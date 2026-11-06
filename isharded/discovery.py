"""Finding a mesh without being told where it is.

A new machine asks the LAN with a broadcast, and asks each online Tailscale peer
directly. Nothing here is trusted. An answer only says where to ask to join, and a
person on the owner's side still has to approve, comparing the key fingerprint both
machines show. A mesh answers only if it was started with `isharded up`, and
`--no-discovery` turns that off.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any
from urllib.parse import urlsplit

import httpx
import psutil

DISCOVERY_PORT = 8208
PROBE = b"isharded: who is there? 1"
LISTEN_SECONDS = 2.0
TAILNET_PORT = 8209
TAILNET_SECONDS = 2.0
LARGEST_ANSWER = 2048


class Beacon(asyncio.DatagramProtocol):
    """Answers a probe with where to ask to join. It says nothing else."""

    def __init__(self, answer: Callable[[], dict[str, Any]]) -> None:
        self.answer = answer
        self.transport: Any = None

    def connection_made(self, transport: Any) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, address: tuple[str, int]) -> None:
        if data == PROBE:
            self.transport.sendto(json.dumps(self.answer()).encode(), address)


async def start_beacon(answer: Callable[[], dict[str, Any]], port: int = DISCOVERY_PORT) -> Any:
    """Listen for probes. Another node on this machine may already be, which is fine."""
    loop = asyncio.get_running_loop()
    try:
        transport, _ = await loop.create_datagram_endpoint(lambda: Beacon(answer), local_addr=("0.0.0.0", port), allow_broadcast=True)
    except OSError:
        return None
    return transport


def valid_answer(raw: bytes) -> dict[str, str] | None:
    """Keep only what a well-formed answer holds. Anyone on the network can send one."""
    try:
        answer = json.loads(raw[:LARGEST_ANSWER])
        url, authority, name = str(answer["url"]).rstrip("/"), str(answer["authority"]), str(answer["name"])[:64]
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.port is None or len(authority) != 64:
            return None
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return {"url": url, "authority": authority, "name": name}


def broadcast_addresses() -> list[str]:
    found = ["255.255.255.255", "127.0.0.1"]
    for addresses in psutil.net_if_addrs().values():
        found += [item.broadcast for item in addresses if item.family == socket.AF_INET and item.broadcast]
    return list(dict.fromkeys(found))


def ask_network(wait: float = LISTEN_SECONDS, port: int = DISCOVERY_PORT, targets: list[str] | None = None) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        probe.settimeout(0.25)
        for target in targets or broadcast_addresses():
            try:
                probe.sendto(PROBE, (target, port))
            except OSError:
                continue  # an interface that is down
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                raw, _ = probe.recvfrom(LARGEST_ANSWER)
            except OSError:
                continue
            answer = valid_answer(raw)
            if answer:
                found.append(answer)
    return found


def tailscale_tool() -> str | None:
    known = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tailscale" / "tailscale.exe"
    return shutil.which("tailscale") or (str(known) if known.is_file() else None)


def tailnet_addresses(run: Any = subprocess.run) -> list[str]:
    """Online peers on this tailnet, which broadcasts do not reach."""
    tool = tailscale_tool()
    if not tool:
        return []
    try:
        done = run([tool, "status", "--json"], capture_output=True, text=True, timeout=5)
        peers = json.loads(done.stdout).get("Peer") or {}
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    return [peer["TailscaleIPs"][0] for peer in peers.values() if peer.get("Online") and peer.get("TailscaleIPs")]


def ask_peer(address: str, port: int = TAILNET_PORT, wait: float = TAILNET_SECONDS) -> dict[str, str] | None:
    from .up import authority_hash

    url = f"http://{address}:{port}"
    try:
        answer = httpx.get(url + "/join/authority", timeout=wait, trust_env=False)
        if answer.status_code != 200:
            return None
        body = answer.json()
        return {"url": url, "authority": authority_hash(body["authority"]), "name": str(body.get("name", address))[:64]}
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return None


def ask_tailnet() -> list[dict[str, str]]:
    addresses = tailnet_addresses()
    if not addresses:
        return []
    with ThreadPoolExecutor(max_workers=16) as pool:
        return [answer for answer in pool.map(ask_peer, addresses) if answer]


def find_meshes(wait: float = LISTEN_SECONDS) -> list[dict[str, str]]:
    """Every mesh this machine can see, one entry each, however many of its nodes answered."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        lan, tailnet = pool.submit(ask_network, wait), pool.submit(ask_tailnet)
        answers = [*lan.result(), *tailnet.result()]
    meshes: dict[str, dict[str, str]] = {}
    for answer in answers:
        meshes.setdefault(answer["authority"], answer)
    return list(meshes.values())
