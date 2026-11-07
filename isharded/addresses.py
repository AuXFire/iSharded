"""Every address a node can be reached on, and proof that one of them is that node.

A node registers through one address, and that is often a relay. Two machines on one
LAN would then send every pass the long way. So a node also lists the addresses it has,
and a stage can be told a shorter way to the next stage.

A private address that answers proves nothing, because two sites can reuse it. Nothing
here counts until the far end has signed for it, by the identity the pool already checks.
Until then no credential goes to that address either: both ends sign a claim that names
one challenge and one address, and that is no use to anyone who reads it.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
import socket
from typing import Any
from urllib.parse import urlsplit

import httpx
import psutil

from .access import node_of
from .pool import LOOPBACK, _private_link
from .tickets import AUDIENCE, PREFIX, TicketMaker, make_checker
from .tls import node_transport

PROVE_PATH = "/internal/diag/prove"
ECHO_PATH = "/internal/diag/echo"
MOST_ADDRESSES = 16
# An address that takes longer than this to connect is no shorter way.
CONNECT_SECONDS = 1.0
REPLY_SECONDS = 5.0
EVERY_INTERFACE = {"0.0.0.0", "::"}


def host_port(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def local_hosts(bind_host: str) -> list[str]:
    """What this machine answers on, given where it listens."""
    if bind_host in LOOPBACK:
        return []
    if bind_host not in EVERY_INTERFACE:
        return [bind_host]
    found = []
    for addresses in psutil.net_if_addrs().values():
        for item in addresses:
            if item.family not in {socket.AF_INET, socket.AF_INET6}:
                continue
            try:
                address = ipaddress.ip_address(item.address)
            except ValueError:
                continue  # a scoped IPv6 address, which means nothing to another machine
            if address.version == 4 and not (address.is_loopback or address.is_link_local) or address.is_global:
                found.append(item.address)
    return found


def reach_urls(setting: str, advertised: str, bind_host: str, port: int, plain_allowed: bool) -> tuple[list[str], int]:
    """The URLs to register beside the advertised one, and how many were left out.

    A pool that may only send credentials over TLS or an overlay must not lose that on
    a shorter path, so an address with neither is left out unless the operator said
    this network is safe. The ring links are encrypted either way, but HTTP passes are not.
    """
    scheme = urlsplit(advertised).scheme or "http"
    hosts: list[str] = []
    for item in setting.split(","):
        item = item.strip().lower()
        if item == "none":
            return [], 0
        hosts += local_hosts(bind_host) if item == "auto" else [item] if item else []
    usable = [host for host in dict.fromkeys(hosts) if scheme == "https" or plain_allowed or _private_link(host)]
    urls = [f"{scheme}://{host_port(host, port)}" for host in usable]
    return [url for url in urls if url != advertised.rstrip("/")][:MOST_ADDRESSES], len(hosts) - len(usable)


def reach_options(via: str) -> dict[str, Any]:
    """Requests to a shorter address still speak to the node at its advertised one.

    TLS then checks the certificate against the advertised name, and a ticket names that
    address too. A ticket for a private address could be replayed at another site.
    """
    parts = urlsplit(via)
    return {"sni_hostname": parts.hostname or "", AUDIENCE: parts.netloc}


def direct_client(auth: Any) -> httpx.AsyncClient:
    """A client for one address alone.

    A TLS check happens when a connection opens. A connection shared with another
    address's traffic would carry its proof over to a node that never gave one.
    """
    return httpx.AsyncClient(transport=node_transport(), timeout=httpx.Timeout(REPLY_SECONDS, connect=CONNECT_SECONDS), auth=auth)


def claim(side: str, nonce: str, url: str, worker_id: str) -> str:
    """What one end signs. The side stops an answer being played back as a question."""
    return f"{side}:{nonce}:{url}:{worker_id}"


class NodeProof:
    """Signs and checks claims with whatever this pool uses to tell its nodes apart."""

    def __init__(self, token: str | None, maker: TicketMaker | None) -> None:
        self.token = token
        self.maker = maker

    @staticmethod
    def _named() -> bool:
        return bool(os.getenv("ISHARDED_OWNERLESS") or os.getenv("ISHARDED_CA_FILE"))

    def _mac(self, text: str) -> str:
        assert self.token is not None
        return hmac.new(self.token.encode(), text.encode(), "sha256").hexdigest()

    def sign(self, text: str) -> str | None:
        """Nothing, if this pool has no identity to prove."""
        if self._named():
            return self.maker.fresh(text) if self.maker else None
        return self._mac(text) if self.token else None

    def signer(self, proof: str, text: str) -> str | None:
        """The node that signed, or an empty name for any holder of the pool token."""
        if self._named():
            checker = make_checker({text})
            return checker.name(proof) if checker and proof.startswith(PREFIX) else None
        return "" if self.token and hmac.compare_digest(proof, self._mac(text)) else None


async def prove(client: httpx.AsyncClient, proof: NodeProof, url: str, via: str, worker_id: str, revoked: set[str]) -> bool:
    """Whether the node at this address is the one that registered as this worker."""
    nonce = secrets.token_hex(16)
    asked = proof.sign(claim("ask", nonce, url, worker_id))
    if asked is None:
        return False
    response = await client.post(
        url + PROVE_PATH,
        json={"nonce": nonce, "url": url, "worker_id": worker_id},
        headers={"Authorization": f"Bearer {asked}"},
        extensions=reach_options(via),
    )
    if response.status_code != 200:
        return False
    answer = response.json()
    if not isinstance(answer, dict):
        return False
    signer = proof.signer(str(answer.get("proof")), claim("answer", nonce, url, worker_id))
    return signer == "" or (signer == node_of(worker_id) and signer not in revoked)
