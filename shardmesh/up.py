"""Starting a mesh with one command, and adding a machine with another.

`shardmesh up` works out everything `shardmesh node` asks for. It finds this
machine's hardware, picks an address, makes certificates, and remembers the result,
so running it again just starts the node.

An invite adds a machine without anyone copying files. The code carries the
addresses of the mesh, a hash of its authority, and a secret. The new machine makes
its own key, proves it holds the secret over that exact request, and gets a
certificate. The owner's machine signs it automatically, so the authority's key
stays there. Nothing else in the mesh can admit anyone.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import webbrowser

import httpx
import psutil

from .access import digest
from .discovery import find_meshes
from .node import add_node_arguments, detect_address, run_node
from .pool import _private_link, _write_private, add_node, approve_membership, home as pool_home, init_pool, request_membership, sign_entry

DEFAULT_PORT = 8209
INVITE_PREFIX = "smi-"
INVITE_ADDRESSES = 4
APPROVAL_SECONDS = 3.0
APPROVAL_WAIT_SECONDS = 180.0
PERSON_WAIT_SECONDS = 1800.0  # someone has to come and click
HEALTH_WAIT_SECONDS = 180.0
SETTLE_WAIT_SECONDS = 60.0  # a node answers about 9 s before it can take a load


def up_home() -> Path:
    configured = os.getenv("SHARDMESH_UP_HOME")
    return Path(configured) if configured else pool_home() / "up"


def read_settings(home: Path) -> dict[str, Any] | None:
    try:
        return json.loads((home / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_settings(home: Path, settings: dict[str, Any]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


def machine_name() -> str:
    """This machine's name, in the characters a node name allows."""
    name = re.sub(r"[^a-z0-9-]+", "-", socket.gethostname().lower()).strip("-")
    return (name or "node")[:48]


def model_folders() -> list[str]:
    """Where models already on this machine are likely to be, so nothing downloads twice."""
    user = Path.home()
    candidates = [user / ".lmstudio" / "models", user / ".cache" / "lm-studio" / "models", user / "models"]
    for settings in (user / ".lmstudio" / "settings.json", user / ".cache" / "lm-studio" / "settings.json"):
        try:
            folder = json.loads(settings.read_text(encoding="utf-8")).get("downloadsFolder")
        except (OSError, ValueError, AttributeError):
            continue
        if folder:
            candidates.append(Path(folder))
    return [str(path) for path in dict.fromkeys(candidates) if path.is_dir()]


def cpu_threads() -> int:
    """Half the cores, so a machine stays usable while it contributes."""
    return max((psutil.cpu_count(logical=False) or 2) // 2, 1)


def needs_tls(address: str, trusted_network: bool) -> bool:
    """Tailscale already encrypts. Anything else gets TLS unless the owner says it is private."""
    return not trusted_network and not _private_link(address)


def make_owner(home: Path, name: str, address: str) -> None:
    """A new mesh: its authority, this node's certificate, and the owner's key."""
    init_pool(str(home))
    add_node(str(home), name, [address, "127.0.0.1"])
    key = "smk-" + secrets.token_urlsafe(32)
    _write_private(home / "admin.key", key.encode())
    signed = sign_entry(str(home), {"type": "key", "name": "owner", "role": "admin", "sha256": digest(key), "requests_per_minute": 0})
    (home / "access.json").write_text(json.dumps({"version": 1, "keys": [], "signed": [signed]}), encoding="utf-8")


def authority_hash(pem: str) -> str:
    return hashlib.sha256(pem.replace("\r\n", "\n").strip().encode()).hexdigest()


def read_invites(home: Path) -> list[dict[str, Any]]:
    try:
        return json.loads((home / "invites.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def write_invites(home: Path, invites: list[dict[str, Any]]) -> None:
    _write_private(home / "invites.json", json.dumps(invites).encode())


def create_invite(home: Path, urls: list[str], uses: int = 1, hours: float = 24.0, clock: Any = time.time) -> dict[str, Any]:
    """A code for adding machines. Only the machine holding the authority can honor it."""
    secret = secrets.token_urlsafe(24)
    expires_at = clock() + hours * 3600
    invites = [invite for invite in read_invites(home) if invite["expires_at"] > clock() and invite["uses"] > 0]
    write_invites(home, [*invites, {"secret": secret, "uses": max(int(uses), 1), "expires_at": expires_at}])
    body = {"urls": urls[:INVITE_ADDRESSES], "authority": authority_hash((home / "ca.pem").read_text(encoding="utf-8")), "secret": secret}
    code = INVITE_PREFIX + base64.urlsafe_b64encode(json.dumps(body, separators=(",", ":")).encode()).decode().rstrip("=")
    return {"code": code, "command": f"shardmesh up --invite {code}", "expires_at": expires_at}


def read_invite(code: str) -> dict[str, Any]:
    if not code.startswith(INVITE_PREFIX):
        raise ValueError(f"an invite starts with {INVITE_PREFIX}")
    text = code[len(INVITE_PREFIX):]
    try:
        body = json.loads(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)))
        return {"urls": [str(url).rstrip("/") for url in body["urls"]], "authority": str(body["authority"]), "secret": str(body["secret"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("this invite is damaged; copy the whole code") from exc


def invite_proof(secret: str, request: str, key_hash: str) -> str:
    """Ties the secret to one request, so seeing a proof pass by lets nobody reuse it."""
    return hmac.new(secret.encode(), f"{request}\n{key_hash}".encode(), hashlib.sha256).hexdigest()


def matching_invite(invites: list[dict[str, Any]], item: dict[str, Any], clock: Any = time.time) -> dict[str, Any] | None:
    for invite in invites:
        if invite["uses"] < 1 or invite["expires_at"] <= clock():
            continue
        expected = invite_proof(invite["secret"], str(item.get("request", "")), str(item.get("key_sha256", "")))
        if hmac.compare_digest(expected, str(item.get("proof", ""))):
            return invite
    return None


def client_for(settings: dict[str, Any], home: Path) -> httpx.Client:
    verify: Any = str(home / "ca.pem") if settings.get("tls") else True
    return httpx.Client(timeout=30.0, trust_env=False, verify=verify)


def local_url(settings: dict[str, Any]) -> str:
    """How this machine reaches its own node. A certificate names its address and loopback."""
    return f"{'https' if settings.get('tls') else 'http'}://127.0.0.1:{settings['port']}"


def approve_invited(home: Path, client: httpx.Client, base: str, key: str, clock: Any = time.time) -> list[str]:
    """Sign in whoever holds a good invite or was approved. Runs only where the authority's key is."""
    headers = {"Authorization": f"Bearer {key}"}
    response = client.get(base + "/join/requests", headers=headers)
    if response.status_code != 200:
        return []
    added = []
    for item in response.json().get("requests", []):
        invites = read_invites(home)
        invite = matching_invite(invites, item, clock)
        name = str(item["name"])
        if item.get("denied"):
            client.post(base + "/access/entries", json=sign_entry(str(home), {"type": "deny", "name": name, "fingerprint": item["fingerprint"]}))
            continue
        if invite is None and not item.get("approved"):
            announce_waiting(name, item)
            continue
        certificate = approve_membership(str(home), str(item["request"]))
        client.post(base + "/access/entries", json=sign_entry(str(home), {"type": "member", "name": name, "certificate": certificate}))
        key_hash = str(item.get("key_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", key_hash):
            entry = {"type": "key", "name": f"{name}-user", "role": "user", "sha256": key_hash, "requests_per_minute": 0}
            client.post(base + "/access/entries", json=sign_entry(str(home), entry))
        if invite is not None:
            invite["uses"] -= 1
            write_invites(home, invites)
        added.append(name)
    return added


ANNOUNCED: set[str] = set()


def announce_waiting(name: str, item: dict[str, Any]) -> None:
    if name not in ANNOUNCED:
        ANNOUNCED.add(name)
        print(f"{name} is asking to join, with key {item.get('fingerprint')}. Check that matches what it shows, then approve it on the dashboard or with: shardmesh approve {name}", flush=True)


def fetch_authority(invite: dict[str, Any]) -> tuple[str, str]:
    """The mesh's authority, checked against the hash in the invite. Trust starts here."""
    problems = []
    for url in invite["urls"]:
        try:
            # Nothing is trusted yet, so the connection proves nothing. The hash does.
            answer = httpx.get(url + "/join/authority", timeout=15.0, verify=False, trust_env=False)
            answer.raise_for_status()
            pem = answer.json()["authority"]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            problems.append(f"{url}: {exc}")
            continue
        if authority_hash(pem) != invite["authority"]:
            raise ValueError(f"{url} is not the mesh this invite is for; refusing to join it")
        return url, pem
    raise ValueError("could not reach the mesh: " + "; ".join(problems))


def certificate_matches_key(certificate: str, key_file: Path) -> bool:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    public = x509.load_pem_x509_certificate(certificate.encode()).public_key()
    private = serialization.load_pem_private_key(key_file.read_bytes(), password=None)
    raw = (serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return public.public_bytes(*raw) == private.public_key().public_bytes(*raw)


def join_mesh(home: Path, invite: dict[str, Any], name: str, address: str) -> str:
    """Ask for a place in the mesh, with an invite's secret or for a person to approve.

    Returns the address that answered.
    """
    url, pem = fetch_authority(invite)
    home.mkdir(parents=True, exist_ok=True)
    (home / "ca.pem").write_text(pem, encoding="utf-8")
    verify: Any = str(home / "ca.pem") if url.startswith("https") else True
    key = "smk-" + secrets.token_urlsafe(32)
    request = request_membership(str(home), name, [address, "127.0.0.1"])
    key_hash = digest(key)
    secret = invite.get("secret")
    asking = {"request": request, "key_sha256": key_hash}
    if secret:
        asking["proof"] = invite_proof(secret, request, key_hash)
    with httpx.Client(timeout=30.0, trust_env=False, verify=verify) as client:
        answer = client.post(url + "/join/requests", json=asking)
        if answer.status_code != 200:
            raise ValueError(answer.text[:200])
        state = answer.json().get("status")
        if state in {"refused", "busy"}:
            raise ValueError("this mesh refused this machine" if state == "refused" else "too many machines are waiting; try again soon")
        if secret:
            print(f"Asked to join as {name}. Waiting for the owner's machine to sign.", flush=True)
        else:
            print(f"Asked to join as {name}. Approve it on the mesh's dashboard, checking the key shown there is {answer.json().get('fingerprint')}.", flush=True)
        deadline = time.monotonic() + (APPROVAL_WAIT_SECONDS if secret else PERSON_WAIT_SECONDS)
        while time.monotonic() < deadline:
            found = client.get(f"{url}/join/certificate/{name}")
            if found.status_code == 200:
                certificate = found.json()["certificate"]
                if not certificate_matches_key(certificate, home / f"{name}.key"):
                    raise ValueError(f"another machine already uses the name {name}; run again with --name")
                (home / f"{name}.pem").write_text(certificate, encoding="utf-8")
                _write_private(home / "user.key", key.encode())
                return url
            time.sleep(2.0)
    raise ValueError("nobody signed this request. The owner's machine must be running, and an invite only works while it lasts")


def node_arguments(settings: dict[str, Any], home: Path) -> argparse.Namespace:
    """Everything `shardmesh node` would have asked for."""
    name, port = settings["name"], int(settings["port"])
    arguments = [
        "--node-id", name,
        "--control-port", str(port),
        "--base-port", str(port - 9),
        "--advertise-host", settings["address"],
        "--keys-file", str(home / "access.json"),
        "--weights-ram-gb", settings.get("ram_gb") or "auto",
        "--cpu-threads", str(settings["cpu_threads"]),
        "--log-file", str(home / "node.log"),
    ]
    if settings.get("ownerless"):
        arguments += ["--ownerless", str(home / "commons")]
    else:
        arguments += ["--ca-file", str(home / "ca.pem"), "--identity-cert", str(home / f"{name}.pem"), "--identity-key", str(home / f"{name}.key")]
    if settings.get("join"):
        arguments += ["--join", ",".join(settings["join"])]
    if settings.get("tls"):
        arguments += ["--tls-cert", str(home / f"{name}.pem"), "--tls-key", str(home / f"{name}.key")]
    elif settings.get("trusted_network"):
        arguments.append("--trusted-network")
    if settings.get("no_gpu"):
        arguments += ["--gpu-device", "none", "--cpu-active"]
    if settings.get("gpu_gb"):
        arguments += ["--gpu-memory-gb", str(settings["gpu_gb"])]
    for folder in settings.get("gguf_dirs", []):
        arguments += ["--gguf-dir", folder]
    parser = argparse.ArgumentParser()
    add_node_arguments(parser)
    return parser.parse_args(arguments)


def sign_in_key(home: Path) -> str | None:
    for name in ("admin.key", "user.key"):
        path = home / name
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def wait_until_healthy(client: httpx.Client, base: str) -> bool:
    deadline = time.monotonic() + HEALTH_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            if client.get(base + "/health", timeout=3.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def can_load(answer: httpx.Response) -> bool:
    """Whether a coordinator is known and a worker is there to load onto.

    Only a node with no coordinator answers 503. A key too weak to read the status
    still gets past that point, which is all it can show.
    """
    if answer.status_code != 200:
        return answer.status_code != 503
    return any(worker.get("live") for worker in answer.json().get("workers", {}).values())


def wait_until_settled(client: httpx.Client, base: str, key: str | None) -> bool:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    deadline = time.monotonic() + SETTLE_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            if can_load(client.get(base + "/admin/status", headers=headers, timeout=5.0)):
                return True
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(1.0)
    return False


def load_first_model(settings: dict[str, Any], home: Path, key: str | None, model: str) -> None:
    """Follows the load to its end, since a refused or failed model would otherwise say nothing."""
    base = local_url(settings)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with client_for(settings, home) as client:
        try:
            answer = client.post(base + "/admin/load", json={"model_id": model, "wait": False}, headers=headers, timeout=30.0)
        except httpx.HTTPError as exc:
            print(f"Could not load {model}: {exc}", flush=True)
            return
        if answer.status_code >= 400:
            print(f"Could not load {model}: {answer.text[:200]}", flush=True)
            return
        print(f"Loading {model}.", flush=True)
        while True:
            time.sleep(2.0)
            try:
                job = client.get(base + "/admin/status", headers=headers, timeout=30.0).json().get("load_job") or {}
            except (httpx.HTTPError, ValueError):
                continue  # a busy or changing coordinator is no reason to give up on the load
            if job.get("phase") == "ready":
                print(f"{model} is ready.", flush=True)
                return
            if job.get("phase") in {"failed", "cancelled"}:
                print(f"Could not load {model}: {job.get('error')}", flush=True)
                return


def keep_watch(home: Path, settings: dict[str, Any], args: argparse.Namespace) -> None:
    """Runs beside the node: says it is ready, and signs in invited machines."""
    base = local_url(settings)
    key = sign_in_key(home)
    with client_for(settings, home) as client:
        if not wait_until_healthy(client, base):
            print(f"The node did not come up. Its log is {home / 'node.log'}.", flush=True)
            return
        settled = wait_until_settled(client, base, key)
        page = f"{base}/chat"
        print(f"\nShardMesh is up as {settings['name']}.\n  Chat and dashboard: {page}", flush=True)
        if (home / "ca.key").is_file():
            print("  Add a machine:      shardmesh invite\n", flush=True)
        if not settled:
            print(f"It has no coordinator or worker yet, so it cannot load a model. Its log is {home / 'node.log'}.", flush=True)
        if key and not args.no_browser:
            webbrowser.open(f"{page}#key={key}")
        if args.model and settled:
            # A big model takes minutes, and invited machines should not wait behind it.
            threading.Thread(target=load_first_model, args=(settings, home, key, args.model), daemon=True).start()
        if settings.get("ownerless") or not (home / "ca.key").is_file():
            return
        while True:
            time.sleep(APPROVAL_SECONDS)
            try:
                for name in approve_invited(home, client, base, key or ""):
                    print(f"Added {name} to the mesh.", flush=True)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                print(f"Could not check for new machines: {exc}", flush=True)


def choose_mesh(found: list[dict[str, str]], args: argparse.Namespace) -> dict[str, str] | None:
    """Whether to join a mesh this machine can see. Never without being asked to."""
    if not found:
        return None
    if args.find:
        if len(found) > 1:
            raise ValueError(f"found {len(found)} meshes, so use an invite to choose one")
        return found[0]
    print("Found on this network:", flush=True)
    for number, mesh in enumerate(found, 1):
        print(f"  {number}. {mesh['name']} at {mesh['url']}, authority {mesh['authority'][:12]}", flush=True)
    if not sys.stdin.isatty():
        print("Starting a new mesh instead. Run shardmesh up --find to ask to join.", flush=True)
        return None
    answer = input(f"Join which? Enter a number, or n for a new mesh [{1}]: ").strip().lower()
    if answer in {"n", "no"}:
        return None
    index = int(answer) - 1 if answer.isdigit() else 0
    return found[index] if 0 <= index < len(found) else None


def first_settings(args: argparse.Namespace, home: Path) -> dict[str, Any]:
    address = args.address or detect_address()
    tls = bool(args.tls) or needs_tls(address, args.trusted_network)
    return {
        "name": args.name or machine_name(),
        "address": address,
        "port": args.port,
        "tls": tls and not args.ownerless,
        "trusted_network": args.trusted_network,
        "cpu_threads": args.cpu_threads or cpu_threads(),
        "no_gpu": args.no_gpu,
        "gpu_gb": args.gpu_gb,
        "ram_gb": str(args.ram_gb) if args.ram_gb else None,
        "gguf_dirs": [*(args.gguf_dir or []), *model_folders()],
        "ownerless": bool(args.ownerless),
        "join": [args.join] if args.join else [],
        "discoverable": not args.no_discovery and not args.ownerless,
    }


def make_ownerless(home: Path, settings: dict[str, Any], joining: str) -> None:
    from .access import create_key
    from .commons import identity_name, make_identity, write_policy

    make_identity(home / "commons")
    write_policy(home / "commons", joining)
    settings["name"] = identity_name(home / "commons")
    key = create_key(str(home / "access.json"), "owner", "admin")
    _write_private(home / "admin.key", key.encode())


def become_member(home: Path, settings: dict[str, Any], invite: dict[str, Any]) -> None:
    settings["tls"] = any(url.startswith("https") for url in invite["urls"])
    answered = join_mesh(home, invite, settings["name"], settings["address"])
    settings["join"] = list(dict.fromkeys([answered, *invite["urls"]]))
    write_settings(home, settings)


def run_up(args: argparse.Namespace) -> int:
    home = up_home()
    os.environ["SHARDMESH_UP_HOME"] = str(home)
    # A mesh's records stay with it, so an earlier pool on this machine cannot leak in.
    os.environ["SHARDMESH_STATE_FILE"] = str(home / "orchestrator-state.json")
    os.environ["SHARDMESH_AUDIT_FILE"] = str(home / "audit.jsonl")
    settings = read_settings(home)
    try:
        if args.invite:
            if settings:
                print(f"This machine already belongs to a mesh, kept in {home}. Remove that folder to join another.", file=sys.stderr)
                return 1
            settings = first_settings(args, home)
            become_member(home, settings, read_invite(args.invite))
        elif settings is None:
            settings = first_settings(args, home)
            mesh = None if args.new_mesh or args.ownerless or args.no_discovery else choose_mesh(find_meshes(), args)
            if mesh:
                become_member(home, settings, {"urls": [mesh["url"]], "authority": mesh["authority"]})
            else:
                if settings["ownerless"]:
                    make_ownerless(home, settings, args.joining)
                else:
                    make_owner(home, settings["name"], settings["address"])
                write_settings(home, settings)
                print(f"Made a new mesh in {home}. This machine owns it. Keep that folder safe.", flush=True)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    address = settings["address"]
    if not settings["tls"] and not _private_link(address) and not settings.get("trusted_network"):
        print(f"{address} is not an encrypted network, so this node needs TLS or --trusted-network.", file=sys.stderr)
        return 1
    if settings["tls"]:
        print("Using TLS with this mesh's own certificate, so a browser will warn once. Tailscale avoids that.", flush=True)
    if settings.get("discoverable"):
        os.environ["SHARDMESH_DISCOVERABLE"] = "1"
    threading.Thread(target=keep_watch, args=(home, settings, args), daemon=True).start()
    return run_node(node_arguments(settings, home))


def run_invite(args: argparse.Namespace) -> int:
    home = up_home()
    settings = read_settings(home)
    if settings is None or not (home / "ca.key").is_file():
        print("Only the machine that made the mesh can invite. Run this there.", file=sys.stderr)
        return 1
    url = f"{'https' if settings['tls'] else 'http'}://{settings['address']}:{settings['port']}"
    invite = create_invite(home, [url], args.uses, args.hours)
    print(f"On the new machine, run:\n\n  {invite['command']}\n\nIt works {args.uses} time(s), for {args.hours:g} hours.")
    return 0


def run_approve(args: argparse.Namespace) -> int:
    home = up_home()
    settings = read_settings(home)
    key = sign_in_key(home)
    if settings is None or key is None:
        print("Run this on the machine that made the mesh.", file=sys.stderr)
        return 1
    with client_for(settings, home) as client:
        answer = client.post(f"{local_url(settings)}/admin/join/{args.name}/{args.join_command}", headers={"Authorization": f"Bearer {key}"})
    print(f"Done: {args.join_command} {args.name}." if answer.status_code < 400 else f"Refused: {answer.text[:200]}")
    return 0 if answer.status_code < 400 else 1


def service_definition(python: str, system: str, user_home: Path) -> tuple[Path | None, str]:
    """Where a start-at-login definition goes, and what it says. None means a scheduled task."""
    command = f'"{python}" -m shardmesh up --no-browser'
    if system == "Windows":
        return None, command
    if system == "Darwin":
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict>'
            "<key>Label</key><string>ai.shardmesh.node</string>"
            f"<key>ProgramArguments</key><array><string>{python}</string><string>-m</string><string>shardmesh</string><string>up</string><string>--no-browser</string></array>"
            "<key>RunAtLoad</key><true/><key>KeepAlive</key><true/></dict></plist>\n"
        )
        return user_home / "Library" / "LaunchAgents" / "ai.shardmesh.node.plist", body
    body = f"[Unit]\nDescription=ShardMesh node\nAfter=network-online.target\n\n[Service]\nExecStart={python} -m shardmesh up --no-browser\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n"
    return user_home / ".config" / "systemd" / "user" / "shardmesh.service", body


def run_service(args: argparse.Namespace) -> int:
    """Start this node at login, for this user only. Run it on purpose: it changes what starts."""
    system = platform.system()
    path, body = service_definition(sys.executable, system, Path.home())
    if system == "Windows":
        command = ["schtasks", "/Create", "/F", "/TN", "ShardMesh", "/SC", "ONLOGON", "/TR", body] if args.service_command == "install" else ["schtasks", "/Delete", "/F", "/TN", "ShardMesh"]
        return subprocess.run(command, check=False).returncode
    assert path is not None
    start = ["systemctl", "--user"] if system == "Linux" else ["launchctl"]
    if args.service_command == "install":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        action = [*start, "enable", "--now", "shardmesh"] if system == "Linux" else [*start, "load", str(path)]
        return subprocess.run(action, check=False).returncode
    action = [*start, "disable", "--now", "shardmesh"] if system == "Linux" else [*start, "unload", str(path)]
    code = subprocess.run(action, check=False).returncode
    path.unlink(missing_ok=True)
    return code


def checks() -> list[tuple[bool, str, str]]:
    """What usually stops a node starting: each with what to do about it."""
    found: list[tuple[bool, str, str]] = []

    def check(good: bool, text: str, fix: str) -> None:
        found.append((good, text, "" if good else fix))

    version = sys.version_info
    check(version >= (3, 10), f"Python {version.major}.{version.minor}", "ShardMesh needs Python 3.10 or newer")
    try:
        import cryptography  # noqa: F401

        found.append((True, "cryptography is installed", ""))
    except ImportError:
        found.append((False, "cryptography is missing", "pip install shardmesh[pool]"))
    address = detect_address()
    private = _private_link(address)
    found.append((True, f"This machine would advertise {address}" + (" (encrypted by Tailscale)" if private else ""), ""))
    if not private:
        found.append((True, "No Tailscale address found", "Nodes will use TLS with their own certificates. Tailscale is the easier way to link machines in different places"))
    with socket.socket() as probe:
        free = probe.connect_ex(("127.0.0.1", DEFAULT_PORT)) != 0
    check(free, f"Port {DEFAULT_PORT} is free" if free else f"Port {DEFAULT_PORT} is in use", "Stop what uses it, or run shardmesh up --port with another port")
    memory = psutil.virtual_memory()
    check(memory.available > 2 * 1024**3, f"{memory.available / 1024**3:.1f} GiB of RAM free", "A node needs at least 2 GiB free")
    disk = shutil.disk_usage(str(Path.home()))
    check(disk.free > 10 * 1024**3, f"{disk.free / 1024**3:.0f} GiB of disk free", "Models need room. Free some disk first")
    folders = model_folders()
    found.append((True, f"Found models in {', '.join(folders)}" if folders else "No local model folders found", "" if folders else "Models will download when you load one"))
    if platform.system() == "Linux":
        missing = not any(Path(folder, "libgomp.so.1").exists() for folder in ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64", "/usr/lib"))
        check(not missing, "libgomp is installed" if not missing else "libgomp is missing", "sudo apt-get install libgomp1")
    return found


def run_doctor(_: argparse.Namespace) -> int:
    results = checks()
    for good, text, advice in results:
        print(f"{'ok  ' if good else 'FIX '} {text}" + (f"\n     {advice}" if advice else ""))
    bad = sum(1 for good, _, _ in results if not good)
    print("Ready to run shardmesh up." if not bad else f"{bad} thing(s) to fix first.")
    return 1 if bad else 0


def add_arguments(subparsers: Any) -> None:
    up = subparsers.add_parser("up", help="start a node with everything worked out for you; the easy way in")
    up.add_argument("--invite", default=None, help="join a mesh, with a code from shardmesh invite")
    up.add_argument("--name", default=None, help="this node's name; default is the machine's name")
    up.add_argument("--port", type=int, default=DEFAULT_PORT, help="port for the chat page and API")
    up.add_argument("--address", default=None, help="the address other nodes reach this one on; found for you if not set")
    up.add_argument("--tls", action="store_true", help="use TLS even on Tailscale")
    up.add_argument("--trusted-network", action="store_true", help="skip TLS, because nobody else can read this network")
    up.add_argument("--no-gpu", action="store_true", help="contribute CPU and RAM only")
    up.add_argument("--gpu-gb", type=float, default=0, help="VRAM to give; default is a share of what is free")
    up.add_argument("--ram-gb", type=float, default=0, help="RAM to give for weights; default is a share of what is free")
    up.add_argument("--cpu-threads", type=int, default=0, help="threads to give; default is half the cores")
    up.add_argument("--gguf-dir", action="append", default=None, help="a folder of GGUF files; LM Studio's is found for you. Repeatable")
    up.add_argument("--model", default=None, help="load this model once the node is up")
    up.add_argument("--no-browser", action="store_true", help="do not open the chat page")
    up.add_argument("--find", action="store_true", help="ask to join the one mesh on this network, without asking you first")
    up.add_argument("--new-mesh", action="store_true", help="start a new mesh even if one is on this network")
    up.add_argument("--no-discovery", action="store_true", help="do not look for a mesh, and do not answer machines looking for this one")
    up.add_argument("--ownerless", action="store_true", help="make or join a network nobody owns")
    up.add_argument("--joining", choices=["open", "vouched"], default="vouched", help="with --ownerless: who this node accepts")
    up.add_argument("--join", default=None, help="with --ownerless: any node already in the network")
    up.set_defaults(handler=run_up)
    invite = subparsers.add_parser("invite", help="make a code that adds a machine to this mesh")
    invite.add_argument("--uses", type=int, default=1, help="how many machines the code can add")
    invite.add_argument("--hours", type=float, default=24.0, help="how long the code lasts")
    invite.set_defaults(handler=run_invite)
    for name, text in (("approve", "let a machine that asked into the mesh"), ("deny", "turn a machine down")):
        change = subparsers.add_parser(name, help=text)
        change.add_argument("name", help="the machine's name, as it asked")
        change.set_defaults(handler=run_approve, join_command=name)
    doctor = subparsers.add_parser("doctor", help="check this machine, and say what to fix")
    doctor.set_defaults(handler=run_doctor)
    service = subparsers.add_parser("service", help="start this node when you log in")
    service_commands = service.add_subparsers(dest="service_command", required=True)
    service_commands.add_parser("install", help="start at login, for this user").set_defaults(handler=run_service)
    service_commands.add_parser("remove", help="stop starting at login").set_defaults(handler=run_service)

