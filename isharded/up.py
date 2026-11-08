"""Starting a mesh with one command, and adding a machine with another.

`isharded up` works out everything `isharded node` asks for. It finds this
machine's hardware, picks an address, makes certificates, and remembers the result,
so running it again just starts the node.

An invite adds a machine without anyone copying files. The code carries the
addresses of the mesh, a hash of its authority, and a secret. The new machine makes
its own key, proves it holds the secret over that exact request, and gets a
certificate. The owner's machine signs it automatically, so the authority's key
stays there. Nothing else in the mesh can admit anyone.
"""
from __future__ import annotations

import _thread
import argparse
from contextlib import suppress
import ctypes
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
from typing import Any, Callable
import webbrowser

import httpx
import psutil

from .access import create_key, digest, new_key
from .commons import identity_name, make_identity, write_policy
from .discovery import find_meshes
from .engine import ABI_VERSION, find_library
from .geo import country_code
from .inventory import ram
from .node import add_node_arguments, detect_address, run_node
from .optional import has_torch
from .tls import OWN_ADDRESS, stay_on_this_machine
from .pool import _private_link, _write_private, add_node, admission_entries, bearer_headers, home as pool_home, init_pool, request_membership, sign_entry
from .statements import VERIFIER_LENGTH, key_verifier
from .tickets import _decode, _encode

DEFAULT_PORT = 8209
INVITE_PREFIX = "isi-"
INVITE_ADDRESSES = 4
APPROVAL_SECONDS = 3.0
APPROVAL_WAIT_SECONDS = 180.0
PERSON_WAIT_SECONDS = 1800.0  # someone has to come and click
HEALTH_WAIT_SECONDS = 180.0
SETTLE_WAIT_SECONDS = 60.0  # a node answers about 9 s before it can take a load
STOP_WAIT_SECONDS = 60.0  # each process gets 10 s, and a node can run several
ENGINE_ADVICE = "Tokens take the slower Python path. Run the installer again for a release that has the engine, or build it with python scripts/build_engine.py"


def up_home() -> Path:
    configured = os.getenv("ISHARDED_UP_HOME")
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
    key = new_key()
    _write_private(home / "admin.key", key.encode())
    signed = sign_entry(str(home), {"type": "key", "name": "owner", "role": "admin", "sha256": digest(key), "requests_per_minute": 0, "verifier": key_verifier(key)})
    (home / "access.json").write_text(json.dumps({"version": 1, "keys": [], "signed": [signed]}), encoding="utf-8")


def give_owner_a_verifier(home: Path, client: httpx.Client, base: str, key: str) -> None:
    """A mesh made before verifiers has an owner key that only commands the node it is
    given to. The owner's machine holds that key and the authority, so it can fix this.
    """
    document = json.loads((home / "access.json").read_text(encoding="utf-8"))
    entries = [held["entry"] for held in document.get("signed", [])]
    mine = [entry for entry in entries if entry.get("type") == "key" and entry.get("sha256") == digest(key)]
    if mine and not any(entry.get("verifier") for entry in mine):
        kept = {field: mine[0][field] for field in ("type", "name", "role", "sha256", "requests_per_minute") if field in mine[0]}
        client.post(base + "/access/entries", json=sign_entry(str(home), {**kept, "verifier": key_verifier(key)}))


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
    code = INVITE_PREFIX + _encode(json.dumps(body, separators=(",", ":")).encode())
    return {"code": code, "command": f"isharded up --invite {code}", "expires_at": expires_at}


def read_invite(code: str) -> dict[str, Any]:
    if not code.startswith(INVITE_PREFIX):
        raise ValueError(f"an invite starts with {INVITE_PREFIX}")
    try:
        body = json.loads(_decode(code[len(INVITE_PREFIX):]))
        return {"urls": [str(url).rstrip("/") for url in body["urls"]], "authority": str(body["authority"]), "secret": str(body["secret"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("this invite is damaged; copy the whole code") from exc


def invite_proof(secret: str, request: str, key_hash: str, verifier: str = "") -> str:
    """Ties the secret to one request, so seeing a proof pass by lets nobody reuse it.

    A machine running older code sends no verifier, and its proof still has to pass.
    """
    message = f"{request}\n{key_hash}\n{verifier}" if verifier else f"{request}\n{key_hash}"
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def matching_invite(invites: list[dict[str, Any]], item: dict[str, Any], clock: Any = time.time) -> dict[str, Any] | None:
    for invite in invites:
        if invite["uses"] < 1 or invite["expires_at"] <= clock():
            continue
        expected = invite_proof(invite["secret"], str(item.get("request", "")), str(item.get("key_sha256", "")), str(item.get("key_verifier", "")))
        if hmac.compare_digest(expected, str(item.get("proof", ""))):
            return invite
    return None


def mesh_client(verify: Any, timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(event_hooks={"request": [stay_on_this_machine]}, timeout=timeout, trust_env=False, verify=verify)


def client_for(settings: dict[str, Any], home: Path) -> httpx.Client:
    return mesh_client(str(home / "ca.pem") if settings.get("tls") else True)


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
        for entry in admission_entries(str(home), item):
            client.post(base + "/access/entries", json=sign_entry(str(home), entry))
        key_hash = str(item.get("key_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", key_hash):
            entry = {"type": "key", "name": f"{name}-user", "role": "user", "sha256": key_hash, "requests_per_minute": 0}
            verifier = str(item.get("key_verifier", ""))
            if re.fullmatch(rf"[A-Za-z0-9_-]{{{VERIFIER_LENGTH}}}", verifier):
                entry["verifier"] = verifier
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
        print(f"{name} is asking to join, with key {item.get('fingerprint')}. Check that matches what it shows, then approve it on the dashboard or with: isharded approve {name}", flush=True)


def fetch_authority(invite: dict[str, Any]) -> tuple[str, str]:
    """The mesh's authority, checked against the hash in the invite. Trust starts here."""
    problems = []
    for url in invite["urls"]:
        try:
            # Nothing is trusted yet, so the connection proves nothing. The hash does.
            with mesh_client(False, timeout=15.0) as client:
                answer = client.get(url + "/join/authority")
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


def ask_to_join(client: httpx.Client, url: str, asking: dict[str, str], deadline: float) -> httpx.Response:
    """Machines behind one address share its limit, so a batch of them takes turns."""
    while True:
        answer = client.post(url + "/join/requests", json=asking)
        wait = float(answer.headers.get("Retry-After", 30))
        if answer.status_code != 429 or time.monotonic() + wait > deadline:
            return answer
        time.sleep(wait)


def join_mesh(home: Path, invite: dict[str, Any], name: str, address: str) -> str:
    """Ask for a place in the mesh, with an invite's secret or for a person to approve.

    Returns the address that answered.
    """
    url, pem = fetch_authority(invite)
    home.mkdir(parents=True, exist_ok=True)
    (home / "ca.pem").write_text(pem, encoding="utf-8")
    key = new_key()
    request = request_membership(str(home), name, [address, "127.0.0.1"])
    key_hash, verifier = digest(key), key_verifier(key)
    secret = invite.get("secret")
    asking = {"request": request, "key_sha256": key_hash, "key_verifier": verifier}
    if secret:
        asking["proof"] = invite_proof(secret, request, key_hash, verifier)
    with mesh_client(str(home / "ca.pem") if url.startswith("https") else True) as client:
        answer = ask_to_join(client, url, asking, time.monotonic() + APPROVAL_WAIT_SECONDS)
        if answer.status_code != 200:
            raise ValueError(answer.text[:200])
        state = answer.json().get("status")
        if state in {"refused", "busy"}:
            raise ValueError("this mesh refused this machine" if state == "refused" else "too many machines are waiting; try again soon")
        replaces = answer.json().get("replaces")
        if replaces:
            # A machine that lost its folder comes back under its name like this. So would a
            # different machine taking the name, which is why it needs an invite or a person.
            print(f"{name} is already a member, with key {replaces}. Joining with this new key retires that one.", flush=True)
        if secret:
            print(f"Asked to join as {name}. Waiting for the owner's machine to sign.", flush=True)
        else:
            print(f"Asked to join as {name}. Approve it on the mesh's dashboard, checking the key shown there is {answer.json().get('fingerprint')}.", flush=True)
        deadline = time.monotonic() + (APPROVAL_WAIT_SECONDS if secret else PERSON_WAIT_SECONDS)
        old = False
        while time.monotonic() < deadline:
            found = client.get(f"{url}/join/certificate/{name}")
            if found.status_code == 200:
                certificate = found.json()["certificate"]
                if certificate_matches_key(certificate, home / f"{name}.key"):
                    (home / f"{name}.pem").write_text(certificate, encoding="utf-8")
                    _write_private(home / "user.key", key.encode())
                    return url
                old = True  # the name's certificate before the owner answers, for the key it had
            time.sleep(2.0)
    if old:
        raise ValueError(f"another machine holds the name {name} and nobody let this one take it over. Use an invite, have the owner approve it, or run again with --name")
    raise ValueError("nobody signed this request. The owner's machine must be running, and an invite only works while it lasts")


def node_arguments(settings: dict[str, Any], home: Path) -> argparse.Namespace:
    """Everything `isharded node` would have asked for."""
    name, port = settings["name"], int(settings["port"])
    arguments = [
        "--node-id", name,
        "--control-port", str(port),
        "--base-port", str(settings.get("worker_port") or port - 9),
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
    if settings.get("location"):
        arguments += ["--location", settings["location"]]
    if settings.get("reach_hosts"):
        arguments += ["--reach-hosts", settings["reach_hosts"]]
    parser = argparse.ArgumentParser()
    add_node_arguments(parser)
    return parser.parse_args(arguments)


def sign_in_key(home: Path) -> str | None:
    for name in ("admin.key", "user.key"):
        path = home / name
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def keep_asking(ask: Callable[[], bool], seconds: float) -> bool:
    """Ask once a second until the answer is yes. A failed attempt counts as a no."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if ask():
                return True
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(1.0)
    return False


def wait_until_healthy(client: httpx.Client, base: str) -> bool:
    return keep_asking(lambda: client.get(base + "/health", timeout=3.0).status_code == 200, HEALTH_WAIT_SECONDS)


def can_load(answer: httpx.Response) -> bool:
    """Whether a coordinator is known and a worker is there to load onto.

    Only a node with no coordinator answers 503. A key too weak to read the status
    still gets past that point, which is all it can show.
    """
    if answer.status_code != 200:
        return answer.status_code != 503
    return any(worker.get("live") for worker in answer.json().get("workers", {}).values())


def wait_until_settled(client: httpx.Client, base: str, key: str | None) -> bool:
    headers = bearer_headers(key)
    return keep_asking(lambda: can_load(client.get(base + "/admin/status", headers=headers, timeout=5.0)), SETTLE_WAIT_SECONDS)


def load_first_model(settings: dict[str, Any], home: Path, key: str | None, model: str) -> None:
    """Follows the load to its end, since a refused or failed model would otherwise say nothing."""
    base = local_url(settings)
    headers = bearer_headers(key)
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


def sign_in_machines(home: Path, client: httpx.Client, base: str, key: str) -> None:
    """The owner's machine, for as long as it runs: nobody else can sign anyone in."""
    try:
        give_owner_a_verifier(home, client, base, key)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        print(f"Could not update the owner's key: {exc}", flush=True)
    while True:
        time.sleep(APPROVAL_SECONDS)
        try:
            for name in approve_invited(home, client, base, key):
                print(f"Added {name} to the mesh.", flush=True)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            print(f"Could not check for new machines: {exc}", flush=True)


def say_ready(home: Path, settings: dict[str, Any], client: httpx.Client, no_browser: bool) -> bool:
    """Says where the node is and what to do next, and opens its page. True once it can load."""
    base, key = local_url(settings), sign_in_key(home)
    settled = wait_until_settled(client, base, key)
    page = f"{base}/chat"
    print(f"\niSharded is up as {settings['name']}.\n  Chat and dashboard: {page}\n  Next:               choose a model on that page", flush=True)
    if (home / "ca.key").is_file():
        print("  Add a machine:      isharded invite\n", flush=True)
    if not settled:
        print(f"It has no coordinator or worker yet, so it cannot load a model. Its log is {home / 'node.log'}.", flush=True)
    if not no_browser:
        webbrowser.open(f"{page}#key={key}" if key else page)
    return settled


def keep_watch(home: Path, settings: dict[str, Any], args: argparse.Namespace) -> None:
    """Runs beside the node: says it is ready, and signs in invited machines."""
    base = local_url(settings)
    key = sign_in_key(home)
    with client_for(settings, home) as client:
        if not wait_until_healthy(client, base):
            print(f"The node did not come up. Its log is {home / 'node.log'}.", flush=True)
            return
        settled = say_ready(home, settings, client, args.no_browser)
        holds_authority = (home / "ca.key").is_file()
        if args.model and settled:
            # A big model takes minutes, and invited machines should not wait behind it.
            threading.Thread(target=load_first_model, args=(settings, home, key, args.model), daemon=True).start()
        if holds_authority and not settings.get("ownerless"):
            sign_in_machines(home, client, base, key or "")


def running_node(home: Path) -> psutil.Process | None:
    """The node started from this folder, if it is still alive. Its start time rules out a reused PID."""
    try:
        pid, started = json.loads((home / "up.pid").read_text(encoding="utf-8"))
        process = psutil.Process(pid)
        return process if abs(process.create_time() - started) < 1 else None
    except (OSError, TypeError, ValueError, psutil.Error):
        return None


def claim_node(home: Path) -> bool:
    """Makes this process the folder's one node. False if another has it."""
    path = home / "up.pid"
    if running_node(home):
        return False
    path.unlink(missing_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, "w", encoding="utf-8") as output:
        json.dump([os.getpid(), psutil.Process().create_time()], output)
    (home / "stop.request").unlink(missing_ok=True)
    return True


def watch_for_stop(home: Path, finished: threading.Event) -> None:
    """Ctrl+C by file: a node started in the background has no console to send one to."""
    while not finished.wait(1.0):
        if (home / "stop.request").exists():
            _thread.interrupt_main()  # run_node stops its services on this, as for a person's Ctrl+C
            return


def detached() -> dict[str, Any]:
    """How to start a process that outlives this one, with no window of its own."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def start_in_background(home: Path, settings: dict[str, Any], args: argparse.Namespace) -> int:
    """Starts the node apart from this process, then answers once it is up.

    The installers and shortcuts use this, so running it again only opens the page.
    """
    if running_node(home) is None:
        command = [sys.executable, "-m", "isharded", "up", "--no-browser", *(["--model", args.model] if args.model else [])]
        # Run from the mesh's folder: python -m would prefer any isharded folder in this one.
        code = str(Path(__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(item for item in (code, os.getenv("PYTHONPATH")) if item)}
        with open(home / "up.log", "ab") as log:
            subprocess.Popen(command, cwd=home, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **detached())
    with client_for(settings, home) as client:
        if not wait_until_healthy(client, local_url(settings)):
            print(f"The node did not come up. Its logs are {home / 'up.log'} and {home / 'node.log'}.", file=sys.stderr)
            return 1
        say_ready(home, settings, client, args.no_browser)
    print("It runs in the background. Stop it with: isharded stop", flush=True)
    return 0


def native_engine() -> tuple[bool, str]:
    """Whether the fast path is here. Loading it is the real test: a file can be for another machine."""
    path = find_library()
    if path is None:
        return False, "turned off by ISHARDED_ENGINE=0" if os.getenv("ISHARDED_ENGINE") == "0" else "not installed"
    try:
        library = ctypes.CDLL(str(path))
        library.isharded_engine_abi.restype = ctypes.c_uint32
        if library.isharded_engine_abi() != ABI_VERSION:
            return False, f"{path} is a different version"
    except (OSError, AttributeError) as exc:
        return False, f"{path} will not load: {exc}"
    return True, str(path)


def describe_gpus(settings: dict[str, Any]) -> str:
    if settings.get("no_gpu"):
        return "not used"
    from .memory import discover_accelerators  # brings in PyTorch where it is installed

    try:
        found = discover_accelerators("auto")
    except (OSError, RuntimeError, ValueError) as exc:
        return f"could not look: {exc}"
    return ", ".join(f"{item.name} ({item.total_bytes / 1024**3:.0f} GiB)" for item in found) or "none found, so this node uses the CPU"


def describe_machine(settings: dict[str, Any], gpus: str) -> list[str]:
    """What was worked out, for someone seeing this machine set up for the first time."""
    address = settings["address"]
    link = "Tailscale or another private link" if _private_link(address) and not settings["tls"] else "TLS with this mesh's own certificate" if settings["tls"] else "none, as you said this network is trusted"
    memory = ram()
    has_engine, engine_note = native_engine()
    folders = settings.get("gguf_dirs") or []
    return [
        f"  Name:       {settings['name']}, at {address}",
        f"  Encryption: {link}",
        f"  GPU:        {gpus}",
        f"  Memory:     {memory.available / 1024**3:.1f} GiB of RAM free, {settings['cpu_threads']} of {os.cpu_count() or 1} CPU threads",
        f"  Models:     {', '.join(folders) if folders else 'no local folders, so models download when you load one'}",
        f"  Engine:     {'native, the fast path' if has_engine else f'Python, the slower path ({engine_note})'}",
    ]


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
        print("Starting a new mesh instead. Run isharded up --find to ask to join.", flush=True)
        return None
    answer = input("Join which? Enter a number, or n for a new mesh [1]: ").strip().lower()
    if answer in {"n", "no"}:
        return None
    index = int(answer) - 1 if answer.isdigit() else 0
    return found[index] if 0 <= index < len(found) else None


def first_settings(args: argparse.Namespace) -> dict[str, Any]:
    address = args.address or detect_address()
    tls = bool(args.tls) or needs_tls(address, args.trusted_network)
    return {
        "name": args.name or machine_name(),
        "address": address,
        "location": args.location,
        "reach_hosts": args.reach_hosts,
        "port": args.port,
        "worker_port": args.worker_port,
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


def make_ownerless(home: Path, settings: dict[str, Any], joining: str, trust: list[str] | None = None) -> None:
    make_identity(home / "commons")
    write_policy(home / "commons", joining, trust=trust)
    settings["name"] = identity_name(home / "commons")
    key = create_key(str(home / "access.json"), "owner", "admin")
    _write_private(home / "admin.key", key.encode())


def become_member(home: Path, settings: dict[str, Any], invite: dict[str, Any]) -> None:
    settings["tls"] = any(url.startswith("https") for url in invite["urls"])
    os.environ.setdefault(OWN_ADDRESS, settings["address"])  # a mesh reached through this machine's own address
    answered = join_mesh(home, invite, settings["name"], settings["address"])
    settings["join"] = list(dict.fromkeys([answered, *invite["urls"]]))
    write_settings(home, settings)


def start_mesh(home: Path, settings: dict[str, Any], args: argparse.Namespace) -> None:
    if settings["ownerless"]:
        make_ownerless(home, settings, args.joining, args.trust)
    else:
        make_owner(home, settings["name"], settings["address"])
    write_settings(home, settings)
    owns = "Nobody owns it, and this node is named " + settings["name"] if settings["ownerless"] else "This machine owns it"
    print(f"Made a new mesh in {home}. {owns}. Keep that folder safe.", flush=True)


def run_up(args: argparse.Namespace) -> int:
    home = up_home()
    os.environ["ISHARDED_UP_HOME"] = str(home)
    # A mesh's records stay with it, so an earlier pool on this machine cannot leak in.
    os.environ["ISHARDED_STATE_FILE"] = str(home / "orchestrator-state.json")
    os.environ["ISHARDED_AUDIT_FILE"] = str(home / "audit.jsonl")
    settings = read_settings(home)
    is_new = settings is None
    try:
        if args.invite:
            if settings:
                print(f"This machine already belongs to a mesh, kept in {home}. Remove that folder to join another.", file=sys.stderr)
                return 1
            settings = first_settings(args)
            become_member(home, settings, read_invite(args.invite))
        elif settings is None:
            settings = first_settings(args)
            mesh = None if args.new_mesh or args.ownerless or args.no_discovery else choose_mesh(find_meshes(), args)
            if mesh:
                become_member(home, settings, {"urls": [mesh["url"]], "authority": mesh["authority"]})
            else:
                start_mesh(home, settings, args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.location and args.location != settings.get("location"):  # a node that already exists can say where it is
        settings["location"] = args.location
        write_settings(home, settings)
    if args.reach_hosts and args.reach_hosts != settings.get("reach_hosts"):
        settings["reach_hosts"] = args.reach_hosts
        write_settings(home, settings)
    address = settings["address"]
    if not settings["tls"] and not _private_link(address) and not settings.get("trusted_network"):
        print(f"{address} is not an encrypted network, so this node needs TLS or --trusted-network.", file=sys.stderr)
        return 1
    if settings["tls"]:
        print("Using TLS with this mesh's own certificate, so a browser will warn once. Tailscale avoids that.", flush=True)
    if settings.get("discoverable"):
        os.environ["ISHARDED_DISCOVERABLE"] = "1"
    if is_new:
        print("Found on this machine:", *describe_machine(settings, describe_gpus(settings)), sep="\n", flush=True)
    if args.background:
        return start_in_background(home, settings, args)
    if not claim_node(home):
        print(f"A node is already running from {home}, at {local_url(settings)}/chat. Stop it with: isharded stop", flush=True)
        return 0
    finished = threading.Event()
    threading.Thread(target=watch_for_stop, args=(home, finished), daemon=True).start()
    threading.Thread(target=keep_watch, args=(home, settings, args), daemon=True).start()
    try:
        return run_node(node_arguments(settings, home))
    finally:
        finished.set()
        (home / "up.pid").unlink(missing_ok=True)


def run_stop(_: argparse.Namespace) -> int:
    """Stops the node started from this machine's folder, foreground or background."""
    home = up_home()
    process = running_node(home)
    if process is None:
        print(f"No node is running from {home}.")
        return 0
    (home / "stop.request").touch()
    try:
        process.wait(STOP_WAIT_SECONDS)
    except psutil.TimeoutExpired:
        for item in [*process.children(recursive=True), process]:
            with suppress(psutil.Error):
                item.kill()
        print("The node did not stop in time, so it was killed.", flush=True)
    (home / "stop.request").unlink(missing_ok=True)
    print("Stopped.")
    return 0


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
    command = f'"{python}" -m isharded up --no-browser'
    if system == "Windows":
        return None, command
    if system == "Darwin":
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict>'
            "<key>Label</key><string>ai.isharded.node</string>"
            f"<key>ProgramArguments</key><array><string>{python}</string><string>-m</string><string>isharded</string><string>up</string><string>--no-browser</string></array>"
            # Restart on a crash only, as the systemd unit does, so isharded stop stays stopped.
            "<key>RunAtLoad</key><true/><key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict></dict></plist>\n"
        )
        return user_home / "Library" / "LaunchAgents" / "ai.isharded.node.plist", body
    body = f"[Unit]\nDescription=iSharded node\nAfter=network-online.target\n\n[Service]\nExecStart={python} -m isharded up --no-browser\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n"
    return user_home / ".config" / "systemd" / "user" / "isharded.service", body


def run_service(args: argparse.Namespace) -> int:
    """Start this node at login, for this user only. Run it on purpose: it changes what starts."""
    system = platform.system()
    path, body = service_definition(sys.executable, system, Path.home())
    if system == "Windows":
        command = ["schtasks", "/Create", "/F", "/TN", "iSharded", "/SC", "ONLOGON", "/TR", body] if args.service_command == "install" else ["schtasks", "/Delete", "/F", "/TN", "iSharded"]
        return subprocess.run(command, check=False).returncode
    assert path is not None
    start = ["systemctl", "--user"] if system == "Linux" else ["launchctl"]
    if args.service_command == "install":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        action = [*start, "enable", "--now", "isharded"] if system == "Linux" else [*start, "load", str(path)]
        return subprocess.run(action, check=False).returncode
    action = [*start, "disable", "--now", "isharded"] if system == "Linux" else [*start, "unload", str(path)]
    code = subprocess.run(action, check=False).returncode
    path.unlink(missing_ok=True)
    return code


def checks() -> list[tuple[bool, str, str]]:
    """What usually stops a node starting: each with what to do about it."""
    found: list[tuple[bool, str, str]] = []

    def check(good: bool, text: str, fix: str) -> None:
        found.append((good, text, "" if good else fix))

    version = sys.version_info
    check(version >= (3, 10), f"Python {version.major}.{version.minor}", "iSharded needs Python 3.10 or newer")
    try:
        import cryptography  # noqa: F401

        check(True, "cryptography is installed", "")
    except ImportError:
        check(False, "cryptography is missing", "pip install cryptography")
    found.append((True, "PyTorch is installed" if has_torch() else "No PyTorch, so this node runs GGUF models only", "" if has_torch() else 'To run safetensors models too: pip install "isharded[torch]"'))
    address = detect_address()
    private = _private_link(address)
    check(True, f"This machine would advertise {address}" + (" (encrypted by Tailscale)" if private else ""), "")
    if not private:
        found.append((True, "No Tailscale address found", "Nodes will use TLS with their own certificates. Tailscale is the easier way to link machines in different places"))
    if running_node(up_home()):
        found.append((True, "A node is running here", ""))
    else:
        with socket.socket() as probe:
            free = probe.connect_ex(("127.0.0.1", DEFAULT_PORT)) != 0
        check(free, f"Port {DEFAULT_PORT} is free" if free else f"Port {DEFAULT_PORT} is in use", "Stop what uses it, or run isharded up --port with another port")
    memory = ram()
    check(memory.available > 2 * 1024**3, f"{memory.available / 1024**3:.1f} GiB of RAM free", "A node needs at least 2 GiB free")
    disk = shutil.disk_usage(str(Path.home()))
    check(disk.free > 10 * 1024**3, f"{disk.free / 1024**3:.0f} GiB of disk free", "Models need room. Free some disk first")
    folders = model_folders()
    found.append((True, f"Found models in {', '.join(folders)}" if folders else "No local model folders found", "" if folders else "Models will download when you load one"))
    has_engine, engine_note = native_engine()
    found.append((True, f"Native engine: {engine_note}" if has_engine else f"No native engine: {engine_note}", "" if has_engine else ENGINE_ADVICE))
    if platform.system() == "Linux":
        missing = not any(Path(folder, "libgomp.so.1").exists() for folder in ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64", "/usr/lib"))
        check(not missing, "libgomp is installed" if not missing else "libgomp is missing", "sudo apt-get install libgomp1")
    return found


def run_doctor(_: argparse.Namespace) -> int:
    results = checks()
    for good, text, advice in results:
        print(f"{'ok  ' if good else 'FIX '} {text}" + (f"\n     {advice}" if advice else ""))
    bad = sum(1 for good, _, _ in results if not good)
    print("Ready to run isharded up." if not bad else f"{bad} thing(s) to fix first.")
    return 1 if bad else 0


def add_arguments(subparsers: Any) -> None:
    up = subparsers.add_parser("up", help="start a node with everything worked out for you; the easy way in")
    up.add_argument("--invite", default=None, help="join a mesh, with a code from isharded invite")
    up.add_argument("--name", default=None, help="this node's name; default is the machine's name")
    up.add_argument("--worker-port", type=int, default=None, help="this node's worker port, for clouds that hand out ports; default is nine below --port")
    up.add_argument("--port", type=int, default=DEFAULT_PORT, help="port for the chat page and API")
    up.add_argument("--address", default=None, help="the address other nodes reach this one on; found for you if not set")
    up.add_argument("--location", type=country_code, default=None, help="two letter country code for the dashboard's flag, or none to show no flag, kept for next time; default is looked up from --address")
    up.add_argument("--reach-hosts", default=None, help="other hosts this machine answers on, comma separated, tried between stages; found for you if not set, none uses only --address")
    up.add_argument("--tls", action="store_true", help="use TLS even on Tailscale")
    up.add_argument("--trusted-network", action="store_true", help="skip TLS, because nobody else can read this network")
    up.add_argument("--no-gpu", action="store_true", help="contribute CPU and RAM only")
    up.add_argument("--gpu-gb", type=float, default=0, help="VRAM to give; default is a share of what is free")
    up.add_argument("--ram-gb", type=float, default=0, help="RAM to give for weights; default is a share of what is free")
    up.add_argument("--cpu-threads", type=int, default=0, help="threads to give; default is half the cores")
    up.add_argument("--gguf-dir", action="append", default=None, help="a folder of GGUF files; LM Studio's is found for you. Repeatable")
    up.add_argument("--model", default=None, help="load this model once the node is up")
    up.add_argument("--no-browser", action="store_true", help="do not open the chat page")
    up.add_argument("--background", action="store_true", help="leave the node running on its own and return once it is up; if it is already up, only open its page")
    up.add_argument("--find", action="store_true", help="ask to join the one mesh on this network, without asking you first")
    up.add_argument("--new-mesh", action="store_true", help="start a new mesh even if one is on this network")
    up.add_argument("--no-discovery", action="store_true", help="do not look for a mesh, and do not answer machines looking for this one")
    up.add_argument("--ownerless", action="store_true", help="make or join a network nobody owns")
    up.add_argument("--joining", choices=["open", "vouched"], default="vouched", help="with --ownerless: who this node accepts")
    up.add_argument("--trust", action="append", default=[], help="with --ownerless: a node to accept from the start; repeatable")
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
    subparsers.add_parser("stop", help="stop the node isharded up started on this machine").set_defaults(handler=run_stop)
    service = subparsers.add_parser("service", help="start this node when you log in")
    service_commands = service.add_subparsers(dest="service_command", required=True)
    service_commands.add_parser("install", help="start at login, for this user").set_defaults(handler=run_service)
    service_commands.add_parser("remove", help="stop starting at login").set_defaults(handler=run_service)

