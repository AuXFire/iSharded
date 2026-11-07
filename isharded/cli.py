from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
from typing import Any

import httpx

from .access import create_key, digest, new_key
from .assistant.cli import add_arguments as add_assistant_arguments, run as run_assistant
from .commons import make_identity, sign_vouch, write_policy
from .node import add_node_arguments, run_node
from .pool import add_node, approve_membership, bearer_headers, init_pool, load_token, read_membership_request, request_membership, sign_entry
from .statements import key_verifier
from .up import add_arguments as add_up_arguments

ASK_AGAIN_SECONDS = 600.0
LOCAL_CONTROL = "http://127.0.0.1:8000"
ANY_NODE = "any node's control address"


def wait_for(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            response.raise_for_status()
            return
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


def wait_for_workers(url: str, headers: dict[str, str], count: int) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        answer = httpx.get(url + "/admin/status", headers=headers)
        answer.raise_for_status()
        if len(answer.json()["workers"]) == count:
            return
        time.sleep(0.5)
    raise RuntimeError("workers did not register")


def print_demo_guide(url: str, model: str) -> None:
    escaped_model = model.replace("'", "''")
    print(
        "\niSharded demo is ready.\n"
        f"Chat:   {url}/chat\n"
        f"Mesh:   {url}/dashboard\n"
        f"API:    {url}/v1\n"
        f"Status: {url}/admin/status\n"
        "Press Ctrl+C to stop every local process.\n\n"
        "Try in PowerShell:\n"
        f"$body = @{{ model = '{escaped_model}'; prompt = 'Once upon a time'; max_tokens = 20 }} | ConvertTo-Json\n"
        f"Invoke-RestMethod -Method Post -Uri {url}/v1/completions -ContentType 'application/json' -Body $body",
        flush=True,
    )


def run_demo(args: argparse.Namespace) -> int:
    # A demo stands alone. A real pool's keys on this machine must not lock its
    # owner out of it, and a saved route must not come back to life inside it.
    scratch = Path(tempfile.mkdtemp(prefix="isharded-demo-"))
    base_env = os.environ.copy()
    for name in ("ISHARDED_POOL_TOKEN", "ISHARDED_CA_FILE", "ISHARDED_IDENTITY_CERT", "ISHARDED_IDENTITY_KEY", "ISHARDED_TLS_CERT", "ISHARDED_TLS_KEY"):
        base_env.pop(name, None)
    base_env["ISHARDED_POOL_TOKEN_FILE"] = str(scratch / "no-token")
    if args.pool_token:
        base_env["ISHARDED_POOL_TOKEN"] = args.pool_token
    orchestrator_url = f"http://127.0.0.1:{args.port}"
    processes: list[subprocess.Popen[bytes]] = []
    headers = bearer_headers(args.pool_token)

    def start(module: str, *arguments: str) -> None:
        processes.append(subprocess.Popen([sys.executable, "-m", module, "--host", "127.0.0.1", *arguments], env=base_env))

    try:
        start(
            "isharded.orchestrator", "--port", str(args.port),
            "--keys-file", str(scratch / "access.json"),
            "--state-file", str(scratch / "state.json"),
            "--audit-file", str(scratch / "audit.jsonl"),
        )  # fmt: skip
        wait_for(orchestrator_url + "/health")
        for index in range(args.workers):
            worker_url = f"http://127.0.0.1:{args.port + 100 + index}"
            start(
                "isharded.worker", "--port", str(args.port + 100 + index),
                "--worker-id", f"demo-{index + 1}",
                "--orchestrator-url", orchestrator_url,
                "--advertise-url", worker_url,
                "--device", args.device,
            )  # fmt: skip
            wait_for(worker_url + "/health")
        wait_for_workers(orchestrator_url, headers, args.workers)

        print(f"Loading {args.model}; the first run may download model files...", flush=True)
        response = httpx.post(orchestrator_url + "/admin/load", json={"model_id": args.model}, headers=headers, timeout=None)
        response.raise_for_status()
        print_demo_guide(orchestrator_url, args.model)
        # Keep the pool alive when a worker exits so the orchestrator can prove
        # its failover path. Only an orchestrator exit ends the demo.
        while processes[0].poll() is None:
            time.sleep(1)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        shutil.rmtree(scratch, ignore_errors=True)


def load_body(args: argparse.Namespace) -> dict[str, object]:
    """What a load asks for. A flag left unset leaves the pool's own default standing."""
    body: dict[str, object] = {
        "model_id": args.model,
        "wait": False,
        "placement": args.placement,
        "context_tokens": args.context_tokens,
        "merge_passes": args.merge_passes,
        "prompt_cache": args.prompt_cache,
        "kv_type": args.kv_type,
        "model_hashes": json.loads(Path(args.hashes).read_text(encoding="utf-8")) if args.hashes else None,
    }
    for field, value in (("entry_worker", args.entry), ("exit_worker", args.exit), ("parallel", args.parallel), ("mmproj", args.mmproj)):
        if value:
            body[field] = value
    if args.no_reuse_cached:
        body["reuse_cached"] = False
    if args.no_peer_fetch:
        body["peer_fetch"] = False
    if args.no_draft:
        body["draft_model"] = "none"
    elif args.draft:
        body["draft_model"] = args.draft
    if args.embedding:
        body["embedding"] = True
    if args.lane:
        body.update({"lane": args.lane, "lane_gpu_gb": args.lane_gpu_gb, "lane_ram_gb": args.lane_ram_gb, "lane_apart": args.lane_apart})
    return body


def run_load(args: argparse.Namespace) -> int:
    """Load a model on a running pool, printing progress until it finishes."""
    base = args.url.rstrip("/")
    headers = bearer_headers(args.pool_token)
    response = httpx.post(base + "/admin/load", json=load_body(args), headers=headers, timeout=120.0)
    if response.status_code >= 400:
        print(f"Load refused: {response.text}", file=sys.stderr)
        return 1
    while True:
        time.sleep(2)
        try:
            answer = httpx.get(base + "/admin/status", headers=headers, timeout=30.0)
            status = answer.json()
        except (httpx.HTTPError, ValueError):
            continue  # a busy or changing coordinator is no reason to give up on the load
        if answer.status_code >= 500:
            continue  # nobody coordinates this moment, so this is not a status
        job = status.get("load_job")
        if job is None and not args.lane:
            continue
        if args.lane:
            lane = next((item for item in status["lanes"] if item["name"] == args.lane), {})
            if lane.get("error"):
                print(lane["error"], file=sys.stderr)
                return 1
            if lane.get("ready") and lane.get("model") == args.model:
                print(f"Lane {args.lane} is ready.", flush=True)
                return 0  # a change of coordinator can end the job that reported progress
            job = lane.get("load_job")
            if job is None or job.get("model") != args.model:
                print(f"Starting lane {args.lane}: {lane.get('workers', 0)} workers so far.", flush=True)
                continue
        gib = 1024**3
        print(
            f"{job['phase']}: {job['bytes_done'] / gib:.2f} of {job['bytes_total'] / gib:.2f} GiB, "
            f"{job['bytes_from_peers'] / gib:.2f} GiB from peers, {job['bytes_reused'] / gib:.2f} GiB reused",
            flush=True,
        )
        if job["phase"] == "ready":
            return 0
        if job["phase"] in {"failed", "cancelled"}:
            print(job["error"], file=sys.stderr)
            return 1


def show_answer(response: httpx.Response, done: str) -> int:
    """Say how a request went, and return the exit code for it."""
    good = response.status_code < 400
    print(done if good else f"Refused: {response.text}")
    return 0 if good else 1


def run_net(args: argparse.Namespace) -> int:
    if args.net_command == "init":
        name = make_identity(args.dir)
        write_policy(args.dir, args.joining, args.quorum, args.trust)
        print(f"This node is {name}. Give that name to anyone who should vouch for it or trust it.")
        return 0
    url = args.url.rstrip("/")
    if args.net_command == "vouch":
        return show_answer(httpx.post(url + "/access/entries", json=sign_vouch(args.dir, args.name), timeout=30.0), f"Vouched for {args.name}.")
    headers = bearer_headers(args.pool_token)
    if args.net_command == "show":
        response = httpx.get(url + "/admin/network", headers=headers, timeout=30.0)
        print(json.dumps(response.json(), indent=2) if response.status_code < 400 else f"Refused: {response.text}")
        return 0 if response.status_code < 400 else 1
    method = httpx.post if args.net_command == "block" else httpx.delete
    return show_answer(method(f"{url}/admin/nodes/{args.name}/revoke", headers=headers, timeout=30.0), f"Done: {args.net_command} {args.name}.")


def run_audit(args: argparse.Namespace) -> int:
    headers = bearer_headers(args.pool_token)
    while True:
        response = httpx.post(args.url.rstrip("/") + "/admin/audit", json={"model": args.model, "samples": args.samples}, headers=headers, timeout=1800.0)
        if response.status_code >= 400:
            print(f"Refused: {response.text}", file=sys.stderr)
            return 1
        result = response.json()
        for pair in result["pairs"]:
            print(f"{pair['routes'][0]} and {pair['routes'][1]}: {pair['agreement']:.0%} agreement over {result['samples']} questions")
        if result["verdict"] == "agree":
            print("The routes agree.")
        elif result["suspects"]:
            for name in result["suspects"]:
                print(f"{name} disagrees with the others. It runs on: {', '.join(result['routes'].get(name, []))}")
        else:
            print("The routes disagree, and with two it cannot be said which is wrong. Load a third with --lane-apart.")
        if not args.every:
            return 0 if result["verdict"] == "agree" else 1
        time.sleep(args.every)


def run_lane(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    headers = bearer_headers(args.pool_token)
    if args.lane_command == "remove":
        return show_answer(httpx.delete(f"{base}/admin/lanes/{args.name}", headers=headers, timeout=30.0), f"Removed lane {args.name}.")
    for lane in httpx.get(base + "/admin/status", headers=headers, timeout=30.0).json()["lanes"]:
        state = "ready" if lane["ready"] else lane["error"] or "starting"
        print(f"{lane['name']}: {lane['model'] or 'no model'}, {lane['workers']} workers, {state}")
    return 0


def send_signed(args: argparse.Namespace, entry: dict[str, Any]) -> None:
    """Sign a change with the pool authority and hand it to any node."""
    authority = str(Path(args.dir) / "ca.pem")
    response = httpx.post(args.url.rstrip("/") + "/access/entries", json=sign_entry(args.dir, entry), verify=ssl.create_default_context(cafile=authority), timeout=30, trust_env=False)
    if response.status_code != 200:
        raise SystemExit(f"{response.status_code}: {response.text[:200]}")


def run_model_hash(args: argparse.Namespace) -> int:
    """Hash a model the way workers build it. Reads all of it, so run it near the file."""
    from .models import create_adapter
    from .weights import build_manifest, hash_model, open_reader

    manifest = build_manifest(args.source)
    units = create_adapter(manifest.config).partition(manifest.tensors)
    out = Path(args.out).resolve()
    reader = open_reader(args.source)
    try:
        hashes = hash_model(manifest, units, reader, out.parent)
    finally:
        reader.close()
    out.write_text(json.dumps(hashes, indent=1), encoding="utf-8")
    print(f"Hashed {len(units)} units into {out}.")
    return 0


def request_join(args: argparse.Namespace, url: str) -> int:
    request = request_membership(args.dir, args.name, args.address)
    answer = httpx.post(url + "/join/requests", json={"request": request}, timeout=30, trust_env=False)
    if answer.status_code != 200:
        print(answer.json().get("detail", answer.text))
        return 1
    state = answer.json()["status"]
    if state in {"refused", "busy"}:
        print("This pool has refused this node." if state == "refused" else "Too many nodes are waiting. Try again later.")
        return 1
    print(f"Asked to join as {args.name}. Tell the owner this key fingerprint: {answer.json()['fingerprint']}")
    asked_at = time.monotonic()
    while args.wait:
        found = httpx.get(f"{url}/join/certificate/{args.name}", timeout=30, trust_env=False)
        if found.status_code == 200:
            folder = Path(args.dir)
            (folder / f"{args.name}.pem").write_text(found.json()["certificate"], encoding="utf-8")
            (folder / "ca.pem").write_text(found.json()["authority"], encoding="utf-8")
            print(f"Approved. Start with --ca-file, --identity-cert, and --identity-key from {folder}.")
            return 0
        if time.monotonic() - asked_at > ASK_AGAIN_SECONDS:  # a request nobody answers expires
            httpx.post(url + "/join/requests", json={"request": request}, timeout=30, trust_env=False)
            asked_at = time.monotonic()
        time.sleep(5)
    return 0


def run_membership(args: argparse.Namespace) -> int:
    """Asking to join a pool, and answering."""
    url = args.url.rstrip("/")
    if args.pool_command == "request-join":
        return request_join(args, url)
    waiting = httpx.get(url + "/join/requests", headers={"Authorization": f"Bearer {args.token}"}, timeout=30, trust_env=False)
    if waiting.status_code != 200:
        print(waiting.text[:200])
        return 1
    requests = {item["name"]: item for item in waiting.json()["requests"]}
    if args.pool_command == "requests":
        for item in requests.values():
            print(f"{item['name']}  key {item['fingerprint']}  from {', '.join(item['addresses'])}")
        print(f"{len(requests)} waiting.")
        return 0
    held = requests.get(args.name)
    if held is None:
        print(f"{args.name} has not asked to join.")
        return 1
    if args.pool_command == "deny":
        send_signed(args, {"type": "deny", "name": args.name, "fingerprint": read_membership_request(held["request"])["fingerprint"]})
        print(f"Denied {args.name}.")
        return 0
    send_signed(args, {"type": "member", "name": args.name, "certificate": approve_membership(args.dir, held["request"])})
    print(f"Approved {args.name}, key {held['fingerprint']}.")
    return 0


def run_pool(args: argparse.Namespace) -> None:
    if args.pool_command == "revoke":
        send_signed(args, {"type": "revoke", "name": args.name})
        print(f"{args.name} is revoked.")
    elif args.pool_command == "init":
        folder = init_pool(args.dir)
        print(f"Created pool credentials in {folder}. For a shared-token pool, copy pool.token to each node. With certificates, nodes need only ca.pem and their own files. Keep ca.key off them.")
    else:
        certificate, key_file = add_node(args.dir, args.name, args.address)
        print(f"Issued {certificate} and {key_file}. Give the node those and ca.pem.")


def run_key(args: argparse.Namespace) -> None:
    if args.key_command == "remove":
        send_signed(args, {"type": "remove_key", "name": args.name})
        print(f"Removed {args.name}.")
        return
    if args.dir:
        key = new_key()
        send_signed(args, {"type": "key", "name": args.name, "role": args.role, "sha256": digest(key), "requests_per_minute": args.requests_per_minute, "verifier": key_verifier(key)})
    else:
        key = create_key(args.file, args.name, args.role, args.requests_per_minute)
    print(f"Key for {args.name} ({args.role}). Save it now: it is stored only as a hash.")
    print(key)


def run_cordon(args: argparse.Namespace) -> int:
    """Take a node out of service, or put it back. Replies finish first."""
    url = f"{args.url.rstrip('/')}/admin/nodes/{args.name}/{args.command}"
    params = {"force": "true"} if getattr(args, "force", False) else {}
    response = httpx.post(url, params=params, headers=bearer_headers(args.token or load_token()), timeout=30, trust_env=False)
    if response.status_code != 200:
        print(response.json().get("detail", response.text))
        return 1
    print(f"{args.name} is {response.json()['status']}. Watch the move with the dashboard.")
    return 0


def run_mcp(args: argparse.Namespace) -> int:
    """Serve a mesh to an MCP client over stdio. Stdout belongs to the protocol."""
    import asyncio

    from .mcp_server import run_server

    asyncio.run(run_server(args.url, args.token))
    return 0


def run_mcp_agent(args: argparse.Namespace) -> int:
    from .mcp_client import run_agent

    return run_agent(args)


def add_url(parser: argparse.ArgumentParser, text: str = ANY_NODE) -> None:
    parser.add_argument("--url", default=LOCAL_CONTROL, help=text)


def add_token(parser: argparse.ArgumentParser, token: str | None, text: str) -> None:
    parser.add_argument("--token", "--pool-token", dest="pool_token", default=token, help=text)


def add_secrets_folder(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dir", required=True, help="the pool's secrets folder")


def add_mcp_commands(subparsers: Any, token: str | None) -> None:
    mcp = subparsers.add_parser("mcp", help="serve a mesh to MCP clients over stdio, or give a model the tools of MCP servers")
    add_url(mcp)
    mcp.add_argument("--token", default=token, help="a user key; defaults to this machine's pool token")
    mcp.set_defaults(handler=run_mcp)
    actions = mcp.add_subparsers(dest="mcp_command")
    run = actions.add_parser("run", help="let a model on the mesh use the tools of MCP stdio servers, and print its answer")
    run.add_argument("prompt", help="what to ask the model")
    # Unset here means the value given before `run` stands.
    run.add_argument("--url", default=argparse.SUPPRESS, help=ANY_NODE)
    run.add_argument("--token", default=argparse.SUPPRESS, help="a user key; defaults to this machine's pool token")
    run.add_argument("--model", required=True, help="the model to use; it must support tool calling")
    run.add_argument("--server", action="append", required=True, help="the command that starts an MCP stdio server; repeatable")
    run.add_argument("--allow", action="append", default=[], help="a tool that may run without asking; repeatable")
    run.add_argument("--pass-env", action="append", default=[], help="an environment variable servers may see, beyond the basics; repeatable")
    run.add_argument("--max-turns", type=int, default=8, help="most times the model may be asked before giving up")
    run.add_argument("--max-tokens", type=int, default=None, help="longest reply on each turn")
    run.add_argument("--tool-timeout", type=float, default=60.0, help="seconds a tool may take")
    run.set_defaults(handler=run_mcp_agent)


def add_pool_commands(subparsers: Any) -> None:
    pool = subparsers.add_parser("pool", help="create a pool's credentials")
    pool.set_defaults(handler=run_pool)
    pool_commands = pool.add_subparsers(dest="pool_command", required=True)
    init = pool_commands.add_parser("init", help="make a pool token and a private authority")
    init.add_argument("--dir", required=True, help="keep this folder off the nodes: it signs their certificates")
    issue = pool_commands.add_parser("add-node", help="issue a node its certificate")
    issue.add_argument("name", help="the node's --node-id")
    add_secrets_folder(issue)
    issue.add_argument("--address", action="append", required=True, help="an address peers use for this node; repeatable")
    ask = pool_commands.add_parser("request-join", help="ask to join a pool; the owner decides")
    ask.add_argument("name", help="this node's --node-id")
    ask.add_argument("--dir", required=True, help="where to keep this node's key and certificate")
    ask.add_argument("--address", action="append", required=True, help="an address peers use for this node; repeatable")
    ask.add_argument("--url", required=True, help="any member's control address")
    ask.add_argument("--wait", action="store_true", help="wait for the owner's answer")
    ask.set_defaults(handler=run_membership)
    listing = pool_commands.add_parser("requests", help="show who is asking to join")
    add_url(listing)
    listing.add_argument("--token", required=True, help="an admin key")
    listing.set_defaults(handler=run_membership)
    for name, summary in (("approve", "let a node join"), ("deny", "turn down a node's request")):
        answer = pool_commands.add_parser(name, help=summary)
        answer.add_argument("name", help="the node asking to join")
        add_secrets_folder(answer)
        add_url(answer)
        answer.add_argument("--token", required=True, help="an admin key, to read the request")
        answer.set_defaults(handler=run_membership)
    revoke = pool_commands.add_parser("revoke", help="throw a node out for good, signed by the pool authority")
    revoke.add_argument("name", help="the node to throw out")
    add_secrets_folder(revoke)
    add_url(revoke)


def add_key_commands(subparsers: Any) -> None:
    key = subparsers.add_parser("key", help="manage API keys")
    key.set_defaults(handler=run_key)
    key_commands = key.add_subparsers(dest="key_command", required=True)
    create = key_commands.add_parser("create", help="add a key; it is shown once")
    create.add_argument("name", help="who the key is for")
    create.add_argument("--role", choices=["user", "admin"], default="user", help="what the key may do")
    create.add_argument("--file", help="the key file nodes read with --keys-file")
    create.add_argument("--dir", help="the pool's secrets folder: sign this with its authority and send it to --url")
    add_url(create, "any node's control address, with --dir")
    create.add_argument("--requests-per-minute", type=int, default=0, help="0 is no limit")
    remove = key_commands.add_parser("remove", help="remove a key for good, signed by the pool authority")
    remove.add_argument("name", help="the key's name")
    add_secrets_folder(remove)
    add_url(remove)


def add_load_command(subparsers: Any, token: str | None) -> None:
    load = subparsers.add_parser("load", help="load a model on a running pool")
    load.add_argument("model", help="owner/repo, owner/repo:file.gguf, a local path, or a file name another node holds")
    add_url(load)
    load.add_argument("--placement", choices=["auto", "spread", "compact"], default="auto", help="the fewest close workers that hold the model in VRAM, every worker, or the fewest that fit")
    load.add_argument("--context-tokens", type=int, default=4096, help="tokens each reply can hold")
    load.add_argument("--entry", help="worker to hold the entry stage")
    load.add_argument("--exit", help="worker to hold the exit stage")
    load.add_argument("--no-reuse-cached", action="store_true", help="plan as if no worker held any shards")
    load.add_argument("--draft", help="small GGUF for speculative decoding; chosen automatically if not set")
    load.add_argument("--no-draft", action="store_true", help="turn speculative decoding off")
    load.add_argument("--parallel", type=int, help="replies to run at once; each costs its own context memory")
    load.add_argument("--no-peer-fetch", action="store_true", help="download every shard from the model host")
    add_token(load, token, "an admin key; defaults to this machine's pool token")
    load.add_argument("--no-merge-passes", dest="merge_passes", action="store_false", help="make conversations take turns on each stage")
    load.add_argument("--no-prompt-cache", dest="prompt_cache", action="store_false", help="read every prompt in full instead of reusing saved states")
    load.add_argument("--kv-type", choices=["f16", "q8_0"], default="f16", help="how a GGUF model keeps attention KV; q8_0 takes about half the memory")
    load.add_argument("--lane", default=None, help="load beside the running model, in a lane of this name")
    load.add_argument("--lane-gpu-gb", type=float, default=None, help="most VRAM per node for this lane; default is what is spare")
    load.add_argument("--lane-ram-gb", type=float, default=None, help="most RAM per node for this lane; default is what is spare")
    load.add_argument("--lane-apart", action="store_true", help="keep this lane off every node another route uses, so routes can check each other")
    load.add_argument("--hashes", default=None, help="file from `isharded model hash`: every node checks every unit against it")
    load.add_argument("--embedding", action="store_true", help="this is an embedding model: one stage, for /v1/embeddings and not chat")
    load.add_argument("--mmproj", default=None, help="projector GGUF for images, hosted on the entry stage; found by name like other GGUF files")
    load.set_defaults(handler=run_load)


def add_net_commands(subparsers: Any, token: str | None) -> None:
    net = subparsers.add_parser("net", help="take part in a network nobody owns")
    net.set_defaults(handler=run_net)
    net_commands = net.add_subparsers(dest="net_command", required=True)
    start = net_commands.add_parser("init", help="make this node's identity, and choose who it accepts")
    start.add_argument("--dir", required=True, help="where this node keeps its identity")
    start.add_argument("--joining", choices=["open", "vouched"], default="open", help="accept anyone, or only nodes others vouched for")
    start.add_argument("--quorum", type=int, default=1, help="vouches a node needs from nodes this one accepts")
    start.add_argument("--trust", action="append", default=[], help="a node to accept from the start; repeatable")
    vouch = net_commands.add_parser("vouch", help="say this node accepts another, for nodes that accept this one")
    vouch.add_argument("name", help="the node to vouch for")
    vouch.add_argument("--dir", required=True, help="this node's identity folder")
    add_url(vouch)
    show = net_commands.add_parser("show", help="this node's view: who it accepts, blocks, and can see")
    add_url(show, "this node's control address")
    add_token(show, token, "a key for this node")
    for name, text in (("block", "stop this node dealing with another; only this node is affected"), ("unblock", "undo a block on this node")):
        change = net_commands.add_parser(name, help=text)
        change.add_argument("name", help="the node to block or unblock")
        add_url(change, "this node's control address")
        add_token(change, token, "this node's admin key")


def build_parser() -> argparse.ArgumentParser:
    token = load_token()
    parser = argparse.ArgumentParser(prog="isharded")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run an orchestrator and local workers")
    demo.add_argument("--workers", type=int, default=2, help="how many workers to start")
    demo.add_argument("--model", default="distilbert/distilgpt2", help="model to load once they are up")
    demo.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or another device")
    demo.add_argument("--port", type=int, default=8000, help="control plane port; workers use the next hundred up")
    demo.add_argument("--pool-token", default=os.getenv("ISHARDED_POOL_TOKEN"), help="require this token")
    demo.set_defaults(handler=run_demo)
    add_pool_commands(subparsers)
    add_key_commands(subparsers)
    for name, summary in (("cordon", "move a node's share elsewhere, for maintenance"), ("uncordon", "put a node back in service")):
        change = subparsers.add_parser(name, help=summary)
        change.add_argument("name", help="the node's --node-id")
        add_url(change)
        change.add_argument("--token", default=None, help="an admin key; defaults to this machine's pool token")
        if name == "cordon":
            change.add_argument("--force", action="store_true", help="go ahead even if this moves a lot of data between nodes")
        change.set_defaults(handler=run_cordon)
    model = subparsers.add_parser("model", help="work with model files")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    hashes = model_commands.add_parser("hash", help="hash a model on a copy you trust, for load --hashes")
    hashes.add_argument("source", help="a GGUF file or a safetensors folder")
    hashes.add_argument("--out", required=True, help="the file to write")
    hashes.set_defaults(handler=run_model_hash)
    add_load_command(subparsers, token)
    check = subparsers.add_parser("audit", help="ask routes serving a model the same questions, and compare their answers")
    check.add_argument("--model", default=None, help="the model to check; default is the main one")
    check.add_argument("--samples", type=int, default=8, help="questions to ask each route")
    check.add_argument("--every", type=float, default=0, help="keep checking, this many seconds apart")
    add_url(check)
    add_token(check, token, "an admin key for this node")
    check.set_defaults(handler=run_audit)
    lane = subparsers.add_parser("lane", help="see or remove models running beside the main one")
    lane_commands = lane.add_subparsers(dest="lane_command", required=True)
    for name, text in (("list", "show the pool's lanes"), ("remove", "stop a lane and free its memory on every node")):
        command = lane_commands.add_parser(name, help=text)
        if name == "remove":
            command.add_argument("name", help="the lane to remove")
        add_url(command)
        add_token(command, token, "an admin key; defaults to this machine's pool token")
        command.set_defaults(handler=run_lane)
    add_net_commands(subparsers, token)
    node = subparsers.add_parser("node", help="contribute CPU, RAM, GPU, or any combination")
    add_node_arguments(node)
    node.set_defaults(handler=run_node)
    add_up_arguments(subparsers)
    add_mcp_commands(subparsers, token)
    assistant = subparsers.add_parser("assistant", help="a persistent assistant that uses the mesh; see docs/ASSISTANT.md")
    add_assistant_arguments(assistant)
    assistant.set_defaults(handler=run_assistant)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "demo" and args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.command == "key" and args.key_command == "create" and not (args.dir or args.file):
        parser.error("give --file for a key file, or --dir and --url to sign one for a running pool")
    code = args.handler(args)
    if code is not None:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
