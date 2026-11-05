from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import httpx

from .pool import load_token
from .node import add_node_arguments, run_node


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


def run_demo(args: argparse.Namespace) -> int:
    base_env = os.environ.copy()
    if args.pool_token:
        base_env["SHARDMESH_POOL_TOKEN"] = args.pool_token
    orchestrator_url = f"http://127.0.0.1:{args.port}"
    processes: list[subprocess.Popen[bytes]] = []
    headers = {"Authorization": f"Bearer {args.pool_token}"} if args.pool_token else {}
    try:
        processes.append(
            subprocess.Popen(
                [sys.executable, "-m", "shardmesh.orchestrator", "--host", "127.0.0.1", "--port", str(args.port)],
                env=base_env,
            )
        )
        wait_for(orchestrator_url + "/health")
        for index in range(args.workers):
            worker_port = args.port + 100 + index
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "shardmesh.worker",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(worker_port),
                        "--worker-id",
                        f"demo-{index + 1}",
                        "--orchestrator-url",
                        orchestrator_url,
                        "--advertise-url",
                        f"http://127.0.0.1:{worker_port}",
                        "--device",
                        args.device,
                    ],
                    env=base_env,
                )
            )
            wait_for(f"http://127.0.0.1:{worker_port}/health")

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = httpx.get(orchestrator_url + "/admin/status", headers=headers).json()
            if len(status["workers"]) == args.workers:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("workers did not register")

        print(f"Loading {args.model}; the first run may download model files...", flush=True)
        response = httpx.post(
            orchestrator_url + "/admin/load",
            json={"model_id": args.model},
            headers=headers,
            timeout=None,
        )
        response.raise_for_status()
        print("\nShardMesh demo is ready.", flush=True)
        print(f"Chat:   {orchestrator_url}/chat", flush=True)
        print(f"Mesh:   {orchestrator_url}/dashboard", flush=True)
        print(f"API:    {orchestrator_url}/v1", flush=True)
        print(f"Status: {orchestrator_url}/admin/status", flush=True)
        print("Press Ctrl+C to stop every local process.\n", flush=True)
        escaped_model = args.model.replace("'", "''")
        print("Try in PowerShell:", flush=True)
        print(
            "$body = @{ "
            f"model = '{escaped_model}'; "
            "prompt = 'Once upon a time'; max_tokens = 20 } | ConvertTo-Json",
            flush=True,
        )
        print(
            f"Invoke-RestMethod -Method Post -Uri {orchestrator_url}/v1/completions "
            "-ContentType 'application/json' -Body $body",
            flush=True,
        )
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


def run_load(args: argparse.Namespace) -> int:
    """Load a model on a running pool, printing progress until it finishes."""
    base = args.url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.pool_token}"} if args.pool_token else {}
    body: dict[str, object] = {
        "model_id": args.model,
        "wait": False,
        "placement": args.placement,
        "context_tokens": args.context_tokens,
    }
    if args.entry:
        body["entry_worker"] = args.entry
    if args.exit:
        body["exit_worker"] = args.exit
    if args.no_reuse_cached:
        body["reuse_cached"] = False
    if args.parallel:
        body["parallel"] = args.parallel
    if args.no_draft:
        body["draft_model"] = "none"
    elif args.draft:
        body["draft_model"] = args.draft
    if args.no_peer_fetch:
        body["peer_fetch"] = False
    response = httpx.post(base + "/admin/load", json=body, headers=headers, timeout=120.0)
    if response.status_code >= 400:
        print(f"Load refused: {response.text}", file=sys.stderr)
        return 1
    while True:
        time.sleep(2)
        job = httpx.get(base + "/admin/status", headers=headers, timeout=30.0).json()["load_job"]
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


def run_cordon(args: argparse.Namespace) -> int:
    """Take a node out of service, or put it back. Replies finish first."""
    token = args.token or load_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    action = "cordon" if args.command == "cordon" else "uncordon"
    url = f"{args.url.rstrip('/')}/admin/nodes/{args.name}/{action}"
    params = {"force": "true"} if getattr(args, "force", False) else {}
    response = httpx.post(url, params=params, headers=headers, timeout=30, trust_env=False)
    if response.status_code != 200:
        print(response.json().get("detail", response.text))
        return 1
    print(f"{args.name} is {response.json()['status']}. Watch the move with the dashboard.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="shardmesh")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run an orchestrator and local workers")
    demo.add_argument("--workers", type=int, default=2)
    demo.add_argument("--model", default="distilbert/distilgpt2")
    demo.add_argument("--device", default="auto")
    demo.add_argument("--port", type=int, default=8000)
    demo.add_argument("--pool-token", default=os.getenv("SHARDMESH_POOL_TOKEN"))
    demo.set_defaults(handler=run_demo)
    pool = subparsers.add_parser("pool", help="create a pool's credentials")
    pool_commands = pool.add_subparsers(dest="pool_command", required=True)
    init = pool_commands.add_parser("init", help="make a pool token and a private authority")
    init.add_argument("--dir", required=True, help="keep this folder off the nodes: it signs their certificates")
    issue = pool_commands.add_parser("add-node", help="issue a node its certificate")
    issue.add_argument("name")
    issue.add_argument("--dir", required=True)
    issue.add_argument("--address", action="append", required=True, help="an address peers use for this node; repeatable")

    key = subparsers.add_parser("key", help="manage API keys")
    key_commands = key.add_subparsers(dest="key_command", required=True)
    create = key_commands.add_parser("create", help="add a key; it is shown once")
    create.add_argument("name")
    create.add_argument("--role", choices=["user", "admin"], default="user")
    create.add_argument("--file", required=True, help="the key file nodes read with --keys-file")
    create.add_argument("--requests-per-minute", type=int, default=0, help="0 is no limit")

    for name, summary in (("cordon", "move a node's share elsewhere, for maintenance"), ("uncordon", "put a node back in service")):
        change = subparsers.add_parser(name, help=summary)
        change.add_argument("name", help="the node's --node-id")
        change.add_argument("--url", default="http://127.0.0.1:8000", help="any node's control address")
        change.add_argument("--token", default=None, help="an admin key; defaults to this machine's pool token")
        if name == "cordon":
            change.add_argument("--force", action="store_true", help="go ahead even if this moves a lot of data between nodes")
        change.set_defaults(handler=run_cordon)
    load = subparsers.add_parser("load", help="load a model on a running pool")
    load.add_argument("model")
    load.add_argument("--url", default="http://127.0.0.1:8000", help="orchestrator address")
    load.add_argument("--placement", choices=["spread", "compact"], default="spread")
    load.add_argument("--context-tokens", type=int, default=4096)
    load.add_argument("--entry", help="worker to hold the entry stage")
    load.add_argument("--exit", help="worker to hold the exit stage")
    load.add_argument("--no-reuse-cached", action="store_true", help="plan as if no worker held any shards")
    load.add_argument("--draft", help="small GGUF for speculative decoding; chosen automatically if not set")
    load.add_argument("--no-draft", action="store_true", help="turn speculative decoding off")
    load.add_argument("--parallel", type=int, help="replies to run at once; each costs its own context memory")
    load.add_argument("--no-peer-fetch", action="store_true", help="download every shard from the model host")
    load.add_argument("--pool-token", default=load_token())
    load.set_defaults(handler=run_load)
    node = subparsers.add_parser("node", help="contribute CPU, RAM, GPU, or any combination")
    add_node_arguments(node)
    node.set_defaults(handler=run_node)
    args = parser.parse_args()
    if args.command == "pool":
        from .pool import add_node, init_pool

        if args.pool_command == "init":
            folder = init_pool(args.dir)
            print(f"Created pool credentials in {folder}. Copy pool.token to each node, and keep ca.key off them.")
        else:
            certificate, key_file = add_node(args.dir, args.name, args.address)
            print(f"Issued {certificate} and {key_file}. Give the node those and ca.pem.")
        return
    if args.command == "key":
        from .access import create_key

        key = create_key(args.file, args.name, args.role, args.requests_per_minute)
        print(f"Key for {args.name} ({args.role}). Save it now: it is stored only as a hash.")
        print(key)
        return
    if args.command == "demo" and args.workers < 1:
        parser.error("--workers must be at least 1")
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()

