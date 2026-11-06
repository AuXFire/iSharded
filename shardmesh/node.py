from __future__ import annotations

import argparse
import copy
import ctypes
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .lanes import lane_port, lane_url, read_lanes
from .pool import check_exposure, load_token

def is_remote_path(path: str | Path) -> bool:
    """True for a UNC path, an RDP redirected drive, or a mapped network drive."""
    text = str(path)
    if text.startswith(("\\\\", "//")):
        return True
    drive = os.path.splitdrive(text)[0]
    if os.name == "nt" and drive:
        return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == 4  # DRIVE_REMOTE
    return False


def local_code_root(source_root: Path) -> Path:
    """Run workers from local disk, copying the package there if it is remote.

    Python searches every import path entry each time an import fails, and
    libraries retry optional imports inside hot paths. With a path on an RDP
    redirected drive, each search crosses the network: measured at over 500 ms
    per HTTP request. A local copy also survives the RDP session ending.
    """
    if not is_remote_path(source_root):
        return source_root
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
    target = base / "ShardMesh" / "app"
    shutil.rmtree(target / "shardmesh", ignore_errors=True)
    shutil.copytree(
        source_root / "shardmesh",
        target / "shardmesh",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    print(f"Code is on {source_root}, so workers run from a local copy in {target}.", flush=True)
    return target


def add_node_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node-id", default=socket.gethostname(), help="this node's name; keep it stable, because the lowest name coordinates")
    parser.add_argument("--orchestrator-url", default=None, help="old name for --join")
    parser.add_argument("--join", default=None, help="any existing nodes, comma separated; omit on the first node")
    parser.add_argument("--control-port", type=int, default=None, help="port for this node's control plane; default is base port + 9")
    parser.add_argument("--no-control", action="store_true", help="worker only: do not run a control plane here")
    parser.add_argument("--tls-cert", default=None, help="certificate for this node's servers")
    parser.add_argument("--tls-key", default=None, help="key for that certificate")
    parser.add_argument("--ca-file", default=None, help="authority that signed the other nodes' certificates")
    parser.add_argument("--identity-cert", default=None, help="this node's certificate, to prove its name without TLS")
    parser.add_argument("--identity-key", default=None, help="the key for --identity-cert")
    parser.add_argument("--keys-file", default=None, help="where this node keeps the pool's API keys")
    parser.add_argument("--log-file", default=None, help="write this node's output here; the last one is kept as .1")
    parser.add_argument("--pool-token-file", default=None, help="file holding the pool token; default is pool.token in the ShardMesh folder")
    parser.add_argument("--open", dest="open_pool", action="store_true", help="serve a network with no credentials, trusting everyone on it")
    parser.add_argument("--trusted-network", action="store_true", help="send pool credentials unencrypted, because nobody else can read this network")
    parser.add_argument("--model-dir", action="append", default=None, help="a folder of local models peers may ask this node to load; repeatable")
    parser.add_argument("--bind-host", default="0.0.0.0", help="address to listen on")
    parser.add_argument(
        "--advertise-host",
        default="auto",
        help="the address other nodes reach this one on; auto prefers Tailscale, then the LAN",
    )
    parser.add_argument("--advertise-scheme", choices=["http", "https"], default="http", help="set to https for you by --tls-cert")
    parser.add_argument("--base-port", type=int, default=8200, help="first worker's port; later workers take the next ones")
    parser.add_argument(
        "--gpu-device",
        default="auto",
        help="auto (every accelerator), cuda:0, cuda:0,cuda:1, mps, xpu:0, or none. "
        "ROCm uses PyTorch's cuda device API",
    )
    parser.add_argument("--gpu-memory-gb", type=float, default=0, help="VRAM budget per accelerator; 0 means auto")
    parser.add_argument(
        "--dtype",
        "--gpu-dtype",
        dest="gpu_dtype",
        choices=["auto", "bfloat16", "float32", "float16"],
        default="auto",
        help="precision for this node's VRAM and RAM; auto is bfloat16, float32 is full precision",
    )
    parser.add_argument("--cpu-threads", type=int, default=0, help="threads for CPU work; with no GPU, setting this adds a CPU worker")
    parser.add_argument("--cpu-active", action="store_true", help="place CPU on normal routes instead of standby")
    parser.add_argument(
        "--weights-ram-gb",
        default=None,
        help="system RAM for model weights, or 'auto'. With a GPU, one worker spans both",
    )
    parser.add_argument("--weights-disk-gb", type=float, default=0, help="disk for memory-mapped model weights")
    parser.add_argument("--cache-dir", default=None, help="where fetched weights are kept; put it on a fast disk")
    parser.add_argument("--cache-budget-gb", type=float, default=0, help="evict old models past this size; 0 is no limit")
    parser.add_argument("--ram-budget-gb", type=float, default=0, help="RAM for KV-cache checkpoints, not weights")
    parser.add_argument("--no-peer-fetch", action="store_true", help="always download from the model host")
    parser.add_argument("--no-peer-serve", action="store_true", help="do not send shards to other workers")
    parser.add_argument("--no-local-copy", action="store_true", help="run workers from a network path as it is")
    parser.add_argument("--no-lanes", action="store_true", help="host only the main model here, never one loaded with load --lane")
    parser.add_argument("--llama-dir", default=None, help="a llama.cpp build for GGUF models; downloaded if not set")
    parser.add_argument(
        "--gguf-dir",
        action="append",
        default=None,
        help="folder to search for GGUF files already on this machine, such as LM Studio's. Repeatable",
    )


def detect_address() -> str:
    """The address peers are most likely to reach this machine on.

    A Tailscale address works from anywhere on the tailnet, so it comes first.
    After that, the address on the route out, which on a LAN or VLAN is the
    interface peers share. Connecting a UDP socket sends nothing.
    """
    import ipaddress

    import psutil

    tailnet = ipaddress.ip_network("100.64.0.0/10")
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family == socket.AF_INET and ipaddress.ip_address(address.address) in tailnet:
                return address.address
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _seeds(args: argparse.Namespace) -> list[str]:
    joined = getattr(args, "join", None) or getattr(args, "orchestrator_url", None) or ""
    return [url.strip() for url in joined.split(",") if url.strip()]


def _control_port(args: argparse.Namespace) -> int:
    port = getattr(args, "control_port", None)
    return port if port else args.base_port + 9


def _local_control(args: argparse.Namespace) -> str:
    return f"{args.advertise_scheme}://{args.advertise_host}:{_control_port(args)}"


def _control_urls(args: argparse.Namespace) -> str:
    local = [] if getattr(args, "no_control", False) else [_local_control(args)]
    return ",".join(dict.fromkeys([*local, *_seeds(args)]))


def _control_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "shardmesh.orchestrator",
        "--host",
        args.bind_host,
        "--port",
        str(_control_port(args)),
        "--node-id",
        args.node_id,
        "--advertise-url",
        _local_control(args),
        "--join",
        ",".join(_seeds(args)),
        *(["--open"] if getattr(args, "open_pool", False) else []),
        *(["--trusted-network"] if getattr(args, "trusted_network", False) else []),
        *_lane_flags(args),
    ]


def data_folder() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "ShardMesh"


def _lane_flags(args: argparse.Namespace) -> list[str]:
    """A lane keeps its own records. The node's keys stay with the front door."""
    lane = getattr(args, "lane", None)
    if not lane:
        return []
    return [
        "--lane",
        lane,
        "--state-file",
        str(lane_state_file(args.node_id, lane)),
        "--audit-file",
        str(data_folder() / f"audit-{args.node_id}-{lane}.jsonl"),
        "--keys-file",
        str(data_folder() / f"access-{args.node_id}-{lane}.json"),
    ]


def lane_state_file(node_id: str, lane: str) -> Path:
    return data_folder() / f"orchestrator-state-{node_id}-{lane}.json"


def _worker_command(
    args: argparse.Namespace,
    worker_id: str,
    port: int,
    device: str,
    roles: str,
    extra: list[str],
) -> list[str]:
    advertised_url = f"{args.advertise_scheme}://{args.advertise_host}:{port}"
    return [
        sys.executable,
        "-m",
        "shardmesh.worker",
        "--host",
        args.bind_host,
        "--port",
        str(port),
        "--worker-id",
        worker_id,
        "--orchestrator-url",
        _control_urls(args),
        "--advertise-url",
        advertised_url,
        "--device",
        device,
        "--roles",
        roles,
        *(["--open"] if getattr(args, "open_pool", False) else []),
        *(["--trusted-network"] if getattr(args, "trusted_network", False) else []),
        *extra,
    ]


def _weight_memory_flags(args: argparse.Namespace) -> list[str]:
    """Flags that give a worker memory for weights beyond its accelerators."""
    flags: list[str] = []
    weights_ram = getattr(args, "weights_ram_gb", None)
    if weights_ram not in {None, 0, "0"}:
        flags += ["--weights-ram-gb", str(weights_ram)]
    disk = getattr(args, "weights_disk_gb", 0)
    if disk:
        flags += ["--weights-disk-gb", str(disk)]
    return flags


def _cache_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if getattr(args, "cache_dir", None):
        flags += ["--cache-dir", args.cache_dir]
    if getattr(args, "cache_budget_gb", 0):
        flags += ["--cache-budget-gb", str(args.cache_budget_gb)]
    if getattr(args, "no_peer_fetch", False):
        flags.append("--no-peer-fetch")
    if getattr(args, "no_peer_serve", False):
        flags.append("--no-peer-serve")
    if getattr(args, "llama_dir", None):
        flags += ["--llama-dir", args.llama_dir]
    for folder in getattr(args, "gguf_dir", None) or []:
        flags += ["--gguf-dir", folder]
    return flags


LOG_ROTATE_BYTES = 64 * 1024 * 1024


def open_log(path: str | None) -> Any:
    """Open a node's log, starting a new one when the last has grown large."""
    if not path:
        return None
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.is_file() and file.stat().st_size > LOG_ROTATE_BYTES:
        file.replace(file.with_suffix(file.suffix + ".1"))
    return file.open("ab")


class Service:
    """One of a node's processes, restarted when it dies.

    The delay doubles while a service keeps failing, so a broken one cannot spin,
    and resets once it has stayed up long enough to count as healthy.
    """

    FIRST_DELAY_SECONDS = 2.0
    LONGEST_DELAY_SECONDS = 120.0
    HEALTHY_AFTER_SECONDS = 300.0

    def __init__(self, label: str, launch: Callable[[], Any], clock: Callable[[], float] = time.monotonic) -> None:
        self.label = label
        self.launch = launch
        self.clock = clock
        self.process: Any = None
        self.started_at = 0.0
        self.restart_at: float | None = None
        self.delay = self.FIRST_DELAY_SECONDS

    def start(self) -> None:
        self.process = self.launch()
        self.started_at = self.clock()
        self.restart_at = None

    def supervise(self) -> None:
        if self.restart_at is not None:
            if self.clock() >= self.restart_at:
                self.start()
                print(f"Restarted {self.label} (PID {self.process.pid}).", flush=True)
            return
        code = self.process.poll()
        if code is None:
            return
        if self.clock() - self.started_at >= self.HEALTHY_AFTER_SECONDS:
            self.delay = self.FIRST_DELAY_SECONDS
        self.restart_at = self.clock() + self.delay
        print(f"{self.label} exited with code {code}; restarting in {self.delay:.0f} s.", flush=True)
        self.delay = min(self.delay * 2, self.LONGEST_DELAY_SECONDS)

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


def node_specs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """What this node runs: a label and a command for each process."""
    specs: list[tuple[str, list[str]]] = []
    port = args.base_port
    gpu_device = args.gpu_device
    dtype = getattr(args, "gpu_dtype", "auto")
    weight_flags = _weight_memory_flags(args)
    cache_flags = _cache_flags(args)
    if gpu_device == "auto":
        # Import the accelerator runtime only when discovery was requested.
        # CPU/RAM-only contributors should not depend on CUDA/ROCm driver
        # initialization merely to join the pool.
        import torch

        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        gpu_device = "auto" if torch.cuda.is_available() or has_mps or has_xpu else "none"
    # A worker with weights in several memory types executes all of them
    # itself: the CPU runs the RAM and disk tiers, the GPU the VRAM tier.
    spans_tiers = gpu_device != "none" and bool(weight_flags)
    if gpu_device != "none":
        extra = ["--gpu-memory-gb", str(args.gpu_memory_gb), "--dtype", dtype, *cache_flags]
        if spans_tiers:
            extra += weight_flags
            if args.cpu_threads:
                extra += ["--cpu-threads", str(args.cpu_threads)]
        specs.append(
            (
                "GPU + RAM/disk inference" if spans_tiers else "GPU inference",
                _worker_command(args, f"{args.node_id}-gpu", port, gpu_device, "inference", extra),
            )
        )
        port += 1
    if not spans_tiers and (args.cpu_threads > 0 or weight_flags):
        extra = ["--dtype", dtype, *cache_flags, *weight_flags]
        if args.cpu_threads:
            extra += ["--cpu-threads", str(args.cpu_threads)]
        if not args.cpu_active:
            extra.append("--standby")
        specs.append(
            (
                "CPU inference" + ("" if args.cpu_active else " standby"),
                _worker_command(args, f"{args.node_id}-cpu", port, "cpu", "inference", extra),
            )
        )
        port += 1
    if args.ram_budget_gb > 0:
        specs.append(
            (
                "RAM KV cache",
                _worker_command(
                    args,
                    f"{args.node_id}-memory",
                    port,
                    "cpu",
                    "kv-cache,checkpoint",
                    ["--ram-budget-gb", str(args.ram_budget_gb)],
                ),
            )
        )
    if not specs and not getattr(args, "lane", None):
        raise ValueError("node contributes nothing; enable a GPU, CPU threads, or a RAM or disk budget")
    if not getattr(args, "no_control", False):
        specs.insert(0, ("Control plane", _control_command(args)))
    elif not _seeds(args):
        raise ValueError("a node without a control plane needs --join")
    return specs


def lane_arguments(args: argparse.Namespace, name: str, lane: dict[str, Any]) -> argparse.Namespace:
    """This node's settings for one lane: higher ports, and only what is spare."""
    budget = lane["budgets"].get(args.node_id, {})
    slot = lane["slot"]
    changed = copy.copy(args)
    changed.lane = name
    changed.base_port = lane_port(args.base_port, slot)
    changed.control_port = lane_port(_control_port(args), slot)
    changed.join = ",".join(lane_url(seed, slot) for seed in _seeds(args))
    changed.orchestrator_url = None
    changed.gpu_memory_gb = budget.get("gpu_gb", 0.0)
    changed.gpu_device = args.gpu_device if changed.gpu_memory_gb else "none"
    ram = budget.get("ram_gb", 0.0)
    changed.weights_ram_gb = str(ram) if ram else None
    changed.weights_disk_gb = 0
    changed.ram_budget_gb = 0
    changed.cpu_active = True
    if not ram:
        changed.cpu_threads = 0
    return changed


class LaneKeeper:
    """Starts and stops this node's lanes as the pool asks."""

    def __init__(self, args: argparse.Namespace, path: Path, make_service: Callable[[str, list[str]], Service]) -> None:
        self.args = args
        self.path = path
        self.make_service = make_service
        self.running: dict[str, list[Service]] = {}

    def supervise(self) -> None:
        wanted, _ = read_lanes(self.path)
        for name in [name for name in self.running if name not in wanted]:
            for service in reversed(self.running.pop(name)):
                service.stop()
            lane_state_file(self.args.node_id, name).unlink(missing_ok=True)
            print(f"Lane {name} stopped.", flush=True)
        for name, lane in wanted.items():
            if name not in self.running:
                specs = node_specs(lane_arguments(self.args, name, lane))
                self.running[name] = [self.make_service(f"{name}: {label}", command) for label, command in specs]
                for service in self.running[name]:
                    service.start()
                print(f"Lane {name} started on slot {lane['slot']}.", flush=True)
        for services in self.running.values():
            for service in services:
                service.supervise()

    def stop(self) -> None:
        for services in self.running.values():
            for service in reversed(services):
                service.stop()


def run_node(args: argparse.Namespace) -> int:
    if getattr(args, "pool_token_file", None):
        os.environ["SHARDMESH_POOL_TOKEN_FILE"] = str(Path(args.pool_token_file).resolve())
    if getattr(args, "ca_file", None):
        os.environ["SHARDMESH_CA_FILE"] = str(Path(args.ca_file).resolve())
    if getattr(args, "identity_cert", None) and getattr(args, "identity_key", None):
        os.environ["SHARDMESH_IDENTITY_CERT"] = str(Path(args.identity_cert).resolve())
        os.environ["SHARDMESH_IDENTITY_KEY"] = str(Path(args.identity_key).resolve())
    if getattr(args, "model_dir", None):
        os.environ["SHARDMESH_MODEL_DIRS"] = os.pathsep.join(str(Path(item).resolve()) for item in args.model_dir)
    if getattr(args, "advertise_host", "auto") == "auto":
        args.advertise_host = detect_address()
        print(f"Advertising {args.advertise_host}.", flush=True)
    if getattr(args, "tls_cert", None):
        args.advertise_scheme = "https"
        os.environ.setdefault("SHARDMESH_TLS_CERT", str(Path(args.tls_cert).resolve()))
        if getattr(args, "tls_key", None) and os.getenv("SHARDMESH_CA_FILE"):
            # The same certificate also signs what TLS cannot vouch for: gossip replies.
            os.environ.setdefault("SHARDMESH_IDENTITY_CERT", str(Path(args.tls_cert).resolve()))
            os.environ.setdefault("SHARDMESH_IDENTITY_KEY", str(Path(args.tls_key).resolve()))
    check_exposure(
        args.bind_host,
        load_token(),
        os.getenv("SHARDMESH_CA_FILE"),
        getattr(args, "open_pool", False),
        args.advertise_host,
        getattr(args, "trusted_network", False),
    )
    for seed in _seeds(args):  # credentials go to the peers named here, too
        check_exposure(args.bind_host, load_token(), os.getenv("SHARDMESH_CA_FILE"), getattr(args, "open_pool", False), seed, getattr(args, "trusted_network", False))
    specs = node_specs(args)

    env = os.environ.copy()
    # When ShardMesh is launched directly from a checkout (including an RDP
    # redirected drive), the supervisor can import this package because its
    # launcher amended sys.path, but fresh ``python -m shardmesh.worker`` child
    # processes cannot.  Propagate the checkout root without disturbing an
    # existing PYTHONPATH; installed-package launches continue to work too.
    source_root = Path(__file__).resolve().parents[1]
    code_root = source_root if getattr(args, "no_local_copy", False) else local_code_root(source_root)
    entries = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    if code_root != source_root:
        entries = [item for item in entries if not is_remote_path(item)]
    env["PYTHONPATH"] = os.pathsep.join([str(code_root), *entries])
    for option, variable in (
        ("tls_cert", "SHARDMESH_TLS_CERT"),
        ("tls_key", "SHARDMESH_TLS_KEY"),
        ("keys_file", "SHARDMESH_KEYS_FILE"),
    ):
        if getattr(args, option, None):
            env[variable] = str(Path(getattr(args, option)).resolve())
    output = open_log(getattr(args, "log_file", None))
    extra = {"stdout": output, "stderr": subprocess.STDOUT} if output else {}
    def make_service(label: str, command: list[str]) -> Service:
        return Service(label, lambda: subprocess.Popen(command, env=env, cwd=code_root, **extra))

    services = [make_service(label, command) for label, command in specs]
    keeper = None
    if not getattr(args, "no_lanes", False) and not getattr(args, "no_control", False):
        lanes_file = data_folder() / f"lanes-{args.node_id}.json"
        env["SHARDMESH_LANES_FILE"] = str(lanes_file)
        keeper = LaneKeeper(args, lanes_file, make_service)
    try:
        print(f"ShardMesh node {args.node_id} contributing:", flush=True)
        for service in services:
            service.start()
            print(f"  {service.label} (PID {service.process.pid})", flush=True)
        print("Press Ctrl+C to stop this node's services.", flush=True)
        while True:
            for service in services:
                service.supervise()
            if keeper is not None:
                keeper.supervise()
            time.sleep(2)
    except KeyboardInterrupt:
        return 0
    finally:
        if keeper is not None:
            keeper.stop()
        for service in reversed(services):
            service.stop()
