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

from .geo import country_code
from .lanes import lane_port, lane_url, read_lanes
from .optional import has_torch
from .tls import OWN_ADDRESS
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
    target = base / "iSharded" / "app"
    shutil.rmtree(target / "isharded", ignore_errors=True)
    shutil.copytree(
        source_root / "isharded",
        target / "isharded",
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
    parser.add_argument("--pool-token-file", default=None, help="file holding the pool token; default is pool.token in the iSharded folder")
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
    parser.add_argument("--reach-hosts", default=None, help="other hosts this node answers on, comma separated, tried between stages; default auto finds this machine's, none registers only the advertised one")
    parser.add_argument("--location", type=country_code, default=None, help="two letter country code for the dashboard's flag, or none to show no flag; default is looked up from --advertise-host")
    parser.add_argument("--base-port", type=int, default=8200, help="first worker's port; later workers take the next ones")
    parser.add_argument(
        "--gpu-device",
        default="auto",
        help="auto (every accelerator), cuda:0, cuda:0,cuda:1, mps, xpu:0, vulkan, or none. "
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
    parser.add_argument("--state-ram-gb", type=float, default=None, help="RAM for saved prompt states on each stage; default 1")
    parser.add_argument("--state-disk-gb", type=float, default=None, help="disk for saved prompt states; default 8, and 0 keeps them in RAM only")
    parser.add_argument("--state-copy-mb", type=float, default=None, help="MB per second for copies of saved states to another node; default 10, and 0 sends none")
    parser.add_argument("--no-peer-fetch", action="store_true", help="always download from the model host")
    parser.add_argument("--no-peer-serve", action="store_true", help="do not send shards to other workers")
    parser.add_argument("--no-local-copy", action="store_true", help="run workers from a network path as it is")
    parser.add_argument("--no-lanes", action="store_true", help="host only the main model here, never one loaded with load --lane")
    parser.add_argument("--host-models", default="", help="comma-separated patterns such as *Qwen3*; host only models that match")
    parser.add_argument("--ownerless", default=None, metavar="DIR", help="join a network nobody owns, using the identity in DIR from isharded net init")
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


def _trust_flags(args: argparse.Namespace) -> list[str]:
    return [
        *(["--open"] if getattr(args, "open_pool", False) else []),
        *(["--trusted-network"] if getattr(args, "trusted_network", False) else []),
    ]


def _control_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-m", "isharded.orchestrator",
        "--host", args.bind_host,
        "--port", str(_control_port(args)),
        "--node-id", args.node_id,
        "--advertise-url", _local_control(args),
        "--join", ",".join(_seeds(args)),
        *_trust_flags(args),
        *_lane_flags(args),
    ]  # fmt: skip


def data_folder() -> Path:
    configured = os.getenv("ISHARDED_UP_HOME")
    return Path(configured) if configured else Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "iSharded"


def _lane_flags(args: argparse.Namespace) -> list[str]:
    """A lane keeps its own records. The node's keys stay with the front door."""
    lane = getattr(args, "lane", None)
    if not lane:
        return []
    return [
        "--lane", lane,
        "--state-file", str(lane_state_file(args.node_id, lane)),
        "--audit-file", str(data_folder() / f"audit-{args.node_id}-{lane}.jsonl"),
        "--keys-file", str(data_folder() / f"access-{args.node_id}-{lane}.json"),
    ]  # fmt: skip


def lane_state_file(node_id: str, lane: str) -> Path:
    return data_folder() / f"orchestrator-state-{node_id}-{lane}.json"


def _worker_command(args: argparse.Namespace, worker_id: str, port: int, device: str, roles: str, extra: list[str]) -> list[str]:
    return [
        sys.executable, "-m", "isharded.worker",
        "--host", args.bind_host,
        "--port", str(port),
        "--worker-id", worker_id,
        "--orchestrator-url", _control_urls(args),
        "--advertise-url", f"{args.advertise_scheme}://{args.advertise_host}:{port}",
        "--device", device,
        "--roles", roles,
        *_trust_flags(args),
        *(["--reach-hosts", args.reach_hosts] if getattr(args, "reach_hosts", None) else []),
        *(["--host-models", args.host_models] if getattr(args, "host_models", "") else []),
        *(["--location", args.location] if getattr(args, "location", None) else []),
        *extra,
    ]  # fmt: skip


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
    for name in ("state_ram_gb", "state_disk_gb", "state_copy_mb"):
        if getattr(args, name, None) is not None:
            flags += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    flags += [flag for flag, name in (("--no-peer-fetch", "no_peer_fetch"), ("--no-peer-serve", "no_peer_serve")) if getattr(args, name, False)]
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

    STOP_SECONDS = 10.0

    def stop(self) -> None:
        stop_together([self])

    def ask_to_stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def finish(self, deadline: float) -> None:
        """Waits until ``deadline`` for the process to end after ``ask_to_stop``, then kills it."""
        if self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.process.kill()


def stop_together(services: list[Service]) -> None:
    """Asks every service to stop at once, then gives them one shared wait.

    One at a time, each waiting its turn, a node with several processes took up to ten
    seconds apiece to stop. Together, the slowest one sets the time.
    """
    for service in services:
        service.ask_to_stop()
    deadline = time.monotonic() + Service.STOP_SECONDS
    for service in services:
        service.finish(deadline)


def _has_accelerator() -> bool:
    # Import the accelerator runtime only when discovery was requested.
    # CPU/RAM-only contributors should not depend on CUDA/ROCm driver
    # initialization merely to join the pool.
    from .memory import _all_accelerators, vulkan_accelerators

    if not has_torch():
        return bool(_all_accelerators())
    import torch

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    return torch.cuda.is_available() or has_mps or has_xpu or bool(vulkan_accelerators())


def node_specs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """What this node runs: a label and a command for each process."""
    specs: list[tuple[str, list[str]]] = []
    port = args.base_port
    gpu_device = args.gpu_device
    dtype = getattr(args, "gpu_dtype", "auto")
    weight_flags = _weight_memory_flags(args)
    cache_flags = _cache_flags(args)
    thread_flags = ["--cpu-threads", str(args.cpu_threads)] if args.cpu_threads else []
    if gpu_device == "auto" and not _has_accelerator():
        gpu_device = "none"
    # A worker with weights in several memory types executes all of them
    # itself: the CPU runs the RAM and disk tiers, the GPU the VRAM tier.
    spans_tiers = gpu_device != "none" and bool(weight_flags)
    if gpu_device != "none":
        extra = ["--gpu-memory-gb", str(args.gpu_memory_gb), "--dtype", dtype, *cache_flags]
        if spans_tiers:
            extra += [*weight_flags, *thread_flags]
        label = "GPU + RAM/disk inference" if spans_tiers else "GPU inference"
        specs.append((label, _worker_command(args, f"{args.node_id}-gpu", port, gpu_device, "inference", extra)))
        port += 1
    if not spans_tiers and (args.cpu_threads > 0 or weight_flags):
        extra = ["--dtype", dtype, *cache_flags, *weight_flags, *thread_flags]
        if not args.cpu_active:
            extra.append("--standby")
        label = "CPU inference" + ("" if args.cpu_active else " standby")
        specs.append((label, _worker_command(args, f"{args.node_id}-cpu", port, "cpu", "inference", extra)))
        port += 1
    if args.ram_budget_gb > 0:
        # It keeps copies of saved states too, in the same places as any worker here.
        extra = ["--ram-budget-gb", str(args.ram_budget_gb), *cache_flags]
        specs.append(("RAM KV cache", _worker_command(args, f"{args.node_id}-memory", port, "cpu", "kv-cache,checkpoint", extra)))
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
            stop_together(self.running.pop(name))
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

    def services(self) -> list[Service]:
        return [service for services in self.running.values() for service in services]

    def stop(self) -> None:
        stop_together(self.services())


def join_commons(args: argparse.Namespace) -> None:
    """A network nobody owns: this node's identity stands in for a certificate from an owner."""
    from .commons import certificate_file, identity_name, key_file

    folder = Path(args.ownerless).resolve()
    if not certificate_file(folder).is_file():
        raise SystemExit(f"{folder} has no identity; run isharded net init --dir {folder} first")
    args.node_id = identity_name(folder)
    args.ca_file = str(certificate_file(folder))
    args.identity_cert = str(certificate_file(folder))
    args.identity_key = str(key_file(folder))
    os.environ["ISHARDED_OWNERLESS"] = str(folder)
    print(f"This node is {args.node_id}.", flush=True)


def _publish_settings(args: argparse.Namespace) -> None:
    """Put what this node's processes share in the environment, and refuse an unsafe start."""
    if getattr(args, "ownerless", None):
        join_commons(args)
    if getattr(args, "pool_token_file", None):
        os.environ["ISHARDED_POOL_TOKEN_FILE"] = str(Path(args.pool_token_file).resolve())
    if getattr(args, "ca_file", None):
        os.environ["ISHARDED_CA_FILE"] = str(Path(args.ca_file).resolve())
    if getattr(args, "identity_cert", None) and getattr(args, "identity_key", None):
        os.environ["ISHARDED_IDENTITY_CERT"] = str(Path(args.identity_cert).resolve())
        os.environ["ISHARDED_IDENTITY_KEY"] = str(Path(args.identity_key).resolve())
    if getattr(args, "model_dir", None):
        os.environ["ISHARDED_MODEL_DIRS"] = os.pathsep.join(str(Path(item).resolve()) for item in args.model_dir)
    if getattr(args, "advertise_host", "auto") == "auto":
        args.advertise_host = detect_address()
        print(f"Advertising {args.advertise_host}.", flush=True)
    if getattr(args, "tls_cert", None):
        args.advertise_scheme = "https"
        os.environ.setdefault("ISHARDED_TLS_CERT", str(Path(args.tls_cert).resolve()))
        if getattr(args, "tls_key", None) and os.getenv("ISHARDED_CA_FILE"):
            # The same certificate also signs what TLS cannot vouch for: gossip replies.
            os.environ.setdefault("ISHARDED_IDENTITY_CERT", str(Path(args.tls_cert).resolve()))
            os.environ.setdefault("ISHARDED_IDENTITY_KEY", str(Path(args.tls_key).resolve()))
    token, authority = load_token(), os.getenv("ISHARDED_CA_FILE")
    for address in (args.advertise_host, *_seeds(args)):  # credentials go to the peers named here, too
        check_exposure(args.bind_host, token, authority, getattr(args, "open_pool", False), address, getattr(args, "trusted_network", False))


def _child_environment(args: argparse.Namespace) -> tuple[dict[str, str], Path]:
    """What this node's processes run with, and the folder they run from."""
    env = os.environ.copy()
    if args.bind_host in {"0.0.0.0", "::"}:  # loopback reaches this node's services only then
        env.setdefault(OWN_ADDRESS, args.advertise_host)
    # When iSharded is launched directly from a checkout (including an RDP
    # redirected drive), the supervisor can import this package because its
    # launcher amended sys.path, but fresh ``python -m isharded.worker`` child
    # processes cannot.  Propagate the checkout root without disturbing an
    # existing PYTHONPATH; installed-package launches continue to work too.
    source_root = Path(__file__).resolve().parents[1]
    code_root = source_root if getattr(args, "no_local_copy", False) else local_code_root(source_root)
    entries = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    if code_root != source_root:
        entries = [item for item in entries if not is_remote_path(item)]
    env["PYTHONPATH"] = os.pathsep.join([str(code_root), *entries])
    for option, variable in (
        ("tls_cert", "ISHARDED_TLS_CERT"),
        ("tls_key", "ISHARDED_TLS_KEY"),
        ("keys_file", "ISHARDED_KEYS_FILE"),
    ):
        if getattr(args, option, None):
            env[variable] = str(Path(getattr(args, option)).resolve())
    return env, code_root


def run_node(args: argparse.Namespace) -> int:
    _publish_settings(args)
    specs = node_specs(args)
    env, code_root = _child_environment(args)
    output = open_log(getattr(args, "log_file", None))
    extra = {"stdout": output, "stderr": subprocess.STDOUT} if output else {}

    def make_service(label: str, command: list[str]) -> Service:
        return Service(label, lambda: subprocess.Popen(command, env=env, cwd=code_root, **extra))

    services = [make_service(label, command) for label, command in specs]
    keeper = None
    if not getattr(args, "no_lanes", False) and not getattr(args, "no_control", False):
        lanes_file = data_folder() / f"lanes-{args.node_id}.json"
        env["ISHARDED_LANES_FILE"] = str(lanes_file)
        keeper = LaneKeeper(args, lanes_file, make_service)
    try:
        print(f"iSharded node {args.node_id} contributing:", flush=True)
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
        stop_together([*(keeper.services() if keeper is not None else []), *reversed(services)])
