"""Discover the memory a node can hold weights in.

Every node is different, so nothing here assumes a shape: a worker reports a
list of pools and the planner works from that list. Four kinds exist.

========  ==================================  ================================
kind      examples                            weights execute on
========  ==================================  ================================
vram      NVIDIA, AMD ROCm, Intel XPU, Vulkan     that accelerator
unified   Apple MPS, integrated GPUs          that accelerator, sharing RAM
ram       system memory                       the CPU
disk      NVMe or SSD, memory-mapped          the CPU, paged in by the OS
========  ==================================  ================================
"""

from __future__ import annotations

import ctypes.util
import functools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psutil
import torch

from .protocol import MemoryPool

GIB = 1024**3
# Share of currently free memory an automatic budget may claim.
AUTO_ACCELERATOR_SHARE = 0.9
AUTO_RAM_SHARE = 0.6
# Sequential NVMe read rate is not worth measuring at startup: the OS page
# cache makes a quick probe report RAM speed rather than disk speed.
DISK_BANDWIDTH_GBPS = 1.5
VULKAN_PROBE_SECONDS = 30
ASSUMED_VULKAN_GBPS = 300.0


@dataclass(slots=True)
class Accelerator:
    device: str
    kind: str  # vram | unified
    backend: str  # cuda | rocm | mps | xpu | vulkan
    name: str
    total_bytes: int
    free_bytes: int
    runtime_version: str | None


def _cuda_accelerator(index: int) -> Accelerator:
    properties = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    backend = "rocm" if torch.version.hip else "cuda"
    version = str(torch.version.hip or torch.version.cuda)
    integrated = bool(getattr(properties, "is_integrated", False))
    return Accelerator(
        device=f"cuda:{index}",
        kind="unified" if integrated else "vram",
        backend=backend,
        name=properties.name,
        total_bytes=total,
        free_bytes=free,
        runtime_version=version,
    )


def _xpu_accelerator(index: int) -> Accelerator:
    properties = torch.xpu.get_device_properties(index)
    total = int(properties.total_memory)
    return Accelerator(
        device=f"xpu:{index}",
        kind="vram",
        backend="xpu",
        name=properties.name,
        total_bytes=total,
        free_bytes=total,
        runtime_version=None,
    )


def _mps_accelerator() -> Accelerator:
    memory = psutil.virtual_memory()
    return Accelerator(
        device="mps",
        kind="unified",
        backend="mps",
        name="Apple Metal Performance Shaders",
        total_bytes=memory.total,
        free_bytes=memory.available,
        runtime_version=platform.mac_ver()[0],
    )


def _has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _has_mps() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def vulkan_loader_present() -> bool:
    """Whether this machine has a Vulkan driver at all, without starting one."""
    if sys.platform == "darwin":
        return False  # Macs get their GPU through PyTorch and Metal
    return ctypes.util.find_library("vulkan-1" if os.name == "nt" else "vulkan") is not None


def read_vulkan_devices(folder: Path, run: Any = subprocess.run) -> list[Accelerator]:
    """What a Vulkan build sees, asked from a process of its own."""
    try:
        done = run([sys.executable, "-m", "isharded.gpu_probe", str(folder)], capture_output=True, text=True, timeout=VULKAN_PROBE_SECONDS)
        reported = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else []
    except (IndexError, OSError, ValueError, subprocess.SubprocessError):
        return []
    found = []
    for index, item in enumerate(item for item in reported if str(item.get("backend", "")).lower().startswith("vulkan")):
        found.append(
            Accelerator(
                device=f"vulkan:{index}",
                kind="unified" if item.get("integrated") else "vram",
                backend="vulkan",
                name=str(item.get("name", "Vulkan GPU")),
                total_bytes=int(item.get("total_bytes", 0)),
                free_bytes=int(item.get("free_bytes", 0)),
                runtime_version=None,
            )
        )
    return found


@functools.lru_cache(maxsize=1)
def vulkan_accelerators() -> tuple[Accelerator, ...]:
    """GPUs PyTorch cannot see, such as an AMD card on Windows. For GGUF models only."""
    from .llamacpp import LlamaError, ensure_build

    if not vulkan_loader_present():
        return ()
    try:
        return tuple(read_vulkan_devices(ensure_build("vulkan")))
    except (LlamaError, OSError, ValueError, httpx.HTTPError):
        return ()  # no way to fetch the build, as on a closed network


def _all_accelerators() -> list[Accelerator]:
    found: list[Accelerator] = []
    if torch.cuda.is_available():
        found.extend(_cuda_accelerator(index) for index in range(torch.cuda.device_count()))
    if _has_xpu():
        found.extend(_xpu_accelerator(index) for index in range(torch.xpu.device_count()))
    if _has_mps():
        found.append(_mps_accelerator())
    return found or list(vulkan_accelerators())


def discover_accelerators(requested: str) -> list[Accelerator]:
    """Resolve ``auto``, ``all``, ``none``, ``cpu``, or a comma list of devices.

    ROCm builds of PyTorch expose AMD GPUs through the ``cuda`` device API, so
    ``cuda:0`` is the right spelling on both vendors.
    """
    choice = requested.strip().lower()
    if choice in {"none", "cpu", ""}:
        return []
    available = list(vulkan_accelerators()) if choice.startswith("vulkan") else _all_accelerators()
    if choice.startswith("vulkan") and not available:
        raise ValueError("no Vulkan GPU found; check its driver, or that this machine can download llama.cpp's Vulkan build")
    if choice in {"auto", "all"}:
        return available
    selected: list[Accelerator] = []
    for name in (part.strip() for part in choice.split(",") if part.strip()):
        name = {"cuda": "cuda:0", "xpu": "xpu:0", "vulkan": "vulkan:0"}.get(name, name)
        match = next((item for item in available if item.device == name), None)
        if match is None:
            known = ", ".join(item.device for item in available) or "none"
            raise ValueError(f"accelerator {name!r} not found; this host offers: {known}")
        selected.append(match)
    return selected


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_bandwidth(device: str, size_mib: int = 256, rounds: int = 16) -> float:
    """Measure memory bandwidth in GB/s with a device-local copy.

    Single-request decoding reads every active weight once per token, so this
    number predicts token latency far better than a FLOPS figure would. The
    copy has to be large and repeated: a short one mostly times launch overhead,
    and read a 7900 XT as slower than a laptop GPU.
    """
    if device.startswith("vulkan"):
        return ASSUMED_VULKAN_GBPS  # PyTorch cannot reach this GPU to time it
    target = torch.device(device)
    try:
        source = torch.empty(size_mib * 1024 * 1024 // 4, dtype=torch.float32, device=target)
        destination = torch.empty_like(source)
        destination.copy_(source)
        _synchronize(target)
        started = time.perf_counter()
        for _ in range(rounds):
            destination.copy_(source)
        _synchronize(target)
        elapsed = max(time.perf_counter() - started, 1e-6)
        moved = 2 * source.numel() * source.element_size() * rounds
        del source, destination
        return round(moved / elapsed / 1e9, 1)
    except (RuntimeError, MemoryError):
        return 300.0 if target.type != "cpu" else 20.0


def live_free_bytes(pool: MemoryPool, cache_dir: Path) -> int:
    """Current free bytes for a pool, read from the OS or driver."""
    if pool.kind == "disk":
        return shutil.disk_usage(cache_dir).free
    if pool.kind == "ram" or pool.device == "mps":
        return psutil.virtual_memory().available
    if pool.device.startswith("cuda:"):
        return torch.cuda.mem_get_info(int(pool.device.split(":", 1)[1]))[0]
    return pool.free_bytes


def build_pools(
    host_id: str,
    accelerators: list[Accelerator],
    *,
    accelerator_budget_bytes: int,
    compute_dtype: str,
    ram_budget_bytes: int | None,
    disk_budget_bytes: int,
    cache_dir: Path,
) -> list[MemoryPool]:
    """Describe this worker's pools, fastest first.

    ``ram_budget_bytes=None`` means the caller did not choose: a CPU-only
    worker then offers a share of free RAM, because it has nowhere else to put
    weights, while an accelerator worker stays accelerator-only.

    ``compute_dtype`` applies to accelerators and RAM. The disk pool always
    reports ``checkpoint``: it maps files as they are, so it cannot convert.
    """
    pools: list[MemoryPool] = []
    for item in accelerators:
        automatic = int(item.free_bytes * AUTO_ACCELERATOR_SHARE)
        budget = min(accelerator_budget_bytes or automatic, item.total_bytes)
        pools.append(
            MemoryPool(
                pool_id=item.device,
                kind=item.kind,
                device=item.device,
                label=item.name,
                # Unified memory is the same physical RAM the CPU uses.
                capacity_group=f"{host_id}:ram" if item.kind == "unified" else f"{host_id}:{item.device}",
                total_bytes=item.total_bytes,
                budget_bytes=budget,
                free_bytes=item.free_bytes,
                bandwidth_gbps=measure_bandwidth(item.device),
                dtype=compute_dtype,
            )
        )

    memory = psutil.virtual_memory()
    if ram_budget_bytes is None:
        ram_budget_bytes = 0 if accelerators else int(memory.available * AUTO_RAM_SHARE)
    if ram_budget_bytes > 0:
        pools.append(
            MemoryPool(
                pool_id="ram",
                kind="ram",
                device="cpu",
                label="System memory",
                capacity_group=f"{host_id}:ram",
                total_bytes=memory.total,
                budget_bytes=min(ram_budget_bytes, memory.total),
                free_bytes=memory.available,
                bandwidth_gbps=measure_bandwidth("cpu"),
                dtype=compute_dtype,
            )
        )

    if disk_budget_bytes > 0:
        usage = shutil.disk_usage(cache_dir)
        pools.append(
            MemoryPool(
                pool_id="disk",
                kind="disk",
                device="cpu",
                label=f"Memory-mapped from {cache_dir}",
                capacity_group=f"{host_id}:disk:{os.path.splitdrive(str(cache_dir.resolve()))[0] or '/'}",
                total_bytes=usage.total,
                budget_bytes=min(disk_budget_bytes, usage.free),
                free_bytes=usage.free,
                bandwidth_gbps=DISK_BANDWIDTH_GBPS,
                dtype="checkpoint",
            )
        )
    return pools


def apply_accelerator_limits(pools: list[MemoryPool], explicit_budget: bool) -> None:
    """Turn an explicit VRAM budget into a hard allocator limit where possible."""
    if not explicit_budget:
        return
    for pool in pools:
        if pool.device.startswith("cuda:") and pool.total_bytes:
            index = int(pool.device.split(":", 1)[1])
            fraction = min(1.0, pool.budget_bytes / pool.total_bytes)
            torch.cuda.set_per_process_memory_fraction(fraction, index)
