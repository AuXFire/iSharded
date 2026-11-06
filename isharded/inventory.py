"""What a node can say about itself beyond its memory budgets.

Placement uses the memory figures workers register with. This is the rest of the
picture: storage, temperatures, power, load, and what is on battery. Each reading
is best effort, because no two machines expose the same ones, and a missing
sensor is left out, never guessed.
"""
from __future__ import annotations

import csv
import io
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import psutil

CGROUP = Path("/sys/fs/cgroup")

GPU_QUERY_SECONDS = 5.0
NVIDIA_FIELDS = ("name", "temperature.gpu", "power.draw", "utilization.gpu", "memory.used", "memory.total")


def _number(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None  # nvidia-smi says "[N/A]" for what a card cannot report


def nvidia_gpus(run: Any = subprocess.run) -> list[dict[str, Any]]:
    """Live readings from NVIDIA's own tool, if this machine has it."""
    tool = shutil.which("nvidia-smi")
    if not tool:
        return []
    try:
        done = run(
            [tool, f"--query-gpu={','.join(NVIDIA_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=GPU_QUERY_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if done.returncode != 0:
        return []
    gpus = []
    for row in csv.reader(io.StringIO(done.stdout)):
        if len(row) != len(NVIDIA_FIELDS):
            continue
        name, temperature, power, load, used, total = row
        reading = {
            "name": name.strip(),
            "temperature_c": _number(temperature),
            "power_w": _number(power),
            "load_percent": _number(load),
            "memory_used_mib": _number(used),
            "memory_total_mib": _number(total),
        }
        gpus.append({key: value for key, value in reading.items() if value is not None})
    return gpus


def disks() -> list[dict[str, Any]]:
    found = []
    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except OSError:
            continue  # an empty card reader or a dropped network drive
        found.append({"mount": partition.mountpoint, "total_bytes": usage.total, "free_bytes": usage.free})
    return found


def temperatures() -> dict[str, float]:
    reader = getattr(psutil, "sensors_temperatures", None)  # Linux and BSD only
    try:
        groups = reader() if reader else {}
    except OSError:
        return {}
    return {group: max(item.current for item in items) for group, items in groups.items() if items}


def battery() -> dict[str, Any] | None:
    reader = getattr(psutil, "sensors_battery", None)
    try:
        state = reader() if reader else None
    except OSError:
        return None
    return {"percent": state.percent, "on_mains": bool(state.power_plugged)} if state else None


@dataclass(frozen=True, slots=True)
class Ram:
    total: int
    available: int


def _cgroup_number(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None  # "max" means no limit


def _reclaimable(root: Path) -> int:
    """Page cache the kernel gives back on demand, which reading a model fills up."""
    for path in (root / "memory.stat", root / "memory" / "memory.stat"):
        try:
            for line in path.read_text(encoding="ascii").splitlines():
                name, _, value = line.partition(" ")
                if name in {"inactive_file", "total_inactive_file"}:
                    return int(value)
        except (OSError, ValueError):
            continue
    return 0


def ram(root: Path = CGROUP) -> Ram:
    """Memory this process may really use.

    In a container the host's RAM means nothing: a pod limited to 50 GB on a 503 GB
    host reads as 503 GB, and planning on that gets the node killed mid-load.
    """
    host = psutil.virtual_memory()
    limit = _cgroup_number(root / "memory.max") or _cgroup_number(root / "memory" / "memory.limit_in_bytes")
    if not limit or limit >= host.total:
        return Ram(host.total, host.available)
    used = _cgroup_number(root / "memory.current") or _cgroup_number(root / "memory" / "memory.usage_in_bytes") or 0
    return Ram(limit, min(host.available, max(limit - used + _reclaimable(root), 0)))


def inventory() -> dict[str, Any]:
    memory = ram()
    return {
        "taken_at": time.time(),
        "system": platform.platform(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count(),
        "ram_available_bytes": memory.available,
        "disks": disks(),
        "gpus": nvidia_gpus(),
        "temperatures_c": temperatures(),
        "battery": battery(),
    }
