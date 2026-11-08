"""What a token costs a node: its power, its price, and how its GPU runs a pass.

These are for measuring, so `isharded bench` can say what a run cost in joules and money and
someone else can compare. Nothing here changes how a node serves. The power meter only runs
when a node asks for it, because it keeps a vendor tool running beside the worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable

PRICE_VARIABLE = "ISHARDED_PRICE_PER_HOUR"
POWER_VARIABLE = "ISHARDED_POWER_SAMPLE_MS"
# llama.cpp turns CUDA graphs off when this is set to anything at all, "0" included.
GRAPHS_VARIABLE = "GGML_CUDA_DISABLE_GRAPHS"
SHORTEST_SAMPLE_MS = 100
# Linux's running count of a CPU package's energy, in microjoules. Often readable by root only.
RAPL_FILE = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_RANGE_FILE = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
LONGEST_ERROR = 200


def price_per_hour(environment: dict[str, str] | None = None) -> float | None:
    """What this node costs to run, in any currency an hour, as its operator set it."""
    text = (environment if environment is not None else os.environ).get(PRICE_VARIABLE, "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def cuda_graphs_on(environment: dict[str, str] | None = None) -> bool:
    """Whether llama.cpp in this node's stage processes may use CUDA graphs."""
    return GRAPHS_VARIABLE not in (environment if environment is not None else os.environ)


def graphs_environment(on: bool, environment: dict[str, str] | None = None) -> dict[str, str]:
    """An environment for a process that runs with CUDA graphs on or off."""
    out = dict(os.environ if environment is None else environment)
    if on:
        out.pop(GRAPHS_VARIABLE, None)
    else:
        out[GRAPHS_VARIABLE] = "1"
    return out


def sample_ms(environment: dict[str, str] | None = None) -> int:
    """How often to read power, or 0 for not at all."""
    text = (environment if environment is not None else os.environ).get(POWER_VARIABLE, "").strip()
    try:
        value = int(float(text)) if text else 0
    except ValueError:
        return 0
    return max(value, SHORTEST_SAMPLE_MS) if value > 0 else 0


def smi_device(device: str, environment: dict[str, str] | None = None) -> str | None:
    """What a vendor tool calls the card a worker computes on.

    The tools count every card, while CUDA counts only the ones CUDA_VISIBLE_DEVICES leaves
    it, in that order. An entry there can also be a card's UUID, which nvidia-smi takes too.
    """
    if not device.startswith(("cuda:", "rocm:", "hip:")):
        return None
    ordinal = device.split(":", 1)[1]
    if not ordinal.isdigit():
        return None
    environment = environment if environment is not None else os.environ
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        chosen = environment.get(name)
        if chosen is None:
            continue
        picked = [part.strip() for part in chosen.split(",") if part.strip()]
        return picked[int(ordinal)] if int(ordinal) < len(picked) else None
    return ordinal


def parse_watts(line: str) -> float | None:
    """One reading from nvidia-smi or rocm-smi. A card that cannot say prints [N/A]."""
    for part in reversed(line.replace(",", " ").split()):
        try:
            value = float(part.rstrip("W"))
        except ValueError:
            continue
        return value if value >= 0 else None
    return None


@dataclass
class Energy:
    """A running total built from power readings, each held until the next."""

    joules: float = 0.0
    watts: float | None = None
    samples: int = 0
    seconds: float = 0.0
    last_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, watts: float, at: float) -> None:
        with self.lock:
            if self.last_at is not None and self.watts is not None:
                gap = max(at - self.last_at, 0.0)
                self.joules += (self.watts + watts) / 2 * gap
                self.seconds += gap
            self.watts, self.last_at = watts, at
            self.samples += 1

    def add_joules(self, joules: float, at: float) -> None:
        """A reading from a counter of energy, which needs no guessing between samples."""
        with self.lock:
            if self.last_at is not None:
                gap = max(at - self.last_at, 0.0)
                self.joules += joules
                self.seconds += gap
                self.watts = joules / gap if gap > 0 else self.watts
            self.last_at = at
            self.samples += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "joules": round(self.joules, 3),
                "watts": None if self.watts is None else round(self.watts, 2),
                "samples": self.samples,
                "seconds": round(self.seconds, 3),
            }


class PowerMeter:
    """Reads one device's power for as long as the worker runs.

    nvidia-smi keeps running and prints a reading every interval, which costs far less than
    starting it each time. rocm-smi has no such loop, so it is asked each interval. A CPU on
    Linux has a counter of the package's energy, which is the whole package and not only
    this worker's share of it.
    """

    def __init__(self, source: str, device: str | None, interval_ms: int, read: Callable[[], Iterable[tuple[str, float]]], stop: Callable[[], None] = lambda: None) -> None:
        self.source, self.device, self.interval_ms = source, device, interval_ms
        self.energy = Energy()
        self.error: str | None = None
        self._read, self._stop = read, stop
        self._stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="isharded-power", daemon=True)

    def start(self) -> "PowerMeter":
        self.thread.start()
        return self

    def _run(self) -> None:
        try:
            for kind, value in self._read():
                if self._stopping.is_set():
                    return
                if kind == "watts":
                    self.energy.add(value, time.monotonic())
                else:
                    self.energy.add_joules(value, time.monotonic())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.error = str(exc)[:LONGEST_ERROR] or type(exc).__name__
            return
        if not self._stopping.is_set() and self.error is None:
            self.error = f"{self.source} stopped reporting"

    def close(self) -> None:
        self._stopping.set()
        self._stop()

    def report(self) -> dict[str, Any]:
        return {"source": self.source, "device": self.device, "interval_ms": self.interval_ms, **self.energy.snapshot(), "error": self.error}


def _no_window() -> dict[str, Any]:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def nvidia_meter(device: str, interval_ms: int, tool: str = "nvidia-smi") -> PowerMeter:
    command = [tool, "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", device, "-lms", str(interval_ms)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True, **_no_window())

    def read() -> Iterable[tuple[str, float]]:
        assert process.stdout is not None
        for line in process.stdout:
            watts = parse_watts(line)
            if watts is not None:
                yield "watts", watts

    def stop() -> None:
        if process.poll() is None:
            process.kill()

    return PowerMeter("nvidia-smi", device, interval_ms, read, stop)


def rocm_meter(device: str, interval_ms: int, tool: str = "rocm-smi", run: Callable[..., Any] = subprocess.run) -> PowerMeter:
    stopping = threading.Event()

    def read() -> Iterable[tuple[str, float]]:
        while not stopping.is_set():
            done = run([tool, "-d", device, "--showpower", "--csv"], capture_output=True, text=True, timeout=10, **_no_window())
            lines = [line for line in done.stdout.splitlines()[1:] if line.strip()] if done.returncode == 0 else []
            watts = parse_watts(lines[0]) if lines else None
            if watts is not None:
                yield "watts", watts
            stopping.wait(interval_ms / 1000)

    return PowerMeter("rocm-smi", device, interval_ms, read, stopping.set)


def rapl_meter(interval_ms: int, path: Path = RAPL_FILE, range_path: Path = RAPL_RANGE_FILE) -> PowerMeter:
    stopping = threading.Event()
    try:
        wrap = int(range_path.read_text().strip())
    except (OSError, ValueError):
        wrap = 0

    def read() -> Iterable[tuple[str, float]]:
        last = int(path.read_text().strip())
        yield "joules", 0.0
        while not stopping.wait(interval_ms / 1000):
            now = int(path.read_text().strip())
            step = now - last if now >= last else now + wrap - last  # the counter wraps
            last = now
            yield "joules", step / 1e6

    return PowerMeter("rapl", "package-0", interval_ms, read, stopping.set)


def open_meter(device: str, interval_ms: int, environment: dict[str, str] | None = None, which: Callable[[str], str | None] = shutil.which) -> PowerMeter | str:
    """A started meter for a worker's device, or why there is none."""
    card = smi_device(device, environment)
    try:
        if card is not None:
            if tool := which("nvidia-smi"):
                return nvidia_meter(card, interval_ms, tool).start()
            if tool := which("rocm-smi"):
                return rocm_meter(card, interval_ms, tool).start()
            return "no nvidia-smi or rocm-smi on this machine"
        if device == "cpu" and RAPL_FILE.exists():
            int(RAPL_FILE.read_text().strip())  # often readable by root only, so find out now
            return rapl_meter(interval_ms).start()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return str(exc)[:LONGEST_ERROR] or type(exc).__name__
    return f"no power reading for {device} here"


@dataclass
class GuessClock:
    """The draft's time per guess on the entry, as a pipelined reply waits for it.

    A guess for a pipelined pass asks for one token. One that runs a single row is the steady
    case. One that reads more, as after a miss, is counted apart, since the law's pace is the
    steady guess and the miss pays the rest.
    """

    guesses: int = 0
    ms_total: float = 0.0
    steady: int = 0
    steady_ms_total: float = 0.0

    def note(self, ms: float, rows: int | None) -> None:
        self.guesses += 1
        self.ms_total += ms
        if rows == 1:
            self.steady += 1
            self.steady_ms_total += ms

    def report(self) -> dict[str, Any]:
        return {"guesses": self.guesses, "ms_total": round(self.ms_total, 3), "steady": self.steady, "steady_ms_total": round(self.steady_ms_total, 3)}
