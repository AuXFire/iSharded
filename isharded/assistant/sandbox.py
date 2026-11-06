"""Where generated code runs. Never this process.

This picks the strongest walls the machine really has and reports which it got.
A claim about isolation that is not true is worse than having none, because
people decide what to allow by it, so ``level`` is only ever what happened, and
what a level cannot do is said in the result. The levels, strongest first:

- ``container``: docker or podman, no network, read-only root, memory limit.
- ``namespace``: bubblewrap or ``unshare`` on Linux.
- ``process``: a Windows Job Object, or a new session and rlimits on Linux. This
  still runs as the same user. It stops runaway code and hides secrets in the
  environment, but it is no wall against a program that means harm.

Every level gets a scrubbed environment, a time limit that kills everything the
command started, and capped output. Nothing is ever downloaded: a container
image has to be on the machine already.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import functools
import hashlib
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from .contracts import SandboxResult
from .store import canonical

WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")
LEVELS = ("container", "namespace", "process")
DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_MEMORY_MB = 2048
DEFAULT_OUTPUT_BYTES = 1024 * 1024
POLL_SECONDS = 0.05
PROBE_SECONDS = 8
DRAIN_SECONDS = 2
CONTAINER_PIDS = 512
WORK_MOUNT = "/work"

KEPT_EVERYWHERE = ("PATH", "LANG", "LC_ALL", "TZ")
# Where Windows keeps things, which ordinary programs fail without. None of it is secret.
KEPT_ON_WINDOWS = (
    "SYSTEMROOT",
    "WINDIR",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL",
    "PROCESSOR_REVISION",
    "OS",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "PROGRAMDATA",
    "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)",
    "COMMONPROGRAMW6432",
    "ALLUSERSPROFILE",
    "PUBLIC",
)
# What the docker or podman client itself needs. Its child sees only what -e names.
KEPT_FOR_CLIENT = ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "XDG_RUNTIME_DIR", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "CONTAINER_HOST")

NETWORK_NOT_BLOCKED = "[sandbox] network=False was not enforced: the process level cannot block the network on this machine."
SAME_USER = "The process level runs as this user, so the command can read and change whatever this user can."
NO_FILE_WALL = "unshare gives no file system wall, so the command can read and change whatever this user can."


@dataclass(frozen=True)
class SandboxRun(SandboxResult):
    """A contract result, plus what the walls could not do."""

    network_blocked: bool = True
    stopped: bool = False  # someone asked for it to stop, as opposed to the time limit
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Walls:
    level: str
    tool: str  # docker, podman, bwrap, unshare, job, or session


def scrubbed_env(home: str | Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """A short allow-list, so a secret added to the environment tomorrow stays out too."""
    names = (*KEPT_EVERYWHERE, *(KEPT_ON_WINDOWS if WINDOWS else ()))
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.update({name: str(home) for name in ("HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR")})
    environment.update(extra or {})
    return environment


@functools.lru_cache(maxsize=1)
def job_api() -> tuple[Any, Any, Any, Any]:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in ("reads", "writes", "others", "read_bytes", "write_bytes", "other_bytes")
        ]

    class Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", Basic),
            ("IoInfo", Counters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)
    ntdll.NtResumeProcess.restype = ctypes.c_long
    return ctypes, kernel, ntdll, Extended


class WindowsJob:
    """Holds a process and all it starts, and kills them when its handle closes.

    Windows closes that handle when this process dies, however it dies, so a
    crash here cannot leave generated code running.
    """

    EXTENDED_LIMITS = 9
    KILL_ON_CLOSE = 0x2000
    JOB_MEMORY = 0x200
    DIE_ON_UNHANDLED_EXCEPTION = 0x400

    def __init__(self, memory_mb: int | None) -> None:
        self.ctypes, self.kernel, self.ntdll, extended = job_api()
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        limits = extended()
        limits.BasicLimitInformation.LimitFlags = self.KILL_ON_CLOSE | self.DIE_ON_UNHANDLED_EXCEPTION
        if memory_mb:
            limits.BasicLimitInformation.LimitFlags |= self.JOB_MEMORY
            limits.JobMemoryLimit = memory_mb * 1024 * 1024
        pointer = self.ctypes.byref(limits)
        if not self.kernel.SetInformationJobObject(self.handle, self.EXTENDED_LIMITS, pointer, self.ctypes.sizeof(limits)):
            error = self.ctypes.WinError(self.ctypes.get_last_error())
            self.close()
            raise error

    def hold(self, process: subprocess.Popen[bytes]) -> None:
        """Take a suspended process, then let it run, so no child can start outside."""
        handle = int(process._handle)  # type: ignore[attr-defined]
        if not self.kernel.AssignProcessToJobObject(self.handle, handle):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        status = self.ntdll.NtResumeProcess(handle)
        if status != 0:
            raise OSError(f"Windows would not start the suspended process (status {status:#x}).")

    def kill(self) -> None:
        if self.handle:
            self.kernel.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def with_rlimits(command: Sequence[str], memory_mb: int | None) -> list[str]:
    """Set limits in a shell, because preexec_fn can deadlock a program with threads."""
    if not memory_mb:
        return list(command)
    script = 'ulimit -v "$1" 2>/dev/null; ulimit -c 0 2>/dev/null; shift; exec "$@"'
    return ["/bin/sh", "-c", script, "sandbox", str(memory_mb * 1024), *command]


class ProcessTree:
    """One command and everything it starts, so all of it can be killed together."""

    def __init__(
        self,
        command: Sequence[str],
        cwd: str | Path | None,
        env: dict[str, str],
        *,
        memory_mb: int | None = None,
        stdin: bool = False,
    ) -> None:
        self.job: WindowsJob | None = None
        options: dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "stdin": subprocess.PIPE if stdin else subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if WINDOWS:
            self.job = WindowsJob(memory_mb)
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | 0x4  # 0x4 is CREATE_SUSPENDED
            try:
                self.process = subprocess.Popen(list(command), creationflags=flags, **options)
            except BaseException:
                self.job.close()
                raise
            try:
                self.job.hold(self.process)
            except BaseException:  # it must never run outside the job
                self.process.kill()
                self.process.wait()
                self.job.close()
                raise
        else:
            self.process = subprocess.Popen(with_rlimits(command, memory_mb), start_new_session=True, **options)

    def kill(self) -> None:
        """Kill the whole tree. Safe to call after it has already ended."""
        if self.job:
            self.job.kill()
        else:
            self._kill_session()

    def _kill_session(self) -> None:
        import signal

        import psutil

        pid = self.process.pid
        try:
            family = psutil.Process(pid).children(recursive=True)
        except psutil.Error:
            family = []
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        for member in family:  # a child that left the session is still found through its parent
            try:
                member.kill()
            except psutil.Error:
                pass

    def close(self) -> None:
        self.kill()
        try:
            self.process.wait(timeout=DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream:
                try:
                    stream.close()
                except OSError:
                    pass
        if self.job:
            self.job.close()


class CappedReader:
    """Keeps the start of a stream and drops the rest, so a flood cannot fill memory.

    It keeps reading past the cap, because a command blocked on a full pipe would
    look like a hang.
    """

    def __init__(self, stream: Any, limit: int) -> None:
        self.stream = stream
        self.limit = limit
        self.kept = bytearray()
        self.total = 0
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            while True:
                chunk = self.stream.read1(65536)
                if not chunk:
                    return
                self.total += len(chunk)
                room = self.limit - len(self.kept)
                if room > 0:
                    self.kept += chunk[:room]
        except (OSError, ValueError):
            pass  # the pipe closed under us, which is how a killed tree ends

    def text(self) -> str:
        self.thread.join(DRAIN_SECONDS)
        text = bytes(self.kept).decode("utf-8", "replace")
        if self.total > self.limit:
            text += f"\n[output cut: kept {self.limit} of {self.total} bytes]"
        return text


@dataclass(frozen=True)
class TreeResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    stopped: bool


def run_tree(
    command: Sequence[str],
    cwd: str | Path | None,
    env: dict[str, str],
    *,
    timeout: float,
    output_bytes: int = DEFAULT_OUTPUT_BYTES,
    memory_mb: int | None = None,
    stop: threading.Event | None = None,
    abandon: Callable[[], Any] | None = None,
    stdin: bytes | None = None,
) -> TreeResult:
    """Run to the end, the time limit, or a stop, and leave nothing behind.

    ``abandon`` runs before the kill when it did not finish by itself, for what the
    tree cannot reach, such as a container whose client is about to die.
    """
    tree = ProcessTree(command, cwd, env, memory_mb=memory_mb, stdin=stdin is not None)
    stdout = CappedReader(tree.process.stdout, output_bytes)
    stderr = CappedReader(tree.process.stderr, output_bytes)
    if stdin is not None:  # on its own thread, because a child that never reads must not hang this
        threading.Thread(target=feed, args=(tree.process.stdin, stdin), daemon=True).start()
    try:
        timed_out, stopped = wait_for_tree(tree, timeout, stop or threading.Event())
        exit_code = None if timed_out or stopped else tree.process.returncode
        if (timed_out or stopped) and abandon:
            abandon()
        tree.kill()  # leftovers too: nothing a command started outlives the command
        return TreeResult(exit_code, stdout.text(), stderr.text(), timed_out, stopped)
    finally:
        tree.close()


def feed(stream: Any, data: bytes) -> None:
    try:
        stream.write(data)
        stream.close()
    except (OSError, ValueError):
        pass  # it stopped listening, which its exit code will show


def wait_for_tree(tree: ProcessTree, timeout: float, stop: threading.Event) -> tuple[bool, bool]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            tree.process.wait(timeout=POLL_SECONDS)
            return False, False
        except subprocess.TimeoutExpired:
            pass
        if stop.is_set():
            return False, True
        if time.monotonic() >= deadline:
            return True, False


def quiet(command: Sequence[str], timeout: float = PROBE_SECONDS) -> bool:
    try:
        return subprocess.run(list(command), capture_output=True, timeout=timeout).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def container_command(
    engine: str, image: str, name: str, command: Sequence[str], workdir: str, network: bool, env_names: Sequence[str], memory_mb: int
) -> list[str]:
    argv = [
        engine, "run", "--rm", "--init", "--pull=never", "--name", name,
        "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=256m",
        "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m", "--pids-limit", str(CONTAINER_PIDS),
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--network", "bridge" if network else "none",
        "--mount", f"type=bind,source={workdir},target={WORK_MOUNT}", "--workdir", WORK_MOUNT,
    ]  # fmt: skip
    for name_of_variable in env_names:  # by name only, so a value never shows in a process list
        argv += ["--env", name_of_variable]
    return [*argv, image, *command]


def bubblewrap_command(command: Sequence[str], workdir: str, network: bool) -> list[str]:
    argv = [
        "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
        "--tmpfs", "/home", "--tmpfs", "/root", "--bind", workdir, workdir, "--chdir", workdir,
        "--unshare-all", "--die-with-parent", "--new-session",
    ]  # fmt: skip
    if network:
        argv.append("--share-net")
    return [*argv, "--", *command]


def unshare_command(command: Sequence[str], network: bool) -> list[str]:
    argv = ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--kill-child", "--mount-proc", "--ipc", "--uts"]
    if not network:
        argv.append("--net")
    return [*argv, "--", *command]


class LocalSandbox:
    """Implements contracts.Sandbox."""

    def __init__(
        self,
        prefer: str | None = None,
        image: str | None = None,
        memory_mb: int | None = None,
        output_bytes: int | None = None,
        audit: Any = None,
    ) -> None:
        self.prefer = prefer or os.environ.get("ISHARDED_SANDBOX", "auto")
        self.image = image or os.environ.get("ISHARDED_SANDBOX_IMAGE", DEFAULT_IMAGE)
        self.memory_mb = memory_mb or int(os.environ.get("ISHARDED_SANDBOX_MEMORY_MB", DEFAULT_MEMORY_MB))
        self.output_bytes = output_bytes or int(os.environ.get("ISHARDED_SANDBOX_OUTPUT_BYTES", DEFAULT_OUTPUT_BYTES))
        self.audit = audit
        self.skipped: list[str] = []  # why anything stronger was not used
        self.running: dict[str, tuple[str | None, threading.Event]] = {}
        self.lock = threading.Lock()
        self._walls: Walls | None = None
        self._detected = False
        if self.prefer not in ("auto", *LEVELS):
            raise ValueError(f"ISHARDED_SANDBOX must be auto, container, namespace, or process, not {self.prefer}.")

    # Finding the walls. Looked for on first use, so starting the assistant runs nothing.

    @property
    def walls(self) -> Walls | None:
        with self.lock:
            if not self._detected:
                self._walls, self.skipped = self._detect()
                self._detected = True
            return self._walls

    @property
    def level(self) -> str:
        walls = self.walls
        return walls.level if walls else "none"

    def _detect(self) -> tuple[Walls | None, list[str]]:
        skipped: list[str] = []
        for level, tool, problem in self._candidates():
            if self.prefer not in ("auto", level):
                continue
            reason = problem()
            if reason is None:
                return Walls(level, tool), skipped
            skipped.append(reason)
        return None, skipped

    def _candidates(self) -> list[tuple[str, str, Any]]:
        return [
            ("container", "docker", lambda: self._engine_problem("docker")),
            ("container", "podman", lambda: self._engine_problem("podman")),
            ("namespace", "bwrap", self._bubblewrap_problem),
            ("namespace", "unshare", self._unshare_problem),
            ("process", "job" if WINDOWS else "session", lambda: None),
        ]

    def _engine_problem(self, engine: str) -> str | None:
        if not shutil.which(engine):
            return f"{engine} is not installed."
        if not quiet([engine, "info"]):
            return f"{engine} is installed but its service is not running."
        if not quiet([engine, "image", "inspect", self.image]):
            return f"{engine} has no local image {self.image}. Nothing is downloaded for you: pull it, or set ISHARDED_SANDBOX_IMAGE."
        return None

    @staticmethod
    def _bubblewrap_problem() -> str | None:
        if not LINUX:
            return "bubblewrap needs Linux."
        if not shutil.which("bwrap"):
            return "bubblewrap is not installed."
        works = quiet(bubblewrap_command(["true"], tempfile.gettempdir(), False))
        return None if works else "bubblewrap is installed but this system refuses it."

    @staticmethod
    def _unshare_problem() -> str | None:
        if not LINUX:
            return "unshare needs Linux."
        if not shutil.which("unshare"):
            return "unshare is not installed."
        return None if quiet(unshare_command(["true"], False)) else "unshare is installed but user namespaces are off."

    def describe(self) -> dict[str, Any]:
        walls = self.walls
        return {"level": self.level, "tool": walls.tool if walls else None, "weaker_because": list(self.skipped)}

    # Running.

    def run(
        self,
        command: Sequence[str],
        workdir: Path,
        *,
        timeout: float = 120.0,
        network: bool = False,
        env: dict[str, str] | None = None,
        label: str | None = None,
    ) -> SandboxRun:
        """Run to the end or to the time limit. A label lets stop() find this run."""
        words = [str(word) for word in command]
        if not words:
            raise ValueError("Nothing to run: the command is empty.")
        walls = self.walls
        if walls is None:
            reasons = " ".join(self.skipped)
            return self._not_run(126, f"ISHARDED_SANDBOX asks for {self.prefer}, which is not available. {reasons}")
        folder = Path(workdir)
        if not folder.is_dir():
            return self._not_run(126, f"{folder} is not a folder, so there is nowhere to run this.", walls.level)
        with tempfile.TemporaryDirectory(prefix="isharded-sandbox-", ignore_cleanup_errors=True) as home:
            result = self._run_in(walls, words, str(folder.resolve()), home, timeout, network, env or {}, label)
        self._record(walls, words, network, result)
        return result

    def _run_in(
        self, walls: Walls, words: list[str], folder: str, home: str, timeout: float, network: bool, env: dict[str, str], label: str | None
    ) -> SandboxRun:
        name = f"isharded-{secrets.token_hex(6)}"
        argv, cwd, environment, memory = self._prepare(walls, name, words, folder, home, network, env)
        stop = threading.Event()
        # The client dying would leave the container running.
        abandon = (lambda: quiet([walls.tool, "rm", "--force", name], DRAIN_SECONDS * 5)) if walls.level == "container" else None
        with self.lock:
            self.running[name] = (label, stop)
        try:
            ran = run_tree(
                argv, cwd, environment, timeout=timeout, output_bytes=self.output_bytes, memory_mb=memory, stop=stop, abandon=abandon
            )
        except OSError as error:
            return self._not_run(127, f"Could not start {words[0]}: {error}", walls.level)
        finally:
            with self.lock:
                self.running.pop(name, None)
        return self._result(walls, ran.exit_code, ran.stdout, ran.stderr, ran.timed_out, ran.stopped, network)

    def _prepare(
        self, walls: Walls, name: str, words: list[str], folder: str, home: str, network: bool, env: dict[str, str]
    ) -> tuple[list[str], str | None, dict[str, str], int | None]:
        if walls.level == "container":
            client = scrubbed_env(home)
            for variable in KEPT_FOR_CLIENT:  # the client's own settings, not the command's
                if variable in os.environ:
                    client[variable] = os.environ[variable]
            client.update(env)
            argv = container_command(walls.tool, self.image, name, words, folder, network, sorted(env), self.memory_mb)
            return argv, None, client, None
        if walls.tool == "bwrap":
            # /tmp is the sandbox's own there, so the folder made here is out of sight.
            inside = scrubbed_env("/tmp", env)
            return bubblewrap_command(with_rlimits(words, self.memory_mb), folder, network), folder, inside, None
        if walls.tool == "unshare":
            argv = unshare_command(with_rlimits(words, self.memory_mb), network)
            return argv, folder, scrubbed_env(home, env), None
        return words, folder, scrubbed_env(home, env), self.memory_mb

    def _result(
        self, walls: Walls, exit_code: int | None, out: str, err: str, timed_out: bool, stopped: bool, network: bool
    ) -> SandboxRun:
        notes: list[str] = []
        enforced = walls.level != "process"
        if walls.level == "process":
            notes.append(SAME_USER)
            if not network:
                notes.append(NETWORK_NOT_BLOCKED)
                err = f"{err}\n{NETWORK_NOT_BLOCKED}" if err else NETWORK_NOT_BLOCKED
        if walls.tool == "unshare":
            notes.append(NO_FILE_WALL)
        return SandboxRun(exit_code, out, err, timed_out, walls.level, enforced and not network, stopped, tuple(notes))

    def _not_run(self, code: int, why: str, level: str = "none") -> SandboxRun:
        return SandboxRun(code, "", why, False, level, False, False, (why,))

    def _record(self, walls: Walls, words: list[str], network: bool, result: SandboxRun) -> None:
        if self.audit is None:
            return
        digest = hashlib.sha256(canonical(words).encode("utf-8", "surrogatepass")).hexdigest()
        detail = {
            "command_hash": digest,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stopped": result.stopped,
            "network_asked": network,
            "network_blocked": result.network_blocked,
        }
        self.audit.record("sandbox", "sandbox.run", walls.level, detail)

    def stop(self, label: str) -> int:
        """Stop every run with this label. Returns how many were told to stop."""
        with self.lock:
            matching = [event for held, event in self.running.values() if held == label]
        for event in matching:
            event.set()
        return len(matching)

    def stop_all(self) -> None:
        with self.lock:
            matching = [event for _, event in self.running.values()]
        for event in matching:
            event.set()


def attach(assistant: Any) -> None:
    assistant.sandbox = LocalSandbox(audit=assistant.audit)


def detach(assistant: Any) -> None:
    sandbox = getattr(assistant, "sandbox", None)
    if isinstance(sandbox, LocalSandbox):
        sandbox.stop_all()


def add_commands(commands: Any) -> None:
    sandbox = commands.add_parser("sandbox", help="show how strong the walls around generated code really are")
    sandbox.set_defaults(run=run_sandbox)


def run_sandbox(assistant: Any, args: Any) -> int:
    report = assistant.sandbox.describe()
    print(f"Level: {report['level']} ({report['tool'] or 'nothing can run'})")
    for reason in report["weaker_because"]:
        print(f"  {reason}")
    if report["level"] == "process":
        print(f"  {SAME_USER}")
        print("  The network cannot be blocked at this level, and each result says so.")
    return 0 if report["level"] != "none" else 1
