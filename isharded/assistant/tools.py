"""The built-in capabilities, each honest about what it can do.

A declared effect is a promise the broker acts on, so these are stated for the
worst a call could do, not the usual case. ``web.fetch`` is NETWORK and not READ,
because a GET can carry data out in its URL. ``shell.run`` admits to writing and
to the network, because the weakest sandbox stops neither.

For the same reason the read-only git tools refuse a repository whose own config
would make git run something. Sandboxed code can write to the folder it works in,
so otherwise it could plant a command there for a later "read" to run outside.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import ssl
import tempfile
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
import psutil

from .broker import path_within
from .contracts import CallContext, Capability, Effect, Provenance, Reach, ToolResult, Trust
from .sandbox import KEPT_EVERYWHERE, KEPT_ON_WINDOWS, WINDOWS, run_tree, scrubbed_env

MAX_READ_BYTES = 1024 * 1024
MAX_LIST_ENTRIES = 2000
MAX_SEARCH_MATCHES = 200
MAX_SEARCH_FILE_BYTES = 2 * 1024 * 1024
MAX_SEARCH_LINE = 300
SKIPPED_FOLDERS = {".git", "__pycache__", "node_modules", ".venv"}
UNDO_KEEP = 500
SHELL_SECONDS = 120.0
MAX_SHELL_SECONDS = 3600.0
FETCH_BYTES = 2 * 1024 * 1024
MAX_FETCH_BYTES = 20 * 1024 * 1024
FETCH_SECONDS = 30.0
MAX_FETCH_SECONDS = 120.0
MAX_REDIRECTS = 5
REDIRECTS = (301, 302, 303, 307, 308)
NAT64 = ipaddress.ip_network("64:ff9b::/96")
GIT_SECONDS = 60.0
GIT_PUSH_SECONDS = 300.0
GIT_OUTPUT_BYTES = 4 * 1024 * 1024
GIT_REF = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./@{}~^-]*$")
GIT_REMOTE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
# Settings in a repository that would run a program or send somewhere else, which
# no flag below turns off. The rest are pinned on the command line instead.
GIT_UNSAFE_CONFIG = re.compile(
    r"^(filter\..*\.(clean|smudge|process)|diff\..*\.(textconv|command)|merge\..*\.driver|credential\..*|"
    r"core\.(gitproxy|askpass|sshcommand|worktree)|remote\..*\.(uploadpack|receivepack|vcs)|include\..*|includeif\..*)$",
    re.IGNORECASE,
)
DEFAULT_MESH_URL = "http://127.0.0.1:8000"
MESH_SECONDS = 5.0


class Refusal(Exception):
    """The tool will not do this, for a reason a person can act on."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def path_property(words: str) -> dict[str, Any]:
    return {"type": "string", "format": "path", "description": words}


def schema(required: list[str], **properties: Any) -> dict[str, Any]:
    return {"type": "object", "required": required, "properties": properties, "additionalProperties": False}


def is_public(address: str) -> bool:
    """Whether an address is on the open internet, seeing through IPv6 wrappers."""
    ip = ipaddress.ip_address(address)
    if ip.version == 6:
        if ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        elif ip in NAT64:
            ip = ipaddress.ip_address(int(ip) & 0xFFFFFFFF)
    return ip.is_global and not ip.is_multicast


def resolve_host(host: str, port: int) -> list[str]:
    try:
        answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise Refusal(f"Could not find {host}: {error}") from None
    return list(dict.fromkeys(answer[4][0].split("%")[0] for answer in answers))


def trust_store() -> ssl.SSLContext:
    """This machine's trusted certificates, as a browser here would use."""
    return ssl.create_default_context()


class Builtins:
    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.clock = assistant.clock
        self.home = os.path.realpath(assistant.home)
        self.undo_folder = Path(self.home) / "undo"
        self.no_hooks = Path(self.home) / "git-no-hooks"

    def catalog(self) -> list[tuple[Capability, Callable[[dict[str, Any], CallContext], ToolResult]]]:
        def declared(name: str, summary: str, effects: set[Effect], handler: Any, **fields: Any) -> tuple[Capability, Any]:
            return Capability(name, summary, frozenset(effects), **fields), self._guarded(name, handler)

        read, local = {Effect.READ}, {Effect.WRITE_LOCAL}
        repo = {"repo": path_property("the repository folder")}
        return [
            declared(
                "files.read", "Read a text file.", read, self.files_read, reads=("files",), resources=("disk",), scope_field="path", idempotent=True,
                input_schema=schema(["path"], path=path_property("the file"), max_bytes={"type": "integer", "minimum": 1}),
            ),
            declared(
                "files.list", "List a folder.", read, self.files_list, reads=("file names",), resources=("disk",), scope_field="path", idempotent=True,
                input_schema=schema(["path"], path=path_property("the folder"), recursive={"type": "boolean"}),
            ),
            declared(
                "files.search", "Find text in the files of a folder.", read, self.files_search, reads=("files",), resources=("disk",), scope_field="path", idempotent=True,
                input_schema=schema(
                    ["path", "text"],
                    path=path_property("the folder"),
                    text={"type": "string", "minLength": 1},
                    glob={"type": "string", "description": "only file names matching this, such as *.py"},
                    ignore_case={"type": "boolean"},
                ),
            ),
            declared(
                "files.write", "Write a file atomically, keeping the old version so it can be rolled back.", local, self.files_write,
                resources=("disk",), scope_field="path", idempotent=True,
                input_schema=schema(["path", "content"], path=path_property("the file"), content={"type": "string"}),
            ),
            declared(
                "shell.run", "Run a command in the sandbox.", {Effect.EXECUTE, Effect.WRITE_LOCAL, Effect.NETWORK}, self.shell_run, scope_field="cwd",
                input_schema=schema(
                    ["command", "cwd"],
                    command={"type": "array", "items": {"type": "string"}, "minItems": 1},
                    cwd=path_property("the folder to run in"),
                    timeout={"type": "number", "exclusiveMinimum": 0, "maximum": MAX_SHELL_SECONDS},
                    network={"type": "boolean"},
                    env={"type": "object", "additionalProperties": {"type": "string"}},
                ),
            ),
            declared(
                "web.fetch", "Fetch a page with GET.", {Effect.NETWORK}, self.web_fetch, reads=("the web",), resources=("network",), scope_field="url", idempotent=True,
                input_schema=schema(
                    ["url"],
                    url={"type": "string", "format": "uri"},
                    max_bytes={"type": "integer", "minimum": 1, "maximum": MAX_FETCH_BYTES},
                    timeout={"type": "number", "exclusiveMinimum": 0, "maximum": MAX_FETCH_SECONDS},
                ),
            ),
            declared(
                "memory.search", "Search this user's memory.", read, self.memory_search, reads=("memory",), idempotent=True,
                input_schema=schema(["query"], query={"type": "string"}, limit={"type": "integer", "minimum": 1, "maximum": 100}),
            ),
            declared(
                "memory.remember", "Store something in this user's memory.", local, self.memory_remember,
                input_schema=schema(["text"], text={"type": "string", "minLength": 1}, kind={"type": "string"}),
            ),
            declared(
                "library.search", "Search the folders this user indexed.", read, self.library_search, reads=("library",), idempotent=True,
                input_schema=schema(["query"], query={"type": "string"}, limit={"type": "integer", "minimum": 1, "maximum": 100}),
            ),
            declared(
                "system.info", "CPU, memory, GPUs, disks, and temperatures where available.", read, self.system_info, reads=("hardware",), idempotent=True,
                input_schema=schema([]),
            ),
            declared(
                "mesh.status", "Whether the mesh is up, and its models.", read, self.mesh_status, reads=("mesh",), resources=("network",), idempotent=True,
                input_schema=schema([]),
            ),
            declared(
                "git.status", "Show changed files.", read, self.git_status, reads=("repository",), scope_field="repo", idempotent=True,
                input_schema=schema(["repo"], **repo),
            ),
            declared(
                "git.diff", "Show changes.", read, self.git_diff, reads=("repository",), scope_field="repo", idempotent=True,
                input_schema=schema(
                    ["repo"], **repo, base={"type": "string"}, target={"type": "string"}, staged={"type": "boolean"},
                    paths={"type": "array", "items": {"type": "string"}},
                ),
            ),
            declared(
                "git.log", "Show recent commits.", read, self.git_log, reads=("repository",), scope_field="repo", idempotent=True,
                input_schema=schema(["repo"], **repo, ref={"type": "string"}, limit={"type": "integer", "minimum": 1, "maximum": 1000}),
            ),
            declared(
                "git.branch", "List branches, or create and switch to one.", local, self.git_branch, scope_field="repo",
                input_schema=schema(["repo"], **repo, name={"type": "string"}, start={"type": "string"}, switch={"type": "boolean"}),
            ),
            declared(
                "git.commit", "Commit staged changes, or the named paths.", local, self.git_commit, scope_field="repo",
                input_schema=schema(
                    ["repo", "message"], **repo, message={"type": "string", "minLength": 1},
                    paths={"type": "array", "items": {"type": "string"}}, all={"type": "boolean"},
                ),
            ),
            declared(
                "git.push", "Push a branch to a configured remote. Never forces.", {Effect.EXTERNAL}, self.git_push, resources=("network",), scope_field="repo",
                input_schema=schema(["repo", "remote", "branch"], **repo, remote={"type": "string"}, branch={"type": "string"}),
            ),
        ]

    def _guarded(self, name: str, handler: Any) -> Callable[[dict[str, Any], CallContext], ToolResult]:
        def run(arguments: dict[str, Any], context: CallContext) -> ToolResult:
            try:
                return handler(arguments, context)
            except Refusal as refusal:
                return self.result(name, False, str(refusal))

        return run

    def result(
        self,
        name: str,
        ok: bool,
        output: str,
        data: dict[str, Any] | None = None,
        source_id: str = "",
        trust: Trust = Trust.EXTERNAL,
        reach: Reach = Reach.POOL,
    ) -> ToolResult:
        return ToolResult(ok, output, Provenance("tool", source_id or name, name, trust, self.clock(), reach), data or {})

    # Files.

    def _usable(self, path: str) -> str:
        """Nothing here may reach the assistant's own folder, whatever a grant says.

        It holds every user's memories and the grants themselves. A part that keeps
        ordinary work under it adds that folder to ``assistant.open_folders``.
        """
        real = os.path.realpath(path)
        opened = getattr(self.assistant, "open_folders", ())
        if path_within(real, self.home) and not any(path_within(real, os.path.realpath(folder)) for folder in opened):
            raise Refusal("That is inside the assistant's own folder, which no tool may touch.")
        return real

    def files_read(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        path = self._usable(arguments["path"])
        limit = min(arguments.get("max_bytes", MAX_READ_BYTES), MAX_READ_BYTES)
        try:
            with open(path, "rb") as stream:
                data = stream.read(limit + 1)
            size = os.path.getsize(path)
        except OSError as error:
            raise Refusal(f"Could not read that file: {error.strerror or error}") from None
        cut = len(data) > limit
        data = data[:limit]
        text = data.decode("utf-8", "replace")
        return self.result("files.read", True, text, {"bytes": size, "truncated": cut, "sha256": digest(data)}, path)

    def _entries(self, root: str, recursive: bool) -> Iterator[tuple[str, os.DirEntry[str]]]:
        """Walk without leaving the folder, whatever links and junctions are in it."""
        waiting = [root]
        while waiting:
            folder = waiting.pop()
            try:
                entries = sorted(os.scandir(folder), key=lambda entry: entry.name)
            except OSError:
                continue
            for entry in entries:
                if not path_within(os.path.realpath(entry.path), root):
                    continue  # a link out: a grant for this folder does not reach it
                yield folder, entry
                if recursive and entry.is_dir(follow_symlinks=False) and entry.name not in SKIPPED_FOLDERS:
                    waiting.append(entry.path)

    def files_list(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        root = self._usable(arguments["path"])
        if not os.path.isdir(root):
            raise Refusal("That is not a folder.")
        lines = []
        for _, entry in self._entries(root, arguments.get("recursive", False)):
            if len(lines) == MAX_LIST_ENTRIES:
                lines.append(f"[stopped after {MAX_LIST_ENTRIES} entries]")
                break
            name = os.path.relpath(entry.path, root)
            lines.append(f"{name}{os.sep}" if entry.is_dir(follow_symlinks=False) else f"{name}  {self._size(entry)}")
        return self.result("files.list", True, "\n".join(lines), {"entries": len(lines)}, root)

    @staticmethod
    def _size(entry: os.DirEntry[str]) -> int:
        try:
            return entry.stat(follow_symlinks=False).st_size
        except OSError:
            return 0

    def files_search(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        """Plain text only. A regular expression from a model could hang this process."""
        import fnmatch

        root = self._usable(arguments["path"])
        ignore = arguments.get("ignore_case", False)
        wanted = arguments["text"].lower() if ignore else arguments["text"]
        pattern = arguments.get("glob", "*")
        matches: list[str] = []
        for _, entry in self._entries(root, True):
            if len(matches) >= MAX_SEARCH_MATCHES:
                matches.append(f"[stopped after {MAX_SEARCH_MATCHES} matches]")
                break
            if entry.is_file(follow_symlinks=False) and fnmatch.fnmatch(entry.name, pattern):
                matches += self._matches_in(entry, root, wanted, ignore)
        return self.result("files.search", True, "\n".join(matches), {"matches": len(matches)}, root)

    def _matches_in(self, entry: os.DirEntry[str], root: str, wanted: str, ignore: bool) -> list[str]:
        if self._size(entry) > MAX_SEARCH_FILE_BYTES:
            return []
        try:
            with open(entry.path, "rb") as stream:
                data = stream.read()
        except OSError:
            return []
        if b"\0" in data[:8192]:
            return []  # binary
        found = []
        for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if wanted in (line.lower() if ignore else line):
                found.append(f"{os.path.relpath(entry.path, root)}:{number}: {line.strip()[:MAX_SEARCH_LINE]}")
        return found

    def files_write(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        path = self._usable(arguments["path"])
        if os.path.isdir(path):
            raise Refusal("That is a folder, not a file.")
        data = arguments["content"].encode("utf-8", "surrogatepass")
        undo = self._keep_previous(path, digest(data), context.user)
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        replace_atomically(path, data)
        return self.result("files.write", True, f"Wrote {len(data)} bytes.", {"bytes": len(data), "sha256": digest(data), "undo": undo}, path)

    def _keep_previous(self, path: str, written: str, user: str) -> str:
        undo = f"{int(self.clock() * 1000):015d}-{secrets.token_hex(4)}"
        folder = self.undo_folder / undo
        folder.mkdir(parents=True)
        existed = os.path.isfile(path)
        if existed:
            shutil.copy2(path, folder / "previous")
        record = {"id": undo, "path": path, "existed": existed, "written_sha256": written, "at": self.clock(), "user": user}
        (folder / "record.json").write_text(json.dumps(record), encoding="utf-8")
        self._prune_undo()
        return undo

    def _prune_undo(self) -> None:
        kept = sorted(self.undo_folder.iterdir(), key=lambda entry: entry.name)
        for old in kept[:-UNDO_KEEP]:
            shutil.rmtree(old, ignore_errors=True)

    def undo_list(self, user: str | None = None) -> list[dict[str, Any]]:
        if not self.undo_folder.is_dir():
            return []
        records = [json.loads((folder / "record.json").read_text(encoding="utf-8")) for folder in sorted(self.undo_folder.iterdir()) if (folder / "record.json").is_file()]
        return [record for record in records if user is None or record["user"] == user]

    def undo_restore(self, undo: str, by: str, force: bool = False) -> str:
        """Put a file back as it was. Refuses to overwrite a newer change unless told to."""
        folder = self.undo_folder / os.path.basename(undo)
        if not (folder / "record.json").is_file():
            raise ValueError(f"There is no undo {undo}. See: isharded assistant undo list")
        record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
        path = record["path"]
        current = digest(Path(path).read_bytes()) if os.path.isfile(path) else None
        if current != record["written_sha256"] and not force:
            raise ValueError(f"{path} changed after that write. To overwrite it anyway, use --force.")
        if record["existed"]:
            replace_atomically(path, (folder / "previous").read_bytes())
        elif os.path.isfile(path):
            os.unlink(path)
        self.assistant.audit.record(by, "files.undo", undo, {"sha256_before": current})
        return path

    # Running things.

    def shell_run(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        sandbox = getattr(self.assistant, "sandbox", None)
        if sandbox is None:
            raise Refusal("shell.run needs the sandbox part, and nothing runs outside it.")
        folder = self._usable(arguments["cwd"])
        options: dict[str, Any] = {
            "timeout": arguments.get("timeout", SHELL_SECONDS),
            "network": arguments.get("network", False),
            "env": arguments.get("env"),
        }
        if hasattr(sandbox, "stop") and context.task_id:
            options["label"] = context.task_id  # so cancelling the task stops this
        run = sandbox.run(arguments["command"], Path(folder), **options)
        output = "\n".join(part for part in (run.stdout, run.stderr) if part)
        data = {
            "exit_code": run.exit_code,
            "timed_out": run.timed_out,
            "level": run.level,
            "network_blocked": getattr(run, "network_blocked", None),
            "stopped": getattr(run, "stopped", False),
            "notes": list(getattr(run, "notes", ())),
        }
        return self.result("shell.run", run.exit_code == 0, output, data, folder)

    # The web.

    def web_fetch(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        url = arguments["url"]
        limit = min(arguments.get("max_bytes", FETCH_BYTES), MAX_FETCH_BYTES)
        deadline = time.monotonic() + min(arguments.get("timeout", FETCH_SECONDS), MAX_FETCH_SECONDS)
        for hop in range(MAX_REDIRECTS + 1):
            address = self._fetch_address(url, context, hop)
            status, location, content_type, body, cut = self._fetch_once(url, address, limit, deadline)
            if status in REDIRECTS and location:
                url = urljoin(url, location)
                continue
            data = {"status": status, "url": url, "content_type": content_type, "bytes": len(body), "truncated": cut, "redirects": hop}
            return self.result("web.fetch", status < 400, body.decode("utf-8", "replace"), data, url)
        raise Refusal(f"Gave up after {MAX_REDIRECTS} redirects.")

    def _fetch_address(self, url: str, context: CallContext, hop: int) -> str:
        """Judge one URL and settle the address, so what was judged is what is reached."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise Refusal("web.fetch only speaks http and https.")
        if parts.username or parts.password:
            raise Refusal("web.fetch will not send a user name or password in a URL.")
        host = parts.hostname or ""
        broker = self.assistant.broker
        if hop and not broker.covers(context.user, "web.fetch", {"url": url}):
            raise Refusal(f"The page redirected to {host}, which no grant covers, so it was not followed.")
        addresses = resolve_host(host, parts.port or (443 if parts.scheme == "https" else 80))
        private = [address for address in addresses if not is_public(address)]
        if private and not broker.names_host(context.user, "web.fetch", url):
            raise Refusal(
                f"{host} is a private address ({private[0]}). To reach it, a grant must name it: "
                f"isharded assistant grant add web.fetch {host} --user {context.user}"
            )
        return addresses[0]

    def _fetch_once(self, url: str, address: str, limit: int, deadline: float) -> tuple[int, str | None, str, bytes, bool]:
        """Connect to the address already judged, so a name that changes its answer gains nothing."""
        parts = urlsplit(url)
        host = parts.hostname or ""
        literal = f"[{address}]" if ":" in address else address
        pinned = parts._replace(netloc=f"{literal}:{parts.port}" if parts.port else literal).geturl()
        name = host.encode("idna").decode()
        headers = {"Host": f"{name}:{parts.port}" if parts.port else name, "User-Agent": "isharded-assistant"}
        left = max(deadline - time.monotonic(), 0.1)
        verify: Any = trust_store() if parts.scheme == "https" else True
        try:
            with httpx.Client(verify=verify, trust_env=False, follow_redirects=False, timeout=left) as client:
                with client.stream("GET", pinned, headers=headers, extensions={"sni_hostname": name}) as response:
                    body, cut = self._read_limited(response, limit, deadline)
                    return response.status_code, response.headers.get("location"), response.headers.get("content-type", ""), body, cut
        except httpx.HTTPError as error:
            raise Refusal(f"Could not fetch that page: {type(error).__name__}: {error}") from None

    @staticmethod
    def _read_limited(response: httpx.Response, limit: int, deadline: float) -> tuple[bytes, bool]:
        """Stop at the limit, decompressed, so a small download cannot become a huge one."""
        body = bytearray()
        for chunk in response.iter_bytes():
            body += chunk
            if len(body) > limit:
                return bytes(body[:limit]), True
            if time.monotonic() > deadline:
                raise Refusal("The page took too long to arrive.")
        return bytes(body), False

    # Other parts.

    def memory_search(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            raise Refusal("memory.search needs the memory part, which is not installed.")
        return self._items("memory.search", memory.search(context.user, arguments["query"], arguments.get("limit", 8)))

    def library_search(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        library = getattr(self.assistant, "library", None)
        if library is None:
            raise Refusal("library.search needs the knowledge part, which is not installed.")
        return self._items("library.search", library.search(context.user, arguments["query"], arguments.get("limit", 8)))

    def _items(self, name: str, items: Any) -> ToolResult:
        found = list(items)
        lines, sources = [], []
        for number, item in enumerate(found, 1):
            provenance = getattr(item, "provenance", None)
            source = getattr(provenance, "source_id", "") if provenance else ""
            sources.append(source)
            lines.append(f"[{number}] {getattr(item, 'text', item)}" + (f" ({source})" if source else ""))
        # The strictest item decides how far all of this may go.
        reaches = [item.provenance.reach for item in found if getattr(item, "provenance", None)]
        return self.result(name, True, "\n".join(lines), {"count": len(found), "sources": sources}, reach=min(reaches, default=Reach.POOL))

    def memory_remember(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            raise Refusal("memory.remember needs the memory part, which is not installed.")
        # A tool call is a model or a task acting, never the person saying it.
        trust = Trust.EXTERNAL if context.tainted else Trust.MODEL
        # What is kept can be no freer than what it was written from, or a private file
        # would only need to be remembered to travel.
        provenance = Provenance("tool", context.actor, context.actor, trust, self.clock(), min(Reach.POOL, context.reach))
        row = memory.remember(context.user, arguments.get("kind", "semantic"), arguments["text"], provenance)
        return self.result("memory.remember", True, f"Remembered as {row}.", {"id": row})

    # This machine.

    def system_info(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        memory = psutil.virtual_memory()
        info: dict[str, Any] = {
            "cpu": {"logical": psutil.cpu_count(), "physical": psutil.cpu_count(logical=False), "percent": psutil.cpu_percent(interval=None)},
            "memory": {"total": memory.total, "available": memory.available, "percent": memory.percent},
            "disks": disks(),
            "gpus": nvidia_gpus(),
            "temperatures": temperatures(),
        }
        # Measured here, by this process, with no outside text in it.
        return self.result("system.info", True, json.dumps(info), info, trust=Trust.VERIFIED)

    def mesh_status(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        base = os.environ.get("ISHARDED_ORCHESTRATOR_URL", DEFAULT_MESH_URL).rstrip("/")
        info: dict[str, Any] = {"url": base, "up": False, "models": []}
        try:
            with httpx.Client(trust_env=False, timeout=MESH_SECONDS) as client:
                info["up"] = client.get(f"{base}/health").status_code == 200
                listing = client.get(f"{base}/v1/models")
            if listing.status_code == 200:
                info["models"] = [str(model.get("id")) for model in listing.json().get("data", [])]
            else:
                info["models_status"] = listing.status_code
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            info["error"] = type(error).__name__
        return self.result("mesh.status", True, json.dumps(info), info, base)

    # Git.

    def _git_env(self) -> dict[str, str]:
        """The owner's identity and keys, with the settings that could prompt or redirect removed."""
        names = (*KEPT_EVERYWHERE, *(KEPT_ON_WINDOWS if WINDOWS else ()), "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "SSH_AUTH_SOCK")
        environment = {name: os.environ[name] for name in names if name in os.environ}
        environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "GIT_EDITOR": "true", "GIT_ASKPASS": "", "SSH_ASKPASS": "", "LC_ALL": "C"})
        return environment

    def _git_flags(self) -> list[str]:
        self.no_hooks.mkdir(parents=True, exist_ok=True)
        pinned = {
            "core.hooksPath": str(self.no_hooks),
            "core.fsmonitor": "false",
            "core.pager": "cat",
            "diff.external": "",
            "commit.gpgsign": "false",
            "tag.gpgsign": "false",
            "log.showSignature": "false",
            "protocol.ext.allow": "never",
        }
        return ["--no-pager", *(word for key, value in pinned.items() for word in ("-c", f"{key}={value}"))]

    def _git_refuse_unsafe(self, repo: str) -> None:
        command = ["git", "config", "--show-scope", "--name-only", "--list"]
        listed = run_tree(command, repo, self._git_env(), timeout=GIT_SECONDS)
        if listed.exit_code != 0:
            return  # not a repository: the command itself will say so
        for line in listed.stdout.splitlines():
            scope, _, key = line.strip().partition("\t")
            # The owner's own global settings are trusted. What lives in the repository is not.
            if scope in ("local", "worktree") and GIT_UNSAFE_CONFIG.match(key):
                raise Refusal(
                    f"This repository's own config sets {key}, which would make git run or send something. "
                    f"If you trust it, use shell.run, or remove it with: git config --local --unset {key}"
                )

    def _git(self, name: str, repo: str, words: list[str], timeout: float = GIT_SECONDS, data: dict[str, Any] | None = None) -> ToolResult:
        folder = self._usable(repo)
        self._git_refuse_unsafe(folder)
        try:
            ran = run_tree(["git", *self._git_flags(), *words], folder, self._git_env(), timeout=timeout, output_bytes=GIT_OUTPUT_BYTES)
        except OSError:
            raise Refusal("git is not installed.") from None
        output = "\n".join(part for part in (ran.stdout, ran.stderr) if part)
        if ran.timed_out:
            output = f"{output}\nStopped after {timeout:g} seconds.".strip()
        return self.result(name, ran.exit_code == 0, output, {"exit_code": ran.exit_code, "timed_out": ran.timed_out, **(data or {})}, folder)

    @staticmethod
    def _ref(value: str, what: str) -> str:
        """Nothing a model writes here may be read as a flag: --output= would write a file."""
        if not GIT_REF.match(value) or value.endswith((".lock", "/")):
            raise Refusal(f"{value!r} is not a usable {what}.")
        return value

    def _paths(self, arguments: dict[str, Any]) -> list[str]:
        paths = arguments.get("paths") or []
        return ["--", *paths] if paths else []

    def git_status(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        return self._git("git.status", arguments["repo"], ["status", "--porcelain=v1", "--branch", "--ignore-submodules=all"])

    def git_diff(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        words = ["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--ignore-submodules=all"]
        if arguments.get("staged"):
            words.append("--cached")
        words += [self._ref(arguments[key], "revision") for key in ("base", "target") if arguments.get(key)]
        return self._git("git.diff", arguments["repo"], [*words, *self._paths(arguments)])

    def git_log(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        words = ["log", "--no-color", "--date=iso-strict", "--format=%H%x09%an%x09%ad%x09%s", f"-n{arguments.get('limit', 20)}"]
        if arguments.get("ref"):
            words.append(self._ref(arguments["ref"], "revision"))
        return self._git("git.log", arguments["repo"], words)

    def git_branch(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        name = arguments.get("name")
        if not name:
            return self._git("git.branch", arguments["repo"], ["branch", "--list", "--no-color"])
        self._ref(name, "branch name")
        words = ["branch", name]
        if arguments.get("start"):
            words.append(self._ref(arguments["start"], "revision"))
        made = self._git("git.branch", arguments["repo"], words)
        if made.ok and arguments.get("switch"):
            return self._git("git.branch", arguments["repo"], ["switch", name])
        return made

    def git_commit(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        repo = arguments["repo"]
        if arguments.get("paths") or arguments.get("all"):
            words = ["add", *self._paths(arguments)] if arguments.get("paths") else ["add", "--all"]
            staged = self._git("git.commit", repo, words)
            if not staged.ok:
                return staged
        return self._git("git.commit", repo, ["commit", "-m", arguments["message"]])

    def git_push(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        remote, branch = arguments["remote"], arguments["branch"]
        if not GIT_REMOTE.match(remote):
            raise Refusal(f"{remote!r} is not the name of a remote. Only a remote already set up in the repository can be used.")
        self._ref(branch, "branch name")
        if ":" in branch or "+" in branch:
            raise Refusal("git.push takes a plain branch name, and it never forces.")
        return self._git("git.push", arguments["repo"], ["push", remote, branch], timeout=GIT_PUSH_SECONDS)


def replace_atomically(path: str, data: bytes) -> None:
    """Whole or not at all: a crash leaves the old file or the new one, never half."""
    handle, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".isharded-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.exists(path):
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def disks() -> list[dict[str, Any]]:
    found = []
    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except OSError:
            continue  # an empty card reader, or a drive that went away
        found.append({"mount": partition.mountpoint, "total": usage.total, "free": usage.free, "percent": usage.percent})
    return found


def nvidia_gpus() -> list[dict[str, Any]]:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return []
    fields = "name,memory.total,memory.used,utilization.gpu,temperature.gpu"
    try:
        ran = run_tree([tool, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], None, scrubbed_env(tempfile.gettempdir()), timeout=10)
    except OSError:
        return []
    if ran.exit_code != 0:
        return []
    return [parse_gpu_line(line) for line in ran.stdout.splitlines() if line.strip()]


def parse_gpu_line(line: str) -> dict[str, Any]:
    name, total, used, load, temperature = ([*(part.strip() for part in line.split(",")), "", "", "", ""])[:5]

    def number(text: str) -> int | None:
        return int(text) if text.isdigit() else None

    return {"name": name, "memory_total_mb": number(total), "memory_used_mb": number(used), "percent": number(load), "temperature_c": number(temperature)}


def temperatures() -> dict[str, float]:
    reader = getattr(psutil, "sensors_temperatures", None)
    if reader is None:
        return {}  # Windows does not offer these
    found = {}
    for group, sensors in reader().items():
        for number, sensor in enumerate(sensors):
            found[sensor.label or f"{group}-{number}"] = sensor.current
    return found


def attach(assistant: Any) -> None:
    builtins = Builtins(assistant)
    assistant.builtins = builtins
    broker = getattr(assistant, "broker", None)
    if broker is None:
        return
    for capability, handler in builtins.catalog():
        broker.register(capability, handler)


def add_commands(commands: Any) -> None:
    undo = commands.add_parser("undo", help="put back a file the assistant wrote")
    actions = undo.add_subparsers(dest="undo_command", required=True)
    listing = actions.add_parser("list", help="show what can be put back, oldest first")
    listing.add_argument("--user", default=None, help="only writes made for this user")
    listing.set_defaults(run=run_undo_list)
    restore = actions.add_parser("restore", help="put a file back as it was before that write")
    restore.add_argument("id", help="the undo, from undo list")
    restore.add_argument("--force", action="store_true", help="even if the file changed again after that write")
    restore.add_argument("--by", default="owner", help="the person doing this, for the record")
    restore.set_defaults(run=run_undo_restore)


def run_undo_list(assistant: Any, args: Any) -> int:
    rows = assistant.builtins.undo_list(args.user)
    for row in rows:
        kind = "changed" if row["existed"] else "created"
        print(f"{row['id']}  {kind}  {row['user']:<12} {row['path']}")
    if not rows:
        print("Nothing to undo.")
    return 0


def run_undo_restore(assistant: Any, args: Any) -> int:
    try:
        path = assistant.builtins.undo_restore(args.id, args.by, args.force)
    except ValueError as error:
        print(error)
        return 1
    print(f"Restored {path}.")
    return 0
