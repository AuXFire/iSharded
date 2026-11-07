"""Tools from outside: a manifest and a command, or an MCP server, behind the broker.

A plugin is someone else's code, so it runs as a separate process with a scrubbed
environment and a time limit, and nothing it says is believed. Its output is always
EXTERNAL. It cannot name itself into a built-in's place, because its capabilities
are always prefixed. And a plugin that keeps crashing is turned off, while the rest
of the assistant carries on.

An MCP server's own description of itself is not evidence. Its ``readOnlyHint`` says
what it would like to be treated as, so the owner states its effects at install and
the default assumes the worst. What each tool claims at install is fingerprinted: a
server that later changes a tool's description or arguments is refused, because a
person agreed to the old one.

These run with process walls as this user, whatever level the sandbox has, because
their programs are on this machine and not in a container image. So installing one
is trusting its code, and ``plugin add`` is a person's decision.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any

from isharded.mcp_client import START_SECONDS, LineRpc, content_parts

from .broker import DEFAULT_USER
from .contracts import CallContext, Capability, Effect, Provenance, ToolResult, Trust
from .mind import is_person
from .sandbox import CappedReader, ProcessTree, run_tree, scrubbed_env
from .store import canonical

PLUGIN_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
DEFAULT_EFFECTS = (Effect.EXECUTE, Effect.EXTERNAL)
MAX_FAILURES = 3
DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 3600.0
DEFAULT_MEMORY_MB = 2048
MIN_MEMORY_MB = 16
MAX_MEMORY_MB = 256 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_SUMMARY = 200
MCP_PROTOCOL = "2025-06-18"

SCHEMA = [
    """
    CREATE TABLE plugins (
        name TEXT PRIMARY KEY,
        protocol TEXT NOT NULL,
        spec TEXT NOT NULL,
        enabled INTEGER NOT NULL,
        failures INTEGER NOT NULL,
        disabled_reason TEXT,
        installed_by TEXT NOT NULL,
        installed_at REAL NOT NULL
    );
    """,
]


class PluginFailure(Exception):
    """It crashed, hung, or spoke nonsense. Counts toward turning it off."""


class ToolError(Exception):
    """It answered properly, and the answer was an error. The plugin itself is fine."""


class ToolChanged(Exception):
    """A server changed what a tool claims. Not a crash: it did it on purpose, or was changed."""


def require_person(by: str, action: str) -> None:
    if not is_person(by):
        raise ValueError(f"{by or 'Nobody'} is not a person, and only a person can {action} a plugin.")


def memory_limit(manifest: dict[str, Any]) -> int:
    try:
        limit = int(manifest.get("memory_mb", DEFAULT_MEMORY_MB))
    except (TypeError, ValueError, OverflowError):
        limit = 0
    if not MIN_MEMORY_MB <= limit <= MAX_MEMORY_MB:  # 0 would mean no limit at all to the sandbox
        raise ValueError(f"memory_mb must be a whole number from {MIN_MEMORY_MB} to {MAX_MEMORY_MB}.")
    return limit


def parse_effects(words: Sequence[str]) -> list[str]:
    effects = []
    for word in words:
        try:
            effects.append(Effect(word.strip().lower()).value)
        except ValueError:
            raise ValueError(f"{word!r} is not an effect. Use: {', '.join(effect.value for effect in Effect)}.") from None
    if not effects:
        raise ValueError("A capability must state at least one effect.")
    return sorted(set(effects))


def fingerprint(tool: dict[str, Any]) -> str:
    """Covers everything a person agreed to, and nothing that changes without meaning."""
    claim = {key: tool.get(key) for key in ("name", "description", "inputSchema")}
    return hashlib.sha256(canonical(claim).encode("utf-8", "surrogatepass")).hexdigest()


def text_of(content: Any) -> str:
    """The text of an MCP result, saying what was left out."""
    return "\n".join(content_parts(content)) if isinstance(content, list) else ""


class McpSession(LineRpc):
    """A stdio MCP client over a long-lived process, held in the sandbox's process walls."""

    failure = PluginFailure
    refusal = ToolError
    ensure_ascii = True

    def __init__(self, command: Sequence[str], cwd: str | None, env: dict[str, str], memory_mb: int) -> None:
        try:
            self.tree = ProcessTree(command, cwd, env, memory_mb=memory_mb, stdin=True)
        except OSError as error:
            raise PluginFailure(f"it would not start: {error}") from None
        self.stderr = CappedReader(self.tree.process.stderr, MAX_STDERR_BYTES)
        super().__init__(self.tree.process.stdin, self.tree.process.stdout)

    def _too_slow(self, method: str, timeout: float) -> str:
        return f"{method} took longer than {timeout:g} seconds"

    def start(self) -> None:
        identity = {"name": "isharded-assistant", "version": "1"}
        # We offer no sampling and no roots: a server cannot call back into a model.
        self.request("initialize", {"protocolVersion": MCP_PROTOCOL, "capabilities": {}, "clientInfo": identity}, START_SECONDS)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def close(self) -> None:
        self.tree.close()


class Plugins:
    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.audit = assistant.audit
        self.clock = assistant.clock
        self.sessions: dict[str, McpSession] = {}
        self.lock = threading.Lock()
        self.starting: dict[str, threading.Lock] = {}
        self.store.migrate("plugins", SCHEMA)
        for row in self.store.query("SELECT * FROM plugins ORDER BY name"):
            self._register(row["name"], json.loads(row["spec"]))

    @property
    def broker(self) -> Any:
        return getattr(self.assistant, "broker", None)

    # Installing.

    def add(
        self,
        path: str | Path,
        by: str = DEFAULT_USER,
        effects: Sequence[str] | None = None,
        tool_effects: dict[str, Sequence[str]] | None = None,
        env: Sequence[str] = (),
    ) -> list[str]:
        """Install from a manifest, or re-enable it if it is the same. Returns its capabilities."""
        require_person(by, "install")
        file = Path(path)
        try:
            manifest = json.loads(file.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"Could not read {file}: {error}") from None
        except ValueError as error:
            raise ValueError(f"{file} is not JSON: {error}") from None
        spec = self._spec(manifest, file.resolve().parent, env)
        protocol = spec["protocol"]
        spec["capabilities"] = (
            self._mcp_capabilities(manifest, spec, effects, tool_effects or {}) if protocol == "mcp" else self._json_capabilities(manifest, effects, tool_effects or {})
        )
        self._install(spec, by)
        return [entry["name"] for entry in spec["capabilities"]]

    @staticmethod
    def _spec(manifest: Any, folder: Path, env: Sequence[str]) -> dict[str, Any]:
        if not isinstance(manifest, dict):
            raise ValueError("A manifest must be a JSON object.")
        name = manifest.get("name")
        if not isinstance(name, str) or not PLUGIN_NAME.match(name):
            raise ValueError("A manifest needs a name: lower-case letters, digits, dash, or underscore.")
        command = manifest.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(word, str) and word for word in command):
            raise ValueError("A manifest needs a command, as a list of words.")
        protocol = manifest.get("protocol", "json")
        if protocol not in ("json", "mcp"):
            raise ValueError("protocol must be json or mcp.")
        timeout = manifest.get("timeout_seconds", DEFAULT_TIMEOUT)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= MAX_TIMEOUT:
            raise ValueError(f"timeout_seconds must be above 0 and at most {MAX_TIMEOUT:g}.")
        memory_mb = memory_limit(manifest)
        return {
            "name": name,
            "protocol": protocol,
            "command": command,
            "cwd": str(folder),
            "timeout": float(timeout),
            "memory_mb": memory_mb,
            "env": sorted(set(env)),  # names only: the value stays in the owner's environment
        }

    @staticmethod
    def _stated(tool: str, stated_by_owner: Any, fallback: Any, tool_effects: dict[str, Sequence[str]]) -> list[str]:
        if tool in tool_effects:
            return parse_effects(tool_effects[tool])
        if stated_by_owner:
            return parse_effects(stated_by_owner)
        if fallback:
            return parse_effects(fallback)
        return sorted(effect.value for effect in DEFAULT_EFFECTS)

    def _json_capabilities(self, manifest: dict[str, Any], effects: Sequence[str] | None, tool_effects: dict[str, Sequence[str]]) -> list[dict[str, Any]]:
        declared = manifest.get("capabilities")
        if not isinstance(declared, list) or not declared:
            raise ValueError("A manifest needs a list of capabilities.")
        made = []
        for entry in declared:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not TOOL_NAME.match(entry["name"]):
                raise ValueError("Each capability needs a name of letters, digits, dot, dash, or underscore.")
            made.append(
                {
                    "name": f"plugin.{manifest['name']}.{entry['name']}",
                    "tool": entry["name"],
                    "summary": str(entry.get("summary", ""))[:MAX_SUMMARY],
                    "effects": self._stated(entry["name"], effects, entry.get("effects"), tool_effects),
                    "reads": [str(item) for item in entry.get("reads", [])],
                    "resources": [str(item) for item in entry.get("resources", [])],
                    "input_schema": entry.get("input_schema", {}),
                    "scope_field": entry.get("scope_field"),
                    "idempotent": entry.get("idempotent") is True,
                }
            )
        return made

    def _mcp_capabilities(
        self, manifest: dict[str, Any], spec: dict[str, Any], effects: Sequence[str] | None, tool_effects: dict[str, Sequence[str]]
    ) -> list[dict[str, Any]]:
        stated = manifest.get("tools", {})  # the owner's own notes on tools, not anything the server said
        session = self._open_session(spec)
        try:
            session.start()
            tools = session.list_tools()
        except (PluginFailure, ToolError) as error:
            raise ValueError(f"Could not talk to that MCP server: {error}") from None
        finally:
            session.close()
        made = []
        for tool in tools:
            name = tool.get("name")
            if not isinstance(name, str) or not TOOL_NAME.match(name):
                continue
            own = stated.get(name, {}) if isinstance(stated, dict) else {}
            description = str(tool.get("description", ""))[:MAX_SUMMARY]
            made.append(
                {
                    "name": f"mcp.{manifest['name']}.{name}",
                    "tool": name,
                    "summary": f"[described by the server, unverified] {description}",
                    "effects": self._stated(name, effects, own.get("effects"), tool_effects),
                    "reads": [],
                    "resources": [],
                    "input_schema": tool.get("inputSchema", {}) if isinstance(tool.get("inputSchema"), dict) else {},
                    "scope_field": own.get("scope_field"),
                    "idempotent": own.get("idempotent") is True,
                    "fingerprint": fingerprint(tool),
                }
            )
        if not made:
            raise ValueError("That MCP server offers no tools.")
        return made

    def _install(self, spec: dict[str, Any], by: str) -> None:
        name = spec["name"]
        row = self.store.one("SELECT spec FROM plugins WHERE name = ?", (name,))
        existing = json.loads(row["spec"]) if row is not None else None
        same = existing == spec
        with self.lock:
            self._drop_session(name)
        if existing is not None and not same:
            self._forget(existing, by)  # what was granted was for the old claims
        with self.store.transaction():
            self.store.execute(
                "INSERT INTO plugins (name, protocol, spec, enabled, failures, installed_by, installed_at) VALUES (?, ?, ?, 1, 0, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET protocol = excluded.protocol, spec = excluded.spec, enabled = 1, failures = 0, "
                "disabled_reason = NULL, installed_by = excluded.installed_by, installed_at = excluded.installed_at",
                (name, spec["protocol"], json.dumps(spec), by, self.clock()),
            )
            self.audit.record(by, "plugin.add", name, {"protocol": spec["protocol"], "capabilities": [entry["name"] for entry in spec["capabilities"]]})
        if not same:
            self._register(name, spec)

    def _register(self, name: str, spec: dict[str, Any]) -> None:
        broker = self.broker
        if broker is None:
            return
        for entry in spec["capabilities"]:
            capability = Capability(
                name=entry["name"],
                summary=entry["summary"],
                effects=frozenset(Effect(effect) for effect in entry["effects"]),
                reads=tuple(entry["reads"]),
                resources=tuple(entry["resources"]),
                input_schema=entry["input_schema"],
                scope_field=entry["scope_field"],
                idempotent=entry["idempotent"],
            )
            broker.register(capability, partial(self.call, name, entry))

    def _forget(self, spec: dict[str, Any], by: str) -> None:
        broker = self.broker
        if broker is None:
            return
        for entry in spec["capabilities"]:
            broker.unregister(entry["name"])
            broker.revoke_capability(entry["name"], by)

    def remove(self, name: str, by: str = DEFAULT_USER) -> bool:
        require_person(by, "remove")
        row = self.store.one("SELECT spec FROM plugins WHERE name = ?", (name,))
        if row is None:
            return False
        with self.lock:
            self._drop_session(name)
        self._forget(json.loads(row["spec"]), by)
        with self.store.transaction():
            self.store.execute("DELETE FROM plugins WHERE name = ?", (name,))
            self.audit.record(by, "plugin.remove", name, {})
        return True

    def list(self) -> list[dict[str, Any]]:
        rows = self.store.query("SELECT * FROM plugins ORDER BY name")
        return [{**dict(row), "spec": json.loads(row["spec"])} for row in rows]

    # Calling.

    def _environment(self, spec: dict[str, Any]) -> dict[str, str]:
        home = Path(self.assistant.home) / "plugin-data" / spec["name"]
        home.mkdir(parents=True, exist_ok=True)
        chosen = {name: os.environ[name] for name in spec["env"] if name in os.environ}
        return scrubbed_env(home, chosen)

    def _open_session(self, spec: dict[str, Any]) -> McpSession:
        return McpSession(spec["command"], spec["cwd"], self._environment(spec), spec["memory_mb"])

    def _result(self, name: str, ok: bool, output: str, data: dict[str, Any] | None = None) -> ToolResult:
        # Always EXTERNAL. A plugin's claim about its own trust is not looked at.
        return ToolResult(ok, output, Provenance("plugin", name, name, Trust.EXTERNAL, self.clock()), data or {})

    def call(self, name: str, entry: dict[str, Any], arguments: dict[str, Any], context: CallContext) -> ToolResult:
        row = self.store.one("SELECT * FROM plugins WHERE name = ?", (name,))
        if row is None or not row["enabled"]:
            why = row["disabled_reason"] if row else "it was removed"
            return self._result(name, False, f"Plugin {name} is turned off: {why} To turn it back on, install it again: isharded assistant plugin add MANIFEST")
        spec = json.loads(row["spec"])
        try:
            if spec["protocol"] == "mcp":
                result = self._call_mcp(name, spec, entry, arguments)
            else:
                result = self._call_json(name, spec, entry, arguments, context)
        except PluginFailure as failure:
            return self._failed(name, str(failure))
        self.store.execute("UPDATE plugins SET failures = 0 WHERE name = ? AND failures != 0", (name,))
        return result

    def _failed(self, name: str, why: str) -> ToolResult:
        with self.store.transaction():
            self.store.execute("UPDATE plugins SET failures = failures + 1 WHERE name = ?", (name,))
            failures = self.store.one("SELECT failures FROM plugins WHERE name = ?", (name,))["failures"]
            self.audit.record("plugins", "plugin.failure", name, {"failures": failures})
            if failures >= MAX_FAILURES:
                self._disable(name, f"it failed {failures} times in a row, most recently because {why}.")
                return self._result(name, False, f"Plugin {name} failed ({why}), and it is now turned off.")
        return self._result(name, False, f"Plugin {name} failed: {why}.")

    def _disable(self, name: str, why: str) -> None:
        self.store.execute("UPDATE plugins SET enabled = 0, disabled_reason = ? WHERE name = ?", (why, name))
        self.audit.record("plugins", "plugin.disable", name, {"why": why})
        with self.lock:
            self._drop_session(name)

    def _call_json(self, name: str, spec: dict[str, Any], entry: dict[str, Any], arguments: dict[str, Any], context: CallContext) -> ToolResult:
        request = {"capability": entry["tool"], "arguments": arguments, "user": context.user}
        try:
            ran = run_tree(
                spec["command"],
                spec["cwd"],
                self._environment(spec),
                timeout=spec["timeout"],
                output_bytes=MAX_OUTPUT_BYTES,
                memory_mb=spec["memory_mb"],
                stdin=json.dumps(request).encode() + b"\n",
            )
        except OSError as error:
            raise PluginFailure(f"it would not start: {error}") from None
        if ran.timed_out:
            raise PluginFailure(f"it took longer than {spec['timeout']:g} seconds")
        if ran.exit_code in {126, 127} and not ran.stdout:
            raise PluginFailure(f"it would not start: the command could not be run, code {ran.exit_code}")
        try:
            reply = json.loads(ran.stdout)
        except ValueError:
            raise PluginFailure(f"it ended with code {ran.exit_code} and no valid reply") from None
        if not isinstance(reply, dict) or not isinstance(reply.get("ok"), bool):
            raise PluginFailure("its reply had no ok field")
        data = reply.get("data") if isinstance(reply.get("data"), dict) else {}
        return self._result(name, reply["ok"], str(reply.get("output", "")), data)

    def _session(self, name: str, spec: dict[str, Any]) -> McpSession:
        """One start at a time, so no call can reach a server before its handshake."""
        with self.lock:
            starting = self.starting.setdefault(name, threading.Lock())
        with starting:
            with self.lock:
                session = self.sessions.get(name)
                if session is not None and session.alive:
                    return session
                self._drop_session(name)
                session = self._open_session(spec)
                self.sessions[name] = session
            try:
                session.start()
                self._check_claims(name, spec, session)
            except ToolError as error:
                raise PluginFailure(f"it refused to start: {error}") from None
            return session

    def _check_claims(self, name: str, spec: dict[str, Any], session: McpSession) -> None:
        offered = {tool.get("name"): fingerprint(tool) for tool in session.list_tools()}
        for entry in spec["capabilities"]:
            if offered.get(entry["tool"]) not in (None, entry["fingerprint"]):
                self._disable(name, f"tool {entry['tool']} changed what it claims since it was installed, so it needs a person to look again.")
                raise ToolChanged(entry["tool"])

    def _call_mcp(self, name: str, spec: dict[str, Any], entry: dict[str, Any], arguments: dict[str, Any]) -> ToolResult:
        try:
            result = self._session(name, spec).request("tools/call", {"name": entry["tool"], "arguments": arguments}, spec["timeout"])
        except ToolChanged as changed:
            return self._result(name, False, f"Tool {changed} changed what it claims since it was installed, so plugin {name} is turned off.")
        except ToolError as error:
            return self._result(name, False, str(error))
        except PluginFailure:
            with self.lock:
                self._drop_session(name)  # a hung or broken server is not given a second call
            raise
        structured = result.get("structuredContent")
        return self._result(name, not result.get("isError", False), text_of(result.get("content")), structured if isinstance(structured, dict) else {})

    def _drop_session(self, name: str) -> None:
        session = self.sessions.pop(name, None)
        if session is not None:
            session.close()

    def close(self) -> None:
        with self.lock:
            for name in list(self.sessions):
                self._drop_session(name)


def attach(assistant: Any) -> None:
    assistant.plugins = Plugins(assistant)


def detach(assistant: Any) -> None:
    plugins = getattr(assistant, "plugins", None)
    if isinstance(plugins, Plugins):
        plugins.close()


def add_commands(commands: Any) -> None:
    plugin = commands.add_parser("plugin", help="install tools from outside, behind the broker")
    actions = plugin.add_subparsers(dest="plugin_command", required=True)
    add = actions.add_parser("add", help="install a plugin or an MCP server from a manifest, or turn it back on")
    add.add_argument("manifest", help="the manifest file; see docs/ASSISTANT.md")
    add.add_argument("--effects", default=None, help="what every tool in it can do, comma separated, overriding the manifest")
    add.add_argument("--tool-effects", action="append", default=[], metavar="TOOL=EFFECTS", help="what one tool can do; can repeat")
    add.add_argument("--env", action="append", default=[], metavar="NAME", help="an environment variable to pass through, by name; can repeat")
    add.add_argument("--by", default=DEFAULT_USER, help="the person installing it, for the record")
    add.set_defaults(run=run_plugin_add)
    listing = actions.add_parser("list", help="show installed plugins")
    listing.set_defaults(run=run_plugin_list)
    remove = actions.add_parser("remove", help="remove a plugin and end every grant for it")
    remove.add_argument("name", help="the plugin")
    remove.add_argument("--by", default=DEFAULT_USER, help="the person removing it, for the record")
    remove.set_defaults(run=run_plugin_remove)


def run_plugin_add(assistant: Any, args: Any) -> int:
    try:
        effects = args.effects.split(",") if args.effects else None
        per_tool = {}
        for pair in args.tool_effects:
            tool, separator, words = pair.partition("=")
            if not separator:
                raise ValueError(f"--tool-effects takes TOOL=EFFECTS, not {pair}.")
            per_tool[tool] = words.split(",")
        names = assistant.plugins.add(args.manifest, args.by, effects, per_tool, args.env)
    except ValueError as error:
        print(error)
        return 1
    print(f"Installed {len(names)} capabilities. None can do anything until a grant covers it.")
    for name in names:
        capability = assistant.broker.capability(name)
        print(f"  {name}  {','.join(sorted(effect.value for effect in capability.effects))}")
    return 0


def run_plugin_list(assistant: Any, args: Any) -> int:
    rows = assistant.plugins.list()
    for row in rows:
        state = "on" if row["enabled"] else f"off: {row['disabled_reason']}"
        print(f"{row['name']:<20} {row['protocol']:<5} {len(row['spec']['capabilities'])} capabilities  {state}")
    if not rows:
        print("No plugins.")
    return 0


def run_plugin_remove(assistant: Any, args: Any) -> int:
    try:
        removed = assistant.plugins.remove(args.name, args.by)
    except ValueError as error:
        print(error)
        return 1
    print(f"Removed {args.name}." if removed else f"There is no plugin {args.name}.")
    return 0 if removed else 1

