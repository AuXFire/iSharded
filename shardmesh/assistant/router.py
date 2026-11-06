"""Which model does the work.

Roles map to ordered candidates. A request first loses every candidate it may not
use: too far away for its text, down, or over budget. What is left is tried best
first, by measured results. Given the same numbers it always picks the same model,
so a choice can be explained and a test can pin it down.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

from .contracts import ModelReply, ModelRequest, Reach
from .mind import actor, emit, LOCAL_USER, plainly, table
from .providers import VisibleText, estimate_tokens

ROLES = ("conversation", "planning", "coding", "worker", "decision", "vision", "embedding", "verification")
Z_95 = 1.96
# Fewer runs than this say nothing yet. Without it one passing fault would score
# zero, and a good model would never be tried again while anything else worked.
MEASURED_RUNS = 3
BACKOFF_SECONDS = 15.0
MAX_BACKOFF_SECONDS = 300.0
SECONDS_PER_DAY = 86_400

SCHEMA = [
    """
    CREATE TABLE model_routes (
        role TEXT NOT NULL,
        position INTEGER NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        PRIMARY KEY (role, position)
    );
    CREATE TABLE model_stats (
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        role TEXT NOT NULL,
        runs INTEGER NOT NULL DEFAULT 0,
        successes INTEGER NOT NULL DEFAULT 0,
        tokens INTEGER NOT NULL DEFAULT 0,
        seconds REAL NOT NULL DEFAULT 0,
        last_at REAL,
        PRIMARY KEY (provider, model, role)
    );
    CREATE TABLE model_health (
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        failures INTEGER NOT NULL,
        retry_at REAL NOT NULL,
        last_error TEXT NOT NULL,
        PRIMARY KEY (provider, model)
    );
    CREATE TABLE model_usage (
        day INTEGER NOT NULL,
        provider TEXT NOT NULL,
        tokens INTEGER NOT NULL,
        PRIMARY KEY (day, provider)
    );
    """,
]


class Refused(Exception):
    """Nothing may take this request, with the reason for each candidate.

    ``refusal`` marks a decision, as opposed to something going wrong.
    """

    refusal = True

    def __init__(self, message: str, reasons: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reasons = list(reasons)


class AllFailed(Exception):
    """Candidates were allowed, and each one failed."""

    refusal = False

    def __init__(self, message: str, reasons: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reasons = list(reasons)


class StoppedPartway(Exception):
    """A stream failed after text had already gone out, so it cannot move to another model."""

    refusal = False


@dataclass(frozen=True)
class Candidate:
    provider: str
    model: str = ""  # empty means whatever the provider is serving, as a pool changes models

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}" if self.model else self.provider


def wilson_lower_bound(successes: int, runs: int, z: float = Z_95) -> float:
    """How good a success rate is at least, given how little may be known.

    Two out of two is not better than 95 out of 100, and this says so.
    """
    if runs <= 0:
        return 0.0
    rate = min(successes, runs) / runs
    spread = z * z / runs
    centre = rate + spread / 2
    margin = z * math.sqrt(rate * (1 - rate) / runs + z * z / (4 * runs * runs))
    return max((centre - margin) / (1 + spread), 0.0)


def parse_candidate(text: str) -> Candidate:
    # Model names often contain a slash, and provider names never do.
    provider, _, model = text.partition("/")
    return Candidate(provider, model)


class ModelRouter:
    """Implements contracts.Router."""

    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.store.migrate("router", SCHEMA)

    # Routes

    def set_route(self, role: str, candidates: Sequence[str], by: str = LOCAL_USER) -> None:
        if role not in ROLES:
            raise ValueError(f"{role!r} is not a role. Use one of: {', '.join(ROLES)}")
        parsed = [parse_candidate(item) for item in candidates]
        for candidate in parsed:
            if self._provider(candidate.provider) is None:
                raise ValueError(f"There is no provider {candidate.provider}. `provider list` shows them")
        with self.store.transaction():
            self.store.execute("DELETE FROM model_routes WHERE role = ?", (role,))
            for position, candidate in enumerate(parsed):
                self.store.execute(
                    "INSERT INTO model_routes (role, position, provider, model) VALUES (?, ?, ?, ?)",
                    (role, position, candidate.provider, candidate.model),
                )
            self.assistant.audit.record(actor(by), "route.set", role, {"candidates": [str(item) for item in parsed]})

    def _provider(self, name: str) -> Any:
        providers = getattr(self.assistant, "providers", None)
        return providers.get(name) if providers else None

    def _providers(self) -> list[Any]:
        providers = getattr(self.assistant, "providers", None)
        return providers.all() if providers else []

    def candidates(self, role: str) -> list[Candidate]:
        """A role's own route, then the conversation route, then every provider.

        Embedding has no fallback: chat models cannot do it.
        """
        for name in (role,) if role == "embedding" else (role, "conversation"):
            rows = self.store.query("SELECT provider, model FROM model_routes WHERE role = ? ORDER BY position", (name,))
            if rows:
                return [Candidate(row["provider"], row["model"]) for row in rows]
        return [] if role == "embedding" else [Candidate(provider.name) for provider in self._providers()]

    def routes(self) -> dict[str, list[str]]:
        return {role: [str(item) for item in self.candidates(role)] for role in ROLES}

    def can_embed(self) -> bool:
        return bool(self.candidates("embedding"))

    # Choosing

    def _today(self) -> int:
        return int(self.assistant.clock() // SECONDS_PER_DAY)

    def _used_today(self, provider: str) -> int:
        row = self.store.one("SELECT tokens FROM model_usage WHERE day = ? AND provider = ?", (self._today(), provider))
        return row["tokens"] if row else 0

    def _resolve(self, candidate: Candidate, provider: Any) -> tuple[Candidate | None, str | None]:
        """Name the model a provider is serving. Failing to ask counts as being down."""
        if candidate.model:
            return candidate, None
        try:
            model = provider.current_model()
        except Exception as error:
            self._back_off(candidate, error)
            return None, f"{candidate}: {error}"
        return (Candidate(candidate.provider, model), None) if model else (None, f"{candidate}: no model is loaded")

    def _blocked(self, candidate: Candidate, provider: Any, request: ModelRequest) -> str | None:
        if provider.reach > request.reach:
            return (
                f"{candidate} reaches {provider.reach.name.lower()}, and text in this request may go no further than "
                f"{request.reach.name.lower()}"
            )
        row = self.store.one("SELECT retry_at, last_error FROM model_health WHERE provider = ? AND model = ?", (candidate.provider, candidate.model))
        now = self.assistant.clock()
        if row and row["retry_at"] > now:
            return f"{candidate} failed recently and will be tried again in {row['retry_at'] - now:.0f} s ({row['last_error']})"
        limit = getattr(provider, "daily_tokens", None)
        if limit:
            used = self._used_today(candidate.provider)
            # The most this call could cost, so a budget is never overrun.
            need = sum(estimate_tokens(str(message.get("content") or "")) for message in request.messages) + request.max_tokens
            if used + need > limit:
                return f"{candidate} has used {used} of its {limit} tokens today, and this could take {need} more"
        return None

    def _stats(self, candidate: Candidate, role: str) -> tuple[int, int, float]:
        row = self.store.one(
            "SELECT runs, successes, tokens, seconds FROM model_stats WHERE provider = ? AND model = ? AND role = ?",
            (candidate.provider, candidate.model, role),
        )
        if row is None:
            return 0, 0, 0.0
        return row["runs"], row["successes"], (row["tokens"] / row["seconds"] if row["seconds"] else 0.0)

    def _ranked(self, usable: list[Candidate], role: str) -> list[Candidate]:
        """Untested first, since a number can only come from runs. Then the measured best."""

        def key(entry: tuple[int, Candidate]) -> tuple[int, float, float, int]:
            position, candidate = entry
            runs, successes, speed = self._stats(candidate, role)
            if runs < MEASURED_RUNS:
                return (0, 0.0, 0.0, position)
            return (1, -wilson_lower_bound(successes, runs), -speed, position)

        return [candidate for _, candidate in sorted(enumerate(usable), key=key)]

    def plan(self, request: ModelRequest) -> tuple[list[Candidate], list[str]]:
        """The candidates this request may use, best first, and why the others may not."""
        usable, reasons = [], []
        for listed in self.candidates(request.role):
            provider = self._provider(listed.provider)
            if provider is None:
                reasons.append(f"{listed}: this provider has been removed")
                continue
            # Reach and back-off come first: neither should cost a call to the provider.
            blocked = self._blocked(listed, provider, request)
            resolved, problem = (None, blocked) if blocked else self._resolve(listed, provider)
            blocked = problem or (self._blocked(resolved, provider, request) if resolved != listed else None)
            if blocked:
                reasons.append(blocked)
            else:
                usable.append(resolved)
        return self._ranked(usable, request.role), reasons

    def _refuse(self, request: ModelRequest, reasons: list[str]) -> Refused:
        if not reasons:
            message = f"No model is set up for {request.role}. Add one with `provider add`."
        else:
            message = f"No model may take this {request.role} request: " + "; ".join(reasons) + "."
        self.assistant.audit.record(
            "router", "router.refused", request.role, {"reach": request.reach.name.lower(), "reasons": len(reasons)}
        )
        return Refused(message, reasons)

    # Running

    def _record(self, candidate: Candidate, role: str, ok: bool, tokens: int, seconds: float) -> None:
        now = self.assistant.clock()
        with self.store.transaction():
            self.store.execute(
                "INSERT INTO model_stats (provider, model, role, runs, successes, tokens, seconds, last_at) VALUES (?, ?, ?, 1, ?, ?, ?, ?) "
                "ON CONFLICT(provider, model, role) DO UPDATE SET runs = runs + 1, successes = successes + excluded.successes, "
                "tokens = tokens + excluded.tokens, seconds = seconds + excluded.seconds, last_at = excluded.last_at",
                (candidate.provider, candidate.model, role, int(ok), tokens, seconds, now),
            )
            if tokens:
                self.store.execute(
                    "INSERT INTO model_usage (day, provider, tokens) VALUES (?, ?, ?) "
                    "ON CONFLICT(day, provider) DO UPDATE SET tokens = tokens + excluded.tokens",
                    (self._today(), candidate.provider, tokens),
                )

    def _succeeded(self, candidate: Candidate, role: str, reply: ModelReply) -> None:
        self._record(candidate, role, True, reply.prompt_tokens + reply.completion_tokens, reply.seconds)
        self.store.execute("DELETE FROM model_health WHERE provider = ? AND model = ?", (candidate.provider, candidate.model))

    def _back_off(self, candidate: Candidate, error: BaseException) -> int:
        row = self.store.one("SELECT failures FROM model_health WHERE provider = ? AND model = ?", (candidate.provider, candidate.model))
        failures = (row["failures"] if row else 0) + 1
        wait = min(BACKOFF_SECONDS * 2 ** (failures - 1), MAX_BACKOFF_SECONDS)
        self.store.execute(
            "INSERT INTO model_health (provider, model, failures, retry_at, last_error) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(provider, model) DO UPDATE SET failures = excluded.failures, retry_at = excluded.retry_at, last_error = excluded.last_error",
            (candidate.provider, candidate.model, failures, self.assistant.clock() + wait, str(error)[:200]),
        )
        return failures

    def _failed(self, candidate: Candidate, role: str, error: BaseException, seconds: float) -> None:
        self._record(candidate, role, False, 0, seconds)
        failures = self._back_off(candidate, error)
        self.assistant.audit.record("router", "router.failed", str(candidate), {"role": role, "failures": failures})

    def complete(self, request: ModelRequest) -> ModelReply:
        ordered, reasons = self.plan(request)
        if not ordered:
            raise self._refuse(request, reasons)
        failures = []
        for candidate in ordered:
            started = time.perf_counter()
            provider = self._provider(candidate.provider)
            try:
                reply = provider.complete(request, candidate.model)
                if not reply.text.strip() and not reply.tool_calls:
                    raise ValueError(f"{candidate} replied with nothing")
            except Exception as error:  # a provider's bug must not stop the next one being tried
                self._failed(candidate, request.role, error, time.perf_counter() - started)
                failures.append(f"{candidate}: {error}")
                continue
            self._succeeded(candidate, request.role, reply)
            return reply
        raise AllFailed("Every model failed: " + "; ".join(failures) + ".", failures)

    def stream(self, request: ModelRequest) -> Iterator[str | ModelReply]:
        """Text as it comes, then the ModelReply. Moves on only before any text went out."""
        ordered, reasons = self.plan(request)
        if not ordered:
            raise self._refuse(request, reasons)
        failures = []
        for candidate in ordered:
            started = time.perf_counter()
            provider = self._provider(candidate.provider)
            visible = VisibleText(request.tools)
            sent = False
            try:
                for item in provider.stream(request, candidate.model):
                    if isinstance(item, ModelReply):
                        if not item.text.strip() and not item.tool_calls:
                            raise ValueError(f"{candidate} replied with nothing")
                        tail = visible.flush()
                        if tail and not item.tool_calls:
                            yield tail
                        self._succeeded(candidate, request.role, item)
                        yield item
                        return
                    text = visible.feed(item)
                    if text:
                        sent = True
                        yield text
                raise ValueError(f"{candidate} stopped before finishing its reply")
            except Exception as error:
                self._failed(candidate, request.role, error, time.perf_counter() - started)
                if sent:
                    raise StoppedPartway(str(error)) from None
                failures.append(f"{candidate}: {error}")
        raise AllFailed("Every model failed: " + "; ".join(failures) + ".", failures)

    def embed(self, texts: Sequence[str], reach: Reach = Reach.POOL) -> tuple[list[list[float]], str]:
        """Vectors from the best allowed embedding model, and that model's name."""
        request = ModelRequest(role="embedding", messages=[{"role": "user", "content": text} for text in texts], reach=reach, max_tokens=1)
        ordered, reasons = self.plan(request)
        if not ordered:
            raise self._refuse(request, reasons)
        failures = []
        for candidate in ordered:
            started = time.perf_counter()
            try:
                vectors = self._provider(candidate.provider).embed(candidate.model, texts)
                if len(vectors) != len(texts):
                    raise ValueError(f"{candidate} returned {len(vectors)} vectors for {len(texts)} texts")
            except Exception as error:
                self._failed(candidate, "embedding", error, time.perf_counter() - started)
                failures.append(f"{candidate}: {error}")
                continue
            seconds = time.perf_counter() - started
            self._record(candidate, "embedding", True, sum(estimate_tokens(text) for text in texts), seconds)
            return vectors, candidate.model
        raise AllFailed("Every embedding model failed: " + "; ".join(failures) + ".", failures)

    def record_outcome(
        self,
        model: str | ModelReply,
        role: str,
        success: bool,
        provider: str | None = None,
    ) -> None:
        """Feed back what checking the work found.

        A reply counts as a success when it arrives. If verification then fails
        it, that success is taken back, so the rate means work that held up. Pass
        the ModelReply, or the model name with its provider.
        """
        if isinstance(model, ModelReply):
            model, provider = model.model, model.provider
        with self.store.transaction():
            rows = self.store.query(
                "SELECT provider FROM model_stats WHERE model = ? AND role = ?" + (" AND provider = ?" if provider else "") + " ORDER BY provider",
                [model, role, provider] if provider else [model, role],
            )
            name = rows[0]["provider"] if rows else (provider or "")
            if rows and not success:
                self.store.execute(
                    "UPDATE model_stats SET successes = max(successes - 1, 0) WHERE provider = ? AND model = ? AND role = ?", (name, model, role)
                )
            elif not rows:
                self.store.execute(
                    "INSERT INTO model_stats (provider, model, role, runs, successes, last_at) VALUES (?, ?, ?, 1, ?, ?)",
                    (name, model, role, int(success), self.assistant.clock()),
                )
            self.assistant.audit.record("router", "router.outcome", f"{name}/{model}", {"role": role, "success": success})

    def stats(self, role: str | None = None) -> list[dict[str, Any]]:
        clause, parameters = ("WHERE role = ?", [role]) if role else ("", [])
        rows = self.store.query(f"SELECT * FROM model_stats {clause} ORDER BY role, provider, model", parameters)
        result = []
        for row in rows:
            item = dict(row)
            item["wilson"] = wilson_lower_bound(row["successes"], row["runs"])
            item["tokens_per_second"] = row["tokens"] / row["seconds"] if row["seconds"] else 0.0
            result.append(item)
        return result


def attach(assistant: Any) -> None:
    assistant.router = ModelRouter(assistant)


def add_commands(commands: Any) -> None:
    models = commands.add_parser("models", help="show how each model has really done, per role")
    models.add_argument("--role", default=None, choices=ROLES, help="only this role")
    models.set_defaults(run=plainly(run_models))
    route = commands.add_parser("route", help="choose which models serve each role")
    actions = route.add_subparsers(dest="route_command", required=True)
    change = actions.add_parser("set", help="set a role's candidates, as PROVIDER or PROVIDER/MODEL")
    change.add_argument("role", choices=ROLES, help="the role")
    change.add_argument("candidates", nargs="+", help="its candidates; on equal results the first listed wins")
    change.set_defaults(run=plainly(run_route_set))
    show = actions.add_parser("show", help="show each role's candidates")
    show.set_defaults(run=plainly(run_route_show))


def run_models(assistant: Any, args: argparse.Namespace) -> int:
    rows = [
        {
            "role": row["role"],
            "model": f"{row['provider']}/{row['model']}",
            "runs": row["runs"],
            "ok": row["successes"],
            "at least": f"{row['wilson']:.2f}",
            "tokens": row["tokens"],
            "tok/s": f"{row['tokens_per_second']:.1f}",
        }
        for row in assistant.router.stats(args.role)
    ]
    emit(table(rows, ["role", "model", "runs", "ok", "at least", "tokens", "tok/s"]) or "No runs recorded yet.")
    return 0


def run_route_set(assistant: Any, args: argparse.Namespace) -> int:
    assistant.router.set_route(args.role, args.candidates)
    emit(f"{args.role}: {', '.join(args.candidates)}")
    return 0


def run_route_show(assistant: Any, args: argparse.Namespace) -> int:
    for role, candidates in assistant.router.routes().items():
        emit(f"{role:<14}{', '.join(candidates) or 'nothing set'}")
    return 0
