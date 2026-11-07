"""isharded bench: numbers about a running mesh that others can compare.

It asks a node the way any client does, through /v1/chat/completions and
/admin/status, and changes nothing. People will post what it writes, so a report
holds only the fields picked out here: no addresses, keys, or machine paths.
Two saved reports can be set side by side with --compare.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import secrets
import statistics
import sys
import time
from typing import Any, Callable

import httpx

from . import __version__, source_commit
from .pool import bearer_headers, load_token
from .tls import client_verify
from .up import local_url, mesh_client, read_settings, sign_in_key, up_home

LOCAL_NODE = "http://127.0.0.1:8000"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
READ_SECONDS = 900.0  # a long prompt behind seven others can stay silent for minutes
SIZING_TRIES = 5
SIZING_FIRST_TOKENS_PER_UNIT = 64  # above any text here, so the first try cannot overrun a size that fits
CACHED_REPLY_TOKENS = 8  # only the first token is being timed
LONGEST_ERROR = 200
REPORT_FORMAT = 2

PROSE = (
    "The ferry left the harbour an hour late because the tide had turned early.",
    "A thin rain settled over the market while the traders packed away their stalls.",
    "Nobody in the village could say who had first planted the row of pear trees.",
    "The surveyor measured the valley twice and wrote down two different answers.",
    "By the third winter the old mill had stopped turning and the river froze around it.",
    "She kept every letter in a biscuit tin under the stairs and never read them again.",
    "The train slowed at the viaduct so that the passengers could see the lake below.",
    "Grain prices fell that autumn, and half the farms on the ridge changed hands.",
    "He repaired clocks for a living but was late to almost everything that mattered.",
    "Fog rolled down from the moor and swallowed the road before anyone reached the inn.",
    "The librarian stamped the book, looked over her glasses, and said nothing at all.",
    "Two brothers argued for forty years about where the boundary stone had once stood.",
    "At low water you could walk out to the wreck and touch the ribs of its hull.",
    "The choir practised in the cold church because the hall roof had finally come down.",
    "A fox crossed the frozen yard at dawn and left a line of prints across the snow.",
    "When the mine closed, the town kept its brass band and little else besides.",
)
CODE = (
    "def clamp(value, low, high):\n    return max(low, min(value, high))",
    "def chunks(items, size):\n    return [items[start:start + size] for start in range(0, len(items), size)]",
    "class Counter:\n    def __init__(self):\n        self.seen = {}\n\n    def add(self, name):\n        self.seen[name] = self.seen.get(name, 0) + 1",
    "def read_rows(path):\n    with open(path, encoding='utf-8') as handle:\n        return [line.rstrip().split(',') for line in handle]",
    "def median(values):\n    ordered = sorted(values)\n    middle = len(ordered) // 2\n    return ordered[middle]",
    "def retry(call, times=3):\n    for attempt in range(times):\n        try:\n            return call()\n        except OSError:\n            continue\n    raise TimeoutError('gave up')",
    "def flatten(groups):\n    return [item for group in groups for item in group]",
    "def invert(table):\n    return {value: key for key, value in table.items()}",
    "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()",
    "def is_prime(number):\n    return number > 1 and all(number % factor for factor in range(2, int(number ** 0.5) + 1))",
)
# A reply that copied its prompt would flatter a draft, so each one asks for new text.
REQUESTS = {
    "prose": "Write a long new story that picks up details from the notes above. Do not stop early.",
    "code": "Write a long new Python module in the style of the code above, with many more functions. Do not stop early.",
    "counting": "Keep counting upward from the last number, one number per line, with no comment. Do not stop.",
}
QUANTIZATION = re.compile(r"(?<![A-Za-z0-9])(I?Q\d(?:_[A-Za-z0-9]+)*|BF16|FP16|F16|FP32|F32|FP8)(?![A-Za-z0-9])", re.IGNORECASE)
NOT_FOR_POSTING = (
    re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE),
    re.compile(r"\bbearer\s+\S+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]:[\\/]\S*"),
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"),
    re.compile(r"\[[0-9A-Fa-f:]*:[0-9A-Fa-f:]*\](?::\d+)?"),
    re.compile(r"(?<![\w.])/(?:[\w.-]+/)+[\w.-]*"),
    re.compile(r"(?:\\\\)?[\w.-]+(?:\\[\w.-]+)+"),  # also what is left of a path with a space in it
    re.compile(r"\b[\w-]+(?:\.[\w-]+)*:\d{2,5}\b"),
    re.compile(r"(?<![\w:])[0-9A-Fa-f]*:[0-9A-Fa-f]*:[0-9A-Fa-f:]*"),
)


class BenchError(RuntimeError):
    """Why there is nothing to measure."""


def machine_names(status: dict[str, Any]) -> re.Pattern[str] | None:
    """The mesh's own names. Its errors say which worker failed, and an id is often a machine's name."""
    workers = status.get("workers") or {}
    mesh = status.get("mesh") or {}
    names = {*workers, *(mesh.get("members") or {}), mesh.get("node"), mesh.get("leader")}
    names |= {((worker or {}).get("hardware") or {}).get("host_id") for worker in workers.values()}
    found = sorted(filter(None, names), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, found)) + r")(?![\w-])", re.IGNORECASE) if found else None


def clean(text: str, names: re.Pattern[str] | None = None) -> str:
    """A server's words, safe to post. Errors are the one free text in a report."""
    if names:
        text = names.sub("[hidden]", text)
    for pattern in NOT_FOR_POSTING:
        text = pattern.sub("[hidden]", text)
    return text[:LONGEST_ERROR]


def public_name(model: str | None) -> str | None:
    """A model's name without the folders it sits in on someone's machine."""
    if not model:
        return None
    if re.fullmatch(r"[\w.-]+/[\w.-]+:[\w./-]+", model) or re.fullmatch(r"[\w.-]+/[\w.-]+(?<!\.gguf)", model):
        return model  # a hub name, which is the same for everyone
    return re.split(r"[\\/]", model.rstrip("\\/"))[-1]


def quantization(name: str | None) -> str | None:
    found = QUANTIZATION.findall(name or "")
    return found[-1].upper() if found else None


def positive(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number above zero")
    return number


def comma_list(text: str) -> list[int]:
    return [positive(part) for part in text.split(",")]


def build_prompt(text: str, units: int, rng: random.Random) -> str:
    """Fixed text in a varied order, behind a prefix unique to this prompt.

    The prefix comes from outside the seed. A second run with the same seed must
    not find its prompts in the cache the first run left behind.
    """
    if text == "counting":
        start = rng.randrange(1, 1000)
        body = [str(number) for number in range(start, start + units)]
    else:
        pool = list(PROSE if text == "prose" else CODE)
        body = []
        while len(body) < units:
            rng.shuffle(pool)
            body += pool
        body = body[:units]
    return f"Run {secrets.token_hex(4)}.\n\n" + "\n".join(body) + f"\n\n{REQUESTS[text]}"


@dataclass
class Reply:
    first_token_seconds: float | None = None
    last_token_seconds: float | None = None
    prompt_tokens: int = 0
    tokens: int = 0
    text: str = ""
    error: str | None = None

    @property
    def tokens_per_second(self) -> float | None:
        """Decoding alone: the first token belongs to reading the prompt."""
        if self.first_token_seconds is None or self.last_token_seconds is None or self.tokens < 2:
            return None
        span = self.last_token_seconds - self.first_token_seconds
        return (self.tokens - 1) / span if span > 0 else None


def summary(values: list[float | None], digits: int = 1) -> dict[str, Any] | None:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return {
        "median": round(statistics.median(numbers), digits),
        "min": round(min(numbers), digits),
        "max": round(max(numbers), digits),
        "samples": len(numbers),
    }


def milliseconds(replies: list[Reply]) -> dict[str, Any] | None:
    return summary([None if reply.first_token_seconds is None else reply.first_token_seconds * 1000 for reply in replies], 0)


def level_result(parallel: int, rounds: list[tuple[list[Reply], float]], wanted_tokens: int) -> dict[str, Any]:
    replies = [reply for group, _ in rounds for reply in group]
    good = [reply for reply in replies if not reply.error]
    return {
        "parallel": parallel,
        "first_token_ms": milliseconds(good),
        "reply_tokens_per_second": summary([reply.tokens_per_second for reply in good]),
        "total_tokens_per_second": summary([sum(reply.tokens for reply in group if not reply.error) / wall if wall > 0 else None for group, wall in rounds]),
        "prompt_tokens": round(statistics.median(reply.prompt_tokens for reply in good)) if good else None,
        "reply_tokens": round(statistics.median(reply.tokens for reply in good)) if good else None,
        "short_replies": sum(reply.tokens < wanted_tokens for reply in good),
        "failed": len(replies) - len(good),
        "errors": sorted({reply.error for reply in replies if reply.error}),
    }


def telemetry_of(status: dict[str, Any], worker_id: str) -> dict[str, Any]:
    worker = (status.get("workers") or {}).get(worker_id) or {}
    return (worker.get("runtime") or {}).get("telemetry") or {}


def route_ids(status: dict[str, Any]) -> list[str]:
    return [stage["worker_id"] for stage in status.get("route") or []]


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def stage_precision(hardware: dict[str, Any]) -> str | None:
    """What a stage computes in, from its fastest pool's setting. A disk pool has none of its own."""
    kinds = sorted({pool.get("dtype") for pool in hardware.get("memory_pools") or [] if pool.get("dtype") not in (None, "checkpoint")})
    return ", ".join(kinds) or None


def route_facts(status: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per stage, numbered. A worker's id is often its machine's name."""
    rows = []
    route = status.get("route") or []
    probed = {(link.get("source"), link.get("target")): link for link in status.get("links") or []}
    for index, stage in enumerate(route, 1):
        worker = (status.get("workers") or {}).get(stage["worker_id"]) or {}
        hardware = worker.get("hardware") or {}
        telemetry = telemetry_of(status, stage["worker_id"])
        round_trip = (telemetry.get("ring") or {}).get("link_rtt_us")
        following = route[index % len(route)]["worker_id"]
        rows.append({
            "stage": index,
            "start_layer": stage.get("start_layer"),
            "end_layer": stage.get("end_layer"),
            "country": worker.get("country"),
            "gpu": hardware.get("accelerator_name") or hardware.get("device"),
            "precision": stage_precision(hardware),
            "link_ms": round(round_trip / 1000, 2) if round_trip else None,
            "probe_ms": rounded((probed.get((stage["worker_id"], following)) or {}).get("median_ms")),
            "next_stage_ms": rounded(telemetry.get("downstream_ms_ema")),
            "compute_ms": rounded(telemetry.get("compute_ms_ema")),
            "control_ms": worker.get("control_latency_ms"),
        })  # fmt: skip
    return rows


def decoding_facts(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """What this run did. The counters run from when each worker started."""
    made = misses = depth = passes = 0
    per_pass = None
    for worker_id in route_ids(after):
        old, new = telemetry_of(before, worker_id), telemetry_of(after, worker_id)
        old_pipe, new_pipe = old.get("pipeline") or {}, new.get("pipeline") or {}
        made += max(new_pipe.get("tokens", 0) - old_pipe.get("tokens", 0), 0)
        misses += max(new_pipe.get("misses", 0) - old_pipe.get("misses", 0), 0)
        depth = max(depth, new_pipe.get("depth", 0))
        passes += max(new.get("draft_passes", 0) - old.get("draft_passes", 0), 0)
        per_pass = new.get("draft_tokens_per_pass") or per_pass
    return {"pipeline_tokens": made, "pipeline_misses": misses, "pipeline_depth": depth, "draft_passes": passes, "draft_tokens_per_pass": per_pass}


def native_engine(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Whether this run's tokens went over the engine's direct links.

    Only the entry stage says its ring is live, and a live ring proves nothing
    until passes have crossed it.
    """
    ids = route_ids(after)
    if not ids:
        return False

    def passes(status: dict[str, Any]) -> int:
        return (telemetry_of(status, ids[0]).get("ring") or {}).get("passes", 0)

    return bool(telemetry_of(after, ids[0]).get("ring_live")) and passes(after) > passes(before)


def machine_count(status: dict[str, Any]) -> int:
    """How many machines the route spans. Stages on one host count once, and no name leaves here."""
    workers = status.get("workers") or {}
    hosts = {((workers.get(worker_id) or {}).get("hardware") or {}).get("host_id") or worker_id for worker_id in route_ids(status)}
    return len(hosts)


class Bench:
    def __init__(self, client: httpx.Client, url: str, headers: dict[str, str], options: argparse.Namespace, clock: Callable[[], float] = time.perf_counter) -> None:
        self.client, self.url, self.headers, self.options, self.clock = client, url, headers, options, clock
        self.rng = random.Random(options.seed)
        self.names: re.Pattern[str] | None = None

    def status(self) -> dict[str, Any]:
        try:
            answer = self.client.get(self.url + "/admin/status", headers=self.headers)
        except httpx.HTTPError as exc:
            raise BenchError(f"no answer from the node: {clean(str(exc)) or type(exc).__name__}") from exc
        if answer.status_code in (401, 403):
            raise BenchError("the node wants a key; give one with --key")
        if answer.status_code >= 400:
            raise BenchError(f"the node refused: HTTP {answer.status_code}")
        try:
            status = answer.json()
        except ValueError:
            status = None
        if not isinstance(status, dict):
            raise BenchError("that address did not answer like an iSharded node")
        return status

    def ask(self, prompt: str, tokens: int) -> Reply:
        """One streamed reply, timed here, where a person would feel it."""
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": self.options.temperature,
            "seed": self.options.seed,
            "enable_thinking": False,
            "stream": True,
        }
        reply = Reply()
        pieces = []
        began = self.clock()
        try:
            with self.client.stream("POST", self.url + "/v1/chat/completions", json=body, headers=self.headers) as answer:
                if answer.status_code >= 400:
                    reply.error = refusal(answer.status_code, answer.read(), self.names)
                    return reply
                for line in answer.iter_lines():
                    if not line.startswith("data: {"):
                        continue
                    data = json.loads(line[6:])
                    if "error" in data:
                        reply.error = clean(str(data["error"].get("message")), self.names)
                        break
                    choice = (data.get("choices") or [{}])[0]
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        reply.last_token_seconds = self.clock() - began
                        if reply.first_token_seconds is None:
                            reply.first_token_seconds = reply.last_token_seconds
                        pieces.append(piece)
                    if data.get("usage"):
                        reply.prompt_tokens = data["usage"]["prompt_tokens"]
                        reply.tokens = data["usage"]["completion_tokens"]
        except (httpx.HTTPError, ValueError, KeyError, AttributeError, TypeError) as exc:
            reply.error = clean(str(exc), self.names) or type(exc).__name__
        reply.text = "".join(pieces)
        if not reply.error and not reply.prompt_tokens:
            reply.error = "the reply ended without its usage line"
        return reply

    def fit(self, wanted: int) -> tuple[int | None, str | None]:
        """How much text comes to the wanted prompt size on this model's tokenizer.

        Only the node can say, and it has no route for counting, so this asks for
        one token at a time and reads what the prompt came to.
        """
        units = max(wanted // SIZING_FIRST_TOKENS_PER_UNIT, 1)
        best, best_gap = None, None
        for _ in range(SIZING_TRIES):
            reply = self.ask(build_prompt(self.options.text, units, self.rng), 1)
            if reply.error:
                return None, reply.error  # a smaller prompt must not pass for this size
            gap = abs(reply.prompt_tokens - wanted)
            if best_gap is None or gap < best_gap:
                best, best_gap = units, gap
            if gap <= max(16, wanted // 50):
                break
            units = max(round(units * wanted / reply.prompt_tokens), 1)
        return best, None

    def round_of(self, prompts: list[str]) -> tuple[list[Reply], float]:
        with ThreadPoolExecutor(len(prompts)) as pool:
            began = self.clock()
            replies = list(pool.map(lambda prompt: self.ask(prompt, self.options.tokens), prompts))
            return replies, self.clock() - began

    def measure(self, wanted: int) -> dict[str, Any]:
        options = self.options
        units, problem = self.fit(wanted)
        if units is None:
            return {"wanted_prompt_tokens": wanted, "skipped": problem}
        print(f"Measuring prompts of about {wanted} tokens...", file=sys.stderr, flush=True)

        def prompts(count: int) -> list[str]:
            return [build_prompt(options.text, units, self.rng) for _ in range(count)]

        for prompt in prompts(options.warmup):
            self.ask(prompt, options.tokens)
        cached: list[Reply] = []
        levels = []
        # One at a time always runs, as the numbers for a single reply come from it.
        for parallel in sorted({1, *options.parallel}):
            rounds = []
            for _ in range(options.repeat):
                batch = prompts(parallel)
                rounds.append(self.round_of(batch))
                if options.cached and parallel == 1:
                    cached.append(self.ask(batch[0], CACHED_REPLY_TOKENS))
            levels.append(level_result(parallel, rounds, options.tokens))
        result = {"wanted_prompt_tokens": wanted, "units": units, "levels": levels}
        if options.cached:
            result["cached_first_token_ms"] = milliseconds([reply for reply in cached if not reply.error])
            result["cached_failed"] = sum(1 for reply in cached if reply.error)
        return result

    def repeat_check(self, units: int) -> dict[str, Any]:
        """Whether the same seeded request gives the same reply twice."""
        prompt = build_prompt(self.options.text, units, self.rng)
        first, second = (self.ask(prompt, self.options.tokens) for _ in range(2))
        problem = first.error or second.error
        if problem:
            return {"identical": None, "error": problem}
        if not first.text:
            return {"identical": None, "error": "the reply was empty"}
        return {"identical": first.text == second.text, "reply_tokens": first.tokens}

    def run(self) -> dict[str, Any]:
        options = self.options
        before = self.status()
        if before.get("status") != "ready":
            raise BenchError(f"no model is ready here; the node says {clean(str(before.get('status')))}")
        self.names = machine_names(before)
        results = [self.measure(wanted) for wanted in options.prompt_tokens]
        lost = None
        try:
            after = self.status()
        except BenchError as exc:  # what was measured is still true, and a run can take an hour
            after, lost = before, str(exc)
        measured = [result for result in results if "levels" in result]
        repeat = self.repeat_check(measured[0]["units"]) if measured else None
        for result in measured:
            del result["units"]  # only meaningful inside this run
        report = build_report(options, before, after, results, repeat)
        if lost:
            report["notes"].append(f"The node's status could not be read after the run, so the native engine and decoding lines mean nothing here: {lost}")
        return report


def refusal(status_code: int, body: bytes, names: re.Pattern[str] | None = None) -> str:
    if status_code == 429:
        return "the node's rate limit refused this reply"  # its words name the key
    try:
        detail = json.loads(body).get("detail")
    except (ValueError, AttributeError):
        detail = None
    return f"HTTP {status_code}: {clean(str(detail), names)}" if detail else f"HTTP {status_code}"


def build_report(options: argparse.Namespace, before: dict[str, Any], after: dict[str, Any], results: list[dict[str, Any]], repeat: dict[str, Any] | None) -> dict[str, Any]:
    plan = after.get("active_plan") or {}
    name = public_name(after.get("model"))
    own_url = httpx.URL(options.url or LOCAL_NODE).host in LOOPBACK
    ids = route_ids(after)
    versions = sorted({(after.get("workers") or {}).get(worker_id, {}).get("code_version") or "unknown" for worker_id in ids})
    old_cache = (before.get("metrics") or {}).get("cache") or {}
    new_cache = (after.get("metrics") or {}).get("cache") or {}
    reused = max(new_cache.get("prompt_tokens_reused", 0) - old_cache.get("prompt_tokens_reused", 0), 0)
    notes = []
    if ids != route_ids(before) or after.get("epoch") != before.get("epoch"):
        notes.append("The route moved during this run, so these numbers mix two routes.")
    if reused and not options.cached:
        notes.append(f"{reused} prompt tokens came from the cache, though nothing should have.")
    if options.cached and not reused:
        notes.append("No prompt tokens came from the cache, so the cached first token times show no saving. Is the node keeping prompt states?")
    for result in results:
        if "skipped" in result:
            notes.append(f"Prompts of {result['wanted_prompt_tokens']} tokens were skipped: {result['skipped']}")
        for level in result.get("levels", []):
            for error in level["errors"]:
                notes.append(f"{level['failed']} replies failed at {result['wanted_prompt_tokens']} tokens, {level['parallel']} at once: {error}")
            if level["short_replies"]:
                notes.append(f"{level['short_replies']} replies at {result['wanted_prompt_tokens']} tokens, {level['parallel']} at once, ended before --tokens.")
        if result.get("cached_failed"):
            notes.append(f"{result['cached_failed']} cached replies failed at {result['wanted_prompt_tokens']} tokens.")
    if repeat and repeat.get("error"):
        notes.append(f"The repeat of a seeded reply could not be checked: {repeat['error']}")
    return {
        "format": REPORT_FORMAT,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": ", ".join(versions) if ids else __version__,
        "bench_version": __version__,
        "commit": after.get("commit"),
        "bench_commit": source_commit(),
        "model": {
            "name": name,
            "quantization": plan.get("quantization") or quantization(name),
            "architecture": plan.get("architecture"),
            "layers": plan.get("layers"),
            "draft_model": public_name(plan.get("draft_model")),
            "slots": plan.get("slots"),
            "context_tokens": plan.get("context_tokens"),
        },
        "machines": machine_count(after),
        "asked_from": "on the node's own machine" if own_url else "over the network",
        "route": route_facts(after),
        "native_engine": native_engine(before, after),
        "decoding": decoding_facts(before, after),
        "prompt_cache": {"tokens_reused": reused},
        "options": {
            "tokens": options.tokens,
            "repeat": options.repeat,
            "warmup": options.warmup,
            "seed": options.seed,
            "temperature": options.temperature,
            "text": options.text,
            "cached": options.cached,
        },
        "results": results,
        "repeat_check": repeat,
        "notes": notes,
    }


def spread(value: dict[str, Any] | None) -> str:
    return f"{value['median']:g} ({value['min']:g}-{value['max']:g})" if value else "-"


def table(report: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Prompt tokens", "Replies at once", "First token ms", "Tokens/s per reply", "Total tokens/s"]
    cached = report["options"]["cached"]
    if cached:
        headers.insert(3, "Cached first token ms")
    rows = []
    for result in report["results"]:
        for level in result.get("levels", []):
            row = [
                str(level["prompt_tokens"] or result["wanted_prompt_tokens"]),
                str(level["parallel"]),
                spread(level["first_token_ms"]),
                spread(level["reply_tokens_per_second"]),
                spread(level["total_tokens_per_second"]),
            ]
            if cached:
                row.insert(3, spread(result.get("cached_first_token_ms")) if level["parallel"] == 1 else "")
            rows.append(row)
    return headers, rows


def stage_line(stage: dict[str, Any]) -> str:
    where = stage["country"] or "country not shown"
    link = f", link {stage['link_ms']} ms" if stage["link_ms"] else ""
    probe = f", probe {stage['probe_ms']} ms" if stage.get("probe_ms") else ""
    precision = f", {stage['precision']}" if stage.get("precision") else ""
    return f"{stage['stage']}: layers {stage['start_layer']}-{stage['end_layer'] - 1}, {stage['gpu']}{precision}, {where}{link}{probe}"


def code_line(report: dict[str, Any]) -> str:
    commit, bench_commit = report.get("commit"), report.get("bench_commit")
    return report["version"] + (f", node at commit {commit}" if commit else ", node reported no commit") + (f", bench at commit {bench_commit}" if bench_commit else "")


def facts(report: dict[str, Any]) -> list[tuple[str, str]]:
    """What explains the numbers, in words."""
    model, decoding, repeat = report["model"], report["decoding"], report["repeat_check"]
    stages = [stage_line(stage) for stage in report["route"]]
    way = f"pipeline {decoding['pipeline_tokens']} tokens, {decoding['pipeline_misses']} misses, depth {decoding['pipeline_depth']}"
    if decoding["draft_passes"]:
        way += f"; draft {decoding['draft_passes']} passes, {decoding['draft_tokens_per_pass']} tokens per pass since the stage started"
    options = report["options"]
    same = "not checked" if repeat is None else "failed to run" if repeat["identical"] is None else "yes" if repeat["identical"] else "NO"
    rows = [
        ("Model", model["name"] or "unknown"),
        ("Quantization", model["quantization"] or "not known"),
        ("Slots", str(model.get("slots") or "not reported")),
        ("Machines", str(report.get("machines") or "not known")),
        ("Stages", "; ".join(stages) or "none"),
        ("Native engine", "on" if report["native_engine"] else "off"),
        ("Decoding", way),
        ("Reply length", f"{options['tokens']} tokens of {options['text']}, temperature {options['temperature']:g}, seed {options['seed']}"),
        ("Repeat of a seeded reply identical", same),
        ("iSharded", code_line(report)),
    ]
    if model["draft_model"]:
        rows.insert(2, ("Draft model", model["draft_model"]))
    return rows


def method(report: dict[str, Any]) -> list[str]:
    """How the numbers were taken, so someone else can say whether theirs compare."""
    options = report["options"]
    rounds = f"{options['repeat']} rounds" if options["repeat"] != 1 else "1 round"
    return [
        f"Measured {report.get('asked_from', 'from an unknown place')} at {report['at']}, as any client would feel it: streamed replies from /v1/chat/completions, timed where they arrive.",
        "First token is the time from sending a request to its first text. Tokens/s per reply leaves that out: tokens after the first, over the time from the first text to the last. Total tokens/s is every token of a round, over the whole of that round.",
        f"Each case ran {rounds}. Replies thrown away first, for each prompt size: {options['warmup']}. Every prompt starts differently, so none is read from the cache unless the table says cached.",
        "A link is the direct link's own round trip to the next stage, where one was up. A probe is the same trip over HTTP, measured when the route was planned. Each number is the median, with the lowest and highest in brackets.",
    ]


def as_text(report: dict[str, Any]) -> str:
    lines = [f"{label}: {value}" for label, value in facts(report)] + [""]
    headers, rows = table(report)
    lines += columns(headers, rows)
    lines += ["", *method(report)] if rows else []
    lines += report["notes"]
    return "\n".join(lines)


def columns(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[column]) for row in [headers, *rows]) for column in range(len(headers))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in [headers, *rows]]


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(row) + " |" for row in [headers, ["---"] * len(headers), *rows]]


def as_markdown(report: dict[str, Any]) -> str:
    title = report["model"]["name"] or "unknown model"
    headers, rows = table(report)
    lines = [f"## iSharded bench: {title}", ""] + [f"- **{label}:** {value}" for label, value in facts(report)] + [""]
    lines += markdown_table(headers, rows)
    lines += ["", "### How this was measured", ""] + [f"{sentence}\n" for sentence in method(report)]
    lines += [f"- {note}" for note in report["notes"]]
    return "\n".join(lines).rstrip() + "\n"


# What must match for two runs' numbers to mean the same thing, as (label, how to read it from a report).
SETUP: tuple[tuple[str, Callable[[dict[str, Any]], Any]], ...] = (
    ("Model", lambda report: report["model"]["name"]),
    ("Quantization", lambda report: report["model"]["quantization"]),
    ("Draft model", lambda report: report["model"]["draft_model"]),
    ("Slots", lambda report: report["model"].get("slots")),
    ("Machines", lambda report: report.get("machines")),
    ("Stages", lambda report: [(stage["start_layer"], stage["end_layer"], stage["gpu"], stage.get("precision")) for stage in report["route"]]),
    ("Native engine", lambda report: report["native_engine"]),
    ("Reply tokens", lambda report: report["options"]["tokens"]),
    ("Temperature", lambda report: report["options"]["temperature"]),
    ("Text", lambda report: report["options"]["text"]),
    ("Asked from", lambda report: report.get("asked_from")),
)
# Label, the level's field, and whether more is better.
MEASURES = (("First token ms", "first_token_ms", False), ("Tokens/s per reply", "reply_tokens_per_second", True), ("Total tokens/s", "total_tokens_per_second", True))


def setup_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(f"layers {start}-{end - 1} on {gpu}" + (f" in {precision}" if precision else "") for start, end, gpu, precision in value) or "none"
    return "not reported" if value is None else str(value)


def change(old: dict[str, Any] | None, new: dict[str, Any] | None, higher_is_better: bool) -> dict[str, Any] | None:
    """How one measure moved. Ranges that overlap are noise until a longer run says otherwise."""
    if not old or not new:
        return None
    percent = round((new["median"] - old["median"]) / old["median"] * 100, 1) if old["median"] else None
    overlap = old["min"] <= new["max"] and new["min"] <= old["max"] or min(old.get("samples", 2), new.get("samples", 2)) < 2  # one sample has no range
    better = new["median"] > old["median"] if higher_is_better else new["median"] < old["median"]
    verdict = "within noise" if overlap or new["median"] == old["median"] else "better" if better else "worse"
    return {"old": old["median"], "new": new["median"], "percent": percent, "verdict": verdict}


def levels_of(report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(result["wanted_prompt_tokens"], level["parallel"]): level for result in report["results"] for level in result.get("levels", [])}


def compare_reports(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """What differs between two saved runs: first what ran, then the numbers."""
    setup = [{"what": label, "old": setup_text(read(old)), "new": setup_text(read(new))} for label, read in SETUP]
    old_levels, new_levels = levels_of(old), levels_of(new)
    rows = []
    for key in sorted(old_levels.keys() & new_levels.keys()):
        changes = {field: change(old_levels[key][field], new_levels[key][field], better) for _, field, better in MEASURES}
        rows.append({"prompt_tokens": key[0], "parallel": key[1], **changes})
    notes = [f"Prompts of {size} tokens, {parallel} at once, ran in only one of these." for size, parallel in sorted(old_levels.keys() ^ new_levels.keys())]
    if any(item["old"] != item["new"] for item in setup):
        notes.insert(0, "These runs differ in what ran, so a change here may come from that and not from the code.")
    return {
        "format": REPORT_FORMAT,
        "old": {"at": old["at"], "commit": old.get("commit"), "version": old["version"]},
        "new": {"at": new["at"], "commit": new.get("commit"), "version": new["version"]},
        "setup": setup,
        "results": rows,
        "notes": notes,
    }


def run_name(run: dict[str, Any]) -> str:
    return f"{run['version']}, {run['commit'] or 'no commit reported'}, {run['at']}"


def moved(item: dict[str, Any] | None) -> str:
    if not item:
        return "-"
    percent = "" if item["percent"] is None else f" ({item['percent']:+g}%)"
    return f"{item['old']:g} -> {item['new']:g}{percent} {item['verdict']}"


def compared_rows(comparison: dict[str, Any]) -> tuple[list[str], list[list[str]], list[list[str]]]:
    headers = ["Prompt tokens", "Replies at once", *(label for label, _, _ in MEASURES)]
    rows = [[str(row["prompt_tokens"]), str(row["parallel"]), *(moved(row[field]) for _, field, _ in MEASURES)] for row in comparison["results"]]
    differs = [[item["what"], item["old"], item["new"]] for item in comparison["setup"] if item["old"] != item["new"]]
    return headers, rows, differs


COMPARE_RULE = "Each cell is the old median, then the new one. A change counts only if the two runs' lowest-to-highest ranges do not overlap, and a single sample has no range."


def comparison_text(comparison: dict[str, Any]) -> str:
    headers, rows, differs = compared_rows(comparison)
    lines = [f"Old: {run_name(comparison['old'])}", f"New: {run_name(comparison['new'])}", ""]
    lines += ["What ran differently:", *columns(["", "Old", "New"], differs), ""] if differs else ["The same setup ran both times.", ""]
    lines += [*columns(headers, rows), "", COMPARE_RULE] if rows else ["The two runs have no case in common."]
    return "\n".join(lines + comparison["notes"])


def comparison_markdown(comparison: dict[str, Any]) -> str:
    headers, rows, differs = compared_rows(comparison)
    lines = ["## iSharded bench comparison", "", f"- **Old:** {run_name(comparison['old'])}", f"- **New:** {run_name(comparison['new'])}", ""]
    lines += ["What ran differently:", "", *markdown_table(["", "Old", "New"], differs), ""] if differs else ["The same setup ran both times.", ""]
    lines += [*markdown_table(headers, rows), "", COMPARE_RULE, ""] if rows else ["The two runs have no case in common.", ""]
    return "\n".join(lines + [f"- {note}" for note in comparison["notes"]]).rstrip() + "\n"


def saved_report(name: str) -> dict[str, Any]:
    try:
        report = json.loads(Path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchError(f"cannot read {name}: {type(exc).__name__}") from exc
    if not isinstance(report, dict) or not {"format", "results", "route", "options", "model"} <= report.keys():
        raise BenchError(f"{name} is not a report from isharded bench --json")
    return report


def connect(args: argparse.Namespace) -> tuple[str, dict[str, str], httpx.Client]:
    """Where to ask, and with what. A saved key goes only to this machine's own node."""
    home = up_home()
    settings = read_settings(home)
    url = (args.url or (local_url(settings) if settings else LOCAL_NODE)).rstrip("/")
    key = args.key
    try:
        own_node = httpx.URL(url).host in LOOPBACK
    except httpx.InvalidURL as exc:
        raise BenchError(f"that is not an address: {clean(str(exc))}") from exc
    if not key and own_node:
        key = sign_in_key(home) or load_token()
    verify = str(home / "ca.pem") if own_node and settings and settings.get("tls") and url.startswith("https:") else client_verify()
    return url, bearer_headers(key), mesh_client(verify, READ_SECONDS)


def went_wrong(report: dict[str, Any]) -> bool:
    """Whether a reply failed, or nothing was measured. A size with no room is the load's limit, not a failure."""
    results = report["results"]
    levels = [level for item in results for level in item.get("levels", [])]
    failed_sizes = [item for item in results if "skipped" in item and not item["skipped"].startswith("HTTP 400")]
    repeat = report["repeat_check"] or {}
    return not levels or any(level["failed"] for level in levels) or bool(failed_sizes) or any(item.get("cached_failed") for item in results) or "error" in repeat


def run_compare(args: argparse.Namespace) -> int:
    try:
        comparison = compare_reports(*map(saved_report, args.compare))
    except (BenchError, KeyError, TypeError, AttributeError) as exc:
        print(f"Cannot compare these: {exc}", file=sys.stderr)
        return 1
    print(comparison_text(comparison))
    return write_files(args, comparison, comparison_markdown(comparison))


def run_bench(args: argparse.Namespace) -> int:
    if args.compare:
        return run_compare(args)
    try:
        url, headers, client = connect(args)
        with client:
            result = Bench(client, url, headers, args).run()
    except BenchError as exc:
        print(f"Cannot run the bench: {exc}", file=sys.stderr)
        return 1
    print(as_text(result))
    return write_files(args, result, as_markdown(result)) or (1 if went_wrong(result) else 0)


def write_files(args: argparse.Namespace, result: dict[str, Any], markdown: str) -> int:
    code = 0
    for name, text in ((args.json_file, json.dumps(result, indent=2) + "\n"), (args.markdown_file, markdown)):
        try:
            if name:
                Path(name).write_text(text, encoding="utf-8")
        except OSError as exc:  # the table is already on screen, and a run can take an hour
            print(f"Could not write {name}: {exc}", file=sys.stderr)
            code = 1
    return code


def add_arguments(subparsers: Any) -> None:
    bench = subparsers.add_parser("bench", help="measure a running mesh, with numbers others can compare")
    bench.add_argument("--prompt-tokens", type=comma_list, default=[128, 3400], help="prompt sizes to try, as a comma list")
    bench.add_argument("--tokens", type=positive, default=200, help="how many tokens each reply runs to")
    bench.add_argument("--parallel", type=comma_list, default=[1, 4, 8], help="replies at once to try, as a comma list")
    bench.add_argument("--repeat", type=positive, default=3, help="how many times to measure each case")
    bench.add_argument("--warmup", type=int, default=1, help="replies to throw away before each prompt size")
    bench.add_argument("--seed", type=int, default=1, help="seed for the prompts and the replies")
    bench.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    bench.add_argument("--text", choices=sorted(REQUESTS), default="prose", help="what the prompts are made of")
    bench.add_argument("--cached", action="store_true", help="also send each prompt a second time, to time its first token from the cache")
    bench.add_argument("--json", dest="json_file", default=None, help="also write the results to this file as JSON")
    bench.add_argument("--markdown", dest="markdown_file", default=None, help="also write the results to this file as Markdown")
    bench.add_argument("--url", default=None, help="any node's address; default is this machine's node")
    bench.add_argument("--key", default=None, help="an API key; default is this machine's own, for its own node only")
    bench.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"), default=None, help="set two runs saved with --json side by side, and measure nothing")
    bench.set_defaults(handler=run_bench)
