"""isharded bench against a fake node: its clock, and what it dares to print."""

from __future__ import annotations

import json
from pathlib import Path
import random
import threading
import time

import httpx
import pytest

from isharded import bench, cli

SECRET_KEY = "sk-do-not-print"
MACHINE = "desk-of-somebody"
MODEL_PATH = r"C:\Users\somebody\models\Qwen3-8B-Q4_K_M.gguf"
DRAFT_PATH = "/home/somebody/models/Qwen3-0.6B-Q8_0.gguf"
ADDRESS = "http://203.0.113.9:8301"
PREFILL_SECONDS = 0.25
TOKEN_SECONDS = 0.01
CACHED_PREFILL_SECONDS = 0.05
OPENING_TOKENS = 5  # what a chat template adds around a message


def status(pipeline_tokens: int = 100, epoch: int = 3, reused: int = 0, ready: str = "ready") -> dict:
    def worker(gpu: str, country: str | None, tokens: int, rtt: int, entry: bool) -> dict:
        return {
            "advertised_url": ADDRESS,
            "code_version": "0.3.0",
            "country": country,
            "control_latency_ms": 2.5,
            "hardware": {"host_id": MACHINE, "device": "cuda:0", "accelerator_name": gpu, "memory_pools": [{"label": MACHINE, "dtype": "bfloat16"}, {"label": MACHINE, "dtype": "checkpoint"}]},
            "runtime": {
                "telemetry": {
                    "pipeline": {"tokens": tokens, "misses": tokens // 10, "depth": 3},
                    "draft_passes": tokens // 2,
                    "draft_tokens_per_pass": 2.5,
                    "compute_ms_ema": 11.0,
                    "downstream_ms_ema": 22.0,
                    "ring": {"link_rtt_us": rtt, "passes": pipeline_tokens},
                    "ring_live": entry,  # as a real route reports it: only the entry stage says so
                    "last_error": f"lost {ADDRESS}",
                }
            },
        }

    return {
        "status": ready,
        "commit": "0123456789ab",
        "model": MODEL_PATH,
        "epoch": epoch,
        "active_plan": {"architecture": "qwen3", "layers": 36, "draft_model": DRAFT_PATH, "slots": 4, "quantization": "Q4_K_M", "context_tokens": 8192},
        "links": [{"source": f"{MACHINE}-1", "target": f"{MACHINE}-2", "median_ms": 3.25, "max_ms": 4.0, "cost_ms": 3.6, "checked_at": 1}],
        "route": [
            {"worker_id": f"{MACHINE}-1", "url": ADDRESS, "start_layer": 0, "end_layer": 20},
            {"worker_id": f"{MACHINE}-2", "url": ADDRESS, "start_layer": 20, "end_layer": 36},
        ],
        "workers": {f"{MACHINE}-1": worker("RTX 3060", "DE", pipeline_tokens, 1500, True), f"{MACHINE}-2": worker("RX 7900 XT", None, 0, 0, False)},
        "metrics": {"cache": {"prompt_tokens_reused": reused}},
        "events": [{"title": "Direct links up", "detail": ADDRESS}],
    }


class Clock:
    """Time passes only when the fake node says so, so timings come out exact."""

    def __init__(self) -> None:
        self.now = 100.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.now += seconds


class FakeNode:
    def __init__(self) -> None:
        self.clock = Clock()
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self.seen: set[str] = set()
        self.statuses = [status(100), status(350, reused=0)]
        self.status_code = 200
        self.context = 100_000
        self.tokens_per_word = 1
        self.reply_tokens: int | None = None
        self.text_of = lambda body, count: ["word "] * count
        self.fail: dict[str, str] = {}
        self.hold_for = 0
        self.in_flight = 0
        self.peak = 0
        self.lock = threading.Lock()

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/admin/status":
            if self.status_code != 200:
                return httpx.Response(self.status_code, json={"detail": "no"})
            answer = self.statuses[0] if len(self.statuses) == 1 else self.statuses.pop(0)
            return answer if isinstance(answer, httpx.Response) else httpx.Response(200, json=answer)
        body = json.loads(request.content)
        self.bodies.append(body)
        prompt = body["messages"][0]["content"]
        prompt_tokens = len(prompt.split()) * self.tokens_per_word + OPENING_TOKENS
        if prompt_tokens > self.context:
            return httpx.Response(400, json={"detail": f"the prompt is {prompt_tokens} tokens, and this model was loaded with room for {self.context}"})
        if "big" in self.fail and prompt_tokens > 1000:
            return httpx.Response(503, json={"detail": self.fail["big"]})
        if "cached" in self.fail and body["max_tokens"] == bench.CACHED_REPLY_TOKENS:
            return httpx.Response(503, json={"detail": self.fail["cached"]})
        if "repeat" in self.fail and body["max_tokens"] > 1 and prompt in self.seen:
            return httpx.Response(503, json={"detail": self.fail["repeat"]})
        if body["max_tokens"] > 1:
            if "http" in self.fail:
                return httpx.Response(503, json={"detail": self.fail["http"]})
            self.hold(body)
        with self.lock:
            cached = prompt in self.seen
            self.seen.add(prompt)
        return httpx.Response(200, content=self.stream(body, prompt_tokens, cached))

    def hold(self, body: dict) -> None:
        with self.lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        deadline = time.monotonic() + 0.3
        while self.hold_for and self.in_flight < self.hold_for and time.monotonic() < deadline:
            time.sleep(0.001)
        time.sleep(0.01 if self.hold_for else 0)
        with self.lock:
            self.in_flight -= 1

    def stream(self, body: dict, prompt_tokens: int, cached: bool):
        count = min(self.reply_tokens or body["max_tokens"], body["max_tokens"])
        self.clock.advance(CACHED_PREFILL_SECONDS if cached else PREFILL_SECONDS)
        for index, piece in enumerate(self.text_of(body, count)):
            if index:
                self.clock.advance(TOKEN_SECONDS)
            if "stream" in self.fail and body["max_tokens"] > 1:
                yield f'data: {json.dumps({"error": {"message": self.fail["stream"], "code": 503}})}\n\n'.encode()
                return
            if "plain" in self.fail and body["max_tokens"] > 1:
                yield f'data: {json.dumps({"error": self.fail["plain"]})}\n\n'.encode()
                return
            if "cut" in self.fail and body["max_tokens"] > 1:
                return
            yield f'data: {json.dumps({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})}\n\n'.encode()
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": count, "total_tokens": prompt_tokens + count}
        yield f'data: {json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}], "usage": usage})}\n\n'.encode()
        yield b"data: [DONE]\n\n"


def options(*extra: str):
    return cli.build_parser().parse_args(["bench", "--prompt-tokens", "200", "--tokens", "11", "--parallel", "1", "--repeat", "2", "--warmup", "0", *extra])


def run(node: FakeNode, *extra: str) -> dict:
    return bench.Bench(node.client(), "http://node", {}, options(*extra), node.clock).run()


def measured(node: FakeNode) -> list[dict]:
    """The replies that count, leaving out sizing, which asks for a single token."""
    return [body for body in node.bodies if body["max_tokens"] > 1]


def test_defaults_are_the_published_ones():
    args = cli.build_parser().parse_args(["bench"])
    assert (args.prompt_tokens, args.tokens, args.parallel, args.repeat, args.warmup) == ([128, 3400], 200, [1, 4, 8], 3, 1)
    assert (args.temperature, args.text, args.cached, args.url, args.key, args.json_file, args.markdown_file) == (0.0, "prose", False, None, None, None, None)


@pytest.mark.parametrize("arguments", [["--parallel", "1,x"], ["--parallel", "0"], ["--prompt-tokens", ""], ["--repeat", "0"], ["--text", "poetry"]])
def test_bad_options_are_refused(arguments, capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["bench", *arguments])


@pytest.mark.parametrize(
    "name, public, quantized",
    [
        (r"I:\models\small\Qwen3-0.6B-Q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf", "Q8_0"),
        ("/home/me/models/gemma-IQ4_XS.gguf", "gemma-IQ4_XS.gguf", "IQ4_XS"),
        ("models/Qwen3.6-27B-UD-Q4_K_XL.gguf", "Qwen3.6-27B-UD-Q4_K_XL.gguf", "Q4_K_XL"),
        ("ggml-org/Qwen3.8-27B-GGUF:Qwen3.8-27B-Q4_K_M.gguf", "ggml-org/Qwen3.8-27B-GGUF:Qwen3.8-27B-Q4_K_M.gguf", "Q4_K_M"),
        ("distilbert/distilgpt2", "distilbert/distilgpt2", None),
        ("Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B", None),
        ("some/Model-bf16", "some/Model-bf16", "BF16"),
    ],
)
def test_names_lose_their_folders_and_give_the_quantization(name, public, quantized):
    assert bench.public_name(name) == public
    assert bench.quantization(public) == quantized


def test_timing_comes_from_the_stream_not_the_chunks():
    level = run(FakeNode())["results"][0]["levels"][0]
    assert level["first_token_ms"] == {"median": 250, "min": 250, "max": 250, "samples": 2}
    assert level["reply_tokens_per_second"]["median"] == pytest.approx(100.0)
    # Eleven tokens in the whole reply, prompt reading included.
    assert level["total_tokens_per_second"]["median"] == pytest.approx(11 / (PREFILL_SECONDS + 10 * TOKEN_SECONDS), abs=0.1)
    assert level["reply_tokens"] == 11


def test_a_token_count_comes_from_usage_when_chunks_carry_several():
    node = FakeNode()
    node.text_of = lambda body, count: ["several words "] * (count // 2)  # a draft lands tokens in bursts
    level = run(node)["results"][0]["levels"][0]
    assert level["reply_tokens"] == 11
    assert level["reply_tokens_per_second"]["median"] == pytest.approx(10 / (4 * TOKEN_SECONDS))


def test_spread_shows_the_lowest_and_highest():
    assert bench.summary([3.0, 1.0, None, 2.0]) == {"median": 2.0, "min": 1.0, "max": 3.0, "samples": 3}
    assert bench.spread({"median": 2.0, "min": 1.0, "max": 3.5, "samples": 3}) == "2 (1-3.5)"
    assert bench.summary([None]) is None and bench.spread(None) == "-"


def test_parallel_replies_really_run_together_and_add_up():
    node = FakeNode()
    node.hold_for = 3
    result = run(node, "--parallel", "3", "--repeat", "1")["results"][0]
    assert [level["parallel"] for level in result["levels"]] == [1, 3]
    assert node.peak == 3
    together = result["levels"][1]
    # This clock adds the three replies' time up, so the expected rate is one reply's.
    assert together["total_tokens_per_second"]["median"] == pytest.approx(11 / (PREFILL_SECONDS + 10 * TOKEN_SECONDS), abs=0.1)
    assert together["total_tokens_per_second"]["samples"] == 1
    assert together["first_token_ms"]["samples"] == 3


def test_a_case_runs_as_often_as_asked_with_warmups_thrown_away():
    node = FakeNode()
    result = run(node, "--parallel", "2", "--repeat", "3", "--warmup", "2")["results"][0]
    assert [level["first_token_ms"]["samples"] for level in result["levels"]] == [3, 6]
    assert len(measured(node)) == 2 + 3 + 6 + 2  # warmups, one at a time, two at a time, the repeat check


def test_nothing_is_served_from_the_cache_unless_asked():
    node = FakeNode()
    result = run(node, "--parallel", "1,3", "--warmup", "1")
    timed = [body["messages"][0]["content"] for body in measured(node)][:-2]  # the repeat check repeats on purpose
    assert len(set(timed)) == len(timed)
    # Replies at once share this clock, so one can read longer, but only a cached one reads shorter.
    assert all(level["first_token_ms"]["min"] >= 250 for level in result["results"][0]["levels"])
    assert "cached_first_token_ms" not in result["results"][0]


def test_the_same_seed_does_not_find_the_last_runs_prompts():
    first, second = FakeNode(), FakeNode()
    run(first)
    run(second)
    assert not {body["messages"][0]["content"] for body in first.bodies} & {body["messages"][0]["content"] for body in second.bodies}


@pytest.mark.parametrize("text", ["prose", "code", "counting"])
def test_prompts_are_fixed_text_behind_a_varied_prefix(text):
    prompts = [bench.build_prompt(text, 40, random.Random(7)) for _ in range(2)]
    first, second = (prompt.split("\n", 1) for prompt in prompts)
    assert first[0] != second[0]
    assert first[1] == second[1]
    assert bench.REQUESTS[text] in prompts[0]


def test_cached_sends_each_prompt_again_and_times_it():
    node = FakeNode()
    result = run(node, "--cached")
    entry = result["results"][0]
    assert entry["cached_first_token_ms"] == {"median": 50, "min": 50, "max": 50, "samples": 2}
    assert entry["levels"][0]["first_token_ms"]["median"] == 250
    asks = [body["messages"][0]["content"] for body in measured(node)][:-2]
    assert asks[0] == asks[1] and asks[2] == asks[3] and asks[0] != asks[2]
    assert "Cached first token ms" in bench.as_text(result)


def test_a_repeat_of_a_seeded_reply_is_compared():
    node = FakeNode()
    assert run(node)["repeat_check"] == {"identical": True, "reply_tokens": 11}
    assert measured(node)[-1] == measured(node)[-2]
    assert measured(node)[-1]["seed"] == 1 and measured(node)[-1]["temperature"] == 0.0


def test_a_repeat_that_differs_is_said_plainly():
    node = FakeNode()
    replies = iter(range(1000))
    node.text_of = lambda body, count: [f"{next(replies)} "] * count
    result = run(node)
    assert result["repeat_check"]["identical"] is False
    assert "Repeat of a seeded reply identical: NO" in bench.as_text(result)


def test_seed_and_temperature_reach_the_node():
    node = FakeNode()
    run(node, "--seed", "42", "--temperature", "0.7")
    assert all(body["seed"] == 42 and body["temperature"] == 0.7 and body["stream"] for body in node.bodies)
    assert all(body["enable_thinking"] is False for body in node.bodies)


def test_prompts_are_sized_on_the_models_own_count():
    node = FakeNode()
    level = run(node, "--prompt-tokens", "3400")["results"][0]["levels"][0]
    assert abs(level["prompt_tokens"] - 3400) <= 68


@pytest.mark.parametrize("wanted", [1, 3400])
def test_sizing_ends_even_when_it_cannot_hit_the_size(wanted):
    node = FakeNode()
    assert run(node, "--prompt-tokens", str(wanted), "--text", "counting")["results"][0]["levels"]


def test_a_first_guess_too_big_does_not_cost_a_size_that_fits():
    node = FakeNode()
    node.context, node.tokens_per_word = 4096, 3  # code runs to about 30 tokens a line
    level = run(node, "--prompt-tokens", "3400", "--text", "code")["results"][0]["levels"][0]
    assert abs(level["prompt_tokens"] - 3400) <= 68


def test_a_size_the_model_has_no_room_for_is_skipped_and_said():
    node = FakeNode()
    node.context = 1000
    result = run(node, "--prompt-tokens", "200,3400")
    assert [("levels" in item) for item in result["results"]] == [True, False]
    assert "loaded with room for 1000" in result["results"][1]["skipped"]
    assert any("skipped" in note for note in result["notes"])


def test_a_short_reply_is_noted():
    node = FakeNode()
    node.reply_tokens = 5
    result = run(node)
    assert result["results"][0]["levels"][0]["short_replies"] == 2
    assert any("ended before --tokens" in note for note in result["notes"])


def test_the_report_explains_the_numbers():
    node = FakeNode()
    result = run(node)
    assert result["model"] == {"name": "Qwen3-8B-Q4_K_M.gguf", "quantization": "Q4_K_M", "architecture": "qwen3", "layers": 36, "draft_model": "Qwen3-0.6B-Q8_0.gguf", "slots": 4, "context_tokens": 8192}
    assert result["route"][0] == {
        "stage": 1, "start_layer": 0, "end_layer": 20, "country": "DE", "gpu": "RTX 3060", "precision": "bfloat16",
        "link_ms": 1.5, "probe_ms": 3.25, "next_stage_ms": 22.0, "compute_ms": 11.0, "control_ms": 2.5,
        "cuda_graphs": None, "price_per_hour": None,
        "trace": {"link_rtt_us": 1500, "keeps": 0, "keep_bytes": None, "keep_ms_per_copy": None, "keep_share_of_compute": None},
    }  # fmt: skip
    assert result["route"][1]["gpu"] == "RX 7900 XT" and result["route"][1]["link_ms"] is None
    assert result["native_engine"] is True
    assert result["version"] == "0.3.0"
    # This run's share, since these counters run from when a worker started.
    assert result["decoding"] == {"pipeline_tokens": 250, "pipeline_misses": 25, "pipeline_depth": 3, "draft_passes": 125, "draft_tokens_per_pass": 2.5, "draft_guess": None}
    assert not result["notes"]


def test_the_native_engine_is_off_when_the_entry_has_no_ring():
    node = FakeNode()
    node.statuses[1]["workers"][f"{MACHINE}-1"]["runtime"]["telemetry"]["ring_live"] = False
    assert run(node)["native_engine"] is False


def test_a_live_ring_no_pass_crossed_does_not_count():
    node = FakeNode()
    node.statuses = [status(100), status(100)]
    assert run(node)["native_engine"] is False


def test_layers_are_shown_as_people_count_them():
    assert "1: layers 0-19, RTX 3060, bfloat16, DE, link 1.5 ms, probe 3.25 ms; 2: layers 20-35, RX 7900 XT, bfloat16, country not shown" in bench.as_text(run(FakeNode()))


def test_a_route_that_moved_is_noted():
    node = FakeNode()
    node.statuses = [status(100, epoch=3), status(100, epoch=4)]
    assert any("route moved" in note for note in run(node)["notes"])


def test_cache_use_nobody_asked_for_is_noted():
    node = FakeNode()
    node.statuses = [status(reused=10), status(reused=500)]
    result = run(node)
    assert result["prompt_cache"] == {"tokens_reused": 490}
    assert any("490 prompt tokens came from the cache" in note for note in result["notes"])
    node = FakeNode()
    node.statuses = [status(reused=10), status(reused=500)]
    assert not any("cache" in note for note in run(node, "--cached")["notes"])


def test_cached_times_with_no_cache_behind_them_are_noted():
    assert any("No prompt tokens came from the cache" in note for note in run(FakeNode(), "--cached")["notes"])


def test_it_only_reads_status_and_asks_for_replies():
    node = FakeNode()
    run(node, "--cached")
    assert {(request.method, request.url.path) for request in node.requests} == {("GET", "/admin/status"), ("POST", "/v1/chat/completions")}


def everything_written(result: dict) -> str:
    return "\n".join((json.dumps(result), bench.as_text(result), bench.as_markdown(result)))


@pytest.mark.parametrize("failure", ["http", "stream"])
def test_nothing_private_reaches_a_report(failure):
    node = FakeNode()
    node.fail[failure] = f"stage at {ADDRESS} failed, see C:\\Users\\somebody\\logs\\w1.log or /home/somebody/w1.log, Bearer {SECRET_KEY} from 203.0.113.9:8301 and [2001:db8::1]:80"
    written = everything_written(run(node))
    assert "failed" in written
    for private in (MACHINE, ADDRESS, "203.0.113", "2001:db8", "somebody", SECRET_KEY, "w1.log"):
        assert private not in written


def test_nothing_private_reaches_a_clean_report_either():
    written = everything_written(run(FakeNode()))
    for private in (MACHINE, ADDRESS, "203.0.113", "somebody", "Users"):
        assert private not in written


@pytest.mark.parametrize("failure", ["http", "stream"])
def test_a_machine_named_in_a_failure_stays_out_of_a_report(failure):
    node = FakeNode()
    node.fail[failure] = f"the entry worker could not answer: {{'failed_worker': '{MACHINE}-1', 'message': 'lost'}} on {MACHINE}"
    written = everything_written(run(node))
    assert "could not answer" in written and MACHINE not in written


@pytest.mark.parametrize(
    "private",
    [r"C:\Users\John Smith\logs\w1.log", r"\\fileserver\share\w1.log", "db-node.example.org:8301", "2001:db8::1"],
)
def test_addresses_and_paths_without_a_clean_edge_are_hidden(private):
    shown = bench.clean(f"lost {private} just now")
    assert shown.startswith("lost ") and shown.endswith(" just now")
    for part in ("Smith", "w1.log", "fileserver", "example.org", "db8"):
        assert part not in shown


def test_a_rate_limit_does_not_name_the_key():
    assert bench.refusal(429, b'{"detail": "rate limit reached for somebody"}') == "the node's rate limit refused this reply"


@pytest.mark.parametrize("failure, words", [("http", "HTTP 503: busy"), ("stream", "busy"), ("cut", "without its usage line")])
def test_failed_replies_are_counted_and_not_timed(failure, words):
    node = FakeNode()
    node.fail[failure] = "busy"
    result = run(node)
    level = result["results"][0]["levels"][0]
    assert level["failed"] == 2 and level["first_token_ms"] is None
    assert words in level["errors"][0]
    assert any("2 replies failed" in note for note in result["notes"])
    assert "error" in result["repeat_check"]
    assert "failed to run" in bench.as_text(result)


def run_command(monkeypatch, node: FakeNode, *extra: str) -> int:
    monkeypatch.setattr(bench, "connect", lambda args: ("http://node", {}, node.client()))
    return bench.run_bench(options(*extra))


def test_the_command_prints_a_table_and_writes_both_files(monkeypatch, tmp_path, capsys):
    node = FakeNode()
    data, page = tmp_path / "out.json", tmp_path / "out.md"
    assert run_command(monkeypatch, node, "--json", str(data), "--markdown", str(page)) == 0
    shown = capsys.readouterr().out
    assert "Tokens/s per reply" in shown and "Model: Qwen3-8B-Q4_K_M.gguf" in shown and "|" not in shown
    saved = json.loads(data.read_text(encoding="utf-8"))
    assert saved["format"] == 3 and saved["options"]["tokens"] == 11
    lines = page.read_text(encoding="utf-8").splitlines()
    assert "- **Quantization:** Q4_K_M" in lines
    table = [line for line in lines if line.startswith("|")]
    assert len(table) == 3 and table[1].startswith("| --- |")
    assert {line.count("|") for line in table} == {6}


def test_a_report_says_what_ran():
    report = run(FakeNode())
    assert (report["model"]["slots"], report["model"]["quantization"], report["model"]["context_tokens"]) == (4, "Q4_K_M", 8192)
    assert report["commit"] == "0123456789ab" and report["machines"] == 1
    assert report["route"][0]["probe_ms"] == 3.25 and report["route"][1]["probe_ms"] is None
    assert report["route"][0]["precision"] == "bfloat16"
    shown = dict(bench.facts(report))
    assert shown["Slots"] == "4" and shown["Machines"] == "1" and "node at commit 0123456789ab" in shown["iSharded"]
    assert "probe 3.25 ms" in shown["Stages"] and "bfloat16" in shown["Stages"]


def test_status_gives_what_a_report_needs(monkeypatch):
    from types import SimpleNamespace

    from isharded import orchestrator
    from isharded.models import GgufAdapter
    from isharded.protocol import StageAssignment

    monkeypatch.setattr(orchestrator, "source_commit", lambda: "0123456789ab")
    assert orchestrator.Orchestrator().status()["commit"] == "0123456789ab"
    adapter = object.__new__(GgufAdapter)
    adapter.header = {"general.file_type": 15}
    adapter.config, adapter.num_layers = {"model_type": "qwen3"}, 2
    target = SimpleNamespace(
        model_id="m", adapter=adapter, slots=4, parallel={}, draft=None, embedding=False, projector=None, units={}, held={},
        profile=SimpleNamespace(total_bytes=1, ring=False, context_tokens=8192),
    )  # fmt: skip
    route = [StageAssignment(worker_id="w", url="http://worker", start_layer=0, end_layer=2)]
    plan = orchestrator.describe_plan(target, route)
    assert (plan["slots"], plan["quantization"], plan["context_tokens"]) == (4, "Q4_K_M", 8192)


def test_a_checkout_knows_its_commit_and_an_install_does_not(monkeypatch, tmp_path):
    import isharded

    isharded.source_commit.cache_clear()
    assert isharded.source_commit() is None or len(isharded.source_commit()) == 12
    monkeypatch.setattr(isharded, "__file__", str(tmp_path / "isharded" / "__init__.py"))
    isharded.source_commit.cache_clear()
    assert isharded.source_commit() is None
    isharded.source_commit.cache_clear()


def test_the_quantization_a_node_reports_outranks_the_file_name():
    node = FakeNode()
    node.statuses[1]["active_plan"]["quantization"] = "Q6_K"
    assert run(node)["model"]["quantization"] == "Q6_K"


def test_a_node_too_old_to_say_is_reported_as_silent():
    node = FakeNode()
    del node.statuses[1]["commit"]
    for name in ("slots", "quantization", "context_tokens"):
        del node.statuses[1]["active_plan"][name]
    report = run(node)
    assert report["model"]["quantization"] == "Q4_K_M"  # still in the file's name
    shown = dict(bench.facts(report))
    assert shown["Slots"] == "not reported" and "node reported no commit" in shown["iSharded"]


def test_machines_are_counted_without_being_named():
    status_now = status()
    status_now["workers"][f"{MACHINE}-2"]["hardware"]["host_id"] = "another-desk"
    assert bench.machine_count(status_now) == 2


def test_a_report_says_how_it_was_measured():
    report = run(FakeNode())
    page = bench.as_markdown(report)
    assert page.startswith("## iSharded bench: Qwen3-8B-Q4_K_M.gguf\n") and "### How this was measured" in page
    assert "/v1/chat/completions" in page and "2 rounds" in page and report["at"] in page
    assert "/v1/chat/completions" in bench.as_text(report)


def test_where_it_was_asked_from_is_said_and_no_address_with_it():
    assert run(FakeNode())["asked_from"] == "on the node's own machine"
    node = FakeNode()
    remote = bench.Bench(node.client(), "http://node", {}, options("--url", ADDRESS), node.clock).run()
    assert remote["asked_from"] == "over the network" and "203.0.113" not in json.dumps(remote)


def saved_run(tmp_path, name: str, **changes) -> str:
    report = run(FakeNode())
    report |= {"commit": name * 12}
    for level, scale in changes.get("scale", {}).items():
        for result in report["results"]:
            for row in result["levels"]:
                row[level] = {key: value * scale if key != "samples" else value for key, value in row[level].items()}
    report["model"] |= changes.get("model", {})
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def compare(*arguments: str) -> int:
    return bench.run_bench(cli.build_parser().parse_args(["bench", "--compare", *arguments]))


def test_compare_sets_two_runs_side_by_side(tmp_path, capsys):
    old = saved_run(tmp_path, "a")
    new = saved_run(tmp_path, "b", scale={"reply_tokens_per_second": 2.0, "first_token_ms": 2.0})
    assert compare(old, new) == 0
    shown = capsys.readouterr().out
    assert "aaaaaaaaaaaa" in shown and "bbbbbbbbbbbb" in shown and "The same setup ran both times." in shown
    assert "+100%) better" in shown and "+100%) worse" in shown


def test_compare_calls_overlapping_ranges_noise():
    wide = {"median": 10.0, "min": 8.0, "max": 12.0, "samples": 3}
    near = {"median": 11.0, "min": 9.0, "max": 13.0, "samples": 3}
    assert bench.change(wide, near, True) == {"old": 10.0, "new": 11.0, "percent": 10.0, "verdict": "within noise"}
    far = {"median": 20.0, "min": 19.0, "max": 21.0, "samples": 3}
    assert bench.change(wide, far, True)["verdict"] == "better" and bench.change(wide, far, False)["verdict"] == "worse"
    assert bench.change(None, far, True) is None and bench.change(wide, None, True) is None
    alone = {"median": 20.0, "min": 20.0, "max": 20.0, "samples": 1}
    assert bench.change(wide, alone, True)["verdict"] == "within noise" and bench.change(alone, wide, True)["verdict"] == "within noise"


def test_compare_warns_when_the_runs_measured_different_things(tmp_path, capsys):
    assert compare(saved_run(tmp_path, "a"), saved_run(tmp_path, "b", model={"quantization": "Q8_0", "slots": 8})) == 0
    shown = capsys.readouterr().out
    assert "What ran differently:" in shown and "Q4_K_M" in shown and "Q8_0" in shown
    assert "may come from that and not from the code" in shown


def test_compare_says_what_only_one_run_measured(tmp_path, capsys):
    old, new = saved_run(tmp_path, "a"), tmp_path / "b.json"
    report = json.loads(Path(old).read_text(encoding="utf-8"))
    report["results"][0]["wanted_prompt_tokens"] = 999
    new.write_text(json.dumps(report), encoding="utf-8")
    assert compare(old, str(new)) == 0
    shown = capsys.readouterr().out
    assert "no case in common" in shown and "Prompts of 999 tokens, 1 at once, ran in only one of these." in shown


def test_compare_reads_a_report_from_before_nodes_said_what_ran(tmp_path, capsys):
    old = tmp_path / "old.json"
    report = json.loads(Path(saved_run(tmp_path, "a")).read_text(encoding="utf-8"))
    report["format"] = 1
    for name in ("machines", "asked_from", "bench_commit"):
        del report[name]
    del report["model"]["slots"]
    for stage in report["route"]:
        del stage["precision"], stage["probe_ms"]
    old.write_text(json.dumps(report), encoding="utf-8")
    assert compare(str(old), saved_run(tmp_path, "b")) == 0
    assert "not reported" in capsys.readouterr().out


def test_compare_writes_both_files_and_measures_nothing(tmp_path, capsys):
    data, page = tmp_path / "out.json", tmp_path / "out.md"
    assert compare(saved_run(tmp_path, "a"), saved_run(tmp_path, "b"), "--json", str(data), "--markdown", str(page)) == 0
    assert json.loads(data.read_text(encoding="utf-8"))["new"]["commit"] == "b" * 12
    text = page.read_text(encoding="utf-8")
    assert text.startswith("## iSharded bench comparison\n") and "| --- |" in text


@pytest.mark.parametrize("content, words", [(None, "cannot read"), ("not json", "cannot read"), ('{"a": 1}', "not a report"), ("[1]", "not a report")])
def test_compare_refuses_what_is_not_a_report(tmp_path, capsys, content, words):
    bad = tmp_path / "bad.json"
    if content is not None:
        bad.write_text(content, encoding="utf-8")
    assert compare(str(bad), saved_run(tmp_path, "a")) == 1
    assert words in capsys.readouterr().err


def test_a_file_it_cannot_write_does_not_cost_the_results(monkeypatch, tmp_path, capsys):
    gone = tmp_path / "no-folder" / "out.json"
    assert run_command(monkeypatch, FakeNode(), "--json", str(gone)) == 1
    shown = capsys.readouterr()
    assert "Tokens/s per reply" in shown.out and "Could not write" in shown.err


def test_the_command_fails_when_a_reply_failed(monkeypatch, capsys):
    node = FakeNode()
    node.fail["stream"] = "busy"
    assert run_command(monkeypatch, node) == 1


def test_the_command_fails_when_nothing_could_be_measured(monkeypatch, capsys):
    node = FakeNode()
    node.context = 10
    assert run_command(monkeypatch, node) == 1


@pytest.mark.parametrize(
    "code, ready, words",
    [(401, "ready", "--key"), (403, "ready", "--key"), (500, "ready", "HTTP 500"), (200, "loading", "no model is ready")],
)
def test_a_node_that_cannot_be_measured_says_why(monkeypatch, capsys, code, ready, words):
    node = FakeNode()
    node.status_code = code
    node.statuses = [status(ready=ready)]
    assert run_command(monkeypatch, node) == 1
    assert words in capsys.readouterr().err
    assert not node.bodies


def test_a_size_that_failed_is_a_failure_though_one_with_no_room_is_not(monkeypatch, capsys):
    node = FakeNode()
    node.fail["big"] = "stage lost"
    assert run_command(monkeypatch, node, "--prompt-tokens", "200,3400") == 1
    node = FakeNode()
    node.context = 1000
    assert run_command(monkeypatch, node, "--prompt-tokens", "200,3400") == 0


def test_a_failed_cached_reply_fails_the_command_and_is_noted(monkeypatch, capsys):
    node = FakeNode()
    node.fail["cached"] = "busy"
    assert run_command(monkeypatch, node, "--cached") == 1
    assert "cached replies failed" in capsys.readouterr().out


def test_a_repeat_check_that_could_not_run_fails_the_command_and_says_why(monkeypatch, capsys):
    node = FakeNode()
    node.fail["repeat"] = "stage lost"
    assert run_command(monkeypatch, node) == 1
    assert "stage lost" in capsys.readouterr().out


def test_empty_replies_prove_nothing_about_repeating(monkeypatch, capsys):
    node = FakeNode()
    node.text_of = lambda body, count: []
    assert run_command(monkeypatch, node) == 1
    assert "identical: failed to run" in capsys.readouterr().out


def test_a_plain_error_in_the_stream_is_a_failed_reply():
    node = FakeNode()
    node.fail["plain"] = "busy"
    assert run(node)["results"][0]["levels"][0]["failed"] == 2


def test_losing_the_node_at_the_end_keeps_what_was_measured(monkeypatch, capsys):
    node = FakeNode()
    node.statuses = [status(100), httpx.Response(500, json={"detail": "no"})]
    run_command(monkeypatch, node)
    shown = capsys.readouterr().out
    assert "Tokens/s per reply" in shown and "could not be read after the run" in shown


def test_something_that_is_not_a_node_says_so(monkeypatch, capsys):
    node = FakeNode()
    node.statuses = [httpx.Response(200, text="<html>hello</html>")]
    assert run_command(monkeypatch, node) == 1
    assert "did not answer like" in capsys.readouterr().err


def test_a_bad_address_says_so(up_folder, capsys):
    arguments = cli.build_parser().parse_args(["bench", "--url", "http://:abc"])
    assert bench.run_bench(arguments) == 1
    assert "Cannot run the bench" in capsys.readouterr().err


def test_a_node_that_does_not_answer_says_so(monkeypatch, capsys):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {ADDRESS}")

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    monkeypatch.setattr(bench, "connect", lambda args: ("http://node", {}, client))
    assert bench.run_bench(options()) == 1
    shown = capsys.readouterr().err
    assert "no answer from the node" in shown and ADDRESS not in shown


@pytest.fixture
def up_folder(monkeypatch, tmp_path) -> Path:
    home = tmp_path / "up"
    home.mkdir()
    monkeypatch.setenv("ISHARDED_UP_HOME", str(home))
    monkeypatch.setenv("ISHARDED_POOL_TOKEN_FILE", str(tmp_path / "no-token"))
    monkeypatch.delenv("ISHARDED_POOL_TOKEN", raising=False)
    monkeypatch.delenv("ISHARDED_CA_FILE", raising=False)
    return home


def connect(*extra: str) -> tuple[str, dict[str, str]]:
    url, headers, client = bench.connect(cli.build_parser().parse_args(["bench", *extra]))
    client.close()
    return url, headers


def test_it_finds_this_machines_node_and_key(up_folder):
    (up_folder / "settings.json").write_text(json.dumps({"port": 8209}), encoding="utf-8")
    (up_folder / "user.key").write_text("saved-key\n", encoding="utf-8")
    assert connect() == ("http://127.0.0.1:8209", {"Authorization": "Bearer saved-key"})


def test_without_an_up_folder_it_asks_the_usual_port(up_folder):
    assert connect() == ("http://127.0.0.1:8000", {})


def test_the_pool_token_stands_in_for_a_saved_key(up_folder, monkeypatch):
    monkeypatch.setenv("ISHARDED_POOL_TOKEN", "pool-token")
    assert connect()[1] == {"Authorization": "Bearer pool-token"}


def test_a_key_given_wins(up_folder):
    (up_folder / "admin.key").write_text("saved-key", encoding="utf-8")
    assert connect("--key", "given")[1] == {"Authorization": "Bearer given"}


def test_a_saved_key_never_leaves_this_machine(up_folder, monkeypatch):
    monkeypatch.setenv("ISHARDED_POOL_TOKEN", "pool-token")
    (up_folder / "admin.key").write_text("saved-key", encoding="utf-8")
    assert connect("--url", "http://somebody-elses-mesh:8209/") == ("http://somebody-elses-mesh:8209", {})
    assert connect("--url", "http://somebody-elses-mesh:8209", "--key", "given")[1] == {"Authorization": "Bearer given"}
    assert connect("--url", "http://localhost:8209")[1] == {"Authorization": "Bearer saved-key"}


def test_a_tls_node_is_checked_against_the_meshs_authority(up_folder, monkeypatch):
    (up_folder / "settings.json").write_text(json.dumps({"port": 8209, "tls": True}), encoding="utf-8")
    seen = {}
    monkeypatch.setattr(bench, "mesh_client", lambda verify, timeout: seen.update(verify=verify, timeout=timeout) or httpx.Client())
    assert connect()[0] == "https://127.0.0.1:8209"
    assert seen == {"verify": str(up_folder / "ca.pem"), "timeout": bench.READ_SECONDS}


def test_this_meshs_authority_is_not_forced_on_a_node_elsewhere(up_folder, monkeypatch):
    (up_folder / "settings.json").write_text(json.dumps({"port": 8209, "tls": True}), encoding="utf-8")
    seen = {}
    monkeypatch.setattr(bench, "mesh_client", lambda verify, timeout: seen.update(verify=verify) or httpx.Client())
    connect("--url", "https://somebody-elses-mesh:8209")
    assert seen == {"verify": True}


def test_the_guide_covers_the_command():
    guide = (Path(__file__).resolve().parents[1] / "docs" / "GUIDE.md").read_text(encoding="utf-8")
    assert "### isharded bench" in guide


def with_telemetry(base: dict, worker: str, **fields) -> dict:
    """A status in which one worker's telemetry says more."""
    copied = json.loads(json.dumps(base))
    copied["workers"][worker]["runtime"]["telemetry"].update(fields)
    return copied


ENTRY, EXIT = f"{MACHINE}-1", f"{MACHINE}-2"
# Run one level of one round, so the clock gives exact walls: 0.25 s to the first token and
# 0.01 s a token after it, 11 tokens.
ONE_ROUND = ("--repeat", "1")
ROUND_SECONDS = PREFILL_SECONDS + 10 * TOKEN_SECONDS


def test_the_hop_trace_and_this_runs_kept_state_come_from_each_stages_engine():
    node = FakeNode()
    ring = {"compute_us": 9000, "queue_us": 12, "write_us": 30, "enqueue_us": 8, "wait_us": 5, "hop_us": 800, "link_rtt_us": 700, "keep_us": 4100, "keep_bytes": 52 * 2**20}
    before = with_telemetry(status(100), EXIT, ring={**ring, "keeps_total": 50, "keep_us_total": 200_000, "compute_us_total": 1_000_000})
    after = with_telemetry(status(350), EXIT, ring={**ring, "keeps_total": 150, "keep_us_total": 600_000, "compute_us_total": 2_000_000})
    node.statuses = [before, after]
    report = run(node)
    trace = report["route"][1]["trace"]
    assert {name: trace[name] for name in ("compute_us", "queue_us", "write_us", "enqueue_us", "wait_us", "hop_us", "keep_us")} == {
        "compute_us": 9000, "queue_us": 12, "write_us": 30, "enqueue_us": 8, "wait_us": 5, "hop_us": 800, "keep_us": 4100,
    }  # fmt: skip
    # 100 copies this run took 400 ms of its 1000 ms of compute.
    assert (trace["keeps"], trace["keep_ms_per_copy"], trace["keep_share_of_compute"], trace["keep_bytes"]) == (100, 4.0, 0.4, 52 * 2**20)
    shown = dict(bench.facts(report))
    assert shown["Kept state"] == "stage 2: 100 copies of 52.0 MiB, 4 ms each, 40% of compute"
    assert "2: compute 9000, keep 4100, queue 12, write 30, enqueue 8, wait 5, hop 800, link_rtt 700" in shown["Stage times, us"]


def test_the_drafts_real_guess_time_is_this_runs_share():
    node = FakeNode()
    guess = {"guesses": 10, "ms_total": 100.0, "steady": 8, "steady_ms_total": 60.0}
    later = {"guesses": 110, "ms_total": 1300.0, "steady": 88, "steady_ms_total": 860.0}
    node.statuses = [with_telemetry(status(100), ENTRY, draft_guess=guess), with_telemetry(status(350), ENTRY, draft_guess=later)]
    report = run(node, "--pace")
    # 100 guesses in 1200 ms; 80 steady in 800 ms; 20 after a miss in 400 ms.
    assert report["decoding"]["draft_guess"] == {"guesses": 100, "ms_per_guess": 12.0, "steady_guesses": 80, "steady_ms_per_guess": 10.0, "catch_up_ms_per_guess": 20.0}
    assert dict(bench.facts(report))["Draft guess (g)"] == "12 ms over 100 guesses; 10 ms steady over 80, 20 ms after a miss"
    assert report["options"]["pace"] is True and not report["notes"]


def test_pace_runs_one_reply_at_a_time_only_and_says_when_nothing_was_guessed():
    node = FakeNode()
    report = run(node, "--pace", "--parallel", "1,4,8")
    assert [level["parallel"] for level in report["results"][0]["levels"]] == [1]
    assert report["decoding"]["draft_guess"] is None
    assert any("--pace found no guesses" in note for note in report["notes"])


def test_cost_per_token_is_the_machines_price_over_the_levels_tokens_and_time():
    node = FakeNode()
    node.statuses = [with_telemetry(status(100), ENTRY, price_per_hour=0.59)]
    level = run(node, *ONE_ROUND)["results"][0]["levels"][0]
    # Both stages are on one machine, so its price counts once.
    assert level["cost_per_million_tokens"] == round(0.59 / 3600 * ROUND_SECONDS / 11 * 1e6, 4)
    assert "joules_per_token" not in level


def test_a_machine_with_no_price_takes_the_benchs_and_otherwise_there_is_no_cost():
    two_machines = status(100)
    two_machines["workers"][EXIT]["hardware"]["host_id"] = "another"
    priced = with_telemetry(two_machines, ENTRY, price_per_hour=0.59)
    node = FakeNode()
    node.statuses = [priced]
    report = run(node, *ONE_ROUND)
    assert "cost_per_million_tokens" not in report["results"][0]["levels"][0]
    assert any("1 of the route's machines have no price" in note for note in report["notes"])
    node = FakeNode()
    node.statuses = [priced]
    level = run(node, *ONE_ROUND, "--price-per-hour", "0.41")["results"][0]["levels"][0]
    assert level["cost_per_million_tokens"] == round(1.0 / 3600 * ROUND_SECONDS / 11 * 1e6, 4)


def test_energy_per_token_is_read_around_each_level_and_a_shared_card_counts_once():
    def reading(joules: float, seconds: float, samples: int, older: float = 0.0) -> dict:
        base = status(100)
        # Two stages on one machine's one card. The second worker's meter started later and
        # has counted less, and the oldest is the one read.
        for worker, less in ((ENTRY, 0.0), (EXIT, older)):
            base["workers"][worker]["runtime"]["telemetry"]["power"] = {
                "source": "nvidia-smi", "device": "0", "joules": joules - less, "seconds": seconds - less / 100, "samples": samples, "watts": 100.0,
            }  # fmt: skip
        return base

    node = FakeNode()
    # Before the run, then around the one level, then after it. The readings around the level
    # cover 1.5 s at 22 W on average, whatever the level's own wall.
    node.statuses = [reading(1000.0, 10.0, 20), reading(1500.0, 15.0, 30, older=300.0), reading(1533.0, 16.5, 33, older=300.0), reading(9999.0, 99.0, 99)]
    level = run(node, *ONE_ROUND)["results"][0]["levels"][0]
    assert (level["watts"], level["power_devices"]) == (22.0, 1)
    assert level["joules_per_token"] == round(22.0 * ROUND_SECONDS / 11, 3), "the level's own wall at the readings' power"
    report = run(FakeNode(), *ONE_ROUND)
    assert "joules_per_token" not in report["results"][0]["levels"][0], "no node measured power"


def test_a_level_shorter_than_a_few_power_readings_gives_no_energy_and_says_so():
    def reading(joules: float, seconds: float, samples: int) -> dict:
        base = status(100)
        base["workers"][ENTRY]["runtime"]["telemetry"]["power"] = {"source": "nvidia-smi", "device": "0", "joules": joules, "seconds": seconds, "samples": samples}
        return base

    node = FakeNode()
    node.statuses = [reading(1000.0, 10.0, 20), reading(1500.0, 15.0, 30), reading(1550.0, 15.5, 31), reading(1600.0, 16.0, 32)]
    report = run(node, *ONE_ROUND)
    level = report["results"][0]["levels"][0]
    assert "joules_per_token" not in level and level["power_too_short"] is True
    assert any("too few power readings" in note for note in report["notes"])


def test_cost_and_energy_columns_show_only_when_measured():
    report = run(FakeNode(), *ONE_ROUND)
    assert bench.table(report)[0][-1] == "Total tokens/s"
    report["results"][0]["levels"][0] |= {"cost_per_million_tokens": 5.2, "joules_per_token": 3.0, "watts": 94.3}
    headers, rows = bench.table(report)
    assert headers[-3:] == ["Cost per M tokens", "J/token", "Watts"] and rows[0][-3:] == ["5.2", "3", "94.3"]
    assert any("J/token is that power" in sentence for sentence in bench.method(report))


def test_cuda_graphs_are_reported_and_a_change_is_a_change_in_setup(tmp_path, capsys):
    node = FakeNode()
    node.statuses = [with_telemetry(with_telemetry(status(100), ENTRY, cuda_graphs=False), EXIT, cuda_graphs=False)]
    off = run(node)
    assert dict(bench.facts(off))["CUDA graphs"] == "off"
    on = run(FakeNode())
    assert "CUDA graphs" not in dict(bench.facts(on)), "a node too old to say says nothing"
    paths = []
    for name, report in (("a", on), ("b", off)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(str(path))
    assert compare(*paths) == 0
    shown = capsys.readouterr().out
    assert "CUDA graphs" in shown and "may come from that and not from the code" in shown


def test_a_price_below_zero_is_refused(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["bench", "--price-per-hour", "-2"])
