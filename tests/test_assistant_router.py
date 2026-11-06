"""The router: reach, health, budget, measured results, and failing over."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import pytest

from shardmesh.assistant.app import Assistant
from shardmesh.assistant.cli import add_arguments, run
from shardmesh.assistant.contracts import ModelReply, ModelRequest, Reach
from shardmesh.assistant.router import (
    BACKOFF_SECONDS,
    MAX_BACKOFF_SECONDS,
    MEASURED_RUNS,
    AllFailed,
    Candidate,
    Refused,
    StoppedPartway,
    parse_candidate,
    wilson_lower_bound,
)
from test_assistant_fakes import FakeModelServer, FakeProvider, Reply

DAY = 86_400.0


class Clock:
    def __init__(self, now: float = 1_000 * DAY) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def assistant(tmp_path, clock):
    with Assistant(tmp_path, clock=clock) as instance:
        yield instance


def ask(role: str = "conversation", reach: Reach = Reach.CLOUD, text: str = "Hello", **options) -> ModelRequest:
    return ModelRequest(role=role, messages=[{"role": "user", "content": text}], reach=reach, **options)


def provide(assistant, *providers: FakeProvider, role: str = "conversation") -> None:
    for provider in providers:
        assistant.providers.register(provider)
    assistant.router.set_route(role, [provider.name for provider in providers])


def run_cli(home: Path, *arguments: str) -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return run(parser.parse_args(["--home", str(home), *arguments]))


# Measuring


@pytest.mark.parametrize(("successes", "runs", "expected"), [(0, 0, 0.0), (0, 10, 0.0), (1, 1, 0.2065), (95, 100, 0.8882), (5, 10, 0.2366)])
def test_the_wilson_bound_matches_known_values(successes, runs, expected):
    assert wilson_lower_bound(successes, runs) == pytest.approx(expected, abs=0.0005)


def test_a_little_evidence_is_worth_less_than_a_lot():
    assert wilson_lower_bound(2, 2) < wilson_lower_bound(95, 100)
    assert wilson_lower_bound(100, 100) > wilson_lower_bound(10, 10)


def test_every_run_is_recorded_with_tokens_and_time(assistant):
    provide(assistant, FakeProvider("a", tokens=(40, 10), seconds=2.0))
    assistant.router.complete(ask())
    assistant.router.complete(ask())
    row = assistant.router.stats("conversation")[0]
    assert (row["runs"], row["successes"], row["tokens"], row["seconds"]) == (2, 2, 100, 4.0)
    assert row["tokens_per_second"] == 25.0 and row["wilson"] == pytest.approx(wilson_lower_bound(2, 2))


def test_stats_are_kept_per_role(assistant):
    provide(assistant, FakeProvider("a"))
    assistant.router.complete(ask("conversation"))
    assistant.router.complete(ask("worker"))
    assert {row["role"] for row in assistant.router.stats()} == {"conversation", "worker"}


# Choosing


def seed(assistant, name: str, runs: int, successes: int, role: str = "conversation", model: str = "m", tokens: int = 0, seconds: float = 0.0) -> None:
    assistant.store.execute(
        "INSERT INTO model_stats (provider, model, role, runs, successes, tokens, seconds) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (name, model, role, runs, successes, tokens, seconds),
    )


def test_the_best_measured_candidate_is_picked_whatever_the_listed_order(assistant):
    weak, strong = FakeProvider("weak"), FakeProvider("strong")
    provide(assistant, weak, strong)
    seed(assistant, "weak", 100, 60)
    seed(assistant, "strong", 100, 95)
    assert assistant.router.complete(ask()).provider == "strong"
    assert weak.calls == 0


def test_a_long_record_beats_a_short_perfect_one(assistant):
    provide(assistant, FakeProvider("lucky"), FakeProvider("proven"))
    seed(assistant, "lucky", 5, 5)
    seed(assistant, "proven", 100, 90)
    assert assistant.router.plan(ask())[0][0].provider == "proven"


def test_untested_candidates_are_tried_first_until_there_is_something_to_measure(assistant):
    provide(assistant, FakeProvider("proven"), FakeProvider("new"))
    seed(assistant, "proven", 1000, 1000)
    assert [assistant.router.complete(ask()).provider for _ in range(MEASURED_RUNS + 1)] == ["new"] * MEASURED_RUNS + ["proven"]


def test_one_passing_fault_does_not_bury_a_good_model(assistant, clock):
    """A failed first run scores zero, which would otherwise lose to any single success."""
    good, other = FakeProvider("good", fail=True), FakeProvider("other")
    provide(assistant, good, other)
    assert assistant.router.complete(ask()).provider == "other"
    good.fail = False
    clock.now += BACKOFF_SECONDS + 1
    assert assistant.router.complete(ask()).provider == "good"


def test_with_equal_results_the_faster_then_the_first_listed_wins(assistant):
    provide(assistant, FakeProvider("a"), FakeProvider("b"), FakeProvider("c"))
    seed(assistant, "a", 10, 10, tokens=100, seconds=10.0)
    seed(assistant, "b", 10, 10, tokens=100, seconds=1.0)
    seed(assistant, "c", 10, 10, tokens=100, seconds=1.0)
    assert [item.provider for item in assistant.router.plan(ask())[0]] == ["b", "c", "a"]


def test_the_choice_is_deterministic_given_the_same_stats(assistant):
    provide(assistant, *(FakeProvider(name) for name in "abcdef"))
    for index, name in enumerate("abcdef"):
        seed(assistant, name, 20, 10 + index % 3)
    first = assistant.router.plan(ask())[0]
    assert all(assistant.router.plan(ask())[0] == first for _ in range(25))


def test_the_same_stats_in_another_install_choose_the_same(tmp_path, clock):
    orders = []
    for folder in ("one", "two"):
        with Assistant(tmp_path / folder, clock=clock) as other:
            provide(other, FakeProvider("a"), FakeProvider("b"), FakeProvider("c"))
            seed(other, "a", 50, 25)
            seed(other, "b", 50, 45)
            seed(other, "c", 50, 35)
            orders.append(other.router.plan(ask())[0])
    assert orders[0] == orders[1] and orders[0][0].provider == "b"


def test_verification_feeds_back_into_the_choice(assistant):
    provide(assistant, FakeProvider("a"), FakeProvider("b"))
    seed(assistant, "a", 10, 10)
    seed(assistant, "b", 10, 9)
    assert assistant.router.plan(ask())[0][0].provider == "a"
    for _ in range(4):
        assistant.router.record_outcome("m", "conversation", False, provider="a")
    assert assistant.router.plan(ask())[0][0].provider == "b"
    row = next(item for item in assistant.router.stats() if item["provider"] == "a")
    assert (row["runs"], row["successes"]) == (10, 6)


def test_record_outcome_accepts_the_reply_itself(assistant):
    provide(assistant, FakeProvider("a"))
    reply = assistant.router.complete(ask())
    assistant.router.record_outcome(reply, "conversation", False)
    row = assistant.router.stats()[0]
    assert (row["runs"], row["successes"]) == (1, 0)


def test_a_passed_check_does_not_count_a_run_twice(assistant):
    provide(assistant, FakeProvider("a"))
    reply = assistant.router.complete(ask())
    assistant.router.record_outcome(reply, "conversation", True)
    row = assistant.router.stats()[0]
    assert (row["runs"], row["successes"]) == (1, 1)


def test_an_outcome_for_an_unseen_model_starts_a_record_and_never_goes_negative(assistant):
    assistant.router.record_outcome("new", "coding", False, provider="p")
    assistant.router.record_outcome("new", "coding", False, provider="p")
    row = assistant.router.stats("coding")[0]
    assert (row["runs"], row["successes"]) == (1, 0)
    assert any(item["action"] == "router.outcome" for item in assistant.audit.read())


# Reach


def test_a_provider_beyond_the_reach_is_never_used(assistant):
    cloud = FakeProvider("cloud", Reach.CLOUD)
    pool = FakeProvider("pool", Reach.POOL)
    provide(assistant, cloud, pool)
    assert assistant.router.complete(ask(reach=Reach.POOL)).provider == "pool"
    assert cloud.calls == 0


def test_reach_holds_when_the_allowed_provider_fails(assistant):
    """Failing over must never be a way around a limit."""
    cloud = FakeProvider("cloud", Reach.CLOUD)
    pool = FakeProvider("pool", Reach.POOL, fail=True)
    provide(assistant, pool, cloud)
    with pytest.raises(AllFailed):
        assistant.router.complete(ask(reach=Reach.POOL))
    assert cloud.calls == 0 and cloud.model_asks == 0


def test_a_refusal_says_why_and_is_audited(assistant):
    cloud, mesh = FakeProvider("cloud", Reach.CLOUD), FakeProvider("mesh", Reach.POOL)
    provide(assistant, cloud, mesh)
    with pytest.raises(Refused) as refused:
        assistant.router.complete(ask(reach=Reach.MACHINE, text="the private words"))
    message = str(refused.value)
    assert "cloud reaches cloud" in message and "mesh reaches pool" in message and "no further than machine" in message
    assert refused.value.refusal is True and cloud.calls == 0 and mesh.calls == 0
    rows = [row for row in assistant.audit.read() if row["action"] == "router.refused"]
    assert len(rows) == 1 and "private words" not in str(rows)


def test_a_provider_beyond_the_reach_is_not_even_asked_its_model(assistant):
    cloud = FakeProvider("cloud", Reach.CLOUD)
    provide(assistant, cloud)
    with pytest.raises(Refused):
        assistant.router.complete(ask(reach=Reach.POOL))
    assert cloud.model_asks == 0


@pytest.mark.parametrize("limit", list(Reach))
def test_every_reach_admits_exactly_what_is_no_further(assistant, limit):
    providers = [FakeProvider(item.name, item) for item in Reach]
    provide(assistant, *providers)
    usable = {candidate.provider for candidate in assistant.router.plan(ask(reach=limit))[0]}
    assert usable == {item.name for item in Reach if item <= limit}


# Being up


def test_failing_over_records_both_runs(assistant):
    down, up = FakeProvider("down", fail=True), FakeProvider("up")
    provide(assistant, down, up)
    assert assistant.router.complete(ask()).provider == "up"
    stats = {row["provider"]: row for row in assistant.router.stats()}
    assert (stats["down"]["runs"], stats["down"]["successes"]) == (1, 0)
    assert (stats["up"]["runs"], stats["up"]["successes"]) == (1, 1)
    assert any(row["action"] == "router.failed" for row in assistant.audit.read())


def test_an_empty_reply_counts_as_a_failure(assistant):
    empty, good = FakeProvider("empty", text="   "), FakeProvider("good")
    provide(assistant, empty, good)
    assert assistant.router.complete(ask()).provider == "good"


def test_a_failed_provider_is_left_alone_then_tried_again(assistant, clock):
    down, up = FakeProvider("down", fail=True), FakeProvider("up")
    provide(assistant, down, up)
    assistant.router.complete(ask())
    assert down.calls == 1
    assistant.router.complete(ask())
    assert down.calls == 1
    clock.now += BACKOFF_SECONDS + 1
    down.fail = False
    assistant.router.complete(ask())
    assistant.router.complete(ask())
    assert down.calls >= 2


def test_back_off_grows_with_each_failure_and_has_a_limit(assistant, clock):
    down = FakeProvider("down", fail=True)
    provide(assistant, down)
    waits = []
    for _ in range(8):
        with pytest.raises(AllFailed):
            assistant.router.complete(ask())
        row = assistant.store.one("SELECT retry_at FROM model_health")
        waits.append(row["retry_at"] - clock.now)
        clock.now = row["retry_at"] + 1
    assert waits[:3] == [BACKOFF_SECONDS, BACKOFF_SECONDS * 2, BACKOFF_SECONDS * 4]
    assert max(waits) == MAX_BACKOFF_SECONDS


def test_a_success_clears_the_back_off(assistant, clock):
    down = FakeProvider("down", fail=True)
    provide(assistant, down)
    with pytest.raises(AllFailed):
        assistant.router.complete(ask())
    clock.now += BACKOFF_SECONDS + 1
    down.fail = False
    assistant.router.complete(ask())
    assert assistant.store.one("SELECT count(*) AS total FROM model_health")["total"] == 0


def test_when_everything_is_backing_off_the_refusal_says_when_it_will_retry(assistant):
    provide(assistant, FakeProvider("down", fail=True))
    with pytest.raises(AllFailed):
        assistant.router.complete(ask())
    with pytest.raises(Refused, match="tried again in 15 s"):
        assistant.router.complete(ask())


def test_back_off_survives_a_restart(tmp_path, clock):
    with Assistant(tmp_path, clock=clock) as first:
        down = FakeProvider("down", fail=True)
        provide(first, down)
        with pytest.raises(AllFailed):
            first.router.complete(ask())
    with Assistant(tmp_path, clock=clock) as second:
        again = FakeProvider("down", fail=True)
        second.providers.register(again)
        with pytest.raises(Refused):
            second.router.complete(ask())
        assert again.calls == 0


def test_a_provider_that_has_no_model_loaded_is_skipped_with_a_reason(assistant):
    empty = FakeProvider("empty", model="")
    provide(assistant, empty)
    with pytest.raises(Refused, match="no model is loaded"):
        assistant.router.complete(ask())


def test_failing_to_ask_for_a_model_backs_off_too(assistant, clock):
    """A silent pool costs one wait, not one wait for every message."""

    class Silent(FakeProvider):
        def current_model(self):
            self.model_asks += 1
            raise RuntimeError("went silent")

    silent = Silent("silent")
    provide(assistant, silent)
    for _ in range(3):
        with pytest.raises(Refused):
            assistant.router.complete(ask())
    assert silent.model_asks == 1
    clock.now += BACKOFF_SECONDS + 1
    with pytest.raises(Refused):
        assistant.router.complete(ask())
    assert silent.model_asks == 2


def test_a_real_silent_link_fails_over_to_the_next_candidate(assistant):
    with FakeModelServer([Reply(silent=True)], mesh_shape=True) as silent, FakeModelServer([Reply(text="Rescued.")]) as healthy:
        assistant.providers.add("first", silent.url, mesh=True, read_seconds=0.4, model="fake-model")
        assistant.providers.add("second", healthy.url, reach="pool", tool_mode="prompt", model="fake-model")
        assistant.router.set_route("conversation", ["first", "second"])
        started = time.monotonic()
        reply = assistant.router.complete(ask())
        assert reply.text == "Rescued." and reply.provider == "second"
        assert time.monotonic() - started < 5


def test_a_real_refused_connection_fails_over_with_no_wait(assistant):
    with FakeModelServer() as healthy:
        assistant.providers.add("gone", "http://127.0.0.1:1", mesh=True, model="m")
        assistant.providers.add("second", healthy.url, reach="pool", model="m")
        assistant.router.set_route("conversation", ["gone", "second"])
        assert assistant.router.complete(ask()).provider == "second"


# Budget


def test_a_provider_over_its_daily_budget_is_skipped(assistant):
    capped = FakeProvider("capped", daily_tokens=1000, tokens=(500, 300))
    spare = FakeProvider("spare")
    provide(assistant, capped, spare)
    seed(assistant, "capped", 100, 100)
    seed(assistant, "spare", 100, 50)
    request = ask(max_tokens=300)
    assert assistant.router.complete(request).provider == "capped"
    assert assistant.router.complete(request).provider == "spare"
    assert capped.calls == 1


def test_a_budget_is_never_overrun_by_a_large_request(assistant):
    capped = FakeProvider("capped", daily_tokens=1000)
    provide(assistant, capped)
    with pytest.raises(Refused, match="could take"):
        assistant.router.complete(ask(max_tokens=2000))
    assert capped.calls == 0


def test_a_refusal_for_budget_says_how_much_is_used(assistant):
    provide(assistant, FakeProvider("capped", daily_tokens=500, tokens=(300, 100)))
    assistant.router.complete(ask(max_tokens=50))
    with pytest.raises(Refused, match="used 400 of its 500 tokens"):
        assistant.router.complete(ask(max_tokens=200))


def test_a_budget_starts_again_the_next_day(assistant, clock):
    capped = FakeProvider("capped", daily_tokens=500, tokens=(300, 100))
    provide(assistant, capped)
    assistant.router.complete(ask(max_tokens=50))
    with pytest.raises(Refused):
        assistant.router.complete(ask(max_tokens=200))
    clock.now += DAY
    assert assistant.router.complete(ask(max_tokens=200)).provider == "capped"


def test_usage_is_counted_across_models_of_one_provider(assistant):
    capped = FakeProvider("capped", daily_tokens=500, tokens=(300, 100))
    assistant.providers.register(capped)
    assistant.router.set_route("conversation", ["capped/one"])
    assistant.router.set_route("worker", ["capped/two"])
    assistant.router.complete(ask("conversation", max_tokens=50))
    with pytest.raises(Refused):
        assistant.router.complete(ask("worker", max_tokens=200))


def test_failed_runs_cost_no_budget(assistant):
    provide(assistant, FakeProvider("a", fail=True, daily_tokens=100))
    with pytest.raises(AllFailed):
        assistant.router.complete(ask(max_tokens=10))
    assert assistant.store.one("SELECT count(*) AS total FROM model_usage")["total"] == 0


# Routes


def test_a_model_name_may_contain_slashes():
    assert parse_candidate("mesh/Qwen/Qwen3-8B") == Candidate("mesh", "Qwen/Qwen3-8B")
    assert parse_candidate("mesh") == Candidate("mesh", "")
    assert str(Candidate("mesh", "a/b")) == "mesh/a/b"


def test_a_role_falls_back_to_conversation_then_to_every_provider(assistant):
    assistant.providers.register(FakeProvider("a"))
    assistant.providers.register(FakeProvider("b"))
    assert assistant.router.candidates("worker") == [Candidate("a"), Candidate("b")]
    assistant.router.set_route("conversation", ["b"])
    assert assistant.router.candidates("worker") == [Candidate("b")]
    assistant.router.set_route("worker", ["a"])
    assert assistant.router.candidates("worker") == [Candidate("a")]


def test_embedding_has_no_fallback(assistant):
    assistant.providers.register(FakeProvider("a"))
    assistant.router.set_route("conversation", ["a"])
    assert assistant.router.candidates("embedding") == [] and assistant.router.can_embed() is False


def test_a_route_needs_a_real_role_and_provider(assistant):
    with pytest.raises(ValueError, match="not a role"):
        assistant.router.set_route("wizard", ["a"])
    with pytest.raises(ValueError, match="provider list"):
        assistant.router.set_route("conversation", ["nobody"])


def test_with_nothing_set_up_the_message_says_what_to_do(assistant):
    with pytest.raises(Refused, match="provider add"):
        assistant.router.complete(ask())


def test_a_provider_removed_after_routing_is_explained(assistant):
    assistant.providers.add("a", "http://127.0.0.1:9000")
    assistant.router.set_route("conversation", ["a/m"])
    assistant.providers.remove("a")
    with pytest.raises(Refused, match="has been removed"):
        assistant.router.complete(ask())


def test_the_provider_that_serves_a_pool_can_change_model(assistant):
    pool = FakeProvider("pool", model="first")
    provide(assistant, pool)
    assert assistant.router.complete(ask()).model == "first"
    pool.model = "second"
    assert assistant.router.complete(ask()).model == "second"
    assert {row["model"] for row in assistant.router.stats()} == {"first", "second"}


# Streaming


def collect(iterator) -> tuple[str, ModelReply]:
    items = list(iterator)
    return "".join(items[:-1]), items[-1]


def test_a_stream_gives_text_then_the_reply_and_is_recorded(assistant):
    provide(assistant, FakeProvider("a", text="Streamed words."))
    text, reply = collect(assistant.router.stream(ask()))
    assert text == "Streamed words." and reply.text == "Streamed words."
    assert assistant.router.stats()[0]["runs"] == 1


def test_a_stream_fails_over_before_any_text_has_gone_out(assistant):
    provide(assistant, FakeProvider("down", fail=True), FakeProvider("up", text="From up."))
    text, reply = collect(assistant.router.stream(ask()))
    assert text == "From up." and reply.provider == "up"


def test_a_stream_that_fails_partway_cannot_silently_become_another_models_answer(assistant):
    with FakeModelServer([Reply(text="Partial answer here", cut_after=2)]) as broken, FakeModelServer() as healthy:
        assistant.providers.add("broken", broken.url, mesh=True, model="m")
        assistant.providers.add("healthy", healthy.url, reach="pool", model="m")
        assistant.router.set_route("conversation", ["broken", "healthy"])
        received = []
        with pytest.raises(StoppedPartway, match="before finishing"):
            for item in assistant.router.stream(ask()):
                received.append(item)
        assert "".join(received) == "Partial an"
        assert healthy.requests == []
    assert next(row for row in assistant.router.stats() if row["provider"] == "broken")["successes"] == 0


def test_a_stream_honours_reach_and_says_why(assistant):
    provide(assistant, FakeProvider("cloud", Reach.CLOUD))
    with pytest.raises(Refused, match="no further than pool"):
        list(assistant.router.stream(ask(reach=Reach.POOL)))


# Embedding


class Embedder(FakeProvider):
    def embed(self, model, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


def test_embedding_uses_the_role_and_respects_reach(assistant):
    cloud = Embedder("cloud", Reach.CLOUD)
    local = Embedder("local", Reach.MACHINE)
    provide(assistant, cloud, local, role="embedding")
    vectors, model = assistant.router.embed(["a", "b"], Reach.MACHINE)
    assert vectors == [[1.0, 0.0], [1.0, 0.0]] and model == "m"
    assert cloud.calls == 0 and local.calls == 1


def test_embedding_refuses_when_nothing_is_within_reach(assistant):
    provide(assistant, Embedder("cloud", Reach.CLOUD), role="embedding")
    with pytest.raises(Refused):
        assistant.router.embed(["private"], Reach.MACHINE)


def test_the_library_uses_the_real_router_for_vectors_within_reach(tmp_path, clock):
    with FakeModelServer() as server, Assistant(tmp_path / "home", clock=clock) as assistant:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.txt").write_text("Some words", encoding="utf-8")
        assistant.providers.add("embed", server.url, reach="machine", model="e")
        assistant.router.set_route("embedding", ["embed/e"])
        assistant.library.add("alice", tmp_path / "docs", Reach.MACHINE)
        assert assistant.library.index("alice").embedded == 1
        assert assistant.library.search("alice", "words", reach=Reach.MACHINE)


# Commands


def test_commands_route_and_models(tmp_path, capsys):
    with FakeModelServer() as server:
        assert run_cli(tmp_path, "provider", "add", "pool", server.url, "--mesh") == 0
        assert run_cli(tmp_path, "route", "set", "conversation", "pool/fake-model") == 0
        assert run_cli(tmp_path, "route", "show") == 0
        with Assistant(tmp_path) as assistant:
            assistant.router.complete(ask())
        assert run_cli(tmp_path, "models") == 0
    shown = capsys.readouterr().out
    assert "pool/fake-model" in shown and "conversation" in shown and "at least" in shown


def test_models_with_no_runs_says_so(tmp_path, capsys):
    assert run_cli(tmp_path, "models") == 0
    assert "No runs recorded yet" in capsys.readouterr().out
