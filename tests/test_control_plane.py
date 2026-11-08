"""The control plane stays responsive: status never waits on a worker, and planning runs off the loop."""

import asyncio
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest

from isharded import orchestrator as orchestrator_module
from isharded.orchestrator import Orchestrator, WorkerRecord
from isharded.protocol import PROTOCOL_VERSION, HardwareInfo, StageAssignment, WorkerRegistration

from test_parallel import target as slotted_target
from test_parallel import worker as candidate


def registration(worker_id):
    hardware = HardwareInfo(device="cpu", backend="cpu", cpu_cores=4, ram_total_bytes=1)
    return WorkerRegistration(worker_id=worker_id, advertised_url=f"http://{worker_id}:1", hardware=hardware, protocol_version=PROTOCOL_VERSION)


def record(worker_id, silent_for=0.0, **fields):
    return WorkerRecord(registration=registration(worker_id), last_seen=time.monotonic() - silent_for, **fields)


def stage(worker_id, start=0, end=1):
    return StageAssignment(worker_id=worker_id, url=f"http://{worker_id}:1", start_layer=start, end_layer=end, estimated_ms_per_token=1.0, placements=[])


class Recorder:
    """A client that answers /health at once, and remembers every call and how many ran together."""

    def __init__(self, delay=0.0):
        self.urls = []
        self.delay = delay
        self.running = 0
        self.most = 0

    async def get(self, url, **options):
        self.urls.append(url)
        self.running += 1
        self.most = max(self.most, self.running)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.running -= 1
        return httpx.Response(200, json={"status": "ok"}, request=httpx.Request("GET", url))

    async def aclose(self):
        return None


@pytest.fixture
def pool():
    runtime = Orchestrator()
    yield runtime
    asyncio.run(runtime.close())


def test_status_makes_no_call_to_any_worker(pool, monkeypatch):
    """Status read every worker's health on every call, and a dead one held it PROBE_SECONDS."""
    pool.workers = {"live": record("live"), "gone": record("gone", silent_for=60.0)}
    pool.client = Recorder()
    monkeypatch.setattr(orchestrator_module, "runtime", pool)
    client = TestClient(orchestrator_module.app)
    answer = client.get("/admin/status")
    assert answer.status_code == 200
    assert pool.client.urls == []
    body = answer.json()
    assert body["workers"]["gone"]["state"] == "offline"
    assert "health_age_seconds" in body and "loop_lag_ms" in body


def test_a_health_check_skips_a_worker_that_stopped_checking_in(pool):
    pool.workers = {"live": record("live"), "gone": record("gone", silent_for=60.0)}
    pool.client = Recorder()
    asyncio.run(pool.refresh_worker_metrics())
    assert pool.client.urls == ["http://live:1/health"]
    assert pool.workers["live"].health == {"status": "ok"}
    assert pool.health_checked_at is not None
    assert pool.status()["health_age_seconds"] < 5


def test_status_shows_how_old_its_health_is(pool):
    assert pool.status()["health_age_seconds"] is None
    pool.health_checked_at = time.time() - 12.0
    assert 11.5 < pool.status()["health_age_seconds"] < 13.0


def test_a_health_check_asks_a_few_workers_at_a_time(pool):
    pool.workers = {f"w{index}": record(f"w{index}") for index in range(3 * orchestrator_module.MAX_PROBES_AT_ONCE)}
    pool.client = Recorder(delay=0.02)
    asyncio.run(pool.refresh_worker_metrics())
    assert len(pool.client.urls) == 3 * orchestrator_module.MAX_PROBES_AT_ONCE
    assert pool.client.most == orchestrator_module.MAX_PROBES_AT_ONCE


def test_the_background_check_keeps_health_current(pool, monkeypatch):
    monkeypatch.setattr(orchestrator_module, "HEALTH_SECONDS", 0.05)
    pool.workers = {"live": record("live")}
    pool.client = Recorder()

    async def scenario():
        task = asyncio.ensure_future(pool._watch_health())
        await asyncio.sleep(0.3)
        task.cancel()

    asyncio.run(scenario())
    assert len(pool.client.urls) >= 3


def test_a_worker_gone_ten_minutes_is_forgotten_unless_a_route_names_it(pool):
    forget = orchestrator_module.FORGET_SECONDS
    pool.workers = {
        "fresh": record("fresh"),
        "lately": record("lately", silent_for=forget / 2),
        "long": record("long", silent_for=forget + 1),
        "routed": record("routed", silent_for=forget + 1),
    }
    pool.route = [stage("routed")]
    pool.links = {("long", "fresh"): 1.0, ("fresh", "routed"): 2.0}
    pool.link_readings = {("fresh", "long"): {"cost_ms": 1.0}}
    pool.lost_from_route = {"long"}
    pool.absent_since = {"long": 1.0}
    pool.client = Recorder()
    asyncio.run(pool.refresh_worker_metrics())
    assert set(pool.workers) == {"fresh", "lately", "routed"}
    assert pool.links == {("fresh", "routed"): 2.0} and pool.link_readings == {}
    assert not pool.lost_from_route and not pool.absent_since
    assert any(event["title"] == "long forgotten" for event in pool.events)
    assert "long" not in pool.status()["workers"] and 'worker="long"' not in pool.metrics_text()


def test_a_loading_or_saved_route_keeps_its_workers(pool):
    forget = orchestrator_module.FORGET_SECONDS
    pool.workers = {"loading": record("loading", silent_for=forget + 1), "saved": record("saved", silent_for=forget + 1)}
    pool.job = orchestrator_module.LoadJob(job_id="j", model_id="m", route=[stage("loading")])
    pool.saved_state = {"epoch": 1, "request": {}, "route": [stage("saved").model_dump()]}
    assert pool.forget_silent_workers() == []
    pool.job.finished_at = time.time()
    pool.saved_state = None
    assert sorted(pool.forget_silent_workers()) == ["loading", "saved"]


def test_being_forgotten_is_no_way_out_of_quarantine(pool):
    pool.workers = {"bad": record("bad", silent_for=orchestrator_module.FORGET_SECONDS + 1, quarantined=True, quarantine_reason="out of memory", quarantine_count=2)}
    assert pool.forget_silent_workers() == ["bad"]
    pool.register(registration("bad"))
    back = pool.workers["bad"]
    assert back.quarantined and back.quarantine_reason == "out of memory" and back.quarantine_count == 2


class Ticks:
    """A 10 Hz coroutine, as a heartbeat or a status call would be, noting how late each tick ran."""

    def __init__(self):
        self.late = []

    async def run(self, stop):
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            planned = loop.time() + 0.1
            await asyncio.sleep(0.1)
            self.late.append(loop.time() - planned)


def slow_plan(seconds):
    """A planner that holds the interpreter for ``seconds``, as a plan on a big pool does."""

    def plan(self, target, **options):
        ended = time.perf_counter() + seconds
        while time.perf_counter() < ended:
            sum(range(200))  # pure Python, holding the GIL as planning does
        return [stage("a")]

    return plan


def while_ticking(work):
    async def scenario():
        ticks, stop = Ticks(), asyncio.Event()
        ticker = asyncio.ensure_future(ticks.run(stop))
        await asyncio.sleep(0.25)
        began = time.perf_counter()
        result = await work()
        took = time.perf_counter() - began
        stop.set()
        await ticker
        return ticks, took, result

    return asyncio.run(scenario())


def test_the_loop_keeps_ticking_while_a_plan_runs(pool, monkeypatch):
    monkeypatch.setattr(Orchestrator, "plan", slow_plan(1.5))
    target = SimpleNamespace(draft=None, request=None)
    ticks, took, route = while_ticking(lambda: pool.planned(target))
    assert took >= 1.5 and [item.worker_id for item in route] == ["a"]
    assert len(ticks.late) >= 15
    assert max(ticks.late) < 0.1, f"a tick ran {max(ticks.late) * 1000:.0f} ms late"


def test_the_loop_keeps_ticking_while_a_cordon_is_priced(pool, monkeypatch):
    monkeypatch.setattr(Orchestrator, "plan", slow_plan(1.0))
    monkeypatch.setattr(orchestrator_module, "describe_plan", lambda target, route, links=None: {"download_bytes": 5})
    pool.workers = {"a": record("a")}
    pool.active = SimpleNamespace(draft=None, request=None)
    pool.route = [stage("a")]
    ticks, took, cost = while_ticking(lambda: pool.cost_of_losing("a"))
    assert cost == 5 and took >= 1.0
    assert max(ticks.late) < 0.1, f"a tick ran {max(ticks.late) * 1000:.0f} ms late"


def test_the_loop_keeps_ticking_through_the_slot_search(pool, monkeypatch):
    """The slot search plans once a step, and each plan held the loop."""
    pool.offered = [candidate("a"), candidate("b")]
    pool.candidate_groups = lambda: (pool.offered, [])
    pool.links = {("a", "b"): 200.0, ("b", "a"): 200.0}
    real_plan = Orchestrator.plan
    plans = []

    def plan(self, target, **options):
        plans.append(threading.current_thread() is threading.main_thread())
        ended = time.perf_counter() + 0.3
        while time.perf_counter() < ended:
            sum(range(200))
        return real_plan(self, target, **options)

    monkeypatch.setattr(Orchestrator, "plan", plan)
    item = slotted_target()
    ticks, took, _ = while_ticking(lambda: pool.choose_slots(item))
    assert len(plans) >= 3 and not any(plans), "every plan of the search ran off the loop"
    assert item.parallel["auto"] and item.slots > 1, "what the search settled reached the target"
    assert max(ticks.late) < 0.1, f"a tick ran {max(ticks.late) * 1000:.0f} ms late"


def test_a_plan_overtaken_by_a_cordon_is_made_again(pool, monkeypatch):
    pool.workers = {"a": record("a"), "b": record("b")}
    seen = []

    def plan(self, target, **options):
        seen.append(sorted(self.access.cordoned))
        self.add_event("rebalance", "Planned", f"attempt {len(seen)}")
        time.sleep(0.2)
        return [stage(name) for name in sorted(self.workers) if not self._cordoned(name)]

    monkeypatch.setattr(Orchestrator, "plan", plan)

    async def scenario():
        planning = asyncio.ensure_future(pool.planned(SimpleNamespace(draft=None, request=None)))
        await asyncio.sleep(0.05)
        pool.access.cordon("b")  # lands while the first plan is in its thread
        return await planning

    route = asyncio.run(scenario())
    assert seen == [[], ["b"]]
    assert [item.worker_id for item in route] == ["a"]
    told = [event["detail"] for event in pool.events if event["title"] == "Planned"]
    assert told == ["attempt 2"], "only the plan that was used tells what it did"


def test_a_plan_never_touches_the_pool_it_reads(pool, monkeypatch):
    pool.workers = {"a": record("a"), "b": record("b")}
    pool.route = [stage("a")]
    pool.links = {("a", "b"): 1.0}

    def plan(self, target, **options):
        self.workers["a"].quarantined = True
        self.access.cordoned.add("b")
        self.route.clear()
        self.links.clear()
        return []

    monkeypatch.setattr(Orchestrator, "plan", plan)
    asyncio.run(pool.planned(SimpleNamespace(draft=None, request=None)))
    assert not pool.workers["a"].quarantined and not pool.access.cordoned
    assert [item.worker_id for item in pool.route] == ["a"] and pool.links == {("a", "b"): 1.0}


def test_a_refused_plan_still_tells_why(pool, monkeypatch):
    def plan(self, target, **options):
        raise orchestrator_module.PlacementError("no room anywhere")

    monkeypatch.setattr(Orchestrator, "plan", plan)
    with pytest.raises(orchestrator_module.HTTPException) as refused:
        asyncio.run(pool.planned(SimpleNamespace(draft=None, request=None)))
    assert refused.value.status_code == 507 and "no room anywhere" in refused.value.detail


def test_a_held_loop_is_measured_logged_and_exported(pool, capsys):
    async def scenario():
        watcher = asyncio.ensure_future(pool._watch_loop())
        await asyncio.sleep(0.25)
        time.sleep(0.4)  # holds the loop, as a plan on it did
        await asyncio.sleep(0.25)
        watcher.cancel()

    asyncio.run(scenario())
    assert pool.loop_stalls == 1
    assert pool.worst_loop_lag() >= 0.3
    assert "event loop was held" in capsys.readouterr().out
    text = pool.metrics_text()
    assert "isharded_event_loop_lag_seconds " in text
    assert "isharded_event_loop_stalls_total 1" in text
    worst = float(next(line for line in text.splitlines() if line.startswith("isharded_event_loop_lag_worst_seconds ")).split()[1])
    assert worst >= 0.3
    assert pool.status()["loop_lag_ms"]["held_over_250ms"] == 1


def test_the_worst_lag_forgets_after_a_minute(pool, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: clock[0])
    pool.note_loop_lag(0.2)
    pool.note_loop_lag(0.01)
    assert pool.worst_loop_lag() == 0.2
    clock[0] += 61.0
    pool.note_loop_lag(0.02)
    assert pool.worst_loop_lag() == 0.02


def test_the_probe_limit_works_on_a_later_event_loop(pool):
    """A semaphore binds to the first loop that waits on it, and a node's tests and restarts use several."""
    pool.workers = {f"w{index}": record(f"w{index}") for index in range(2 * orchestrator_module.MAX_PROBES_AT_ONCE)}
    for _ in range(2):
        pool.client = Recorder(delay=0.02)
        asyncio.run(pool.refresh_worker_metrics())
        assert pool.client.most == orchestrator_module.MAX_PROBES_AT_ONCE


def test_a_held_loop_says_where_it_was_held(pool, capsys):
    """A lag number alone does not say what to fix, so the log line names the code that held the loop."""

    def hold_the_loop_here():
        time.sleep(1.0)  # well past the limit, so a busy machine still gives the watcher a turn

    async def scenario():
        watcher = asyncio.ensure_future(pool._watch_loop())
        await asyncio.sleep(0.25)
        hold_the_loop_here()
        await asyncio.sleep(0.25)
        watcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watcher

    asyncio.run(scenario())
    line = next(line for line in capsys.readouterr().out.splitlines() if "event loop was held" in line)
    assert "hold_the_loop_here" in line and "test_control_plane.py" in line
    assert pool.loop_stall_where is None, "a place is told once, with the stall it belongs to"
    assert not any(thread.name == "isharded-loop-watch" for thread in threading.enumerate()), "the watcher's thread ends with it"


def test_a_place_caught_after_a_stall_is_not_told_with_the_next(pool, capsys):
    """The watcher can catch the loop just after a stall's line was written; that place belongs to no stall."""
    pool.loop_stall_where = (1.0, "late.py:1 caught_after_the_line")  # caught after the tick that followed 1.0
    pool.note_loop_lag(0.0, since=2.0)
    pool.loop_stall_where = (3.0, "plan.py:9 held_here")
    pool.note_loop_lag(0.5, since=3.0)
    pool.loop_stall_where = (3.5, "late.py:1 caught_after_the_line")
    pool.note_loop_lag(0.6, since=4.0)
    lines = [line for line in capsys.readouterr().out.splitlines() if "event loop was held" in line]
    assert len(lines) == 2
    assert "plan.py:9 held_here" in lines[0]
    assert "caught_after_the_line" not in lines[1] and ", in " not in lines[1]


def test_planning_that_changed_nothing_undoes_nothing(pool, monkeypatch):
    """A cordon priced while a load settles slots must not put the old slots back when it ends."""
    monkeypatch.setattr(orchestrator_module, "PLAN_ATTEMPTS", 1)
    monkeypatch.setattr(orchestrator_module, "describe_plan", lambda target, route, links=None: {"download_bytes": 5})

    def plan(self, target, **options):
        time.sleep(0.2)
        return [stage("a")]

    monkeypatch.setattr(Orchestrator, "plan", plan)
    pool.workers = {"a": record("a"), "b": record("b")}
    pool.active = SimpleNamespace(draft=None, request=None, slots=1, parallel={})
    pool.route = [stage("a")]

    async def scenario():
        pricing = asyncio.ensure_future(pool.cost_of_losing("a"))
        await asyncio.sleep(0.05)
        pool.active.slots, pool.active.parallel = 4, {"auto": True}  # settled on the loop meanwhile
        return await pricing

    assert asyncio.run(scenario()) == 5
    assert pool.active.slots == 4 and pool.active.parallel == {"auto": True}
