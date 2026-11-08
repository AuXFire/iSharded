"""The control plane stays responsive: status never waits on a worker, and a held loop is noticed."""

import asyncio
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from isharded import orchestrator as orchestrator_module
from isharded.orchestrator import Orchestrator, WorkerRecord
from isharded.protocol import PROTOCOL_VERSION, HardwareInfo, StageAssignment, WorkerRegistration


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
