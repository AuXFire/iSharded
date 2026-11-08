"""A stage's load has a deadline from its size, kept for as long as it shows progress.

Before, configure waited on a client with an hour's timeout, and every re-plan empties the
route first, so one suspended worker could hold the whole pool with no route for an hour.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
from fastapi.testclient import TestClient
import pytest

from isharded import orchestrator as orchestrator_module, worker
from isharded.orchestrator import STALLED, ConfigureStalled, configure_seconds
from isharded.protocol import StageAssignment, UnitPlacement
from test_resilience import deploy_pool, pool, record, run_deploy


@pytest.fixture
def quick(monkeypatch):
    """Deadlines short enough to watch: a third of a second, looked at every 50 ms."""
    monkeypatch.setattr(orchestrator_module, "CONFIGURE_FLOOR_SECONDS", 0.3)
    monkeypatch.setattr(orchestrator_module, "CONFIGURE_STALL_SECONDS", 0.3)
    monkeypatch.setattr(orchestrator_module, "CONFIGURE_LOOK_SECONDS", 0.05)
    monkeypatch.setattr(orchestrator_module, "GIVE_UP_TELL_SECONDS", 1.0)


def stage_of(size: int = 0) -> StageAssignment:
    placements = [UnitPlacement(unit="layer-00000.dense", pool_id="cpu", bytes=size)] if size else []
    return StageAssignment(worker_id="w1", url="http://w1:1", start_layer=0, end_layer=1, placements=placements)


def request_for(stage: StageAssignment) -> MagicMock:
    request = MagicMock(load_id="load-1")
    request.model_dump.return_value = {"load_id": "load-1"}
    return request


class Worker:
    """A worker's configure and progress routes, scripted."""

    def __init__(self, loads_for: float, signs: str = "moving") -> None:
        self.loads_for = loads_for
        self.signs = signs  # moving, still, silent, or other: another load's signs
        self.looks = 0
        self.given_up: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/internal/configure":
            await asyncio.sleep(self.loads_for)
            return httpx.Response(200, json={"status": "configured"})
        if path == "/internal/configure/progress":
            self.looks += 1
            if self.signs == "silent":
                raise httpx.ReadTimeout("nothing came back", request=request)
            if self.signs == "suspended":
                # A suspended process takes the connection and never answers: the asker's
                # own timeout is all that ends the wait, as on a real socket.
                await asyncio.sleep(request.extensions["timeout"]["read"])
                raise httpx.ReadTimeout("nothing came back", request=request)
            moved = self.looks if self.signs in {"moving", "other"} else 0
            return httpx.Response(200, json={"load_id": "load-9" if self.signs == "other" else "load-1", "done": 0, "total": 1, "read_mib": moved, "children_cpu_s": 0.0})
        if path.endswith("/give-up"):
            self.given_up.append(path.split("/")[3])
            return httpx.Response(200, json={"status": "given up"})
        return httpx.Response(404)


def configure(scripted: Worker, size: int = 0) -> tuple[float, object, object]:
    plane = pool(w1=record("w1"))

    async def scenario():
        plane.client = httpx.AsyncClient(transport=httpx.MockTransport(scripted))
        stage = stage_of(size)
        began = asyncio.get_running_loop().time()
        try:
            outcome = await plane._configure_in_time(stage, request_for(stage))
        except ConfigureStalled as exc:
            outcome = exc
        took = asyncio.get_running_loop().time() - began
        await asyncio.gather(*plane.background, return_exceptions=True)
        await plane.close()
        return took, outcome

    took, outcome = asyncio.run(scenario())
    return took, outcome, plane


def test_the_deadline_grows_with_the_stage_and_a_measured_rate_halves():
    floor = orchestrator_module.CONFIGURE_FLOOR_SECONDS
    assert configure_seconds(0) == floor
    assert configure_seconds(5 * 10**9) == pytest.approx(floor + 100)  # 5 GB at the assumed 50 MB/s
    assert configure_seconds(5 * 10**9, measured_rate=1e9) == pytest.approx(floor + 10)


def test_defaults_stay_inside_the_wait_a_reply_gives_a_route():
    # A small stage that stalls is given up on, and the route planned again, while replies still wait.
    assert configure_seconds(2 * 1024**3) < orchestrator_module.ROUTE_WAIT_SECONDS


def test_a_load_that_keeps_moving_is_waited_for_past_its_deadline(quick, monkeypatch):
    # Room for a busy machine to look late, and still five times the first deadline.
    monkeypatch.setattr(orchestrator_module, "CONFIGURE_STALL_SECONDS", 1.0)
    took, outcome, _ = configure(Worker(loads_for=1.5, signs="moving"))
    assert isinstance(outcome, httpx.Response) and outcome.status_code == 200
    assert took >= 1.4  # Windows clocks tick in 15 ms steps


def test_a_load_that_stops_moving_is_given_up_on_and_told(quick):
    scripted = Worker(loads_for=30, signs="still")
    took, outcome, _ = configure(scripted)
    assert isinstance(outcome, ConfigureStalled) and str(outcome).startswith(STALLED)
    assert 0.3 <= took < 1.0
    assert scripted.given_up == ["load-1"], "a worker that wakes later lets the weights go"


def test_a_worker_that_answers_nothing_is_given_up_on_at_its_deadline(quick):
    took, outcome, _ = configure(Worker(loads_for=30, signs="silent"))
    assert isinstance(outcome, ConfigureStalled)
    assert took < 1.0


def test_a_suspended_worker_is_let_go_at_its_deadline_and_not_a_question_later(quick, monkeypatch):
    # Measured on a real pool: a load given 35 s was let go 42 s in, the last question to
    # the suspended worker running its full 5 s past the deadline.
    monkeypatch.setattr(orchestrator_module, "CONFIGURE_FLOOR_SECONDS", 1.0)
    monkeypatch.setattr(orchestrator_module, "CONFIGURE_LOOK_SECONDS", 0.6)
    monkeypatch.setattr(orchestrator_module, "CONFIGURE_SIGNS_SECONDS", 3.0)
    took, outcome, _ = configure(Worker(loads_for=30, signs="suspended"))
    assert isinstance(outcome, ConfigureStalled)
    assert took < 1.9, f"let go {took:.2f} s into a 1 s deadline"


def test_another_loads_signs_are_no_progress_for_this_one(quick):
    took, outcome, _ = configure(Worker(loads_for=30, signs="other"))
    assert isinstance(outcome, ConfigureStalled) and took < 1.0


def test_a_big_load_that_finishes_sets_the_workers_load_rate(quick):
    took, outcome, plane = configure(Worker(loads_for=1.0), size=512 * 1024**2)
    rate = plane.workers["w1"].load_bytes_per_second
    assert rate == pytest.approx(512 * 1024**2 / took, rel=0.05)


def test_the_route_is_planned_again_without_a_stalled_worker_and_not_retried(monkeypatch):
    # Moving 15 GiB is refused for a node that is alive and failed to load. A stalled
    # load is different: waiting on it would leave the pool with no route.
    stall = {"w1": f"{STALLED}: nothing moved for 30 s, 46 s into a load given 46 s for 0.80 GiB"}
    orchestrator, attempts = deploy_pool(monkeypatch, [stall, {}], 15 * 1024**3)
    assert run_deploy(orchestrator)
    assert len(attempts) == 2, "the same route is not tried again on a stalled worker"
    assert orchestrator.workers["w1"].quarantined
    assert orchestrator.workers["w1"].quarantine_reason.startswith(STALLED)



def test_each_plan_of_a_load_fetches_under_its_own_name(monkeypatch):
    # Measured on a real pool: after a stalled worker, the new plan gave the other worker
    # the layers the stalled one had. A worker takes a known fetch name as a repeat, so it
    # answered with the first plan's finished download and the load failed for want of them.
    stall = {"w1": f"{STALLED}: nothing moved for 30 s, 46 s into a load given 46 s for 0.80 GiB"}
    orchestrator, _ = deploy_pool(monkeypatch, [stall, {}], 0)
    names = []

    async def prefetch(target, route, job):
        names.append(job.fetch_id)
        return {}

    orchestrator._prefetch = prefetch
    assert run_deploy(orchestrator)
    assert names == ["j", "j-1"]


# --- the worker's side


def test_a_worker_shows_its_load_moving_without_counting_its_own_answers():
    runtime = SimpleNamespace(configuring="load-1", materialize_done=2, materialize_total=5)
    signs = worker.WorkerRuntime.configure_signs(runtime)
    assert signs["load_id"] == "load-1" and (signs["done"], signs["total"]) == (2, 5)
    assert isinstance(signs["read_mib"], int) and isinstance(signs["children_cpu_s"], float)
    # Asking again moves nothing: this process's own CPU, which answering spends, is left out.
    assert worker.WorkerRuntime.configure_signs(runtime) == signs


@pytest.fixture
def node(monkeypatch):
    runtime = SimpleNamespace(pool_token="secret", requires_identity=False, revoked_nodes=set(), authorized=None, configuring=None, materialize_done=0, materialize_total=0, given_up={}, holding_load=None, lock=asyncio.Lock())
    runtime.configure_signs = lambda: worker.WorkerRuntime.configure_signs(runtime)
    monkeypatch.setattr(worker, "runtime", runtime)
    return runtime


def test_progress_and_give_up_need_the_pool_token(node):
    client = TestClient(worker.app)
    assert client.get("/internal/configure/progress").status_code == 401
    assert client.post("/internal/configure/load-1/give-up").status_code == 401
    answer = client.get("/internal/configure/progress", headers={"authorization": "Bearer secret"})
    assert answer.status_code == 200 and answer.json()["load_id"] is None


def test_a_given_up_load_is_remembered_and_only_the_newest_kept(node):
    client = TestClient(worker.app)
    for index in range(70):
        assert client.post(f"/internal/configure/load-{index}/give-up", headers={"authorization": "Bearer secret"}).status_code == 200
    assert len(node.given_up) == 64 and "load-69" in node.given_up and "load-0" not in node.given_up


def test_a_give_up_heard_after_the_load_finished_lets_its_weights_go(node):
    """A suspended worker wakes, finishes its load, and only then hears it was given up on."""
    released = []
    node._release_weights = lambda: released.append(True)
    node.holding_load = "load-1"
    client = TestClient(worker.app)
    headers = {"authorization": "Bearer secret"}
    assert client.post("/internal/configure/load-0/give-up", headers=headers).status_code == 200
    assert released == [] and node.holding_load == "load-1", "an older load's give-up leaves a newer one alone"
    assert client.post("/internal/configure/load-1/give-up", headers=headers).status_code == 200
    assert released == [True] and node.holding_load is None


def test_a_load_finishing_after_it_was_given_up_on_lets_its_weights_go(monkeypatch):
    released = []
    runtime = MagicMock()
    runtime.given_up = {"load-1": None}
    runtime.roles = {"inference"}
    runtime.worker_id = "w1"
    runtime.term = 0
    runtime.host_models = None
    runtime._materialize = lambda request, adapter: None
    runtime._release_weights = lambda: released.append(True)
    monkeypatch.setattr(worker, "hosts_model", lambda hosts, model: True)
    monkeypatch.setattr(worker, "create_adapter", lambda architecture: SimpleNamespace(num_layers=4))
    monkeypatch.setattr(worker, "has_torch", lambda: True)
    request = SimpleNamespace(model_id="m", term=0, architecture={}, load_id="load-1", assignment=SimpleNamespace(worker_id="w1", start_layer=0, end_layer=2))
    with pytest.raises(ValueError, match="gave up on this load"):
        asyncio.run(worker.WorkerRuntime.configure(runtime, request))
    assert released == [True] and runtime.configuring is None


def test_a_configure_request_carries_its_load_id_and_an_older_one_has_none():
    from isharded.protocol import ConfigureRequest

    assert ConfigureRequest.model_fields["load_id"].default is None
