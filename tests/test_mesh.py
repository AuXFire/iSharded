import asyncio
import json

from fastapi.testclient import TestClient
import httpx

from shardmesh.serving import MAX_REQUEST_BYTES
import pytest

from shardmesh import orchestrator as orchestrator_module
from shardmesh.mesh import GOSSIP_SECONDS, JOIN_GRACE_SECONDS, PEER_TIMEOUT_SECONDS, ControlMesh
from shardmesh.orchestrator import Orchestrator
from shardmesh.protocol import LoadModelRequest, StageAssignment


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Network:
    """Nodes that gossip through a list, with links that can be cut."""

    def __init__(self, *names):
        self.clock = Clock()
        self.nodes = {
            name: ControlMesh(name, f"http://{name}", tuple(f"http://{other}" for other in names), self.clock)
            for name in names
        }
        self.down = set()
        self.cut = set()
        self.stepped_down = []
        self.took_over = []

    def reachable(self, a, b):
        return a not in self.down and b not in self.down and frozenset((a, b)) not in self.cut

    def step(self, rounds=1):
        for _ in range(rounds):
            self.clock.now += GOSSIP_SECONDS
            for name, node in self.nodes.items():
                if name in self.down:
                    continue
                for other, peer in self.nodes.items():
                    if other != name and self.reachable(name, other):
                        if peer.observe(node.message()):
                            self.stepped_down.append(other)
                        if node.observe(peer.message()):
                            self.stepped_down.append(name)
                if node.elect():
                    self.took_over.append(name)

    def leaders(self):
        return {name for name, node in self.nodes.items() if name not in self.down and node.is_leader}

    def settled(self):
        return self.step(4)


def test_nodes_agree_on_one_leader():
    network = Network("a", "b", "c")
    network.settled()
    assert network.leaders() == {"a"}
    assert {node.leader_id for node in network.nodes.values()} == {"a"}


def test_nobody_claims_leadership_before_listening():
    network = Network("a", "b")
    network.nodes["a"].started_at = network.clock.now
    assert not network.nodes["a"].elect()
    network.clock.now += JOIN_GRACE_SECONDS
    assert network.nodes["a"].elect()


def test_a_lone_node_leads_itself():
    clock = Clock()
    node = ControlMesh("solo", "http://solo", (), clock)
    clock.now += JOIN_GRACE_SECONDS
    assert node.elect() and node.is_leader
    assert node.term == 1


def test_the_next_node_takes_over_when_the_leader_dies():
    network = Network("a", "b", "c")
    network.settled()
    network.down.add("a")
    network.step(rounds=int(PEER_TIMEOUT_SECONDS / GOSSIP_SECONDS) + 2)
    assert network.leaders() == {"b"}
    assert network.nodes["c"].leader_id == "b"
    assert network.nodes["b"].term == 2


def test_a_returning_leader_does_not_take_its_job_back():
    network = Network("a", "b")
    network.settled()
    network.down.add("a")
    network.step(rounds=6)
    assert network.leaders() == {"b"}
    network.down.discard("a")
    network.step(rounds=2)
    assert network.leaders() == {"b"}, "the lower term must defer, even with the better name"
    assert "a" in network.stepped_down


def test_a_restarted_leader_must_win_an_election_again():
    clock = Clock()
    restarted = ControlMesh("a", "http://a", ("http://b",), clock)
    restarted.observe({"node_id": "b", "url": "http://b", "term": 4, "leader_id": "a"})
    assert not restarted.is_leader, "a peer's memory of its leadership is not an election"
    assert restarted.term == 4
    clock.now += JOIN_GRACE_SECONDS
    assert restarted.elect(), "the election is what runs the takeover"
    assert restarted.term == 5


def test_a_leader_that_restarts_quickly_does_not_leave_the_mesh_leaderless():
    """It returns before anyone timed it out, with no memory of leading."""
    clock = Clock()
    follower = ControlMesh("b", "http://b", (), clock)
    follower.leader_id, follower.term = "c", 3
    follower.started_at -= JOIN_GRACE_SECONDS
    follower.observe({"node_id": "c", "url": "http://c", "term": 3, "leader_id": None})
    assert follower.leader_id is None
    assert follower.elect() and follower.term == 4


def test_a_healed_split_ends_with_one_leader():
    network = Network("a", "b", "c", "d")
    network.settled()
    for left in ("a", "b"):
        for right in ("c", "d"):
            network.cut.add(frozenset((left, right)))
    network.step(rounds=6)
    assert network.leaders() == {"a", "c"}, "each side keeps serving what it can reach"
    network.cut.clear()
    network.step(rounds=2)
    assert network.leaders() == {"c"}, "the higher term wins, whatever the names"
    assert {node.term for node in network.nodes.values()} == {2}


def test_equal_terms_settle_on_the_lower_name():
    clock = Clock()
    node = ControlMesh("b", "http://b", (), clock)
    node.term, node.leader_id = 3, "b"
    assert node.observe({"node_id": "a", "url": "http://a", "term": 3, "leader_id": "a"})
    assert node.leader_id == "a"


def test_membership_spreads_from_a_single_seed():
    clock = Clock()
    a = ControlMesh("a", "http://a", (), clock)
    b = ControlMesh("b", "http://b", ("http://a",), clock)
    c = ControlMesh("c", "http://c", ("http://a",), clock)
    a.observe(b.message())
    c.observe(a.message())
    assert "http://b" in c.targets(), "c learned of b from a without ever being told its address"


def test_a_node_never_gossips_to_itself():
    node = ControlMesh("a", "http://a", ("http://a", "http://b/"), Clock())
    assert node.targets() == {"http://b"}


def test_followers_keep_the_leaders_state():
    network = Network("a", "b")
    network.settled()
    network.nodes["a"].publish({"epoch": 7})
    network.step()
    assert network.nodes["b"].replica == {"epoch": 7}
    assert network.nodes["b"].message()["replica"] is None, "only the leader speaks for the state"


def test_an_old_leaders_state_does_not_overwrite_a_new_one():
    clock = Clock()
    node = ControlMesh("b", "http://b", (), clock)
    node.leader_id, node.term = "a", 5
    node.replica, node.replica_version = {"epoch": 9}, 4
    node.observe({"node_id": "z", "url": "http://z", "term": 1, "leader_id": "z", "replica": {"epoch": 1}, "replica_version": 9})
    assert node.replica == {"epoch": 9}


def test_a_term_learned_from_workers_outlives_the_control_plane():
    clock = Clock()
    node = ControlMesh("a", "http://a", (), clock)
    node.learn_term(9)
    clock.now += JOIN_GRACE_SECONDS
    assert node.elect()
    assert node.term == 10


@pytest.fixture
def mesh_orchestrator(monkeypatch):
    monkeypatch.setenv("SHARDMESH_NODE_URL", "http://me:1")
    monkeypatch.setenv("SHARDMESH_NODE_ID", "me")
    monkeypatch.delenv("SHARDMESH_STATE_FILE", raising=False)
    orchestrator = Orchestrator()
    yield orchestrator
    asyncio.run(orchestrator.close())


def test_stepping_down_stops_all_coordination(mesh_orchestrator):
    orchestrator = mesh_orchestrator
    orchestrator.mesh.leader_id, orchestrator.mesh.term = "me", 1
    orchestrator.route = [StageAssignment(worker_id="w", url="http://w", start_layer=0, end_layer=1)]
    orchestrator.model_id = "m"
    orchestrator.receive_gossip({"node_id": "other", "url": "http://other", "term": 2, "leader_id": "other"})
    assert not orchestrator.coordinating
    assert orchestrator.route, "replies in progress still need the route"


def test_taking_over_picks_up_the_replicated_state(mesh_orchestrator):
    orchestrator = mesh_orchestrator
    orchestrator.mesh.replica = {"epoch": 8, "request": {"model_id": "m"}, "route": []}
    orchestrator._take_over()
    assert orchestrator.saved_state["epoch"] == 8
    assert orchestrator.epoch == 8


def test_followers_do_not_repair_anything(mesh_orchestrator, monkeypatch):
    orchestrator = mesh_orchestrator
    orchestrator.mesh.leader_id = "other"
    called = []

    async def refresh():
        called.append("refresh")

    orchestrator.refresh_worker_metrics = refresh
    orchestrator._review_quarantine = lambda: called.append("review")
    asyncio.run(orchestrator.maintenance_pass())
    assert called == ["refresh"]


def test_state_reaches_followers_on_disk(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    monkeypatch.setenv("SHARDMESH_NODE_URL", "http://me:1")
    monkeypatch.setenv("SHARDMESH_NODE_ID", "me")
    monkeypatch.setenv("SHARDMESH_STATE_FILE", str(path))
    orchestrator = Orchestrator()
    state = {"epoch": 4, "request": LoadModelRequest(model_id="m").model_dump(), "route": []}
    orchestrator.receive_gossip(
        {"node_id": "a", "url": "http://a", "term": 1, "leader_id": "a", "replica": state, "replica_version": 1}
    )
    assert json.loads(path.read_text(encoding="utf-8"))["epoch"] == 4
    asyncio.run(orchestrator.close())


def forwarding(monkeypatch, leader_url, handler):
    runtime = Orchestrator()
    runtime.mesh = ControlMesh("me", "http://me", (), Clock())
    runtime.mesh.leader_id = "boss"
    if leader_url:
        runtime.mesh.observe({"node_id": "boss", "url": leader_url, "term": 1, "leader_id": "boss"})
    runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    return TestClient(orchestrator_module.app)


def test_any_node_serves_the_api_by_forwarding_it(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["forwarded"] = request.headers.get("x-shardmesh-forwarded")
        return httpx.Response(200, stream=httpx.ByteStream(b'{"answer": 42}'), headers={"content-type": "application/json"})

    client = forwarding(monkeypatch, "http://boss:9", handler)
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.json() == {"answer": 42}
    assert seen == {"url": "http://boss:9/v1/chat/completions", "body": {"messages": []}, "forwarded": "me"}


def test_a_follower_serves_replies_itself_once_it_has_the_route(monkeypatch):
    def handler(request):
        raise AssertionError("a reply must not depend on the coordinator")

    client = forwarding(monkeypatch, "http://boss:9", handler)
    orchestrator_module.runtime.route = [StageAssignment(worker_id="w", url="http://w", start_layer=0, end_layer=1)]
    assert client.get("/v1/models").status_code == 200


def test_management_still_goes_to_the_coordinator(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"), headers={"content-type": "application/json"})

    client = forwarding(monkeypatch, "http://boss:9", handler)
    orchestrator_module.runtime.route = [StageAssignment(worker_id="w", url="http://w", start_layer=0, end_layer=1)]
    client.get("/admin/status")
    assert seen == ["/admin/status"]


def test_a_follower_installs_each_new_route_once(mesh_orchestrator, monkeypatch):
    orchestrator = mesh_orchestrator
    orchestrator.mesh.leader_id = "other"
    orchestrator.mesh.replica = {"epoch": 4, "request": LoadModelRequest(model_id="m").model_dump(), "route": []}
    assert orchestrator._should_follow()
    orchestrator.installed_epoch = 4
    assert not orchestrator._should_follow()
    orchestrator.installed_epoch = 3
    orchestrator.following = True
    assert not orchestrator._should_follow()


def test_pages_and_gossip_stay_local_on_a_follower(monkeypatch):
    def handler(request):
        raise AssertionError("must not be forwarded")

    client = forwarding(monkeypatch, "http://boss:9", handler)
    assert client.get("/health").status_code == 200
    assert client.get("/chat").status_code == 200
    assert client.get("/internal/mesh").json()["leader"] == "boss"


def test_no_coordinator_gives_a_clear_answer(monkeypatch):
    def handler(request):
        raise AssertionError("nowhere to forward")

    client = forwarding(monkeypatch, None, handler)
    response = client.get("/admin/status")
    assert response.status_code == 503
    assert "no coordinator" in response.json()["detail"]


def test_forwarding_cannot_loop(monkeypatch):
    def handler(request):
        raise AssertionError("must not forward twice")

    client = forwarding(monkeypatch, "http://boss:9", handler)
    assert client.get("/admin/status", headers={"x-shardmesh-forwarded": "x"}).status_code == 503


def test_a_dead_coordinator_is_reported_not_raised(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("gone")

    client = forwarding(monkeypatch, "http://boss:9", handler)
    assert client.get("/admin/status").status_code == 503


def test_workers_refuse_orders_from_an_older_term(monkeypatch):
    from shardmesh import worker
    from shardmesh.protocol import ConfigureRequest

    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.roles = {"inference"}
    runtime.worker_id = "w"
    runtime.term = 5
    stage = StageAssignment(worker_id="w", url="http://w", start_layer=0, end_layer=1)
    stale = ConfigureRequest(model_id="m", revision="r", architecture={}, epoch=9, assignment=stage, route=[stage], term=4)
    with pytest.raises(ValueError, match="term 5 already took over"):
        asyncio.run(runtime.configure(stale))


def test_only_mesh_members_may_cross_origins(monkeypatch):
    def handler(request):
        raise AssertionError("not forwarded")

    client = forwarding(monkeypatch, "http://boss:9", handler)
    allowed = client.options("/v1/chat/completions", headers={"Origin": "http://boss:9", "Access-Control-Request-Method": "POST"})
    assert allowed.status_code == 204
    assert allowed.headers["access-control-allow-origin"] == "http://boss:9"
    stranger = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in stranger.headers


def test_a_cordoned_node_gets_no_work(mesh_orchestrator):
    orchestrator = mesh_orchestrator
    orchestrator.access.cordon("laptop")
    assert orchestrator._cordoned("laptop-gpu")
    assert orchestrator._cordoned("laptop")
    assert not orchestrator._cordoned("laptop2-gpu")
    assert not orchestrator._cordoned("desktop-gpu")


def test_cordons_reach_every_node():
    from shardmesh.access import AccessControl

    first, second = AccessControl(), AccessControl()
    first.cordon("laptop")
    second.merge(first.document())
    assert "laptop" in second.cordoned


def test_oversized_requests_are_refused_unread(monkeypatch):
    def handler(request):
        raise AssertionError("must not get this far")

    client = forwarding(monkeypatch, "http://boss:9", handler)
    assert client.post("/v1/chat/completions", content=b"x" * (MAX_REQUEST_BYTES + 1)).status_code == 413
