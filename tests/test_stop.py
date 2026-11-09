"""A planned stop moves the route off the node first, then stops its services together."""

from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import httpx
import pytest

from isharded import node as node_module, up
from isharded.node import Service, stop_together

NAME = "laptop"
SETTINGS = {"name": NAME, "port": 8209, "tls": False}
ON_ME = [{"worker_id": "desk-gpu", "start_layer": 0}, {"worker_id": f"{NAME}-cpu", "start_layer": 14}]
OFF_ME = [{"worker_id": "desk-gpu", "start_layer": 0}]


class Coordinator:
    """What a node's own control plane answers, which it passes on to the coordinator."""

    def __init__(self, routes: list[list[dict]], cordon_status: int = 200, cordoned: list[str] | None = None, down: bool = False) -> None:
        self.routes = routes  # one per status call; the last repeats
        self.cordon_status = cordon_status
        self.cordoned = cordoned or []
        self.down = down
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path, dict(request.url.params)))
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        if request.url.path == "/admin/status":
            route = self.routes.pop(0) if len(self.routes) > 1 else self.routes[0]
            return httpx.Response(200, json={"route": route, "cordoned_nodes": self.cordoned})
        if request.url.path.endswith("/cordon"):
            body = {"status": "cordoned"} if self.cordon_status == 200 else {"detail": "the rest of the pool cannot hold this model without laptop"}
            return httpx.Response(self.cordon_status, json=body)
        if request.url.path.endswith("/uncordon"):
            return httpx.Response(200, json={"status": "uncordoned"})
        return httpx.Response(404)

    def posted(self) -> list[str]:
        return [path for method, path, _ in self.calls if method == "POST"]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "admin.key").write_text("isk-admin", encoding="utf-8")
    monkeypatch.setattr(up, "up_home", lambda: tmp_path)
    return tmp_path


def serve(monkeypatch: pytest.MonkeyPatch, coordinator: Coordinator) -> None:
    monkeypatch.setattr(up, "client_for", lambda settings, home: httpx.Client(transport=httpx.MockTransport(coordinator)))


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_the_route_moves_off_before_the_stop_and_the_cordon_is_remembered(home, monkeypatch):
    coordinator = Coordinator([ON_ME, ON_ME, ON_ME, OFF_ME])
    serve(monkeypatch, coordinator)
    clock = Clock()
    said = up.move_route_off(SETTINGS, home, 30, clock, clock.sleep)
    assert said.startswith("the route moved off this node in 0.8 s")
    assert coordinator.posted() == [f"/admin/nodes/{NAME}/cordon"]
    assert coordinator.calls[1][2] == {"force": "true"}, "a node that is leaving anyway moves whatever it must"
    assert (home / up.STOPPED_CORDON).read_text(encoding="utf-8") == NAME


def test_a_route_with_no_stage_yet_has_not_moved_off(home, monkeypatch):
    # Between the old route and the new one, the status shows none; that is not the move done.
    coordinator = Coordinator([ON_ME, [], [], OFF_ME])
    serve(monkeypatch, coordinator)
    clock = Clock()
    assert "moved off this node in 0.8 s" in up.move_route_off(SETTINGS, home, 30, clock, clock.sleep)


def test_with_no_coordinator_the_stop_goes_on_as_before(home, monkeypatch):
    coordinator = Coordinator([ON_ME], down=True)
    serve(monkeypatch, coordinator)
    assert up.move_route_off(SETTINGS, home, 30) == "no coordinator answered, so nothing was moved first"
    assert not coordinator.posted() and not (home / up.STOPPED_CORDON).exists()


def test_a_refused_cordon_stops_at_once_and_says_why(home, monkeypatch):
    coordinator = Coordinator([ON_ME], cordon_status=409)
    serve(monkeypatch, coordinator)
    said = up.move_route_off(SETTINGS, home, 30)
    assert said.startswith("the cordon was refused (the rest of the pool cannot hold") and not (home / up.STOPPED_CORDON).exists()
    assert [path for _, path, _ in coordinator.calls].count("/admin/status") == 1, "nothing to wait for"


def test_a_route_that_never_moves_is_waited_for_only_so_long(home, monkeypatch):
    serve(monkeypatch, Coordinator([ON_ME]))
    clock = Clock()
    said = up.move_route_off(SETTINGS, home, 5, clock, clock.sleep)
    assert said == "the route was still on this node after 5 s, so it stops anyway"
    assert 5 <= clock.now < 5 + up.STOP_LOOK_SECONDS * 2
    assert (home / up.STOPPED_CORDON).exists(), "the next start must still put it back"


def test_a_node_the_route_does_not_use_is_not_cordoned(home, monkeypatch):
    coordinator = Coordinator([OFF_ME])
    serve(monkeypatch, coordinator)
    assert up.move_route_off(SETTINGS, home, 30) == "the route does not use this node"
    assert not coordinator.posted()


def test_a_cordon_someone_set_by_hand_is_waited_on_and_left_in_place(home, monkeypatch):
    coordinator = Coordinator([ON_ME, OFF_ME], cordoned=[NAME])
    serve(monkeypatch, coordinator)
    clock = Clock()
    assert "moved off" in up.move_route_off(SETTINGS, home, 30, clock, clock.sleep)
    assert not coordinator.posted()
    assert not (home / up.STOPPED_CORDON).exists(), "the next start must not undo someone else's cordon"


def test_the_next_start_puts_a_node_its_stop_cordoned_back_in_service(home, monkeypatch):
    coordinator = Coordinator([OFF_ME])
    client = httpx.Client(transport=httpx.MockTransport(coordinator))
    up.back_in_service(SETTINGS, home, client)
    assert not coordinator.posted(), "no stop cordoned it"
    (home / up.STOPPED_CORDON).write_text(NAME, encoding="utf-8")
    up.back_in_service(SETTINGS, home, client)
    assert coordinator.posted() == [f"/admin/nodes/{NAME}/uncordon"]
    assert not (home / up.STOPPED_CORDON).exists()


def stop_arguments(*words: str):
    parser = up.argparse.ArgumentParser()
    up.add_arguments(parser.add_subparsers())
    arguments = parser.parse_args(["stop", *words])
    assert arguments.handler is up.run_stop
    return arguments


class Running:
    """The node's process, as run_stop sees it."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.asked_at: float | None = None

    def wait(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not (self.home / "stop.request").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        self.asked_at = time.monotonic()


def test_stop_cordons_first_and_only_then_asks_the_node_to_stop(home, monkeypatch, capsys):
    (home / "settings.json").write_text(json.dumps(SETTINGS), encoding="utf-8")
    process = Running(home)
    monkeypatch.setattr(up, "running_node", lambda folder: process)
    moved_at = []

    def move(settings, folder, wait):
        assert not (folder / "stop.request").exists(), "the route moves before the node is told to stop"
        assert wait == 7.0
        moved_at.append(time.monotonic())
        return "the route moved off this node in 1.0 s"

    monkeypatch.setattr(up, "move_route_off", move)
    assert up.run_stop(stop_arguments("--wait", "7")) == 0
    assert process.asked_at is not None and moved_at and moved_at[0] <= process.asked_at
    assert "The route moved off this node in 1.0 s." in capsys.readouterr().out


def test_stop_with_no_wait_skips_the_move(home, monkeypatch):
    (home / "settings.json").write_text(json.dumps(SETTINGS), encoding="utf-8")
    monkeypatch.setattr(up, "running_node", lambda folder: Running(home))
    monkeypatch.setattr(up, "move_route_off", lambda *args: pytest.fail("--wait 0 stops at once"))
    assert up.run_stop(up.argparse.Namespace(wait=0.0)) == 0


def test_the_stop_command_waits_thirty_seconds_unless_told(monkeypatch):
    assert stop_arguments().wait == 30.0
    assert stop_arguments("--wait", "0").wait == 0.0


# --- the services stop together


class Slow:
    """A process that takes ``seconds`` to end once asked, or never ends."""

    def __init__(self, seconds: float | None) -> None:
        self.seconds = seconds
        self.asked: float | None = None
        self.killed = False
        self.pid = 1

    def poll(self):
        if self.killed:
            return -9
        if self.asked is not None and self.seconds is not None and time.monotonic() - self.asked >= self.seconds:
            return 0
        return None

    def terminate(self) -> None:
        self.asked = time.monotonic()

    def wait(self, timeout: float) -> int:
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise node_module.subprocess.TimeoutExpired("x", timeout)
            time.sleep(0.005)
        return 0

    def kill(self) -> None:
        self.killed = True


def started(process: Slow) -> Service:
    service = Service("test", lambda: process)
    service.start()
    return service


def test_services_are_asked_together_so_the_slowest_sets_the_time():
    processes = [Slow(0.4), Slow(0.4), Slow(0.4)]
    began = time.monotonic()
    stop_together([started(process) for process in processes])
    took = time.monotonic() - began
    assert all(process.poll() == 0 for process in processes)
    assert took < 0.8, f"one at a time would take 1.2 s, and this took {took:.2f}"
    assert max(process.asked for process in processes) - min(process.asked for process in processes) < 0.1


def test_a_service_that_will_not_stop_is_killed_at_the_shared_deadline(monkeypatch):
    monkeypatch.setattr(Service, "STOP_SECONDS", 0.3)
    stuck, quick = Slow(None), Slow(0.05)
    began = time.monotonic()
    stop_together([started(stuck), started(quick)])
    assert stuck.killed and not quick.killed
    assert time.monotonic() - began < 0.6


def test_a_service_that_already_ended_is_left_alone():
    ended = Slow(0)
    ended.asked = time.monotonic() - 1
    service = started(ended)
    stop_together([service])
    assert not ended.killed


def test_ctrl_c_on_a_node_stops_every_service_together(monkeypatch):
    processes = [Slow(0.3), Slow(0.3)]
    launched = iter(processes)
    monkeypatch.setattr(node_module, "node_specs", lambda args: [("Control plane", ["x"]), ("Worker", ["y"])])
    monkeypatch.setattr(node_module, "_publish_settings", lambda args: None)
    monkeypatch.setattr(node_module, "_child_environment", lambda args: ({}, Path(".")))
    monkeypatch.setattr(node_module.subprocess, "Popen", lambda *args, **kwargs: next(launched))

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    # Its supervising loop sleeps between looks, and Ctrl+C lands there.
    monkeypatch.setattr(node_module, "time", SimpleNamespace(sleep=interrupt, monotonic=time.monotonic))
    args = up.argparse.Namespace(no_lanes=True, no_control=False, node_id=NAME, log_file=None)
    began = time.monotonic()
    assert node_module.run_node(args) == 0
    assert all(process.poll() == 0 for process in processes)
    assert time.monotonic() - began < 0.55


# An invited machine holds only a user key, which may not cordon anything. Its stop then
# asks as the node itself, with a ticket from its own certificate, straight to the coordinator.


def node_certificate(home: Path, name: str = NAME) -> None:
    pytest.importorskip("cryptography")
    from isharded.pool import add_node, init_pool

    folder = init_pool(str(home / "authority"))
    add_node(str(folder), name, ["127.0.0.1"])
    for suffix in (".pem", ".key"):
        (home / f"{name}{suffix}").write_bytes((folder / f"{name}{suffix}").read_bytes())


class InvitedNode:
    """A joined node's control plane, which refuses its operator's user key a cordon, and the coordinator."""

    LEADER = "http://desk:8300"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        asked = request.headers.get("authorization", "")
        where = f"{request.url.scheme}://{request.url.netloc.decode()}"
        self.calls.append((request.method, where + request.url.path, asked[:11]))
        ticket = asked.startswith("Bearer smn-")
        if request.url.path == "/admin/status":
            return httpx.Response(200, json={"route": OFF_ME if self.moved else ON_ME, "cordoned_nodes": []})
        if request.url.path == "/internal/lane":
            return httpx.Response(200, json={"leader_url": self.LEADER}) if ticket else httpx.Response(401)
        if request.url.path.endswith(("/cordon", "/uncordon")):
            if where == self.LEADER and ticket:
                self.moved = request.url.path.endswith("/cordon")
                return httpx.Response(200, json={"status": "done"})
            return httpx.Response(403, json={"detail": "this needs the member role"})
        return httpx.Response(404)

    moved = False


def test_an_invited_node_moves_the_route_off_as_itself_when_its_key_cannot(home, monkeypatch):
    (home / "admin.key").unlink()
    (home / "user.key").write_text("isk-user", encoding="utf-8")
    node_certificate(home)
    node = InvitedNode()
    serve(monkeypatch, node)
    clock = Clock()
    said = up.move_route_off(SETTINGS, home, 30, clock, clock.sleep)
    assert said.startswith("the route moved off this node"), said
    posted = [(where, asked) for method, where, asked in node.calls if method == "POST"]
    assert posted == [
        ("http://127.0.0.1:8209/admin/nodes/laptop/cordon", "Bearer isk-"),
        ("http://desk:8300/admin/nodes/laptop/cordon", "Bearer smn-"),
    ], "the key first, then the node's own ticket, to the coordinator"
    up.back_in_service(SETTINGS, home, up.client_for(SETTINGS, home))
    assert not node.moved and not (home / up.STOPPED_CORDON).exists()


class SelfCordoned(InvitedNode):
    """The coordinator holds this node cordoned, by its own stop or by someone else."""

    def __init__(self, by_itself: bool) -> None:
        super().__init__()
        self.by_itself = by_itself
        self.moved = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/uncordon"):
            where = f"{request.url.scheme}://{request.url.netloc.decode()}"
            asked = request.headers.get("authorization", "")
            self.calls.append((request.method, where + request.url.path, asked[:11]))
            if where == self.LEADER and asked.startswith("Bearer smn-") and self.by_itself:
                self.moved = False
                return httpx.Response(200, json={"status": "uncordoned"})
            return httpx.Response(403, json={"detail": f"{NAME} was cordoned by someone else, who has to lift it"})
        return super().__call__(request)


def test_a_node_that_rejoined_after_its_stop_lifts_its_own_cordon_as_itself(home, monkeypatch):
    """A rejoin is for a lost folder, and the stop's marker went with it."""
    node_certificate(home)
    (home / up.REJOINED_CORDON).write_text(NAME, encoding="utf-8")
    node = SelfCordoned(by_itself=True)
    up.back_in_service(SETTINGS, home, httpx.Client(transport=httpx.MockTransport(node)))
    assert not node.moved, "the node is back in service"
    posted = [(where, asked) for method, where, asked in node.calls if method == "POST"]
    assert posted == [("http://desk:8300/admin/nodes/laptop/uncordon", "Bearer smn-")], "only as itself, never with the key"
    assert not (home / up.REJOINED_CORDON).exists()


def test_a_node_that_rejoined_leaves_a_cordon_someone_else_set(home, monkeypatch, capsys):
    """An owner's key could lift any cordon, so a rejoin asks only as the node, which cannot."""
    node_certificate(home)
    (home / up.REJOINED_CORDON).write_text(NAME, encoding="utf-8")
    node = SelfCordoned(by_itself=False)
    up.back_in_service(SETTINGS, home, httpx.Client(transport=httpx.MockTransport(node)))
    assert node.moved and not (home / up.REJOINED_CORDON).exists()
    assert "stays out of service" in capsys.readouterr().out


def test_without_its_certificate_a_refused_node_stops_as_before(home, monkeypatch):
    (home / "admin.key").unlink()
    (home / "user.key").write_text("isk-user", encoding="utf-8")
    serve(monkeypatch, InvitedNode())
    said = up.move_route_off(SETTINGS, home, 30)
    assert said.startswith("the cordon was refused (this needs the member role)")


@pytest.fixture
def coordinator_seen_by(monkeypatch):
    from fastapi.testclient import TestClient

    from isharded import orchestrator as orchestrator_module
    from isharded.access import AccessControl
    from isharded.tls import PEER

    runtime = orchestrator_module.Orchestrator()
    runtime.access = AccessControl(identities=True)
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)

    def seen_by(peer: str) -> TestClient:
        async def with_identity(scope, receive, send):
            PEER.set(peer)  # what the TLS layer or a good ticket does
            await orchestrator_module.app(scope, receive, send)

        return TestClient(with_identity)

    return runtime, seen_by


def test_a_node_takes_only_itself_out_and_back(coordinator_seen_by):
    runtime, seen_by = coordinator_seen_by
    laptop = seen_by(NAME)
    assert laptop.post("/admin/nodes/desk/cordon").status_code == 403
    assert laptop.post(f"/admin/nodes/{NAME}/cordon", params={"force": "true"}).status_code == 200
    assert runtime.access.cordoned == {NAME} and runtime.access.leaving == {NAME}
    assert seen_by("desk").post(f"/admin/nodes/{NAME}/uncordon").status_code == 403
    assert laptop.post(f"/admin/nodes/{NAME}/uncordon").status_code == 200
    assert not runtime.access.cordoned and not runtime.access.leaving


def test_a_node_cannot_lift_a_cordon_someone_else_set(coordinator_seen_by):
    runtime, seen_by = coordinator_seen_by
    runtime.access.cordon(NAME)
    laptop = seen_by(NAME)
    assert laptop.post(f"/admin/nodes/{NAME}/cordon").status_code == 200, "taking itself out again is fine"
    refused = laptop.post(f"/admin/nodes/{NAME}/uncordon")
    assert refused.status_code == 403 and "someone else" in refused.json()["detail"]
    assert runtime.access.cordoned == {NAME}


def test_a_cordon_set_over_a_nodes_own_becomes_the_setters():
    from isharded.access import AccessControl

    access = AccessControl()
    access.cordon(NAME, by_itself=True)
    assert access.document()["leaving_nodes"] == [NAME]
    access.cordon(NAME)
    assert access.leaving == set() and access.cordoned == {NAME}
    assert "leaving_nodes" not in access.document(), "documents stay as they were for pools that never use it"


def test_a_nodes_own_cordon_travels_with_the_newer_document():
    from isharded.access import AccessControl

    left, right = AccessControl(), AccessControl()
    left.cordon(NAME, by_itself=True)
    assert right.merge(left.document())
    assert right.cordoned == {NAME} and right.leaving == {NAME}
    right.uncordon(NAME)
    assert left.merge(right.document())
    assert not left.cordoned and not left.leaving
