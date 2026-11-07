"""What a person manages over HTTP: the same keys and rules as the gateway, and the refusals first."""
from __future__ import annotations

import json
from pathlib import Path
import re

from fastapi.testclient import TestClient
import pytest

from isharded.assistant import manage_http
from isharded.assistant.app import Assistant
from isharded.assistant.contracts import Capability, Effect, Reach
from isharded.assistant.gateway import Users, create_app
from test_assistant_coder import FIX_AS_FILE, GitBroker, ProcessSandbox, repo  # noqa: F401
from test_assistant_coder import ScriptedRouter as CodingRouter
from test_assistant_fakes import FakeModelServer, FakeProvider, ScriptedRouter
from test_assistant_gateway import Live, client

READ_ONLY_FOR_THE_OWNER = ["/assistant/providers", "/assistant/plugins", "/assistant/peers"]
OPEN_TO_ANY_KEY = {"POST /assistant/plan"}  # it asks a model and runs nothing, as a chat turn does


@pytest.fixture
def assistant(tmp_path):
    with Assistant(tmp_path / "home") as instance:
        yield instance


@pytest.fixture
def keys(assistant):
    users = Users(assistant)
    return {"owner": users.add("owner"), "alice": users.add("alice"), "bob": users.add("bob")}


@pytest.fixture
def app(assistant, keys):
    return create_app(assistant)


def signed_in(app, key: str) -> TestClient:
    return TestClient(app, headers={"Authorization": f"Bearer {key}"})


@pytest.fixture
def owner(app, keys):
    return signed_in(app, keys["owner"])


@pytest.fixture
def alice(app, keys):
    return signed_in(app, keys["alice"])


@pytest.fixture
def bob(app, keys):
    return signed_in(app, keys["bob"])


def acting_routes(assistant) -> list[tuple[str, str]]:
    """Every route here that changes anything, so a new one cannot forget who may call it."""
    found = [
        (method, re.sub(r"\{[^}]+\}", "x", route.path))
        for route in manage_http.routes(assistant).routes
        for method in route.methods
        if method != "GET"
    ]
    return [(method, path) for method, path in found if f"{method} {path}" not in OPEN_TO_ANY_KEY] + [("POST", "/assistant/tasks/x/reflect")]


def test_every_route_that_acts_refuses_a_member(assistant, alice):
    routes = acting_routes(assistant)
    assert len(routes) >= 19
    before = assistant.store.one("SELECT count(*) AS total FROM audit")["total"]
    for method, path in routes:
        answer = alice.request(method, path, json={})
        assert answer.status_code == 403, f"{method} {path}: {answer.text}"
        assert answer.json()["detail"] == "Only the owner can do this"
    assert assistant.store.one("SELECT count(*) AS total FROM audit")["total"] == before


@pytest.mark.parametrize("path", READ_ONLY_FOR_THE_OWNER)
def test_what_shows_this_machine_is_for_the_owner(alice, owner, path):
    assert alice.get(path).status_code == 403
    assert owner.get(path).status_code == 200


def test_every_new_route_needs_a_key(assistant, app):
    anonymous = TestClient(app)
    for method, path in acting_routes(assistant) + [("GET", "/assistant/grants"), ("GET", "/assistant/skills"), ("GET", "/assistant/identity/history")]:
        assert anonymous.request(method, path, json={}).status_code == 401, path


@pytest.mark.parametrize("name", ["model-one", "conversation-two"])
def test_an_owners_key_with_a_machines_name_decides_nothing(assistant, app, name):
    """Such a user can be made, and the broker would not take them for a person."""
    key = Users(assistant).add(name, "owner")
    machine = signed_in(app, key)
    for method, path in acting_routes(assistant):
        if path != "/assistant/providers/x/test":  # asks a server if it is up, and decides nothing
            answer = machine.request(method, path, json={})
            assert answer.status_code == 403, f"{method} {path}: {answer.text}"
            assert "only a person can decide" in answer.json()["detail"]


# Grants


def test_the_owner_grants_and_revokes_in_their_own_name(assistant, owner, alice, tmp_path):
    # A request cannot say who is deciding, whatever it sends.
    made = owner.post("/assistant/grants", json={"capability": "files.read", "scope": str(tmp_path), "for_user": "alice", "uses": 2, "by": "skill:evil"})
    assert made.status_code == 200, made.text
    grant = made.json()["id"]
    row = assistant.broker.grants("alice")[0]
    assert (row["id"], row["granted_by"], row["uses_left"]) == (grant, "owner", 2)
    assert [item["id"] for item in alice.get("/assistant/grants").json()["grants"]] == [grant]

    assert owner.post(f"/assistant/grants/{grant}/revoke").json() == {"revoked": True}
    assert alice.get("/assistant/grants").json()["grants"] == []
    assert alice.get("/assistant/grants", params={"ended": True}).json()["grants"][0]["revoked_by"] == "owner"
    assert owner.post(f"/assistant/grants/{grant}/revoke").status_code == 404


def test_grants_are_shown_only_to_their_user_and_the_owner(assistant, owner, alice, bob, tmp_path):
    assistant.broker.add_grant("alice", "files.read", str(tmp_path), "owner")
    assert bob.get("/assistant/grants").json()["grants"] == []
    assert bob.get("/assistant/grants", params={"user": "alice"}).status_code == 403
    assert len(owner.get("/assistant/grants").json()["grants"]) == 1
    assert owner.get("/assistant/grants", params={"user": "bob"}).json()["grants"] == []


def test_a_member_cannot_grant_themselves_anything(assistant, alice, tmp_path):
    answer = alice.post("/assistant/grants", json={"capability": "files.read", "scope": str(tmp_path), "for_user": "alice"})
    assert answer.status_code == 403
    assert assistant.broker.grants("alice") == []


def test_a_grant_the_broker_refuses_says_why(assistant, owner):
    assistant.broker.register(Capability("door.open", "x", frozenset({Effect.PHYSICAL})), lambda a, c: "open")
    for body, words in (
        ({"capability": "door.open"}, "no standing grant"),
        ({"capability": "nothing.here"}, "not a registered capability"),
        ({"capability": "files.read"}, "needs a scope"),
    ):
        answer = owner.post("/assistant/grants", json=body)
        assert answer.status_code == 400 and words in answer.json()["detail"]


def test_any_key_can_read_what_a_grant_can_name(alice):
    rows = {row["name"]: row for row in alice.get("/assistant/capabilities").json()["capabilities"]}
    assert rows["files.read"]["effects"] == ["read"] and rows["files.read"]["scope"] == "path"


# Approvals


def waiting(assistant, user: str = "alice") -> str:
    return assistant.broker.request_approval(user, "files.read", {"path": "/x", "api_key": "hunter2"}, "task:t1", reason="asked")


def test_decided_approvals_can_be_read_by_their_person_with_secrets_hidden(assistant, alice, bob, owner):
    approval = waiting(assistant)
    assert owner.post(f"/assistant/approvals/{approval}", json={"approve": True}).status_code == 404  # not the owner's to decide
    assert alice.post(f"/assistant/approvals/{approval}", json={"approve": False, "by": "peer:evil"}).json()["decided"] is True
    row = alice.get("/assistant/approvals", params={"decided": True}).json()["approvals"][0]
    assert (row["approval_id"], row["status"], row["decided_by"]) == (approval, "denied", "alice")
    assert row["arguments"]["api_key"] == "***"
    assert bob.get("/assistant/approvals", params={"decided": True}).json()["approvals"] == []
    assert bob.get("/assistant/approvals", params={"user": "alice"}).status_code == 403
    assert owner.get("/assistant/approvals", params={"user": "alice", "decided": True}).json()["approvals"][0]["id"] == approval


def test_what_a_turn_is_waiting_on_keeps_its_old_shape(assistant, alice):
    approval = waiting(assistant)
    row = alice.get("/assistant/approvals").json()["approvals"][0]
    assert row["approval_id"] == approval and row["arguments"]["api_key"] == "hunter2"  # the action they are deciding, as it stands


# Providers and routes


def test_the_owner_adds_tests_and_removes_a_provider(assistant, owner, monkeypatch):
    monkeypatch.setenv("SOME_KEY", "a value")
    with FakeModelServer() as server:
        added = owner.post("/assistant/providers", json={"name": "local", "url": server.url, "reach": "machine", "model": "fake-model", "key_env": "SOME_KEY"})
        assert added.status_code == 200, added.text
        assert added.json()["reach"] == "machine" and added.json()["model"] == "fake-model"
        listed = owner.get("/assistant/providers").json()["providers"]
        assert [row["name"] for row in listed] == ["local"] and "health" not in listed[0]
        assert "SOME_KEY" not in json.dumps(listed)
        tested = owner.post("/assistant/providers/local/test").json()
        assert tested["up"] is True and tested["models"] == ["fake-model"]
        assert owner.get("/assistant/providers", params={"check": True}).json()["providers"][0]["health"]["up"] is True
    assert owner.post("/assistant/providers/local/test").json()["up"] is False
    assert owner.delete("/assistant/providers/local").json() == {"removed": True}
    assert owner.get("/assistant/providers").json()["providers"] == []
    assert [row["subject"] for row in assistant.audit.read() if row["action"] == "provider.remove"] == ["local"]
    assert next(row for row in assistant.audit.read() if row["action"] == "provider.add")["actor"] == "user:owner"


def test_a_provider_is_refused_for_the_same_reasons_as_on_the_command_line(owner):
    url = "http://example.com:8000"
    for body, words in (
        ({"name": "x", "url": url, "reach": "machine"}, "not on this machine"),
        ({"name": "x y", "url": url}, "not a usable name"),
        ({"name": "x", "url": url, "tools": "magic"}, "Tool mode"),
        ({"name": "x", "url": url, "key_env": "A", "key_file": "/b"}, "one of key_env and key_file"),
    ):
        answer = owner.post("/assistant/providers", json=body)
        assert answer.status_code == 400 and words in answer.json()["detail"]
    assert owner.delete("/assistant/providers/nobody").status_code == 404
    assert owner.post("/assistant/providers/nobody/test").status_code == 404


def test_a_provider_a_part_registered_cannot_be_removed_here(assistant, owner):
    assistant.providers.register(FakeProvider("peer:ann"))
    assert owner.delete("/assistant/providers/peer:ann").status_code == 409
    assert [row["name"] for row in owner.get("/assistant/providers").json()["providers"]] == ["peer:ann"]


def test_routes_are_read_by_any_key_and_set_by_the_owner(assistant, owner, alice):
    assistant.providers.register(FakeProvider("a"))
    assert owner.put("/assistant/routes/coding", json={"candidates": ["a/big"]}).json()["routes"]["coding"] == ["a/big"]
    assert alice.get("/assistant/routes").json()["routes"]["coding"] == ["a/big"]
    assert alice.put("/assistant/routes/coding", json={"candidates": ["a"]}).status_code == 403
    assert assistant.router.routes()["coding"] == ["a/big"]
    assert next(row for row in assistant.audit.read() if row["action"] == "route.set")["actor"] == "user:owner"


def test_a_route_is_refused_when_it_could_not_work(owner):
    assert owner.put("/assistant/routes/juggling", json={"candidates": ["a"]}).status_code == 400
    assert owner.put("/assistant/routes/coding", json={"candidates": ["nobody"]}).status_code == 400
    assert owner.put("/assistant/routes/coding", json={"candidates": []}).status_code == 422


# Skills


GREETS = [{"name": "greets", "input": "hi", "checks": [{"kind": "contains", "value": "hello"}]}]


def candidate(assistant, user: str = "alice"):
    return assistant.skills.propose("greeter", "Say hello.", ("files.read",), tests=GREETS, user=user)


def test_a_skill_is_listed_shown_evaluated_and_promoted_only_by_the_owner(assistant, owner, alice):
    version = candidate(assistant)
    assert [row["label"] for row in alice.get("/assistant/skills").json()["skills"]] == ["greeter@1"]
    assert alice.get("/assistant/skills/greeter@1").json()["instructions"] == "Say hello."
    assert owner.get("/assistant/skills").json()["skills"] == []  # skills grow out of one person's tasks

    early = owner.post("/assistant/skills/greeter@1/promote", params={"user": "alice"})
    assert early.status_code == 409 and "has not been evaluated" in early.json()["detail"]

    assistant.router = ScriptedRouter("hello there")
    report = owner.post("/assistant/skills/greeter@1/evaluate", params={"user": "alice"})
    assert report.status_code == 200 and report.json()["eligible"] is True
    assert alice.post("/assistant/skills/greeter@1/promote").status_code == 403
    promoted = owner.post(f"/assistant/skills/{version.id}/promote", params={"user": "alice"})
    assert promoted.status_code == 200 and promoted.json()["state"] == "active"
    assert next(row for row in assistant.audit.read() if row["action"] == "skills.promoted")["actor"] == "owner"
    assert alice.get("/assistant/skills/greeter@1").json()["evaluation"]["report"]["eligible"] is True


def test_an_id_does_not_reach_across_users(assistant, owner, bob):
    """A version can be found by id alone, so the route has to ask whose it is."""
    version = candidate(assistant)
    assert bob.get(f"/assistant/skills/{version.id}").status_code == 404
    assert bob.get("/assistant/skills", params={"user": "alice"}).status_code == 403
    assert owner.get(f"/assistant/skills/{version.id}").status_code == 404  # the owner says whose they mean
    for action in ("evaluate", "promote", "reject"):
        assert owner.post(f"/assistant/skills/{version.id}/{action}").status_code == 404
    assert assistant.skills.version(version.id).state == "candidate"


def test_a_skill_can_be_rejected_with_a_reason(assistant, owner):
    candidate(assistant)
    answer = owner.post("/assistant/skills/greeter@1/reject", params={"user": "alice"}, json={"reason": "Too polite."})
    assert answer.json()["state"] == "rejected" and answer.json()["reason"] == "Too polite."
    assert owner.post("/assistant/skills/greeter@1/reject", params={"user": "alice"}).status_code == 409


def finished_task(assistant, user: str, path: Path) -> str:
    assistant.broker.add_grant(user, "files.read", str(path.parent), "owner")
    plan = {"title": "Read the notes", "goal": "Read the notes", "steps": [{"id": "read", "kind": "tool", "capability": "files.read", "arguments": {"path": str(path)}}]}
    task = assistant.tasks.submit(user, plan)
    assert assistant.tasks.run_until_idle(30)
    return task


def test_reflecting_is_on_a_real_task_of_the_person_named(assistant, owner, bob, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    task = finished_task(assistant, "alice", notes)
    assert bob.post(f"/assistant/tasks/{task}/reflect", json={"draft": False}).status_code == 403
    assert owner.post(f"/assistant/tasks/{task}/reflect", json={"draft": False}).status_code == 404  # not the owner's task
    record = owner.post(f"/assistant/tasks/{task}/reflect", params={"user": "alice"}, json={"draft": False}).json()
    assert (record["task_type"], record["outcome"], record["candidate"]) == ("files", "succeeded", None)


# Plugins


def manifest(folder: Path) -> Path:
    path = folder / "weather.json"
    capability = {"name": "forecast", "summary": "Tomorrow.", "effects": ["network"]}
    path.write_text(json.dumps({"name": "weather", "command": ["python", "weather.py"], "capabilities": [capability]}), encoding="utf-8")
    return path


def test_the_owner_installs_and_removes_a_plugin(assistant, owner, tmp_path):
    answer = owner.post("/assistant/plugins", json={"manifest": str(manifest(tmp_path))})
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"capabilities": ["plugin.weather.forecast"]}
    assert [row["name"] for row in owner.get("/assistant/plugins").json()["plugins"]] == ["weather"]
    assert assistant.broker.grants() == []  # installed, and able to do nothing yet
    assert next(row for row in assistant.audit.read() if row["action"] == "plugin.remove" or row["action"] == "plugin.add")["actor"] == "owner"
    assert owner.delete("/assistant/plugins/weather").json() == {"removed": True}
    assert owner.delete("/assistant/plugins/weather").status_code == 404
    assert assistant.broker.capability("plugin.weather.forecast") is None


def test_a_member_cannot_put_a_program_on_this_machine(assistant, alice, tmp_path):
    assert alice.post("/assistant/plugins", json={"manifest": str(manifest(tmp_path))}).status_code == 403
    assert assistant.plugins.list() == []


def test_a_plugin_that_cannot_be_read_says_so(owner, tmp_path):
    answer = owner.post("/assistant/plugins", json={"manifest": str(tmp_path / "nothing.json")})
    assert answer.status_code == 400 and "Could not read" in answer.json()["detail"]


# The coder


@pytest.fixture
def ready_run(assistant, repo):  # noqa: F811
    assistant.sandbox = ProcessSandbox()
    assistant.router = CodingRouter([FIX_AS_FILE])
    assistant.broker = GitBroker()
    assistant.broker.grants.add(("alice", "git.commit"))
    run = assistant.coder.start(repo, "Fix add so test_add passes", user="alice")
    assert run.state == "ready"
    return run


def test_runs_are_shown_to_their_user(assistant, ready_run, alice, bob, owner):
    assert [row["id"] for row in alice.get("/assistant/code/runs").json()["runs"]] == [ready_run.id]
    assert bob.get("/assistant/code/runs").json()["runs"] == []
    assert bob.get("/assistant/code/runs", params={"user": "alice"}).status_code == 403
    assert bob.get(f"/assistant/code/runs/{ready_run.id}").status_code == 404
    shown = alice.get(f"/assistant/code/runs/{ready_run.id}").json()
    assert "return a + b" in shown["diff"] and shown["models"] and "output" not in shown["tests"][0]
    assert "assert" in alice.get(f"/assistant/code/runs/{ready_run.id}", params={"output": True}).json()["tests"][0]["output"]
    assert owner.get(f"/assistant/code/runs/{ready_run.id}", params={"user": "alice"}).status_code == 200


def test_only_the_owner_commits_a_run_and_in_their_own_name(assistant, ready_run, alice, owner):
    assert alice.post(f"/assistant/code/runs/{ready_run.id}/approve").status_code == 403
    assert assistant.coder.run(ready_run.id).state == "ready"
    assert owner.post(f"/assistant/code/runs/{ready_run.id}/approve").status_code == 404  # the owner says whose they mean
    done = owner.post(f"/assistant/code/runs/{ready_run.id}/approve", params={"user": "alice"}, json={"message": "Fix add", "by": "model:evil"})
    assert done.status_code == 200, done.text
    assert done.json()["state"] == "committed" and done.json()["commit"]
    assert next(row for row in assistant.audit.read() if row["action"] == "coder.committed")["actor"] == "owner"
    again = owner.post(f"/assistant/code/runs/{ready_run.id}/approve", params={"user": "alice"})
    assert again.status_code == 409 and "Only a ready run" in again.json()["detail"]


def test_a_commit_the_broker_holds_says_so_and_commits_nothing(assistant, ready_run, owner):
    assistant.broker.ask.add("git.commit")
    held = owner.post(f"/assistant/code/runs/{ready_run.id}/approve", params={"user": "alice"}).json()
    assert held["state"] == "ready" and held["approval_id"]


def test_a_run_can_be_rejected_by_the_owner(assistant, ready_run, alice, owner):
    assert alice.post(f"/assistant/code/runs/{ready_run.id}/reject").status_code == 403
    gone = owner.post(f"/assistant/code/runs/{ready_run.id}/reject", params={"user": "alice"}, json={"reason": "Not wanted."})
    assert gone.json()["state"] == "rejected" and gone.json()["problem"] == "Not wanted."
    assert not ready_run.tree.exists()


# Peers


def test_two_owners_pair_and_end_it_over_their_gateways(tmp_path):
    with Assistant(tmp_path / "one") as first, Assistant(tmp_path / "two") as second:
        first_key, second_key = Users(first).add("owner"), Users(second).add("owner")
        member_key = Users(first).add("alice")
        with Live(first) as one, Live(second) as two, client(one, first_key) as ann, client(two, second_key) as ben, client(one, member_key) as alice:
            invited = ann.post("/assistant/peers/invite", json={"name": "ben", "address": one.url})
            assert invited.status_code == 200, invited.text
            assert alice.post("/assistant/peers/accept", json={"code": invited.json()["code"]}).status_code == 403
            accepted = ben.post("/assistant/peers/accept", json={"code": invited.json()["code"], "name": "ann", "address": two.url})
            assert accepted.status_code == 200, accepted.text
            assert accepted.json()["state"] != "paired" and accepted.json()["own_fingerprint"] == ben.get("/assistant/peers").json()["fingerprint"]

            listed = ann.get("/assistant/peers").json()
            assert [row["name"] for row in listed["peers"]] == ["ben"]
            wrong = ann.post("/assistant/peers/ben/approve", json={"fingerprint": "0" * 64})
            assert wrong.status_code == 409 and "does not match" in wrong.json()["detail"]
            assert alice.post("/assistant/peers/ben/approve").status_code == 403
            real = ann.post("/assistant/peers/ben/approve", json={"fingerprint": listed["peers"][0]["fingerprint"]})
            assert real.json()["state"] == "paired"
            assert ben.post("/assistant/peers/ann/approve").json()["state"] == "paired"

            assert alice.post("/assistant/peers/ben/revoke").status_code == 403
            assert ann.post("/assistant/peers/ben/revoke").json()["state"] == "revoked"
            assert ann.post("/assistant/peers/nobody/revoke").status_code == 404
            assert ann.post("/assistant/peers/nobody/approve").status_code == 404


def test_an_invite_needs_an_address_and_a_bad_code_is_refused(assistant, owner):
    assert owner.post("/assistant/peers/invite", json={}).status_code == 409
    assert owner.post("/assistant/peers/invite", json={"minutes": 0}).status_code == 422
    assert owner.post("/assistant/peers/accept", json={"code": "not a code"}).status_code == 409


# Planning


def planned(assistant, step: dict) -> ScriptedRouter:
    assistant.router = ScriptedRouter(json.dumps({"title": "Read it", "steps": [step]}))
    return assistant.router


def read_step(path: str, **fields) -> dict:
    return {"id": "a", "kind": "tool", "capability": "files.read", "arguments": {"path": path}, **fields}


def test_a_plan_is_asked_for_by_any_key_and_runs_nothing(assistant, alice, tmp_path):
    router = planned(assistant, read_step(str(tmp_path / "x.txt")))
    assistant.context.set_default_reach("alice", Reach.MACHINE)
    plan = alice.post("/assistant/plan", json={"goal": "Read it"}).json()
    assert [entry["capability"] for entry in plan["missing_grants"]] == ["files.read"]
    assert router.requests[0].reach is Reach.MACHINE  # a person's words go no further than theirs may
    assert assistant.tasks.list_tasks("alice") == []


def test_a_plan_nothing_can_write_says_why(alice):
    answer = alice.post("/assistant/plan", json={"goal": "Read it"})
    assert answer.status_code == 503 and "No model" in answer.json()["detail"]


def test_one_yes_from_the_owner_allows_what_a_plan_needs_and_no_more(assistant, owner, alice, tmp_path):
    planned(assistant, read_step(str(tmp_path / "x.txt"), retries=1))
    plan = alice.post("/assistant/plan", json={"goal": "Read it"}).json()
    assert alice.post("/assistant/plan/allow", json={"plan": plan}).status_code == 403
    assert assistant.broker.grants("alice") == []
    answer = owner.post("/assistant/plan/allow", json={"plan": plan, "for_user": "alice"})
    assert answer.status_code == 200, answer.text
    grant = assistant.broker.grants("alice")[0]
    assert answer.json() == {"granted": [grant["id"]], "left_for_a_person": []}
    assert (grant["uses_left"], grant["granted_by"], grant["scope"]) == (2, "owner", str(tmp_path / "x.txt"))
    assert grant["expires_at"] is not None


def test_a_plan_cannot_ask_for_more_than_its_steps_do(assistant, owner, tmp_path):
    """Whoever sends a plan writes its list of what is missing, so that list is not believed."""
    wider = {"capability": "shell.run", "scope": "*", "uses": 1000, "steps": ["a"]}
    plan = {"title": "x", "steps": [read_step(str(tmp_path / "x.txt"))], "missing_grants": [wider]}
    assert owner.post("/assistant/plan/allow", json={"plan": plan}).status_code == 200
    assert [row["capability"] for row in assistant.broker.grants("owner")] == ["files.read"]


def test_a_plan_that_could_not_run_is_refused(assistant, owner):
    answer = owner.post("/assistant/plan/allow", json={"plan": {"steps": [{"id": "a", "kind": "tool", "capability": "nothing.here", "arguments": {}}]}})
    assert answer.status_code == 400 and "not a capability" in answer.json()["detail"]
    assert assistant.broker.grants() == []


# Who the assistant is


def test_the_history_of_who_the_assistant_is_can_be_read(assistant, alice):
    assistant.identity.set("owner", name="Ada")
    rows = alice.get("/assistant/identity/history").json()["history"]
    assert [(row["key"], row["new"]) for row in rows] == [("name", "Ada")]


def test_a_relationship_history_belongs_to_its_person(assistant, owner, alice, bob):
    assistant.identity.set_relationship("alice", "New here.", "owner")
    assistant.identity.set_relationship("alice", "Trusted.", "owner")
    assert [row["note"] for row in alice.get("/assistant/identity/relationship").json()["history"]] == ["Trusted.", "New here."]
    assert [row["note"] for row in alice.get("/assistant/identity/relationship", params={"limit": 1}).json()["history"]] == ["Trusted."]
    assert bob.get("/assistant/identity/relationship").json()["history"] == []
    assert bob.get("/assistant/identity/relationship", params={"user": "alice"}).status_code == 403
    assert len(owner.get("/assistant/identity/relationship", params={"user": "alice"}).json()["history"]) == 2


# A conversation's tasks


def test_a_conversation_lists_the_tasks_it_handed_off(assistant, alice, bob):
    assistant.router = ScriptedRouter(json.dumps({"title": "Check", "steps": [{"id": "a", "kind": "wait", "seconds": 1}]}), skip_workers=True)
    said = alice.post("/assistant/say", json={"text": "/task check the backups", "conversation_id": "work"}).json()
    assert said["status"] == "task"
    rows = alice.get("/assistant/conversations/work/tasks").json()["tasks"]
    assert [(row["task_id"], row["turn_id"], row["title"], row["status"]) for row in rows] == [(said["task"], said["turn"], "Check", "pending")]
    assert alice.get("/assistant/conversations/other/tasks").json()["tasks"] == []
    assert bob.get("/assistant/conversations/work/tasks").json()["tasks"] == []


# Without a part


@pytest.mark.parametrize("path", ["/assistant/skills", "/assistant/code/runs", "/assistant/peers", "/assistant/plugins"])
def test_a_part_that_is_not_there_says_so(assistant, owner, path):
    name = {"skills": "skills", "code": "coder", "peers": "federation", "plugins": "plugins"}[path.split("/")[2]]
    saved = getattr(assistant, name)
    delattr(assistant, name)
    try:
        answer = owner.get(path)
    finally:
        setattr(assistant, name, saved)
    assert answer.status_code == 404 and "not installed" in answer.json()["detail"]


def test_a_reply_never_shows_where_a_providers_key_is_kept(assistant, owner):
    assistant.providers.add("cloudy", "http://127.0.0.1:9", reach="machine", key="file:/secret/place.key")
    assert "place.key" not in owner.get("/assistant/providers").text


def test_nothing_here_needs_a_person_to_trust_a_request_for_their_name():
    source = (Path(manage_http.__file__)).read_text(encoding="utf-8")
    assert "body.by" not in source and "by: str" not in source

