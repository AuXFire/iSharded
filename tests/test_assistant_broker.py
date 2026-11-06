"""Each rule of the broker, on a real database and a real file system."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from shardmesh.assistant import broker as broker_part
from shardmesh.assistant import cli
from shardmesh.assistant.app import Assistant
from shardmesh.assistant.broker import argument_hash, path_within, redacted, scope_kind, url_host
from shardmesh.assistant.contracts import (
    CallContext,
    Capability,
    Effect,
    Provenance,
    ToolCall,
    ToolResult,
    Trust,
    Verdict,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def assistant(tmp_path, clock):
    with Assistant(tmp_path / "home", clock=clock) as made:
        yield made


@pytest.fixture
def broker(assistant):
    return assistant.broker


def context(user="ann", actor="conversation", tainted=False, task_id=None):
    return CallContext(user=user, actor=actor, tainted=tainted, task_id=task_id)


def make(broker, name="demo.read", effects=(Effect.READ,), scope_field=None, handler=None, **fields):
    calls = []

    def record(arguments, call_context):
        calls.append((arguments, call_context))
        return handler(arguments, call_context) if handler else "done"

    broker.register(Capability(name=name, summary="demo", effects=frozenset(effects), scope_field=scope_field, **fields), record)
    return calls


def call(name="demo.read", **arguments):
    return ToolCall(name, arguments)


def test_an_unregistered_capability_is_denied(broker):
    decision = broker.check(call("nothing.here"), context())
    assert decision.verdict is Verdict.DENY
    assert not broker.invoke(call("nothing.here"), context()).ok


def test_a_registered_capability_with_no_grant_is_denied_and_never_runs(broker):
    calls = make(broker)
    result = broker.invoke(call(), context())
    assert not result.ok and result.data["verdict"] == "deny"
    assert "grant add demo.read" in result.output  # says what a person can do about it
    assert calls == []


def test_a_grant_is_for_one_user_and_one_capability(broker):
    calls = make(broker)
    make(broker, "demo.other")
    broker.add_grant("ann", "demo.read", None, "ann")
    assert broker.invoke(call(), context("ann")).ok
    assert broker.check(call(), context("bob")).verdict is Verdict.DENY
    assert broker.check(call("demo.other"), context("ann")).verdict is Verdict.DENY
    assert len(calls) == 1


def test_a_capability_must_declare_what_it_does(broker):
    with pytest.raises(ValueError, match="declares no effects"):
        broker.register(Capability(name="demo.vague", summary="?", effects=frozenset()), lambda a, c: "x")


def test_a_tool_cannot_replace_a_registered_one(broker):
    make(broker)
    with pytest.raises(ValueError, match="already registered"):
        make(broker)


# Scope: paths.


@pytest.fixture
def scoped(broker, tmp_path):
    allowed = tmp_path / "work"
    allowed.mkdir()
    (tmp_path / "work-private").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("x", encoding="utf-8")
    calls = make(broker, "demo.file", scope_field="path")
    broker.add_grant("ann", "demo.file", str(allowed), "ann")
    return allowed, calls


def test_a_path_inside_the_scope_is_allowed(broker, scoped):
    allowed, calls = scoped
    assert broker.invoke(call("demo.file", path=str(allowed / "a" / "b.txt")), context()).ok
    assert broker.invoke(call("demo.file", path=str(allowed)), context()).ok
    assert len(calls) == 2


def test_dots_cannot_climb_out_of_a_path_scope(broker, scoped):
    allowed, calls = scoped
    escape = os.path.join(str(allowed), "..", "outside", "secret.txt")
    assert broker.check(call("demo.file", path=escape), context()).verdict is Verdict.DENY


def test_a_sibling_with_the_same_prefix_is_outside_the_scope(broker, scoped):
    allowed, _ = scoped
    assert broker.check(call("demo.file", path=str(allowed) + "-private"), context()).verdict is Verdict.DENY


def link_directory(link: Path, target: Path) -> None:
    """A real link, using a junction where Windows refuses symlinks to ordinary users."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def test_a_link_cannot_lead_out_of_a_path_scope(broker, scoped, tmp_path):
    allowed, _ = scoped
    link_directory(allowed / "door", tmp_path / "outside")
    escape = str(allowed / "door" / "secret.txt")
    assert broker.check(call("demo.file", path=escape), context()).verdict is Verdict.DENY


def test_a_link_that_stays_inside_the_scope_is_fine(broker, scoped):
    allowed, _ = scoped
    (allowed / "real").mkdir()
    link_directory(allowed / "alias", allowed / "real")
    assert broker.check(call("demo.file", path=str(allowed / "alias" / "x.txt")), context()).verdict is Verdict.ALLOW


def test_the_handler_gets_the_path_that_was_judged(broker, scoped):
    allowed, calls = scoped
    broker.invoke(call("demo.file", path=os.path.join(str(allowed), "sub", "..", "x.txt")), context())
    assert calls[0][0]["path"] == os.path.realpath(allowed / "x.txt")


def test_a_relative_path_is_refused_not_guessed(broker, scoped):
    decision = broker.check(call("demo.file", path="notes.txt"), context())
    assert decision.verdict is Verdict.DENY and "absolute" in decision.reason


def test_a_relative_path_is_refused_even_under_a_grant_for_everything(broker):
    make(broker, "demo.any", scope_field="path")
    broker.add_grant("ann", "demo.any", "*", "ann")
    assert broker.check(call("demo.any", path="../up.txt"), context()).verdict is Verdict.DENY


def test_a_narrow_grant_does_not_cover_a_missing_argument(broker, scoped):
    assert broker.check(call("demo.file"), context()).verdict is Verdict.DENY


def test_every_path_in_a_list_must_be_inside_the_scope(broker, scoped, tmp_path):
    allowed, _ = scoped
    good, bad = str(allowed / "a"), str(tmp_path / "outside" / "b")
    assert broker.check(call("demo.file", path=[good, good]), context()).verdict is Verdict.ALLOW
    assert broker.check(call("demo.file", path=[good, bad]), context()).verdict is Verdict.DENY
    assert broker.check(call("demo.file", path=[]), context()).verdict is Verdict.DENY


def test_a_grant_keeps_the_place_it_named_when_a_link_is_moved_later(broker, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    link = tmp_path / "moving"
    link_directory(link, first)
    make(broker, "demo.linked", scope_field="path")
    broker.add_grant("ann", "demo.linked", str(link), "ann")
    os.rmdir(link) if sys.platform == "win32" else os.unlink(link)
    link_directory(link, second)
    assert broker.check(call("demo.linked", path=str(link / "x")), context()).verdict is Verdict.DENY


def test_path_comparison_helper():
    root = os.path.abspath(os.sep + "work")
    assert path_within(root, root)
    assert path_within(os.path.join(root, "a"), root)
    assert not path_within(root + "2", root)
    assert path_within(os.path.join(root, "a"), os.path.abspath(os.sep))


# Scope: URLs.


@pytest.fixture
def web(broker):
    calls = make(broker, "demo.fetch", effects=(Effect.READ,), scope_field="url")
    broker.add_grant("ann", "demo.fetch", "example.com", "ann")
    return calls


@pytest.mark.parametrize("url", ["https://example.com/a", "http://EXAMPLE.com:8443/b?c=d", "https://example.com./"])
def test_a_url_is_compared_by_host(broker, web, url):
    assert broker.check(call("demo.fetch", url=url), context()).verdict is Verdict.ALLOW


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.com/example.com",
        "https://example.com.evil.com/",
        "https://example.com@evil.com/",
        "https://evil.com#@example.com/",
        "https://evil.com?example.com",
        "https://example.com\\@evil.com/",
        "https://sub.example.com/",
        "not a url",
        "",
    ],
)
def test_a_url_that_only_looks_right_is_denied(broker, web, url):
    assert broker.check(call("demo.fetch", url=url), context()).verdict is Verdict.DENY


def test_a_host_wildcard_covers_subdomains_only(broker):
    make(broker, "demo.sub", scope_field="url")
    broker.add_grant("ann", "demo.sub", "*.example.com", "ann")
    assert broker.check(call("demo.sub", url="https://a.example.com/"), context()).verdict is Verdict.ALLOW
    assert broker.check(call("demo.sub", url="https://example.com/"), context()).verdict is Verdict.DENY


def test_a_scope_with_a_port_covers_that_port_only(broker):
    make(broker, "demo.port", scope_field="url")
    broker.add_grant("ann", "demo.port", "localhost:9000", "ann")
    assert broker.check(call("demo.port", url="http://localhost:9000/"), context()).verdict is Verdict.ALLOW
    assert broker.check(call("demo.port", url="http://localhost:9001/"), context()).verdict is Verdict.DENY


def test_names_host_needs_the_host_written_out(broker):
    make(broker, "demo.lan", scope_field="url")
    broker.add_grant("ann", "demo.lan", "*.lan", "ann")
    broker.add_grant("ann", "demo.lan", "192.168.1.5", "ann")
    assert broker.names_host("ann", "demo.lan", "http://192.168.1.5/x")
    assert not broker.names_host("ann", "demo.lan", "http://printer.lan/")
    assert not broker.names_host("bob", "demo.lan", "http://192.168.1.5/x")


def test_url_host_refuses_what_it_cannot_read_plainly():
    assert url_host("http://[::1") is None
    assert url_host("http://exa mple.com/") is None
    assert url_host(7) is None
    assert url_host("https://Example.COM:444/x") == ("example.com", 444)


# Scope: everything else.


def test_other_scopes_are_globs(broker):
    make(broker, "demo.tag", scope_field="topic")
    broker.add_grant("ann", "demo.tag", "project-*", "ann")
    assert broker.check(call("demo.tag", topic="project-a"), context()).verdict is Verdict.ALLOW
    assert broker.check(call("demo.tag", topic="Project-a"), context()).verdict is Verdict.DENY
    assert broker.check(call("demo.tag", topic="other"), context()).verdict is Verdict.DENY
    assert broker.check(call("demo.tag", topic={"not": "text"}), context()).verdict is Verdict.DENY


def test_scope_kind_trusts_a_declared_format_over_the_name():
    plain = Capability("a.b", "", frozenset({Effect.READ}), scope_field="target")
    declared = Capability(
        "a.c", "", frozenset({Effect.READ}), scope_field="target", input_schema={"properties": {"target": {"format": "uri"}}}
    )
    assert scope_kind(plain) == "glob"
    assert scope_kind(declared) == "url"


def test_a_capability_with_no_scope_field_takes_no_scope(broker):
    make(broker)
    with pytest.raises(ValueError, match="nothing to scope"):
        broker.add_grant("ann", "demo.read", "something", "ann")


def test_a_scoped_capability_needs_a_scope_stated(broker):
    make(broker, "demo.file", scope_field="path")
    with pytest.raises(ValueError, match="needs a scope"):
        broker.add_grant("ann", "demo.file", None, "ann")


def test_arguments_are_checked_against_the_declared_schema(broker):
    calls = make(
        broker,
        "demo.typed",
        input_schema={"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}, "additionalProperties": False},
    )
    broker.add_grant("ann", "demo.typed", None, "ann")
    assert broker.check(call("demo.typed"), context()).verdict is Verdict.DENY
    assert broker.check(call("demo.typed", n="1"), context()).verdict is Verdict.DENY
    assert broker.check(call("demo.typed", n=1, extra=True), context()).verdict is Verdict.DENY
    assert broker.invoke(call("demo.typed", n=1), context()).ok
    assert len(calls) == 1


def test_arguments_that_are_not_json_are_denied_without_raising(broker):
    make(broker)
    broker.add_grant("ann", "demo.read", None, "ann")
    assert broker.check(ToolCall("demo.read", {"x": object()}), context()).verdict is Verdict.DENY
    assert broker.check(ToolCall("demo.read", "not a dict"), context()).verdict is Verdict.DENY  # type: ignore[arg-type]


# Taint.


def test_read_runs_under_a_grant_even_when_tainted(broker):
    calls = make(broker)
    broker.add_grant("ann", "demo.read", None, "ann")
    assert broker.invoke(call(), context(tainted=True)).ok
    assert len(calls) == 1


@pytest.mark.parametrize("effect", [Effect.WRITE_LOCAL, Effect.EXECUTE, Effect.NETWORK, Effect.EXTERNAL])
def test_a_tainted_context_never_uses_a_standing_grant_for_anything_consequential(broker, effect):
    calls = make(broker, "demo.act", effects=(effect,))
    broker.add_grant("ann", "demo.act", None, "ann")
    decision = broker.check(call("demo.act"), context(tainted=True))
    assert decision.verdict is Verdict.ASK and decision.approval_id
    result = broker.invoke(call("demo.act"), context(tainted=True))
    assert not result.ok and result.data["verdict"] == "ask"
    assert calls == []
    assert broker.check(call("demo.act"), context()).verdict is Verdict.ALLOW  # the grant is still fine untainted


def test_a_tainted_call_that_nothing_covers_is_denied_not_asked(broker):
    make(broker, "demo.act", effects=(Effect.EXECUTE,))
    assert broker.check(call("demo.act"), context(tainted=True)).verdict is Verdict.DENY
    assert broker.approvals() == []


def test_a_tainted_call_outside_the_scope_is_denied_not_asked(broker, tmp_path):
    make(broker, "demo.put", effects=(Effect.WRITE_LOCAL,), scope_field="path")
    broker.add_grant("ann", "demo.put", str(tmp_path / "a"), "ann")
    decision = broker.check(call("demo.put", path=str(tmp_path / "b")), context(tainted=True))
    assert decision.verdict is Verdict.DENY


def test_a_mixed_read_and_write_capability_counts_as_consequential(broker):
    make(broker, "demo.both", effects=(Effect.READ, Effect.WRITE_LOCAL))
    broker.add_grant("ann", "demo.both", None, "ann")
    assert broker.check(call("demo.both"), context(tainted=True)).verdict is Verdict.ASK


# Approvals.


def approved(broker, decision, by="ann"):
    assert broker.decide(decision.approval_id, True, by)


def test_an_approval_works_exactly_once(broker):
    calls = make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    the_call = call("demo.act", x=1)
    approved(broker, broker.check(the_call, context(tainted=True)))
    assert broker.invoke(the_call, context(tainted=True)).ok
    second = broker.invoke(the_call, context(tainted=True))
    assert not second.ok and second.data["verdict"] == "ask"
    assert len(calls) == 1


def test_changed_arguments_need_a_new_approval(broker):
    calls = make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    approved(broker, broker.check(call("demo.act", x=1), context(tainted=True)))
    changed = broker.invoke(call("demo.act", x=2), context(tainted=True))
    assert not changed.ok and changed.data["verdict"] == "ask"
    assert calls == []


def test_argument_order_does_not_change_the_action():
    assert argument_hash({"a": 1, "b": 2}) == argument_hash({"b": 2, "a": 1})
    assert argument_hash({"a": 1}) != argument_hash({"a": 2})


def test_an_approval_is_not_for_another_actor_or_task_or_user(broker):
    make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    broker.add_grant("bob", "demo.act", None, "bob")
    the_call = call("demo.act")
    approved(broker, broker.check(the_call, context(actor="task:1", task_id="1", tainted=True)))
    for other in (
        context(actor="task:2", task_id="2", tainted=True),
        context(actor="conversation", tainted=True),
        context(user="bob", actor="task:1", task_id="1", tainted=True),
    ):
        assert broker.check(the_call, other).verdict is Verdict.ASK


def test_moving_a_link_after_an_approval_makes_it_a_different_action(broker, tmp_path):
    allowed, elsewhere = tmp_path / "allowed", tmp_path / "elsewhere"
    allowed.mkdir()
    elsewhere.mkdir()
    link = allowed / "out"
    link_directory(link, allowed)
    calls = make(broker, "demo.put", effects=(Effect.WRITE_LOCAL,), scope_field="path")
    broker.add_grant("ann", "demo.put", str(allowed), "ann")
    the_call = call("demo.put", path=str(link / "note.txt"))
    approved(broker, broker.check(the_call, context(tainted=True)))
    os.rmdir(link) if sys.platform == "win32" else os.unlink(link)
    link_directory(link, elsewhere)
    assert not broker.invoke(the_call, context(tainted=True)).ok
    assert calls == [], "a yes given for one place was spent on another"


def test_a_person_is_shown_where_a_path_really_leads(broker, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    make(broker, "demo.put", effects=(Effect.WRITE_LOCAL,), scope_field="path")
    broker.add_grant("ann", "demo.put", str(allowed), "ann")
    written = os.path.join(str(allowed), "sub", "..", "note.txt")
    decision = broker.check(call("demo.put", path=written), context(tainted=True))
    assert broker.approval(decision.approval_id)["arguments"]["path"] == os.path.realpath(allowed / "note.txt")


def test_asking_twice_is_one_question(broker):
    make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    first = broker.check(call("demo.act"), context(tainted=True))
    second = broker.check(call("demo.act"), context(tainted=True))
    assert first.approval_id == second.approval_id
    assert len(broker.approvals()) == 1


def test_a_denied_approval_allows_nothing(broker):
    calls = make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    decision = broker.check(call("demo.act"), context(tainted=True))
    assert broker.decide(decision.approval_id, False, "ann")
    assert not broker.invoke(call("demo.act"), context(tainted=True)).ok
    assert calls == []


def test_no_can_withdraw_a_yes_that_was_not_used(broker):
    calls = make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    decision = broker.check(call("demo.act"), context(tainted=True))
    approved(broker, decision)
    assert broker.decide(decision.approval_id, False, "ann")
    assert not broker.invoke(call("demo.act"), context(tainted=True)).ok
    assert calls == []


def test_a_used_approval_cannot_be_decided_again(broker):
    make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    decision = broker.check(call("demo.act"), context(tainted=True))
    approved(broker, decision)
    broker.invoke(call("demo.act"), context(tainted=True))
    assert not broker.decide(decision.approval_id, True, "ann")
    assert not broker.decide(decision.approval_id, False, "ann")


def test_one_approval_cannot_be_spent_twice_at_the_same_moment(broker):
    calls = make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    approved(broker, broker.check(call("demo.act"), context(tainted=True)))
    results = []

    def go():
        results.append(broker.invoke(call("demo.act"), context(tainted=True)).ok)
        broker.store.close()

    threads = [threading.Thread(target=go) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == [False] * 7 + [True]
    assert len(calls) == 1


@pytest.mark.parametrize("who", ["skill:tidy", "peer:bob", "task:1", "conversation", "model:x", "plugin:y", ""])
def test_only_a_person_can_decide_or_grant(broker, who):
    make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    decision = broker.check(call("demo.act"), context(tainted=True))
    with pytest.raises(ValueError, match="not a person"):
        broker.decide(decision.approval_id, True, who)
    with pytest.raises(ValueError, match="not a person"):
        broker.add_grant("ann", "demo.act", None, who)
    assert broker.approval(decision.approval_id)["status"] == "pending"


def test_a_decision_can_be_limited_to_the_users_own_approvals(broker):
    make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    decision = broker.check(call("demo.act"), context(tainted=True))
    assert not broker.decide(decision.approval_id, True, "bob", user="bob")
    assert broker.decide(decision.approval_id, True, "ann", user="ann")


# Physical.


def test_physical_always_asks_and_can_have_no_grant(broker):
    calls = make(broker, "door.open", effects=(Effect.PHYSICAL,))
    with pytest.raises(ValueError, match="no standing grant"):
        broker.add_grant("ann", "door.open", None, "ann")
    decision = broker.check(call("door.open"), context())
    assert decision.verdict is Verdict.ASK
    assert not broker.invoke(call("door.open"), context()).ok
    assert calls == []


def test_physical_runs_once_per_approval(broker):
    calls = make(broker, "door.open", effects=(Effect.PHYSICAL,))
    approved(broker, broker.check(call("door.open"), context()))
    assert broker.invoke(call("door.open"), context()).ok
    assert broker.check(call("door.open"), context()).verdict is Verdict.ASK
    assert len(calls) == 1


def test_a_grant_row_written_some_other_way_still_cannot_move_hardware(broker, assistant, clock):
    calls = make(broker, "door.open", effects=(Effect.PHYSICAL,))
    assistant.store.execute(
        "INSERT INTO grants (id, user, capability, scope, granted_by, granted_at) VALUES ('g-x', 'ann', 'door.open', '*', 'ann', ?)",
        (clock.now,),
    )
    assert broker.check(call("door.open"), context()).verdict is Verdict.ASK
    assert calls == []


# Peers and skills.


def test_a_peer_is_tainted_whatever_its_context_says(broker):
    calls = make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    assert broker.check(call("demo.act"), context(actor="peer:bob", tainted=False)).verdict is Verdict.ASK
    assert calls == []
    audit = [row for row in broker.audit.read() if row["action"] == "broker.check"]
    assert audit[-1]["detail"]["tainted"] is True


def test_a_peer_can_still_read_what_a_grant_covers(broker):
    make(broker)
    broker.add_grant("ann", "demo.read", None, "ann")
    assert broker.invoke(call(), context(actor="peer:bob")).ok


def test_a_skill_gets_what_a_grant_covers(broker):
    make(broker)
    broker.add_grant("ann", "demo.read", None, "ann")
    assert broker.invoke(call(), context(actor="skill:tidy")).ok


def test_a_skill_gets_nothing_more_and_cannot_even_ask(broker):
    make(broker, "demo.act", effects=(Effect.EXECUTE,))
    assert broker.check(call("demo.act"), context(actor="skill:tidy")).verdict is Verdict.DENY
    assert broker.check(call("demo.act"), context(actor="skill:tidy", tainted=True)).verdict is Verdict.DENY
    make(broker, "door.open", effects=(Effect.PHYSICAL,))
    assert broker.check(call("door.open"), context(actor="skill:tidy")).verdict is Verdict.DENY
    assert broker.approvals() == []


def test_a_skill_with_a_grant_still_cannot_act_tainted_without_a_person(broker):
    make(broker, "demo.act", effects=(Effect.EXECUTE,))
    broker.add_grant("ann", "demo.act", None, "ann")
    assert broker.check(call("demo.act"), context(actor="skill:tidy", tainted=True)).verdict is Verdict.ASK


def test_a_skill_loses_an_unspent_approval_when_the_grant_is_revoked(broker):
    calls = make(broker, "demo.act", effects=(Effect.EXECUTE,))
    grant = broker.add_grant("ann", "demo.act", None, "ann")
    skill = context(actor="skill:tidy", tainted=True)
    approved(broker, broker.check(call("demo.act"), skill))
    broker.revoke_grant(grant, "ann")
    assert not broker.invoke(call("demo.act"), skill).ok
    assert calls == []


# Limits take effect at once.


def test_expiry_takes_effect_at_once(broker, clock):
    calls = make(broker)
    broker.add_grant("ann", "demo.read", None, "ann", expires_at=clock.now + 10)
    assert broker.invoke(call(), context()).ok
    clock.now += 10
    assert not broker.invoke(call(), context()).ok
    assert len(calls) == 1


def test_a_use_count_runs_out(broker):
    calls = make(broker)
    broker.add_grant("ann", "demo.read", None, "ann", uses=2)
    assert [broker.invoke(call(), context()).ok for _ in range(3)] == [True, True, False]
    assert len(calls) == 2


def test_check_spends_nothing(broker):
    make(broker)
    broker.add_grant("ann", "demo.read", None, "ann", uses=1)
    for _ in range(3):
        assert broker.check(call(), context()).verdict is Verdict.ALLOW
    assert broker.invoke(call(), context()).ok


def test_a_limited_grant_is_kept_when_an_unlimited_one_also_covers_the_call(broker):
    make(broker)
    limited = broker.add_grant("ann", "demo.read", None, "ann", uses=1)
    broker.add_grant("ann", "demo.read", None, "ann")
    broker.invoke(call(), context())
    assert {row["id"]: row["uses_left"] for row in broker.grants("ann")}[limited] == 1


def test_a_use_cannot_be_spent_twice_at_the_same_moment(broker):
    calls = make(broker)
    broker.add_grant("ann", "demo.read", None, "ann", uses=3)
    results = []

    def go():
        results.append(broker.invoke(call(), context()).ok)
        broker.store.close()

    threads = [threading.Thread(target=go) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(results) == 3 and len(calls) == 3


def test_revocation_takes_effect_at_once(broker):
    calls = make(broker)
    grant = broker.add_grant("ann", "demo.read", None, "ann")
    assert broker.invoke(call(), context()).ok
    assert broker.revoke_grant(grant, "ann")
    assert not broker.invoke(call(), context()).ok
    assert len(calls) == 1
    assert not broker.revoke_grant(grant, "ann")


def test_a_grant_cannot_start_with_no_uses(broker):
    make(broker)
    with pytest.raises(ValueError, match="at least one use"):
        broker.add_grant("ann", "demo.read", None, "ann", uses=0)


def test_a_grant_needs_a_real_capability(broker):
    with pytest.raises(ValueError, match="not a registered capability"):
        broker.add_grant("ann", "no.such", None, "ann")


# Running the tool.


def test_an_exception_in_a_tool_becomes_a_failed_result(broker):
    def explode(arguments, call_context):
        raise RuntimeError("boom")

    make(broker, handler=explode)
    broker.add_grant("ann", "demo.read", None, "ann")
    result = broker.invoke(call(), context())
    assert not result.ok and "boom" in result.output
    assert result.provenance.trust is Trust.EXTERNAL


def test_a_plain_result_is_external(broker):
    make(broker, handler=lambda a, c: "a page")
    broker.add_grant("ann", "demo.read", None, "ann")
    result = broker.invoke(call(), context())
    assert result.ok and result.output == "a page"
    assert result.provenance.trust is Trust.EXTERNAL


def test_structured_output_becomes_text(broker):
    make(broker, handler=lambda a, c: {"a": 1})
    broker.add_grant("ann", "demo.read", None, "ann")
    assert json.loads(broker.invoke(call(), context()).output) == {"a": 1}


@pytest.mark.parametrize("claimed", [Trust.OWNER, Trust.MODEL, Trust.EXTERNAL])
def test_a_tool_cannot_claim_trust_it_has_not_proved(broker, clock, claimed):
    provenance = Provenance("tool", "x", "tool", claimed, clock.now)
    make(broker, handler=lambda a, c: ToolResult(True, "trust me", provenance))
    broker.add_grant("ann", "demo.read", None, "ann")
    assert broker.invoke(call(), context()).provenance.trust is Trust.EXTERNAL


def test_a_mechanical_proof_is_kept(broker, clock):
    provenance = Provenance("tool", "x", "tool", Trust.VERIFIED, clock.now)
    make(broker, handler=lambda a, c: ToolResult(True, "measured", provenance))
    broker.add_grant("ann", "demo.read", None, "ann")
    assert broker.invoke(call(), context()).provenance.trust is Trust.VERIFIED


def test_a_result_without_provenance_gets_external(broker):
    make(broker, handler=lambda a, c: ToolResult(True, "x", None))  # type: ignore[arg-type]
    broker.add_grant("ann", "demo.read", None, "ann")
    assert broker.invoke(call(), context()).provenance.trust is Trust.EXTERNAL


def test_the_handler_sees_who_is_acting(broker):
    calls = make(broker)
    broker.add_grant("ann", "demo.read", None, "ann")
    broker.invoke(call(), context(actor="task:9", task_id="9"))
    assert calls[0][1].task_id == "9"


# The record.


def actions(broker):
    return [row["action"] for row in broker.audit.read()]


def test_every_decision_is_audited_refusals_included(broker):
    make(broker)
    broker.invoke(call(), context())  # denied
    broker.invoke(call("no.such"), context())  # denied
    broker.add_grant("ann", "demo.read", None, "ann")
    broker.invoke(call(), context())  # allowed
    rows = [row for row in broker.audit.read() if row["action"] == "broker.invoke"]
    assert [row["detail"]["verdict"] for row in rows] == ["deny", "deny", "allow"]
    assert rows[1]["subject"] == "no.such"
    assert "broker.result" in actions(broker)
    assert broker.audit.verify() is None


def test_the_record_holds_the_hash_and_never_the_values(broker):
    def fail(arguments, call_context):
        raise RuntimeError("password was hunter2")

    make(broker, handler=fail)
    broker.add_grant("ann", "demo.read", None, "ann")
    arguments = {"password": "hunter2", "note": "sk-very-secret"}
    broker.invoke(ToolCall("demo.read", arguments), context())
    broker.check(ToolCall("demo.read", arguments), context())
    record = json.dumps(broker.audit.read())
    assert "hunter2" not in record and "sk-very-secret" not in record
    assert argument_hash(arguments) in record


def test_a_person_sees_the_action_with_secret_looking_names_hidden():
    shown = redacted({"path": "/x", "api_key": "abc", "Password": "p", "token": "t"})
    assert shown == {"path": "/x", "api_key": "***", "Password": "***", "token": "***"}


def test_grant_and_approval_changes_are_audited(broker):
    make(broker, "demo.act", effects=(Effect.EXECUTE,))
    grant = broker.add_grant("ann", "demo.act", None, "ann")
    decision = broker.check(call("demo.act"), context(tainted=True))
    broker.decide(decision.approval_id, True, "ann")
    broker.revoke_grant(grant, "ann")
    assert {"grant.add", "grant.revoke", "approval.request", "approval.approve"} <= set(actions(broker))


def test_a_consumed_use_and_its_record_commit_together(broker, assistant, monkeypatch):
    make(broker)
    grant = broker.add_grant("ann", "demo.read", None, "ann", uses=1)

    def fail(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(assistant.audit, "record", fail)
    with pytest.raises(RuntimeError):
        broker.invoke(call(), context())
    monkeypatch.undo()
    assert next(row for row in broker.grants("ann") if row["id"] == grant)["uses_left"] == 1


def test_decisions_survive_a_restart(tmp_path, clock):
    home = tmp_path / "home"
    with Assistant(home, clock=clock) as first:
        make(first.broker, "demo.act", effects=(Effect.EXECUTE,))
        first.broker.add_grant("ann", "demo.act", None, "ann", uses=2)
        first.broker.invoke(call("demo.act"), context())
        pending = first.broker.check(call("demo.act"), context(tainted=True)).approval_id
    with Assistant(home, clock=clock) as second:
        make(second.broker, "demo.act", effects=(Effect.EXECUTE,))
        assert second.broker.grants("ann")[0]["uses_left"] == 1
        assert second.broker.approval(pending)["status"] == "pending"


def test_users_listings_are_separate(broker):
    make(broker)
    broker.add_grant("ann", "demo.read", None, "ann")
    broker.add_grant("bob", "demo.read", None, "bob")
    assert {row["user"] for row in broker.grants("ann")} == {"ann"}


# The command line, through the real parser.


def run(home, *words):
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    return cli.run(parser.parse_args(["--home", str(home), *words]))


def register_on_every_start(monkeypatch):
    attach = broker_part.attach

    def attach_and_register(assistant):
        attach(assistant)
        make(assistant.broker, "demo.act", effects=(Effect.EXECUTE,))
        make(assistant.broker, "door.open", effects=(Effect.PHYSICAL,))

    monkeypatch.setattr(broker_part, "attach", attach_and_register)


def test_commands_add_list_and_revoke_a_grant(tmp_path, capsys, monkeypatch):
    register_on_every_start(monkeypatch)
    home = tmp_path / "home"
    assert run(home, "grant", "add", "demo.act", "--user", "ann", "--uses", "3") == 0
    grant = capsys.readouterr().out.split(" as ")[1].strip(" .\n")
    assert run(home, "grant", "list") == 0
    assert grant in capsys.readouterr().out
    assert run(home, "grant", "revoke", grant) == 0
    capsys.readouterr()
    assert run(home, "grant", "list") == 0
    assert "No grants" in capsys.readouterr().out
    assert run(home, "grant", "list", "--all") == 0
    assert "revoked by" in capsys.readouterr().out
    assert run(home, "grant", "revoke", grant) == 1


def test_commands_explain_a_refusal(tmp_path, capsys, monkeypatch):
    register_on_every_start(monkeypatch)
    assert run(tmp_path / "home", "grant", "add", "door.open") == 1
    assert "no standing grant" in capsys.readouterr().out


def test_commands_show_capabilities(tmp_path, capsys, monkeypatch):
    register_on_every_start(monkeypatch)
    assert run(tmp_path / "home", "grant", "list", "--capabilities") == 0
    assert "demo.act" in capsys.readouterr().out


def test_commands_list_and_decide_approvals(tmp_path, capsys, monkeypatch):
    register_on_every_start(monkeypatch)
    home = tmp_path / "home"
    with Assistant(home) as assistant:
        asked = assistant.broker.check(ToolCall("door.open", {"token": "s3cret", "door": "front"}), context()).approval_id
        other = assistant.broker.check(ToolCall("door.open", {"door": "back"}), context()).approval_id
    assert run(home, "approvals", "list") == 0
    shown = capsys.readouterr().out
    assert asked in shown and "s3cret" not in shown and "front" in shown
    assert run(home, "approvals", "approve", asked) == 0
    assert run(home, "approvals", "deny", other) == 0
    capsys.readouterr()
    assert run(home, "approvals", "list") == 0
    assert "Nothing is waiting" in capsys.readouterr().out
    assert run(home, "approvals", "list", "--all") == 0
    assert "approved" in capsys.readouterr().out
    assert run(home, "approvals", "approve", other) == 1
