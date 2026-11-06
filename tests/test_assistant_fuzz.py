"""Plans and schemas come from models and strangers, so junk must be refused and never raise."""
from __future__ import annotations

import copy
import json
import random

import pytest

from isharded.assistant.app import Assistant
from isharded.assistant.contracts import CallContext, Capability, Effect, ToolCall, Verdict
from isharded.assistant.planner import plan_problems
from isharded.assistant.schema import errors
from isharded.assistant.tasks import normalised, plan_errors

JUNK = [
    None, True, False, 0, -1, 1, 3.5, 1e308, float("inf"), float("nan"), "", "x", "{{steps.a.output}}", "{{inputs.nope}}", "{{",
    [], [[]], [{}], [None], ["a"], [["a"]], [1, 2], {}, {"a": 1}, {"a": []}, {"type": "regex"}, {"type": []}, {"steps": []}, {"steps": [[]]},
]  # fmt: skip

BASE = {
    "title": "t",
    "priority": 1,
    "reach": "pool",
    "inputs": {"x": "y"},
    "run_at": 2_000_000_000,
    "every_seconds": 60,
    "on_missed": "skip",
    "steps": [
        {"id": "a", "kind": "tool", "capability": "files.read", "arguments": {"path": "/x"}, "retries": 1, "backoff_seconds": 1, "on_failure": "stop", "priority": 1, "resources": ["gpu:1"], "expect": {"type": "regex", "pattern": "x"}},
        {"id": "b", "kind": "model", "role": "worker", "prompt": "{{steps.a.output}}", "system": "s", "max_tokens": 5, "temperature": 0.1, "escalate_to": ["planning"], "review": True, "depends_on": ["a"]},
        {"id": "c", "kind": "check", "check": {"type": "exit_code", "step": "a", "equals": 0}},
        {"id": "d", "kind": "check", "check": {"type": "json_schema", "text": "{{steps.a.output}}", "schema": {"type": "object"}}},
        {"id": "e", "kind": "approval", "message": "go"},
        {"id": "f", "kind": "wait", "seconds": 1},
        {"id": "g", "kind": "wait", "until": 2_000_000_000},
        {"id": "h", "kind": "subtask", "inputs": {"t": "{{steps.a.output}}"}, "plan": {"steps": [{"id": "i", "kind": "model", "prompt": "{{inputs.t}}"}]}},
    ],
}  # fmt: skip


def paths(value, where=()):
    """Every place in a plan a value could be swapped for junk."""
    yield where
    if isinstance(value, dict):
        for key, item in value.items():
            yield from paths(item, (*where, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from paths(item, (*where, index))


def replaced(plan, where, junk):
    changed = copy.deepcopy(plan)
    if not where:
        return junk
    node = changed
    for key in where[:-1]:
        node = node[key]
    node[where[-1]] = junk
    return changed


PLACES = list(paths(BASE))


def test_the_base_plan_is_valid():
    assert plan_errors(BASE) == []


@pytest.mark.parametrize("junk", JUNK, ids=repr)
def test_junk_anywhere_in_a_plan_never_raises(junk):
    for where in PLACES:
        plan = replaced(BASE, where, junk)
        problems = plan_errors(plan, {"gpu": 1})
        if not problems:
            normalised(plan)  # what passes must also survive what comes next
        plan_problems(plan, {"files.read": Capability("files.read", "", frozenset({Effect.READ}))}, {"gpu": 1})


@pytest.mark.parametrize("seed", range(30))
def test_several_kinds_of_junk_at_once_never_raise(seed):
    chooser = random.Random(seed)
    plan = BASE
    for _ in range(6):
        where = chooser.choice(list(paths(plan)))
        plan = replaced(plan, where, chooser.choice(JUNK))
    if isinstance(plan, dict) and not plan_errors(plan):
        normalised(plan)


@pytest.mark.parametrize("value", ["Infinity", "-Infinity", "NaN"])
@pytest.mark.parametrize("field", ["run_at", "every_seconds"])
def test_numbers_python_json_allows_but_time_cannot_use_are_refused(field, value):
    plan = {**json.loads(json.dumps(BASE)), field: json.loads(value)}
    assert plan_errors(plan)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_a_wait_or_backoff_must_be_finite(value):
    assert plan_errors({"steps": [{"id": "a", "kind": "wait", "seconds": value}]})
    assert plan_errors({"steps": [{"id": "a", "kind": "wait", "until": value}]})
    assert plan_errors({"steps": [{"id": "a", "kind": "wait", "seconds": 1, "backoff_seconds": value}]})


@pytest.mark.parametrize(
    "field, value",
    [
        ("role", ["worker"]),
        ("role", 5),
        ("escalate_to", [["planning"]]),
        ("escalate_to", [5]),
        ("system", 5),
        ("max_tokens", "many"),
        ("max_tokens", 0),
        ("temperature", "hot"),
    ],
)
def test_model_step_fields_have_the_right_types(field, value):
    assert plan_errors({"steps": [{"id": "a", "kind": "model", "prompt": "x", field: value}]})


def test_an_approval_message_must_be_text():
    assert plan_errors({"steps": [{"id": "a", "kind": "approval", "message": 5}]})


SCHEMA_JUNK = [
    None, True, False, 5, "x", [], {}, {"type": 5}, {"type": ["string", 5]}, {"type": {}}, {"properties": []}, {"properties": 5},
    {"properties": {"a": 5}}, {"required": 5}, {"required": [[]]}, {"enum": 5}, {"const": []}, {"items": 5}, {"allOf": 5}, {"anyOf": 5},
    {"oneOf": 5}, {"not": 5}, {"minimum": "x"}, {"minLength": "x"}, {"maxItems": []}, {"additionalProperties": 5},
]  # fmt: skip
VALUES = [None, True, 5, 2.5, "x", [], [1], {}, {"a": 1}, {"a": {"b": []}}]


@pytest.mark.parametrize("schema", SCHEMA_JUNK, ids=repr)
def test_a_junk_schema_never_raises(schema):
    for value in VALUES:
        assert isinstance(errors(value, schema), list)


def test_a_schema_that_refers_to_itself_forever_cannot_crash_a_call(tmp_path):
    loop: dict = {"type": "object"}
    loop["properties"] = {"a": loop}
    nested: dict = {"a": {}}
    node = nested
    for _ in range(5000):
        node["a"] = {"a": {}}
        node = node["a"]
    with Assistant(tmp_path / "home") as assistant:
        assistant.broker.register(Capability("demo.loop", "", frozenset({Effect.READ}), input_schema=loop), lambda a, c: "ran")
        assistant.broker.add_grant("ann", "demo.loop", None, "ann")
        decision = assistant.broker.check(ToolCall("demo.loop", nested), CallContext("ann", "conversation"))
    assert decision.verdict is Verdict.DENY


@pytest.mark.parametrize("schema", SCHEMA_JUNK, ids=repr)
def test_a_junk_declared_schema_cannot_take_down_invoke(tmp_path, schema):
    with Assistant(tmp_path / "home") as assistant:
        declared = schema if isinstance(schema, dict) else {}
        assistant.broker.register(Capability("demo.odd", "", frozenset({Effect.READ}), input_schema=declared, scope_field="path"), lambda a, c: "ran")
        assistant.broker.add_grant("ann", "demo.odd", "*", "ann")
        result = assistant.broker.invoke(ToolCall("demo.odd", {"path": "/x", "a": 1}), CallContext("ann", "conversation"))
    assert isinstance(result.ok, bool)
