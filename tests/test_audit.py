"""Comparing routes: honest ones agree, and blame takes a majority."""
from __future__ import annotations

import random

from shardmesh import audit
from shardmesh.lanes import LaneDoor

HONEST = ["Rivers carry water to the sea and shape the land around them over time"] * 3
BENT = ["Purple seven under quickly nothing the of of a and"] * 3


def test_questions_read_like_chat_and_vary() -> None:
    questions = audit.canaries(40, random.Random(1))
    assert len(questions) == 40 and len(set(questions)) > 20
    assert all("{}" not in question for question in questions)


def test_agreement_is_how_far_replies_run_together() -> None:
    assert audit.agreement("a b c d", "a b c d") == 1.0
    assert audit.agreement("a b c d", "a b x d") == 0.5
    assert audit.agreement("x b c d", "a b c d") == 0.0
    assert audit.agreement("a b", "a b c d e f") == 1.0  # a shorter reply that matches is no disagreement
    assert audit.agreement("", "") == 1.0
    assert audit.agreement("", "a") == 0.0


def test_routes_that_part_late_at_a_close_call_still_agree() -> None:
    late = ["Rivers carry water to the sea and shape the land around us over time"] * 3
    assert audit.judge({"main": HONEST, "check": late})["verdict"] == "agree"


def test_two_routes_can_only_say_something_is_wrong() -> None:
    result = audit.judge({"main": HONEST, "check": BENT})
    assert result["verdict"] == "disagree" and result["suspects"] == []


def test_three_routes_name_the_one_out_of_line() -> None:
    result = audit.judge({"main": HONEST, "one": HONEST, "two": BENT})
    assert result["verdict"] == "suspect" and result["suspects"] == ["two"]


def test_nobody_is_named_when_all_three_differ() -> None:
    other = ["Nothing here matches either of the others at all today"] * 3
    result = audit.judge({"main": HONEST, "one": BENT, "two": other})
    assert result["verdict"] == "disagree" and result["suspects"] == []


def test_a_lane_kept_apart_avoids_every_node_on_a_route() -> None:
    from types import SimpleNamespace

    door = LaneDoor.__new__(LaneDoor)
    door.runtime = SimpleNamespace(route=[SimpleNamespace(worker_id="alpha-gpu")])
    door.seen = {"one": {"route": [{"worker_id": "beta-cpu"}]}}
    assert door.nodes_in_use() == {"alpha", "beta"}


def test_a_lane_with_a_route_stops_holding_the_nodes_it_does_not_use() -> None:
    door = LaneDoor.__new__(LaneDoor)
    door.seen = {"one": {"route": [{"worker_id": "beta-cpu"}]}}
    budget = {"gpu_gb": 0.0, "ram_gb": 2.0}
    lanes = {"one": {"slot": 1, "budgets": {"beta": budget, "gamma": budget}, "by": ""}, "new": {"slot": 2, "budgets": {"gamma": budget}, "by": ""}}
    settled = door.settled(lanes)
    assert list(settled["one"]["budgets"]) == ["beta"]
    assert list(settled["new"]["budgets"]) == ["gamma"]  # still starting, so it keeps what it asked for
