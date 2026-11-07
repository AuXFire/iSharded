"""Shorter loops: tours on measured tables, and proposals that leave out a node nobody needs."""

from itertools import permutations

import pytest

import test_planner as fixtures
from isharded import geometry, planner
from isharded.planner import Candidate

MIB = 1024**2
GIB = 1024**3

# The seven-node mesh of the 235B run as the WAN lab replays it: round trips in ms along the
# chain it ran in, from measured hop costs, and 110 for every other pair, which is a guess.
MESH7 = [f"s{index}" for index in range(1, 8)]
MESH7_CHAIN = {("s1", "s2"): 106, ("s2", "s3"): 105, ("s3", "s4"): 64, ("s4", "s5"): 111, ("s5", "s6"): 74, ("s6", "s7"): 195}


def mesh7_links() -> dict[tuple[str, str], float]:
    links = {(left, right): 110.0 for left in MESH7 for right in MESH7 if left != right}
    for (left, right), milliseconds in MESH7_CHAIN.items():
        links[left, right] = links[right, left] = milliseconds
    return links


def loop_rtt(order: list[str], links: dict[tuple[str, str], float]) -> float:
    return sum(links[left, right] for left, right in zip(order, order[1:] + order[:1]))


def test_the_tour_of_the_seven_node_mesh_is_the_shortest_loop_and_skips_its_worst_link() -> None:
    links = mesh7_links()
    workers = planner.with_links([Candidate(worker_id=name, url=f"http://{name}") for name in MESH7], links)
    order = [item.worker_id for item in geometry.tour(workers, links)]
    best = min(loop_rtt(["s1", *rest], links) for rest in permutations(MESH7[1:]))
    assert sorted(order) == MESH7
    assert loop_rtt(order, links) == best == 679
    assert ("s6", "s7") not in fixtures.touching(fixtures.stages_on(*order))


def test_on_the_seven_node_mesh_the_law_charges_half_of_each_round_trip_and_a_hop() -> None:
    links = mesh7_links()
    today = planner.route_network_ms(fixtures.stages_on(*MESH7), links, ring=True)
    toured = planner.route_network_ms(fixtures.stages_on("s1", "s2", "s3", "s4", "s6", "s5", "s7"), links, ring=True)
    assert today == 765 / 2 + 7 * planner.RING_HOP_MS
    assert toured == 679 / 2 + 7 * planner.RING_HOP_MS


# Two near nodes hold the model between them, and a third sits far off.
NEAR_FAR = {("a", "b"): 2.0, ("b", "c"): 40.0, ("a", "c"): 40.0}
NEAR_FAR |= {(right, left): milliseconds for (left, right), milliseconds in NEAR_FAR.items()}


def three_cards() -> list[Candidate]:
    return [fixtures.worker(name, fixtures.pool(f"{name}-gpu", "vram", GIB, 300, f"{name}:vram")) for name in "abc"]


def spread_route(model):
    """A route on all three nodes, as a repair or a node joining can leave one."""
    return fixtures.zigzag(model, ["a", "b", "c"], three_cards(), NEAR_FAR)


def ring_profile(dense: int = 150 * MIB):
    """Eight layers. At 150 MiB each they need two of the three cards, at 250 MiB all three."""
    model = fixtures.profile(dense=dense)
    model.ring = True
    return model


def test_a_node_the_model_does_not_need_is_left_out_when_shedding_is_allowed() -> None:
    model = ring_profile()
    route, workers = spread_route(model)
    found = geometry.propose_route(route, workers, NEAR_FAR, model, shed=True)
    assert found is not None
    new_route, proposal = found
    assert proposal["left_out"] == ["c"]
    assert sorted(proposal["proposed"]["order"]) == ["a", "b"]
    assert [stage.worker_id for stage in new_route] == proposal["proposed"]["order"]
    assert [(stage.start_layer, stage.end_layer) for stage in new_route] == [(item["start_layer"], item["end_layer"]) for item in proposal["proposed"]["stages"]]
    assert new_route[0].start_layer == 0 and new_route[-1].end_layer == model.num_layers
    # Three links of the loop become two near ones, and every token pays one hop less.
    loop_before, loop_after = (2 + 40 + 40) / 2 + 3 * planner.RING_HOP_MS, (2 + 2) / 2 + 2 * planner.RING_HOP_MS
    assert proposal["saved_ms_per_token"] == pytest.approx(loop_before - loop_after, abs=0.05)
    assert proposal["moved_bytes"] > 0 and proposal["break_even_tokens"] > 0


def test_without_shedding_a_proposal_keeps_every_node() -> None:
    model = ring_profile()
    route, workers = spread_route(model)
    # Every loop through three nodes crosses the same links.
    assert geometry.propose(route, workers, NEAR_FAR, model) is None


def test_a_pinned_node_is_never_left_out() -> None:
    model = ring_profile()
    route, workers = spread_route(model)
    proposal = geometry.propose(route, workers, NEAR_FAR, model, exit_="c", shed=True)
    assert proposal is None or "c" in proposal["proposed"]["order"]


def test_shedding_never_takes_a_node_the_model_needs() -> None:
    model = ring_profile(250 * MIB)
    route, workers = spread_route(model)
    assert geometry.propose(route, workers, NEAR_FAR, model, shed=True) is None


def test_a_reorder_leaves_nothing_out() -> None:
    model = fixtures.ring_model()
    route, workers = fixtures.zigzag(model)
    proposal = geometry.propose(route, workers, fixtures.LINKS, model, shed=True)
    assert proposal["left_out"] == []
    assert sorted(proposal["proposed"]["order"]) == sorted(fixtures.ZIGZAG)


# A near node joined after a repair put a far one on the route.
FOUR = {**NEAR_FAR, ("a", "d"): 2.0, ("b", "d"): 2.0, ("c", "d"): 40.0}
FOUR |= {(right, left): milliseconds for (left, right), milliseconds in FOUR.items()}


def four_cards() -> list[Candidate]:
    return [*three_cards(), fixtures.worker("d", fixtures.pool("d-gpu", "vram", GIB, 300, "d:vram"))]


def test_a_spare_that_joined_can_take_the_far_node_s_place() -> None:
    model = ring_profile(250 * MIB)
    route, workers = fixtures.zigzag(model, ["a", "b", "c"], four_cards(), FOUR)
    found = geometry.propose_route(route, workers, FOUR, model, shed=True, spares=["d"])
    assert found is not None
    new_route, proposal = found
    assert proposal["left_out"] == ["c"] and proposal["joined"] == ["d"]
    assert sorted(stage.worker_id for stage in new_route) == ["a", "b", "d"]
    assert proposal["moved_bytes"] > 0  # the newcomer has nothing yet


def test_a_spare_is_weighed_only_when_shedding_and_only_on_measured_links() -> None:
    model = ring_profile(250 * MIB)
    route, workers = fixtures.zigzag(model, ["a", "b", "c"], four_cards(), FOUR)
    assert geometry.propose(route, workers, FOUR, model, spares=["d"]) is None
    unmeasured = {link: milliseconds for link, milliseconds in FOUR.items() if "d" not in link or "c" in link}
    assert geometry.propose(route, workers, unmeasured, model, shed=True, spares=["d"]) is None


def ram_worker(name: str) -> Candidate:
    return fixtures.worker(name, fixtures.pool(f"{name}-ram", "ram", GIB, 30, f"{name}:ram"))


def test_on_workers_with_no_vram_a_spare_can_still_take_the_far_node_s_place() -> None:
    model = ring_profile(250 * MIB)
    workers = [ram_worker(name) for name in "abcd"]
    route, workers = fixtures.zigzag(model, ["a", "b", "c"], workers, FOUR)
    proposal = geometry.propose(route, workers, FOUR, model, shed=True, spares=["d"])
    assert proposal is not None and proposal["left_out"] == ["c"] and proposal["joined"] == ["d"]
