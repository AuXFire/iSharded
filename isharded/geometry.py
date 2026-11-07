"""Find a shorter loop through the nodes a route already uses, and say what it would cost.

Every token goes around the loop, so a route's order decides most of its speed on a mesh.
But a route that runs has its weights in place, and they must not move or reload unless
they must. So this only ever proposes: it tells an operator what a better order would save
and what it would move, and nothing here changes a route.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from math import ceil
from typing import Any

from .planner import (
    MAX_LINKED_WORKERS,
    Candidate,
    ModelProfile,
    _shortest_order,
    _untangled,
    holds_entry,
    link_ms,
    moved_weights,
    ms_per_token,
    plan_through,
    route_faults,
    route_times,
    with_links,
)
from .protocol import StageAssignment

# Links at or under this round trip make one site. A judgment, not a measurement.
SITE_MS = 15.0
# A saving smaller than this share of a token's time is within what a link's reading swings.
MIN_SAVED_SHARE = 0.01
# Entries to plan in full. Each is a walk through every worker, on a route of any length.
MAX_ENTRIES = 8

Links = dict[tuple[str, str], float]


def nearest_groups(names: list[str], links: Links) -> list[list[str]]:
    """Group workers by their shortest measured links, shortest first, up to one site apart."""
    parent = {name: name for name in names}

    def root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for (source, target), milliseconds in sorted(links.items(), key=lambda item: item[1]):
        if milliseconds > SITE_MS:
            break
        if source in parent and target in parent:
            parent[root(source)] = root(target)
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(root(name), []).append(name)
    return list(groups.values())


def _walked(groups: list[list[str]], links: Links) -> list[str]:
    """Group after group, each time on to whatever is nearest to where the walk has got to."""
    left = list(groups)
    walk: list[str] = []
    while left:
        group = min(left, key=lambda item: min(link_ms(links, walk[-1], name) for name in item)) if walk else left[0]
        left.remove(group)
        here = list(group)
        while here:
            name = min(here, key=lambda item: link_ms(links, walk[-1], item)) if walk else here[0]
            here.remove(name)
            walk.append(name)
    return walk


def tour(workers: list[Candidate], links: Links, ring: bool = True) -> list[Candidate]:
    """The shortest way around these workers, the edge that closes the loop included.

    Up to a dozen workers that is exact. Past that it goes group by group, since a loop
    should not cross between sites more than it must, and then uncrosses what is left.
    """
    if len(workers) <= MAX_LINKED_WORKERS:
        return _shortest_order([], workers, [], ring)
    by_name = {item.worker_id: item for item in workers}
    walk = _walked(nearest_groups(list(by_name), links), links)
    hops = [[link_ms(links, left, right) for right in walk] for left in walk]
    return [by_name[walk[index]] for index in _untangled(list(range(len(walk))), hops)]


def _orders(
    route: list[Candidate], links: Links, profile: ModelProfile, entry: str | None, exit_: str | None
) -> list[list[Candidate]]:
    """Ways through these workers worth planning in full, the entry they have now first.

    Turning a loop costs the same to cross, so the entry is free to go where the draft
    fits, and the two ways around differ only in which stage reads what.
    """
    first = [item for item in route if item.worker_id == entry]
    last = [item for item in route if item.worker_id == exit_ and item.worker_id != entry]
    if (first or last) and len(route) <= MAX_LINKED_WORKERS:
        middle = [item for item in route if item not in first and item not in last]
        return [[*first, *_shortest_order(first, middle, last, profile.ring), *last]] if middle else []
    loop = tour(route, links, profile.ring)
    ways = [loop, loop[::-1]]
    if profile.ring:
        start = loop.index(route[0])
        turns = [loop[start + index :] + loop[: start + index] for index in range(len(loop))]
        fitting = [turn for turn in turns if holds_entry(turn[0], profile)] or turns[:1]
        ways = [way for turn in fitting[:MAX_ENTRIES] for way in (turn, [turn[0], *turn[:0:-1]])]
    return [way for way in ways if (not first or way[0] in first) and (not last or way[-1] in last)]


def _times(route: list[StageAssignment], links: Links, profile: ModelProfile, tokens_per_miss: float) -> dict[str, Any]:
    loop, pace = route_times(route, links, profile.ring)
    return {
        "order": [stage.worker_id for stage in route],
        "loop_ms": round(loop, 2),
        "pace_ms": round(pace, 2),
        "ms_per_token": round(ms_per_token(loop, pace, tokens_per_miss), 2),
    }


def propose(
    route: list[StageAssignment],
    candidates: list[Candidate],
    links: Links,
    profile: ModelProfile,
    entry: str | None = None,
    exit_: str | None = None,
    entry_bytes: int = 0,
    tokens_per_miss: float = 1.0,
) -> dict[str, Any] | None:
    """A better order of the nodes in ``route``, or None if there is none worth showing.

    ``entry`` and ``exit_`` are pins, which a proposal keeps. ``entry_bytes`` is what would
    follow a new entry, such as a draft model. A proposal never needs more links that
    failed, or more weights outside VRAM, than the route it would replace.
    """
    profile = replace(profile, tokens_per_miss=tokens_per_miss)
    workers = {item.worker_id: item for item in with_links(candidates, links)}
    names = [stage.worker_id for stage in route]
    # A link nobody measured counts as a guess, which is no reason to move weights.
    measured = all((left, right) in links or (right, left) in links for left, right in combinations(names, 2))
    if len(route) < 2 or not measured or any(name not in workers for name in names):
        return None
    members = [workers[name] for name in names]
    found = plan_through(_orders(members, links, profile, entry, exit_), members, profile)
    if found is None or [stage.worker_id for stage in found] == names:
        return None
    if any(then > now for then, now in zip(route_faults(found, members), route_faults(route, members))):
        return None
    current, proposed = _times(route, links, profile, tokens_per_miss), _times(found, links, profile, tokens_per_miss)
    saved = current["ms_per_token"] - proposed["ms_per_token"]
    if saved < current["ms_per_token"] * MIN_SAVED_SHARE:
        return None
    moved, seconds = moved_weights(found, members, profile, entry_bytes if found[0].worker_id != names[0] else 0)
    kept = {(stage.worker_id, stage.start_layer, stage.end_layer) for stage in route}
    return {
        "groups": nearest_groups(names, links),
        "current": current,
        "proposed": {
            **proposed,
            "stages": [{"worker_id": stage.worker_id, "start_layer": stage.start_layer, "end_layer": stage.end_layer} for stage in found],
        },
        "saved_ms_per_token": round(saved, 2),
        "moved_bytes": moved,
        "move_seconds": round(seconds, 1),
        "stages_changed": sum((stage.worker_id, stage.start_layer, stage.end_layer) not in kept for stage in found),
        "break_even_tokens": ceil(seconds * 1000 / saved),
    }
