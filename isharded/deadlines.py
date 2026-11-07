"""How long a pass may take. Apart from the protocol, because a stage process cannot load torch."""

from __future__ import annotations

# Far beyond any real pass, and far short of leaving a reply hung on a dead link.
PASS_SECONDS = 30.0
PASS_SECONDS_PER_TOKEN = 1.0


def pass_seconds(tokens: int, hops: int = 1) -> float:
    """How long a pass may take before the stage after this one counts as silent.

    A lost link often gives no error at all: packets just stop. Each hop waits
    longer than the next one in, so the failure is blamed on the right stage.
    """
    return (PASS_SECONDS + tokens * PASS_SECONDS_PER_TOKEN) * max(hops, 1)
