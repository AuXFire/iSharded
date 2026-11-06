"""The control mesh: which nodes exist, and which one coordinates.

ShardMesh has no fixed orchestrator. Every node runs the control plane, and
coordinating is a role the nodes agree on among themselves. Membership spreads by
gossip, so a node joins through any peer. The coordinator's state is copied to
every peer, so losing any one machine loses nothing but what it was running.

This module is pure logic with a clock passed in, so an election can be tested
without a network. ``term`` is the generation of leadership: it only goes up, and
the higher term always wins. When two sides of a split both elected a leader, the
term decides who steps down once they can hear each other again.

Two nodes cannot form a majority once split, so each side of a split elects its
own leader and carries on with the workers it can reach. That favors staying up.
Nothing can be corrupted by it: a worker takes orders only from the highest term
it has seen, and a stale coordinator's requests fail its epoch check.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import time
from typing import Any
from urllib.parse import urlsplit

# A peer gossips every few seconds, so this many silent seconds means it is gone.
PEER_TIMEOUT_SECONDS = 12.0
GOSSIP_SECONDS = 3.0
# Listen before claiming anything, so a newcomer learns who already leads.
JOIN_GRACE_SECONDS = 7.0
# A rumor nobody repeats stops being worth calling, and there is no reason for many.
RUMOR_SECONDS = 60.0
MAX_RUMORS = 256


def clean_url(value: Any) -> str:
    """A node's address, or ValueError. These arrive from the network."""
    url = str(value).rstrip("/")
    parts = urlsplit(url)
    plain = not (parts.path or parts.query or parts.fragment or parts.username)
    if parts.scheme not in {"http", "https"} or not parts.hostname or not plain or any(ord(c) <= 32 or ord(c) == 127 for c in url):
        raise ValueError(f"not a node address: {url[:80]!r}")
    _ = parts.port  # raises on a bad port
    return url


@dataclass(slots=True)
class Peer:
    node_id: str
    url: str
    last_seen: float = 0.0


class ControlMesh:
    def __init__(
        self,
        node_id: str,
        url: str,
        seeds: tuple[str, ...] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.node_id = node_id
        self.url = url.rstrip("/")
        self.seeds = tuple(seed.rstrip("/") for seed in seeds if seed.rstrip("/") != self.url)
        self.clock = clock
        self.started_at = clock()
        self.term = 0
        self.leader_id: str | None = None
        self.peers: dict[str, Peer] = {}
        # A peer's address we were told about but have not heard from yet.
        self.rumors: dict[str, float] = {}
        # Addresses of revoked nodes, which nobody calls again.
        self.barred: set[str] = set()
        # The coordinator's latest state, kept by every peer.
        self.replica: dict[str, Any] | None = None
        self.lanes: dict[str, Any] = {}
        self.lanes_version = 0
        self.replica_version = 0

    @property
    def is_leader(self) -> bool:
        return self.leader_id == self.node_id

    @property
    def leader_url(self) -> str | None:
        if self.is_leader:
            return self.url
        peer = self.peers.get(self.leader_id or "")
        return peer.url if peer else None

    def alive(self) -> dict[str, str]:
        """Everyone reachable right now, including this node, as id to address."""
        now = self.clock()
        members = {self.node_id: self.url}
        for peer in self.peers.values():
            if now - peer.last_seen < PEER_TIMEOUT_SECONDS:
                members[peer.node_id] = peer.url
        return members

    def known_urls(self) -> set[str]:
        """Every member's address, alive or not, short of the revoked."""
        return {peer.url for peer in self.peers.values()} | {self.url}

    def targets(self) -> set[str]:
        """Every address worth gossiping to, including ones not heard from yet."""
        known = {peer.url for peer in self.peers.values()}
        now = self.clock()
        self.rumors = {url: heard for url, heard in self.rumors.items() if now - heard < RUMOR_SECONDS}
        return (known | set(self.rumors) | set(self.seeds)) - {self.url} - self.barred

    def forget(self, names: Iterable[str]) -> None:
        """Stop talking to revoked nodes, so nothing of theirs comes back in a reply."""
        for name in names:
            peer = self.peers.pop(name, None)
            if peer is not None:
                self.barred.add(peer.url)
                self.rumors.pop(peer.url, None)

    def message(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "url": self.url,
            "term": self.term,
            "leader_id": self.leader_id,
            "peers": sorted(self.alive().values()),
            "replica": self.replica if self.is_leader else None,
            "replica_version": self.replica_version if self.is_leader else 0,
            "lanes": self.lanes if self.is_leader else None,
            "lanes_version": self.lanes_version if self.is_leader else 0,
        }

    def observe(self, message: dict[str, Any]) -> bool:
        """Take in a peer's gossip. Returns True if this node just lost leadership."""
        sender = str(message["node_id"])
        if sender == self.node_id:
            return False
        self.peers[sender] = Peer(sender, clean_url(message["url"]), self.clock())
        known = {peer.url for peer in self.peers.values()} | {self.url} | self.barred
        for url in message.get("peers", []):
            try:
                heard = clean_url(url)
            except ValueError:
                continue  # one bad address must not cost the rest of the message
            if heard not in known and (heard in self.rumors or len(self.rumors) < MAX_RUMORS):
                self.rumors[heard] = self.clock()
        for url in known:
            self.rumors.pop(url, None)

        term = int(message.get("term", 0))
        claimed = message.get("leader_id")
        was_leader = self.is_leader
        if sender == self.leader_id and claimed != sender and term <= self.term:
            # The leader says it no longer leads, because it restarted. Someone
            # still hearing from it would otherwise wait for it forever.
            self.leader_id = None
        if claimed == self.node_id and not was_leader:
            # Old news from before this node restarted. Leading takes an election,
            # because taking over is what restores the route.
            self.term = max(self.term, term)
        elif claimed and self._outranks(term, str(claimed)):
            self.term = term
            self.leader_id = str(claimed)
        elif term > self.term:
            self.term = term
        replica = message.get("replica")
        if replica is not None and sender == self.leader_id and message.get("replica_version", 0) >= self.replica_version:
            self.replica = replica
            self.replica_version = int(message["replica_version"])
        lanes = message.get("lanes")
        if isinstance(lanes, dict) and sender == self.leader_id and int(message.get("lanes_version", 0)) >= self.lanes_version:
            self.lanes = lanes
            self.lanes_version = int(message["lanes_version"])
        return was_leader and not self.is_leader

    def _outranks(self, term: int, candidate: str) -> bool:
        """Whether a claimed leader beats the one this node follows."""
        if term != self.term:
            return term > self.term
        if self.leader_id is None:
            return True
        return candidate < self.leader_id

    def elect(self) -> bool:
        """Settle who leads. Returns True if this node just took over."""
        members = self.alive()
        if self.leader_id in members:
            return False
        self.leader_id = None
        if self.clock() - self.started_at < JOIN_GRACE_SECONDS:
            return False
        # Everyone applies the same rule to the same list, so no vote is needed.
        if min(members) != self.node_id:
            return False
        self.term += 1
        self.leader_id = self.node_id
        return True

    def publish(self, state: dict[str, Any]) -> None:
        """Record the coordinator's state for the next round of gossip."""
        self.replica = state
        self.replica_version += 1

    def publish_lanes(self, lanes: dict[str, Any]) -> None:
        """Record which lanes the pool runs, for the next round of gossip."""
        self.lanes = lanes
        self.lanes_version += 1

    def learn_term(self, term: int) -> None:
        """Workers remember the highest term, which outlives any control process."""
        self.term = max(self.term, term)
