"""A network nobody owns: nobody can command it, and nobody can speak for anyone else."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import time

import pytest

pytest.importorskip("cryptography")

from isharded.access import AccessControl, ROLE_MEMBER, Caller, canonical
from isharded.commons import (
    Commons,
    CommonsChecker,
    certificate_file,
    identity_name,
    key_file,
    make_identity,
    name_of,
    read_policy,
    sign_vouch,
    write_policy,
)
from isharded.lanes import proposer
from isharded.protocol import hosts_model
from isharded.tickets import TicketMaker


def node(tmp_path: Path, label: str, joining: str = "vouched", quorum: int = 1, trust: list[str] | None = None) -> tuple[Path, str, Commons]:
    folder = tmp_path / label
    name = make_identity(folder)
    write_policy(folder, joining, quorum, trust)
    return folder, name, Commons(folder)


def control(commons: Commons, path: Path | None = None) -> AccessControl:
    return AccessControl(authority=commons, identities=True, path=path)


def test_a_name_comes_from_the_key_and_survives_a_restart(tmp_path: Path) -> None:
    first = make_identity(tmp_path)
    assert first.startswith("n-")
    assert make_identity(tmp_path) == first == identity_name(tmp_path)
    assert first != make_identity(tmp_path / "other")


def test_a_vouch_is_checked_against_the_key_that_made_it(tmp_path: Path) -> None:
    folder, name, commons = node(tmp_path, "a")
    signed = sign_vouch(folder, "n-someone")
    assert commons.verified(signed) == signed["entry"]
    forged = {**signed, "entry": {**signed["entry"], "name": "n-another"}}
    assert commons.verified(forged) is None
    other, other_name, _ = node(tmp_path, "b")
    borrowed = {**sign_vouch(other, "n-someone")}
    borrowed["entry"] = {**borrowed["entry"], "by": name}  # claiming to be someone else
    assert commons.verified(borrowed) is None


def test_nothing_but_a_vouch_can_be_signed(tmp_path: Path) -> None:
    """Whoever holds a key may sign anything, so no other kind of entry may count."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    folder, name, commons = node(tmp_path, "a")
    key = serialization.load_pem_private_key(key_file(folder).read_bytes(), password=None)
    for entry in ({"type": "key", "name": "evil", "role": "admin", "sha256": "0" * 64, "by": name}, {"type": "revoke", "name": "n-victim", "by": name}):
        signed = {
            "entry": entry,
            "signature": base64.b64encode(key.sign(canonical(entry), ec.ECDSA(hashes.SHA256()))).decode(),
            "certificate": certificate_file(folder).read_text(encoding="utf-8"),
        }
        assert commons.verified(signed) is None
        access = control(commons)
        assert not access.add_signed(signed)
        assert not access.keys and not access.revoked


def test_gossip_changes_nothing_but_vouches(tmp_path: Path) -> None:
    _, _, commons = node(tmp_path, "a")
    access = control(commons)
    peer = {
        "version": 999,
        "keys": [{"name": "evil", "role": "admin", "sha256": "0" * 64}],
        "revoked_nodes": ["n-victim"],
        "removed_keys": [],
        "cordoned_nodes": ["n-victim"],
        "signed": [],
        "join_requests": [],
    }
    access.merge(peer)
    assert not access.keys and not access.revoked and not access.cordoned


def test_acceptance_starts_from_this_node_and_needs_a_quorum(tmp_path: Path) -> None:
    a_folder, a, a_commons = node(tmp_path, "a", quorum=2)
    b_folder, b, _ = node(tmp_path, "b")
    c_folder, c, _ = node(tmp_path, "c")
    d_folder, d, _ = node(tmp_path, "d")
    access = control(a_commons)
    assert a_commons.members(access.signed) == {a}
    access.add_signed(sign_vouch(a_folder, b))
    assert b not in a_commons.members(access.signed)  # one vouch, and this node needs two
    write_policy(a_folder, "vouched", 1, [c])
    one = Commons(a_folder)
    access = control(one)
    assert one.members(access.signed) == {a, c}
    access.add_signed(sign_vouch(c_folder, b))
    access.add_signed(sign_vouch(d_folder, b))  # d is nobody here, so its word does not count
    assert one.members(access.signed) == {a, b, c}
    access.add_signed(sign_vouch(b_folder, d))  # but b's does, now that b is in
    assert one.members(access.signed) == {a, b, c, d}


def test_two_nodes_can_disagree_and_neither_overrules_the_other(tmp_path: Path) -> None:
    a_folder, a, a_commons = node(tmp_path, "a")
    b_folder, b, b_commons = node(tmp_path, "b")
    c_folder, c, _ = node(tmp_path, "c")
    entry = sign_vouch(a_folder, c)
    a_view, b_view = control(a_commons), control(b_commons)
    a_view.add_signed(entry)
    b_view.add_signed(entry)
    assert c in a_commons.members(a_view.signed)
    assert c not in b_commons.members(b_view.signed)  # b does not accept a, so a's word means nothing to it


def test_open_joining_accepts_anyone_except_who_this_node_blocks(tmp_path: Path) -> None:
    _, _, commons = node(tmp_path, "a", joining="open")
    access = control(commons, tmp_path / "access.json")
    access.revoke_node("n-bad")
    assert commons.accepts("n-anyone")
    assert not commons.accepts("n-bad")
    access.unblock_node("n-bad")
    assert commons.accepts("n-bad")


def test_a_ticket_needs_a_self_made_identity_with_the_right_name(tmp_path: Path) -> None:
    folder, name, commons = node(tmp_path, "a", joining="open")
    checker = CommonsChecker(folder, {"here:1"})
    good = TicketMaker(str(certificate_file(folder)), str(key_file(folder))).ticket("here:1")
    assert checker.name(good) == name
    elsewhere = TicketMaker(str(certificate_file(folder)), str(key_file(folder))).ticket("there:1")
    assert checker.name(elsewhere) is None


def test_a_certificate_cannot_claim_another_nodes_name(tmp_path: Path) -> None:
    """Self-signing is free, so the name must be the key's own, or anyone could be anyone."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    folder, victim, _ = node(tmp_path, "victim", joining="open")
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, victim)])
    now = datetime.datetime.now(datetime.timezone.utc)
    forged = (
        x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=30)).sign(key, hashes.SHA256())
    )
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    (attacker / "c.pem").write_bytes(forged.public_bytes(serialization.Encoding.PEM))
    (attacker / "c.key").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    ticket = TicketMaker(str(attacker / "c.pem"), str(attacker / "c.key")).ticket("here:1")
    assert CommonsChecker(folder, {"here:1"}).name(ticket) is None
    assert name_of(key.public_key()) != victim


def test_workers_learn_the_verdict_from_this_nodes_control_plane(tmp_path: Path) -> None:
    folder, a, commons = node(tmp_path, "a")
    _, b, _ = node(tmp_path, "b")
    worker_side = Commons(folder)
    assert not worker_side.accepts(b)
    control(commons).add_signed(sign_vouch(folder, b))
    time.sleep(0)
    worker_side.checked_at = float("-inf")
    assert worker_side.accepts(b)
    assert json.loads((folder / "members.json").read_text(encoding="utf-8"))["accepted"] == sorted({a, b})


def test_a_lane_never_overwrites_the_nodes_view(tmp_path: Path, monkeypatch) -> None:
    # Found on real processes: a lane knows no vouches, and its empty view locked members out.
    folder, a, commons = node(tmp_path, "a")
    _, b, _ = node(tmp_path, "b")
    control(commons).add_signed(sign_vouch(folder, b))
    monkeypatch.setenv("ISHARDED_LANE", "small")
    lane = Commons(folder)
    AccessControl.load(None, None, identities=True, authority=lane)
    assert sorted(json.loads((folder / "members.json").read_text(encoding="utf-8"))["accepted"]) == sorted({a, b})
    assert lane.accepts(b)


def test_a_bad_read_keeps_the_last_good_view(tmp_path: Path) -> None:
    folder, _, commons = node(tmp_path, "a")
    _, b, _ = node(tmp_path, "b")
    control(commons).add_signed(sign_vouch(folder, b))
    worker_side = Commons(folder)
    assert worker_side.accepts(b)
    (folder / "members.json").write_text("{half a fi", encoding="utf-8")
    worker_side.checked_at = float("-inf")
    assert worker_side.accepts(b)


def test_a_member_ranks_above_a_user_and_below_an_admin() -> None:
    access = AccessControl()
    member = Caller("n-1/alice", ROLE_MEMBER)
    assert access.permits(member, "user") and access.permits(member, ROLE_MEMBER)
    assert not access.permits(member, "admin")
    assert proposer(member, "n-me") == "n-1"
    assert proposer(Caller("root", "admin"), "n-me") == "n-me"


def test_policy_defaults_are_open() -> None:
    assert read_policy("/nowhere") == {"joining": "open", "quorum": 1, "trust": []}


def test_each_owner_chooses_what_their_node_hosts() -> None:
    assert hosts_model([], "anything.gguf")
    assert hosts_model(["*qwen3*"], "Qwen3-0.6B-Q8_0.gguf")
    assert not hosts_model(["*llama*"], "Qwen3-0.6B-Q8_0.gguf")
