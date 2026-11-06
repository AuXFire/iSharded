"""One-command setup: invites must be easy to use and hard to abuse."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

pytest.importorskip("cryptography")

from isharded import up
from isharded.access import AccessControl, Authority, digest
from isharded.pool import read_membership_request, request_membership


def owner(tmp_path: Path) -> Path:
    home = tmp_path / "owner"
    up.make_owner(home, "first", "100.64.0.1")
    return home


def test_a_new_mesh_gets_an_authority_a_certificate_and_a_signed_owner_key(tmp_path: Path) -> None:
    home = owner(tmp_path)
    assert all((home / name).is_file() for name in ("ca.pem", "ca.key", "first.pem", "first.key", "admin.key"))
    key = (home / "admin.key").read_text(encoding="utf-8")
    access = AccessControl.load(None, str(home / "access.json"), identities=True, authority=Authority(str(home / "ca.pem")))
    caller = access.keys[digest(key)]
    assert caller.role == "admin"
    assert key not in (home / "access.json").read_text(encoding="utf-8")  # only its hash leaves the owner's file
    assert access.document()["signed"]  # signed, so gossip carries it to every node


def test_an_invite_survives_the_trip_and_says_what_it_can(tmp_path: Path) -> None:
    home = owner(tmp_path)
    made = up.create_invite(home, ["http://100.64.0.1:8209/"], uses=2, hours=1)
    assert made["command"] == f"isharded up --invite {made['code']}"
    invite = up.read_invite(made["code"])
    assert invite["urls"] == ["http://100.64.0.1:8209"]
    assert invite["authority"] == up.authority_hash((home / "ca.pem").read_text(encoding="utf-8"))
    assert up.read_invites(home)[0]["uses"] == 2


@pytest.mark.parametrize("code", ["nonsense", "isi-!!!!", "isi-e30"])
def test_a_damaged_invite_says_so(code: str) -> None:
    with pytest.raises(ValueError):
        up.read_invite(code)


def test_a_proof_is_bound_to_one_request(tmp_path: Path) -> None:
    home = owner(tmp_path)
    secret = up.read_invite(up.create_invite(home, ["http://a:1"])["code"])["secret"]
    good = request_membership(str(tmp_path / "joiner"), "joiner", ["100.64.0.2"])
    other = request_membership(str(tmp_path / "attacker"), "attacker", ["100.64.0.3"])
    proof = up.invite_proof(secret, good, "a" * 64)
    invites = up.read_invites(home)
    assert up.matching_invite(invites, {"request": good, "key_sha256": "a" * 64, "proof": proof})
    # Whoever watches gossip sees the proof. It is no use on their own request or key.
    assert up.matching_invite(invites, {"request": other, "key_sha256": "a" * 64, "proof": proof}) is None
    assert up.matching_invite(invites, {"request": good, "key_sha256": "b" * 64, "proof": proof}) is None
    assert up.matching_invite(invites, {"request": good}) is None


def test_expired_and_used_invites_admit_nobody(tmp_path: Path) -> None:
    home = owner(tmp_path)
    secret = up.read_invite(up.create_invite(home, ["http://a:1"], hours=1)["code"])["secret"]
    request = request_membership(str(tmp_path / "joiner"), "joiner", ["100.64.0.2"])
    item = {"request": request, "key_sha256": "", "proof": up.invite_proof(secret, request, "")}
    invites = up.read_invites(home)
    assert up.matching_invite(invites, item, clock=lambda: invites[0]["expires_at"] + 1) is None
    invites[0]["uses"] = 0
    assert up.matching_invite(invites, item) is None


def transport(home: Path, requests: list[dict[str, str]], posted: list[dict[str, str]]) -> httpx.Client:
    def answer(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"requests": requests})
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "accepted"})

    return httpx.Client(transport=httpx.MockTransport(answer))


def test_the_owners_machine_signs_a_good_request_once_and_ignores_the_rest(tmp_path: Path) -> None:
    home = owner(tmp_path)
    secret = up.read_invite(up.create_invite(home, ["http://a:1"], uses=1)["code"])["secret"]
    good = request_membership(str(tmp_path / "joiner"), "joiner", ["100.64.0.2"])
    bad = request_membership(str(tmp_path / "stranger"), "stranger", ["100.64.0.3"])
    key_hash = "c" * 64
    requests = [
        {**read_membership_request(bad), "request": bad, "proof": "0" * 64},
        {**read_membership_request(good), "request": good, "key_sha256": key_hash, "proof": up.invite_proof(secret, good, key_hash)},
    ]
    posted: list[dict[str, str]] = []
    with transport(home, requests, posted) as client:
        assert up.approve_invited(home, client, "http://a:1", "key") == ["joiner"]
        assert up.approve_invited(home, client, "http://a:1", "key") == []  # the invite is spent
    kinds = [item["entry"]["type"] for item in posted]
    assert kinds == ["member", "key"]
    assert posted[1]["entry"]["role"] == "user"  # the new machine's owner can chat, not administer
    access = AccessControl(authority=Authority(str(home / "ca.pem")), identities=True)
    assert all(access.add_signed(item) for item in posted)
    assert access.certificates["joiner"]


def test_the_invite_fields_travel_with_a_request_through_gossip(tmp_path: Path) -> None:
    text = request_membership(str(tmp_path / "joiner"), "joiner", ["100.64.0.2"])
    held = {**read_membership_request(text), "request": text, "proof": "p" * 64, "key_sha256": "k" * 64, "role": "admin"}
    home = owner(tmp_path)
    access = AccessControl(authority=Authority(str(home / "ca.pem")), identities=True)
    access.merge({"join_requests": [held], "signed": []})
    kept = access.join_requests["joiner"]
    assert kept["proof"] == "p" * 64 and kept["key_sha256"] == "k" * 64
    assert "role" not in kept  # nothing else rides along


def test_a_name_taken_by_another_key_is_noticed(tmp_path: Path) -> None:
    home = owner(tmp_path)
    certificate = (home / "first.pem").read_text(encoding="utf-8")
    assert up.certificate_matches_key(certificate, home / "first.key")
    request_membership(str(tmp_path / "other"), "first", ["100.64.0.2"])
    assert not up.certificate_matches_key(certificate, tmp_path / "other" / "first.key")


def settings(**changes: object) -> dict[str, object]:
    return {"name": "first", "address": "100.64.0.1", "port": 8209, "tls": False, "cpu_threads": 4, **changes}


def test_settings_become_a_full_node_command(tmp_path: Path) -> None:
    args = up.node_arguments(settings(gguf_dirs=["D:/models"]), tmp_path)
    assert args.node_id == "first" and args.control_port == 8209 and args.base_port == 8200
    assert args.ca_file.endswith("ca.pem") and args.identity_key.endswith("first.key")
    assert args.weights_ram_gb == "auto" and args.gguf_dir == ["D:/models"]
    assert args.tls_cert is None


def test_tls_uses_the_nodes_own_certificate(tmp_path: Path) -> None:
    args = up.node_arguments(settings(tls=True), tmp_path)
    assert args.tls_cert.endswith("first.pem") and args.tls_key.endswith("first.key")


def test_only_tailscale_or_a_stated_private_network_skips_tls() -> None:
    assert not up.needs_tls("100.64.0.1", False)
    assert up.needs_tls("192.168.1.20", False)
    assert not up.needs_tls("192.168.1.20", True)


def test_machine_names_are_safe_for_a_node(monkeypatch) -> None:
    monkeypatch.setattr(up.socket, "gethostname", lambda: "Alice's PC.local")
    assert up.machine_name() == "alice-s-pc-local"


def test_a_machine_in_a_mesh_refuses_to_join_another(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ISHARDED_UP_HOME", str(tmp_path))
    up.write_settings(tmp_path, settings())
    args = type("Args", (), {"invite": "isi-x"})()
    assert up.run_up(args) == 1
    assert "already belongs" in capsys.readouterr().err


def test_only_the_owners_machine_can_invite(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ISHARDED_UP_HOME", str(tmp_path))
    up.write_settings(tmp_path, settings())
    assert up.run_invite(type("Args", (), {"uses": 1, "hours": 1})()) == 1
    assert "Only the machine that made the mesh" in capsys.readouterr().err


@pytest.mark.parametrize("system,suffix", [("Linux", "isharded.service"), ("Darwin", "ai.isharded.node.plist")])
def test_autostart_files_run_up_for_this_user(system: str, suffix: str, tmp_path: Path) -> None:
    path, body = up.service_definition("/opt/venv/bin/python", system, tmp_path)
    assert path is not None and path.name == suffix and str(path).startswith(str(tmp_path))
    assert "/opt/venv/bin/python" in body and "--no-browser" in body


def test_doctor_gives_a_fix_for_everything_it_flags() -> None:
    assert all(advice for good, _, advice in up.checks() if not good)


def test_a_person_approving_needs_no_invite_and_a_denial_is_signed(tmp_path: Path) -> None:
    home = owner(tmp_path)
    asked = request_membership(str(tmp_path / "joiner"), "joiner", ["100.64.0.2"])
    refused = request_membership(str(tmp_path / "other"), "other", ["100.64.0.3"])
    waiting = request_membership(str(tmp_path / "waiter"), "waiter", ["100.64.0.4"])
    requests = [
        {**read_membership_request(asked), "request": asked, "approved": True},
        {**read_membership_request(refused), "request": refused, "denied": True},
        {**read_membership_request(waiting), "request": waiting},
    ]
    posted: list[dict[str, str]] = []
    with transport(home, requests, posted) as client:
        assert up.approve_invited(home, client, "http://a:1", "key") == ["joiner"]
    assert sorted(item["entry"]["type"] for item in posted) == ["deny", "member"]
    assert {item["entry"]["name"] for item in posted} == {"joiner", "other"}


def test_nothing_joins_a_found_mesh_without_being_asked_to(capsys, monkeypatch) -> None:
    found = [{"url": "http://a:1", "authority": "a" * 64, "name": "first"}]
    args = type("Args", (), {"find": False})()
    monkeypatch.setattr(up.sys.stdin, "isatty", lambda: False, raising=False)
    assert up.choose_mesh(found, args) is None
    assert "Starting a new mesh" in capsys.readouterr().out
    assert up.choose_mesh([], args) is None


def test_find_joins_the_only_mesh_and_refuses_to_guess_between_several() -> None:
    one = {"url": "http://a:1", "authority": "a" * 64, "name": "first"}
    args = type("Args", (), {"find": True})()
    assert up.choose_mesh([one], args) == one
    with pytest.raises(ValueError, match="use an invite"):
        up.choose_mesh([one, {**one, "authority": "b" * 64}], args)


def test_up_waits_for_a_coordinator_and_a_worker_before_it_says_it_can_load() -> None:
    def reply(status, body=None):
        return httpx.Response(status, json=body or {})

    assert not up.can_load(reply(503))  # nobody coordinates yet
    assert not up.can_load(reply(200, {"workers": {}}))
    assert not up.can_load(reply(200, {"workers": {"a-cpu": {"live": False}}}))
    assert up.can_load(reply(200, {"workers": {"a-cpu": {"live": True}}}))
    assert up.can_load(reply(401))  # a key is only checked once a coordinator is known


def test_a_node_behind_mapped_ports_can_name_both_of_them(tmp_path: Path) -> None:
    settings = {"name": "a", "address": "203.0.113.5", "port": 22184, "worker_port": 22185, "tls": False, "trusted_network": True, "cpu_threads": 2, "no_gpu": True, "gpu_gb": None, "ram_gb": None, "gguf_dirs": [], "ownerless": False, "join": [], "discoverable": False}
    arguments = up.node_arguments(settings, tmp_path)
    assert (arguments.base_port, arguments.control_port) == (22185, 22184)
    settings["worker_port"] = None
    assert up.node_arguments(settings, tmp_path).base_port == 22175
