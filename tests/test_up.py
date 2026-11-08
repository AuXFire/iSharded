"""One-command setup: invites must be easy to use and hard to abuse."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import psutil
import pytest

pytest.importorskip("cryptography")

from isharded import up
from isharded.access import AccessControl, Authority, digest
from isharded.pool import read_membership_request, request_membership
from isharded.statements import key_verifier


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
    held = {**read_membership_request(text), "request": text, "proof": "p" * 64, "key_sha256": "k" * 64, "key_verifier": "v" * 43, "role": "admin"}
    home = owner(tmp_path)
    access = AccessControl(authority=Authority(str(home / "ca.pem")), identities=True)
    access.merge({"join_requests": [held], "signed": []})
    kept = access.join_requests["joiner"]
    assert kept["proof"] == "p" * 64 and kept["key_sha256"] == "k" * 64 and kept["key_verifier"] == "v" * 43
    assert "role" not in kept  # nothing else rides along


def test_a_new_owner_key_can_prove_itself_to_other_nodes(tmp_path: Path) -> None:
    home = owner(tmp_path)
    key = (home / "admin.key").read_text(encoding="utf-8")
    access = AccessControl.load(None, str(home / "access.json"), identities=True, authority=Authority(str(home / "ca.pem")))
    assert access.keys[digest(key)].verifier == key_verifier(key)


def test_an_invited_machine_gets_a_key_that_can_prove_itself(tmp_path: Path) -> None:
    home = owner(tmp_path)
    secret = up.read_invite(up.create_invite(home, ["http://a:1"], uses=1)["code"])["secret"]
    good = request_membership(str(tmp_path / "joiner"), "joiner", ["100.64.0.2"])
    key = "isk-joiner"
    fields = {"key_sha256": digest(key), "key_verifier": key_verifier(key)}
    proof = up.invite_proof(secret, good, fields["key_sha256"], fields["key_verifier"])
    posted: list[dict[str, str]] = []
    with transport(home, [{**read_membership_request(good), "request": good, **fields, "proof": proof}], posted) as client:
        assert up.approve_invited(home, client, "http://a:1", "key") == ["joiner"]
    assert posted[1]["entry"]["verifier"] == key_verifier(key)


def test_a_member_cannot_swap_the_verifier_on_a_request_passing_through(tmp_path: Path) -> None:
    home = owner(tmp_path)
    secret = up.read_invite(up.create_invite(home, ["http://a:1"])["code"])["secret"]
    good = request_membership(str(tmp_path / "joiner"), "joiner", ["100.64.0.2"])
    proof = up.invite_proof(secret, good, "a" * 64, key_verifier("isk-real"))
    item = {"request": good, "key_sha256": "a" * 64, "key_verifier": key_verifier("isk-real"), "proof": proof}
    invites = up.read_invites(home)
    assert up.matching_invite(invites, item)
    assert up.matching_invite(invites, {**item, "key_verifier": key_verifier("isk-theirs")}) is None
    assert up.matching_invite(invites, {**item, "key_verifier": ""}) is None


def test_a_machine_running_older_code_can_still_use_an_invite(tmp_path: Path) -> None:
    home = owner(tmp_path)
    secret = up.read_invite(up.create_invite(home, ["http://a:1"], uses=1)["code"])["secret"]
    good = request_membership(str(tmp_path / "joiner"), "joiner", ["100.64.0.2"])
    item = {**read_membership_request(good), "request": good, "key_sha256": "c" * 64, "proof": up.invite_proof(secret, good, "c" * 64)}
    posted: list[dict[str, str]] = []
    with transport(home, [item], posted) as client:
        assert up.approve_invited(home, client, "http://a:1", "key") == ["joiner"]
    assert "verifier" not in posted[1]["entry"]


def test_a_bad_verifier_is_left_out_of_a_signed_entry(tmp_path: Path) -> None:
    home = owner(tmp_path)
    secret = up.read_invite(up.create_invite(home, ["http://a:1"], uses=1)["code"])["secret"]
    good = request_membership(str(tmp_path / "joiner"), "joiner", ["100.64.0.2"])
    item = {**read_membership_request(good), "request": good, "key_sha256": "c" * 64, "key_verifier": "not one", "proof": up.invite_proof(secret, good, "c" * 64, "not one")}
    posted: list[dict[str, str]] = []
    with transport(home, [item], posted) as client:
        up.approve_invited(home, client, "http://a:1", "key")
    assert "verifier" not in posted[1]["entry"]


def older_owner(tmp_path: Path) -> tuple[Path, str]:
    """A mesh made before keys had verifiers."""
    from isharded.pool import sign_entry

    home = owner(tmp_path)
    key = (home / "admin.key").read_text(encoding="utf-8")
    signed = sign_entry(str(home), {"type": "key", "name": "owner", "role": "admin", "sha256": digest(key), "requests_per_minute": 0})
    (home / "access.json").write_text(json.dumps({"version": 1, "keys": [], "signed": [signed]}), encoding="utf-8")
    return home, key


def test_the_owners_machine_gives_an_older_owner_key_a_verifier_once(tmp_path: Path) -> None:
    home, key = older_owner(tmp_path)
    posted: list[dict[str, str]] = []
    with transport(home, [], posted) as client:
        up.give_owner_a_verifier(home, client, "http://a:1", key)
    access = AccessControl.load(None, str(home / "access.json"), identities=True, authority=Authority(str(home / "ca.pem")))
    assert access.keys[digest(key)].verifier == ""
    assert access.add_signed(posted[0])
    assert access.keys[digest(key)].verifier == key_verifier(key)
    assert access.keys[digest(key)].role == "admin"
    with transport(home, [], posted) as client:
        up.give_owner_a_verifier(home, client, "http://a:1", key)
    assert len(posted) == 1  # the node saved it, so restarting signs nothing new


def test_someone_elses_key_is_not_given_a_verifier_by_accident(tmp_path: Path) -> None:
    home, _ = older_owner(tmp_path)
    posted: list[dict[str, str]] = []
    with transport(home, [], posted) as client:
        up.give_owner_a_verifier(home, client, "http://a:1", "isk-not-on-the-list")
    assert not posted


def test_a_batch_behind_one_address_takes_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = [httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200, json={"status": "waiting"})]
    waited: list[float] = []
    monkeypatch.setattr(up.time, "sleep", waited.append)
    with httpx.Client(transport=httpx.MockTransport(lambda request: answers.pop(0))) as client:
        answer = up.ask_to_join(client, "http://a:1", {}, up.time.monotonic() + 60)
    assert answer.status_code == 200 and waited == [1.0]


def test_asking_to_join_gives_up_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(up.time, "sleep", lambda _: pytest.fail("must not wait past the deadline"))
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(429, headers={"Retry-After": "100"}))) as client:
        assert up.ask_to_join(client, "http://a:1", {}, up.time.monotonic() + 60).status_code == 429


def test_waiting_for_an_answer_keeps_the_request_from_expiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from isharded import cli

    from isharded.pool import approve_membership

    authority = owner(tmp_path)
    asked: list[dict[str, str]] = []

    def certificate() -> httpx.Response:
        return httpx.Response(200, json={"certificate": approve_membership(str(authority), asked[0]["request"]), "authority": "a"})

    looks = iter([lambda: httpx.Response(404), certificate])

    def post(url: str, **options: object) -> httpx.Response:
        asked.append(options["json"])  # type: ignore[arg-type]
        return httpx.Response(200, json={"status": "waiting", "fingerprint": read_membership_request(asked[0]["request"])["fingerprint"]})

    monkeypatch.setattr(cli, "ASK_AGAIN_SECONDS", -1.0)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    monkeypatch.setattr(cli.httpx, "post", post)
    monkeypatch.setattr(cli.httpx, "get", lambda *_, **__: next(looks)())
    args = argparse.Namespace(url="http://a:1", pool_command="request-join", dir=str(tmp_path / "joiner"), name="joiner", address=["100.64.0.2"], wait=True)
    assert cli.run_membership(args) == 0
    assert len(asked) == 2 and asked[0] == asked[1]


def test_a_key_signed_for_a_running_pool_gets_a_verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from isharded import cli

    sent: list[dict[str, str]] = []
    monkeypatch.setattr(cli, "send_signed", lambda args, entry: sent.append(entry))
    monkeypatch.setattr(cli.sys, "argv", ["isharded", "key", "create", "alice", "--role", "admin", "--dir", str(tmp_path), "--url", "http://a:1"])
    cli.main()
    key = capsys.readouterr().out.strip().splitlines()[-1]
    assert sent[0]["sha256"] == digest(key) and sent[0]["verifier"] == key_verifier(key)


def test_a_mesh_with_no_name_to_give_is_listed_by_its_address(monkeypatch: pytest.MonkeyPatch) -> None:
    from isharded import discovery

    pem = "-----BEGIN CERTIFICATE-----"
    monkeypatch.setattr(discovery.httpx, "get", lambda *_, **__: httpx.Response(200, json={"authority": pem}))
    assert discovery.ask_peer("100.64.0.5") == {"url": "http://100.64.0.5:8209", "authority": up.authority_hash(pem), "name": "100.64.0.5"}


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


def test_a_launch_agent_restarts_a_crash_but_not_a_stop(tmp_path: Path) -> None:
    _, body = up.service_definition("/opt/venv/bin/python", "Darwin", tmp_path)
    assert "<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>" in body  # isharded stop exits cleanly


def test_doctor_reports_the_engine_without_calling_it_a_failure(monkeypatch) -> None:
    monkeypatch.setenv("ISHARDED_ENGINE", "0")
    line = next(item for item in up.checks() if "native engine" in item[1].lower())
    assert line[0] is True and "turned off" in line[1] and "build_engine.py" in line[2]


def test_doctor_does_not_blame_a_port_on_this_machines_own_node(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ISHARDED_UP_HOME", str(tmp_path))
    up.claim_node(tmp_path)
    assert (True, "A node is running here", "") in up.checks()
    assert not any("Port" in item[1] for item in up.checks())


def test_an_engine_only_counts_if_it_loads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ISHARDED_ENGINE", raising=False)
    monkeypatch.setattr(up, "find_library", lambda: None)
    assert up.native_engine() == (False, "not installed")
    bad = tmp_path / "engine"
    bad.write_bytes(b"not a library")
    monkeypatch.setattr(up, "find_library", lambda: bad)
    good, why = up.native_engine()
    assert not good and "will not load" in why


def test_what_was_found_is_said_in_a_few_lines(monkeypatch) -> None:
    monkeypatch.setenv("ISHARDED_ENGINE", "0")
    found = settings() | {"address": "100.64.0.1", "tls": False, "cpu_threads": 4, "gguf_dirs": ["/models"]}
    lines = "\n".join(up.describe_machine(found, "A GPU (8 GiB)"))
    for text in ("100.64.0.1", "Tailscale", "A GPU (8 GiB)", "4 of", "/models", "slower path"):
        assert text in lines
    assert "TLS with this mesh's own certificate" in "\n".join(up.describe_machine(found | {"address": "192.168.1.5", "tls": True}, ""))


def test_one_folder_runs_one_node(tmp_path: Path) -> None:
    assert up.running_node(tmp_path) is None
    assert up.claim_node(tmp_path)
    assert up.running_node(tmp_path).pid == os.getpid()
    assert not up.claim_node(tmp_path)


@pytest.mark.parametrize("record", ["[1, 0]", "not json", "[]"])
def test_a_stale_claim_does_not_block_a_new_node(tmp_path: Path, record: str) -> None:
    (tmp_path / "up.pid").write_text(record, encoding="utf-8")
    assert up.running_node(tmp_path) is None  # PID 1 lives on, but this is not the process that claimed
    assert up.claim_node(tmp_path)


def test_a_reused_pid_is_not_taken_for_the_node(tmp_path: Path) -> None:
    (tmp_path / "up.pid").write_text(json.dumps([os.getpid(), psutil.Process().create_time() - 100]), encoding="utf-8")
    assert up.running_node(tmp_path) is None


def fake_node_args(*flags: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    up.add_arguments(parser.add_subparsers())
    return parser.parse_args(["up", *flags])


def test_up_points_at_a_node_that_is_already_running(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ISHARDED_UP_HOME", str(tmp_path))
    up.write_settings(tmp_path, settings() | {"address": "100.64.0.1", "tls": False})
    up.claim_node(tmp_path)
    monkeypatch.setattr(up, "run_node", lambda _: pytest.fail("a second node started"))
    assert up.run_up(fake_node_args()) == 0
    assert "already running" in capsys.readouterr().out
    assert (tmp_path / "up.pid").is_file()  # the running node keeps its claim


def test_up_gives_up_its_claim_when_the_node_ends(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ISHARDED_UP_HOME", str(tmp_path))
    up.write_settings(tmp_path, settings() | {"address": "100.64.0.1", "tls": False})
    monkeypatch.setattr(up, "keep_watch", lambda *_: None)
    claimed = []
    monkeypatch.setattr(up, "run_node", lambda _: claimed.append((tmp_path / "up.pid").is_file()) or 0)
    assert up.run_up(fake_node_args()) == 0
    assert claimed == [True] and not (tmp_path / "up.pid").exists()


def background(tmp_path: Path, monkeypatch, healthy: bool = True) -> tuple[list, list]:
    started: list = []
    opened: list = []
    monkeypatch.setattr(up.subprocess, "Popen", lambda command, **options: started.append((command, options)))
    monkeypatch.setattr(up, "wait_until_healthy", lambda *_: healthy)
    monkeypatch.setattr(up, "wait_until_settled", lambda *_: True)
    monkeypatch.setattr(up.webbrowser, "open", opened.append)
    return started, opened


def test_background_starts_a_node_that_outlives_it_and_opens_the_page(tmp_path: Path, monkeypatch, capsys) -> None:
    started, opened = background(tmp_path, monkeypatch)
    (tmp_path / "admin.key").write_text("the-key", encoding="utf-8")
    found = settings() | {"address": "100.64.0.1", "port": 8209, "tls": False}
    assert up.start_in_background(tmp_path, found, fake_node_args("--background", "--model", "m")) == 0
    command, options = started[0]
    assert command[-4:] == ["up", "--no-browser", "--model", "m"] and "--background" not in command  # or it would start itself for ever
    assert options["stdin"] == subprocess.DEVNULL and ("creationflags" in options or options["start_new_session"])
    assert options["cwd"] == tmp_path  # not this folder, which may hold another copy of the package
    assert str(Path(up.__file__).resolve().parents[1]) in options["env"]["PYTHONPATH"].split(os.pathsep)
    assert opened == ["http://127.0.0.1:8209/chat#key=the-key"]
    said = capsys.readouterr().out
    assert "http://127.0.0.1:8209/chat" in said and "the-key" not in said and "isharded stop" in said and "Next:" in said


def test_background_opens_the_page_of_a_running_node_and_starts_nothing(tmp_path: Path, monkeypatch) -> None:
    started, opened = background(tmp_path, monkeypatch)
    up.claim_node(tmp_path)
    found = settings() | {"address": "100.64.0.1", "port": 8209, "tls": False}
    assert up.start_in_background(tmp_path, found, fake_node_args("--background")) == 0
    assert not started and opened == ["http://127.0.0.1:8209/chat"]  # an invited machine holds no key, and still opens


def test_background_says_where_to_look_when_the_node_never_comes_up(tmp_path: Path, monkeypatch, capsys) -> None:
    _, opened = background(tmp_path, monkeypatch, healthy=False)
    found = settings() | {"address": "100.64.0.1", "port": 8209, "tls": False}
    assert up.start_in_background(tmp_path, found, fake_node_args("--background")) == 1
    assert "up.log" in capsys.readouterr().err and not opened


def test_no_browser_opens_nothing(tmp_path: Path, monkeypatch) -> None:
    _, opened = background(tmp_path, monkeypatch)
    found = settings() | {"address": "100.64.0.1", "port": 8209, "tls": False}
    assert up.start_in_background(tmp_path, found, fake_node_args("--background", "--no-browser")) == 0
    assert not opened


STOPPABLE = """
import sys, threading, time
from pathlib import Path
from isharded import up
home = Path(sys.argv[1])
assert up.claim_node(home)
threading.Thread(target=up.watch_for_stop, args=(home, threading.Event()), daemon=True).start()
print("ready", flush=True)
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    (home / "stopped").touch()  # run_node stops its services on exactly this
"""


def test_stop_reaches_a_real_node_process_as_a_ctrl_c(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ISHARDED_UP_HOME", str(tmp_path))
    process = subprocess.Popen([sys.executable, "-c", STOPPABLE, str(tmp_path)], stdout=subprocess.PIPE, text=True, cwd=Path(__file__).resolve().parents[1])
    try:
        assert process.stdout.readline().strip() == "ready"
        assert up.run_stop(argparse.Namespace()) == 0
        assert process.wait(timeout=10) == 0
    finally:
        process.kill()
    assert (tmp_path / "stopped").is_file() and not (tmp_path / "stop.request").exists()
    assert "Stopped." in capsys.readouterr().out


def test_stop_kills_a_node_that_will_not_stop(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ISHARDED_UP_HOME", str(tmp_path))
    monkeypatch.setattr(up, "STOP_WAIT_SECONDS", 0.2)
    code = "import json, os, sys, time, psutil; open(sys.argv[1], 'w').write(json.dumps([os.getpid(), psutil.Process().create_time()])); time.sleep(120)"
    process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "up.pid")])
    try:
        deadline = time.monotonic() + 30
        while up.running_node(tmp_path) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert up.run_stop(argparse.Namespace()) == 0
        process.wait(timeout=10)
    finally:
        process.kill()
    assert "killed" in capsys.readouterr().out


def test_stop_with_nothing_running_says_so(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ISHARDED_UP_HOME", str(tmp_path))
    assert up.run_stop(argparse.Namespace()) == 0
    assert "No node is running" in capsys.readouterr().out


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
