"""A node that lost its folder comes back under its own name.

A different machine after that name needs the same invite or a person's word, and either
way the owner sees it happen and the old key stops proving the name anywhere.
"""
from __future__ import annotations

import contextlib
import ipaddress
from pathlib import Path
import shutil

import pytest

pytest.importorskip("cryptography")

from fastapi.testclient import TestClient

from isharded import orchestrator, tls, up
from isharded.access import AccessControl, Authority, digest
from isharded.pool import certificate_fingerprint, request_membership, sign_entry
from isharded.tickets import TicketChecker, TicketMaker


class Shared:
    """One in-process mesh node for every client ``up`` opens, without its startup running."""

    def __init__(self, client: TestClient) -> None:
        self.client = client

    def __enter__(self) -> TestClient:
        return self.client

    def __exit__(self, *_: object) -> None:
        return None


@pytest.fixture
def owners_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "owner"
    up.make_owner(home, "first", "100.64.0.1")
    monkeypatch.setenv("ISHARDED_CA_FILE", str(home / "ca.pem"))
    monkeypatch.setenv("ISHARDED_UP_HOME", str(home))
    runtime = orchestrator.Orchestrator()
    runtime.access = AccessControl.load(None, str(home / "access.json"), identities=True, authority=Authority(str(home / "ca.pem")))
    monkeypatch.setattr(orchestrator, "runtime", runtime)
    client = TestClient(orchestrator.app)
    monkeypatch.setattr(up, "mesh_client", lambda *_, **__: Shared(client))
    key = (home / "admin.key").read_text(encoding="utf-8")
    # The owner's machine signs in whoever it should each time a joiner looks again.
    monkeypatch.setattr(up.time, "sleep", lambda _: up.approve_invited(home, client, "http://testserver", key))
    monkeypatch.setattr(up, "PERSON_WAIT_SECONDS", 1.0)
    monkeypatch.setattr(up, "APPROVAL_WAIT_SECONDS", 3.0)
    yield home, runtime, client, key
    tls.RETIRED_KEYS.clear()


def invite_for(home: Path, secret: bool = True) -> dict[str, object]:
    invite = up.read_invite(up.create_invite(home, ["http://testserver"], uses=1)["code"])
    return invite if secret else {key: value for key, value in invite.items() if key != "secret"}


def key_of(folder: Path, name: str) -> str:
    return str(certificate_fingerprint((folder / f"{name}.pem").read_text(encoding="utf-8")))


def events(runtime: orchestrator.Orchestrator) -> list[str]:
    return [f"{event['title']}: {event['detail']}" for event in runtime.events]


def request_with_key(key_file: Path, name: str) -> str:
    """A request made with a key someone kept, as whoever copied a lost folder could."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.x509.oid import NameOID

    key = serialization.load_pem_private_key(key_file.read_bytes(), password=None)
    addresses = x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("100.64.0.2"))])
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    request = x509.CertificateSigningRequestBuilder().subject_name(subject).add_extension(addresses, critical=False).sign(key, hashes.SHA256())
    return request.public_bytes(serialization.Encoding.PEM).decode()


def test_a_node_whose_folder_was_wiped_rejoins_under_its_name_with_one_invite(owners_node, tmp_path: Path) -> None:
    home, runtime, client, _ = owners_node
    folder = tmp_path / "second"
    up.join_mesh(folder, invite_for(home), "second", "100.64.0.2")
    first_key, first_user_key = key_of(folder, "second"), digest((folder / "user.key").read_text(encoding="utf-8"))
    kept = tmp_path / "kept"  # what someone who copied the lost folder would still hold
    shutil.copytree(folder, kept)
    shutil.rmtree(folder)

    up.join_mesh(folder, invite_for(home), "second", "100.64.0.2")

    assert up.certificate_matches_key((folder / "second.pem").read_text(encoding="utf-8"), folder / "second.key")
    assert key_of(folder, "second") != first_key
    assert runtime.access.retired_keys == {first_key}
    assert client.get("/join/certificate/second").json()["certificate"] == (folder / "second.pem").read_text(encoding="utf-8")
    # The old folder's API key goes with its node key, and the new one works.
    assert first_user_key not in runtime.access.keys and first_user_key in runtime.access.removed
    assert digest((folder / "user.key").read_text(encoding="utf-8")) in runtime.access.keys
    # The old key cannot speak for the name any more, and the new one can.
    checker = TicketChecker(str(home / "ca.pem"), {"b:1"})
    assert checker.name(TicketMaker(str(kept / "second.pem"), str(kept / "second.key")).ticket("b:1")) is None
    assert checker.name(TicketMaker(str(folder / "second.pem"), str(folder / "second.key")).ticket("b:1")) == "second"
    assert any(line.startswith("A node rejoined with a new key: second") and first_key in line for line in events(runtime))
    assert not runtime.access.join_requests
    # Nor can the old key ask to come back.
    assert client.post("/join/requests", json={"request": request_with_key(kept / "second.key", "second")}).json()["status"] == "refused"


def test_another_machine_cannot_take_a_name_without_an_invite_or_the_owner(owners_node, tmp_path: Path) -> None:
    home, runtime, client, key = owners_node
    member = tmp_path / "second"
    up.join_mesh(member, invite_for(home), "second", "100.64.0.2")
    held = key_of(member, "second")
    stranger = tmp_path / "stranger"
    with pytest.raises(ValueError, match="another machine holds the name second"):
        up.join_mesh(stranger, invite_for(home, secret=False), "second", "100.64.0.9")
    # Nothing changed hands, and nobody had to notice by chance.
    assert runtime.access.certificate_key("second") == held and not runtime.access.retired_keys
    assert not (stranger / "second.pem").exists()
    waiting = client.get("/join/requests", headers={"Authorization": f"Bearer {key}"}).json()["requests"]
    assert [(item["name"], item["retires"]["fingerprint"]) for item in waiting] == [("second", held)]
    assert any(line.startswith("A member's name was asked for with a new key: second") and held in line for line in events(runtime))
    # The owner can still let it in, and that retires the key the name had.
    asked = waiting[0]["fingerprint"]
    assert client.post("/admin/join/second/approve", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    assert up.approve_invited(home, client, "http://testserver", key) == ["second"]
    assert runtime.access.retired_keys == {held}
    assert runtime.access.certificate_key("second") == asked


def test_a_person_approving_one_machine_does_not_approve_the_next_to_ask_for_its_name(owners_node, tmp_path: Path) -> None:
    home, runtime, client, key = owners_node
    headers = {"Authorization": f"Bearer {key}"}
    first = request_membership(str(tmp_path / "a"), "second", ["100.64.0.2"])
    assert client.post("/join/requests", json={"request": first}).json()["status"] == "waiting"
    assert client.post("/admin/join/second/approve", headers=headers).status_code == 200
    assert up.approve_invited(home, client, "http://testserver", key) == ["second"]
    held = runtime.access.certificate_key("second")
    later = request_membership(str(tmp_path / "b"), "second", ["100.64.0.9"])
    answer = client.post("/join/requests", json={"request": later}).json()
    assert answer["status"] == "waiting" and answer["replaces"] == held
    assert up.approve_invited(home, client, "http://testserver", key) == []
    assert runtime.access.certificate_key("second") == held and not runtime.access.retired_keys


def test_the_same_key_asking_again_is_given_its_certificate(owners_node, tmp_path: Path) -> None:
    home, runtime, client, _ = owners_node
    folder = tmp_path / "second"
    up.join_mesh(folder, invite_for(home), "second", "100.64.0.2")
    again = request_with_key(folder / "second.key", "second")
    assert client.post("/join/requests", json={"request": again}).json() == {"status": "approved", "fingerprint": key_of(folder, "second")}
    assert not runtime.access.join_requests and not runtime.access.retired_keys


def test_a_request_cannot_choose_which_key_it_retires(owners_node, tmp_path: Path) -> None:
    """What is retired comes from the owner's own node, never from what travels with a request."""
    home, runtime, client, key = owners_node
    up.join_mesh(tmp_path / "second", invite_for(home), "second", "100.64.0.2")
    up.join_mesh(tmp_path / "third", invite_for(home), "third", "100.64.0.3")
    victim = key_of(tmp_path / "third", "third")
    text = request_membership(str(tmp_path / "new"), "fourth", ["100.64.0.4"])
    runtime.access.merge({"join_requests": [{"request": text, "retires": {"fingerprint": victim, "keys": []}}], "signed": []})
    listed = client.get("/join/requests", headers={"Authorization": f"Bearer {key}"}).json()["requests"]
    assert [item.get("retires") for item in listed] == [None]


def entries(folder: str, request: str, retired: str | None = None) -> list[dict[str, object]]:
    from isharded.pool import admission_entries

    held = {"request": request, **({"retires": {"fingerprint": retired, "keys": []}} if retired else {})}
    return [sign_entry(folder, entry) for entry in admission_entries(folder, held)]


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1)])
def test_every_node_settles_on_the_same_certificate_whatever_order_entries_come_in(tmp_path: Path, order: tuple[int, ...]) -> None:
    import time

    home = tmp_path / "owner"
    up.make_owner(home, "first", "100.64.0.1")
    old = entries(str(home), request_membership(str(tmp_path / "a"), "second", ["100.64.0.2"]))
    time.sleep(1.1)  # the newer signature carries a later time
    newer = request_membership(str(tmp_path / "b"), "second", ["100.64.0.2"])
    retired = certificate_fingerprint(old[0]["entry"]["certificate"])
    retire, member = entries(str(home), newer, retired)
    arriving = [old[0], retire, member]
    access = AccessControl(identities=True, authority=Authority(str(home / "ca.pem")))
    for index in order:
        assert access.add_signed(arriving[index])
    assert access.certificates["second"] == member["entry"]["certificate"]
    assert access.retired_keys == {retired}


def test_a_connection_whose_key_was_retired_proves_no_name(tmp_path: Path) -> None:
    from isharded.tls import retired_certificate

    home = tmp_path / "owner"
    up.make_owner(home, "first", "100.64.0.1")
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    der = x509.load_pem_x509_certificate((home / "first.pem").read_bytes()).public_bytes(serialization.Encoding.DER)
    try:
        assert not retired_certificate(der)
        tls.retire_keys(["0" * 16])
        assert not retired_certificate(der)
        tls.retire_keys([str(certificate_fingerprint(der))])
        assert retired_certificate(der)
    finally:
        tls.RETIRED_KEYS.clear()


def test_a_ticket_checked_before_its_key_was_retired_stops_working(tmp_path: Path) -> None:
    home = tmp_path / "owner"
    up.make_owner(home, "first", "100.64.0.1")
    checker = TicketChecker(str(home / "ca.pem"), {"b:1"})
    ticket = TicketMaker(str(home / "first.pem"), str(home / "first.key")).ticket("b:1")
    try:
        assert checker.name(ticket) == "first"
        tls.retire_keys([key_of(home, "first")])
        assert checker.name(ticket) is None
    finally:
        tls.RETIRED_KEYS.clear()


def test_a_worker_takes_only_signed_retirements(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from isharded import worker

    home = tmp_path / "owner"
    up.make_owner(home, "first", "100.64.0.1")
    runtime = SimpleNamespace(authority=Authority(str(home / "ca.pem")), revoked_nodes=set())
    good = sign_entry(str(home), {"type": "retire_key", "name": "x", "fingerprint": "a" * 16, "keys": []})
    forged = {**good, "entry": {**good["entry"], "fingerprint": "b" * 16}}
    try:
        worker.WorkerRuntime._take_revocations(runtime, {"signed_retirements": [good, forged]})
        assert tls.RETIRED_KEYS == {"a" * 16}
    finally:
        tls.RETIRED_KEYS.clear()


def test_a_joiner_on_the_command_line_waits_for_a_certificate_for_its_own_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    import httpx

    from isharded import cli

    home = tmp_path / "owner"
    up.make_owner(home, "first", "100.64.0.1")
    someone_elses = (home / "first.pem").read_text(encoding="utf-8")
    looked: list[str] = []

    def get(url: str, **_: object) -> httpx.Response:
        looked.append(url)
        if len(looked) == 1:
            return httpx.Response(200, json={"certificate": someone_elses, "authority": "a"})
        from isharded.pool import approve_membership

        mine = approve_membership(str(home), (tmp_path / "joiner" / "asked.csr").read_text(encoding="utf-8"))
        return httpx.Response(200, json={"certificate": mine, "authority": "a"})

    def post(url: str, **options: object) -> httpx.Response:
        request = options["json"]["request"]  # type: ignore[index]
        (tmp_path / "joiner" / "asked.csr").write_text(request, encoding="utf-8")
        from isharded.pool import read_membership_request

        return httpx.Response(200, json={"status": "waiting", "fingerprint": read_membership_request(request)["fingerprint"], "replaces": key_of(home, "first")})

    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    monkeypatch.setattr(cli.httpx, "post", post)
    monkeypatch.setattr(cli.httpx, "get", get)
    args = argparse.Namespace(url="http://a:1", pool_command="request-join", dir=str(tmp_path / "joiner"), name="first", address=["100.64.0.2"], wait=True)
    assert cli.run_membership(args) == 0
    assert len(looked) == 2, "the name's old certificate was not taken for its own"
    assert up.certificate_matches_key((tmp_path / "joiner" / "first.pem").read_text(encoding="utf-8"), tmp_path / "joiner" / "first.key")


@contextlib.contextmanager
def serving_at(app, **options):
    """A real server with the node's own protocol, where a connection's name is read."""
    import asyncio
    import socket
    import threading
    import time

    import uvicorn

    from isharded.serving import NoDelayHTTPProtocol

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off", http=NoDelayHTTPProtocol, **options))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_a_tls_connection_stops_proving_its_name_once_its_key_is_retired(tmp_path: Path) -> None:
    """Over real mutual TLS, on one kept-alive connection, as nodes hold theirs for an hour."""
    import ssl

    import httpx

    from isharded.pool import add_node

    home = tmp_path / "owner"
    up.make_owner(home, "first", "100.64.0.1")
    certificate, key = add_node(str(home), "second", ["127.0.0.1"])

    async def who(scope, receive, send):
        name = (tls.PEER.get() or "nobody").encode()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", str(len(name)).encode())]})
        await send({"type": "http.response.body", "body": name})

    options = {"ssl_certfile": str(home / "first.pem"), "ssl_keyfile": str(home / "first.key"), "ssl_ca_certs": str(home / "ca.pem"), "ssl_cert_reqs": ssl.CERT_OPTIONAL}
    context = ssl.create_default_context(cafile=str(home / "ca.pem"))
    context.load_cert_chain(str(certificate), str(key))
    try:
        with serving_at(who, **options) as port, httpx.Client(verify=context, trust_env=False) as client:
            assert client.get(f"https://127.0.0.1:{port}/").text == "second"
            tls.retire_keys([str(certificate_fingerprint(Path(certificate).read_text(encoding="utf-8")))])
            assert client.get(f"https://127.0.0.1:{port}/").text == "nobody"
    finally:
        tls.RETIRED_KEYS.clear()
