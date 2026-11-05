import json
from pathlib import Path
import shutil
import ssl
import subprocess
import threading
import time

from fastapi.testclient import TestClient
import httpx
import pytest
import uvicorn

from shardmesh import orchestrator as orchestrator_module
from shardmesh.access import ROLE_ADMIN, ROLE_NODE, ROLE_USER, AccessControl, AuditLog, create_key, digest
from shardmesh.orchestrator import Orchestrator


@pytest.fixture
def keys(tmp_path):
    path = str(tmp_path / "keys.json")
    return path, create_key(path, "alice", ROLE_USER), create_key(path, "root", ROLE_ADMIN)


def test_a_key_is_stored_only_as_a_hash(keys):
    path, alice, _ = keys
    stored = Path(path).read_text(encoding="utf-8")
    assert alice not in stored
    assert digest(alice) in stored


def test_duplicate_names_are_refused(keys):
    with pytest.raises(ValueError, match="already exists"):
        create_key(keys[0], "alice", ROLE_USER)


def test_roles_rank_as_expected(keys):
    path, alice, root = keys
    access = AccessControl.load("pool-secret", path)
    user = access.identify(f"Bearer {alice}")
    admin = access.identify(f"Bearer {root}")
    node = access.identify("Bearer pool-secret")
    assert user.name == "alice" and access.permits(user, ROLE_USER) and not access.permits(user, ROLE_ADMIN)
    assert access.permits(admin, ROLE_ADMIN) and not access.permits(admin, ROLE_NODE)
    assert access.permits(node, ROLE_NODE)
    assert access.identify("Bearer wrong") is None
    assert access.identify(None) is None


def test_an_open_pool_needs_nothing():
    access = AccessControl()
    assert access.permits(access.identify(None), ROLE_NODE)


def test_rate_limits_refill():
    now = [0.0]
    path_free = AccessControl(clock=lambda: now[0])
    from shardmesh.access import Caller

    caller = Caller("c", ROLE_USER, requests_per_minute=2)
    assert path_free.spend(caller) == 0
    assert path_free.spend(caller) == 0
    assert path_free.spend(caller) > 0
    now[0] += 30
    assert path_free.spend(caller) == 0


def test_the_audit_log_rotates(tmp_path, monkeypatch):
    monkeypatch.setattr("shardmesh.access.AUDIT_ROTATE_BYTES", 10)
    path = tmp_path / "audit.jsonl"
    log = AuditLog(str(path), "n")
    log.record("a", "POST", "/admin/load", 200, "1.2.3.4")
    log.record("a", "POST", "/admin/load", 200, "1.2.3.4")
    assert (tmp_path / "audit.jsonl.1").is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["actor"] == "a"


@pytest.fixture
def secured(monkeypatch, tmp_path, keys):
    path, alice, root = keys
    monkeypatch.setenv("SHARDMESH_KEYS_FILE", path)
    monkeypatch.setenv("SHARDMESH_POOL_TOKEN", "pool-secret")
    monkeypatch.setenv("SHARDMESH_AUDIT_FILE", str(tmp_path / "audit.jsonl"))
    monkeypatch.delenv("SHARDMESH_NODE_URL", raising=False)
    runtime = Orchestrator()
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    return TestClient(orchestrator_module.app), alice, root, tmp_path / "audit.jsonl"


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


def test_users_can_read_but_not_manage(secured):
    client, alice, _, _ = secured
    assert client.get("/admin/status").status_code == 401
    assert client.get("/admin/status", headers=bearer(alice)).status_code == 200
    assert client.post("/admin/rebalance", headers=bearer(alice)).status_code == 403
    assert client.post("/admin/load/cancel", headers=bearer(alice)).status_code == 403
    assert client.post("/admin/plan", headers=bearer(alice), json={"model_id": "m"}).status_code == 403


def test_keys_never_reach_the_node_protocol(secured):
    client, _, root, _ = secured
    assert client.post("/internal/heartbeat", headers=bearer(root), json={"worker_id": "w"}).status_code == 403
    assert client.post("/internal/gossip", headers=bearer(root), json={}).status_code == 403


def test_metrics_need_admin_and_report_the_counters_that_matter(secured):
    client, alice, root, _ = secured
    assert client.get("/metrics", headers=bearer(alice)).status_code == 403
    text = client.get("/metrics", headers=bearer(root)).text
    assert "shardmesh_stream_fallbacks_total 0" in text
    assert "shardmesh_workers_live 0" in text
    assert text.count("# HELP shardmesh_requests_total") == 1


def test_ready_reports_whether_this_node_can_serve(secured):
    client = secured[0]
    response = client.get("/ready")
    assert response.status_code == 503 and response.json()["ready"] is False


def test_changes_are_audited_without_their_content(secured):
    client, alice, root, audit = secured
    client.post("/admin/load/cancel", headers=bearer(root))
    client.post("/v1/chat/completions", headers=bearer(alice), json={"messages": [{"role": "user", "content": "my secret"}]})
    text = audit.read_text(encoding="utf-8")
    entries = [json.loads(line) for line in text.splitlines()]
    assert [entry["actor"] for entry in entries] == ["root", "alice"]
    assert entries[0]["path"] == "/admin/load/cancel"
    assert "my secret" not in text


def test_a_rate_limited_key_is_told_when_to_return(monkeypatch, tmp_path):
    path = str(tmp_path / "keys.json")
    key = create_key(path, "slow", ROLE_USER, requests_per_minute=1)
    monkeypatch.setenv("SHARDMESH_KEYS_FILE", path)
    monkeypatch.delenv("SHARDMESH_POOL_TOKEN", raising=False)
    monkeypatch.delenv("SHARDMESH_NODE_URL", raising=False)
    monkeypatch.setattr(orchestrator_module, "runtime", Orchestrator())
    client = TestClient(orchestrator_module.app)
    assert client.get("/v1/models", headers=bearer(key)).status_code == 200
    limited = client.get("/v1/models", headers=bearer(key))
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_addresses_prefer_the_tailnet(monkeypatch):
    import socket

    import psutil

    from shardmesh.node import detect_address

    class Address:
        family = socket.AF_INET

        def __init__(self, address):
            self.address = address

    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {"lan": [Address("192.168.1.5")], "ts": [Address("100.100.1.2")]})
    assert detect_address() == "100.100.1.2"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")
def test_nodes_can_talk_over_tls_with_a_private_authority(tmp_path, monkeypatch):
    """A real server and client, as a pool of remote machines would use."""
    from shardmesh.serving import NoDelayHTTPProtocol
    from shardmesh.tls import client_verify, server_options

    certificate, key = tmp_path / "node.pem", tmp_path / "node.key"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-keyout", str(key), "-out", str(certificate),
         "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"],
        check=True, capture_output=True,
    )
    monkeypatch.setenv("SHARDMESH_TLS_CERT", str(certificate))
    monkeypatch.setenv("SHARDMESH_TLS_KEY", str(key))
    monkeypatch.setenv("SHARDMESH_CA_FILE", str(certificate))

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"secure"})

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=8466, http=NoDelayHTTPProtocol, lifespan="off", log_level="warning", **server_options())
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        served = ssl.get_server_certificate(("127.0.0.1", 8466))
        if ssl.PEM_cert_to_DER_cert(served) != ssl.PEM_cert_to_DER_cert(certificate.read_text()):
            pytest.skip("something on this machine intercepts TLS, as antivirus web shields do")
        assert httpx.get("https://127.0.0.1:8466/", verify=client_verify(), trust_env=False).text == "secure"
        with pytest.raises(httpx.ConnectError):
            httpx.get("https://127.0.0.1:8466/", verify=ssl.create_default_context(), trust_env=False)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_tls_needs_both_halves(monkeypatch):
    from shardmesh.tls import server_options

    monkeypatch.setenv("SHARDMESH_TLS_CERT", "x")
    monkeypatch.delenv("SHARDMESH_TLS_KEY", raising=False)
    with pytest.raises(ValueError):
        server_options()


def test_an_exposed_pool_with_no_credentials_refuses_to_start():
    from shardmesh.pool import ExposedPool, check_exposure

    with pytest.raises(ExposedPool, match="pool init"):
        check_exposure("0.0.0.0", None, None, False)
    check_exposure("127.0.0.1", None, None, False)
    check_exposure("0.0.0.0", "token", None, False, "http://100.64.0.12:8300")
    check_exposure("0.0.0.0", None, "ca.pem", False, "http://100.64.0.12:8300")
    check_exposure("0.0.0.0", None, None, True)


def test_the_token_comes_from_the_nodes_file(tmp_path, monkeypatch):
    from shardmesh.pool import load_token

    monkeypatch.delenv("SHARDMESH_POOL_TOKEN", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("SHARDMESH_POOL_TOKEN_FILE", raising=False)
    assert load_token() is None
    (tmp_path / "ShardMesh").mkdir()
    (tmp_path / "ShardMesh" / "pool.token").write_text("secret\n", encoding="utf-8")
    assert load_token() == "secret"


def test_a_change_on_one_node_reaches_another(tmp_path):
    first = AccessControl(path=tmp_path / "a.json")
    second = AccessControl(path=tmp_path / "b.json")
    key = first.add_key("alice", ROLE_USER)
    first.revoke_node("bad-node")
    assert second.merge(first.document())
    assert second.identify(f"Bearer {key}").name == "alice"
    assert "bad-node" in second.revoked
    assert json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["version"] == 2


def test_an_old_document_cannot_undo_a_revocation():
    new = AccessControl()
    new.revoke_node("bad-node")
    stale = AccessControl()
    assert not new.merge(stale.document())
    assert "bad-node" in new.revoked


def test_both_sides_of_a_split_settle_on_one_document():
    left, right = AccessControl(), AccessControl()
    left.add_key("l", ROLE_USER)
    right.add_key("r", ROLE_USER)
    left_doc, right_doc = left.document(), right.document()
    left.merge(right_doc)
    right.merge(left_doc)
    assert left.document() == right.document()


def test_a_key_can_be_taken_back():
    access = AccessControl(pool_token="p")
    key = access.add_key("alice", ROLE_USER)
    assert access.remove_key("alice")
    assert access.identify(f"Bearer {key}") is None
    assert not access.remove_key("alice")


def test_a_verified_node_is_trusted_until_revoked():
    access = AccessControl(identities=True)
    assert access.identify(None) is None, "certificates mean nothing is open"
    assert access.permits(access.identify(None, "node-a"), ROLE_NODE)
    access.revoke_node("node-a")
    assert access.identify(None, "node-a") is None


def test_keys_are_managed_through_the_api_and_shown_once(secured):
    client, alice, root, _ = secured
    assert client.post("/admin/keys", headers=bearer(alice), json={"name": "bob"}).status_code == 403
    created = client.post("/admin/keys", headers=bearer(root), json={"name": "bob", "role": "user"}).json()
    assert client.get("/v1/models", headers=bearer(created["key"])).status_code == 200
    listing = client.get("/admin/keys", headers=bearer(root)).json()
    assert {"name": "bob", "role": "user", "requests_per_minute": 0} in listing["keys"]
    assert created["key"] not in json.dumps(listing)
    assert client.delete("/admin/keys/bob", headers=bearer(root)).status_code == 200
    assert client.get("/v1/models", headers=bearer(created["key"])).status_code == 401


def test_revoking_a_node_quarantines_its_workers_and_keeps_them_out(secured):
    from shardmesh.orchestrator import WorkerRecord
    from shardmesh.protocol import HardwareInfo, WorkerRegistration

    client, _, root, _ = secured
    hardware = HardwareInfo(device="cpu", backend="cpu", cpu_cores=1, ram_total_bytes=1)
    registration = WorkerRegistration(worker_id="bad-gpu", advertised_url="http://bad:1", hardware=hardware)
    runtime = orchestrator_module.runtime
    runtime.workers["bad-gpu"] = WorkerRecord(registration=registration, last_seen=time.monotonic(), health={"status": "ok"})
    assert client.post("/admin/nodes/bad/revoke", headers=bearer(root)).status_code == 200
    assert runtime.workers["bad-gpu"].quarantined
    for _ in range(100):
        runtime._review_quarantine()
    assert runtime.workers["bad-gpu"].quarantined, "probation does not undo a revocation"


def test_worker_health_says_nothing_to_strangers(monkeypatch):
    from shardmesh import worker

    class Runtime:
        pool_token = "secret"
        requires_identity = False
        revoked_nodes = set()

    monkeypatch.setattr(worker, "runtime", Runtime())
    client = TestClient(worker.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_pool_credentials_issue_working_certificates(tmp_path):
    from cryptography import x509

    from shardmesh.pool import add_node, init_pool

    folder = str(tmp_path / "pool")
    init_pool(folder)
    with pytest.raises(ValueError, match="already"):
        init_pool(folder)
    certificate_path, key_path = add_node(folder, "node-a", ["10.0.0.5", "node-a.lan"])
    certificate = x509.load_pem_x509_certificate(certificate_path.read_bytes())
    authority = x509.load_pem_x509_certificate((tmp_path / "pool" / "ca.pem").read_bytes())
    certificate.verify_directly_issued_by(authority)
    assert certificate.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier)
    assert authority.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
    assert certificate.subject.rfc4514_string() == "CN=node-a"
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(str(certificate_path), str(key_path))


def test_a_certificate_name_is_read_from_its_subject():
    from shardmesh.tls import peer_name

    assert peer_name({"subject": ((("commonName", "node-a"),),)}) == "node-a"
    assert peer_name(None) is None
    assert peer_name({"subject": ()}) is None
