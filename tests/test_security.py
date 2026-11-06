"""Each test here is an attack from the security review that must stay closed."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.testclient import TestClient
import httpx
from jinja2.exceptions import TemplateError
from pydantic import ValidationError
import pytest

from shardmesh import orchestrator as orchestrator_module
from shardmesh import worker as worker_module
from shardmesh.access import ROLE_ADMIN, ROLE_NODE, ROLE_USER, AccessControl, node_of
from shardmesh.llamacpp import TokenEmbedding
from shardmesh.mesh import RUMOR_SECONDS, ControlMesh, clean_url
from shardmesh.orchestrator import FORWARD_MARK, Orchestrator, verify_worker_name
from shardmesh.pool import ExposedPool, check_exposure
from shardmesh.protocol import ChatCompletionRequest, CompletionRequest, DraftModel, TensorRef
from shardmesh.tls import PEER
from shardmesh.tokenizer import template_json
from shardmesh.weights import UnitCache


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


def connected_as(app, peer):
    """What the server's TLS layer does for a verified client certificate."""

    async def with_identity(scope, receive, send):
        PEER.set(peer)
        await app(scope, receive, send)

    return with_identity


@pytest.fixture
def pool_with_authority(monkeypatch):
    runtime = Orchestrator()
    runtime.access = AccessControl(identities=True)
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    return runtime


def test_a_forwarded_request_gets_no_credit_for_the_connection(pool_with_authority):
    client = TestClient(connected_as(orchestrator_module.app, "follower"))
    made = client.post("/admin/keys", json={"name": "x", "role": "admin"}, headers={FORWARD_MARK: "follower"})
    assert made.status_code == 401
    assert not pool_with_authority.access.keys


def test_a_certificate_is_good_for_node_routes_only(pool_with_authority):
    client = TestClient(connected_as(orchestrator_module.app, "node-a"))
    assert client.post("/admin/keys", json={"name": "x", "role": "admin"}).status_code == 403
    assert client.get("/v1/models").status_code == 403
    assert client.get("/internal/mesh").status_code == 200


def test_a_key_still_works_over_a_certified_connection():
    access = AccessControl(identities=True)
    key = access.add_key("alice", ROLE_ADMIN)
    assert access.permits(access.identify(f"Bearer {key}", "node-a"), ROLE_ADMIN)


def test_dot_segments_cannot_turn_a_forward_into_a_node_route(monkeypatch):
    def handler(request):
        raise AssertionError("must not be forwarded")

    runtime = Orchestrator()
    runtime.mesh = ControlMesh("me", "http://me", (), Clock())
    runtime.mesh.leader_id = "boss"
    runtime.mesh.observe({"node_id": "boss", "url": "http://boss:9", "term": 1, "leader_id": "boss"})
    runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    client = TestClient(orchestrator_module.app)
    assert client.post("/admin/%2e%2e/internal/register", json={}).status_code == 400


def test_the_pool_token_means_nothing_once_nodes_have_certificates():
    access = AccessControl(pool_token="stolen", identities=True)
    assert access.identify("Bearer stolen") is None


def test_the_token_stays_a_node_credential_without_an_authority():
    access = AccessControl(pool_token="p")
    assert access.permits(access.identify("Bearer p"), ROLE_NODE)


def test_a_node_speaks_only_for_its_own_workers(pool_with_authority):
    held = PEER.set("a")
    try:
        verify_worker_name("a-gpu")
        with pytest.raises(HTTPException) as refused:
            verify_worker_name("b-gpu")
        assert refused.value.status_code == 403
    finally:
        PEER.reset(held)


def test_a_revoked_node_cannot_register_under_a_new_worker_name(pool_with_authority):
    pool_with_authority.access.revoke_node("a")
    with pytest.raises(HTTPException):
        verify_worker_name("a-cpu")


def test_node_names_come_from_the_whole_worker_name():
    assert node_of("desktop-2-gpu") == "desktop-2"
    assert node_of("node-a-memory") == "node-a"
    assert node_of("standalone") == "standalone"


def test_quarantine_survives_going_quiet_and_registering_again():
    from test_resilience import pool, record, registration

    held = record("w1", quarantined=True, quarantine_reason="bad", quarantine_count=2)
    held.last_seen -= 60
    orchestrator = pool(w1=held)
    orchestrator.register(registration("w1"))
    assert orchestrator.workers["w1"].quarantined
    assert orchestrator.workers["w1"].quarantine_count == 2
    asyncio.run(orchestrator.close())


def test_gossip_cannot_claim_another_nodes_name(pool_with_authority):
    pool_with_authority.mesh = ControlMesh("me", "https://me", (), Clock())
    client = TestClient(connected_as(orchestrator_module.app, "node-a"))
    claim = {"node_id": "node-b", "url": "https://evil", "term": 9, "leader_id": "node-b"}
    assert client.post("/internal/gossip", json=claim).status_code == 403
    assert "node-b" not in pool_with_authority.mesh.peers


@pytest.mark.parametrize("bad", ["http://a\x01", "ftp://a", "http://", "http://a/path", "http://u@a", "http://a:port"])
def test_bad_node_addresses_are_refused(bad):
    with pytest.raises(ValueError):
        clean_url(bad)


def test_a_bad_rumor_costs_nothing_and_old_ones_fade():
    clock = Clock()
    mesh = ControlMesh("me", "http://me", (), clock)
    mesh.observe({"node_id": "a", "url": "http://a", "peers": ["http://b", "http://bad\x01"]})
    assert mesh.targets() == {"http://a", "http://b"}
    clock.now += RUMOR_SECONDS + 1
    assert mesh.targets() == {"http://a"}


def test_a_bad_address_cannot_stop_an_election():
    orchestrator = Orchestrator()
    clock = Clock()
    orchestrator.mesh = ControlMesh("me", "http://me", (), clock)
    orchestrator.mesh.rumors["http://[bad"] = clock.now
    clock.now += 8

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
        await orchestrator._gossip_round()
        await orchestrator.close()

    asyncio.run(scenario())
    assert orchestrator.mesh.is_leader


def test_revoked_nodes_are_not_called_or_heard_again():
    orchestrator = Orchestrator()
    orchestrator.mesh = ControlMesh("me", "http://me", (), Clock())
    orchestrator.mesh.observe({"node_id": "bad", "url": "http://bad"})
    orchestrator.access.revoke_node("bad")
    orchestrator.mesh.forget(orchestrator.access.revoked)
    assert orchestrator.mesh.targets() == set()
    orchestrator.receive_gossip({"node_id": "bad", "url": "http://bad", "access": {"version": 99, "keys": []}})
    assert "bad" not in orchestrator.mesh.peers
    assert orchestrator.access.version == 1
    orchestrator.mesh.observe({"node_id": "a", "url": "http://a", "peers": ["http://bad"]})
    assert orchestrator.mesh.targets() == {"http://a"}
    asyncio.run(orchestrator.close())


def test_a_newer_document_cannot_undo_a_revocation_or_a_removed_key():
    access = AccessControl(pool_token="p")
    key = access.add_key("alice", ROLE_USER)
    held = access.document()
    access.remove_key("alice")
    access.revoke_node("bad")
    forged = {**held, "version": 500, "revoked_nodes": []}
    assert access.merge(forged)
    assert "bad" in access.revoked
    assert access.identify(f"Bearer {key}") is None
    assert access.version == 500


def test_both_sides_of_a_split_keep_what_the_other_did():
    left, right = AccessControl(), AccessControl()
    left.revoke_node("a")
    first = left.add_key("alice", ROLE_USER)
    right.revoke_node("b")
    second = right.add_key("bob", ROLE_USER)
    theirs = right.document()
    right.merge(left.document())
    left.merge(theirs)
    assert left.document() == right.document()
    assert left.revoked == {"a", "b"}
    assert left.identify(f"Bearer {first}") and left.identify(f"Bearer {second}")


def test_cordons_follow_the_newer_document():
    left, right = AccessControl(), AccessControl()
    left.cordon("a")
    right.merge(left.document())
    right.uncordon("a")
    left.merge(right.document())
    assert not left.cordoned


@pytest.mark.parametrize("model", [CompletionRequest, ChatCompletionRequest])
def test_a_seed_that_would_crash_a_stage_is_refused_at_the_door(model):
    fields = {"prompt": "x"} if model is CompletionRequest else {"messages": []}
    with pytest.raises(ValidationError):
        model(seed=2**64, **fields)
    with pytest.raises(ValidationError):
        model(seed=-1, **fields)
    assert model(seed=2**63 - 1, **fields).seed == 2**63 - 1


def test_a_stage_refusing_a_request_costs_nobody_their_place():
    from test_resilience import pool, record, stage
    from shardmesh.orchestrator import SessionState

    orchestrator = pool(w1=record("w1"))

    async def scenario():
        orchestrator.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(500, json={"detail": {"stage_error": "bad seed"}}))
        )
        orchestrator.route = [stage("w1")]
        session = SessionState(tokens=[1], epoch=orchestrator.epoch, needs_reset=False)
        with pytest.raises(HTTPException) as failed:
            await orchestrator.next_token(session, 0.0, 1.0, 0, None)
        await orchestrator.close()
        return failed.value

    assert asyncio.run(scenario()).status_code == 502
    assert not orchestrator.workers["w1"].quarantined
    assert orchestrator.stage_errors == 1


@pytest.mark.parametrize("name", [r"C:\Users\x\node.key", "/etc/passwd", "../up.bin", r"a\..\b", "a/../b", ""])
def test_a_tensor_cannot_point_outside_its_model(name):
    with pytest.raises(ValidationError):
        TensorRef(name="t", file=name, dtype="U8", shape=[1], start=0, end=1)


def test_a_tensor_may_sit_in_a_subfolder():
    assert TensorRef(name="t", file="q4/model.safetensors", dtype="U8", shape=[1], start=0, end=1).file


@pytest.mark.parametrize("name", [r"..\Startup\run.bat", "run.bat", "../x.gguf"])
def test_a_draft_cannot_write_anything_but_a_gguf_inside_the_cache(name):
    with pytest.raises(ValidationError):
        DraftModel(source="s", file=name, file_size=1, revision="r")


def test_the_cache_refuses_names_that_climb_out(tmp_path):
    cache = UnitCache(tmp_path)
    with pytest.raises(ValueError):
        cache.directory("m", r"..\..\out")
    with pytest.raises(ValueError):
        cache.path("m", "r", "../unit")
    assert cache.held("m", "../..") == {}


def test_only_offered_folders_can_be_a_model_source(tmp_path, monkeypatch):
    offered, private = tmp_path / "models", tmp_path / "private"
    (offered / "one").mkdir(parents=True)
    private.mkdir()
    runtime = SimpleNamespace(
        pool_token="t",
        requires_identity=False,
        unit_cache=SimpleNamespace(local_dirs=(offered,), root=tmp_path / "cache"),
    )
    monkeypatch.setattr(worker_module, "runtime", runtime)
    worker_module.verify_source(str(offered / "one"))
    worker_module.verify_source("Qwen/Qwen3-8B")
    worker_module.verify_source("held-by-a-peer.gguf")
    with pytest.raises(HTTPException) as refused:
        worker_module.verify_source(str(private))
    assert refused.value.status_code == 403
    monkeypatch.setenv("SHARDMESH_MODEL_DIRS", str(private))
    worker_module.verify_source(str(private))


def test_a_worker_with_an_authority_ignores_the_token(monkeypatch):
    runtime = SimpleNamespace(pool_token="stolen", requires_identity=True, revoked_nodes=set())
    monkeypatch.setattr(worker_module, "runtime", runtime)
    assert not worker_module.authorized("Bearer stolen")
    held = PEER.set("a")
    try:
        assert worker_module.authorized(None)
    finally:
        PEER.reset(held)


def test_credentials_do_not_cross_a_plain_network(monkeypatch):
    monkeypatch.delenv("SHARDMESH_TLS_CERT", raising=False)
    with pytest.raises(ExposedPool, match="unencrypted"):
        check_exposure("0.0.0.0", "token", None, False, "http://192.168.1.20:8000")
    check_exposure("0.0.0.0", "token", None, False, "http://192.168.1.20:8000", trusted_network=True)
    check_exposure("0.0.0.0", "token", None, False, "http://100.64.0.12:8300")
    check_exposure("0.0.0.0", "token", None, False, "desktop.tail1234.ts.net")
    check_exposure("0.0.0.0", "token", None, False, "http://127.0.0.1:8100")
    monkeypatch.setenv("SHARDMESH_TLS_CERT", "node.pem")
    check_exposure("0.0.0.0", "token", None, False, "https://192.168.1.20:8000")


def test_a_chat_template_cannot_pass_code_to_json():
    assert template_json({"b": 1, "a": 2}, indent=1, sort_keys=True).startswith("{\n")
    with pytest.raises(TypeError):
        template_json({}, default=str)


def test_a_chat_template_cannot_reach_the_process_through_tojson():
    from shardmesh.tokenizer import ChatTokenizer as Holder

    source = '{{ raise_exception|tojson(default="{0.__globals__}".format) }}'
    tokenizer = Holder.__new__(Holder)
    tokenizer.chat_template, tokenizer.special_tokens = source, {}
    with pytest.raises((TypeError, TemplateError)):
        tokenizer.render_chat([{"role": "user", "content": "x"}])


def test_token_ids_stay_inside_the_embedding_table():
    table = TokenEmbedding.__new__(TokenEmbedding)
    table.width, table.row_bytes, table.data, table.to_float = 2, 8, bytes(16), None
    assert table.rows([1]).shape == (1, 2)
    with pytest.raises(ValueError):
        table.rows([2])
    with pytest.raises(ValueError):
        table.rows([-1])


def test_pages_escape_what_workers_report():
    pages = Path(orchestrator_module.__file__).parent
    assert "${esc(TIER[kind] || kind)}" in (pages / "chat.html").read_text(encoding="utf-8")
    dashboard = (pages / "dashboard.html").read_text(encoding="utf-8")
    assert "${esc(TIERS[kind]?.[0]||kind)}" in dashboard
    assert "Number(assignment.start_layer)" in dashboard


def test_a_page_on_a_dead_node_can_still_fail_over_but_a_revoked_one_cannot():
    clock = Clock()
    mesh = ControlMesh("me", "http://me", (), clock)
    mesh.observe({"node_id": "a", "url": "http://a"})
    clock.now += 1000
    assert "http://a" in mesh.known_urls()
    mesh.forget(["a"])
    assert "http://a" not in mesh.known_urls()


@pytest.fixture
def signed_pool(tmp_path):
    from shardmesh.access import Authority
    from shardmesh.pool import init_pool

    pytest.importorskip("cryptography")
    folder = init_pool(str(tmp_path / "pool"))
    return str(folder), AccessControl(identities=True, authority=Authority(str(folder / "ca.pem")))


def key_entry(name="alice", role="admin", key="smk-test"):
    from shardmesh.access import digest

    return {"type": "key", "name": name, "role": role, "sha256": digest(key)}


def test_a_signed_key_spreads_and_works(signed_pool):
    from shardmesh.access import Authority
    from shardmesh.pool import sign_entry

    folder, access = signed_pool
    assert access.add_signed(sign_entry(folder, key_entry()))
    assert access.permits(access.identify("Bearer smk-test"), ROLE_ADMIN)
    peer = AccessControl(identities=True, authority=Authority(folder + "/ca.pem"))
    assert peer.merge(access.document())
    assert peer.permits(peer.identify("Bearer smk-test"), ROLE_ADMIN)


def test_a_compromised_node_cannot_add_a_key_through_gossip(signed_pool):
    from shardmesh.access import digest

    _, access = signed_pool
    forged = {
        "version": 999,
        "keys": [{"name": "evil", "role": "admin", "sha256": digest("evil")}],
        "revoked_nodes": ["good-node"],
        "signed": [{"entry": key_entry("evil", key="evil"), "signature": "AAAA"}],
    }
    access.merge(forged)
    assert access.identify("Bearer evil") is None
    assert "good-node" not in access.revoked


def test_an_entry_signed_by_another_authority_is_ignored(signed_pool, tmp_path):
    from shardmesh.pool import init_pool, sign_entry

    _, access = signed_pool
    other = init_pool(str(tmp_path / "other"))
    assert not access.add_signed(sign_entry(str(other), key_entry()))
    assert access.identify("Bearer smk-test") is None


def test_changing_a_signed_entry_breaks_it(signed_pool):
    from shardmesh.pool import sign_entry

    folder, access = signed_pool
    signed = sign_entry(folder, key_entry(role="user"))
    signed["entry"]["role"] = "admin"
    assert not access.add_signed(signed)


def test_a_signed_revocation_and_removal_stick(signed_pool):
    from shardmesh.pool import sign_entry

    folder, access = signed_pool
    held = access.document()
    access.add_signed(sign_entry(folder, key_entry()))
    access.add_signed(sign_entry(folder, {"type": "remove_key", "name": "alice"}))
    access.add_signed(sign_entry(folder, {"type": "revoke", "name": "bad"}))
    access.merge({**held, "version": 500})
    assert "bad" in access.revoked
    assert access.identify("Bearer smk-test") is None
    late = sign_entry(folder, key_entry(key="again"))
    access.add_signed(late)
    assert access.identify("Bearer again") is None, "a removed name stays removed"


def test_signed_entries_survive_a_restart(signed_pool, tmp_path):
    from shardmesh.access import Authority
    from shardmesh.pool import sign_entry

    folder, _ = signed_pool
    path = tmp_path / "access.json"
    authority = Authority(folder + "/ca.pem")
    access = AccessControl.load(None, str(path), identities=True, authority=authority)
    access.add_signed(sign_entry(folder, key_entry()))
    again = AccessControl.load(None, str(path), identities=True, authority=authority)
    assert again.permits(again.identify("Bearer smk-test"), ROLE_ADMIN)


def test_with_an_authority_nodes_cannot_change_access_themselves(signed_pool):
    from shardmesh.access import NeedsAuthority

    _, access = signed_pool
    for change in (lambda: access.add_key("x", ROLE_USER), lambda: access.remove_key("x"), lambda: access.revoke_node("x")):
        with pytest.raises(NeedsAuthority):
            change()


def test_the_api_takes_signed_entries_and_refuses_the_rest(signed_pool, monkeypatch):
    from shardmesh.pool import sign_entry

    folder, access = signed_pool
    runtime = Orchestrator()
    runtime.access = access
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    client = TestClient(orchestrator_module.app)
    assert client.post("/access/entries", json={"entry": key_entry(), "signature": "AAAA"}).status_code == 403
    assert client.post("/access/entries", json=sign_entry(folder, key_entry())).status_code == 200
    admin = {"Authorization": "Bearer smk-test"}
    assert client.post("/admin/keys", json={"name": "x"}, headers=admin).status_code == 409
    assert client.post("/admin/nodes/x/revoke", headers=admin).status_code == 409


@pytest.fixture
def ticket_pool(tmp_path):
    from shardmesh.pool import add_node, init_pool

    pytest.importorskip("cryptography")
    folder = init_pool(str(tmp_path / "pool"))
    add_node(str(folder), "node-a", ["10.0.0.1"])
    return folder


def maker(folder, name="node-a", clock=None):
    from shardmesh.tickets import TicketMaker

    options = {"clock": clock} if clock else {}
    return TicketMaker(str(folder / f"{name}.pem"), str(folder / f"{name}.key"), **options)


def checker(folder, audiences=("b:1",), clock=None):
    from shardmesh.tickets import TicketChecker

    options = {"clock": clock} if clock else {}
    return TicketChecker(str(folder / "ca.pem"), set(audiences), **options)


def test_a_ticket_proves_a_nodes_name(ticket_pool):
    assert checker(ticket_pool).name(maker(ticket_pool).ticket("b:1")) == "node-a"


def test_a_ticket_only_works_at_the_node_it_names(ticket_pool):
    assert checker(ticket_pool).name(maker(ticket_pool).ticket("elsewhere:1")) is None


def test_a_ticket_expires(ticket_pool):
    clock = Clock()
    clock.now = 1_800_000_000.0
    ticket = maker(ticket_pool, clock=clock).ticket("b:1")
    check = checker(ticket_pool, clock=clock)
    assert check.name(ticket) == "node-a"
    clock.now += 121
    assert check.name(ticket) is None


def test_a_ticket_from_another_authority_is_refused(ticket_pool, tmp_path):
    from shardmesh.pool import add_node, init_pool

    other = init_pool(str(tmp_path / "other"))
    add_node(str(other), "node-a", ["10.0.0.1"])
    assert checker(ticket_pool).name(maker(other).ticket("b:1")) is None


def test_an_altered_ticket_is_refused(ticket_pool):
    import json

    from shardmesh.tickets import PREFIX, _decode, _encode

    ticket = maker(ticket_pool).ticket("b:1")
    body, signature = ticket[len(PREFIX):].split(".")
    claims = json.loads(_decode(body))
    claims["aud"] = "c:1"
    forged = PREFIX + _encode(json.dumps(claims).encode()) + "." + signature
    assert checker(ticket_pool, ("c:1",)).name(forged) is None
    assert checker(ticket_pool).name("smn-garbage") is None
    assert checker(ticket_pool).name("smn-a.b") is None


def test_a_far_future_ticket_is_refused(ticket_pool):
    clock = Clock()
    clock.now = 1_800_000_000.0
    ahead = Clock()
    ahead.now = clock.now + 3600
    assert checker(ticket_pool, clock=clock).name(maker(ticket_pool, clock=ahead).ticket("b:1")) is None


def test_a_ticket_identifies_a_node_to_a_control_plane(ticket_pool, monkeypatch):
    from shardmesh.tickets import TicketIdentity

    monkeypatch.setenv("SHARDMESH_CA_FILE", str(ticket_pool / "ca.pem"))
    monkeypatch.setenv("SHARDMESH_NODE_URL", "http://b:1")
    runtime = Orchestrator()
    runtime.mesh = ControlMesh("me", "http://b:1", (), Clock())
    runtime.mesh.leader_id = "me"
    runtime.access = AccessControl(identities=True)
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    client = TestClient(TicketIdentity(orchestrator_module.app))
    good = {"Authorization": "Bearer " + maker(ticket_pool).ticket("b:1")}
    assert client.get("/internal/mesh", headers=good).status_code == 200
    assert client.post("/admin/keys", json={"name": "x"}, headers=good).status_code == 403
    assert client.get("/internal/mesh").status_code == 401
    runtime.access.revoke_node("node-a")
    assert client.get("/internal/mesh", headers=good).status_code == 401


def test_a_worker_takes_tickets_and_refuses_the_revoked(ticket_pool, monkeypatch):
    from shardmesh.tickets import TicketIdentity

    monkeypatch.setenv("SHARDMESH_CA_FILE", str(ticket_pool / "ca.pem"))
    monkeypatch.setenv("SHARDMESH_ADVERTISE_URL", "http://b:1")
    runtime = SimpleNamespace(pool_token=None, requires_identity=True, revoked_nodes=set())
    monkeypatch.setattr(worker_module, "runtime", runtime)
    seen = {}

    async def site(scope, receive, send):
        seen["allowed"] = worker_module.authorized(None)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    client = TestClient(TicketIdentity(site))
    client.get("/", headers={"Authorization": "Bearer " + maker(ticket_pool).ticket("b:1")})
    assert seen["allowed"]
    client.get("/")
    assert not seen["allowed"]
    runtime.revoked_nodes.add("node-a")
    client.get("/", headers={"Authorization": "Bearer " + maker(ticket_pool).ticket("b:1")})
    assert not seen["allowed"]


def test_tickets_leave_a_callers_own_credentials_alone(ticket_pool):
    from shardmesh.tickets import NodeTickets

    auth = NodeTickets(maker(ticket_pool))
    plain = httpx.Request("GET", "http://b:1/x")
    assert next(auth.auth_flow(plain)).headers["authorization"].startswith("Bearer smn-")
    theirs = httpx.Request("GET", "http://b:1/x", headers={"Authorization": "Bearer user-key"})
    assert next(auth.auth_flow(theirs)).headers["authorization"] == "Bearer user-key"


def test_certificate_interception_gets_a_plain_explanation():
    orchestrator = Orchestrator()
    clock = Clock()
    orchestrator.mesh = ControlMesh("me", "https://me", ("https://peer",), clock)

    def handler(request):
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await orchestrator._gossip_round()
        await orchestrator._gossip_round()
        await orchestrator.close()

    asyncio.run(scenario())
    warnings = [event for event in orchestrator.events if "antivirus" in event["detail"]]
    assert len(warnings) == 1


def test_a_control_plane_cannot_clear_or_invent_a_workers_revocations(signed_pool):
    from shardmesh.access import Authority
    from shardmesh.pool import sign_entry
    from shardmesh.worker import WorkerRuntime

    folder, _ = signed_pool
    worker = SimpleNamespace(authority=Authority(folder + "/ca.pem"), revoked_nodes={"bad"})
    real = sign_entry(folder, {"type": "revoke", "name": "worse"})
    forged = {"entry": {"type": "revoke", "name": "good"}, "signature": real["signature"]}
    WorkerRuntime._take_revocations(worker, {"revoked_nodes": ["good"], "signed_revocations": [real, forged]})
    assert worker.revoked_nodes == {"bad", "worse"}
    WorkerRuntime._take_revocations(worker, {"revoked_nodes": [], "signed_revocations": []})
    assert worker.revoked_nodes == {"bad", "worse"}


def test_without_an_authority_a_workers_revocations_only_add_up():
    from shardmesh.worker import WorkerRuntime

    worker = SimpleNamespace(authority=None, revoked_nodes={"bad"})
    WorkerRuntime._take_revocations(worker, {"revoked_nodes": []})
    assert worker.revoked_nodes == {"bad"}


def proof_pool(ticket_pool, signed_access):
    orchestrator = Orchestrator()
    orchestrator.access = signed_access
    orchestrator.mesh = ControlMesh("me", "http://me:1", (), Clock())
    return orchestrator


def test_a_gossip_reply_must_prove_who_sent_it(ticket_pool, monkeypatch):
    from shardmesh.access import Authority
    from shardmesh.tickets import reply_audience

    monkeypatch.setenv("SHARDMESH_CA_FILE", str(ticket_pool / "ca.pem"))
    orchestrator = proof_pool(ticket_pool, AccessControl(identities=True, authority=Authority(str(ticket_pool / "ca.pem"))))
    proof = maker(ticket_pool).ticket(reply_audience("http://me:1"))
    assert orchestrator._reply_is_genuine({"node_id": "node-a", "proof": proof})
    assert not orchestrator._reply_is_genuine({"node_id": "ghost", "proof": proof})
    assert not orchestrator._reply_is_genuine({"node_id": "node-a"})
    wrong = maker(ticket_pool).ticket(reply_audience("http://someone-else:1"))
    assert not orchestrator._reply_is_genuine({"node_id": "node-a", "proof": wrong})
    asyncio.run(orchestrator.close())


def test_a_persons_key_never_leaves_the_node_they_gave_it_to(signed_pool, monkeypatch):
    from shardmesh.pool import sign_entry

    folder, access = signed_pool
    access.add_signed(sign_entry(folder, key_entry()))
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, stream=httpx.ByteStream(b'{"ok": true}'), headers={"content-type": "application/json"})

    runtime = Orchestrator()
    runtime.access = access
    runtime.mesh = ControlMesh("me", "http://me", (), Clock())
    runtime.mesh.leader_id = "evil"
    runtime.mesh.observe({"node_id": "evil", "url": "http://evil:9", "term": 1, "leader_id": "evil"})
    runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    client = TestClient(orchestrator_module.app)
    assert client.get("/admin/status").status_code == 401
    assert client.get("/admin/status", headers={"Authorization": "Bearer smk-test"}).status_code == 200
    assert "authorization" not in seen
    assert seen["x-shardmesh-on-behalf"] == "admin:alice"


def test_a_verified_node_can_vouch_for_a_caller_but_a_stranger_cannot(signed_pool, monkeypatch):
    _, access = signed_pool
    runtime = Orchestrator()
    runtime.access = access
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    headers = {FORWARD_MARK: "node-b", "x-shardmesh-on-behalf": "admin:alice"}
    assert TestClient(orchestrator_module.app).get("/admin/keys", headers=headers).status_code == 401
    node = TestClient(connected_as(orchestrator_module.app, "node-b"))
    assert node.get("/admin/keys", headers=headers).status_code == 200


def test_a_signed_entry_with_a_new_signature_changes_nothing(signed_pool):
    from shardmesh.pool import sign_entry

    folder, access = signed_pool
    signed = sign_entry(folder, key_entry())
    access.add_signed(signed)
    version = access.version
    again = dict(signed)
    assert access.add_signed(again)
    assert access.version == version


def test_a_report_about_a_healthy_worker_quarantines_nobody():
    from test_resilience import pool, record

    orchestrator = pool(w1=record("w1"))
    orchestrator.active = object()

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok"})))
        await orchestrator.handle_report("w1", "a peer says so")
        await orchestrator.close()

    asyncio.run(scenario())
    assert not orchestrator.workers["w1"].quarantined


def test_a_tiny_temperature_is_greedy():
    assert ChatCompletionRequest(messages=[], temperature=1e-300).temperature == 0.0
    assert ChatCompletionRequest(messages=[], temperature=0.7).temperature == 0.7


def test_a_ticket_with_no_real_expiry_is_refused(ticket_pool):
    import json

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    from shardmesh.tickets import PREFIX, _encode

    made = maker(ticket_pool)
    body = json.dumps({"aud": "b:1", "exp": float("nan"), "cert": made.certificate.decode()}).encode()
    ticket = f"{PREFIX}{_encode(body)}.{_encode(made.key.sign(body, ec.ECDSA(hashes.SHA256())))}"
    assert checker(ticket_pool).name(ticket) is None


def test_a_stage_cannot_be_built_from_units_that_fail_their_hash(tmp_path):
    from shardmesh.gguf_stage import build_stage_file
    from shardmesh.weights import UnitCache

    cache = UnitCache(tmp_path / "cache")
    path = cache.path("m", "r", "unit")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"altered")
    request = SimpleNamespace(model_id="m", revision="r", unit_hashes={"unit": "0" * 64})
    with pytest.raises(ValueError, match="does not match"):
        build_stage_file(cache, request, None, ["unit"], 0, 4, True)


def test_a_stage_built_with_hashes_does_not_reuse_an_older_file(tmp_path):
    from shardmesh.gguf_stage import build_stage_file
    from shardmesh.weights import UnitCache, file_sha256

    cache = UnitCache(tmp_path / "cache")
    folder = cache.directory("m", "r") / "stages"
    folder.mkdir(parents=True)
    poisoned = folder / "stage-0-4.gguf"
    poisoned.write_bytes(b"poisoned")
    unit = cache.path("m", "r", "unit")
    unit.write_bytes(b"good")
    request = SimpleNamespace(model_id="m", revision="r", unit_hashes={"unit": "0" * 64})
    with pytest.raises(ValueError):
        build_stage_file(cache, request, None, ["unit"], 0, 4, True)
    assert not poisoned.exists()
    assert file_sha256(unit)


@pytest.fixture
def owned(signed_pool, tmp_path, monkeypatch):
    folder, access = signed_pool
    monkeypatch.setenv("SHARDMESH_CA_FILE", folder + "/ca.pem")
    runtime = Orchestrator()
    runtime.access = access
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    return folder, access, TestClient(orchestrator_module.app), tmp_path / "new-node"


def test_a_node_joins_by_asking_and_the_owner_approving(owned):
    from shardmesh.pool import approve_membership, request_membership, sign_entry
    from shardmesh.tickets import TicketChecker, TicketMaker

    folder, access, client, home = owned
    request = request_membership(str(home), "newcomer", ["10.0.0.9"])
    asked = client.post("/join/requests", json={"request": request})
    assert asked.json()["status"] == "waiting"
    assert client.get("/join/certificate/newcomer").status_code == 404
    held = access.join_requests["newcomer"]
    assert held["addresses"] == ["10.0.0.9"] and held["fingerprint"] == asked.json()["fingerprint"]
    certificate = approve_membership(folder, held["request"])
    entry = sign_entry(folder, {"type": "member", "name": "newcomer", "certificate": certificate})
    assert client.post("/access/entries", json=entry).status_code == 200
    assert "newcomer" not in access.join_requests
    issued = client.get("/join/certificate/newcomer").json()
    (home / "newcomer.pem").write_text(issued["certificate"])
    ticket = TicketMaker(str(home / "newcomer.pem"), str(home / "newcomer.key")).ticket("b:1")
    assert TicketChecker(folder + "/ca.pem", {"b:1"}).name(ticket) == "newcomer"


def test_asking_to_join_grants_nothing(owned):
    from shardmesh.pool import request_membership

    _, access, client, home = owned
    client.post("/join/requests", json={"request": request_membership(str(home), "newcomer", ["10.0.0.9"])})
    assert client.get("/internal/mesh").status_code == 401
    assert client.get("/join/requests").status_code == 401
    assert not access.certificates


def test_a_forged_join_request_is_refused(owned):
    _, _, client, _ = owned
    assert client.post("/join/requests", json={"request": "not a request"}).status_code == 400
    assert client.post("/join/requests", json={}).status_code == 400


def test_a_revoked_name_cannot_ask_again(owned):
    from shardmesh.pool import request_membership, sign_entry

    folder, access, client, home = owned
    access.add_signed(sign_entry(folder, {"type": "revoke", "name": "banned"}))
    answer = client.post("/join/requests", json={"request": request_membership(str(home), "banned", ["10.0.0.9"])})
    assert answer.json()["status"] == "refused"
    assert "banned" not in access.join_requests


def test_a_denied_key_stays_denied(owned):
    from shardmesh.pool import request_membership, sign_entry

    folder, access, client, home = owned
    request = request_membership(str(home), "newcomer", ["10.0.0.9"])
    fingerprint = client.post("/join/requests", json={"request": request}).json()["fingerprint"]
    access.add_signed(sign_entry(folder, {"type": "deny", "name": "newcomer", "fingerprint": fingerprint}))
    assert "newcomer" not in access.join_requests
    assert client.post("/join/requests", json={"request": request}).json()["status"] == "refused"


def test_only_the_owner_can_admit_a_node(owned, tmp_path):
    from shardmesh.pool import approve_membership, init_pool, request_membership, sign_entry

    folder, access, client, home = owned
    other = init_pool(str(tmp_path / "other"))
    request = request_membership(str(home), "newcomer", ["10.0.0.9"])
    forged = sign_entry(str(other), {"type": "member", "name": "newcomer", "certificate": approve_membership(str(other), request)})
    assert client.post("/access/entries", json=forged).status_code == 403
    assert client.get("/join/certificate/newcomer").status_code == 404


def test_join_requests_are_bounded_and_spread(owned):
    from shardmesh.access import AccessControl, Authority, MAX_JOIN_REQUESTS
    from shardmesh.pool import read_membership_request, request_membership

    folder, access, _, home = owned
    text = request_membership(str(home), "one", ["10.0.0.9"])
    access.add_join_request(read_membership_request(text), text)
    peer = AccessControl(identities=True, authority=Authority(folder + "/ca.pem"))
    peer.merge(access.document())
    assert "one" in peer.join_requests
    peer.merge({"join_requests": [{"request": "forged"}]})
    assert list(peer.join_requests) == ["one"]
    for index in range(MAX_JOIN_REQUESTS):
        access.join_requests[f"n{index}"] = {}
    second = request_membership(str(home), "late", ["10.0.0.9"])
    assert access.add_join_request(read_membership_request(second), second) == "busy"
