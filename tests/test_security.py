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

from isharded import orchestrator as orchestrator_module
from isharded import worker as worker_module
from isharded.access import ROLE_ADMIN, ROLE_NODE, ROLE_USER, AccessControl, node_of
from isharded.llamacpp import TokenEmbedding
from isharded.mesh import RUMOR_SECONDS, ControlMesh, clean_url
from isharded.orchestrator import FORWARD_MARK, Orchestrator, verify_worker_name
from isharded.pool import ExposedPool, check_exposure
from isharded.protocol import ChatCompletionRequest, CompletionRequest, DraftModel, TensorRef
from isharded.tls import PEER
from isharded.tokenizer import template_json
from isharded.weights import UnitCache


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
    assert node_of("laptop-memory") == "laptop"
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
    from isharded.orchestrator import SessionState

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
    monkeypatch.setenv("ISHARDED_MODEL_DIRS", str(private))
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
    monkeypatch.delenv("ISHARDED_TLS_CERT", raising=False)
    with pytest.raises(ExposedPool, match="unencrypted"):
        check_exposure("0.0.0.0", "token", None, False, "http://192.168.1.20:8000")
    check_exposure("0.0.0.0", "token", None, False, "http://192.168.1.20:8000", trusted_network=True)
    check_exposure("0.0.0.0", "token", None, False, "http://100.64.0.2:8300")
    check_exposure("0.0.0.0", "token", None, False, "desktop.tail1234.ts.net")
    check_exposure("0.0.0.0", "token", None, False, "http://127.0.0.1:8100")
    monkeypatch.setenv("ISHARDED_TLS_CERT", "node.pem")
    check_exposure("0.0.0.0", "token", None, False, "https://192.168.1.20:8000")


def test_a_chat_template_cannot_pass_code_to_json():
    assert template_json({"b": 1, "a": 2}, indent=1, sort_keys=True).startswith("{\n")
    with pytest.raises(TypeError):
        template_json({}, default=str)


def test_a_chat_template_cannot_reach_the_process_through_tojson():
    from isharded.tokenizer import ChatTokenizer as Holder

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
    from isharded.access import Authority
    from isharded.pool import init_pool

    pytest.importorskip("cryptography")
    folder = init_pool(str(tmp_path / "pool"))
    return str(folder), AccessControl(identities=True, authority=Authority(str(folder / "ca.pem")))


def key_entry(name="alice", role="admin", key="isk-test"):
    from isharded.access import digest

    return {"type": "key", "name": name, "role": role, "sha256": digest(key)}


def test_a_signed_key_spreads_and_works(signed_pool):
    from isharded.access import Authority
    from isharded.pool import sign_entry

    folder, access = signed_pool
    assert access.add_signed(sign_entry(folder, key_entry()))
    assert access.permits(access.identify("Bearer isk-test"), ROLE_ADMIN)
    peer = AccessControl(identities=True, authority=Authority(folder + "/ca.pem"))
    assert peer.merge(access.document())
    assert peer.permits(peer.identify("Bearer isk-test"), ROLE_ADMIN)


def test_a_compromised_node_cannot_add_a_key_through_gossip(signed_pool):
    from isharded.access import digest

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
    from isharded.pool import init_pool, sign_entry

    _, access = signed_pool
    other = init_pool(str(tmp_path / "other"))
    assert not access.add_signed(sign_entry(str(other), key_entry()))
    assert access.identify("Bearer isk-test") is None


def test_changing_a_signed_entry_breaks_it(signed_pool):
    from isharded.pool import sign_entry

    folder, access = signed_pool
    signed = sign_entry(folder, key_entry(role="user"))
    signed["entry"]["role"] = "admin"
    assert not access.add_signed(signed)


def test_a_signed_revocation_and_removal_stick(signed_pool):
    from isharded.pool import sign_entry

    folder, access = signed_pool
    held = access.document()
    access.add_signed(sign_entry(folder, key_entry()))
    access.add_signed(sign_entry(folder, {"type": "remove_key", "name": "alice"}))
    access.add_signed(sign_entry(folder, {"type": "revoke", "name": "bad"}))
    access.merge({**held, "version": 500})
    assert "bad" in access.revoked
    assert access.identify("Bearer isk-test") is None
    late = sign_entry(folder, key_entry(key="again"))
    access.add_signed(late)
    assert access.identify("Bearer again") is None, "a removed name stays removed"


def test_signed_entries_survive_a_restart(signed_pool, tmp_path):
    from isharded.access import Authority
    from isharded.pool import sign_entry

    folder, _ = signed_pool
    path = tmp_path / "access.json"
    authority = Authority(folder + "/ca.pem")
    access = AccessControl.load(None, str(path), identities=True, authority=authority)
    access.add_signed(sign_entry(folder, key_entry()))
    again = AccessControl.load(None, str(path), identities=True, authority=authority)
    assert again.permits(again.identify("Bearer isk-test"), ROLE_ADMIN)


def test_with_an_authority_nodes_cannot_change_access_themselves(signed_pool):
    from isharded.access import NeedsAuthority

    _, access = signed_pool
    for change in (lambda: access.add_key("x", ROLE_USER), lambda: access.remove_key("x"), lambda: access.revoke_node("x")):
        with pytest.raises(NeedsAuthority):
            change()


def test_the_api_takes_signed_entries_and_refuses_the_rest(signed_pool, monkeypatch):
    from isharded.pool import sign_entry

    folder, access = signed_pool
    runtime = Orchestrator()
    runtime.access = access
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    client = TestClient(orchestrator_module.app)
    assert client.post("/access/entries", json={"entry": key_entry(), "signature": "AAAA"}).status_code == 403
    assert client.post("/access/entries", json=sign_entry(folder, key_entry())).status_code == 200
    admin = {"Authorization": "Bearer isk-test"}
    assert client.post("/admin/keys", json={"name": "x"}, headers=admin).status_code == 409
    assert client.post("/admin/nodes/x/revoke", headers=admin).status_code == 409


@pytest.fixture
def ticket_pool(tmp_path):
    from isharded.pool import add_node, init_pool

    pytest.importorskip("cryptography")
    folder = init_pool(str(tmp_path / "pool"))
    add_node(str(folder), "node-a", ["10.0.0.1"])
    return folder


def maker(folder, name="node-a", clock=None):
    from isharded.tickets import TicketMaker

    options = {"clock": clock} if clock else {}
    return TicketMaker(str(folder / f"{name}.pem"), str(folder / f"{name}.key"), **options)


def checker(folder, audiences=("b:1",), clock=None):
    from isharded.tickets import TicketChecker

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
    from isharded.pool import add_node, init_pool

    other = init_pool(str(tmp_path / "other"))
    add_node(str(other), "node-a", ["10.0.0.1"])
    assert checker(ticket_pool).name(maker(other).ticket("b:1")) is None


def test_an_altered_ticket_is_refused(ticket_pool):
    import json

    from isharded.tickets import PREFIX, _decode, _encode

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
    from isharded.tickets import TicketIdentity

    monkeypatch.setenv("ISHARDED_CA_FILE", str(ticket_pool / "ca.pem"))
    monkeypatch.setenv("ISHARDED_NODE_URL", "http://b:1")
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
    from isharded.tickets import TicketIdentity

    monkeypatch.setenv("ISHARDED_CA_FILE", str(ticket_pool / "ca.pem"))
    monkeypatch.setenv("ISHARDED_ADVERTISE_URL", "http://b:1")
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
    from isharded.tickets import NodeTickets

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
    from isharded.access import Authority
    from isharded.pool import sign_entry
    from isharded.worker import WorkerRuntime

    folder, _ = signed_pool
    worker = SimpleNamespace(authority=Authority(folder + "/ca.pem"), revoked_nodes={"bad"})
    real = sign_entry(folder, {"type": "revoke", "name": "worse"})
    forged = {"entry": {"type": "revoke", "name": "good"}, "signature": real["signature"]}
    WorkerRuntime._take_revocations(worker, {"revoked_nodes": ["good"], "signed_revocations": [real, forged]})
    assert worker.revoked_nodes == {"bad", "worse"}
    WorkerRuntime._take_revocations(worker, {"revoked_nodes": [], "signed_revocations": []})
    assert worker.revoked_nodes == {"bad", "worse"}


def test_without_an_authority_a_workers_revocations_only_add_up():
    from isharded.worker import WorkerRuntime

    worker = SimpleNamespace(authority=None, revoked_nodes={"bad"})
    WorkerRuntime._take_revocations(worker, {"revoked_nodes": []})
    assert worker.revoked_nodes == {"bad"}


def proof_pool(ticket_pool, signed_access):
    orchestrator = Orchestrator()
    orchestrator.access = signed_access
    orchestrator.mesh = ControlMesh("me", "http://me:1", (), Clock())
    return orchestrator


def test_a_gossip_reply_must_prove_who_sent_it(ticket_pool, monkeypatch):
    from isharded.access import Authority
    from isharded.tickets import reply_audience

    monkeypatch.setenv("ISHARDED_CA_FILE", str(ticket_pool / "ca.pem"))
    orchestrator = proof_pool(ticket_pool, AccessControl(identities=True, authority=Authority(str(ticket_pool / "ca.pem"))))
    proof = maker(ticket_pool).ticket(reply_audience("http://me:1"))
    assert orchestrator._reply_is_genuine({"node_id": "node-a", "proof": proof})
    assert not orchestrator._reply_is_genuine({"node_id": "ghost", "proof": proof})
    assert not orchestrator._reply_is_genuine({"node_id": "node-a"})
    wrong = maker(ticket_pool).ticket(reply_audience("http://someone-else:1"))
    assert not orchestrator._reply_is_genuine({"node_id": "node-a", "proof": wrong})
    asyncio.run(orchestrator.close())


def test_a_persons_key_never_leaves_the_node_they_gave_it_to(signed_pool, monkeypatch):
    from isharded.pool import sign_entry

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
    assert client.get("/admin/status", headers={"Authorization": "Bearer isk-test"}).status_code == 200
    assert "authorization" not in seen
    assert "isk-test" not in str(seen)


def proven_key_entry(name="alice", role="admin", key="isk-test"):
    from isharded.statements import key_verifier

    return {**key_entry(name, role, key), "verifier": key_verifier(key)}


@pytest.fixture
def coordinator(signed_pool, monkeypatch):
    """A node that coordinates. Alice's key has a verifier, and an older one does not."""
    from isharded.pool import sign_entry

    folder, access = signed_pool
    access.add_signed(sign_entry(folder, proven_key_entry()))
    access.add_signed(sign_entry(folder, proven_key_entry("bob", "user", "isk-bob")))
    access.add_signed(sign_entry(folder, key_entry("old", "admin", "isk-old")))
    runtime = Orchestrator()
    runtime.access = access
    runtime.mesh = ControlMesh("boss", "http://boss:9", (), Clock())
    runtime.mesh.leader_id = "boss"
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    return runtime


@pytest.fixture
def follow(coordinator, monkeypatch):
    """What a real follower sends the coordinator for a person's request."""

    def send(method, path, key="isk-test", body=b"", follower="follower"):
        seen = {}

        def handler(request):
            seen.update(request.headers)
            seen["sent"] = request.content
            return httpx.Response(200, stream=httpx.ByteStream(b"{}"), headers={"content-type": "application/json"})

        node = Orchestrator()
        node.access = coordinator.access
        node.mesh = ControlMesh(follower, f"http://{follower}:1", (), Clock())
        node.mesh.observe({"node_id": "boss", "url": "http://boss:9", "term": 1, "leader_id": "boss"})
        node.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(orchestrator_module, "runtime", node)
        headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        assert TestClient(orchestrator_module.app).request(method, path, headers=headers, content=body).status_code == 200
        monkeypatch.setattr(orchestrator_module, "runtime", coordinator)
        sent = seen.pop("sent")
        return {name: value for name, value in seen.items() if name not in {"host", "content-length"}}, sent

    return send


def from_node(name):
    return TestClient(connected_as(orchestrator_module.app, name))


def plain_statement(**changed):
    """A statement any node could write, signed with whatever it holds."""
    from isharded.statements import make_statement

    fields = {"key": "isk-test", "sender": "follower", "target": "boss", "method": "GET", "path": "/admin/keys"}
    return make_statement(**{**fields, **changed})


def forwarded(statement, by="follower"):
    return {FORWARD_MARK: by, "x-isharded-on-behalf": statement}


def test_a_real_forward_works_with_the_key_left_behind(follow):
    headers, _ = follow("GET", "/admin/keys")
    assert "isk-test" not in str(headers)
    assert from_node("follower").get("/admin/keys", headers=headers).status_code == 200


@pytest.mark.parametrize(("method", "tries", "status"), [("GET", 2, 200), ("POST", 1, 503)])
def test_a_dead_keep_alive_is_retried_for_a_read_only(coordinator, monkeypatch, method, tries, status):
    statements = []

    def handler(request):
        statements.append(request.headers.get("x-isharded-on-behalf"))
        if len(statements) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"), headers={"content-type": "application/json"})

    node = Orchestrator()
    node.access = coordinator.access
    node.mesh = ControlMesh("follower", "http://follower:1", (), Clock())
    node.mesh.observe({"node_id": "boss", "url": "http://boss:9", "term": 1, "leader_id": "boss"})
    node.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(orchestrator_module, "runtime", node)
    answer = TestClient(orchestrator_module.app).request(method, "/admin/keys", headers={"Authorization": "Bearer isk-test"})
    assert answer.status_code == status
    assert len(statements) == tries
    assert len(set(statements)) == tries  # a statement is good once


def test_a_real_forward_carries_its_body_and_query(follow):
    body = b'{"name": "x", "role": "user"}'
    headers, sent = follow("POST", "/admin/keys?note=a%20b&first=1", body=body)
    assert sent == body
    assert from_node("follower").post("/admin/keys?first=1&note=a+b", headers=headers, content=sent).status_code == 409  # past the key check


def test_a_statement_works_once(follow):
    headers, _ = follow("GET", "/admin/keys")
    node = from_node("follower")
    assert node.get("/admin/keys", headers=headers).status_code == 200
    replayed = node.get("/admin/keys", headers=headers)
    assert replayed.status_code == 401 and "already been used" in replayed.text


def test_a_statement_for_another_node_is_refused(follow):
    headers, _ = follow("GET", "/admin/keys")
    orchestrator_module.runtime.mesh = ControlMesh("other", "http://other:9", (), Clock())
    orchestrator_module.runtime.mesh.leader_id = "other"
    refused = from_node("follower").get("/admin/keys", headers=headers)
    assert refused.status_code == 401 and "this statement is for boss" in refused.text


def test_a_statement_is_no_use_to_a_node_that_saw_it_pass(follow):
    headers, _ = follow("GET", "/admin/keys")
    assert from_node("thief").get("/admin/keys", headers=headers).status_code == 401


@pytest.mark.parametrize(
    "changed",
    [
        {"method": "POST"},
        {"path": "/admin/status"},
        {"path": "/admin/keys?force=true"},
        {"body": b"something else"},
    ],
)
def test_a_statement_covers_exactly_one_request(coordinator, changed):
    statement = plain_statement(**changed)
    assert from_node("follower").get("/admin/keys", headers=forwarded(statement)).status_code == 401


def test_a_stale_statement_is_refused_and_a_fresh_one_from_a_slow_clock_is_not(coordinator):
    import time

    from isharded.statements import STATEMENT_SECONDS
    from isharded.tickets import CLOCK_SKEW_SECONDS

    node = from_node("follower")
    good = plain_statement(clock=lambda: time.time() - STATEMENT_SECONDS)
    assert node.get("/admin/keys", headers=forwarded(good)).status_code == 200
    stale = plain_statement(clock=lambda: time.time() - STATEMENT_SECONDS - CLOCK_SKEW_SECONDS - 5)
    assert node.get("/admin/keys", headers=forwarded(stale)).status_code == 401
    ahead = plain_statement(clock=lambda: time.time() + CLOCK_SKEW_SECONDS + 5)
    assert node.get("/admin/keys", headers=forwarded(ahead)).status_code == 401
    nonsense = plain_statement(clock=lambda: float("nan"))
    assert node.get("/admin/keys", headers=forwarded(nonsense)).status_code == 401


def test_a_node_cannot_sign_for_a_key_it_does_not_hold(coordinator):
    """Everything a member can see of the key: its hash and its verifier."""
    import json

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from isharded.access import digest
    from isharded.statements import body_hash
    from isharded.tickets import _encode

    claims = {"key": digest("isk-test"), "from": "follower", "to": "boss", "method": "GET", "path": "/admin/keys", "body": body_hash(b""), "at": __import__("time").time(), "nonce": "n"}
    raw = json.dumps(claims).encode()
    forged = f"{_encode(raw)}.{_encode(Ed25519PrivateKey.generate().sign(raw))}"
    assert from_node("follower").get("/admin/keys", headers=forwarded(forged)).status_code == 401


def test_a_statement_that_cannot_be_read_proves_nothing(coordinator):
    for junk in ("", ".", "a.b", "e30.AA", "W10.AA", "admin:alice"):
        assert from_node("follower").get("/admin/keys", headers=forwarded(junk)).status_code == 401


def test_the_role_comes_from_the_signed_entry(follow):
    headers, _ = follow("GET", "/admin/status", key="isk-bob")
    node = from_node("follower")
    assert node.get("/admin/status", headers=headers).status_code == 200
    headers, _ = follow("GET", "/admin/status", key="isk-bob")
    attempt = dict(headers, **{"x-isharded-role": "admin"})
    assert node.get("/admin/keys", headers=attempt).status_code == 401
    assert node.get("/admin/keys", headers=forwarded(plain_statement(key="isk-bob"))).status_code == 403


def test_a_removed_key_cannot_be_passed_on(coordinator, signed_pool):
    from isharded.pool import sign_entry

    coordinator.access.add_signed(sign_entry(signed_pool[0], {"type": "remove_key", "name": "alice"}))
    assert from_node("follower").get("/admin/keys", headers=forwarded(plain_statement())).status_code == 401


def test_a_revoked_node_cannot_pass_anything_on(coordinator, signed_pool):
    from isharded.pool import sign_entry

    coordinator.access.add_signed(sign_entry(signed_pool[0], {"type": "revoke", "name": "follower"}))
    assert from_node("follower").get("/admin/keys", headers=forwarded(plain_statement())).status_code == 401


def test_a_stranger_gets_nothing_from_a_good_statement(coordinator):
    client = TestClient(orchestrator_module.app)
    assert client.get("/admin/keys", headers=forwarded(plain_statement())).status_code == 401


def test_a_key_made_before_verifiers_is_a_user_through_a_forward(follow):
    node = from_node("follower")
    headers, _ = follow("GET", "/admin/status", key="isk-old")
    assert node.get("/admin/status", headers=headers).status_code == 200
    headers, _ = follow("GET", "/admin/keys", key="isk-old")
    assert node.get("/admin/keys", headers=headers).status_code == 403
    direct = TestClient(orchestrator_module.app).get("/admin/keys", headers={"Authorization": "Bearer isk-old"})
    assert direct.status_code == 200


def test_the_audit_log_records_what_was_proven(follow, coordinator, tmp_path):
    import json

    from isharded.access import AuditLog

    coordinator.audit = AuditLog(str(tmp_path / "audit.jsonl"), "boss")
    headers, sent = follow("POST", "/admin/keys", body=b"{}")
    from_node("follower").post("/admin/keys", headers=headers, content=sent)
    from_node("member").post("/admin/keys", headers=forwarded("admin:root", "member"), content=b"{}")
    actors = [json.loads(line)["actor"] for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert actors == ["admin:alice via follower", "unknown"]


def test_one_key_cannot_use_up_what_others_are_remembered_by(coordinator, monkeypatch):
    from isharded import statements

    monkeypatch.setattr(statements, "MAX_NONCES", 2)
    node = from_node("follower")
    for _ in range(2):
        assert node.get("/admin/status", headers=forwarded(plain_statement(key="isk-bob", path="/admin/status"))).status_code == 200
    assert node.get("/admin/status", headers=forwarded(plain_statement(key="isk-bob", path="/admin/status"))).status_code == 401
    assert node.get("/admin/keys", headers=forwarded(plain_statement())).status_code == 200


def test_a_key_keeps_its_verifier_whatever_order_entries_arrive_in(signed_pool):
    from isharded.access import digest
    from isharded.pool import sign_entry

    folder, access = signed_pool
    access.add_signed(sign_entry(folder, proven_key_entry()))
    access.add_signed(sign_entry(folder, key_entry()))
    assert access.keys[digest("isk-test")].verifier


def test_every_way_of_making_a_key_gives_it_a_verifier(tmp_path):
    from isharded.access import create_key, digest
    from isharded.statements import key_verifier

    path = str(tmp_path / "keys.json")
    key = create_key(path, "alice", ROLE_ADMIN)
    loaded = AccessControl.load(None, path)
    assert loaded.keys[digest(key)].verifier == key_verifier(key)
    other = loaded.add_key("bob", ROLE_USER)
    again = AccessControl.load(None, path)
    assert again.keys[digest(other)].verifier == key_verifier(other)


def test_a_verifier_says_nothing_about_the_key_it_came_from():
    from isharded.statements import VERIFIER_LENGTH, key_verifier

    assert len(key_verifier("isk-one")) == VERIFIER_LENGTH
    assert key_verifier("isk-one") == key_verifier("isk-one") != key_verifier("isk-two")
    assert "isk" not in key_verifier("isk-one")


@pytest.fixture
def lane(coordinator):
    """The same runtime, as the lane called small on a node called lane-node."""
    coordinator.lane_name = "small"
    coordinator.mesh = ControlMesh("lane-node", "http://lane-node:1009", (), Clock())
    coordinator.mesh.leader_id = "lane-node"
    return coordinator


def test_a_statement_for_a_node_is_no_use_on_its_lane_and_the_other_way(coordinator):
    for_lane = plain_statement(target="boss/small")
    assert from_node("follower").get("/admin/keys", headers=forwarded(for_lane)).status_code == 401
    coordinator.lane_name = "small"
    coordinator.mesh.node_id = coordinator.mesh.leader_id = "boss"
    for_node = plain_statement(target="boss")
    assert from_node("follower").get("/admin/keys", headers=forwarded(for_node)).status_code == 401
    assert from_node("follower").get("/admin/keys", headers=forwarded(for_lane)).status_code == 200


def test_a_lane_takes_a_passed_on_statement_only_from_the_node_it_was_for(lane):
    statement = plain_statement(path="/admin/keys", target="boss", sender="follower")
    assert from_node("member").get("/admin/keys", headers=forwarded(statement, "member")).status_code == 401
    assert from_node("follower").get("/admin/keys", headers=forwarded(statement)).status_code == 401
    assert from_node("boss").get("/admin/keys", headers=forwarded(statement, "boss")).status_code == 200
    assert from_node("boss").get("/admin/keys", headers=forwarded(statement, "boss")).status_code == 401


def test_only_a_lane_takes_a_passed_on_statement(coordinator):
    statement = plain_statement(target="other-coordinator")
    assert from_node("other-coordinator").get("/admin/keys", headers=forwarded(statement, "other-coordinator")).status_code == 401


def test_a_passed_on_statement_must_be_for_the_same_request(lane):
    statement = plain_statement(path="/admin/status", target="boss")
    assert from_node("boss").get("/admin/keys", headers=forwarded(statement, "boss")).status_code == 401


def test_a_passed_on_statement_outlasts_a_lanes_workers_starting(lane):
    import time

    from isharded.lanes import LANE_WORKERS_WAIT_SECONDS
    from isharded.statements import PASSED_ON_SECONDS

    assert PASSED_ON_SECONDS > LANE_WORKERS_WAIT_SECONDS
    late = plain_statement(target="boss", clock=lambda: time.time() - LANE_WORKERS_WAIT_SECONDS - 20)
    assert from_node("boss").get("/admin/keys", headers=forwarded(late, "boss")).status_code == 200
    too_late = plain_statement(target="boss", clock=lambda: time.time() - 1000)
    assert from_node("boss").get("/admin/keys", headers=forwarded(too_late, "boss")).status_code == 401


def lane_door(coordinator, handler):
    """A front door for a lane whose control planes answer through ``handler``."""
    coordinator.lanes.path = Path("unused")
    coordinator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    coordinator.mesh.lanes = {"small": {"slot": 1, "budgets": {}}}
    coordinator.lanes.seen["small"] = {"ready": False, "leader": "lane-boss", "leader_url": "http://lane-boss:1009"}
    return coordinator


def test_a_front_door_holding_the_key_signs_for_the_lane_it_is_sending_to(coordinator, signed_pool):
    from isharded.access import Caller
    from isharded.statements import Statements
    from isharded.tls import KEY_HELD

    seen = {}

    def handler(request):
        seen.update(request.headers)
        seen["url"], seen["content"] = str(request.url), request.content
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    door = lane_door(coordinator, handler)
    held = KEY_HELD.set("isk-test")
    try:
        answer = asyncio.run(door.lanes.ask("small", {"messages": []}, Caller("alice", ROLE_ADMIN)))
    finally:
        KEY_HELD.reset(held)
    assert answer == "hello"
    assert "isk-test" not in str(seen)
    proven = Statements().check(door.access, seen["x-isharded-on-behalf"], "boss", "lane-boss/small", "POST", "/v1/chat/completions", seen["content"], in_lane=True)
    assert proven == ("alice", ROLE_ADMIN)


def test_a_front_door_with_no_key_passes_on_what_it_was_given(coordinator):
    from isharded.access import Caller
    from isharded.tls import STATEMENT

    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200)

    door = lane_door(coordinator, handler)
    held = STATEMENT.set("the statement")
    try:
        asyncio.run(door.lanes.tell_all("/admin/nodes/x/cordon?force=true", Caller("alice", ROLE_ADMIN)))
    finally:
        STATEMENT.reset(held)
    assert seen["x-isharded-on-behalf"] == "the statement"


def test_the_key_is_still_there_for_a_load_that_starts_later(coordinator):
    """A lane's load waits for workers, long after the request that asked for it returned."""
    from isharded.access import Caller
    from isharded.lanes import LANE_SETTLED_LOOKS
    from isharded.statements import Statements
    from isharded.tls import KEY_HELD

    sent = []

    def handler(request):
        if request.url.path == "/internal/lane":
            return httpx.Response(200, json={"leader": "lane-boss", "leader_url": "http://lane-boss:1009", "workers": 1})
        sent.append(request)
        return httpx.Response(202)

    door = lane_door(coordinator, handler)
    door.lanes.seen.clear()

    async def scenario():
        KEY_HELD.set("isk-test")
        door.mesh.lanes = {"small": {"slot": 1, "budgets": {"n": {"gpu_gb": 1, "ram_gb": 0}}}}
        task = door.spawn(door.lanes._start_load("small", {"model_id": "m"}, Caller("alice", ROLE_ADMIN)))
        KEY_HELD.set(None)
        await task

    real_sleep = asyncio.sleep
    asyncio.sleep = lambda _: real_sleep(0)
    try:
        asyncio.run(scenario())
    finally:
        asyncio.sleep = real_sleep
    assert LANE_SETTLED_LOOKS and sent[0].url.path == "/admin/load"
    assert Statements().check(door.access, sent[0].headers["x-isharded-on-behalf"], "boss", "lane-boss/small", "POST", "/admin/load", sent[0].content, in_lane=True)[1] == ROLE_ADMIN


@pytest.fixture
def door_with_a_lane(coordinator):
    """A coordinating front door whose lane serves a model, recording what reaches the lane."""
    seen = {}

    def handler(request):
        seen.update(request.headers)
        seen["content"] = request.content
        return httpx.Response(200, stream=httpx.ByteStream(b'{"ok": true}'), headers={"content-type": "application/json"})

    door = lane_door(coordinator, handler)
    door.lanes.seen["small"] = {"ready": True, "model": "lane-model", "leader": "boss", "leader_url": "http://boss:1009"}
    return door, seen


CHAT = {"model": "lane-model", "messages": [{"role": "user", "content": "hello"}]}


def test_a_key_given_to_a_front_door_signs_for_its_lane_through_the_real_route(door_with_a_lane):
    from isharded.statements import Statements

    door, seen = door_with_a_lane
    answer = TestClient(orchestrator_module.app).post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer isk-test"})
    assert answer.status_code == 200 and answer.json() == {"ok": True}
    assert "isk-test" not in str(seen) and "authorization" not in seen
    proven = Statements().check(door.access, seen["x-isharded-on-behalf"], "boss", "boss/small", "POST", "/v1/chat/completions", seen["content"], in_lane=True)
    assert proven == ("alice", ROLE_ADMIN)


def test_a_forward_reaches_a_lane_through_the_coordinator_it_was_sent_to(door_with_a_lane, follow):
    """Follower to coordinator to lane, with the person's key still on the follower."""
    import json

    from isharded.statements import Statements

    door, seen = door_with_a_lane
    headers, sent = follow("POST", "/v1/chat/completions", body=json.dumps(CHAT).encode())
    answer = from_node("follower").post("/v1/chat/completions", headers=headers, content=sent)
    assert answer.status_code == 200
    assert seen["x-isharded-on-behalf"] == headers["x-isharded-on-behalf"]
    door.lane_name = "small"
    proven = Statements().check(door.access, seen["x-isharded-on-behalf"], "boss", "other-node/small", "POST", "/v1/chat/completions", seen["content"], in_lane=True)
    assert proven == ("alice", ROLE_ADMIN)  # a lane on any node, from the node it was addressed to


def test_a_door_tells_a_lane_that_started_after_the_last_change(coordinator):
    pushed = []

    def handler(request):
        pushed.append(request.url.path)
        return httpx.Response(200, json={"status": "merged"})

    door = lane_door(coordinator, handler)
    door.mesh.url = "http://boss:9"

    async def scenario():
        await door.lanes._tell_lanes(["small"])
        assert len(pushed) == 1
        door.lanes.seen["small"]["entries_held"] = door.access.entries_held
        await door.lanes._tell_lanes(["small"])
        assert len(pushed) == 1  # a lane that has it all is left alone
        door.lanes.seen["small"]["entries_held"] = 0  # it restarted, or has only just started
        await door.lanes._tell_lanes(["small"])
        assert pushed == ["/internal/lane/access"] * 2

    asyncio.run(scenario())


def test_a_lane_says_how_to_name_its_coordinator_and_what_it_holds(lane):
    answer = from_node("lane-node").get("/internal/lane").json()
    assert answer["leader"] == "lane-node"
    assert answer["entries_held"] == lane.access.entries_held


def test_a_lane_takes_file_keys_from_its_own_node_and_nobody_else(lane):
    from isharded.access import digest
    from isharded.statements import key_verifier

    document = {"keys": [{"name": "root", "role": "admin", "sha256": digest("isk-root"), "verifier": key_verifier("isk-root")}]}
    assert from_node("member").post("/internal/lane/access", json=document).status_code == 200
    assert digest("isk-root") not in lane.access.keys
    assert from_node("lane-node").post("/internal/lane/access", json=document).status_code == 200
    assert lane.access.keys[digest("isk-root")].name == "root"
    assert lane.access.keys[digest("isk-root")].verifier == key_verifier("isk-root")


def test_nobody_is_given_a_role_a_lane_never_had(lane):
    from isharded.access import digest

    document = {"keys": [{"name": "root", "role": "node", "sha256": digest("isk-root")}]}
    from_node("lane-node").post("/internal/lane/access", json=document)
    assert digest("isk-root") not in lane.access.keys


def test_where_nobody_owns_the_network_a_forwarded_caller_is_reduced(tmp_path, monkeypatch):
    from isharded.commons import Commons, make_identity

    make_identity(tmp_path)
    runtime = Orchestrator()
    runtime.access = AccessControl(identities=True, authority=Commons(tmp_path))
    runtime.mesh = ControlMesh("n-boss", "http://boss:9", (), Clock())
    runtime.mesh.leader_id = "n-boss"
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    seen = []
    real = orchestrator_module.require

    def watching(authorization, role):
        seen.append(real(authorization, ROLE_USER))
        return real(authorization, role)

    monkeypatch.setattr(orchestrator_module, "require", watching)
    node = from_node("n-peer")
    assert node.get("/admin/keys", headers=forwarded("admin:owner", "n-peer")).status_code == 403
    assert (seen[-1].name, seen[-1].role) == ("n-peer/owner", "member")
    assert node.get("/admin/status", headers=forwarded("user:someone", "n-peer")).status_code == 200
    assert (seen[-1].name, seen[-1].role) == ("n-peer/someone", ROLE_USER)
    assert node.get("/admin/keys", headers=forwarded("nonsense", "n-peer")).status_code == 401


def test_where_nobody_owns_the_network_a_forward_still_says_who_asked(tmp_path, monkeypatch):
    from isharded.access import Caller
    from isharded.commons import Commons, make_identity

    make_identity(tmp_path)
    runtime = Orchestrator()
    runtime.access = AccessControl(identities=True, authority=Commons(tmp_path))
    assert runtime.speak_for(Caller("owner", ROLE_ADMIN), "n-boss", "GET", "/admin/status") == "admin:owner"


@pytest.mark.parametrize("claim", ["admin:x", "admin:alice", "member:x", "user:x", "node:x"])
def test_a_member_cannot_claim_a_role_with_a_header(coordinator, claim):
    """A node certificate proves which node is talking, and says nothing about who asked."""
    member = TestClient(connected_as(orchestrator_module.app, "member"))
    forged = {FORWARD_MARK: "member", "x-isharded-on-behalf": claim}
    assert member.get("/admin/keys", headers=forged).status_code == 401
    assert member.post("/admin/keys", json={"name": "x", "role": "admin"}, headers=forged).status_code == 401
    assert member.post("/admin/nodes/victim/revoke", headers=forged).status_code == 401
    assert member.post("/admin/load", json={"model_id": "evil"}, headers=forged).status_code == 401
    assert member.get("/admin/status", headers=forged).status_code == 401
    assert "victim" not in coordinator.access.revoked


def test_a_signed_entry_with_a_new_signature_changes_nothing(signed_pool):
    from isharded.pool import sign_entry

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

    from isharded.tickets import PREFIX, _encode

    made = maker(ticket_pool)
    body = json.dumps({"aud": "b:1", "exp": float("nan"), "cert": made.certificate.decode()}).encode()
    ticket = f"{PREFIX}{_encode(body)}.{_encode(made.key.sign(body, ec.ECDSA(hashes.SHA256())))}"
    assert checker(ticket_pool).name(ticket) is None


def test_a_stage_cannot_be_built_from_units_that_fail_their_hash(tmp_path):
    from isharded.gguf_stage import build_stage_file
    from isharded.weights import UnitCache

    cache = UnitCache(tmp_path / "cache")
    path = cache.path("m", "r", "unit")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"altered")
    request = SimpleNamespace(model_id="m", revision="r", unit_hashes={"unit": "0" * 64})
    with pytest.raises(ValueError, match="does not match"):
        build_stage_file(cache, request, None, ["unit"], 0, 4, True)


def test_a_stage_built_with_hashes_does_not_reuse_an_older_file(tmp_path):
    from isharded.gguf_stage import build_stage_file
    from isharded.weights import UnitCache, file_sha256

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
    monkeypatch.setenv("ISHARDED_CA_FILE", folder + "/ca.pem")
    runtime = Orchestrator()
    runtime.access = access
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    return folder, access, TestClient(orchestrator_module.app), tmp_path / "new-node"


def test_a_node_joins_by_asking_and_the_owner_approving(owned):
    from isharded.pool import approve_membership, request_membership, sign_entry
    from isharded.tickets import TicketChecker, TicketMaker

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
    from isharded.pool import request_membership

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
    from isharded.pool import request_membership, sign_entry

    folder, access, client, home = owned
    access.add_signed(sign_entry(folder, {"type": "revoke", "name": "banned"}))
    answer = client.post("/join/requests", json={"request": request_membership(str(home), "banned", ["10.0.0.9"])})
    assert answer.json()["status"] == "refused"
    assert "banned" not in access.join_requests


def test_a_denied_key_stays_denied(owned):
    from isharded.pool import request_membership, sign_entry

    folder, access, client, home = owned
    request = request_membership(str(home), "newcomer", ["10.0.0.9"])
    fingerprint = client.post("/join/requests", json={"request": request}).json()["fingerprint"]
    access.add_signed(sign_entry(folder, {"type": "deny", "name": "newcomer", "fingerprint": fingerprint}))
    assert "newcomer" not in access.join_requests
    assert client.post("/join/requests", json={"request": request}).json()["status"] == "refused"


def test_only_the_owner_can_admit_a_node(owned, tmp_path):
    from isharded.pool import approve_membership, init_pool, request_membership, sign_entry

    folder, access, client, home = owned
    other = init_pool(str(tmp_path / "other"))
    request = request_membership(str(home), "newcomer", ["10.0.0.9"])
    forged = sign_entry(str(other), {"type": "member", "name": "newcomer", "certificate": approve_membership(str(other), request)})
    assert client.post("/access/entries", json=forged).status_code == 403
    assert client.get("/join/certificate/newcomer").status_code == 404


def test_join_requests_are_bounded_and_spread(owned):
    from isharded.access import AccessControl, Authority, MAX_JOIN_REQUESTS
    from isharded.pool import read_membership_request, request_membership

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


def from_address(app, host):
    async def with_address(scope, receive, send):
        scope["client"] = (host, 5000)
        await app(scope, receive, send)

    return with_address


def fake_request(home, name):
    from isharded.pool import request_membership

    return {"request": request_membership(str(home), name, ["10.0.0.9"])}


def test_a_stranger_cannot_send_much_to_the_routes_anyone_can_reach(owned):
    from isharded.serving import MAX_STRANGER_BYTES

    _, _, client, _ = owned
    big = b'{"request": "' + b"x" * MAX_STRANGER_BYTES + b'"}'
    assert client.post("/join/requests", content=big).status_code == 413
    assert client.post("/access/entries", content=big).status_code == 413


def test_the_largest_real_entry_and_request_fit_under_the_limit(owned):
    import json

    from isharded.pool import approve_membership, request_membership, sign_entry
    from isharded.serving import MAX_STRANGER_BYTES

    folder, _, _, home = owned
    addresses = [f"10.0.0.{index}" for index in range(1, 30)] + ["a-long-name.tailnet-with-a-long-name.ts.net"]
    request = request_membership(str(home), "newcomer", addresses)
    asking = {"request": request, "key_sha256": "a" * 64, "key_verifier": "v" * 43, "proof": "p" * 64}
    entry = sign_entry(folder, {"type": "member", "name": "newcomer", "certificate": approve_membership(folder, request)})
    assert len(json.dumps(asking)) < MAX_STRANGER_BYTES and len(json.dumps(entry)) < MAX_STRANGER_BYTES


def test_one_address_can_only_ask_to_join_so_often(owned):
    from isharded.orchestrator import JOIN_BURST

    _, access, _, home = owned
    client = TestClient(from_address(orchestrator_module.app, "198.51.100.7"))
    for index in range(JOIN_BURST):
        assert client.post("/join/requests", json=fake_request(home, f"n{index}")).status_code == 200
    refused = client.post("/join/requests", json=fake_request(home, "one-too-many"))
    assert refused.status_code == 429 and int(refused.headers["Retry-After"]) >= 1
    assert "one-too-many" not in access.join_requests
    other = TestClient(from_address(orchestrator_module.app, "198.51.100.8"))
    assert other.post("/join/requests", json=fake_request(home, "elsewhere")).status_code == 200


def test_junk_costs_as_much_as_a_real_request_to_join(owned):
    from isharded.orchestrator import JOIN_BURST

    client = TestClient(from_address(orchestrator_module.app, "198.51.100.7"))
    for _ in range(JOIN_BURST):
        assert client.post("/join/requests", json={"request": "junk"}).status_code == 400
    assert client.post("/join/requests", json={"request": "junk"}).status_code == 429


def test_one_address_cannot_hold_the_whole_queue():
    from isharded.access import JOIN_REQUEST_SECONDS, MAX_JOIN_REQUESTS
    from isharded.orchestrator import JOIN_BURST, JOINS_PER_MINUTE

    assert JOIN_BURST + JOINS_PER_MINUTE * JOIN_REQUEST_SECONDS / 60 < MAX_JOIN_REQUESTS


def test_the_limit_sees_one_ipv6_customer_as_one_address():
    from isharded.serving import client_address

    assert client_address("2001:db8:1:2::5") == client_address("2001:db8:1:2:ffff::9")
    assert client_address("2001:db8:1:2::5") != client_address("2001:db8:1:3::5")
    assert client_address("::ffff:198.51.100.7") == client_address("198.51.100.7") == "198.51.100.7"
    assert client_address(None) == "unknown" and client_address("testclient") == "testclient"


def test_a_limit_refills_and_forgets_only_what_it_can():
    from isharded import serving
    from isharded.serving import AddressLimit

    clock = Clock()
    limit = AddressLimit(2, 60.0, clock)
    limit.charge("a")
    limit.charge("a")
    assert limit.wait("a") == 1.0 and limit.wait("b") == 0.0
    clock.now += 1.0
    assert limit.wait("a") == 0.0
    limit.held = {f"x{index}": (0.0, clock.now) for index in range(serving.MOST_ADDRESSES)}
    limit.charge("new")
    assert len(limit.held) == serving.MOST_ADDRESSES and "x0" not in limit.held and "new" in limit.held


def test_a_flood_of_requests_expires_and_the_queue_opens_again(owned):
    from isharded.access import JOIN_REQUEST_SECONDS, MAX_JOIN_REQUESTS
    from isharded.pool import read_membership_request

    _, access, _, home = owned
    clock = Clock()
    access.wall_clock = clock
    text = fake_request(home, "fake")["request"]
    details = read_membership_request(text)
    for index in range(MAX_JOIN_REQUESTS):
        assert access.add_join_request({**details, "name": f"fake{index}"}, text) == "waiting"
    real = fake_request(home, "real")["request"]
    assert access.add_join_request(read_membership_request(real), real) == "busy"
    clock.now += JOIN_REQUEST_SECONDS
    assert access.add_join_request(read_membership_request(real), real) == "waiting"
    assert list(access.join_requests) == ["real"]


def test_an_invite_gets_through_a_queue_full_of_strangers(owned):
    from isharded.access import MAX_JOIN_REQUESTS
    from isharded.pool import read_membership_request

    _, access, client, home = owned
    text = fake_request(home, "fake")["request"]
    details = read_membership_request(text)
    for index in range(MAX_JOIN_REQUESTS):
        access.add_join_request({**details, "name": f"fake{index}"}, text)
    asked = client.post("/join/requests", json={**fake_request(home, "invited"), "proof": "p" * 64, "key_sha256": "k" * 64})
    assert asked.json()["status"] == "waiting"
    assert "invited" in access.join_requests and "fake0" not in access.join_requests
    assert len(access.join_requests) == MAX_JOIN_REQUESTS


def test_a_false_proof_buys_only_minutes(owned):
    from isharded.access import INVITED_REQUEST_SECONDS, JOIN_REQUEST_SECONDS
    from isharded.pool import read_membership_request

    _, access, _, home = owned
    clock = Clock()
    access.wall_clock = clock
    text = fake_request(home, "fake")["request"]
    access.add_join_request(read_membership_request(text), text, {"proof": "p" * 64})
    assert "fake" in access.join_requests
    clock.now += INVITED_REQUEST_SECONDS
    access.expire_join_requests()
    assert "fake" not in access.join_requests
    assert INVITED_REQUEST_SECONDS < JOIN_REQUEST_SECONDS


def test_a_real_invite_outlasts_the_machine_waiting_on_it():
    from isharded.access import INVITED_REQUEST_SECONDS, JOIN_REQUEST_SECONDS
    from isharded.up import APPROVAL_WAIT_SECONDS, PERSON_WAIT_SECONDS

    assert INVITED_REQUEST_SECONDS >= APPROVAL_WAIT_SECONDS and JOIN_REQUEST_SECONDS >= PERSON_WAIT_SECONDS


def test_an_expired_request_does_not_come_back_through_gossip(owned):
    from isharded.access import AccessControl, Authority, JOIN_REQUEST_SECONDS

    folder, access, _, home = owned
    clock = Clock()
    access.wall_clock = clock
    peer = AccessControl(identities=True, authority=Authority(folder + "/ca.pem"), wall_clock=clock)
    text = fake_request(home, "fake")["request"]
    from isharded.pool import read_membership_request

    access.add_join_request(read_membership_request(text), text)
    peer.merge(access.document())
    held = access.document()
    assert "fake" in peer.join_requests
    clock.now += JOIN_REQUEST_SECONDS + 1
    access.expire_join_requests()
    peer.merge(held)  # a late copy, from a node yet to notice
    access.merge(held)
    assert not access.join_requests and not peer.join_requests


def test_asking_again_keeps_a_request_alive_on_every_node(owned):
    from isharded.access import AccessControl, Authority, JOIN_REQUEST_SECONDS
    from isharded.pool import read_membership_request

    folder, access, _, home = owned
    clock = Clock()
    access.wall_clock = clock
    peer = AccessControl(identities=True, authority=Authority(folder + "/ca.pem"), wall_clock=clock)
    text = fake_request(home, "patient")["request"]
    access.add_join_request(read_membership_request(text), text)
    peer.merge(access.document())
    clock.now += JOIN_REQUEST_SECONDS / 2 + 1
    stale = peer.document()
    access.add_join_request(read_membership_request(text), text)
    access.merge(stale)  # an older copy must not undo it
    peer.merge(access.document())
    clock.now += JOIN_REQUEST_SECONDS / 2 + 1
    access.expire_join_requests()
    peer.expire_join_requests()
    assert "patient" in access.join_requests and "patient" in peer.join_requests


@pytest.mark.parametrize("asked_at", ["nonsense", float("nan"), float("-inf"), -5.0])
def test_a_bad_time_from_gossip_never_keeps_a_request(owned, asked_at):
    from isharded.pool import read_membership_request

    _, access, _, home = owned
    text = fake_request(home, "fake")["request"]
    state = access.add_join_request(read_membership_request(text), text, None, asked_at)
    if asked_at == "nonsense":
        assert state == "waiting" and access.join_requests["fake"]["asked_at"] <= access.wall_clock()
    else:
        assert state == "expired" and "fake" not in access.join_requests


def test_a_time_from_the_future_counts_as_now(owned):
    from isharded.pool import read_membership_request

    _, access, _, home = owned
    text = fake_request(home, "fake")["request"]
    access.add_join_request(read_membership_request(text), text, None, access.wall_clock() + 10**9)
    assert access.join_requests["fake"]["asked_at"] <= access.wall_clock()


def test_bad_entries_run_out_and_good_ones_never_do(owned):
    from isharded.orchestrator import WASTED_ENTRY_BURST
    from isharded.pool import sign_entry

    folder, access, _, _ = owned
    client = TestClient(from_address(orchestrator_module.app, "198.51.100.7"))
    for _ in range(WASTED_ENTRY_BURST + 5):
        assert client.post("/access/entries", json=sign_entry(folder, {"type": "revoke", "name": "fresh" + str(len(access.signed))})).status_code == 200
    for _ in range(WASTED_ENTRY_BURST):
        assert client.post("/access/entries", json={"entry": key_entry(), "signature": "AAAA"}).status_code == 403
    assert client.post("/access/entries", json={"entry": key_entry(), "signature": "AAAA"}).status_code == 429
    owner = TestClient(from_address(orchestrator_module.app, "198.51.100.8"))
    assert owner.post("/access/entries", json=sign_entry(folder, key_entry())).status_code == 200


def test_sending_a_good_entry_again_is_a_waste_like_any_other(owned):
    from isharded.orchestrator import WASTED_ENTRY_BURST
    from isharded.pool import sign_entry

    folder, _, _, _ = owned
    runtime = orchestrator_module.runtime
    client = TestClient(from_address(orchestrator_module.app, "198.51.100.7"))
    signed = sign_entry(folder, key_entry())
    events = len(runtime.events)
    for _ in range(WASTED_ENTRY_BURST + 1):
        assert client.post("/access/entries", json=signed).status_code == 200
    assert client.post("/access/entries", json=signed).status_code == 429
    assert len(runtime.events) == events + 1  # one change, one line in the log


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_no_description_of_the_api_is_served_to_strangers(path):
    assert TestClient(orchestrator_module.app).get(path).status_code == 404
    assert TestClient(worker_module.app).get(path).status_code in {401, 404}


def test_the_worker_and_control_plane_routes_do_not_depend_on_the_schema():
    for app in (orchestrator_module.app, worker_module.app):
        assert app.docs_url is None and app.redoc_url is None and app.openapi_url is None


def test_a_stranger_learns_no_model_from_the_ready_check(pool_with_authority):
    pool_with_authority.model_id = "someone/private-model"
    answer = TestClient(orchestrator_module.app).get("/ready")
    assert answer.json() == {"ready": False}
    assert "private-model" not in answer.text


def test_the_public_certificate_comes_with_no_node_name(owned):
    _, _, client, _ = owned
    orchestrator_module.runtime.mesh = ControlMesh("secret-machine", "http://me:1", (), Clock())
    answer = client.get("/join/authority")
    assert list(answer.json()) == ["authority"]
    assert "secret-machine" not in answer.text


def handshake(server, client):
    """Over memory, so antivirus cannot step in the way it does on loopback."""
    import ssl

    client_in, client_out, server_in, server_out = (ssl.MemoryBIO() for _ in range(4))
    near = client.wrap_bio(client_in, client_out, server_hostname="10.0.0.1")
    far = server.wrap_bio(server_in, server_out, server_side=True)
    for _ in range(20):
        finished = 0
        for side in (near, far):
            try:
                side.do_handshake()
                finished += 1
            except ssl.SSLWantReadError:
                pass
        if client_out.pending:
            server_in.write(client_out.read())
        if server_out.pending:
            client_in.write(server_out.read())
        if finished == 2:
            return near.version(), near.cipher()[0]
    raise AssertionError("the handshake did not finish")


@pytest.fixture
def tls_contexts(ticket_pool, monkeypatch):
    import ssl

    from uvicorn.config import SSL_PROTOCOL_VERSION, create_ssl_context

    from isharded.tls import server_options

    monkeypatch.setenv("ISHARDED_TLS_CERT", str(ticket_pool / "node-a.pem"))
    monkeypatch.setenv("ISHARDED_TLS_KEY", str(ticket_pool / "node-a.key"))
    monkeypatch.setenv("ISHARDED_CA_FILE", str(ticket_pool / "ca.pem"))
    options = server_options()
    server = create_ssl_context(options["ssl_certfile"], options["ssl_keyfile"], None, SSL_PROTOCOL_VERSION, options["ssl_cert_reqs"], options["ssl_ca_certs"], options["ssl_ciphers"])
    return server, lambda: ssl.create_default_context(cafile=str(ticket_pool / "ca.pem"))


def test_a_node_speaks_tls_1_3(tls_contexts):
    server, client = tls_contexts
    assert handshake(server, client())[0] == "TLSv1.3"


def test_tls_1_2_still_works_with_a_modern_cipher(tls_contexts):
    import ssl

    server, client = tls_contexts
    older = client()
    older.maximum_version = ssl.TLSVersion.TLSv1_2
    version, cipher = handshake(server, older)
    assert version == "TLSv1.2" and "GCM" in cipher


def test_nothing_below_tls_1_2_is_accepted(tls_contexts):
    import ssl
    import warnings

    server, client = tls_contexts
    old = client()
    old.check_hostname = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old.minimum_version = ssl.TLSVersion.TLSv1
        old.maximum_version = ssl.TLSVersion.TLSv1_1
    old.set_ciphers("ALL:@SECLEVEL=0")
    with pytest.raises(ssl.SSLError):
        handshake(server, old)


def test_no_anonymous_or_cbc_cipher_is_offered(tls_contexts):
    server, _ = tls_contexts
    names = [item["name"] for item in server.get_ciphers()]
    assert names and not [name for name in names if name.startswith(("AECDH", "ADH")) or name.endswith("-SHA")]


def test_a_node_reaches_itself_over_loopback_and_everyone_else_as_named(monkeypatch) -> None:
    from isharded.tls import OWN_ADDRESS, stay_on_this_machine

    def sent(url: str) -> httpx.Request:
        request = httpx.Request("GET", url)
        stay_on_this_machine(request)
        return request

    assert sent("https://203.0.113.5:9000/health").url.host == "203.0.113.5"  # unset: nothing changes
    monkeypatch.setenv(OWN_ADDRESS, "203.0.113.5")
    request = sent("https://203.0.113.5:9000/health")
    assert (request.url.host, request.url.port, request.url.scheme) == ("127.0.0.1", 9000, "https")
    assert request.headers["host"] == "203.0.113.5:9000"
    assert sent("https://203.0.113.6:9000/health").url.host == "203.0.113.6"
    monkeypatch.setenv(OWN_ADDRESS, "203.0.113.5:9000,203.0.113.5:9001")  # behind a relay, sharing its host
    assert sent("https://203.0.113.5:9001/health").url.host == "127.0.0.1"
    assert sent("https://203.0.113.5:9002/health").url.host == "203.0.113.5"
