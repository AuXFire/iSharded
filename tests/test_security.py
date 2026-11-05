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
