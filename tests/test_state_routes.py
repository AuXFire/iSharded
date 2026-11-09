"""What a node does with saved states that reach it over the network.

These routes take bytes from another machine and turn them into files, so they get
tested from outside, as a peer or a stranger would reach them.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from isharded import worker
from isharded.protocol import META_HEADER, encode_meta
from isharded.statestore import HISTORY, StateStore

SPACE = "ab12"
KEY = "cd34"


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A node with a real store for copies, and a pool token."""
    held = StateStore(tmp_path / "held", 0, 1024 * 1024)
    runtime = SimpleNamespace(
        held=held, pool_token="secret", requires_identity=False, revoked_nodes=set(), authorized=None,
    )
    monkeypatch.setattr(worker, "runtime", runtime)
    yield runtime
    held.close()


def client():
    return TestClient(worker.app)


def token(value="secret"):
    return {"authorization": f"Bearer {value}"}


def state_file(tmp_path, data=7, size=64):
    """The bytes of a good state, as a stage's store wrote them."""
    source = StateStore(tmp_path / "source", 1024, 1024)
    source.put("x", np.full(size, data, dtype=np.uint8), 300, HISTORY)
    source.flush()
    content = source.file_of("x").read_bytes()
    source.close()
    return content


def send(content, namespace=SPACE, key=KEY, auth=None, **meta):
    headers = {**(token() if auth is None else auth), META_HEADER: encode_meta({"namespace": namespace, "key": key, **meta})}
    return client().put("/internal/state/copy", content=content, headers=headers)


def test_a_good_copy_is_kept_and_can_be_fetched_back(node, tmp_path):
    content = state_file(tmp_path)
    assert send(content).status_code == 200
    listing = client().get(f"/internal/state/held/{SPACE}", headers=token()).json()
    assert listing == {"states": [{"key": KEY, "length": 300, "kind": HISTORY}]}
    assert client().get(f"/internal/state/held/{SPACE}/{KEY}", headers=token()).content == content


def test_a_newer_turn_replaces_its_older_copy(node, tmp_path):
    send(state_file(tmp_path), key="aa")
    send(state_file(tmp_path, data=8), key="bb", replaces="aa")
    keys = [item["key"] for item in client().get(f"/internal/state/held/{SPACE}", headers=token()).json()["states"]]
    assert keys == ["bb"]


def test_copies_for_other_stages_stay_out_of_a_listing(node, tmp_path):
    send(state_file(tmp_path), namespace="ff00")
    assert client().get(f"/internal/state/held/{SPACE}", headers=token()).json() == {"states": []}


@pytest.mark.parametrize(
    "method, path",
    [
        ("get", f"/internal/state/held/{SPACE}"),
        ("get", f"/internal/state/held/{SPACE}/{KEY}"),
        ("get", "/internal/state/index"),
        ("post", "/internal/state/prepare"),
        ("post", "/internal/state/keep"),
        ("post", "/internal/state/restore"),
    ],
)
def test_a_stranger_gets_nothing(node, method, path):
    extra = {"json": {"key": "abc", "namespaces": []}} if method == "post" else {}
    assert getattr(client(), method)(path, headers=token("wrong"), **extra).status_code == 401


def test_a_stranger_cannot_store_anything(node, tmp_path):
    assert send(state_file(tmp_path), auth=token("wrong")).status_code == 401
    assert node.held.listing() == []
    assert not list((tmp_path / "held").iterdir())


@pytest.mark.parametrize(
    "namespace, key",
    [
        ("../../evil", KEY),
        (SPACE, "../../evil"),
        (SPACE, "..\\..\\evil"),
        ("AB12", KEY),  # names are digests, in lower case
        (SPACE, "x" * 65),
        ("", KEY),
    ],
)
def test_a_name_can_only_ever_be_a_digest(node, tmp_path, namespace, key):
    assert send(state_file(tmp_path), namespace=namespace, key=key).status_code == 400
    assert not list(tmp_path.rglob("evil*"))
    assert node.held.listing() == []


def test_a_replaced_name_is_checked_too(node, tmp_path):
    assert send(state_file(tmp_path), replaces="../../evil").status_code == 400


@pytest.mark.parametrize("header", ["", "not base64 at all", encode_meta({"key": KEY})])
def test_a_copy_without_a_proper_header_is_refused(node, tmp_path, header):
    response = client().put("/internal/state/copy", content=state_file(tmp_path), headers={**token(), META_HEADER: header})
    assert response.status_code == 400


def test_a_copy_without_a_header_at_all_is_refused(node, tmp_path):
    assert client().put("/internal/state/copy", content=b"x", headers=token()).status_code == 400


def test_a_corrupt_copy_is_refused_and_nothing_stays_behind(node, tmp_path):
    content = bytearray(state_file(tmp_path))
    content[-1] ^= 1
    assert send(bytes(content)).status_code == 400
    assert node.held.listing() == []
    assert not list((tmp_path / "held").iterdir())


def test_a_copy_too_big_for_this_node_is_refused(node, tmp_path):
    node.held.disk_bytes = 10
    assert send(state_file(tmp_path)).status_code == 413


def test_a_copy_is_refused_rather_than_fill_the_disk(node, tmp_path, monkeypatch):
    monkeypatch.setattr(node.held, "has_room", lambda size: False)
    assert send(state_file(tmp_path)).status_code == 507
    assert node.held.listing() == []


def test_a_node_with_no_disk_for_states_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "runtime", SimpleNamespace(held=StateStore(None, 0, 0), pool_token="secret", requires_identity=False, revoked_nodes=set()))
    assert send(b"x").status_code == 409


def test_a_copy_that_is_not_here_is_a_404(node):
    assert client().get(f"/internal/state/held/{SPACE}/{KEY}", headers=token()).status_code == 404


@pytest.mark.parametrize("key", ["../../evil", "..%2F..%2Fevil", "ZZ"])
def test_a_fetch_cannot_reach_outside_the_store(node, key):
    assert client().get(f"/internal/state/held/{SPACE}/{key}", headers=token()).status_code in {400, 404}


def test_only_copies_for_running_stages_are_kept(node, tmp_path):
    send(state_file(tmp_path), namespace="aa", key="11")
    send(state_file(tmp_path), namespace="bb", key="22")
    assert client().post("/internal/state/keep", json={"namespaces": ["bb"]}, headers=token()).status_code == 200
    assert [item["key"] for item in node.held.listing()] == ["bb.22"]


def test_a_stage_asked_to_prepare_a_bad_name_does_nothing(node):
    async def ensure(key):
        raise AssertionError("a name like that goes nowhere near the store")

    node.ensure_state = ensure
    assert client().post("/internal/state/prepare", json={"key": "../evil"}, headers=token()).json() == {"ready": False}


def test_a_stage_asked_to_prepare_answers(node):
    async def ensure(key):
        return key == "abc"

    node.ensure_state = ensure
    assert client().post("/internal/state/prepare", json={"key": "abc"}, headers=token()).json() == {"ready": True}


@pytest.mark.parametrize(
    "body",
    [
        {"key": "../evil", "session_id": "s", "slot": 0, "epoch": 1, "length": 10},
        {"key": "abc", "session_id": "s", "slot": 2, "epoch": 1, "length": 10},
        {"key": "abc", "session_id": "s", "slot": -1, "epoch": 1, "length": 10},
        {"key": "abc", "session_id": "s", "slot": 0, "epoch": 1, "length": 0},
        {"key": "abc", "slot": 0, "epoch": 1, "length": 10},
        {"key": "abc", "session_id": "s", "slot": "x", "epoch": 1, "length": 10},
    ],
)
def test_a_restore_ahead_for_a_bad_name_or_slot_does_nothing(node, body):
    async def take_restore(*_):
        raise AssertionError("a request like that goes nowhere near a slot")

    node.gate, node.take_restore = SimpleNamespace(size=2), take_restore
    assert client().post("/internal/state/restore", json=body, headers=token()).status_code == 400


def test_a_stage_asked_to_restore_ahead_answers(node):
    asked = []

    async def take_restore(*args):
        asked.append(args)
        return True

    node.gate, node.take_restore = SimpleNamespace(size=2), take_restore
    body = {"key": "abc", "session_id": "s", "slot": 1, "epoch": 3, "length": 10}
    assert client().post("/internal/state/restore", json=body, headers=token()).json() == {"restored": True}
    # A sender that gives no generation is taken as before, with zero.
    assert client().post("/internal/state/restore", json={**body, "generation": 42}, headers=token()).json() == {"restored": True}
    assert asked == [("s", 3, 1, "abc", 10, 0), ("s", 3, 1, "abc", 10, 42)]


def test_the_index_says_what_this_stage_can_restore(node):
    async def restorable():
        return {"abc": {"key": "abc", "length": 300, "kind": HISTORY, "elsewhere": False}}

    node.restorable = restorable
    assert client().get("/internal/state/index", headers=token()).json()["states"][0]["key"] == "abc"
