"""/v1/embeddings: packing, shaping, refusals, and agreement with llama.cpp itself.

Real-model tests need ISHARDED_TEST_LLAMA_DIR and ISHARDED_TEST_EMBED_GGUF. Set
ISHARDED_TEST_EMBED_PORT as well to compare with llama-server on that port.
"""

import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from isharded import orchestrator as orchestrator_module
from isharded import worker as worker_module
from isharded.embedding import (
    MOST_INPUTS,
    PASS_SEQUENCES,
    dimensions_problem,
    encoded,
    inputs_problem,
    length_problem,
    passes,
    shaped,
    smallest_dimensions,
)
from isharded.gguf import GgufHeader, KeyValue, T_BOOL, T_STRING, T_UINT32
from isharded.models import GgufAdapter, UnsupportedModel
from isharded.planner import PlacementError
from isharded.protocol import EmbeddingRequest, LoadModelRequest, StageAssignment
from isharded.tokenizer import gguf_marks

WIDTH = 8


def gguf_header(pooling=3, name="Qwen3 Embedding 0.6B", **extra):
    values = {
        "general.architecture": (T_STRING, "qwen3"),
        "general.name": (T_STRING, name),
        "qwen3.embedding_length": (T_UINT32, WIDTH),
        **extra,
    }
    if pooling is not None:
        values["qwen3.pooling_type"] = (T_UINT32, pooling)
    keys = {key: KeyValue(key, kind, b"", value) for key, (kind, value) in values.items()}
    return GgufHeader(3, keys, {}, 0, 0)


def adapter_for(header):
    made = GgufAdapter.__new__(GgufAdapter)
    made.header = header
    return made


class Tokens:
    """One token per word, and one more to pool at."""

    def encode_input(self, text):
        return [len(word) for word in text.split()] + [99]


def vectors_for(inputs):
    """A row per input that shows which input it came from."""
    rows = np.zeros((len(inputs), WIDTH), dtype=np.float32)
    rows[:, 0] = [len(item) for item in inputs]
    rows[:, 1] = 1.0
    return base64.b64encode(rows.tobytes()).decode()


@pytest.fixture
def serving(monkeypatch):
    """A control plane running an embedding model, with a fake entry worker behind it."""
    runtime = orchestrator_module.runtime
    seen = []

    def entry(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"vectors": vectors_for(body["inputs"])})

    monkeypatch.setattr(runtime, "model_id", "embed.gguf")
    monkeypatch.setattr(runtime, "tokenizer", Tokens())
    monkeypatch.setattr(runtime, "route", [SimpleNamespace(url="http://entry:1", worker_id="w")])
    monkeypatch.setattr(runtime, "epoch", 4)
    monkeypatch.setattr(
        runtime,
        "active",
        SimpleNamespace(embedding=True, adapter=adapter_for(gguf_header()), profile=SimpleNamespace(context_tokens=32)),
    )
    monkeypatch.setattr(runtime, "client", httpx.AsyncClient(transport=httpx.MockTransport(entry)))
    from fastapi.testclient import TestClient

    return TestClient(orchestrator_module.app), seen


def test_inputs_pack_in_order_under_the_budget():
    assert passes([10, 10, 10, 10], budget=20) == [[0, 1], [2, 3]]
    assert passes([], budget=20) == []


def test_a_pass_never_holds_more_inputs_than_sequences():
    groups = passes([1] * (PASS_SEQUENCES + 1), budget=10_000)
    assert [len(group) for group in groups] == [PASS_SEQUENCES, 1]


def test_an_input_past_the_budget_gets_a_pass_of_its_own():
    assert passes([5, 500, 5], budget=20) == [[0], [1], [2]]


def test_a_vector_comes_out_unit_length():
    assert np.linalg.norm(shaped(np.array([3.0, 4.0], dtype=np.float32))) == pytest.approx(1.0)


def test_shortening_cuts_first_and_scales_after():
    assert shaped(np.array([3.0, 4.0, 100.0], dtype=np.float32), 2).tolist() == pytest.approx([0.6, 0.8])


def test_a_zero_vector_stays_zero_instead_of_becoming_nan():
    assert shaped(np.zeros(4, dtype=np.float32)).tolist() == [0, 0, 0, 0]


def test_base64_is_little_endian_float32():
    vector = np.array([1.5, -2.25], dtype=np.float32)
    assert np.frombuffer(base64.b64decode(encoded(vector, "base64")), dtype="<f4").tolist() == [1.5, -2.25]
    assert encoded(vector, "float") == [1.5, -2.25]


@pytest.mark.parametrize("names, expected", [(["Qwen3 Embedding 0.6B"], 32), (["qwen3-embedding"], 32), (["Qwen3_Embedding"], 32), (["bge-m3"], None), ([], None)])
def test_only_families_trained_for_it_can_be_shortened(names, expected):
    assert smallest_dimensions(names) == expected


@pytest.mark.parametrize(
    "asked, smallest, ok",
    [(None, None, True), (1024, None, True), (256, None, False), (256, 32, True), (32, 32, True), (31, 32, False), (2048, 32, False)],
)
def test_dimensions_are_checked_against_the_model(asked, smallest, ok):
    assert (dimensions_problem(asked, 1024, smallest) is None) is ok


def test_input_is_checked():
    assert inputs_problem(["a"]) is None
    assert "at least one" in inputs_problem([])
    assert "input 1 is empty" in inputs_problem(["a", ""])
    assert str(MOST_INPUTS) in inputs_problem(["a"] * (MOST_INPUTS + 1))


def test_text_past_the_context_is_refused_rather_than_cut():
    assert length_problem([4, 8], 8) is None
    problem = length_problem([4, 9], 8)
    assert "input 1 is 9 tokens" in problem and "--context-tokens" in problem


def test_a_request_needs_a_known_encoding():
    assert EmbeddingRequest(input="x").encoding_format == "float"
    with pytest.raises(ValueError):
        EmbeddingRequest(input="x", encoding_format="hex")


def test_marks_follow_what_the_gguf_asks_for():
    header = gguf_header(**{"tokenizer.ggml.add_eos_token": (T_BOOL, True), "tokenizer.ggml.eos_token_id": (T_UINT32, 7), "tokenizer.ggml.bos_token_id": (T_UINT32, 3)})
    assert gguf_marks(header) == ([], [7])
    assert gguf_marks(gguf_header()) == ([], [])


@pytest.mark.parametrize("pooling", [1, 2, 3])
def test_an_embedding_model_has_to_be_loaded_as_one(pooling):
    adapter = adapter_for(gguf_header(pooling=pooling))
    orchestrator_module.check_pooling(LoadModelRequest(model_id="m.gguf", embedding=True), adapter)
    with pytest.raises(UnsupportedModel, match="--embedding"):
        orchestrator_module.check_pooling(LoadModelRequest(model_id="m.gguf"), adapter)


@pytest.mark.parametrize("pooling", [None, 0, 4])
def test_a_model_without_pooling_cannot_be_loaded_for_embeddings(pooling):
    adapter = adapter_for(gguf_header(pooling=pooling))
    orchestrator_module.check_pooling(LoadModelRequest(model_id="m.gguf"), adapter)
    with pytest.raises(UnsupportedModel, match="no pooling"):
        orchestrator_module.check_pooling(LoadModelRequest(model_id="m.gguf", embedding=True), adapter)


def test_a_model_without_a_gguf_cannot_be_an_embedding_model():
    with pytest.raises(UnsupportedModel, match="GGUF"):
        orchestrator_module.check_pooling(LoadModelRequest(model_id="m", embedding=True), object())


def test_an_embedding_model_takes_no_draft():
    request = LoadModelRequest(model_id="m.gguf", embedding=True)
    assert orchestrator_module.resolve_draft(request, None, adapter_for(gguf_header())) is None
    with pytest.raises(UnsupportedModel, match="draft"):
        orchestrator_module.resolve_draft(request.model_copy(update={"draft_model": "d.gguf"}), None, adapter_for(gguf_header()))


def stage(worker, start, end):
    return StageAssignment(worker_id=worker, url="http://x:1", start_layer=start, end_layer=end, placements=[], estimated_ms_per_token=1.0)


@pytest.mark.parametrize("stages, refused", [(1, False), (2, True)])
def test_an_embedding_model_is_never_split(monkeypatch, stages, refused):
    route = [stage("a", 0, 8), stage("b", 8, 16)][:stages]
    monkeypatch.setattr(orchestrator_module, "plan_placement", lambda *args, **kwargs: route)
    runtime = orchestrator_module.Orchestrator()
    runtime.candidate_groups = lambda: ([], [])
    target = SimpleNamespace(
        model_id="m.gguf", embedding=True, entry_worker=None, exit_worker=None, profile=None, spread=False, reuse_cached=False
    )
    if refused:
        with pytest.raises(PlacementError, match="cannot be split"):
            runtime.plan(target)
    else:
        assert runtime.plan(target) == route


def test_vectors_keep_the_order_of_the_input_across_passes(serving, monkeypatch):
    client, seen = serving
    monkeypatch.setattr(orchestrator_module, "passes", lambda lengths: [[1], [0, 2]])
    answer = client.post("/v1/embeddings", json={"input": ["one", "one two", "one two three"]}).json()
    assert len(seen) == 2
    assert [item["index"] for item in answer["data"]] == [0, 1, 2]
    # Tokens per input show through as the first column, before scaling.
    ratios = [item["embedding"][0] / item["embedding"][1] for item in answer["data"]]
    assert ratios == pytest.approx([2.0, 3.0, 4.0])
    assert answer["usage"] == {"prompt_tokens": 9, "total_tokens": 9}
    assert answer["object"] == "list" and answer["model"] == "embed.gguf"


def test_a_string_input_gives_one_vector_at_the_current_epoch(serving):
    client, seen = serving
    answer = client.post("/v1/embeddings", json={"input": "hello there"}).json()
    assert seen[0]["epoch"] == 4 and seen[0]["inputs"] == [[5, 5, 99]]
    assert len(answer["data"]) == 1 and answer["data"][0]["object"] == "embedding"


def test_base64_and_dimensions_reach_the_caller(serving):
    client, _ = serving
    answer = client.post("/v1/embeddings", json={"input": "x", "encoding_format": "base64"}).json()
    assert np.frombuffer(base64.b64decode(answer["data"][0]["embedding"]), dtype="<f4").size == WIDTH
    # This fake is 8 wide, under the family's 32, so asking to shorten it has to be refused.
    assert client.post("/v1/embeddings", json={"input": "x", "dimensions": 4}).status_code == 400


def test_text_past_the_context_is_a_plain_error(serving):
    client, seen = serving
    answer = client.post("/v1/embeddings", json={"input": ["ok", "word " * 40]})
    assert answer.status_code == 400 and "input 1 is 41 tokens" in answer.json()["detail"]
    assert not seen, "nothing should reach a stage"


def test_chat_on_an_embedding_model_is_refused(serving):
    client, _ = serving
    for path, body in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}), ("/v1/completions", {"prompt": "hi"})):
        answer = client.post(path, json=body)
        assert answer.status_code == 400 and "embedding model" in answer.json()["detail"]


def test_a_chat_model_asked_for_embeddings_says_so_and_names_what_can(serving, monkeypatch):
    client, _ = serving
    runtime = orchestrator_module.runtime
    monkeypatch.setattr(runtime, "active", SimpleNamespace(embedding=False))
    answer = client.post("/v1/embeddings", json={"input": "x"})
    assert answer.status_code == 400 and "chat model" in answer.json()["detail"] and "--embedding" in answer.json()["detail"]
    monkeypatch.setattr(runtime.lanes, "model_kinds", lambda: {"e.gguf": {"kind": "embedding"}, "c.gguf": {"kind": "chat"}})
    assert "e.gguf" in client.post("/v1/embeddings", json={"input": "x"}).json()["detail"]


def test_models_say_what_each_is_for(serving, monkeypatch):
    client, _ = serving
    runtime = orchestrator_module.runtime
    monkeypatch.setattr(runtime.lanes, "model_kinds", lambda: {"chat.gguf": {"kind": "chat", "vision": True}})
    rows = {row["id"]: row for row in client.get("/v1/models").json()["data"]}
    assert rows["embed.gguf"]["kind"] == "embedding" and rows["embed.gguf"]["dimensions"] == WIDTH
    assert rows["chat.gguf"]["kind"] == "chat" and rows["chat.gguf"]["vision"] is True


def test_embeddings_are_behind_the_same_rate_limit_as_chat(serving, monkeypatch):
    client, seen = serving
    runtime = orchestrator_module.runtime
    monkeypatch.setattr(runtime.access, "spend", lambda caller: 3.0)
    answer = client.post("/v1/embeddings", json={"input": "x"})
    assert answer.status_code == 429 and not seen


def test_a_request_for_a_lanes_model_goes_to_that_lane(serving, monkeypatch):
    client, seen = serving
    runtime = orchestrator_module.runtime
    sent = {}

    async def locate(model):
        return "small"

    async def forward(lane, path, payload, caller):
        sent.update(lane=lane, path=path, payload=payload)
        return {"object": "list", "data": []}

    monkeypatch.setattr(runtime.lanes, "locate", locate)
    monkeypatch.setattr(runtime.lanes, "forward", forward)
    client.post("/v1/embeddings", json={"input": "x", "model": "lane.gguf", "dimensions": 64})
    assert sent == {"lane": "small", "path": "/v1/embeddings", "payload": {"model": "lane.gguf", "input": "x", "encoding_format": "float", "dimensions": 64}}
    assert not seen


def test_a_restarted_stage_costs_a_retry_and_not_the_request(serving, monkeypatch):
    client, _ = serving
    runtime = orchestrator_module.runtime
    answers = iter([httpx.Response(503, json={"detail": {"retry": True, "message": "the stage process died"}})])

    def entry(request):
        body = json.loads(request.content)
        return next(answers, None) or httpx.Response(200, json={"vectors": vectors_for(body["inputs"])})

    monkeypatch.setattr(runtime, "client", httpx.AsyncClient(transport=httpx.MockTransport(entry)))
    answer = client.post("/v1/embeddings", json={"input": "x"})
    assert answer.status_code == 200 and len(answer.json()["data"]) == 1


def test_a_worker_that_refuses_a_request_is_not_blamed_for_it(serving, monkeypatch):
    client, _ = serving
    runtime = orchestrator_module.runtime
    monkeypatch.setattr(runtime, "client", httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(400, json={"detail": "inputs cannot be empty"}))))
    answer = client.post("/v1/embeddings", json={"input": "x"})
    assert answer.status_code == 400 and answer.json()["detail"] == "inputs cannot be empty"


class FakeStage:
    embedding = True

    def __init__(self, failure=None):
        self.failure = failure

    def pool(self, inputs):
        if self.failure:
            raise self.failure
        return np.ones((len(inputs), WIDTH), dtype=np.float32)


@pytest.fixture
def worker_client(monkeypatch):
    from fastapi.testclient import TestClient

    runtime = worker_module.runtime
    monkeypatch.setattr(runtime, "gguf", FakeStage())
    monkeypatch.setattr(runtime, "epoch", 2)
    monkeypatch.setattr(runtime, "gate", worker_module.SlotGate(1))
    return TestClient(worker_module.app)


def test_a_worker_answers_one_pass(worker_client):
    answer = worker_client.post("/v1/embed", json={"epoch": 2, "inputs": [[1, 2], [3]]})
    assert answer.status_code == 200
    assert np.frombuffer(base64.b64decode(answer.json()["vectors"]), dtype=np.float32).size == 2 * WIDTH


def test_a_worker_on_another_route_says_so(worker_client):
    assert worker_client.post("/v1/embed", json={"epoch": 1, "inputs": [[1]]}).status_code == 409


def test_a_worker_running_a_chat_model_refuses(worker_client, monkeypatch):
    monkeypatch.setattr(worker_module.runtime.gguf, "embedding", False, raising=False)
    assert worker_client.post("/v1/embed", json={"epoch": 2, "inputs": [[1]]}).status_code == 409


def test_a_worker_gives_its_place_back_after_a_failed_pass(worker_client, monkeypatch):
    runtime = worker_module.runtime
    monkeypatch.setattr(runtime, "gguf", FakeStage(worker_module.LlamaError("no")))
    assert worker_client.post("/v1/embed", json={"epoch": 2, "inputs": [[1]]}).status_code == 500
    assert runtime.gate.free == {0}


def test_a_full_queue_is_a_429(worker_client, monkeypatch):
    monkeypatch.setattr(worker_module.runtime, "waiting", worker_module.MAX_WAITING_REPLIES)
    assert worker_client.post("/v1/embed", json={"epoch": 2, "inputs": [[1]]}).status_code == 429


def test_the_chat_page_leaves_embedding_models_out_of_its_picker():
    text = (Path(orchestrator_module.__file__).parent / "chat.html").read_text(encoding="utf-8")
    choice = text[text.index("function renderChoice"):text.index("function setPill")]
    assert "status.kind !== 'embedding'" in choice and "lane.kind !== 'embedding'" in choice


def test_an_embedding_stage_is_rebuilt_when_the_mode_changes():
    from isharded.protocol import ConfigureRequest

    assignment = stage("w", 0, 8)
    plain = ConfigureRequest(model_id="m", revision="r", architecture={}, epoch=1, assignment=assignment, route=[assignment])
    signature = worker_module.WorkerRuntime._stage_signature
    assert signature(plain) != signature(plain.model_copy(update={"embedding": True}))


BUILD = os.environ.get("ISHARDED_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("ISHARDED_TEST_EMBED_GGUF", "")
PORT = os.environ.get("ISHARDED_TEST_EMBED_PORT", "")
needs_real_model = pytest.mark.skipif(
    not (BUILD and MODEL and Path(BUILD).is_dir() and Path(MODEL).is_file()), reason="needs a llama.cpp build and an embedding GGUF"
)
TEXTS = [
    "The cat sat quietly on the warm windowsill.",
    "A kitten was resting on a sunny window ledge.",
    "How do I reset my router to factory settings?",
    "Steps for restoring a wifi router back to its defaults",
    "Quarterly revenue grew by twelve percent in Europe.",
    "Photosynthesis turns sunlight and water into sugar.",
]


def unit_rows(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def real_stage(tmp_path_factory):
    """A whole model in a stage process, built the way a worker builds one."""
    from isharded.gguf import file_reader, plan_stage, read_file_header, write_stage
    from isharded.llama_host import StageHost
    from isharded.tokenizer import tokenizer_from_gguf

    header = read_file_header(MODEL)
    path = tmp_path_factory.mktemp("embed") / "stage.gguf"
    with path.open("wb") as handle:
        write_stage(header, plan_stage(header, 0, header.layer_count, last=True), handle, file_reader(MODEL))
    table = header.tensors["token_embd.weight"]
    host = StageHost()
    host.call(
        "load",
        folder=BUILD,
        extra_paths=[],
        path=str(path),
        last=True,
        context_tokens=2048,
        gpu_layers=0,
        threads=4,
        cpu_tensors=[],
        table={"path": MODEL, "offset": header.data_start + table.offset, "length": table.nbytes, "ggml_type": table.ggml_type},
        map_file=False,
        embedding=True,
    )
    tokenizer = tokenizer_from_gguf(header)

    def embed(texts):
        inputs = [tokenizer.encode_input(text) for text in texts]
        answer, data = host.call("pool", inputs=inputs)
        return unit_rows(np.frombuffer(data, dtype=np.float32).reshape(len(inputs), answer["width"]).astype(np.float64))

    yield embed
    host.close()


@needs_real_model
def test_similar_sentences_score_higher_than_unrelated_ones(real_stage):
    rows = real_stage(TEXTS)
    similar = [rows[0] @ rows[1], rows[2] @ rows[3]]
    unrelated = [rows[a] @ rows[b] for a in (0, 1) for b in (2, 3, 4, 5)] + [rows[a] @ rows[b] for a in (2, 3) for b in (4, 5)]
    assert min(similar) > max(unrelated)


@needs_real_model
def test_a_batch_gives_what_inputs_give_alone(real_stage):
    alone = np.concatenate([real_stage([text]) for text in TEXTS])
    together = real_stage(TEXTS)
    # llama-server's own vectors move this much between batched and alone.
    assert np.sum(alone * together, axis=1).min() > 0.999


@needs_real_model
@pytest.mark.skipif(not PORT, reason="needs ISHARDED_TEST_EMBED_PORT for llama-server")
def test_vectors_match_llama_servers_own(real_stage):
    executable = next(path for path in (Path(BUILD) / "llama-server.exe", Path(BUILD) / "llama-server") if path.is_file())
    server = subprocess.Popen(
        [str(executable), "-m", MODEL, "--embedding", "--pooling", "last", "--port", PORT, "--host", "127.0.0.1", "-ngl", "0", "-t", "4", "-c", "2048", "--no-op-offload"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{PORT}"
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if httpx.get(url + "/health", timeout=4, trust_env=False).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        theirs = []
        for text in TEXTS:
            reply = httpx.post(url + "/v1/embeddings", json={"input": [text], "model": "x"}, timeout=120, trust_env=False)
            theirs.append(reply.json()["data"][0]["embedding"])
    finally:
        server.terminate()
        server.wait(15)
    ours = np.concatenate([real_stage([text]) for text in TEXTS])
    # One input at a time runs the same kernels, so this is agreement, not similarity.
    assert np.sum(ours * unit_rows(np.array(theirs, dtype=np.float64)), axis=1).min() > 0.99999
