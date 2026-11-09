"""Images in chat: what is refused, how rows sit in a prompt, and that a split sees what one stage sees.

Real-model tests need ISHARDED_TEST_LLAMA_DIR, ISHARDED_TEST_VISION_GGUF, and
ISHARDED_TEST_VISION_MMPROJ.
"""

import asyncio
import base64
import ctypes as C
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
import zlib

import httpx
import numpy as np
import pytest
import torch

from isharded import orchestrator as orchestrator_module
from isharded import worker as worker_module
from isharded.gguf import GgufHeader, KeyValue, T_ARRAY, T_BOOL, T_STRING, T_UINT32
from isharded.gguf_stage import ImageEncoder, plan_prompt
from isharded.statestore import HISTORY, PrefixIndex, prefix_key
from isharded.llamacpp import LlamaError, StageLost
from isharded.models import GgufAdapter, UnsupportedModel
from isharded.mtmd import BitmapWrapper, ContextParams, DecoderPosition, InputText, VideoParams
from isharded.protocol import InferRequest, LoadModelRequest, SessionState, decode_pass, encode_activation
from isharded.vision import (
    MOST_IMAGE_BYTES,
    MOST_IMAGE_PIXELS,
    MOST_IMAGES,
    SECTIONS,
    EncodedImage,
    ImageRefused,
    chat_images,
    check_image,
    expand_prompt,
    image_id,
    image_size,
    lay_out,
    projector_marker,
    read_image_url,
)


def png(width=4, height=3, colour=(200, 10, 10)):
    """A real PNG, built by hand so the tests need no imaging library."""

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\0" + bytes(colour) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def data_url(data):
    return "data:image/png;base64," + base64.b64encode(data).decode()


def image_part(data):
    return {"type": "image_url", "image_url": {"url": data_url(data)}}


def test_size_comes_from_each_header():
    assert image_size(png(7, 5)) == (7, 5)
    assert image_size(b"GIF89a" + struct.pack("<HH", 9, 8) + bytes(4)) == (9, 8)
    assert image_size(b"BM" + bytes(16) + struct.pack("<ii", 6, -4)) == (6, 4)  # top-down BMPs are negative


def test_a_jpeg_size_is_found_past_its_other_segments():
    app0 = b"\xff\xe0" + struct.pack(">H", 6) + b"JFIF"
    frame = b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 300, 200) + bytes(12)
    assert image_size(b"\xff\xd8" + app0 + frame) == (200, 300)


@pytest.mark.parametrize("data", [b"", b"hello there", b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"RIFF\0\0\0\0WEBPVP8 "])
def test_anything_else_is_not_an_image(data):
    with pytest.raises(ImageRefused, match="PNG, JPEG"):
        image_size(data)


def test_a_small_file_cannot_claim_a_huge_picture():
    with pytest.raises(ImageRefused, match="smaller"):
        check_image(png(10_000, 10_000)[:40])
    check_image(png(16, 16))
    assert 10_000 * 10_000 > MOST_IMAGE_PIXELS


def test_an_image_with_no_pixels_is_refused():
    with pytest.raises(ImageRefused, match="no pixels"):
        check_image(b"GIF89a" + bytes(10))


@pytest.mark.parametrize("url", ["http://169.254.169.254/latest", "https://example.com/a.png", "HTTP://example.com/a.png"])
def test_a_remote_image_is_refused_and_never_fetched(url, monkeypatch):
    def fetched(*args, **kwargs):
        raise AssertionError("nothing may be fetched")

    monkeypatch.setattr(httpx, "get", fetched)
    with pytest.raises(ImageRefused, match="data URL"):
        read_image_url(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/a.png", "data:image/png,rawtext", "/local/path.png", ""])
def test_only_base64_data_urls_are_read(url):
    with pytest.raises(ImageRefused, match="base64 data URL"):
        read_image_url(url)


def test_bad_base64_says_so():
    with pytest.raises(ImageRefused, match="not valid base64"):
        read_image_url("data:image/png;base64,@@@@")


def test_a_data_url_gives_its_bytes():
    assert read_image_url(data_url(b"abc")) == b"abc"


def test_size_is_refused_before_decoding():
    with pytest.raises(ImageRefused, match="10 MiB"):
        read_image_url("data:image/png;base64," + "A" * (MOST_IMAGE_BYTES * 2))


def test_images_come_out_in_order_with_their_places_marked():
    first, second = png(colour=(1, 2, 3)), png(colour=(4, 5, 6))
    messages = [
        {"role": "user", "content": [image_part(first), {"type": "text", "text": "and"}, image_part(second)]},
        {"role": "assistant", "content": "ok"},
    ]
    plain, images = chat_images(messages)
    assert images == [first, second]
    assert plain[0]["content"] == [{"type": "image"}, {"type": "text", "text": "and"}, {"type": "image"}]
    assert plain[1] == {"role": "assistant", "content": "ok"}


def test_text_parts_alone_become_plain_text():
    plain, images = chat_images([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert plain == [{"role": "user", "content": "ab"}] and images == []


def test_a_message_of_plain_text_is_left_alone():
    message = {"role": "user", "content": "hi"}
    assert chat_images([message]) == ([message], [])


@pytest.mark.parametrize("role", ["system", "assistant", "tool"])
def test_only_user_messages_hold_images(role):
    with pytest.raises(ImageRefused, match="only user"):
        chat_images([{"role": role, "content": [image_part(png())]}])


@pytest.mark.parametrize(
    "part, reason",
    [({"type": "input_audio"}, "not supported"), ("just a string", "not supported"), ({"type": "image_url", "image_url": {}}, "needs a url"), ({"type": "image_url"}, "needs a url")],
)
def test_parts_that_cannot_be_used_are_refused(part, reason):
    with pytest.raises(ImageRefused, match=reason):
        chat_images([{"role": "user", "content": [part]}])


def test_a_conversation_has_a_limit_on_how_many_and_how_much():
    small = png()
    with pytest.raises(ImageRefused, match="16"):
        chat_images([{"role": "user", "content": [image_part(small)] * (MOST_IMAGES + 1)}])
    large = png() + bytes(MOST_IMAGE_BYTES // 2 + 1024)
    with pytest.raises(ImageRefused, match="10 MiB"):
        chat_images([{"role": "user", "content": [image_part(large), image_part(large)]}])


def projector_header(**changes):
    values = {
        "general.architecture": (T_STRING, "clip"),
        "clip.has_vision_encoder": (T_BOOL, True),
        "clip.projector_type": (T_STRING, "qwen3vl_merger"),
        "clip.vision.projection_dim": (T_UINT32, 1024),
        **changes,
    }
    keys = {key: KeyValue(key, kind, b"", value) for key, (kind, value) in values.items()}
    return GgufHeader(3, keys, {}, 0, 0)


def test_a_matching_projector_gives_its_marker():
    assert projector_marker(projector_header(), 1024, "p.gguf") == "<|image_pad|>"


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"general.architecture": (T_STRING, "qwen35")}, "not a projector"),
        ({"clip.has_vision_encoder": (T_BOOL, False)}, "no vision encoder"),
        ({"clip.projector_type": (T_STRING, "mlp")}, "not supported yet"),
        ({"clip.vision.projection_dim": (T_UINT32, 2048)}, "belongs to another model"),
    ],
)
def test_a_projector_that_cannot_work_is_refused_at_load(changes, reason):
    with pytest.raises(ImageRefused, match=reason):
        projector_marker(projector_header(**changes), 1024, "p.gguf")


def test_deepstack_is_refused_because_a_stage_takes_one_hidden_state():
    header = projector_header()
    raw = struct.pack("<IQ", T_BOOL, 2) + bytes([0, 1])
    header.keys["clip.vision.is_deepstack_layers"] = KeyValue("clip.vision.is_deepstack_layers", T_ARRAY, raw, 2, T_BOOL)
    with pytest.raises(ImageRefused, match="deepstack"):
        projector_marker(header, 1024, "p.gguf")


def test_an_image_id_is_never_a_token_and_depends_on_every_byte():
    assert image_id(b"a") < 0 and image_id(b"a") == image_id(b"a")
    assert image_id(b"a") != image_id(b"b")
    assert abs(image_id(b"a")) > 2**32 or abs(image_id(b"b")) > 2**32  # wider than a search could reach


def test_expanding_gives_each_marker_its_rows():
    tokens, boundary = expand_prompt([1, 9, 2, 9, 3], None, 9, [(-7, 3), (-8, 2)])
    assert tokens == [1, -7, -7, -7, 2, -8, -8, 3] and boundary is None


def test_the_boundary_moves_past_rows_before_it_and_no_others():
    # Boundary 4 sits after the first marker, before the second.
    tokens, boundary = expand_prompt([1, 9, 2, 3, 9, 4], 4, 9, [(-7, 3), (-8, 5)])
    assert boundary == 6
    assert tokens[:boundary] == [1, -7, -7, -7, 2, 3]


def encoded(identity=-7, grid=(2, 3), width=4, fill=1.0):
    """An image as M-RoPE places it, with ``grid`` rows by columns."""
    height, wide = grid
    rows = height * wide
    places = np.zeros((SECTIONS, rows), dtype=np.int32)
    places[1] = np.repeat(np.arange(height), wide)
    places[2] = np.tile(np.arange(wide), height)
    return EncodedImage(identity, max(height, wide), places, np.full((rows, width), fill, dtype=np.float32))


def test_text_alone_counts_one_position_a_row():
    rows = lay_out([5, 6, 7], [])
    assert rows.places[0].tolist() == [0, 1, 2]
    assert rows.places[:3].tolist() == [[0, 1, 2]] * 3 and rows.places[3].tolist() == [0, 0, 0]
    assert rows.position_at(3) == 3


def test_an_image_takes_its_grid_and_moves_on_by_its_span():
    image = encoded(grid=(2, 3))
    rows = lay_out([5, 6, *[-7] * 6, 8], [image])
    assert rows.places[0].tolist() == [0, 1, 2, 2, 2, 2, 2, 2, 5]  # the rows share one temporal position
    assert rows.places[1].tolist() == [0, 1, 2, 2, 2, 3, 3, 3, 5]  # height
    assert rows.places[2].tolist() == [0, 1, 2, 3, 4, 2, 3, 4, 5]  # width
    assert rows.places[3].tolist() == [0] * 9
    assert rows.position_at(8) == 5 and rows.position_at(9) == 6


def test_text_after_an_image_follows_its_span_and_not_its_rows():
    rows = lay_out([*[-7] * 6, 8, 9], [encoded(grid=(2, 3))])
    assert rows.places[0, 6:].tolist() == [3, 4]


def test_each_of_several_images_starts_where_the_last_one_ended():
    first, second = encoded(-7, (2, 2), fill=1.0), encoded(-8, (3, 1), fill=2.0)
    rows = lay_out([*[-7] * 4, 5, *[-8] * 3], [first, second])
    assert rows.places[0].tolist() == [0, 0, 0, 0, 2, 3, 3, 3]
    assert rows.vectors[rows.source[5:]].tolist() == [[2.0] * 4] * 3


def test_a_part_of_a_prompt_keeps_its_positions_and_its_vectors():
    image = encoded(grid=(2, 3))
    rows = lay_out([5, *[-7] * 6, 8], [image])
    part = rows.part(3, 8)
    assert part.places.shape == (SECTIONS, 5) and part.places.flags["C_CONTIGUOUS"]
    assert part.images.tolist() == [0, 1, 2, 3] and part.vectors.shape == (4, 4)
    assert part.after == rows.position_at(8)


@pytest.mark.parametrize(
    "tokens, images",
    [([5, -7, 6], [encoded(grid=(2, 3))]), ([*[-9] * 6], [encoded(grid=(2, 3))]), ([5, -7], [])],
)
def test_rows_that_do_not_match_the_images_sent_are_an_error(tokens, images):
    with pytest.raises(ValueError):
        lay_out(tokens, images)


def test_a_different_image_can_never_reuse_a_saved_state():
    red, blue = image_id(b"red"), image_id(b"blue")
    held_tokens = [1, *[red] * 4, 2]
    index = PrefixIndex()
    index.add(prefix_key(held_tokens), len(held_tokens), HISTORY)
    assert plan_prompt(index, [*held_tokens, 3], None, None).start == len(held_tokens)
    assert plan_prompt(index, [1, *[blue] * 4, 2, 3], None, None).start == 0
    assert prefix_key(held_tokens) != prefix_key([1, *[blue] * 4, 2])


def test_a_later_stage_gets_positions_with_the_rows():
    hidden = torch.ones(1, 3, 4, dtype=torch.bfloat16)
    places = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    got_hidden, got_places = decode_pass(encode_activation(hidden, places))
    assert torch.equal(got_hidden, hidden) and torch.equal(got_places, places)


def test_a_text_pass_carries_no_positions():
    hidden, places = decode_pass(encode_activation(torch.ones(1, 1, 4)))
    assert places is None and hidden.shape == (1, 1, 4)


def test_a_payload_with_anything_else_is_refused():
    from safetensors.torch import save as save_safetensors

    with pytest.raises(ValueError):
        decode_pass(save_safetensors({"weights": torch.ones(1)}))
    with pytest.raises(ValueError):
        decode_pass(save_safetensors({"hidden": torch.ones(1), "extra": torch.ones(1)}))


@pytest.mark.skipif(C.sizeof(C.c_void_p) != 8, reason="these offsets are for 64-bit builds")
def test_structures_sit_where_mtmd_h_puts_them():
    """Worked out by hand from mtmd.h and mtmd-helper.h at the pinned build."""
    assert C.sizeof(ContextParams) == 96
    assert [getattr(ContextParams, name).offset for name in ("device", "n_threads", "image_marker", "media_marker", "flash_attn_type", "warmup", "image_min_tokens", "cb_eval", "batch_max_tokens", "progress_callback")] == [8, 20, 24, 32, 40, 44, 48, 56, 72, 80]
    assert C.sizeof(InputText) == 24 and InputText.parse_special.offset == 17
    assert C.sizeof(VideoParams) == 32 and VideoParams.no_soundtrack.offset == 24
    assert C.sizeof(BitmapWrapper) == 16
    assert C.sizeof(DecoderPosition) == 16


class FakeHost:
    def __init__(self, lose=False):
        self.lose = lose
        self.closed = False
        self.calls = []
        self.process = SimpleNamespace(poll=lambda: None)

    def call(self, operation, payload=b"", **meta):
        self.calls.append(operation)
        if self.lose:
            raise StageLost("the stage process died")
        return {"rows": 2, "span": 1}, b""

    def close(self):
        self.closed = True


def encoder_with(host, replacement):
    made = ImageEncoder.__new__(ImageEncoder)
    made.host = host
    made.lock = __import__("threading").Lock()
    made._start = lambda: replacement
    return made


def test_an_image_that_stops_the_decoder_is_refused_and_the_process_comes_back():
    bad, fresh = FakeHost(lose=True), FakeHost()
    encoder = encoder_with(bad, fresh)
    with pytest.raises(LlamaError, match="could not be read"):
        encoder.measure(b"poison")
    assert encoder.host is fresh and bad.closed
    assert encoder.measure(b"fine") == {"rows": 2, "span": 1}


def test_a_process_that_died_on_its_own_is_not_blamed_on_the_next_image():
    gone, fresh = FakeHost(), FakeHost()
    gone.process = SimpleNamespace(poll=lambda: 1)
    encoder = encoder_with(gone, fresh)
    assert encoder.measure(b"fine") == {"rows": 2, "span": 1}
    assert encoder.host is fresh and not gone.calls


def fake_request(**changes):
    return InferRequest(session_id="s", epoch=1, input_ids=[1], **changes)


def merge_runtime():
    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.merge_passes, runtime.exact, runtime.exactness = True, False, None
    runtime.gguf = object()
    return runtime


def test_a_pass_with_image_rows_never_merges():
    hidden = torch.zeros(1, 1, 4)
    assert merge_runtime()._can_merge(hidden, fake_request())
    assert not merge_runtime()._can_merge(hidden, fake_request(next_position=9))


def test_the_entry_stage_puts_vectors_where_the_rows_are():
    rows = lay_out([5, *[-7] * 6, 8], [encoded(grid=(2, 3), fill=3.0)])
    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.gguf = SimpleNamespace(width=4, embed=lambda tokens: np.array([[float(token)] * 4 for token in tokens], dtype=np.float32))
    runtime._session = lambda request: None
    hidden = runtime._prepare_entry(fake_request().model_copy(update={"input_ids": [5, *[-7] * 6, 8]}), rows.part(0, 8))
    assert hidden.shape == (1, 8, 4)
    assert hidden[0, :, 0].tolist() == [5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 8.0]


class FakeStage:
    last = False

    def __init__(self):
        self.seen = []

    def forward(self, rows, position, slot, places=None, cuts=None, saves=None):
        self.seen.append((position, None if places is None else places.tolist()))
        return rows


def stage_runtime(stage):
    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.gguf = stage
    runtime.adapter = object()
    runtime.assignment = SimpleNamespace(start_layer=4)
    runtime.caches = {"s": worker_module.CacheEntry(epoch=1, slot=0, length=7)}
    runtime.inference_calls = runtime.tokens_processed = 0
    runtime.compute_ms_ema = runtime.row_ms_ema = None
    runtime.batch_ms = {}
    return runtime


def test_a_later_stage_follows_the_positions_it_is_sent():
    stage = FakeStage()
    runtime = stage_runtime(stage)
    places = np.arange(8, dtype=np.int32).reshape(4, 2)
    runtime._run_local(torch.zeros(1, 2, 4), fake_request(next_position=20), places)
    assert stage.seen == [(7, places.tolist())]
    assert runtime.caches["s"].length == 20, "an image's rows move a sequence on by their span"
    runtime._run_local(torch.zeros(1, 1, 4), fake_request())
    assert stage.seen[-1] == (20, None)
    assert runtime.caches["s"].length == 21


def test_images_stay_out_of_what_a_stage_passes_along(monkeypatch):
    sent = {}

    class Client:
        async def post(self, url, content, headers, timeout):
            sent.update(worker_module.decode_meta(headers[worker_module.META_HEADER]), body=content)
            return httpx.Response(200, json={"token_id": 3}, request=httpx.Request("POST", url))

    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.assignment, runtime.model_id, runtime.epoch = object(), "m", 1
    runtime.gguf, runtime.dtype_name, runtime.pool_token = object(), "bfloat16", None
    runtime.merge_passes = False
    runtime.activation_bytes_sent = 0
    runtime.downstream_ms_ema = None
    runtime.last_error = None
    runtime.client = Client()
    runtime._next_stage = lambda: SimpleNamespace(url="http://next:1")
    runtime._stages_after = lambda: 1
    runtime._record = lambda *args: None
    runtime.schedule_checkpoint = lambda request: None

    async def alone(hidden, request, places=None):
        return hidden

    runtime._step_alone = alone
    places = np.zeros((4, 2), dtype=np.int32)

    async def go():
        runtime.lock = asyncio.Lock()
        return await runtime.execute(torch.zeros(1, 2, 4), fake_request(images=["A" * 4096], next_position=5), places)

    assert asyncio.run(go()) == {"token_id": 3}
    assert "images" not in sent and sent["next_position"] == 5
    assert decode_pass(sent["body"])[1].shape == (4, 2)


def vision_target(tokenizer, marker="<|image_pad|>"):
    return SimpleNamespace(tokenizer=tokenizer, image_marker=marker, projector=SimpleNamespace(file="p.gguf"))


def test_a_template_with_no_marker_is_refused_at_load():
    with_marker = SimpleNamespace(chat_template="x <|image_pad|> y", encode=lambda text: [7])
    orchestrator_module.check_image_marker(vision_target(with_marker))
    without = SimpleNamespace(chat_template="no images here", encode=lambda text: [7])
    with pytest.raises(UnsupportedModel, match="no <|image_pad|>"):
        orchestrator_module.check_image_marker(vision_target(without))
    split = SimpleNamespace(chat_template="x <|image_pad|> y", encode=lambda text: [7, 8])
    with pytest.raises(UnsupportedModel):
        orchestrator_module.check_image_marker(vision_target(split))


def gguf_adapter():
    keys = {
        "general.architecture": KeyValue("general.architecture", T_STRING, b"", "qwen35"),
        "qwen35.embedding_length": KeyValue("qwen35.embedding_length", T_UINT32, b"", 1024),
    }
    made = GgufAdapter.__new__(GgufAdapter)
    made.header = GgufHeader(3, keys, {}, 0, 0)
    return made


def test_no_projector_named_means_no_vision():
    assert orchestrator_module.resolve_projector(LoadModelRequest(model_id="m.gguf"), gguf_adapter()) == (None, None)


def test_a_projector_needs_a_gguf_chat_model():
    with pytest.raises(UnsupportedModel, match="GGUF chat model"):
        orchestrator_module.resolve_projector(LoadModelRequest(model_id="m", mmproj="p.gguf"), object())
    with pytest.raises(UnsupportedModel, match="GGUF chat model"):
        orchestrator_module.resolve_projector(LoadModelRequest(model_id="m.gguf", mmproj="p.gguf", embedding=True), gguf_adapter())


def test_a_projector_is_found_by_name_like_any_gguf(monkeypatch):
    seen = {}

    def manifest(source, revision, holders, token):
        seen.update(source=source, holders=holders)
        return SimpleNamespace(revision="r1", config={"gguf": "skeleton", "gguf_file_size": 1234})

    monkeypatch.setattr(orchestrator_module, "build_manifest", manifest)
    monkeypatch.setattr(orchestrator_module.GgufHeader, "from_skeleton", classmethod(lambda cls, data: projector_header()))
    request = LoadModelRequest(model_id="m.gguf", mmproj="p.gguf")
    projector, marker = orchestrator_module.resolve_projector(request, gguf_adapter(), {"p.gguf": ["http://peer:1"]})
    assert marker == "<|image_pad|>"
    assert (projector.file, projector.file_size, projector.file_sources) == ("p.gguf", 1234, ["http://peer:1"])
    assert seen == {"source": "p.gguf", "holders": ["http://peer:1"]}


def test_a_file_that_is_not_a_projector_fails_the_load(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "build_manifest", lambda *args: SimpleNamespace(revision="r", config={"gguf": "s", "gguf_file_size": 1}))
    wrong = projector_header(**{"general.architecture": (T_STRING, "qwen35")})
    monkeypatch.setattr(orchestrator_module.GgufHeader, "from_skeleton", classmethod(lambda cls, data: wrong))
    with pytest.raises(UnsupportedModel, match="not a projector"):
        orchestrator_module.resolve_projector(LoadModelRequest(model_id="m.gguf", mmproj="m.gguf"), gguf_adapter())


class Words:
    """A tokenizer where the marker is one token."""

    stop_token_ids = {0}
    chat_template = tool_template = "<|image_pad|>"

    def encode(self, text):
        return [7 if word == "<|image_pad|>" else 1 for word in text.split()]

    def decode(self, ids):
        return "".join(chr(96 + token) for token in ids)

    def render_chat(self, messages, enable_thinking=None, add_generation_prompt=True):
        pieces = ["<|image_pad|>" if part.get("type") == "image" else part.get("text", "") for message in messages for part in (message["content"] if isinstance(message["content"], list) else [{"text": message["content"]}])]
        return " ".join(pieces)


@pytest.fixture
def control(monkeypatch):
    """A control plane running a model, with a fake entry worker that records what it gets."""
    from fastapi.testclient import TestClient

    runtime = orchestrator_module.runtime
    seen = []

    def entry(request):
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path == "/internal/images/measure":
            return httpx.Response(200, json={"images": [{"rows": 3, "span": 2} for _ in body["images"]]})
        if request.url.path == "/v1/generate":
            return httpx.Response(200, content=b'{"token_id": 2}\n{"token_id": 0}\n')
        return httpx.Response(200, json={"token_id": 0})

    monkeypatch.setattr(runtime, "model_id", "see.gguf")
    monkeypatch.setattr(runtime, "tokenizer", Words())
    monkeypatch.setattr(runtime, "route", [SimpleNamespace(url="http://entry:1", worker_id="w", start_layer=0, end_layer=2)])
    monkeypatch.setitem(runtime.workers, "w", SimpleNamespace(live=True, quarantined=False, health=None))
    monkeypatch.setattr(runtime, "epoch", 4)
    active = SimpleNamespace(
        embedding=False,
        projector=SimpleNamespace(file="p.gguf"),
        image_marker="<|image_pad|>",
        adapter=gguf_adapter(),
        profile=SimpleNamespace(context_tokens=24),
        speculate_tokens=0,
        lookup_guesses=0,
        request=None,
        manifest=SimpleNamespace(revision="r1"),
        exact=False,
    )
    monkeypatch.setattr(runtime, "active", active)
    monkeypatch.setattr(runtime, "client", httpx.AsyncClient(transport=httpx.MockTransport(entry)))
    return TestClient(orchestrator_module.app), seen, active


def ask_about(client, *parts):
    return client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": list(parts)}], "max_tokens": 4})


def test_a_model_with_no_projector_refuses_and_never_pretends(control, monkeypatch):
    client, seen, active = control
    monkeypatch.setattr(active, "projector", None)
    answer = ask_about(client, image_part(png()), {"type": "text", "text": "what is this"})
    assert answer.status_code == 400 and "--mmproj" in answer.json()["detail"]
    assert not seen, "nothing may reach a stage"


def test_a_remote_image_is_a_plain_error_at_the_door(control):
    client, seen, _ = control
    answer = ask_about(client, {"type": "image_url", "image_url": {"url": "http://10.0.0.1/a.png"}})
    assert answer.status_code == 400 and "data URL" in answer.json()["detail"] and not seen


def test_images_reach_the_entry_worker_with_their_rows_in_the_prompt(control):
    client, seen, _ = control
    data = png()
    answer = ask_about(client, image_part(data), {"type": "text", "text": "what"})
    assert answer.status_code == 200
    paths = [path for path, _ in seen]
    assert paths[:2] == ["/internal/images/measure", "/v1/generate"]
    generate = seen[1][1]
    assert generate["images"] == [base64.b64encode(data).decode()]
    assert generate["input_ids"] == [image_id(data)] * 3 + [1]
    assert answer.json()["usage"]["prompt_tokens"] == 4, "image rows count, as they do against the context"


def test_a_stray_marker_in_the_text_is_refused_rather_than_misplaced(control):
    client, seen, _ = control
    answer = ask_about(client, image_part(png()), {"type": "text", "text": "<|image_pad|>"})
    assert answer.status_code == 400 and "markers" in answer.json()["detail"]


def test_images_past_the_context_are_refused_before_any_stage_runs(control, monkeypatch):
    client, seen, active = control
    monkeypatch.setattr(active.profile, "context_tokens", 3)
    answer = ask_about(client, image_part(png()), {"type": "text", "text": "what"})
    assert answer.status_code == 400 and "4 tokens" in answer.json()["detail"]
    assert [path for path, _ in seen] == ["/internal/images/measure"]


def test_a_replay_carries_the_images_and_the_next_step_does_not(monkeypatch):
    runtime = orchestrator_module.Orchestrator()
    sent = []

    def entry(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"token_id": 5})

    runtime.route = [SimpleNamespace(url="http://entry:1", worker_id="w")]
    runtime.epoch = 1

    async def go():
        runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(entry))
        session = SessionState(tokens=[image_id(b"x")] * 3 + [4], epoch=1, needs_reset=True, images=["AAAA"])
        await runtime.next_token(session, 0.0, 1.0, 0, None)
        session.tokens.append(5)
        await runtime.next_token(session, 0.0, 1.0, 0, None)

    asyncio.run(go())
    assert sent[0]["images"] == ["AAAA"] and sent[0]["reset"] is True
    assert sent[1]["images"] == [] and sent[1]["input_ids"] == [5]


def test_a_projector_is_rebuilt_when_it_changes_and_only_on_the_entry_stage():
    from isharded.protocol import ConfigureRequest, DraftModel, StageAssignment

    def stage(start):
        return StageAssignment(worker_id="w", url="http://x:1", start_layer=start, end_layer=start + 4, placements=[], estimated_ms_per_token=1.0)

    projector = DraftModel(source="p.gguf", file="p.gguf", file_size=1, revision="r")
    signature = worker_module.WorkerRuntime._stage_signature

    def configure(assignment, **changes):
        return ConfigureRequest(model_id="m", revision="r", architecture={}, epoch=1, assignment=assignment, route=[assignment], **changes)

    assert signature(configure(stage(0))) != signature(configure(stage(0), projector=projector))
    assert signature(configure(stage(4))) == signature(configure(stage(4), projector=projector))


def test_the_chat_page_offers_images_only_when_the_model_can_see():
    text = (Path(orchestrator_module.__file__).parent / "chat.html").read_text(encoding="utf-8")
    assert 'id="attach" type="button"' in text and 'id="imageFile" type="file"' in text
    sync = text[text.index("function syncComposer"):text.index("function renderAttachments")]
    assert "$('attach').hidden = !sees" in sync and "chatSees()" in sync
    assert "image_url: { url }" in text, "images go as OpenAI parts"
    assert "turn.images" in text[text.index("function renderTurns"):text.index("function renderLastTurn")]


BUILD = os.environ.get("ISHARDED_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("ISHARDED_TEST_VISION_GGUF", "")
MMPROJ = os.environ.get("ISHARDED_TEST_VISION_MMPROJ", "")
needs_real_model = pytest.mark.skipif(
    not (BUILD and MODEL and MMPROJ and Path(BUILD).is_dir() and Path(MODEL).is_file() and Path(MMPROJ).is_file()),
    reason="needs a llama.cpp build, a vision GGUF, and its projector",
)
QUESTION = "What color is the large square in this image? Answer with one word."


def square_png(colour):
    """A large coloured square on white, whose answer can be checked."""
    size, edge = 256, 48
    rows = []
    for y in range(size):
        line = bytearray(b"\xff" * (size * 3))
        if edge <= y < size - edge:
            line[edge * 3 : (size - edge) * 3] = bytes(colour) * (size - 2 * edge)
        rows.append(b"\0" + bytes(line))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


class RealPipeline:
    """Stage processes and a projector process, joined the way workers join them."""

    def __init__(self, folder):
        from isharded.gguf import file_reader, plan_stage, read_file_header, write_stage
        from isharded.llama_host import StageHost
        from isharded.tokenizer import tokenizer_from_gguf

        self.header = read_file_header(MODEL)
        self.tokenizer = tokenizer_from_gguf(self.header)
        self.layers = self.header.layer_count
        table = self.header.tensors["token_embd.weight"]
        self.table = {"path": MODEL, "offset": self.header.data_start + table.offset, "length": table.nbytes, "ggml_type": table.ggml_type}
        self.hosts = {}

        def stage(name, start, end, last):
            path = folder / f"{name}.gguf"
            with path.open("wb") as handle:
                write_stage(self.header, plan_stage(self.header, start, end, last=last), handle, file_reader(MODEL))
            host = StageHost()
            self.hosts[name] = host
            host.call(
                "load", folder=BUILD, extra_paths=[], path=str(path), last=last, context_tokens=2048, gpu_layers=0, threads=4,
                cpu_tensors=[], table=self.table, map_file=False,
            )
            return path

        self.middle = (self.layers // 2) // 4 * 4
        whole = stage("whole", 0, self.layers, True)
        entry = stage("entry", 0, self.middle, False)
        stage("exit", self.middle, self.layers, True)
        self.encoder = ImageEncoder(BUILD, (), entry, Path(MMPROJ), self.header.hidden_size, 4, False)
        self.whole_path = whole

    def prompt(self, images):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": QUESTION}]
        text = self.tokenizer.render_chat([{"role": "user", "content": content}], False)
        tokens = self.tokenizer.encode(text)
        marker = self.tokenizer.encode("<|image_pad|>")[0]
        encoded = [self.encoder.encode(data) for data in images]
        sizes = [(item.identity, item.places.shape[1]) for item in encoded]
        expanded, _ = expand_prompt(tokens, None, marker, sizes)
        return expanded, lay_out(expanded, encoded)

    def rows(self, tokens, rows):
        ids = np.array(tokens, dtype=np.int64)
        hidden = np.empty((ids.size, self.header.hidden_size), dtype=np.float32)
        text = ids >= 0
        _, data = self.hosts["whole"].call("embed", tokens=ids[text].tolist())
        hidden[text] = np.frombuffer(data, dtype=np.float32).reshape(-1, self.header.hidden_size)
        part = rows.part(0, ids.size)
        hidden[part.images] = part.vectors
        return hidden, part

    def run_whole(self, hidden, position, slot_places):
        where = {} if slot_places is None else {"places": slot_places.tolist()}
        _, data = self.hosts["whole"].call("forward", hidden.tobytes(), position=position, **where)
        return np.frombuffer(data, dtype=np.float32)

    def run_split(self, hidden, position, entry_places, exit_places):
        where = {} if entry_places is None else {"places": entry_places.tolist()}
        _, data = self.hosts["entry"].call("forward", hidden.tobytes(), position=position, **where)
        where = {} if exit_places is None else {"places": exit_places.tolist()}
        _, data = self.hosts["exit"].call("forward", data, position=position, **where)
        return np.frombuffer(data, dtype=np.float32)

    def clear(self):
        for host in self.hosts.values():
            host.call("clear")

    def answer(self, step, tokens, rows, count=8):
        """Greedy text, through ``step(hidden, position, places)``."""
        self.clear()
        hidden, part = self.rows(tokens, rows)
        logits = step(hidden, 0, part.places)
        position = part.after
        produced = []
        for _ in range(count):
            token = int(np.argmax(logits))
            if token in self.tokenizer.stop_token_ids:
                break
            produced.append(token)
            _, data = self.hosts["whole"].call("embed", tokens=[token])
            logits = step(np.frombuffer(data, dtype=np.float32).reshape(1, -1), position, None)
            position += 1
        return self.tokenizer.decode(produced), logits

    def close(self):
        self.encoder.close()
        for host in self.hosts.values():
            host.close()


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    made = RealPipeline(tmp_path_factory.mktemp("vision"))
    yield made
    made.close()


RED, BLUE = (220, 20, 20), (20, 40, 220)


@needs_real_model
def test_a_stub_vocabulary_is_enough_for_the_projector(pipeline):
    measured = pipeline.encoder.measure(square_png(RED))
    image = pipeline.encoder.encode(square_png(RED))
    assert measured["rows"] == image.places.shape[1] == image.vectors.shape[0]
    assert measured["span"] < measured["rows"], "M-RoPE moves on by the grid's side, not its area"
    assert image.vectors.shape[1] == pipeline.header.hidden_size and np.isfinite(image.vectors).all()


@needs_real_model
def test_the_model_sees_what_is_in_the_image(pipeline):
    for colour, word in ((RED, "red"), (BLUE, "blue")):
        tokens, rows = pipeline.prompt([square_png(colour)])
        text, _ = pipeline.answer(lambda hidden, position, places: pipeline.run_whole(hidden, position, places), tokens, rows)
        assert word in text.lower()


@needs_real_model
def test_a_split_answers_as_one_stage_does(pipeline):
    tokens, rows = pipeline.prompt([square_png(RED)])
    alone, _ = pipeline.answer(lambda hidden, position, places: pipeline.run_whole(hidden, position, places), tokens, rows)
    split, _ = pipeline.answer(lambda hidden, position, places: pipeline.run_split(hidden, position, places, places), tokens, rows)
    assert split == alone


@needs_real_model
def test_a_later_stage_given_flat_positions_gets_something_else(pipeline):
    """What keeps the split test honest: positions on a later stage really do matter."""
    tokens, rows = pipeline.prompt([square_png(RED)])
    hidden, part = pipeline.rows(tokens, rows)
    pipeline.clear()
    right = pipeline.run_split(hidden, 0, part.places, part.places)
    pipeline.clear()
    flat = np.arange(part.places.shape[1], dtype=np.int32)
    flat_places = np.stack([flat, flat, flat, np.zeros_like(flat)])
    wrong = pipeline.run_split(hidden, 0, part.places, flat_places)
    assert np.abs(right - wrong).max() > 0.1


@needs_real_model
def test_a_bad_image_is_refused_and_the_next_one_still_works(pipeline):
    with pytest.raises(LlamaError):
        pipeline.encoder.measure(b"\x89PNG\r\n\x1a\n" + b"not really a png" * 8)
    assert pipeline.encoder.measure(square_png(RED))["rows"] > 0
