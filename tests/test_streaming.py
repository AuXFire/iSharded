import asyncio
import json
from pathlib import Path

from fastapi import HTTPException
import httpx
import pytest
from jinja2.exceptions import SecurityError, TemplateError
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace

from shardmesh.models import GgufAdapter
from shardmesh.orchestrator import Orchestrator
from shardmesh.protocol import StageAssignment
from shardmesh.tokenizer import ChatTokenizer, IncrementalText, _parse_tokenizer, load_tokenizer

STOP_TOKEN = 99


class ScriptedTokenizer:
    """Decodes token ids straight to the characters of a script."""

    stop_token_ids = {STOP_TOKEN}

    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces

    def encode(self, text: str) -> list[int]:
        return [1, 2]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[token] for token in ids)


def generate(pieces: dict[int, str], tokens: list[int], stop=None, max_tokens: int = 20) -> tuple[list[dict], list[str]]:
    async def scenario() -> tuple[list[dict], list[str]]:
        orchestrator = Orchestrator()
        orchestrator.tokenizer = ScriptedTokenizer(pieces)
        orchestrator.model_id = "scripted"
        remaining = iter(tokens)
        cleared: list[str] = []

        async def next_token(*args) -> int:
            return next(remaining)

        async def clear_session(session_id: str) -> None:
            cleared.append(session_id)

        orchestrator.next_token = next_token
        orchestrator.clear_session = clear_session
        events = [item async for item in orchestrator.stream_tokens("prompt", max_tokens, 0.0, 1.0, 0, None, stop)]
        await asyncio.gather(*orchestrator.background)
        await orchestrator.close()
        return events, cleared

    return asyncio.run(scenario())


def text_of(events: list[dict]) -> list[str]:
    return [item["text"] for item in events if "text" in item]


def test_text_streams_as_it_is_generated() -> None:
    events, cleared = generate({1: "Hel", 2: "lo", 3: "!"}, [1, 2, 3, STOP_TOKEN])
    assert text_of(events) == ["Hel", "lo", "!"]
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["completion_tokens"] == 3
    assert len(cleared) == 1


def test_stream_ends_at_the_token_limit() -> None:
    events, _ = generate({1: "a"}, [1, 1, 1, 1], max_tokens=3)
    assert "".join(text_of(events)) == "aaa"
    assert events[-1]["finish_reason"] == "length"


def test_a_stop_string_split_across_tokens_never_reaches_the_client() -> None:
    events, _ = generate({1: "Hello ", 2: "EN", 3: "D", 4: " tail"}, [1, 2, 3, 4], stop="END")
    assert "".join(text_of(events)) == "Hello "
    assert events[-1]["finish_reason"] == "stop"


def test_text_that_only_looked_like_a_stop_string_still_arrives() -> None:
    events, _ = generate({1: "EN", 2: "gland"}, [1, 2, STOP_TOKEN], stop="END")
    assert "".join(text_of(events)) == "ENgland"


def test_a_character_split_across_tokens_is_held_until_complete() -> None:
    events, _ = generate({1: "a", 2: "�", 3: "b"}, [1, 2, STOP_TOKEN])
    assert all("�" not in piece for piece in text_of(events)[:-1])


def test_a_client_leaving_mid_stream_still_clears_worker_state() -> None:
    async def scenario() -> list[str]:
        orchestrator = Orchestrator()
        orchestrator.tokenizer = ScriptedTokenizer({1: "a"})
        orchestrator.model_id = "scripted"
        cleared: list[str] = []

        async def next_token(*args) -> int:
            return 1

        async def clear_session(session_id: str) -> None:
            cleared.append(session_id)

        orchestrator.next_token = next_token
        orchestrator.clear_session = clear_session
        stream = orchestrator.stream_tokens("prompt", 100, 0.0, 1.0, 0, None, None)
        await stream.__anext__()
        await stream.aclose()
        await asyncio.gather(*orchestrator.background)
        assert not orchestrator.generation_lock.locked()
        await orchestrator.close()
        return cleared

    assert len(asyncio.run(scenario())) == 1


def run_with_worker_stream(handler, pieces, tokens_after=(), max_tokens=20):
    """Drive stream_tokens against a fake entry worker."""

    async def scenario():
        orchestrator = Orchestrator()
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        orchestrator.tokenizer = ScriptedTokenizer(pieces)
        orchestrator.model_id = "scripted"
        orchestrator.route = [StageAssignment(worker_id="w", url="http://worker", start_layer=0, end_layer=1)]
        remaining = iter(tokens_after)
        replays = []

        async def next_token(session, *args):
            replays.append((session.needs_reset, list(session.tokens)))
            return next(remaining)

        async def clear_session(session_id):
            return None

        orchestrator.next_token = next_token
        orchestrator.clear_session = clear_session
        events = [item async for item in orchestrator.stream_tokens("prompt", max_tokens, 0.0, 1.0, 0, None, None)]
        await asyncio.gather(*orchestrator.background)
        await orchestrator.close()
        return events, replays

    return asyncio.run(scenario())


def stream_of(*lines: str) -> httpx.Response:
    return httpx.Response(200, content="".join(line + "\n" for line in lines).encode())


def test_the_entry_worker_runs_the_token_loop() -> None:
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        assert request.url.path == "/v1/generate"
        return stream_of('{"token_id": 1}', '{"token_id": 2}', f'{{"token_id": {STOP_TOKEN}}}')

    events, replays = run_with_worker_stream(handler, {1: "Hi", 2: "!"})
    assert text_of(events) == ["Hi", "!"]
    assert events[-1]["finish_reason"] == "stop"
    assert not replays
    assert seen["input_ids"] == [1, 2] and seen["reset"] is True
    assert seen["stop_token_ids"] == [STOP_TOKEN]


def test_a_stream_that_fails_part_way_recovers_with_a_full_replay() -> None:
    def handler(request):
        return stream_of('{"token_id": 1}', '{"error": "stage lost"}')

    events, replays = run_with_worker_stream(handler, {1: "a", 2: "b"}, tokens_after=[2, STOP_TOKEN])
    assert "".join(text_of(events)) == "ab"
    assert replays[0] == (True, [1, 2, 1]), "the replay must include the token already streamed"


def test_a_refused_stream_goes_straight_to_the_old_path() -> None:
    def handler(request):
        return httpx.Response(409, json={"detail": "busy"})

    events, replays = run_with_worker_stream(handler, {1: "a"}, tokens_after=[1, STOP_TOKEN])
    assert "".join(text_of(events)) == "a"
    assert replays[0][0] is True


def test_a_connection_lost_mid_stream_also_recovers() -> None:
    def handler(request):
        raise httpx.ConnectError("gone")

    events, replays = run_with_worker_stream(handler, {1: "a"}, tokens_after=[1, STOP_TOKEN])
    assert "".join(text_of(events)) == "a"


def test_the_token_limit_applies_across_a_recovery() -> None:
    def handler(request):
        return stream_of('{"token_id": 1}', '{"token_id": 1}', '{"error": "lost"}')

    events, replays = run_with_worker_stream(handler, {1: "a"}, tokens_after=[1, 1, 1, 1], max_tokens=3)
    assert "".join(text_of(events)) == "aaa"
    assert len(replays) == 1


def gguf_orchestrator(context_tokens: int, speculate_tokens: int = 0) -> Orchestrator:
    class Target:
        adapter = GgufAdapter.__new__(GgufAdapter)
        profile = type("Profile", (), {"context_tokens": context_tokens})()

    Target.speculate_tokens = speculate_tokens
    orchestrator = Orchestrator()
    orchestrator.active = Target()
    return orchestrator


def test_a_prompt_too_long_for_the_context_is_refused_not_blamed_on_a_worker() -> None:
    orchestrator = gguf_orchestrator(100)
    with pytest.raises(HTTPException) as caught:
        orchestrator._fit_context(100, 10)
    assert caught.value.status_code == 400
    assert "100 tokens" in caught.value.detail
    asyncio.run(orchestrator.close())


def test_generation_stops_at_the_context_with_room_for_guesses() -> None:
    orchestrator = gguf_orchestrator(100, speculate_tokens=6)
    assert orchestrator._fit_context(50, 10) == 10
    assert orchestrator._fit_context(90, 10) == 3
    asyncio.run(orchestrator.close())


def test_other_engines_have_no_context_limit_here() -> None:
    orchestrator = Orchestrator()
    assert orchestrator._fit_context(10**6, 10) == 10
    asyncio.run(orchestrator.close())


def test_incremental_text_matches_a_full_decode() -> None:
    class Bytes:
        def decode(self, ids):
            return bytes(ids).decode("utf-8", "replace")

    tokens = list("héllo 🙂 wörld".encode())
    decoder = IncrementalText(Bytes())
    text = ""
    for token in tokens:
        text = decoder.add(token)
    assert text == "héllo 🙂 wörld"


def word_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel({"hello": 0, "world": 1, "<unk>": 2}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def chat_tokenizer(template: str | None) -> ChatTokenizer:
    return ChatTokenizer(word_tokenizer(), template, {0}, {"eos_token": "<end>"})


def test_chat_template_receives_messages_and_the_thinking_switch() -> None:
    template = "{% for m in messages %}{{ m.role }}={{ m.content }};{% endfor %}think={{ enable_thinking }}"
    rendered = chat_tokenizer(template).render_chat([{"role": "user", "content": "hi"}], False)
    assert rendered == "user=hi;think=False"


def test_models_without_a_template_get_a_plain_transcript() -> None:
    rendered = chat_tokenizer(None).render_chat([{"role": "user", "content": "hi"}])
    assert rendered == "User: hi\nAssistant:"


def test_chat_templates_cannot_reach_python_internals() -> None:
    template = "{{ messages.__class__.__mro__[1].__subclasses__() }}"
    with pytest.raises((SecurityError, TemplateError)):
        chat_tokenizer(template).render_chat([{"role": "user", "content": "hi"}])


def test_tokenizer_files_with_paired_merges_load_on_any_tokenizers_release() -> None:
    tokenizer = Tokenizer(BPE({"a": 0, "b": 1, "ab": 2}, [("a", "b")]))
    data = json.loads(tokenizer.to_str())
    data["model"]["merges"] = [["a", "b"]]

    parsed = _parse_tokenizer(json.dumps(data))

    assert parsed.encode("ab").ids == [2]


def test_stop_tokens_merge_every_place_a_checkpoint_declares_them(tmp_path: Path) -> None:
    word_tokenizer().save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": {"content": "world"}}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 8]}))
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 9}))
    (tmp_path / "chat_template.jinja").write_text("separate template")

    loaded = load_tokenizer(str(tmp_path))

    assert loaded.stop_token_ids == {1, 7, 8, 9}
    assert loaded.chat_template == "separate template"
