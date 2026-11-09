"""Fields an API defines and iSharded cannot honour are refused by name, not ignored.

Each API shape is checked through its real route, under the scripted model of
tests/test_tools.py, and with the real client libraries where they are installed (see
tests/test_compat.py for putting recent ones on the path).
"""

from __future__ import annotations

import pytest

from isharded import parameters
from isharded.parameters import CHAT_LIMITS, OLLAMA_OPTION_LIMITS, refusal
from test_compat import served  # noqa: F401
from test_tools import QWEN3, mesh  # noqa: F401

SAY = [{"role": "user", "content": "hi"}]


@pytest.fixture
def plain(mesh):  # noqa: F811
    mesh.use(QWEN3, "Hello there")
    return mesh


def chat(mesh, **body):
    return mesh.client.post("/v1/chat/completions", json={"model": "scripted", "messages": SAY, **body})


def completion(mesh, **body):
    return mesh.client.post("/v1/completions", json={"model": "scripted", "prompt": "hi", **body})


def claude(mesh, **body):
    return mesh.client.post("/v1/messages", json={"model": "scripted", "max_tokens": 16, "messages": SAY, **body})


def responses(mesh, **body):
    return mesh.client.post("/v1/responses", json={"model": "scripted", "input": "hi", **body})


def ollama(mesh, **body):
    return mesh.client.post("/api/chat", json={"model": "scripted", "messages": SAY, "stream": False, **body})


# --- the rule itself


def test_json_equality_keeps_false_and_zero_apart():
    assert parameters.equal(0, 0.0) and parameters.equal(1, 1.0)
    assert not parameters.equal(False, 0) and not parameters.equal(0, False) and not parameters.equal(True, 1)
    assert CHAT_LIMITS["logprobs"].accepts(False) and not CHAT_LIMITS["logprobs"].accepts(0)
    assert CHAT_LIMITS["n"].accepts(1) and CHAT_LIMITS["n"].accepts(None) and not CHAT_LIMITS["n"].accepts(True)


def test_the_first_field_sent_is_the_one_named():
    found = refusal({"metadata": {"a": 1}, "presence_penalty": 1, "n": 3}, CHAT_LIMITS)
    assert found is not None and found.startswith("not supported: presence_penalty (")


def test_unknown_names_are_refused_unless_the_api_ignores_them():
    assert refusal({"foo": 1}, CHAT_LIMITS) == "not supported: foo (this API has no such field, or iSharded does not know it)"
    assert refusal({"foo": 1}, OLLAMA_OPTION_LIMITS, unknown=False) is None
    assert refusal({"model": "x"}, {}, known={"model"}) is None


# --- OpenAI chat completions


@pytest.mark.parametrize(
    "field",
    [
        {"n": 2},
        {"frequency_penalty": 0.5},
        {"presence_penalty": -1},
        {"logit_bias": {"50256": -100}},
        {"logprobs": True},
        {"top_logprobs": 3},
        {"response_format": {"type": "json_object"}},
        {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}},
        {"modalities": ["text", "audio"]},
        {"audio": {"voice": "alloy", "format": "wav"}},
        {"prediction": {"type": "content", "content": "x"}},
        {"reasoning_effort": "high"},
        {"verbosity": "low"},
        {"web_search_options": {}},
        {"functions": [{"name": "f", "parameters": {}}]},
        {"function_call": {"name": "f"}},
        {"stream_options": {"include_usage": True, "chunk_every": 2}},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"min_p": 0.05},
        {"repetition_penalty": 1.2},
        {"best_of": 2},
        {"frequency_penaltyy": 0},
    ],
)
def test_chat_refuses_what_it_would_not_honour_and_names_it(plain, field):
    answer = chat(plain, **field)
    assert answer.status_code == 400, answer.text
    name = next(iter(field))
    assert answer.json()["detail"].startswith(f"not supported: {name} (")
    # OpenAI's libraries read this object, and show only the status without it.
    assert answer.json()["error"] == {"message": answer.json()["detail"], "type": "invalid_request_error", "param": name, "code": "unsupported_parameter"}
    assert not plain.seen, "nothing ran"


def test_chat_takes_defaults_and_fields_that_change_nothing(plain):
    defaults = {
        "n": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.0,
        "logit_bias": {},
        "logprobs": False,
        "top_logprobs": 0,
        "response_format": {"type": "text"},
        "modalities": ["text"],
        "audio": None,
        "verbosity": "medium",
        "stream_options": None,
        "function_call": "auto",
        "metadata": {"run": "7"},
        "user": "someone",
        "store": True,
        "service_tier": "auto",
        "safety_identifier": "abc",
        "prompt_cache_key": "k",
        "prompt_cache_retention": "24h",
    }
    answer = chat(plain, **defaults)
    assert answer.status_code == 200, answer.text
    assert answer.json()["choices"][0]["message"]["content"] == "Hello there"


def test_a_stream_may_ask_for_usage_which_it_gets_anyway(plain):
    answer = chat(plain, stream=True, stream_options={"include_usage": True})
    assert answer.status_code == 200 and '"usage"' in answer.text


# --- OpenAI completions and embeddings


@pytest.mark.parametrize("field", [{"n": 3}, {"best_of": 2}, {"echo": True}, {"logprobs": 5}, {"suffix": "end"}, {"presence_penalty": 0.4}, {"messages": SAY}])
def test_completions_refuse_what_they_would_not_honour(plain, field):
    answer = completion(plain, **field)
    assert answer.status_code == 400 and answer.json()["detail"].startswith(f"not supported: {next(iter(field))} (")


def test_completions_take_their_defaults(plain):
    answer = completion(plain, n=1, best_of=1, echo=False, logprobs=None, suffix=None, frequency_penalty=0, user="u")
    assert answer.status_code == 200 and answer.json()["choices"][0]["text"] == "Hello there"


def test_embeddings_refuse_a_field_they_do_not_have(plain):
    answer = plain.client.post("/v1/embeddings", json={"model": "scripted", "input": "hi", "truncate": True})
    assert answer.status_code == 400 and answer.json()["detail"].startswith("not supported: truncate (")


# --- Claude Messages


@pytest.mark.parametrize(
    "field",
    [
        {"output_format": {"type": "json_schema", "schema": {}}},
        {"output_config": {"format": {"type": "json_schema", "schema": {}}}},
        {"container": "container_1"},
        {"mcp_servers": [{"type": "url", "url": "https://example.com/mcp", "name": "x"}]},
        {"top_logprobs": 2},
    ],
)
def test_claude_refuses_what_it_would_not_honour(plain, field):
    answer = claude(plain, **field)
    assert answer.status_code == 400
    error = answer.json()["error"]
    assert error["type"] == "invalid_request_error" and error["message"].startswith(f"not supported: {next(iter(field))} (")


def test_claude_takes_fields_that_change_nothing(plain):
    extra = {"metadata": {"user_id": "u"}, "service_tier": "auto", "context_management": {"edits": []}, "output_config": {"effort": "high"}, "mcp_servers": []}
    answer = claude(plain, **extra)
    assert answer.status_code == 200 and answer.json()["content"][0]["text"] == "Hello there"


def test_counting_tokens_refuses_the_same(plain):
    answer = plain.client.post("/v1/messages/count_tokens", json={"model": "scripted", "messages": SAY, "container": "c"})
    assert answer.status_code == 400 and "not supported: container" in answer.json()["error"]["message"]


# --- OpenAI Responses


@pytest.mark.parametrize(
    "field, named",
    [
        ({"truncation": "auto"}, "truncation"),
        ({"include": ["message.output_text.logprobs"]}, "include"),
        ({"top_logprobs": 4}, "top_logprobs"),
        ({"max_tool_calls": 3}, "max_tool_calls"),
        ({"prompt": {"id": "pmpt_1"}}, "prompt"),
        ({"reasoning": {"effort": "low", "budget": 3}}, "reasoning"),
        ({"text": {"verbosity": "high"}}, "verbosity"),
        ({"text": {"format": {"type": "text"}, "stops": ["x"]}}, "stops"),
        ({"n": 2}, "n"),
    ],
)
def test_responses_refuse_what_they_would_not_honour(plain, field, named):
    answer = responses(plain, **field)
    assert answer.status_code == 400
    assert answer.json()["error"]["message"].startswith(f"not supported: {named} (")


def test_responses_take_what_codex_sends(plain):
    sent = {
        "instructions": "Be brief.",
        "store": False,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "session",
        "parallel_tool_calls": False,
        "text": {"verbosity": "medium"},
        "truncation": "disabled",
        "metadata": {},
        "service_tier": "auto",
    }
    answer = responses(plain, **sent)
    assert answer.status_code == 200, answer.text
    assert answer.json()["output"][0]["content"][0]["text"] == "Hello there"


# --- Ollama


@pytest.mark.parametrize("options", [{"repeat_penalty": 1.3}, {"presence_penalty": 0.5}, {"frequency_penalty": 0.2}, {"min_p": 0.1}, {"typical_p": 0.7}, {"tfs_z": 0.9}, {"mirostat": 2}])
def test_ollama_refuses_sampling_it_would_not_apply(plain, options):
    answer = ollama(plain, options=options)
    assert answer.status_code == 400 and answer.json()["error"].startswith(f"not supported: {next(iter(options))} (")


def test_ollama_refuses_the_same_on_generate(plain):
    answer = plain.client.post("/api/generate", json={"model": "scripted", "prompt": "hi", "stream": False, "options": {"repeat_penalty": 1.5}})
    assert answer.status_code == 400 and "not supported: repeat_penalty" in answer.json()["error"]


def test_ollama_takes_its_own_defaults_and_ignores_what_it_does_not_know(plain):
    options = {"repeat_penalty": 1.1, "repeat_last_n": 64, "min_p": 0.0, "typical_p": 1.0, "mirostat": 0, "mirostat_tau": 5.0, "num_ctx": 4096, "num_gpu": 99, "made_up": 3}
    answer = ollama(plain, options=options, keep_alive="5m", something_new=True, logprobs=False)
    assert answer.status_code == 200 and answer.json()["message"]["content"] == "Hello there"


def test_ollama_refuses_logprobs_and_the_old_context(plain):
    assert ollama(plain, logprobs=True).status_code == 400
    answer = plain.client.post("/api/generate", json={"model": "scripted", "prompt": "hi", "stream": False, "context": [1, 2, 3]})
    assert answer.status_code == 400 and answer.json()["error"].startswith("not supported: context (")


# --- real clients, which send their own defaults


def test_the_openai_client_with_its_defaults_and_a_refusal(plain, served):  # noqa: F811
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=served + "/v1", api_key="unused", max_retries=0)
    reply = client.chat.completions.create(model="scripted", messages=SAY, n=1, frequency_penalty=0, presence_penalty=0, logprobs=False, user="u")
    assert reply.choices[0].message.content == "Hello there"
    plain.use(QWEN3, "Hello there")
    chunks = list(client.chat.completions.create(model="scripted", messages=SAY, stream=True, stream_options={"include_usage": True}))
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "Hello there"
    with pytest.raises(openai.BadRequestError, match="not supported: n"):
        client.chat.completions.create(model="scripted", messages=SAY, n=2)


def test_the_anthropic_client_with_its_defaults(plain, served):  # noqa: F811
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(base_url=served, api_key="unused", max_retries=0)
    reply = client.messages.create(model="scripted", max_tokens=16, messages=SAY, metadata={"user_id": "u"})
    assert reply.content[0].text == "Hello there"
    with pytest.raises(anthropic.BadRequestError, match="not supported: container"):
        client.messages.create(model="scripted", max_tokens=16, messages=SAY, extra_body={"container": "c"})


def test_the_ollama_client_with_options(plain):
    ollama_library = pytest.importorskip("ollama")
    client = ollama_library.Client(host="http://testserver")
    client._client = plain.client
    assert client.chat(model="scripted", messages=SAY, options={"num_ctx": 2048, "repeat_penalty": 1.1}).message.content == "Hello there"
    with pytest.raises(ollama_library.ResponseError, match="not supported: presence_penalty"):
        client.chat(model="scripted", messages=SAY, options={"presence_penalty": 1.0})
