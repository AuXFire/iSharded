"""Voice: engines are optional, text never runs, and a persona changes only how it sounds.

The system voice is really used where the machine has one. Engines that cannot be
installed here are stand-in programs, run as real child processes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
import httpx
import pytest
import uvicorn

from shardmesh.assistant.app import Assistant
from shardmesh.assistant.cli import add_arguments
from shardmesh.assistant.contracts import Capability, Decision, Effect, Trust, Verdict
from shardmesh.assistant.voice import (
    INSTALL_SPEECH,
    INSTALL_WHISPER,
    SPEAK,
    TRANSCRIBE,
    Voice,
    VoiceError,
    VoiceRefused,
    VoiceUnavailable,
    _user,
    routes,
)

SETTINGS = (
    "SHARDMESH_ASSISTANT_WHISPER_MODEL",
    "SHARDMESH_ASSISTANT_WHISPER_CPP",
    "SHARDMESH_ASSISTANT_WHISPER_CPP_MODEL",
    "SHARDMESH_ASSISTANT_PIPER",
    "SHARDMESH_ASSISTANT_PIPER_MODEL",
)
HAS_SYSTEM_VOICE = bool(shutil.which("powershell") if os.name == "nt" else shutil.which("say") or shutil.which("espeak-ng"))
needs_system_voice = pytest.mark.skipif(not HAS_SYSTEM_VOICE, reason="this machine has no system voice")


class FakeBroker:
    """contracts.Broker: deny by default."""

    def __init__(self) -> None:
        self.declared: dict[str, Capability] = {}
        self.handlers: dict[str, Any] = {}
        self.grants: set[tuple[str, str]] = set()
        self.ask: set[str] = set()
        self.calls: list[Any] = []

    def register(self, capability: Capability, handler: Any) -> None:
        self.declared[capability.name] = capability
        self.handlers[capability.name] = handler

    def capabilities(self) -> list[Capability]:
        return list(self.declared.values())

    def check(self, call: Any, context: Any) -> Decision:
        if call.capability in self.ask:
            return Decision(Verdict.ASK, "a person must approve", approval_id="approval-3")
        if (context.user, call.capability) not in self.grants:
            return Decision(Verdict.DENY, "no grant")
        return Decision(Verdict.ALLOW, "granted")

    def invoke(self, call: Any, context: Any) -> Any:
        self.calls.append((call, context))
        return self.handlers[call.capability](call.arguments, context)


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def assistant(tmp_path):
    with Assistant(tmp_path / "home") as built:
        built.broker = None
        yield built


def stand_in(folder: Path, name: str, body: str) -> Path:
    """A real program standing in for an engine that cannot be installed here."""
    script = folder / f"{name}.py"
    script.write_text(body, encoding="utf-8")
    if os.name == "nt":
        program = folder / f"{name}.cmd"
        program.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        program = folder / name
        program.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        program.chmod(0o755)
    return program


PIPER = """
import json, sys
from pathlib import Path
arguments = sys.argv[1:]
text = sys.stdin.buffer.read().decode()
Path(arguments[arguments.index("--output_file") + 1]).write_bytes(b"RIFF" + json.dumps({"text": text, "model": arguments[arguments.index("--model") + 1]}).encode())
"""
WHISPER_CPP = """
import sys
arguments = sys.argv[1:]
print("  heard from", arguments[arguments.index("-f") + 1].rsplit(".", 1)[-1], "with", arguments[arguments.index("-m") + 1], " ")
"""


def no_whisper(monkeypatch: Any) -> None:
    monkeypatch.setattr(Voice, "_faster_whisper", staticmethod(lambda: None))


def fake_whisper(monkeypatch: Any, recorded: list[Any], fails: bool = False) -> None:
    class WhisperModel:
        def __init__(self, name: str, **options: Any) -> None:
            if fails:
                raise OSError("no such model")
            recorded.append((name, options))

        def transcribe(self, path: str) -> tuple[Any, Any]:
            return iter([SimpleNamespace(text=" hello "), SimpleNamespace(text=" there ")]), None

    monkeypatch.setattr(Voice, "_faster_whisper", staticmethod(lambda: SimpleNamespace(WhisperModel=WhisperModel)))


@pytest.fixture
def audio(tmp_path) -> Path:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF....WAVE")
    return path


# Transcribing


def test_faster_whisper_is_used_when_installed(assistant, monkeypatch, audio):
    recorded: list[Any] = []
    fake_whisper(monkeypatch, recorded)
    assert assistant.voice.transcribe(audio) == "hello there"
    assert recorded[0][0] == "base"


def test_nothing_is_downloaded_and_the_gpu_is_left_alone(assistant, monkeypatch, audio):
    recorded: list[Any] = []
    fake_whisper(monkeypatch, recorded)
    assistant.voice.transcribe(audio)
    assert recorded[0][1]["local_files_only"] is True
    assert recorded[0][1]["device"] == "cpu"


def test_the_model_is_loaded_once(assistant, monkeypatch, audio):
    recorded: list[Any] = []
    fake_whisper(monkeypatch, recorded)
    assistant.voice.transcribe(audio)
    assistant.voice.transcribe(audio)
    assert len(recorded) == 1


def test_the_model_setting_is_used(assistant, monkeypatch, audio):
    recorded: list[Any] = []
    fake_whisper(monkeypatch, recorded)
    monkeypatch.setenv("SHARDMESH_ASSISTANT_WHISPER_MODEL", "/models/small")
    assistant.voice.transcribe(audio)
    assert recorded[0][0] == "/models/small"


def test_the_real_library_with_no_model_says_so_and_touches_no_network(assistant, monkeypatch, audio):
    """This is faster-whisper itself, if it is installed here."""
    pytest.importorskip("faster_whisper")
    monkeypatch.setenv("SHARDMESH_ASSISTANT_WHISPER_MODEL", "no-such-model-here")
    connections: list[Any] = []
    real_connect = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", lambda self, address: connections.append(address) or real_connect(self, address))
    with pytest.raises(VoiceUnavailable, match="nothing is downloaded"):
        assistant.voice.transcribe(audio)
    assert connections == []


def test_whisper_cpp_is_the_fallback(assistant, monkeypatch, audio, tmp_path):
    no_whisper(monkeypatch)
    monkeypatch.setenv("SHARDMESH_ASSISTANT_WHISPER_CPP", str(stand_in(tmp_path, "whisper-cli", WHISPER_CPP)))
    monkeypatch.setenv("SHARDMESH_ASSISTANT_WHISPER_CPP_MODEL", "ggml-base.bin")
    assert assistant.voice.transcribe(audio) == "heard from wav with ggml-base.bin"


def test_whisper_cpp_covers_a_faster_whisper_with_no_model(assistant, monkeypatch, audio, tmp_path):
    fake_whisper(monkeypatch, [], fails=True)
    monkeypatch.setenv("SHARDMESH_ASSISTANT_WHISPER_CPP", str(stand_in(tmp_path, "whisper-cli", WHISPER_CPP)))
    monkeypatch.setenv("SHARDMESH_ASSISTANT_WHISPER_CPP_MODEL", "m.bin")
    assert "heard" in assistant.voice.transcribe(audio)


def test_whisper_cpp_needs_a_model(assistant, monkeypatch, audio, tmp_path):
    no_whisper(monkeypatch)
    monkeypatch.setenv("SHARDMESH_ASSISTANT_WHISPER_CPP", str(stand_in(tmp_path, "whisper-cli", WHISPER_CPP)))
    with pytest.raises(VoiceUnavailable, match="SHARDMESH_ASSISTANT_WHISPER_CPP_MODEL"):
        assistant.voice.transcribe(audio)


def test_with_no_engine_it_says_which_package_to_install(assistant, monkeypatch, audio):
    no_whisper(monkeypatch)
    with pytest.raises(VoiceUnavailable) as caught:
        assistant.voice.transcribe(audio)
    assert "pip install faster-whisper" in str(caught.value)
    assert "whisper.cpp" in str(caught.value)


def test_a_missing_file_says_so(assistant, tmp_path):
    with pytest.raises(VoiceError, match="no audio file"):
        assistant.voice.transcribe(tmp_path / "missing.wav")


def test_a_failing_engine_says_so(assistant, monkeypatch, audio, tmp_path):
    no_whisper(monkeypatch)
    monkeypatch.setenv("SHARDMESH_ASSISTANT_WHISPER_CPP", str(stand_in(tmp_path, "whisper-cli", "import sys\nsys.exit('bad audio')\n")))
    monkeypatch.setenv("SHARDMESH_ASSISTANT_WHISPER_CPP_MODEL", "m.bin")
    with pytest.raises(VoiceError, match="bad audio"):
        assistant.voice.transcribe(audio)


# Speaking


@needs_system_voice
def test_the_system_voice_really_speaks(assistant):
    spoken = assistant.voice.speak("This is a test of the system voice.")
    data = Path(spoken["path"]).read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert len(data) > 5000  # real audio, not an empty header


@needs_system_voice
def test_what_is_said_never_runs(assistant, tmp_path):
    marker = tmp_path / "ran.txt"
    for attack in (
        f"'; New-Item -Path '{marker}'; '",
        f'"; New-Item -Path "{marker}"; "',
        f"$(New-Item -Path '{marker}')",
        f"`n New-Item -Path '{marker}'",
        f"& touch {marker}",
        f"; touch {marker}",
        f"$(touch {marker})",
        "-w /tmp/elsewhere.wav",
    ):
        assert Path(assistant.voice.speak(attack)["path"]).stat().st_size > 0
    assert not marker.exists()


@needs_system_voice
def test_text_outside_ascii_is_spoken(assistant):
    assert Path(assistant.voice.speak("Grüße, 世界")["path"]).stat().st_size > 0


@needs_system_voice
def test_the_text_file_is_cleaned_up(assistant):
    assistant.voice.speak("Tidy.")
    assert list(assistant.voice.folder.glob("*.txt")) == []


def test_piper_comes_first_when_it_has_a_model(assistant, monkeypatch, tmp_path):
    model = tmp_path / "en.onnx"
    model.write_bytes(b"model")
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER", str(stand_in(tmp_path, "piper", PIPER)))
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER_MODEL", str(model))
    spoken = assistant.voice.speak("Grüße")
    assert spoken["engine"] == "piper"
    assert json.loads(Path(spoken["path"]).read_bytes()[4:]) == {"text": "Grüße", "model": str(model)}


def test_piper_with_no_model_falls_back_to_the_system_voice(assistant, monkeypatch, tmp_path):
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER", str(stand_in(tmp_path, "piper", PIPER)))
    if HAS_SYSTEM_VOICE:
        assert assistant.voice.status()["speak"]["engine"] != "piper"


def no_voices(monkeypatch: Any) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)


def test_with_no_engine_it_says_what_to_install(assistant, monkeypatch):
    no_voices(monkeypatch)
    with pytest.raises(VoiceUnavailable) as caught:
        assistant.voice.speak("hello")
    assert "pip install piper-tts" in str(caught.value) and "espeak-ng" in str(caught.value)


@pytest.mark.parametrize("platform, tool", [("posix", "say"), ("posix", "espeak-ng")])
def test_each_system_voice_keeps_text_out_of_its_arguments(assistant, monkeypatch, platform, tool):
    commands: list[list[str]] = []
    monkeypatch.setattr("shardmesh.assistant.voice.os.name", platform)
    monkeypatch.setattr(shutil, "which", lambda name: f"/bin/{name}" if name == tool else None)
    monkeypatch.setattr("shardmesh.assistant.voice._run", lambda command, **_: commands.append(command) or b"")
    assistant.voice.speak("-v secret text", voice="Ada")
    flat = " ".join(commands[0])
    assert "secret" not in flat
    assert "-f" in commands[0] and "Ada" in commands[0]


@pytest.mark.parametrize("text, message", [("", "nothing to say"), ("   ", "nothing to say"), ("x" * 30000, "too long")])
def test_unusable_text_is_refused(assistant, text, message):
    with pytest.raises(VoiceError, match=message):
        assistant.voice.speak(text)


def test_an_out_path_is_used(assistant, monkeypatch, tmp_path):
    model = tmp_path / "en.onnx"
    model.write_bytes(b"model")
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER", str(stand_in(tmp_path, "piper", PIPER)))
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER_MODEL", str(model))
    out = tmp_path / "deep" / "speech.wav"
    assert Path(assistant.voice.speak("hi", out_path=out)["path"]) == out.resolve()
    assert out.exists()


def test_old_speech_is_tidied_away(assistant, monkeypatch, tmp_path):
    speech = assistant.voice.folder / "speech"
    speech.mkdir(parents=True)
    old, fresh = speech / "old.wav", speech / "fresh.wav"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    assistant.voice._new_speech_path()
    assert not old.exists() and fresh.exists()


# A persona changes how it sounds, and nothing else


def persona(assistant: Any, voice: Any) -> None:
    assistant.identity = SimpleNamespace(voice=voice, owner="owner")


def test_a_persona_picks_the_voice(assistant, monkeypatch, tmp_path):
    model = tmp_path / "en.onnx"
    model.write_bytes(b"model")
    (tmp_path / "Ada.onnx").write_bytes(b"model")
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER", str(stand_in(tmp_path, "piper", PIPER)))
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER_MODEL", str(model))
    persona(assistant, "Ada")
    spoken = assistant.voice.speak("hi")
    assert spoken["voice"] == "Ada"
    assert json.loads(Path(spoken["path"]).read_bytes()[4:])["model"] == str(tmp_path / "Ada.onnx")


@pytest.mark.parametrize("voice", ["Ada", {"name": "Ada"}, lambda: "Ada"])
def test_a_persona_voice_can_be_held_in_several_shapes(assistant, voice):
    persona(assistant, voice)
    assert assistant.voice.persona_voice() == "Ada"


@pytest.mark.parametrize("voice", ["../../evil", "C:\\models\\evil", "/etc/passwd", "-v evil", "x" * 100, "", None, 7])
def test_a_persona_voice_is_a_name_and_never_a_path(assistant, voice):
    persona(assistant, voice)
    assert assistant.voice.persona_voice() is None


def test_a_persona_voice_outside_the_model_folder_cannot_be_reached(assistant, monkeypatch, tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "en.onnx").write_bytes(b"model")
    (tmp_path / "evil.onnx").write_bytes(b"model")
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER", str(stand_in(tmp_path, "piper", PIPER)))
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER_MODEL", str(folder / "en.onnx"))
    persona(assistant, "../evil")
    spoken = assistant.voice.speak("hi")
    assert json.loads(Path(spoken["path"]).read_bytes()[4:])["model"] == str(folder / "en.onnx")


def test_changing_voice_changes_no_permission(assistant, monkeypatch, tmp_path):
    model = tmp_path / "en.onnx"
    model.write_bytes(b"model")
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER", str(stand_in(tmp_path, "piper", PIPER)))
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER_MODEL", str(model))
    assistant.broker = FakeBroker()
    assistant.voice.broker()
    declared_before = dict(assistant.broker.declared)

    persona(assistant, "Ada")
    with pytest.raises(VoiceRefused):
        assistant.voice.call(SPEAK, {"text": "hi"}, "owner")
    persona(assistant, "Grace")
    with pytest.raises(VoiceRefused):
        assistant.voice.call(SPEAK, {"text": "hi"}, "owner")

    assistant.broker.grants.add(("owner", SPEAK))
    assert assistant.voice.call(SPEAK, {"text": "hi"}, "owner").ok
    persona(assistant, "Ada")
    assert assistant.voice.call(SPEAK, {"text": "hi"}, "owner").ok
    assert assistant.broker.declared == declared_before
    assert assistant.broker.grants == {("owner", SPEAK)}


# The broker


def test_each_is_declared_to_the_broker(assistant):
    assistant.broker = FakeBroker()
    assistant.voice.broker()
    transcribe, speak = assistant.broker.declared[TRANSCRIBE], assistant.broker.declared[SPEAK]
    assert transcribe.effects == frozenset({Effect.READ}) and transcribe.scope_field == "audio_path"
    assert speak.effects == frozenset({Effect.WRITE_LOCAL}) and speak.consequential


def test_it_works_with_no_broker_present(assistant, monkeypatch, audio):
    fake_whisper(monkeypatch, [])
    assert assistant.voice.call(TRANSCRIBE, {"audio_path": str(audio)}, "owner").output == "hello there"


def test_what_a_recording_says_is_data(assistant, monkeypatch, audio):
    fake_whisper(monkeypatch, [])
    assistant.broker = FakeBroker()
    assistant.broker.grants.add(("owner", TRANSCRIBE))
    result = assistant.voice.call(TRANSCRIBE, {"audio_path": str(audio)}, "owner")
    assert result.provenance.trust is Trust.EXTERNAL
    assert assistant.broker.calls[0][1].actor == "voice"


def test_the_broker_can_ask_for_a_person(assistant):
    assistant.broker = FakeBroker()
    assistant.voice.broker()
    assistant.broker.ask.add(SPEAK)
    with pytest.raises(VoiceRefused) as caught:
        assistant.voice.call(SPEAK, {"text": "hi"}, "owner")
    assert caught.value.approval_id == "approval-3"


# Routes, over real HTTP


class Server:
    def __init__(self, assistant: Any) -> None:
        app = FastAPI()
        app.include_router(routes(assistant))
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        return f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"

    def __exit__(self, *_: Any) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def post(url: str, **options: Any) -> httpx.Response:
    return httpx.post(url, timeout=120, trust_env=False, **options)


def test_transcribing_over_http(assistant, monkeypatch):
    fake_whisper(monkeypatch, [])
    with Server(assistant) as url:
        response = post(f"{url}/assistant/voice/transcribe", content=b"RIFF....WAVE", headers={"Content-Type": "audio/wav"})
    assert response.status_code == 200
    assert response.json() == {"text": "hello there", "trust": "external"}
    assert list((assistant.voice.folder / "incoming").iterdir()) == []


@needs_system_voice
def test_speaking_over_http_gives_real_audio(assistant):
    with Server(assistant) as url:
        response = post(f"{url}/assistant/voice/speak", json={"text": "Hello from the route."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF" and len(response.content) > 5000
    assert list((assistant.voice.folder / "speech").iterdir()) == []


def test_routes_name_a_missing_engine(assistant, monkeypatch):
    no_whisper(monkeypatch)
    no_voices(monkeypatch)
    with Server(assistant) as url:
        heard = post(f"{url}/assistant/voice/transcribe", content=b"x", headers={"Content-Type": "audio/wav"})
        said = post(f"{url}/assistant/voice/speak", json={"text": "hi"})
    assert heard.status_code == said.status_code == 503
    assert heard.json()["error"] == INSTALL_WHISPER
    assert said.json()["error"] == INSTALL_SPEECH


def test_routes_go_through_the_broker(assistant, monkeypatch):
    fake_whisper(monkeypatch, [])
    assistant.broker = FakeBroker()
    assistant.voice.broker()
    with Server(assistant) as url:
        denied = post(f"{url}/assistant/voice/transcribe", content=b"x", headers={"Content-Type": "audio/wav"})
        assert denied.status_code == 403 and "refused" in denied.json()
        assert list((assistant.voice.folder / "incoming").iterdir()) == []

        assistant.broker.ask.add(SPEAK)
        held = post(f"{url}/assistant/voice/speak", json={"text": "hi"})
        assert held.status_code == 403 and held.json()["approval_id"] == "approval-3"

        assistant.broker.grants.add(("owner", TRANSCRIBE))
        assert post(f"{url}/assistant/voice/transcribe", content=b"x", headers={"Content-Type": "audio/wav"}).status_code == 200


def test_a_web_page_cannot_call_the_routes_without_asking_first(assistant):
    """Plain form types need no preflight, so they are refused, and nothing is spoken."""
    with Server(assistant) as url:
        spoken = post(f"{url}/assistant/voice/speak", content=b'{"text": "hi"}', headers={"Content-Type": "text/plain"})
        heard = post(f"{url}/assistant/voice/transcribe", content=b"x", headers={"Content-Type": "text/plain"})
    assert spoken.status_code == heard.status_code == 415


def test_a_path_from_outside_is_never_opened(assistant, monkeypatch, tmp_path):
    """The body is the audio. There is no argument that names a file."""
    opened: list[str] = []
    fake_whisper(monkeypatch, [])
    real = Voice.transcribe
    monkeypatch.setattr(Voice, "transcribe", lambda self, path: opened.append(str(path)) or real(self, path))
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"secret")
    with Server(assistant) as url:
        post(f"{url}/assistant/voice/transcribe?audio_path={secret}", content=str(secret).encode(), headers={"Content-Type": "audio/wav", "X-Audio-Path": str(secret)})
    assert str(secret) not in opened
    assert Path(opened[0]).parent == assistant.voice.folder / "incoming"


def test_bad_bodies_are_refused(assistant):
    with Server(assistant) as url:
        for body in (b"not json", b"[]", b'{"voice": "Ada"}', b'{"text": 7}'):
            assert post(f"{url}/assistant/voice/speak", content=body, headers={"Content-Type": "application/json"}).status_code == 400


def test_too_much_audio_is_refused_before_it_is_kept(assistant, monkeypatch):
    monkeypatch.setattr("shardmesh.assistant.voice_http.MAX_AUDIO_BYTES", 1000)
    with Server(assistant) as url:
        response = post(f"{url}/assistant/voice/transcribe", content=b"x" * 2000, headers={"Content-Type": "audio/wav"})
    assert response.status_code == 413
    assert not (assistant.voice.folder / "incoming").exists() or list((assistant.voice.folder / "incoming").iterdir()) == []


def request_from(host: str, user: Any = None) -> Any:
    return SimpleNamespace(state=SimpleNamespace(**({"user": user} if user else {})), client=SimpleNamespace(host=host))


def test_who_is_asking(assistant):
    assert _user(request_from("127.0.0.1"), assistant.voice) == "owner"
    assert _user(request_from("::1"), assistant.voice) == "owner"
    assert _user(request_from("100.64.0.2"), assistant.voice) is None  # a stranger is nobody
    assert _user(request_from("100.64.0.2", "ada"), assistant.voice) == "ada"  # a gateway signed them in
    assert _user(request_from("127.0.0.1", "ada"), assistant.voice) == "ada"


# Commands


def run_command(assistant: Any, *words: str) -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(list(words))
    return args.run(assistant, args)


def test_status_says_what_works_and_what_to_install(assistant, monkeypatch, capsys):
    no_whisper(monkeypatch)
    no_voices(monkeypatch)
    assert run_command(assistant, "voice", "status") == 0
    out = capsys.readouterr().out
    assert "Transcribing: not available" in out and "pip install faster-whisper" in out
    assert "Speaking: not available" in out and "piper" in out


def test_status_names_a_working_engine(assistant, monkeypatch, capsys):
    fake_whisper(monkeypatch, [])
    assert run_command(assistant, "voice", "status") == 0
    assert "Transcribing: faster-whisper, model base" in capsys.readouterr().out


def test_the_commands_transcribe_and_speak(assistant, monkeypatch, audio, tmp_path, capsys):
    fake_whisper(monkeypatch, [])
    model = tmp_path / "en.onnx"
    model.write_bytes(b"model")
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER", str(stand_in(tmp_path, "piper", PIPER)))
    monkeypatch.setenv("SHARDMESH_ASSISTANT_PIPER_MODEL", str(model))

    assert run_command(assistant, "voice", "transcribe", str(audio)) == 0
    assert "hello there" in capsys.readouterr().out

    out = tmp_path / "said.wav"
    assert run_command(assistant, "voice", "speak", "hello", "--out", str(out), "--voice", "Ada") == 0
    assert out.exists() and "using piper" in capsys.readouterr().out

    assert run_command(assistant, "voice", "transcribe", str(tmp_path / "missing.wav")) == 1
    assert "no audio file" in capsys.readouterr().out


def test_commands_answer_to_the_broker_too(assistant, audio, capsys):
    assistant.broker = FakeBroker()
    assert run_command(assistant, "voice", "transcribe", str(audio)) == 1
    assert "refused voice.transcribe" in capsys.readouterr().out
    assert run_command(assistant, "voice", "--user", "ada", "speak", "hi") == 1
