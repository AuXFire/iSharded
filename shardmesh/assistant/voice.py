"""Speech in and out, local and optional.

Everything here runs on this machine and works without the internet. Any engine can
be missing, and then the error says what to install. Text to be spoken always
travels through a file or a pipe, never inside a command, so nothing it says can
run. A persona's voice picks how speech sounds and changes nothing else.
"""
from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import threading
import time
from typing import Any

from .contracts import CallContext, Capability, Effect, Provenance, Reach, ToolCall, ToolResult, Trust, Verdict

WHISPER_MODEL_SETTING = "SHARDMESH_ASSISTANT_WHISPER_MODEL"
WHISPER_CPP_SETTING = "SHARDMESH_ASSISTANT_WHISPER_CPP"
WHISPER_CPP_MODEL_SETTING = "SHARDMESH_ASSISTANT_WHISPER_CPP_MODEL"
PIPER_SETTING = "SHARDMESH_ASSISTANT_PIPER"
PIPER_MODEL_SETTING = "SHARDMESH_ASSISTANT_PIPER_MODEL"

TRANSCRIBE, SPEAK = "voice.transcribe", "voice.speak"
ENGINE_SECONDS = 600.0
MAX_TEXT = 20_000
MAX_AUDIO_BYTES = 64 << 20
OLD_SPEECH_SECONDS = 3600.0
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
VOICE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}")

INSTALL_WHISPER = (
    f"To transcribe speech, install faster-whisper with: pip install faster-whisper. "
    f"Or build whisper.cpp and point {WHISPER_CPP_SETTING} at its binary and {WHISPER_CPP_MODEL_SETTING} at a model."
)
INSTALL_SPEECH = (
    f"To speak, install piper with: pip install piper-tts, and point {PIPER_MODEL_SETTING} at a voice model. "
    "Or install a system voice: espeak-ng on Linux (apt install espeak-ng). Windows and macOS have one built in."
)

# Reads everything from the environment, so nothing a person or model says becomes part of a script.
SAPI_SCRIPT = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:SHARDMESH_VOICE_NAME) { try { $speaker.SelectVoice($env:SHARDMESH_VOICE_NAME) } catch { } }
$speaker.SetOutputToWaveFile($env:SHARDMESH_VOICE_OUT)
$speaker.Speak([IO.File]::ReadAllText($env:SHARDMESH_VOICE_TEXT, [Text.Encoding]::UTF8))
$speaker.Dispose()
"""


class VoiceError(Exception):
    """Speech failed. The message says why."""


class VoiceUnavailable(VoiceError):
    """An engine is missing. The message says what to install."""


class VoiceRefused(VoiceError):
    """The broker said no, or wants a person."""

    def __init__(self, message: str, approval_id: str | None = None) -> None:
        super().__init__(message)
        self.approval_id = approval_id


def _run(command: list[str], *, stdin: bytes | None = None, environment: dict[str, str] | None = None) -> bytes:
    """Never through a shell, so the arguments stay arguments."""
    try:
        done = subprocess.run(
            command, input=stdin, capture_output=True, timeout=ENGINE_SECONDS, env={**os.environ, **(environment or {})}
        )
    except subprocess.TimeoutExpired as error:
        raise VoiceError(f"{Path(command[0]).name} did not finish in {ENGINE_SECONDS:.0f} seconds.") from error
    except OSError as error:
        raise VoiceError(f"{Path(command[0]).name} could not start: {error}") from error
    if done.returncode != 0:
        raise VoiceError(f"{Path(command[0]).name} failed: {done.stderr.decode(errors='replace').strip()[-500:]}")
    return done.stdout


class Voice:
    """Picks an engine each time, so installing one needs no restart."""

    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.audit = assistant.audit
        self.clock = assistant.clock
        self.folder = Path(assistant.home) / "voice"
        self._declared_to: Any = None
        self._whisper: Any = None
        self._lock = threading.Lock()

    @property
    def owner(self) -> str:
        identity = getattr(self.assistant, "identity", None)
        return getattr(identity, "owner", None) or os.environ.get("SHARDMESH_ASSISTANT_USER") or "owner"

    def broker(self) -> Any:
        broker = getattr(self.assistant, "broker", None)
        if broker is not None and broker is not self._declared_to:
            self._declare(broker)
        return broker

    def _declare(self, broker: Any) -> None:
        self._declared_to = broker
        known = {capability.name for capability in broker.capabilities()}
        if TRANSCRIBE not in known:
            broker.register(
                Capability(
                    name=TRANSCRIBE,
                    summary="Turn a local audio file into text, on this machine",
                    effects=frozenset({Effect.READ}),
                    reads=("audio",),
                    resources=("cpu",),
                    input_schema={"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]},
                    scope_field="audio_path",
                    idempotent=True,
                ),
                self._handle_transcribe,
            )
        if SPEAK not in known:
            broker.register(
                Capability(
                    name=SPEAK,
                    summary="Turn text into a WAV file in the assistant's folder, on this machine",
                    effects=frozenset({Effect.WRITE_LOCAL}),
                    resources=("cpu",),
                    input_schema={"type": "object", "properties": {"text": {"type": "string"}, "voice": {"type": "string"}}, "required": ["text"]},
                ),
                self._handle_speak,
            )

    # Hearing.

    def transcribe(self, audio_path: str | Path) -> str:
        """The words in an audio file. Nothing is downloaded and nothing leaves this machine."""
        path = self._audio_file(audio_path)
        problem = INSTALL_WHISPER
        module = self._faster_whisper()
        if module is not None:
            try:
                return self._with_faster_whisper(module, path)
            except VoiceUnavailable as error:
                problem = str(error)
        binary = os.environ.get(WHISPER_CPP_SETTING)
        if binary:
            return self._with_whisper_cpp(binary, path)
        raise VoiceUnavailable(problem)

    @staticmethod
    def _audio_file(audio_path: str | Path) -> Path:
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise VoiceError(f"There is no audio file at {path}.")
        return path

    @staticmethod
    def _faster_whisper() -> Any:
        try:
            import faster_whisper
        except (ImportError, OSError):  # a broken native library is as good as not installed
            return None
        return faster_whisper

    def _with_faster_whisper(self, module: Any, path: Path) -> str:
        model = self._whisper_model(module)
        try:
            segments, _ = model.transcribe(str(path))
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as error:  # noqa: BLE001 - the library raises many kinds on bad audio
            raise VoiceError(f"That audio could not be read: {error}") from error

    def _whisper_model(self, module: Any) -> Any:
        """On the CPU, as the GPU is the mesh's, and from local files only."""
        name = os.environ.get(WHISPER_MODEL_SETTING) or "base"
        with self._lock:
            if self._whisper is None or self._whisper[0] != name:
                try:
                    self._whisper = (name, module.WhisperModel(name, device="cpu", compute_type="int8", local_files_only=True))
                except Exception as error:  # noqa: BLE001 - which error depends on the version and on what is missing
                    raise VoiceUnavailable(
                        f"faster-whisper has no {name} model on this machine, and nothing is downloaded by itself. "
                        f"Set {WHISPER_MODEL_SETTING} to a folder holding a converted model."
                    ) from error
            return self._whisper[1]

    @staticmethod
    def _with_whisper_cpp(binary: str, path: Path) -> str:
        model = os.environ.get(WHISPER_CPP_MODEL_SETTING)
        if not model:
            raise VoiceUnavailable(f"{WHISPER_CPP_SETTING} is set, so point {WHISPER_CPP_MODEL_SETTING} at a whisper.cpp model too.")
        return _run([binary, "-m", model, "-f", str(path), "-nt", "-np"]).decode(errors="replace").strip()

    # Speaking.

    def persona_voice(self) -> str | None:
        """How the current persona sounds. It is a name, never a path, and it decides nothing."""
        value = getattr(getattr(self.assistant, "identity", None), "voice", None)
        value = value() if callable(value) else value
        value = value.get("name") if isinstance(value, dict) else value
        return value if isinstance(value, str) and VOICE_NAME.fullmatch(value) else None

    def speak(self, text: str, voice: str | None = None, out_path: str | Path | None = None) -> dict[str, Any]:
        """Write speech to a WAV file. The interface decides whether and where to play it."""
        if not isinstance(text, str) or not text.strip():
            raise VoiceError("There is nothing to say.")
        if len(text) > MAX_TEXT:
            raise VoiceError(f"That is too long to speak at once. The limit is {MAX_TEXT} characters.")
        chosen = voice if isinstance(voice, str) and VOICE_NAME.fullmatch(voice) else self.persona_voice()
        out = Path(out_path).resolve() if out_path else self._new_speech_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        engine = self._speech_engine()
        engine[1](text, out, chosen)
        return {"path": str(out), "engine": engine[0], "voice": chosen}

    def _new_speech_path(self) -> Path:
        speech = self.folder / "speech"
        speech.mkdir(parents=True, exist_ok=True)
        for old in speech.glob("*.wav"):  # the caller keeps what it needs, and nobody tidies up for it
            try:
                if time.time() - old.stat().st_mtime > OLD_SPEECH_SECONDS:
                    old.unlink()
            except OSError:
                continue
        return speech / f"{secrets.token_hex(8)}.wav"

    def _speech_engine(self) -> tuple[str, Callable[[str, Path, str | None], None]]:
        piper = os.environ.get(PIPER_SETTING) or shutil.which("piper")
        model = os.environ.get(PIPER_MODEL_SETTING)
        if piper and model:
            return "piper", lambda text, out, voice: self._with_piper(piper, model, text, out, voice)
        if os.name == "nt" and shutil.which("powershell"):
            return "sapi", self._with_sapi
        if shutil.which("say"):
            return "say", self._with_say
        espeak = shutil.which("espeak-ng")
        if espeak:
            return "espeak-ng", lambda text, out, voice: self._with_espeak(espeak, text, out, voice)
        raise VoiceUnavailable(INSTALL_SPEECH)

    @staticmethod
    def _with_piper(binary: str, model: str, text: str, out: Path, voice: str | None) -> None:
        """A persona picks among voices beside the configured one, by name alone."""
        sibling = Path(model).with_name(f"{voice}.onnx") if voice else None
        chosen = str(sibling) if sibling and sibling.is_file() else model
        _run([binary, "--model", chosen, "--output_file", str(out)], stdin=text.encode())

    def _with_text_file(self, text: str, use: Callable[[Path], None]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{secrets.token_hex(8)}.txt"
        path.write_text(text, encoding="utf-8")
        try:
            use(path)
        finally:
            path.unlink(missing_ok=True)

    def _with_sapi(self, text: str, out: Path, voice: str | None) -> None:
        def use(path: Path) -> None:
            environment = {"SHARDMESH_VOICE_OUT": str(out), "SHARDMESH_VOICE_TEXT": str(path), "SHARDMESH_VOICE_NAME": voice or ""}
            _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", SAPI_SCRIPT], environment=environment)

        self._with_text_file(text, use)

    def _with_say(self, text: str, out: Path, voice: str | None) -> None:
        def use(path: Path) -> None:
            command = ["say", "-f", str(path), "-o", str(out), "--data-format=LEI16@22050"]
            _run(command + (["-v", voice] if voice else []))

        self._with_text_file(text, use)

    def _with_espeak(self, binary: str, text: str, out: Path, voice: str | None) -> None:
        def use(path: Path) -> None:
            _run([binary, "-f", str(path), "-w", str(out)] + (["-v", voice] if voice else []))

        self._with_text_file(text, use)

    def status(self) -> dict[str, Any]:
        """What works here, and what to install for what does not."""
        try:
            speech = {"engine": self._speech_engine()[0], "ready": True}
        except VoiceUnavailable as error:
            speech = {"engine": None, "ready": False, "install": str(error)}
        if self._faster_whisper() is not None:
            hearing = {"engine": "faster-whisper", "model": os.environ.get(WHISPER_MODEL_SETTING) or "base"}
        elif os.environ.get(WHISPER_CPP_SETTING):
            hearing = {"engine": "whisper.cpp", "model": os.environ.get(WHISPER_CPP_MODEL_SETTING)}
        else:
            hearing = {"engine": None, "install": INSTALL_WHISPER}
        return {"transcribe": hearing, "speak": speech}

    # Through the broker.

    def _provenance(self, source_id: str, trust: Trust) -> Provenance:
        return Provenance("tool", source_id, "voice", trust, self.clock(), Reach.POOL)

    def _handle_transcribe(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        """What a recording says is data, whoever seems to be speaking in it."""
        text = self.transcribe(str(arguments.get("audio_path", "")))
        return ToolResult(True, text, self._provenance(TRANSCRIBE, Trust.EXTERNAL))

    def _handle_speak(self, arguments: dict[str, Any], context: CallContext) -> ToolResult:
        spoken = self.speak(str(arguments.get("text", "")), arguments.get("voice"))
        return ToolResult(True, spoken["path"], self._provenance(SPEAK, Trust.VERIFIED), spoken)

    def call(self, capability: str, arguments: dict[str, Any], user: str) -> ToolResult:
        """As the broker allows. With no broker, this is a local tool on the owner's own machine."""
        broker = self.broker()
        context = CallContext(user=user, actor="voice")
        if broker is None:
            handler = self._handle_transcribe if capability == TRANSCRIBE else self._handle_speak
            return handler(arguments, context)
        call = ToolCall(capability, arguments)
        decision = broker.check(call, context)
        if decision.verdict is Verdict.DENY:
            self.audit.record("voice", "voice.denied", capability, {"user": user, "reason": decision.reason})
            raise VoiceRefused(f"The broker refused {capability}: {decision.reason}")
        if decision.verdict is Verdict.ASK:
            raise VoiceRefused(f"A person has to approve {capability}: {decision.reason}", decision.approval_id)
        return broker.invoke(call, context)


def attach(assistant: Any) -> None:
    assistant.voice = Voice(assistant)
    assistant.voice.broker()


def _user(request: Any, voice: Voice) -> str | None:
    """Who a gateway says this is, or the owner at their own machine. Nobody else."""
    user = getattr(request.state, "user", None)
    if isinstance(user, str) and user:
        return user
    return voice.owner if request.client is not None and request.client.host in LOOPBACK else None


def routes(assistant: Any) -> Any:
    """The HTTP half is kept apart, because every shardmesh command imports this module."""
    from . import voice_http

    return voice_http.routes(assistant)


def run_status(assistant: Any, args: Any) -> int:
    status = assistant.voice.status()
    hearing, speech = status["transcribe"], status["speak"]
    print(f"Transcribing: {hearing['engine'] or 'not available'}" + (f", model {hearing['model']}" if hearing.get("model") else ""))
    if hearing.get("install"):
        print(hearing["install"])
    print(f"Speaking: {speech['engine'] or 'not available'}")
    if speech.get("install"):
        print(speech["install"])
    return 0


def run_transcribe(assistant: Any, args: Any) -> int:
    print(assistant.voice.call(TRANSCRIBE, {"audio_path": args.audio}, args.user or assistant.voice.owner).output)
    return 0


def run_speak(assistant: Any, args: Any) -> int:
    arguments = {"text": args.text, **({"voice": args.voice} if args.voice else {})}
    result = assistant.voice.call(SPEAK, arguments, args.user or assistant.voice.owner)
    path = Path(result.data["path"])
    if args.out:
        shutil.move(str(path), args.out)
        path = Path(args.out)
    print(f"Wrote {path}, using {result.data['engine']}.")
    return 0


def _guarded(run: Callable[[Any, Any], int]) -> Callable[[Any, Any], int]:
    def guarded(assistant: Any, args: Any) -> int:
        try:
            return run(assistant, args)
        except VoiceError as error:
            print(error)
            return 1

    return guarded


def add_commands(commands: Any) -> None:
    voice = commands.add_parser("voice", help="speech in and out, on this machine")
    voice.add_argument("--user", default=None, help="who is asking; default is the owner")
    actions = voice.add_subparsers(dest="voice_command", required=True)

    actions.add_parser("status", help="which engines work here, and what to install").set_defaults(run=_guarded(run_status))

    transcribe = actions.add_parser("transcribe", help="print the words in an audio file")
    transcribe.add_argument("audio", help="the audio file")
    transcribe.set_defaults(run=_guarded(run_transcribe))

    speak = actions.add_parser("speak", help="write speech to a WAV file")
    speak.add_argument("text", help="what to say")
    speak.add_argument("--voice", default=None, help="a voice name; default is the persona's")
    speak.add_argument("--out", default=None, help="where to put the file")
    speak.set_defaults(run=_guarded(run_speak))
