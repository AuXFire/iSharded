"""Speech over HTTP, kept apart from voice.py for the reason federation_http.py gives."""
from __future__ import annotations

import json
from pathlib import Path
import secrets
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from .contracts import Trust
from .voice import MAX_AUDIO_BYTES, MAX_TEXT, SPEAK, TRANSCRIBE, VoiceError, VoiceRefused, VoiceUnavailable, _user


async def _bounded_body(request: Request, limit: int) -> bytes | None:
    if int(request.headers.get("content-length") or 0) > limit:
        return None
    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return body


def _media_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";")[0].strip().lower()


def _failure(error: VoiceError) -> JSONResponse:
    if isinstance(error, VoiceRefused):
        return JSONResponse({"refused": str(error), "approval_id": error.approval_id}, status_code=403)
    return JSONResponse({"error": str(error)}, status_code=503 if isinstance(error, VoiceUnavailable) else 400)


def routes(assistant: Any) -> APIRouter:
    """Audio arrives as bytes and leaves as bytes. A path from outside is never opened."""
    router = APIRouter(prefix="/assistant/voice")
    voice = assistant.voice

    @router.post("/transcribe")
    async def transcribe(request: Request) -> Response:
        user = _user(request, voice)
        if user is None:
            return JSONResponse({"error": "This needs a signed-in user."}, status_code=401)
        # A web page cannot send these types without asking first, so it cannot call this behind the owner's back.
        kind = _media_type(request)
        if not kind.startswith("audio/") and kind != "application/octet-stream":
            return JSONResponse({"error": "Send the audio as the body, with an audio Content-Type."}, status_code=415)
        body = await _bounded_body(request, MAX_AUDIO_BYTES)
        if body is None:
            return JSONResponse({"error": f"Audio may be at most {MAX_AUDIO_BYTES >> 20} MiB."}, status_code=413)
        incoming = voice.folder / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        path = incoming / f"{secrets.token_hex(8)}.audio"
        path.write_bytes(body)
        try:
            result = await run_in_threadpool(voice.call, TRANSCRIBE, {"audio_path": str(path)}, user)
        except VoiceError as error:
            return _failure(error)
        finally:
            path.unlink(missing_ok=True)
        return JSONResponse({"text": result.output, "trust": Trust.EXTERNAL.value})

    @router.post("/speak")
    async def speak(request: Request) -> Response:
        user = _user(request, voice)
        if user is None:
            return JSONResponse({"error": "This needs a signed-in user."}, status_code=401)
        if _media_type(request) != "application/json":
            return JSONResponse({"error": 'Send JSON: {"text": "...", "voice": "optional name"}.'}, status_code=415)
        body = await _bounded_body(request, MAX_TEXT * 8)
        try:
            fields = json.loads(body) if body is not None else None
        except (RecursionError, ValueError):
            fields = None
        if not isinstance(fields, dict) or not isinstance(fields.get("text"), str):
            return JSONResponse({"error": 'Send JSON: {"text": "...", "voice": "optional name"}.'}, status_code=400)
        arguments = {"text": fields["text"], **({"voice": fields["voice"]} if isinstance(fields.get("voice"), str) else {})}
        try:
            result = await run_in_threadpool(voice.call, SPEAK, arguments, user)
        except VoiceError as error:
            return _failure(error)
        path = Path(result.data["path"])
        audio = path.read_bytes()
        path.unlink(missing_ok=True)
        return Response(audio, media_type="audio/wav", headers={"X-Voice-Engine": str(result.data["engine"])})

    return router
