"""The peer protocol over HTTP.

Kept apart from federation.py, as gateway_cli.py is from gateway.py: every isharded
command builds the assistant's parser, and importing FastAPI there would cost all of
them most of a second. FastAPI reads route annotations from module globals, so these
routes need a module of their own and not a local import.
"""
from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .federation import MAX_MESSAGE_BYTES, PROTOCOL, Refused


async def _read_body(request: Request) -> Any:
    """Bounded before anyone is identified."""
    if int(request.headers.get("content-length") or 0) > MAX_MESSAGE_BYTES:
        raise Refused("this message is too large")
    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_MESSAGE_BYTES:
            raise Refused("this message is too large")
    try:
        return json.loads(body)
    except (RecursionError, ValueError) as error:
        raise Refused("this message could not be opened") from error


def routes(assistant: Any) -> APIRouter:
    """Only what a stranger may reach. Messages prove themselves, so a gateway must not ask them for a key."""
    router = APIRouter(prefix="/federation")
    federation = assistant.federation

    async def answer(request: Request, handler: Callable[[Any], tuple[int, dict[str, Any]]]) -> JSONResponse:
        try:
            status, reply = await run_in_threadpool(handler, await _read_body(request))
        except Refused as refusal:
            return JSONResponse({"refused": str(refusal)}, status_code=403)
        return JSONResponse(reply, status_code=status)

    @router.get("/identity")
    async def identity() -> dict[str, Any]:
        return {"fingerprint": federation.identity.fingerprint, "protocol": PROTOCOL}

    @router.post("/pair")
    async def pair(request: Request) -> JSONResponse:
        return await answer(request, federation.pair)

    @router.post("/request")
    async def incoming(request: Request) -> JSONResponse:
        return await answer(request, federation.receive)

    @router.post("/notice")
    async def notice(request: Request) -> JSONResponse:
        return await answer(request, federation.receive_notice)

    return router
