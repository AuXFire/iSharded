"""Real processes, each worker behind two loopback addresses: a slow relay and a quick one.

Workers listen on 127.0.0.1. Their advertised address is a relay on 127.0.0.2 with a delay,
as a tunnel has. Their other address is a relay on 127.0.0.3 that a test can silence, which
holds every byte and closes nothing, as a dead link does.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import httpx
import pytest

from isharded.engine import find_library

BUILD = os.environ.get("ISHARDED_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("ISHARDED_TEST_GGUF", "")
ROOT = Path(__file__).resolve().parents[1]
ADVERTISED, DIRECT, LISTEN = "127.0.0.2", "127.0.0.3", "127.0.0.1"
TOKEN = "live-test-pool-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
DELAY_SECONDS = 0.05


def second_loopback() -> bool:
    with socket.socket() as probe:
        try:
            probe.bind((DIRECT, 0))
        except OSError:
            return False  # macOS has only 127.0.0.1
    return True


pytestmark = pytest.mark.skipif(not (BUILD and MODEL and find_library() and second_loopback()), reason="needs a real model, the built engine, and more than one loopback address")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((LISTEN, 0))
        return int(probe.getsockname()[1])


class Relay:
    def __init__(self, host: str, port: int, target: int, delay: float = 0.0) -> None:
        self.host, self.port, self.target, self.delay = host, port, target, delay
        self.silent = False
        self.writers: list[asyncio.StreamWriter] = []

    async def _pipe(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        waiting: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue()

        async def deliver() -> None:
            while True:
                due, data = await waiting.get()
                await asyncio.sleep(max(due - time.monotonic(), 0))
                while self.silent:
                    await asyncio.sleep(0.05)
                if not data:
                    writer.close()
                    return
                writer.write(data)
                await writer.drain()

        delivering = asyncio.ensure_future(deliver())
        try:
            while data := await reader.read(65536):
                waiting.put_nowait((time.monotonic() + self.delay, data))
        except OSError:
            pass
        waiting.put_nowait((time.monotonic() + self.delay, b""))
        await asyncio.gather(delivering, return_exceptions=True)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            far_reader, far_writer = await asyncio.open_connection(LISTEN, self.target)
        except OSError:
            writer.close()
            return
        self.writers += [writer, far_writer]
        await asyncio.gather(self._pipe(reader, far_writer), self._pipe(far_reader, writer), return_exceptions=True)

    async def start(self) -> asyncio.AbstractServer:
        return await asyncio.start_server(self._serve, self.host, self.port)


class Pool:
    """An orchestrator and two workers, with relays in front of the workers."""

    def __init__(self, folder: Path, ring: bool, swapped: bool = False) -> None:
        self.folder, self.processes = folder, []
        self.client = httpx.Client(timeout=120, trust_env=False)
        self.control = free_port()
        self.ports, self.doors = [free_port(), free_port()], [free_port(), free_port()]
        token = folder / "pool.token"
        token.write_text(TOKEN)
        self.env = {**os.environ, "PYTHONPATH": str(ROOT), "ISHARDED_POOL_TOKEN_FILE": str(token), "NO_PROXY": "*", "ISHARDED_RING": "1" if ring else "0", "ISHARDED_HOST": LISTEN}
        for name in ("ISHARDED_POOL_TOKEN", "ISHARDED_CA_FILE", "ISHARDED_IDENTITY_CERT", "ISHARDED_IDENTITY_KEY", "ISHARDED_TLS_CERT", "ISHARDED_TLS_KEY"):
            self.env.pop(name, None)
        # Swapped, each address answers, and it is the other worker that answers.
        far = [*reversed(self.ports), *reversed(self.doors)] if swapped else self.ports + self.doors
        self.quick = [Relay(DIRECT, port, target) for port, target in zip(self.ports + self.doors, far)]
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._relay, daemon=True).start()

    def _relay(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.relays = [*self.quick, *(Relay(ADVERTISED, port, port, DELAY_SECONDS) for port in self.ports + self.doors)]
        self.servers = [self.loop.run_until_complete(relay.start()) for relay in self.relays]
        self.loop.run_forever()
        self.loop.close()

    async def _stop_relays(self) -> None:
        for server in self.servers:
            server.close()
        running = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        for relay in self.relays:
            for writer in relay.writers:
                writer.close()
        await asyncio.sleep(0.2)  # a closed transport finishes on the next turns of this loop

    def _start(self, name: str, module: str, arguments: list[str], **env: str) -> None:
        log = (self.folder / f"{name}.log").open("w")
        self.processes.append(subprocess.Popen([sys.executable, "-m", module, *arguments], env={**self.env, **env}, stdout=log, stderr=log, cwd=str(ROOT)))

    def _wait(self, ready: Any, seconds: float = 120.0) -> Any:
        began = time.monotonic()
        while time.monotonic() - began < seconds:
            try:
                found = ready()
            except httpx.HTTPError:
                found = None
            if found:
                return found
            time.sleep(0.25)
        raise AssertionError(f"waited {seconds} s for {ready}; logs are in {self.folder}")

    def up(self) -> None:
        url = f"http://{LISTEN}:{self.control}"
        files = [f"--{kind}-file" for kind in ("keys", "state", "audit")]
        arguments = ["--host", LISTEN, "--port", str(self.control), *sum(([flag, str(self.folder / flag[2:])] for flag in files), [])]
        self._start("orchestrator", "isharded.orchestrator", arguments)
        self._wait(lambda: self.client.get(url + "/health").status_code == 200)
        for index, port in enumerate(self.ports):
            arguments = ["--host", LISTEN, "--port", str(port), "--worker-id", f"w{index + 1}", "--orchestrator-url", url, "--advertise-url", f"http://{ADVERTISED}:{port}", "--reach-hosts", DIRECT]
            arguments += ["--device", "cpu", "--gguf-dir", str(Path(MODEL).parent), "--cache-dir", str(self.folder / f"cache{index}"), "--llama-dir", BUILD, "--state-disk-gb", "0", "--state-ram-gb", "0", "--weights-ram-gb", "1.5"]
            self._start(f"w{index + 1}", "isharded.worker", arguments, ISHARDED_DATA_PORT=str(self.doors[index]))
        self._wait(lambda: len(self.status().get("workers", {})) == 2)
        body = {"model_id": MODEL, "wait": False, "placement": "spread", "context_tokens": 1024, "parallel": 1, "draft_model": "none"}
        self.client.post(url + "/admin/load", json=body, headers=HEADERS).raise_for_status()
        self._wait(self._loaded, 240.0)

    def _loaded(self) -> bool:
        now = self.status()
        assert not (now.get("load_job") or {}).get("error"), now["load_job"]["error"]
        return now.get("status") == "ready"

    def stop(self) -> None:
        for process in self.processes:
            process.kill()
            process.wait()
        asyncio.run_coroutine_threadsafe(self._stop_relays(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.client.close()

    def status(self) -> dict[str, Any]:
        return self.client.get(f"http://{LISTEN}:{self.control}/admin/status", headers=HEADERS).json()

    def healths(self) -> list[str]:
        return [self.client.get(f"http://{LISTEN}:{port}/health", headers=HEADERS, timeout=30).text for port in self.ports]

    def links(self) -> dict[str, dict[str, Any]]:
        return {f"{link['source']}>{link['target']}": link for link in self.status().get("links", [])}

    def wait_until_direct(self) -> None:
        self._wait(lambda: all(link.get("direct") for link in self.links().values()) and len(self.links()) == 2 and any(json.loads(text)["telemetry"]["hop_direct"] for text in self.healths()))

    def silence(self, silent: bool) -> None:
        for relay in self.quick:
            relay.silent = silent

    def ask(self, cut_after: int = 0) -> tuple[str, str | None]:
        body = {"messages": [{"role": "user", "content": "Describe how rivers shape the land."}], "max_tokens": 48, "temperature": 0, "enable_thinking": False, "stream": True}
        pieces: list[str] = []
        error = None
        with self.client.stream("POST", f"http://{LISTEN}:{self.control}/v1/chat/completions", json=body, headers=HEADERS) as response:
            for line in response.iter_lines():
                if not line.startswith("data: {"):
                    continue
                data = json.loads(line[6:])
                error = data.get("error") or error
                piece = (data["choices"][0].get("delta") or {}).get("content") if data.get("choices") else None
                if piece:
                    pieces.append(piece)
                    if len(pieces) == cut_after:
                        self.silence(True)
        return "".join(pieces), error


@pytest.fixture
def running(tmp_path: Path):
    made: list[Pool] = []

    def start(ring: bool, swapped: bool = False) -> Pool:
        pool = Pool(tmp_path, ring, swapped)
        made.append(pool)
        pool.up()
        return pool

    yield start
    for pool in made:
        pool.stop()


@pytest.mark.parametrize("ring", [False, True], ids=["http", "ring"])
def test_a_proven_address_is_used_and_silencing_it_mid_reply_changes_nothing(running, ring: bool) -> None:
    pool = running(ring)
    pool.wait_until_direct()
    links = pool.links()
    assert all(link["median_ms"] < DELAY_SECONDS * 1000 for link in links.values())  # the slow relay costs twice this
    text, error = pool.ask()
    assert text and error is None

    silenced, error = pool.ask(cut_after=12)
    assert error is None
    assert silenced == text
    healths = "".join(pool.healths())
    assert sum(json.loads(item)["telemetry"]["hop_fallbacks"] for item in pool.healths()) + ring >= 1
    assert DIRECT not in healths and DIRECT not in json.dumps(pool.status())


def test_an_address_that_answers_as_another_worker_never_counts(running) -> None:
    pool = running(False, swapped=True)
    began = time.monotonic()
    while time.monotonic() - began < 25:
        assert not any(link.get("direct") for link in pool.links().values())
        time.sleep(1)
    text, error = pool.ask()
    assert text and error is None
    assert not any(json.loads(item)["telemetry"]["hop_direct"] for item in pool.healths())
