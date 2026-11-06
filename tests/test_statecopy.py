"""Copies of saved states: they leave behind the reply, and nothing ever waits for one."""

import asyncio
import time

import httpx
import pytest

from isharded import statecopy
from isharded.protocol import META_HEADER, SavePoint, decode_meta
from isharded.statecopy import COPY_LEAST_TOKENS, QUEUED_COPIES, CopySender, fetch_file
from isharded.statestore import HISTORY, SUFFIX


def save(key, length=COPY_LEAST_TOKENS, replaces=None):
    return SavePoint(at=length, length=length, key=key, kind=HISTORY, replaces=replaces)


def sender_for(handler, tmp_path, rate=10**9):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sender = CopySender(client, lambda: {"authorization": "Bearer t"}, rate)
    sender.aim("http://holder:1/", tmp_path, "space")
    return sender


def state_file(tmp_path, key, size=100):
    (tmp_path / (key + SUFFIX)).write_bytes(b"x" * size)


async def finished(sender):
    while sender.task is not None and not sender.task.done():
        await asyncio.sleep(0.01)


def test_a_copy_reaches_the_holder_with_its_name(tmp_path):
    seen = []

    async def handler(request):
        seen.append((str(request.url), decode_meta(request.headers[META_HEADER]), await request.aread(), request.headers))
        return httpx.Response(200, json={})

    async def scenario():
        sender = sender_for(handler, tmp_path)
        state_file(tmp_path, "a")
        sender.add([save("a", replaces="older")])
        await finished(sender)
        return sender

    sender = asyncio.run(scenario())
    url, meta, body, headers = seen[0]
    assert url == "http://holder:1/internal/state/copy"
    assert meta == {"namespace": "space", "key": "a", "replaces": "older"}
    assert body == b"x" * 100
    assert headers["content-length"] == "100", "a node refuses a body of no stated length"
    assert sender.report()["copies_sent"] == 1 and sender.report()["copy_bytes_sent"] == 100


def test_a_short_state_is_not_worth_a_copy(tmp_path):
    async def scenario():
        sender = sender_for(lambda request: httpx.Response(200), tmp_path)
        sender.add([save("a", COPY_LEAST_TOKENS - 1)])
        return sender.task

    assert asyncio.run(scenario()) is None


def test_with_no_holder_nothing_is_sent(tmp_path):
    async def scenario():
        sender = sender_for(lambda request: httpx.Response(200), tmp_path)
        sender.aim(None, tmp_path, "space")
        sender.add([save("a")])
        return sender.task

    assert asyncio.run(scenario()) is None


def test_a_rate_of_zero_turns_copies_off(tmp_path):
    async def scenario():
        sender = sender_for(lambda request: httpx.Response(200), tmp_path, rate=0)
        sender.add([save("a")])
        return sender.task

    assert asyncio.run(scenario()) is None


def test_copies_keep_to_their_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(statecopy, "CHUNK_BYTES", 1000)

    async def handler(request):
        await request.aread()
        return httpx.Response(200, json={})

    async def scenario():
        sender = sender_for(handler, tmp_path, rate=20_000)
        state_file(tmp_path, "a", 6000)
        begun = time.monotonic()
        sender.add([save("a")])
        await finished(sender)
        return time.monotonic() - begun

    assert asyncio.run(scenario()) >= 0.25


def test_a_newer_turn_replaces_one_still_waiting(tmp_path):
    sent = []
    gate = asyncio.Event

    async def scenario():
        hold = gate()

        async def handler(request):
            sent.append(decode_meta(request.headers[META_HEADER])["key"])
            await request.aread()
            await hold.wait()
            return httpx.Response(200, json={})

        sender = sender_for(handler, tmp_path)
        for name in ("busy", "two", "three"):
            state_file(tmp_path, name)
        sender.add([save("busy")])
        await asyncio.sleep(0.05)  # that one is on its way
        sender.add([save("two")])
        sender.add([save("three", replaces="two")])
        hold.set()
        await finished(sender)
        return sender

    sender = asyncio.run(scenario())
    assert sent == ["busy", "three"], "a slow link falls behind, and never backs up"
    assert sender.report()["copies_dropped"] == 1


def test_a_link_that_cannot_keep_up_queues_only_so_much(tmp_path):
    async def scenario():
        hold = asyncio.Event()

        async def handler(request):
            await hold.wait()
            return httpx.Response(200, json={})

        sender = sender_for(handler, tmp_path)
        state_file(tmp_path, "first")
        sender.add([save("first")])
        await asyncio.sleep(0.05)
        for index in range(QUEUED_COPIES * 3):
            sender.add([save(f"state{index}")])
        waiting = len(sender.queue)
        sender.task.cancel()
        return waiting, sender.report()["copies_dropped"]

    waiting, dropped = asyncio.run(scenario())
    assert waiting == QUEUED_COPIES and dropped == QUEUED_COPIES * 2


@pytest.mark.parametrize(
    "answer",
    [
        lambda request: httpx.Response(500),
        lambda request: httpx.Response(409, json={"detail": "no disk here"}),
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("gone")),
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("silent")),
    ],
)
def test_a_copy_that_fails_is_counted_and_the_rest_still_go(tmp_path, answer):
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return answer(request)
        await request.aread()
        return httpx.Response(200, json={})

    async def scenario():
        sender = sender_for(handler, tmp_path)
        for name in ("a", "b"):
            state_file(tmp_path, name)
        sender.add([save("a"), save("b")])
        await finished(sender)
        return sender.report()

    report = asyncio.run(scenario())
    assert report["copies_failed"] == 1 and report["copies_sent"] == 1


def test_a_state_never_written_is_given_up_on(tmp_path, monkeypatch):
    monkeypatch.setattr(statecopy, "FILE_WAIT_SECONDS", 0.05)

    async def scenario():
        sender = sender_for(lambda request: httpx.Response(200), tmp_path)
        sender.add([save("never")])
        await finished(sender)
        return sender.report()

    assert asyncio.run(scenario())["copies_failed"] == 1


def test_adding_a_copy_never_waits_for_the_link(tmp_path):
    """Called in the middle of a reply, so even a link that has stopped must cost it nothing."""

    async def scenario():
        hold = asyncio.Event()

        async def handler(request):
            await hold.wait()
            return httpx.Response(200, json={})

        sender = sender_for(handler, tmp_path)
        state_file(tmp_path, "a")
        begun = time.perf_counter()
        sender.add([save("a")])
        spent = time.perf_counter() - begun
        sender.task.cancel()
        return spent

    assert asyncio.run(scenario()) < 0.01


def test_moving_to_a_new_stage_forgets_what_was_queued(tmp_path):
    async def scenario():
        hold = asyncio.Event()

        async def handler(request):
            await hold.wait()
            return httpx.Response(200, json={})

        sender = sender_for(handler, tmp_path)
        state_file(tmp_path, "first")
        sender.add([save("first")])
        await asyncio.sleep(0.05)
        sender.add([save("second")])
        sender.aim("http://other:1", tmp_path, "another")
        sender.task.cancel()
        return len(sender.queue)

    assert asyncio.run(scenario()) == 0


def fetching(handler, target):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_file(client, "http://holder:1/x", target, {})

    return asyncio.run(scenario())


def test_a_fetched_state_lands_in_the_file(tmp_path):
    target = tmp_path / "state.part"
    assert fetching(lambda request: httpx.Response(200, content=b"abc"), target)
    assert target.read_bytes() == b"abc"


def test_a_holder_without_the_state_is_a_miss(tmp_path):
    target = tmp_path / "state.part"
    assert not fetching(lambda request: httpx.Response(404), target)
    assert not target.exists()


def test_a_holder_gone_silent_leaves_nothing_behind(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("silent")

    target = tmp_path / "state.part"
    assert not fetching(handler, target)
    assert not target.exists()
