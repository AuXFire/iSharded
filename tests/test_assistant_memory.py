"""Memory: what was said, what was kept, who said it, and who may see it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import threading

import pytest

from isharded.assistant.app import Assistant
from isharded.assistant.cli import add_arguments, run
from isharded.assistant.contracts import Provenance, Reach, Trust
from isharded.assistant.memory import MemoryNotFound, MemoryRefused

DAY = 86_400.0


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def assistant(tmp_path, clock):
    with Assistant(tmp_path, clock=clock) as instance:
        yield instance


def said(trust: Trust = Trust.OWNER, at: float = 1_000_000.0, reach: Reach = Reach.POOL, author: str = "alice") -> Provenance:
    return Provenance("user", "test", author, trust, at, reach)


def texts(items) -> list[str]:
    return [item.text for item in items]


def run_cli(home: Path, *arguments: str) -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return run(parser.parse_args(["--home", str(home), *arguments]))


# What was said


def test_messages_are_kept_word_for_word_with_who_produced_them(assistant):
    content = "  Unicode kept: → café \U0001f40d\n\tand whitespace  "
    assistant.memory.record_message(
        "alice", "c1", "assistant", content, model="qwen", provider="mesh", prompt_tokens=11, completion_tokens=7, turn_id="t1"
    )
    row = assistant.memory.recent("alice", "c1")[0]
    assert row["content"] == content
    assert (row["model"], row["provider"], row["prompt_tokens"], row["completion_tokens"]) == ("qwen", "mesh", 11, 7)
    assert row["created_at"] == 1_000_000.0 and row["turn_id"] == "t1"


def test_recent_returns_the_latest_in_order(assistant):
    for number in range(5):
        assistant.memory.record_message("alice", "c1", "user", f"message {number}")
    assert [row["content"] for row in assistant.memory.recent("alice", "c1", limit=2)] == ["message 3", "message 4"]


def test_what_was_said_can_be_searched(assistant):
    assistant.memory.record_message("alice", "c1", "user", "The launch is on Thursday")
    assistant.memory.record_message("alice", "c2", "user", "Unrelated")
    assert [row["conversation_id"] for row in assistant.memory.search_messages("alice", "launch")] == ["c1"]
    assert assistant.memory.search_messages("alice", "launch", conversation_id="c2") == []


# What was kept


def test_a_memory_keeps_its_whole_provenance(assistant, clock):
    origin = Provenance("tool", "call-7", "qwen", Trust.MODEL, clock.now - 5, Reach.MACHINE)
    memory_id = assistant.memory.remember("alice", "semantic", "Paris is in France", origin, subject="Capital ", expires_at=clock.now + DAY)
    row = assistant.memory.get("alice", memory_id)
    assert (row["source"], row["source_id"], row["author"], row["trust"]) == ("tool", "call-7", "qwen", "model")
    assert row["reach"] == Reach.MACHINE and row["captured_at"] == clock.now - 5
    assert row["subject"] == "capital" and row["expires_at"] == clock.now + DAY and row["superseded_by"] is None
    item = assistant.memory.search("alice", "Paris")[0]
    assert item.provenance.trust is Trust.MODEL and item.provenance.reach is Reach.MACHINE
    assert item.provenance.source_id == f"memory:{memory_id}"


@pytest.mark.parametrize("kind", ["episodic", "semantic", "procedural", "user", "system"])
def test_each_kind_of_memory_can_be_kept_and_filtered(assistant, kind):
    assistant.memory.remember("alice", kind, "zebra fact", said())
    assert len(assistant.memory.search("alice", "zebra", kinds=(kind,))) == 1
    other = "user" if kind != "user" else "system"
    assert assistant.memory.search("alice", "zebra", kinds=(other,)) == []


def test_an_unknown_kind_lists_the_real_ones(assistant):
    with pytest.raises(ValueError, match="episodic, semantic, procedural, user, system"):
        assistant.memory.remember("alice", "gossip", "text", said())


def test_a_models_guess_can_never_overwrite_what_the_owner_said(assistant):
    owner = assistant.memory.remember("alice", "user", "My favourite colour is green", said(Trust.OWNER), subject="favourite colour")
    guess = assistant.memory.remember("alice", "user", "My favourite colour is purple", said(Trust.MODEL, author="qwen"), subject="favourite colour")
    assert texts(assistant.memory.search("alice", "favourite colour")) == ["My favourite colour is green"]
    assert assistant.memory.get("alice", owner)["superseded_by"] is None
    assert assistant.memory.get("alice", guess)["superseded_by"] == int(owner)
    assert {row["text"] for row in assistant.memory.history("alice", owner)} == {"My favourite colour is green", "My favourite colour is purple"}


@pytest.mark.parametrize(
    ("old", "new", "replaced"),
    [
        (Trust.OWNER, Trust.OWNER, True),
        (Trust.MODEL, Trust.MODEL, True),
        (Trust.MODEL, Trust.OWNER, True),
        (Trust.OWNER, Trust.VERIFIED, True),
        (Trust.EXTERNAL, Trust.MODEL, True),
        (Trust.VERIFIED, Trust.OWNER, False),
        (Trust.OWNER, Trust.MODEL, False),
        (Trust.OWNER, Trust.EXTERNAL, False),
        (Trust.MODEL, Trust.EXTERNAL, False),
    ],
)
def test_a_fact_replaces_another_only_if_at_least_as_trusted(assistant, old, new, replaced):
    assistant.memory.remember("alice", "user", "first value", said(old), subject="thing")
    assistant.memory.remember("alice", "user", "second value", said(new), subject="thing")
    assert texts(assistant.memory.search("alice", "value")) == ["second value" if replaced else "first value"]


def test_an_expired_fact_does_not_block_a_lesser_one(assistant, clock):
    assistant.memory.remember("alice", "user", "old value", said(Trust.OWNER), subject="thing", expires_at=clock.now + 10)
    clock.now += 20
    assistant.memory.remember("alice", "user", "new value", said(Trust.MODEL, at=clock.now), subject="thing")
    assert texts(assistant.memory.search("alice", "value")) == ["new value"]


def test_superseded_rows_stay_for_history_but_leave_search(assistant):
    old = assistant.memory.remember("alice", "user", "I live in Leeds", said(), subject="home")
    new = assistant.memory.remember("alice", "user", "I live in York", said(), subject="home")
    assert texts(assistant.memory.search("alice", "live")) == ["I live in York"]
    assert [row["id"] for row in assistant.memory.history("alice", new)] == [int(new), int(old)]
    assert [row["id"] for row in assistant.memory.entries("alice")] == [int(new)]
    assert len(assistant.memory.entries("alice", include_superseded=True)) == 2


def test_one_subject_has_one_current_fact_under_concurrent_writes(tmp_path):
    with Assistant(tmp_path) as assistant:
        def write(number: int) -> None:
            assistant.memory.remember("alice", "user", f"value {number}", said(), subject="thing")
            assistant.store.close()

        threads = [threading.Thread(target=write, args=(number,)) for number in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        rows = assistant.store.query("SELECT id FROM memories WHERE subject = 'thing' AND superseded_by IS NULL")
        assert len(rows) == 1
        assert assistant.store.one("SELECT count(*) AS total FROM memories")["total"] == 8


def test_expired_memories_leave_search(assistant, clock):
    assistant.memory.remember("alice", "user", "the code is valid today", said(), expires_at=clock.now + 10)
    assert assistant.memory.search("alice", "code")
    clock.now += 11
    assert assistant.memory.search("alice", "code") == []
    assert assistant.memory.entries("alice") == []


def test_supersede_obeys_the_same_rule(assistant):
    owner = assistant.memory.remember("alice", "user", "Trust me on this", said())
    with pytest.raises(MemoryRefused, match="cannot replace it"):
        assistant.memory.supersede("alice", owner, "A model says otherwise", said(Trust.MODEL))
    assert assistant.memory.get("alice", owner)["superseded_by"] is None
    newer = assistant.memory.supersede("alice", owner, "A newer statement", said())
    assert assistant.memory.get("alice", owner)["superseded_by"] == int(newer)
    assert texts(assistant.memory.search("alice", "statement")) == ["A newer statement"]


def test_a_replaced_memory_cannot_be_replaced_again(assistant):
    old = assistant.memory.remember("alice", "user", "one", said())
    assistant.memory.supersede("alice", old, "two", said())
    with pytest.raises(MemoryRefused, match="already replaced"):
        assistant.memory.supersede("alice", old, "three", said())


# Confirming


def test_the_owner_confirms_a_models_fact(assistant, clock):
    guess = assistant.memory.remember("alice", "semantic", "Alice likes tea", said(Trust.MODEL, author="qwen"))
    assistant.memory.confirm("alice", guess)
    row = assistant.memory.get("alice", guess)
    assert row["trust"] == "owner" and row["confirmed_at"] == clock.now and row["author"] == "qwen"
    assert assistant.memory.search("alice", "tea")[0].provenance.trust is Trust.OWNER


def test_confirming_a_shelved_guess_makes_it_the_current_fact(assistant):
    owner = assistant.memory.remember("alice", "user", "Lives in Leeds", said(), subject="home")
    guess = assistant.memory.remember("alice", "user", "Lives in York", said(Trust.MODEL), subject="home")
    assistant.memory.confirm("alice", guess)
    assert texts(assistant.memory.search("alice", "Lives")) == ["Lives in York"]
    assert assistant.memory.get("alice", owner)["superseded_by"] == int(guess)


@pytest.mark.parametrize("trust", [Trust.EXTERNAL, Trust.OWNER, Trust.VERIFIED])
def test_only_a_models_fact_can_be_confirmed(assistant, trust):
    memory_id = assistant.memory.remember("alice", "user", "Some text", said(trust))
    with pytest.raises(MemoryRefused, match="Only a model"):
        assistant.memory.confirm("alice", memory_id)
    assert assistant.memory.get("alice", memory_id)["trust"] == trust.value


# Forgetting


def database_bytes(assistant) -> bytes:
    path = assistant.store.path
    return b"".join(file.read_bytes() for file in path.parent.glob(path.name + "*"))


def test_forget_is_a_hard_delete_from_the_row_index_and_disk(assistant):
    secret = "xylophonequartz"
    keep = assistant.memory.remember("alice", "user", "A harmless thing to keep around", said())
    memory_id = assistant.memory.remember("alice", "user", f"My secret word is {secret}", said())
    assert assistant.memory.search("alice", secret)
    assert secret.encode() in database_bytes(assistant)

    assistant.memory.forget("alice", memory_id)

    with pytest.raises(MemoryNotFound):
        assistant.memory.get("alice", memory_id)
    assert assistant.memory.search("alice", secret) == []
    assert assistant.store.one("SELECT count(*) AS total FROM memories_fts WHERE memories_fts MATCH ?", (secret,))["total"] == 0
    assert secret.encode() not in database_bytes(assistant)
    assert assistant.memory.get("alice", keep)
    assert texts(assistant.memory.search("alice", "harmless")) == ["A harmless thing to keep around"]


def test_forget_takes_the_older_versions_with_it(assistant):
    first = assistant.memory.remember("alice", "user", "oldaddressword here", said(), subject="address")
    second = assistant.memory.remember("alice", "user", "newaddressword here", said(), subject="address")
    assert assistant.memory.forget("alice", second) == 2
    for memory_id in (first, second):
        with pytest.raises(MemoryNotFound):
            assistant.memory.get("alice", memory_id)
    assert b"oldaddressword" not in database_bytes(assistant)
    assert assistant.memory.remember("alice", "user", "fresh start", said(Trust.MODEL), subject="address")
    assert texts(assistant.memory.search("alice", "fresh")) == ["fresh start"]


def test_the_audit_trail_never_holds_what_was_forgotten(assistant):
    secret = "xylophonequartz"
    memory_id = assistant.memory.remember("alice", "user", f"My word is {secret}", said())
    assistant.memory.forget("alice", memory_id)
    assert [row["action"] for row in assistant.audit.read()] == ["memory.remember", "memory.forget"]
    assert secret not in json.dumps(assistant.audit.read())
    assert assistant.audit.verify() is None


def test_a_whole_conversation_can_be_forgotten(assistant):
    assistant.memory.record_message("alice", "c1", "user", "marmaladepenguin is the password")
    assistant.memory.record_message("alice", "c2", "user", "something else")
    assert assistant.memory.forget_conversation("alice", "c1") == 1
    assert assistant.memory.recent("alice", "c1") == []
    assert assistant.memory.search_messages("alice", "marmaladepenguin") == []
    assert b"marmaladepenguin" not in database_bytes(assistant)
    assert assistant.memory.recent("alice", "c2")


# Searching


def test_search_finds_word_forms(assistant):
    assistant.memory.remember("alice", "user", "Alice walks her dogs every morning", said())
    assert texts(assistant.memory.search("alice", "walking the dog"))


def test_search_prefers_trust_when_relevance_is_equal(assistant):
    assistant.memory.remember("alice", "user", "alpha beta gamma model", said(Trust.MODEL))
    assistant.memory.remember("alice", "user", "alpha beta gamma owner", said(Trust.OWNER))
    assistant.memory.remember("alice", "user", "alpha beta gamma peer", said(Trust.EXTERNAL))
    assistant.memory.remember("alice", "user", "filler one", said())
    assistant.memory.remember("alice", "user", "filler two", said())
    assistant.memory.remember("alice", "user", "filler three", said())
    assert texts(assistant.memory.search("alice", "alpha")) == ["alpha beta gamma owner", "alpha beta gamma model", "alpha beta gamma peer"]


def test_search_prefers_recent_when_trust_is_equal(assistant, clock):
    assistant.memory.remember("alice", "user", "alpha old", said(at=clock.now - 200 * DAY))
    assistant.memory.remember("alice", "user", "alpha new", said(at=clock.now))
    for number in range(4):
        assistant.memory.remember("alice", "user", f"filler {number}", said())
    assert texts(assistant.memory.search("alice", "alpha")) == ["alpha new", "alpha old"]


def test_relevance_still_beats_trust_for_a_much_better_match(assistant):
    assistant.memory.remember("alice", "user", "tomato tomato tomato tomato", said(Trust.MODEL))
    assistant.memory.remember("alice", "user", "a long sentence that mentions a tomato once along with many other unrelated words here", said(Trust.OWNER))
    for number in range(4):
        assistant.memory.remember("alice", "user", f"filler {number}", said())
    assert assistant.memory.search("alice", "tomato")[0].text.startswith("tomato tomato")


@pytest.mark.parametrize(
    "query",
    ['"', "'", "*", "AND", "NOT", "OR", "NEAR(", "a:b", "col:value", "-", "(", ")", "{}", "^", "alpha*", '"unterminated', "", "   ", "\x00", "\U0001f40d"],
)
def test_search_survives_any_text_a_person_types(assistant, query):
    assistant.memory.remember("alice", "user", "alpha", said())
    assert isinstance(assistant.memory.search("alice", query), list)
    assert isinstance(assistant.memory.search_messages("alice", query), list)


def test_common_words_do_not_match_everything(assistant):
    assistant.memory.remember("alice", "user", "My dog is called Rex", said())
    assistant.memory.remember("alice", "user", "My car is a Volvo", said())
    assert texts(assistant.memory.search("alice", "what is my dog called")) == ["My dog is called Rex"]


def test_the_index_follows_a_changed_row(assistant):
    memory_id = assistant.memory.remember("alice", "user", "before", said())
    assistant.store.execute("UPDATE memories SET text = 'after' WHERE id = ?", (int(memory_id),))
    assert assistant.memory.search("alice", "before") == []
    assert texts(assistant.memory.search("alice", "after")) == ["after"]


# Users are separated


def test_one_user_cannot_reach_anothers_memory_by_any_method(assistant):
    secret = assistant.memory.remember("alice", "user", "Alice's private penguin", said(), subject="pet")
    assistant.memory.record_message("alice", "shared-id", "user", "Alice's private penguin")

    assert assistant.memory.search("bob", "penguin") == []
    assert assistant.memory.search_messages("bob", "penguin") == []
    assert assistant.memory.recent("bob", "shared-id") == []
    assert assistant.memory.entries("bob", include_superseded=True) == []
    assert assistant.memory.conversations("bob") == []
    assert assistant.memory.turn_messages("bob", "t") == []
    assert assistant.memory.message("bob", 1) is None
    exported = assistant.memory.export("bob")
    assert exported["memories"] == [] and exported["messages"] == []
    for attempt in (
        lambda: assistant.memory.get("bob", secret),
        lambda: assistant.memory.history("bob", secret),
        lambda: assistant.memory.confirm("bob", secret),
        lambda: assistant.memory.forget("bob", secret),
        lambda: assistant.memory.supersede("bob", secret, "replaced", said(author="bob")),
        lambda: assistant.memory.forget_conversation("bob", "shared-id"),
    ):
        with pytest.raises(MemoryNotFound):
            attempt()
    assert assistant.memory.get("alice", secret)["text"] == "Alice's private penguin"


def test_two_users_can_hold_the_same_subject_and_conversation_id(assistant):
    assistant.memory.remember("alice", "user", "Alice's cat", said(), subject="pet")
    assistant.memory.remember("bob", "user", "Bob's cat", said(Trust.MODEL), subject="pet")
    assistant.memory.record_message("alice", "same", "user", "from alice")
    assistant.memory.record_message("bob", "same", "user", "from bob")
    assert texts(assistant.memory.search("alice", "cat")) == ["Alice's cat"]
    assert texts(assistant.memory.search("bob", "cat")) == ["Bob's cat"]
    assert [row["content"] for row in assistant.memory.recent("bob", "same")] == ["from bob"]


def test_a_missing_or_malformed_id_looks_like_any_other_miss(assistant):
    for memory_id in ("999", "not-a-number", None, ""):
        with pytest.raises(MemoryNotFound):
            assistant.memory.get("alice", memory_id)


def test_every_statement_that_reads_or_changes_user_data_names_the_user(assistant):
    """Traced from the real statements SQLite ran, so a missing clause cannot hide."""
    seen: list[str] = []
    assistant.store.connection.set_trace_callback(seen.append)
    memory = assistant.memory
    first = memory.remember("alice", "user", "alpha one", said(Trust.MODEL), subject="s")
    second = memory.remember("alice", "user", "alpha two", said(), subject="s")
    third = memory.remember("alice", "user", "alpha three", said(Trust.MODEL))
    memory.record_message("alice", "c", "user", "alpha said", turn_id="t")
    memory.search("alice", "alpha")
    memory.search("alice", "alpha", kinds=("user",))
    memory.search_messages("alice", "alpha")
    memory.search_messages("alice", "alpha", conversation_id="c")
    memory.recent("alice", "c")
    memory.turn_messages("alice", "t")
    memory.message("alice", 1)
    memory.conversations("alice")
    memory.entries("alice")
    memory.entries("alice", include_superseded=True)
    memory.get("alice", first)
    memory.history("alice", second)
    memory.supersede("alice", second, "alpha four", said())
    memory.confirm("alice", third)
    memory.export("alice")
    memory.forget("alice", third)
    memory.forget_conversation("alice", "c")
    assistant.store.connection.set_trace_callback(None)

    touching = re.compile(r"\b(FROM|JOIN|UPDATE|INTO)\s+(memories|messages)\b(?!_)", re.IGNORECASE)
    statements = [sql for sql in seen if not sql.lstrip().startswith("--")]
    checked = [sql for sql in statements if touching.search(sql) and not sql.lstrip().upper().startswith("INSERT")]
    assert len(checked) >= 15
    unscoped = [sql for sql in checked if not re.search(r"\buser\s*=\s*", sql)]
    assert unscoped == []


# Commands


def test_commands_remember_recall_confirm_forget_export(tmp_path, capsys):
    assert run_cli(tmp_path, "remember", "My dog is called Rex", "--subject", "dog", "--reach", "machine") == 0
    capsys.readouterr()
    assert run_cli(tmp_path, "recall", "dog") == 0
    shown = capsys.readouterr().out
    assert "Rex" in shown and "owner" in shown and "machine" in shown
    exported = tmp_path / "out.json"
    assert run_cli(tmp_path, "export", str(exported)) == 0
    assert json.loads(exported.read_text(encoding="utf-8"))["memories"][0]["text"] == "My dog is called Rex"
    assert run_cli(tmp_path, "forget", "1") == 0
    capsys.readouterr()
    assert run_cli(tmp_path, "recall", "dog") == 0
    assert "Nothing remembered" in capsys.readouterr().out
