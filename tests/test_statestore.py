"""The saved-state store: its bounds, its tiers, and what survives a process."""

import subprocess
import sys
import threading

import numpy as np
import pytest

from isharded import statestore
from isharded.statestore import HISTORY, PREFIX, StateStore, clear_other_stores, keys_at, prefix_key, stage_namespace


def blob(size, value=1):
    return np.full(size, value, dtype=np.uint8)


def filled(tmp_path, ram=1000, disk=1000):
    return StateStore(tmp_path / "store", ram, disk)


def test_a_key_names_exact_tokens():
    assert prefix_key([1, 2, 3]) == prefix_key([1, 2, 3])
    assert prefix_key([1, 2, 3]) != prefix_key([1, 2, 4])
    assert prefix_key([1, 2]) != prefix_key([1, 2, 0])


def test_a_key_tells_images_apart():
    # An image's rows are negative digests, so another image can never match.
    assert prefix_key([5, -1234, -1234]) != prefix_key([5, -1235, -1235])


def test_keys_at_several_lengths_match_one_at_a_time():
    tokens = list(range(50))
    found = keys_at(tokens, [40, 10, 50, 60, 0])
    assert found == {10: prefix_key(tokens[:10]), 40: prefix_key(tokens[:40]), 50: prefix_key(tokens)}


def test_namespaces_keep_models_and_layers_apart():
    base = stage_namespace("b1", "model", "rev", 0, 14, 0, "f16")
    assert base == stage_namespace("b1", "model", "rev", 0, 14, 0, "f16")
    assert base != stage_namespace("b1", "model", "rev", 14, 28, 0, "f16")
    assert base != stage_namespace("b1", "other", "rev", 0, 14, 0, "f16")
    assert base != stage_namespace("b2", "model", "rev", 0, 14, 0, "f16"), "another build may lay a state out differently"
    assert base != stage_namespace("b1", "model", "rev", 0, 14, 0, "q8_0"), "a state in 8 bits cannot go in a 16-bit cache"


def test_a_new_rule_for_cuts_retires_old_states(monkeypatch):
    """They would still load, but no longer sit on cuts, and replies would quietly stop matching."""
    base = stage_namespace("b1", "model", "rev", 0, 14, 0, "f16")
    monkeypatch.setattr(statestore, "CUT_LEAST", statestore.CUT_LEAST * 2)
    assert base != stage_namespace("b1", "model", "rev", 0, 14, 0, "f16")


def test_a_state_comes_back_from_ram(tmp_path):
    store = filled(tmp_path)
    store.put("a", blob(10, 7), 5, PREFIX)
    data, tier = store.get("a")
    assert tier == "ram" and data.tolist() == [7] * 10
    store.close()


def test_a_state_leaving_ram_comes_back_from_disk(tmp_path):
    store = filled(tmp_path, ram=100, disk=1000)
    store.put("a", blob(80, 1), 5)
    store.flush()
    store.put("b", blob(80, 2), 5)
    store.flush()
    assert store.entries["a"].data is None, "the least recent state leaves RAM first"
    data, tier = store.get("a")
    assert tier == "disk" and data.tolist() == [1] * 80
    store.close()


def test_ram_alone_drops_the_least_recently_used(tmp_path):
    store = StateStore(None, 100, 0)
    store.put("a", blob(40), 1)
    store.put("b", blob(40), 1)
    store.get("a")  # now b is the older one
    store.put("c", blob(40), 1)
    assert store.has("a") and store.has("c") and not store.has("b")
    assert store.stats()["evicted"] == 1


def test_disk_stays_inside_its_budget(tmp_path):
    store = filled(tmp_path, ram=50, disk=100)
    for name in "abcd":
        store.put(name, blob(40), 1)
        store.flush()
    assert store.stats()["disk_bytes"] <= 100
    assert not store.has("a")
    assert sum(path.stat().st_size > 0 for path in (tmp_path / "store").iterdir()) <= 2
    store.close()


def test_a_disk_that_cannot_keep_up_does_not_fill_the_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(statestore, "WRITE_BACKLOG_BYTES", 100)
    store = filled(tmp_path, ram=10, disk=10_000)
    release = threading.Event()
    real_write = store._write
    monkeypatch.setattr(store, "_write", lambda key: (release.wait(10), real_write(key)))
    for index in range(50):
        store.put(f"s{index}", blob(40), 1)
    held = store.stats()
    release.set()
    store.flush()
    assert held["ram_bytes"] <= 100 + 40, "the backlog stays near its limit however much is saved"
    assert held["evicted"] >= 40
    assert store.has("s49"), "the newest state is the one most likely to be wanted"
    store.close()


def test_a_state_too_big_for_either_tier_is_refused(tmp_path):
    store = filled(tmp_path, ram=10, disk=20)
    assert not store.put("a", blob(30), 1)
    assert store.stats()["refused"] == 1 and not store.has("a")
    store.close()


def test_a_state_too_big_for_ram_can_still_live_on_disk(tmp_path):
    store = filled(tmp_path, ram=10, disk=100)
    assert store.put("a", blob(30, 3), 1)
    store.flush()
    assert store.stats()["ram_bytes"] == 0
    assert store.get("a")[0].tolist() == [3] * 30
    store.close()


def test_a_new_process_finds_what_the_last_one_wrote(tmp_path):
    first = filled(tmp_path)
    first.put("a", blob(10, 9), 6, PREFIX)
    first.close()
    second = filled(tmp_path)
    assert second.listing() == [{"key": "a", "length": 6, "kind": PREFIX}]
    assert second.get("a")[0].tolist() == [9] * 10
    second.close()


def test_a_process_killed_mid_write_leaves_nothing_usable_behind(tmp_path):
    folder = tmp_path / "store"
    folder.mkdir()
    (folder / "a.part").write_bytes(b"half a state")
    store = StateStore(folder, 100, 100)
    assert store.listing() == []
    assert not (folder / "a.part").exists()
    store.close()


def test_a_killed_process_keeps_what_reached_disk(tmp_path):
    """No clean shutdown here: the process is gone the moment the write lands."""
    script = (
        "import os, sys, numpy as np\n"
        "from isharded.statestore import StateStore\n"
        "store = StateStore(sys.argv[1], 1000, 1000)\n"
        "store.put('a', np.full(10, 5, dtype=np.uint8), 4)\n"
        "store.flush()\n"
        "os._exit(9)\n"
    )
    folder = tmp_path / "store"
    assert subprocess.run([sys.executable, "-c", script, str(folder)], timeout=60).returncode == 9
    store = StateStore(folder, 1000, 1000)
    assert store.get("a")[0].tolist() == [5] * 10
    store.close()


@pytest.mark.parametrize("damage", [lambda data: data[:-3], lambda data: data[:-1] + bytes([data[-1] ^ 1])])
def test_a_damaged_file_is_a_miss_and_never_a_wrong_state(tmp_path, damage):
    first = filled(tmp_path)
    first.put("a", blob(50, 4), 1)
    first.close()
    path = tmp_path / "store" / "a.state"
    path.write_bytes(damage(path.read_bytes()))
    second = filled(tmp_path)
    assert second.get("a") is None
    assert second.stats()["damaged"] == 1 and not second.has("a")
    second.close()


def test_rubbish_in_the_folder_is_cleared_out(tmp_path):
    folder = tmp_path / "store"
    folder.mkdir()
    (folder / "b.state").write_bytes(b"not one of ours")
    store = StateStore(folder, 100, 100)
    assert store.listing() == [] and store.stats()["damaged"] == 1
    store.close()


def test_a_full_disk_costs_the_copy_and_not_the_state(tmp_path, monkeypatch):
    store = filled(tmp_path)
    real_open = type(store._file("x")).open

    def refuse(path, mode="r", *args, **kwargs):
        if "w" in mode:
            raise OSError("no space left")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(type(store._file("x")), "open", refuse)
    store.put("a", blob(10, 2), 1)
    store.flush()
    assert store.get("a")[0].tolist() == [2] * 10
    assert not store.entries["a"].on_disk
    store.close()


def test_dropping_removes_the_file_too(tmp_path):
    store = filled(tmp_path)
    store.put("a", blob(10), 1)
    store.flush()
    path = store.file_of("a")
    assert path.is_file()
    store.drop("a")
    assert not path.exists() and store.file_of("a") is None
    store.close()


def test_a_state_being_sent_can_still_be_replaced(tmp_path):
    """A copy can take a while, and the next turn may replace that state meanwhile.

    Windows will not delete a file another process has open, and that must never
    reach the reply which is saving the new state.
    """
    store = filled(tmp_path)
    store.put("old", blob(10), 1)
    store.flush()
    path = store.file_of("old")
    with path.open("rb"):  # the copy going out
        store.drop("old")
        assert not store.has("old")
    store.put("new", blob(10), 1)
    store.flush()
    assert not path.exists(), "it goes once nothing holds it any more"
    store.close()


def test_states_leave_the_disk_its_reserve(tmp_path, monkeypatch):
    """States are worth less than weights, so they must never be what stops a model loading."""
    store = filled(tmp_path)
    usage = type("Usage", (), {"free": statestore.DISK_RESERVE_BYTES + 5})
    monkeypatch.setattr(statestore.shutil, "disk_usage", lambda path: usage)
    store.put("a", blob(10, 6), 1)
    store.flush()
    assert store.file_of("a") is None
    assert store.stats()["disk_full"] == 1
    assert store.get("a")[0].tolist() == [6] * 10, "it stays in RAM, and a reply never notices"
    store.close()


def test_a_disk_with_room_takes_states_as_usual(tmp_path, monkeypatch):
    store = filled(tmp_path)
    usage = type("Usage", (), {"free": statestore.DISK_RESERVE_BYTES + 50})
    monkeypatch.setattr(statestore.shutil, "disk_usage", lambda path: usage)
    store.put("a", blob(10), 1)
    store.flush()
    assert store.file_of("a") is not None
    store.close()


def test_one_bad_write_does_not_stop_the_ones_after_it(tmp_path, monkeypatch):
    store = filled(tmp_path)
    real_write = store._write
    calls = []

    def fragile(key):
        calls.append(key)
        if len(calls) == 1:
            raise RuntimeError("something nobody thought of")
        real_write(key)

    monkeypatch.setattr(store, "_write", fragile)
    store.put("a", blob(10), 1)
    store.put("b", blob(10), 1)
    store.flush()
    assert store.file_of("b") is not None
    assert store.stats()["write_failures"] == 1
    store.close()


def test_a_state_being_sent_can_still_be_evicted(tmp_path):
    store = filled(tmp_path, ram=10, disk=25)
    store.put("old", blob(10), 1)
    store.flush()
    with store.file_of("old").open("rb"):
        store.put("b", blob(10), 1)
        store.flush()
        store.put("c", blob(10), 1)
        store.flush()
    assert store.stats()["disk_bytes"] <= 25
    store.close()


def test_wiping_empties_both_tiers(tmp_path):
    store = filled(tmp_path)
    store.put("a", blob(10), 1)
    store.put("b", blob(10), 1)
    store.flush()
    store.wipe()
    assert store.listing() == []
    assert not list((tmp_path / "store").iterdir())
    store.close()


def test_saving_the_same_state_again_writes_nothing(tmp_path):
    store = filled(tmp_path)
    store.put("a", blob(10), 1)
    store.put("a", blob(10), 1)
    assert store.stats()["saved"] == 1
    store.close()


def test_stats_say_what_is_held_where(tmp_path):
    store = filled(tmp_path, ram=100, disk=200)
    store.put("a", blob(60), 1, HISTORY)
    store.flush()
    stats = store.stats()
    assert (stats["states"], stats["ram_bytes"], stats["disk_bytes"]) == (1, 60, 60)
    assert (stats["ram_budget_bytes"], stats["disk_budget_bytes"]) == (100, 200)
    store.close()


def test_old_stores_are_cleared_when_the_model_or_layers_change(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "a.state").write_bytes(b"x")
    (tmp_path / "current").mkdir()
    clear_other_stores(tmp_path, {"current"})
    assert [path.name for path in tmp_path.iterdir()] == ["current"]


def test_a_state_from_another_node_is_checked_before_it_is_kept(tmp_path):
    source = filled(tmp_path / "a")
    source.put("a", blob(50, 8), 12, PREFIX)
    source.flush()
    arrived = tmp_path / "arrived.part"
    arrived.write_bytes(source.file_of("a").read_bytes())
    holder = filled(tmp_path / "b")
    assert holder.adopt("copy", arrived)
    assert not arrived.exists()
    assert holder.listing() == [{"key": "copy", "length": 12, "kind": PREFIX}]
    assert holder.get("copy")[0].tolist() == [8] * 50
    source.close()
    holder.close()


@pytest.mark.parametrize("content", [b"", b"not a state", b"ISHSTATE" + b"\x01" * 40])
def test_a_bad_copy_is_refused_and_leaves_nothing_behind(tmp_path, content):
    holder = filled(tmp_path)
    arrived = tmp_path / "arrived.part"
    arrived.write_bytes(content)
    assert not holder.adopt("copy", arrived)
    assert holder.listing() == [] and not arrived.exists()
    holder.close()


def test_a_copy_cut_short_on_the_way_is_refused(tmp_path):
    source = filled(tmp_path / "a")
    source.put("a", blob(50), 12)
    source.flush()
    arrived = tmp_path / "arrived.part"
    arrived.write_bytes(source.file_of("a").read_bytes()[:-5])
    holder = filled(tmp_path / "b")
    assert not holder.adopt("copy", arrived)
    source.close()
    holder.close()


def test_the_stage_process_can_use_this_without_torch():
    script = "import sys, isharded.statestore; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", script], timeout=120).returncode == 0
