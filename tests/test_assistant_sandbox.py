"""The claims about processes, tested on real processes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import threading
import time

import psutil
import pytest

from shardmesh.assistant import cli
from shardmesh.assistant import sandbox as sandbox_part
from shardmesh.assistant.app import Assistant
from shardmesh.assistant.sandbox import (
    LocalSandbox,
    NETWORK_NOT_BLOCKED,
    ProcessTree,
    bubblewrap_command,
    container_command,
    scrubbed_env,
    unshare_command,
    with_rlimits,
)

PYTHON = sys.executable


def python(script: str) -> list[str]:
    return [PYTHON, "-c", textwrap.dedent(script)]


def wait_for(condition, seconds=20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@pytest.fixture
def sandbox():
    return LocalSandbox(prefer="process", memory_mb=512)


FAMILY = """
    import os, subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open("pids.tmp", "w") as out:
        out.write(f"{os.getpid()} {child.pid}")
    os.replace("pids.tmp", "pids.txt")
    time.sleep(120)
"""


def family_pids(folder: Path) -> list[int]:
    return [int(pid) for pid in (folder / "pids.txt").read_text().split()]


def test_a_command_runs_and_reports_its_exit_code(sandbox, tmp_path):
    result = sandbox.run(python("import sys; print('hi'); print('oops', file=sys.stderr); sys.exit(3)"), tmp_path)
    assert result.exit_code == 3
    assert result.stdout.strip() == "hi"
    assert "oops" in result.stderr
    assert not result.timed_out
    assert result.level == "process"


def test_it_runs_in_the_workdir(sandbox, tmp_path):
    result = sandbox.run(python("import os; print(os.getcwd())"), tmp_path)
    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


def test_children_die_on_timeout(sandbox, tmp_path):
    started = time.monotonic()
    result = sandbox.run(python(FAMILY), tmp_path, timeout=4)
    assert result.timed_out and result.exit_code is None
    assert time.monotonic() - started < 15
    assert (tmp_path / "pids.txt").exists(), "the command never got as far as starting its child"
    parent, child = family_pids(tmp_path)
    assert wait_for(lambda: gone(parent) and gone(child)), "something outlived the time limit"


def test_a_child_left_behind_by_a_finished_command_dies_too(sandbox, tmp_path):
    script = """
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open("child.txt", "w").write(str(child.pid))
    """
    result = sandbox.run(python(script), tmp_path, timeout=30)
    assert result.exit_code == 0
    child = int((tmp_path / "child.txt").read_text())
    assert wait_for(lambda: gone(child)), "a background child outlived its command"


def test_a_secret_in_the_parents_environment_is_invisible(sandbox, tmp_path, monkeypatch):
    monkeypatch.setenv("SHARDMESH_POOL_TOKEN", "hunter2-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "hunter2-aws")
    monkeypatch.setenv("SOMETHING_NOBODY_THOUGHT_OF", "hunter2-other")
    script = """
        import json, os, subprocess, sys
        grandchild = subprocess.run([sys.executable, "-c", "import os, json; print(json.dumps(dict(os.environ)))"], capture_output=True, text=True)
        print(json.dumps({"child": dict(os.environ), "grandchild": json.loads(grandchild.stdout)}))
    """
    result = sandbox.run(python(script), tmp_path)
    assert result.exit_code == 0, result.stderr
    seen = json.loads(result.stdout)
    assert "hunter2" not in result.stdout
    assert "SHARDMESH_POOL_TOKEN" not in seen["child"] and "SHARDMESH_POOL_TOKEN" not in seen["grandchild"]


def test_the_environment_is_an_allow_list(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAND_NEW_VARIABLE", "x")
    environment = scrubbed_env(tmp_path)
    assert "BRAND_NEW_VARIABLE" not in environment
    assert environment["HOME"] == str(tmp_path)
    assert environment["TEMP"] == str(tmp_path)


def test_a_variable_passed_on_purpose_arrives(sandbox, tmp_path):
    result = sandbox.run(python("import os; print(os.environ['WANTED'])"), tmp_path, env={"WANTED": "yes"})
    assert result.stdout.strip() == "yes"


def test_home_is_not_the_users_home(sandbox, tmp_path):
    result = sandbox.run(python("import os; print(os.path.expanduser('~'))"), tmp_path)
    assert Path(result.stdout.strip()) != Path.home()


def test_output_is_capped_and_a_flood_does_not_block_the_command(tmp_path):
    capped = LocalSandbox(prefer="process", output_bytes=1000)
    result = capped.run(python("import sys\nfor _ in range(2000): sys.stdout.write('x' * 1000)\nsys.stdout.flush()"), tmp_path, timeout=60)
    assert result.exit_code == 0 and not result.timed_out
    assert result.stdout.startswith("x" * 1000)
    assert len(result.stdout) < 1100
    assert "output cut" in result.stdout


def test_the_memory_limit_is_real(tmp_path):
    small = LocalSandbox(prefer="process", memory_mb=64)
    result = small.run(python("data = bytearray(1024 * 1024 * 1024); print('allocated', len(data))"), tmp_path, timeout=60)
    assert result.exit_code != 0
    assert "allocated" not in result.stdout
    assert "MemoryError" in result.stderr


def test_a_program_that_is_missing_is_a_result_not_a_crash(sandbox, tmp_path):
    result = sandbox.run(["no-such-program-anywhere"], tmp_path)
    assert result.exit_code == 127
    assert "no-such-program-anywhere" in result.stderr


def test_a_missing_folder_is_a_result_not_a_crash(sandbox, tmp_path):
    result = sandbox.run(python("print(1)"), tmp_path / "missing")
    assert result.exit_code == 126 and "not a folder" in result.stderr


def test_an_empty_command_is_refused(sandbox, tmp_path):
    with pytest.raises(ValueError, match="empty"):
        sandbox.run([], tmp_path)


def test_stop_ends_a_running_command_and_its_children(sandbox, tmp_path):
    results = []
    thread = threading.Thread(target=lambda: results.append(sandbox.run(python(FAMILY), tmp_path, timeout=120, label="job-1")))
    thread.start()
    assert wait_for(lambda: (tmp_path / "pids.txt").exists()), "the command never started"
    assert sandbox.stop("someone-else") == 0
    assert sandbox.stop("job-1") == 1
    thread.join(20)
    assert not thread.is_alive()
    assert results[0].stopped and not results[0].timed_out and results[0].exit_code is None
    parent, child = family_pids(tmp_path)
    assert wait_for(lambda: gone(parent) and gone(child))


def test_network_refusal_that_cannot_be_enforced_is_stated(sandbox, tmp_path):
    result = sandbox.run(python("print('ok')"), tmp_path, network=False)
    assert result.level == "process"
    assert result.network_blocked is False
    assert NETWORK_NOT_BLOCKED in result.notes
    assert "not enforced" in result.stderr  # also reaches a reader who knows only the contract


def test_the_statement_is_true_the_process_level_really_can_reach_the_network(sandbox, tmp_path):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    script = f"import socket; socket.create_connection(('127.0.0.1', {port}), 10).close(); print('connected')"
    try:
        result = sandbox.run(python(script), tmp_path, network=False, timeout=30)
    finally:
        listener.close()
    assert "connected" in result.stdout


def test_asking_for_the_network_needs_no_warning(sandbox, tmp_path):
    result = sandbox.run(python("print('ok')"), tmp_path, network=True)
    assert NETWORK_NOT_BLOCKED not in result.notes
    assert result.stderr == ""


def test_the_process_level_says_it_runs_as_this_user(sandbox, tmp_path):
    assert any("same" in note or "this user" in note for note in sandbox.run(python("pass"), tmp_path).notes)


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are a Windows feature")
def test_a_child_cannot_break_out_of_the_job(sandbox, tmp_path):
    script = """
        import subprocess, sys
        try:
            subprocess.Popen([sys.executable, "-c", "pass"], creationflags=0x01000000)  # CREATE_BREAKAWAY_FROM_JOB
            print("escaped")
        except OSError:
            print("refused")
    """
    result = sandbox.run(python(script), tmp_path, timeout=30)
    assert result.stdout.strip() == "refused"


@pytest.mark.skipif(sys.platform != "win32", reason="Linux at the process level has no kill-on-close, which the guide states")
def test_a_killed_parent_takes_the_sandbox_with_it(tmp_path):
    """Kill-on-close: a crash here must not leave generated code running."""
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        textwrap.dedent(
            f"""
            import sys, textwrap
            sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
            from pathlib import Path
            from shardmesh.assistant.sandbox import LocalSandbox
            LocalSandbox(prefer="process").run(
                [sys.executable, "-c", textwrap.dedent({FAMILY!r})], Path({str(tmp_path)!r}), timeout=120
            )
            """
        ),
        encoding="utf-8",
    )
    parent = subprocess.Popen([PYTHON, str(parent_script)])
    try:
        assert wait_for(lambda: (tmp_path / "pids.txt").exists(), 30), "the sandboxed command never started"
        pids = family_pids(tmp_path)
        assert not any(gone(pid) for pid in pids)
        parent.kill()  # TerminateProcess
        parent.wait(10)
        assert wait_for(lambda: all(gone(pid) for pid in pids)), "generated code outlived a killed parent"
    finally:
        parent.kill()


def test_process_tree_can_talk_over_stdin(tmp_path):
    tree = ProcessTree(python("print(input().upper())"), tmp_path, scrubbed_env(tmp_path), stdin=True)
    try:
        tree.process.stdin.write(b"hello\n")
        tree.process.stdin.close()
        assert tree.process.stdout.read().strip() == b"HELLO"
    finally:
        tree.close()


# Which walls.


def test_asking_for_a_level_that_does_not_exist_is_an_error():
    with pytest.raises(ValueError, match="must be auto"):
        LocalSandbox(prefer="magic")


def test_detection_reports_why_it_got_less():
    chosen = LocalSandbox()
    report = chosen.describe()
    assert report["level"] in ("container", "namespace", "process")
    if report["level"] == "process":
        assert report["weaker_because"], "falling to the weakest level must say why"


def test_a_demanded_level_that_is_missing_refuses_rather_than_runs_weaker(tmp_path, monkeypatch):
    demanding = LocalSandbox(prefer="container")
    monkeypatch.setattr(demanding, "_engine_problem", lambda engine: f"{engine} is not here.")
    result = demanding.run(python("open('ran.txt', 'w')"), tmp_path)
    assert result.level == "none" and result.exit_code == 126
    assert "not available" in result.stderr
    assert not (tmp_path / "ran.txt").exists()


def test_a_container_engine_with_no_service_is_passed_over(monkeypatch):
    chooser = LocalSandbox()
    monkeypatch.setattr(sandbox_part.shutil, "which", lambda name: f"/bin/{name}" if name == "docker" else None)
    monkeypatch.setattr(sandbox_part, "quiet", lambda command, timeout=8: False)
    assert "not running" in chooser._engine_problem("docker")


def test_a_container_engine_with_no_local_image_is_passed_over_and_nothing_is_pulled(monkeypatch):
    chooser = LocalSandbox(image="nothing:here")
    seen = []

    def fake_quiet(command, timeout=8):
        seen.append(list(command))
        return command[1] == "info"

    monkeypatch.setattr(sandbox_part.shutil, "which", lambda name: "/bin/docker")
    monkeypatch.setattr(sandbox_part, "quiet", fake_quiet)
    assert "no local image nothing:here" in chooser._engine_problem("docker")
    assert not any("pull" in command for command in seen)


def test_the_container_command_has_every_wall():
    argv = container_command("docker", "img:1", "n1", ["python", "x.py"], "/some/work", False, ["WANTED"], 256)
    assert argv[:2] == ["docker", "run"]
    assert "--read-only" in argv and "--rm" in argv and "--pull=never" in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--memory") + 1] == "256m"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in argv
    assert argv[argv.index("--env") + 1] == "WANTED"  # by name, so the value is not in a process list
    assert argv[-3:] == ["img:1", "python", "x.py"]


def test_a_container_can_be_given_the_network_on_request():
    argv = container_command("podman", "img:1", "n1", ["true"], "/w", True, [], 256)
    assert argv[argv.index("--network") + 1] == "bridge"


def test_the_bubblewrap_command_has_every_wall():
    argv = bubblewrap_command(["python", "x.py"], "/work", False)
    assert "--unshare-all" in argv and "--die-with-parent" in argv and "--share-net" not in argv
    assert argv[argv.index("--ro-bind") + 1 : argv.index("--ro-bind") + 3] == ["/", "/"]
    assert argv[argv.index("--bind") + 1 : argv.index("--bind") + 3] == ["/work", "/work"]
    assert "/home" in argv and argv[-3:] == ["--", "python", "x.py"]
    assert "--share-net" in bubblewrap_command(["true"], "/work", True)


def test_the_unshare_command_blocks_the_network_unless_asked():
    assert "--net" in unshare_command(["true"], False)
    assert "--net" not in unshare_command(["true"], True)
    assert "--kill-child" in unshare_command(["true"], False)


def test_rlimits_wrap_the_command_without_a_shell_string():
    argv = with_rlimits(["python", "a b.py"], 100)
    assert argv[:2] == ["/bin/sh", "-c"]
    assert argv[-2:] == ["python", "a b.py"]  # stays separate words, so nothing is reparsed
    assert str(100 * 1024) in argv


@pytest.mark.skipif(LocalSandbox().level != "namespace", reason="no namespace support on this machine")
def test_a_namespace_really_blocks_the_network(tmp_path):
    made = LocalSandbox(prefer="namespace")
    script = "import socket; socket.create_connection(('1.1.1.1', 53), 3)"
    result = made.run(python(script), tmp_path, network=False, timeout=30)
    assert result.level == "namespace" and result.network_blocked
    assert result.exit_code != 0


# As a part.


def test_it_attaches_without_starting_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_part, "quiet", lambda *a, **k: pytest.fail("attach must not probe"))
    with Assistant(tmp_path / "home") as assistant:
        assert isinstance(assistant.sandbox, LocalSandbox)


def test_each_run_is_audited_without_the_command_text(tmp_path):
    with Assistant(tmp_path / "home") as assistant:
        assistant.sandbox = LocalSandbox(prefer="process", audit=assistant.audit)
        assistant.sandbox.run(python("print('classified')"), tmp_path)
        rows = assistant.audit.read(actor="sandbox")
    assert rows[0]["subject"] == "process" and rows[0]["detail"]["exit_code"] == 0
    assert rows[0]["detail"]["network_blocked"] is False
    assert "classified" not in json.dumps(rows)


def test_the_command_says_what_level_this_machine_got(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SHARDMESH_SANDBOX", "process")
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    assert cli.run(parser.parse_args(["--home", str(tmp_path / "home"), "sandbox"])) == 0
    shown = capsys.readouterr().out
    assert "Level: process" in shown and "cannot be blocked" in shown


def test_stopping_the_assistant_stops_what_is_running(tmp_path):
    results = []
    with Assistant(tmp_path / "home") as assistant:
        assistant.sandbox = LocalSandbox(prefer="process")
        thread = threading.Thread(target=lambda: results.append(assistant.sandbox.run(python(FAMILY), tmp_path, timeout=120)))
        thread.start()
        assert wait_for(lambda: (tmp_path / "pids.txt").exists())
    thread.join(20)
    assert results and results[0].stopped
    assert all(wait_for(lambda pid=pid: gone(pid)) for pid in family_pids(tmp_path))


def test_a_setting_chooses_the_limits(monkeypatch):
    monkeypatch.setenv("SHARDMESH_SANDBOX_MEMORY_MB", "333")
    monkeypatch.setenv("SHARDMESH_SANDBOX_OUTPUT_BYTES", "4444")
    monkeypatch.setenv("SHARDMESH_SANDBOX_IMAGE", "mine:1")
    made = LocalSandbox()
    assert (made.memory_mb, made.output_bytes, made.image) == (333, 4444, "mine:1")


def test_the_runs_table_empties_after_each_run(sandbox, tmp_path):
    sandbox.run(python("pass"), tmp_path)
    assert sandbox.running == {}
