"""Supervised coding, on real git repositories, with tests that really fail and get fixed.

The sandbox here is a plain subprocess that implements contracts.Sandbox. The claims
tested are about what the coder hands to it, and what it does around it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from isharded.assistant.app import Assistant
from isharded.assistant.cli import add_arguments
from isharded.assistant.coder import (
    COMMITTED,
    PROPOSING,
    PUSHED,
    READY,
    REJECTED,
    STOPPED,
    CoderError,
    DiffChange,
    FileChange,
    UnusableChange,
    detect_languages,
    detect_test_command,
    parse_changes,
    safe_path,
    slug_of,
)
from isharded.assistant.contracts import (
    Capability,
    Decision,
    Effect,
    ModelReply,
    Provenance,
    Reach,
    SandboxResult,
    ToolResult,
    Trust,
    Verdict,
)

ROOT = Path(__file__).resolve().parents[1]
BROKEN = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
FIX_AS_FILE = f"<<<FILE calc.py\n{FIXED}>>>FILE\n"
FIX_AS_DIFF = "<<<DIFF\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n>>>DIFF\n"


def git(folder: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *arguments], cwd=folder, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path) -> Path:
    """A real repository with a real failing test, on a default branch called main."""
    folder = tmp_path / "project"
    folder.mkdir()
    git(folder, "init", "-q", "-b", "main")
    git(folder, "config", "core.autocrlf", "false")
    (folder / "calc.py").write_bytes(BROKEN.encode())
    (folder / "test_calc.py").write_bytes(TEST.encode())
    (folder / "pytest.ini").write_bytes(b"[pytest]\naddopts = -p no:cacheprovider\n")
    git(folder, "add", "-A")
    git(folder, "commit", "-q", "-m", "A broken adder")
    return folder


class ProcessSandbox:
    """contracts.Sandbox as a child process, keeping a record of what it was given."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.before: Any = None

    def run(self, command: Any, workdir: Path, *, timeout: float = 120.0, network: bool = False, env: dict[str, str] | None = None) -> SandboxResult:
        listing = sorted(path.relative_to(workdir).as_posix() for path in Path(workdir).rglob("*") if path.is_file())
        self.calls.append({"command": list(command), "workdir": Path(workdir), "network": network, "files": listing, "env": dict(env or {})})
        if self.before:
            self.before(self)
        try:
            done = subprocess.run(command, cwd=workdir, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
        except subprocess.TimeoutExpired:
            return SandboxResult(None, "", "", True, "process")
        return SandboxResult(done.returncode, done.stdout, done.stderr, False, "process")


class ScriptedRouter:
    """Replies in turn for each role, and keeps what it was asked."""

    def __init__(self, coding: list[Any] | None = None, verification: list[Any] | None = None) -> None:
        self.replies = {"coding": list(coding or []), "verification": list(verification or ['{"verdict": "accept", "notes": "fine"}'])}
        self.requests: list[Any] = []

    def complete(self, request: Any) -> ModelReply:
        self.requests.append(request)
        queue = self.replies[request.role]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return ModelReply(answer, (), f"{request.role}-model", "local", 10, 20, 0.1)


class GitBroker:
    """contracts.Broker: deny by default, and git.commit and git.push are real git."""

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self.grants: set[tuple[str, str]] = set()
        self.calls: list[tuple[str, dict[str, Any], Any]] = []
        self.ask: set[str] = set()
        self.schema = schema or {"type": "object", "properties": {"repo": {}, "message": {}, "branch": {}, "remote": {}}}

    def capabilities(self) -> list[Capability]:
        return [
            Capability("git.commit", "commit", frozenset({Effect.WRITE_LOCAL}), input_schema=self.schema),
            Capability("git.push", "push", frozenset({Effect.EXTERNAL}), input_schema=self.schema),
        ]

    def register(self, capability: Any, handler: Any) -> None:
        pass

    def check(self, call: Any, context: Any) -> Decision:
        if call.capability in self.ask:
            return Decision(Verdict.ASK, "a person must approve", approval_id="approval-7")
        if (context.user, call.capability) not in self.grants:
            return Decision(Verdict.DENY, f"no grant for {call.capability}")
        return Decision(Verdict.ALLOW, "granted")

    def invoke(self, call: Any, context: Any) -> ToolResult:
        assert self.check(call, context).verdict is Verdict.ALLOW
        self.calls.append((call.capability, dict(call.arguments), context))
        arguments = call.arguments
        folder = Path(arguments.get("repo") or arguments["path"])
        if call.capability == "git.commit":
            git(folder, "commit", "-q", "-m", arguments["message"])
        else:
            git(folder, "push", "-q", arguments["remote"], arguments["branch"])
        return ToolResult(True, "done", Provenance("tool", "git", "git", Trust.EXTERNAL, 0.0))


@pytest.fixture
def assistant(tmp_path):
    with Assistant(tmp_path / "home") as built:
        built.sandbox = ProcessSandbox()
        built.router = ScriptedRouter([FIX_AS_FILE])
        built.broker = GitBroker()
        built.broker.grants.add(("owner", "git.commit"))
        yield built


def snapshot(repo: Path) -> dict[str, Any]:
    """Everything of the owner's that must not change."""
    return {
        "head": git(repo, "rev-parse", "HEAD"),
        "branch": git(repo, "symbolic-ref", "HEAD"),
        "main": git(repo, "rev-parse", "refs/heads/main"),
        "status": git(repo, "status", "--porcelain"),
        "files": {path.name: path.read_bytes() for path in repo.iterdir() if path.is_file()},
    }


# The main path


def test_a_real_failing_test_gets_fixed(assistant, repo):
    before = snapshot(repo)
    run = assistant.coder.start(repo, "Fix add so test_add passes")

    assert run.state == READY
    assert run.branch == "assistant/fix-add-so-test-add-passes"
    assert run.languages == ("Python",)
    assert run.test_command == ("python", "-m", "pytest", "-q")

    tree = run.tree
    assert tree.is_relative_to(assistant.home)
    assert (tree / "calc.py").read_bytes() == FIXED.encode()
    assert snapshot(repo) == before

    tests = assistant.coder.show(run.id)["tests"]
    assert [(test["round"], test["exit_code"] == 0) for test in tests] == [(0, False), (1, True)]
    assert "assert" in tests[0]["output"]  # the baseline really failed, and for the reason expected


def test_the_diff_form_works_too(assistant, repo):
    assistant.router = ScriptedRouter([FIX_AS_DIFF])
    run = assistant.coder.start(repo, "Fix add")
    assert run.state == READY
    assert (run.tree / "calc.py").read_bytes() == FIXED.encode()


def test_diffs_apply_when_the_owners_git_rewrites_line_endings(assistant, repo):
    """This machine's global setting is autocrlf, which breaks LF diffs if it is let in."""
    git(repo, "config", "core.autocrlf", "true")
    assistant.router = ScriptedRouter([FIX_AS_DIFF])
    run = assistant.coder.start(repo, "Fix add")
    assert run.state == READY
    assert b"\r" not in (run.tree / "calc.py").read_bytes()


def test_failures_go_back_to_the_model(assistant, repo):
    wrong = "<<<FILE calc.py\ndef add(a, b):\n    return a * b\n>>>FILE\n"
    assistant.router = ScriptedRouter([wrong, FIX_AS_FILE])
    run = assistant.coder.start(repo, "Fix add")
    assert run.state == READY and run.round == 2
    second = [request for request in assistant.router.requests if request.role == "coding"][1]
    prompt = second.messages[1]["content"]
    assert "The tests failed" in prompt and "assert 6 == 5" in prompt
    assert "return a * b" in prompt  # it sees where the last round left the code


def test_it_stops_at_the_limit_and_reports(assistant, repo):
    wrong = "<<<FILE calc.py\ndef add(a, b):\n    return 0\n>>>FILE\n"
    assistant.router = ScriptedRouter([wrong])
    run = assistant.coder.start(repo, "Fix add", max_rounds=2)
    assert run.state == STOPPED and run.round == 2
    assert "Stopped after 2 round(s)" in run.problem
    assert [test["round"] for test in assistant.coder.show(run.id)["tests"]] == [0, 1, 2]
    with pytest.raises(CoderError, match="Only a ready run"):
        assistant.coder.approve(run.id)


def test_a_stopped_run_can_be_given_more_rounds(assistant, repo):
    wrong = "<<<FILE calc.py\ndef add(a, b):\n    return 0\n>>>FILE\n"
    assistant.router = ScriptedRouter([wrong, FIX_AS_FILE])
    run = assistant.coder.start(repo, "Fix add", max_rounds=1)
    assert run.state == STOPPED
    with pytest.raises(CoderError, match="--rounds"):
        assistant.coder.resume(run.id)
    assert assistant.coder.resume(run.id, rounds=1).state == READY


def test_the_review_comes_after_the_tests(assistant, repo):
    assistant.coder.start(repo, "Fix add")
    roles = [request.role for request in assistant.router.requests]
    assert roles == ["coding", "verification"]


def test_a_reviewer_can_send_it_back(assistant, repo):
    better = f"<<<FILE calc.py\n# Adds two numbers.\n{FIXED}>>>FILE\n"
    assistant.router = ScriptedRouter([FIX_AS_FILE, better], ['{"verdict": "changes", "notes": "add a comment"}', '{"verdict": "accept"}'])
    run = assistant.coder.start(repo, "Fix add")
    assert run.state == READY and run.round == 2
    assert b"# Adds two numbers" in (run.tree / "calc.py").read_bytes()
    assert "add a comment" in [request for request in assistant.router.requests if request.role == "coding"][1].messages[1]["content"]


def test_a_review_decides_nothing_when_rounds_run_out(assistant, repo):
    assistant.router = ScriptedRouter([FIX_AS_FILE], ['{"verdict": "changes", "notes": "I dislike this"}'])
    run = assistant.coder.start(repo, "Fix add", max_rounds=1)
    assert run.state == READY
    assert run.review == {"verdict": "changes", "notes": "I dislike this"}


def test_show_gives_the_diff_tests_and_models(assistant, repo):
    run = assistant.coder.start(repo, "Fix add")
    shown = assistant.coder.show(run.id)
    assert "-    return a - b" in shown["diff"] and "+    return a + b" in shown["diff"]
    assert {(row["role"], row["model"]) for row in shown["models"]} == {("coding", "coding-model"), ("verification", "verification-model")}
    assert all(test["level"] == "process" for test in shown["tests"])


# The owner's decisions


def test_approving_commits_to_the_branch_through_the_broker(assistant, repo):
    before = snapshot(repo)
    run = assistant.coder.approve(assistant.coder.start(repo, "Fix add").id)

    assert run.state == COMMITTED
    capability, arguments, context = assistant.broker.calls[0]
    assert capability == "git.commit"
    assert arguments["repo"] == str(run.tree)
    assert arguments["branch"] == run.branch
    assert context.user == "owner"
    assert "Models: coding: coding-model" in arguments["message"]

    assert git(repo, "rev-parse", run.branch) == run.commit_id
    assert "return a + b" in git(repo, "show", f"{run.branch}:calc.py")
    assert snapshot(repo) == before  # main and the checkout never moved


def test_a_commit_the_broker_denies_does_not_happen(assistant, repo):
    assistant.broker.grants.clear()
    run = assistant.coder.start(repo, "Fix add")
    with pytest.raises(CoderError, match="refused git.commit"):
        assistant.coder.approve(run.id)
    assert assistant.coder.run(run.id).state == READY
    assert git(run.tree, "rev-parse", "HEAD") == run.base


def test_a_commit_that_is_not_what_was_tested_is_reported(assistant, repo):
    """A hook or a tool could change things on the way in. The owner must hear about it."""
    run = assistant.coder.start(repo, "Fix add")
    real_invoke = assistant.broker.invoke

    def tampering(call: Any, context: Any) -> Any:
        (run.tree / "extra.py").write_bytes(b"added = 'on the way in'\n")
        git(run.tree, "add", "-A")
        return real_invoke(call, context)

    assistant.broker.invoke = tampering
    done = assistant.coder.approve(run.id)
    assert done.state == COMMITTED
    assert "not the diff that was tested" in done.problem
    assert [row for row in assistant.audit.read() if row["action"] == "coder.committed"][-1]["detail"]["as_tested"] is False


def test_a_clean_commit_reports_no_problem(assistant, repo):
    run = assistant.coder.approve(assistant.coder.start(repo, "Fix add").id)
    assert run.problem == ""


def test_the_owners_other_worktrees_are_left_alone(assistant, repo, tmp_path):
    """One on a drive that is not plugged in looks missing, and pruning would forget it."""
    other = tmp_path / "other-worktree"
    git(repo, "worktree", "add", "-q", "-b", "owners-work", str(other))
    away = tmp_path / "unplugged"
    other.rename(away)
    run = assistant.coder.start(repo, "Fix add")
    assistant.coder.reject(run.id)
    assert "other-worktree" in git(repo, "worktree", "list")


def test_a_worktree_that_cannot_be_removed_says_so(assistant, repo, monkeypatch):
    run = assistant.coder.start(repo, "Fix add")
    real = assistant.coder._git

    def stuck(folder: Any, *arguments: str, **options: Any) -> Any:
        if arguments[:2] == ("worktree", "remove"):
            return subprocess.CompletedProcess(arguments, 1, b"", b"file in use")
        return real(folder, *arguments, **options)

    monkeypatch.setattr(assistant.coder, "_git", stuck)
    with pytest.raises(CoderError, match="file in use"):
        assistant.coder.reject(run.id)
    assert assistant.coder.run(run.id).state == READY


def test_git_missing_says_so(assistant, repo, monkeypatch):
    def nothing(*_: Any, **__: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", nothing)
    with pytest.raises(CoderError, match="git is not installed"):
        assistant.coder.start(repo, "Fix add")


def test_a_commit_needing_approval_waits_and_can_be_retried(assistant, repo):
    assistant.broker.ask.add("git.commit")
    run = assistant.coder.approve(assistant.coder.start(repo, "Fix add").id)
    assert run.state == READY and run.approval_id == "approval-7"
    assistant.broker.ask.clear()
    assert assistant.coder.approve(run.id).state == COMMITTED


def test_nothing_is_committed_without_a_broker(tmp_path, repo):
    with Assistant(tmp_path / "bare") as bare:
        bare.sandbox = ProcessSandbox()
        bare.router = ScriptedRouter([FIX_AS_FILE])
        bare.broker = None
        run = bare.coder.start(repo, "Fix add")
        with pytest.raises(CoderError, match="no broker"):
            bare.coder.approve(run.id)


def test_a_change_after_the_tests_passed_cannot_be_committed(assistant, repo):
    run = assistant.coder.start(repo, "Fix add")
    (run.tree / "calc.py").write_bytes(b"def add(a, b):\n    return 'stolen'\n")
    with pytest.raises(CoderError, match="not what was shown"):
        assistant.coder.approve(run.id)
    assert assistant.broker.calls == []


@pytest.mark.parametrize("actor", ["skill:sk_1", "peer:abc", "task:9", "model", "model:x", "plugin:weather", "conversation"])
def test_only_the_owner_can_approve_push_or_reject(assistant, repo, actor):
    run = assistant.coder.start(repo, "Fix add")
    with pytest.raises(CoderError, match="Only the owner"):
        assistant.coder.approve(run.id, by=actor)
    with pytest.raises(CoderError, match="Only the owner"):
        assistant.coder.push(run.id, by=actor)
    with pytest.raises(CoderError, match="Only the owner"):
        assistant.coder.reject(run.id, by=actor)
    assert assistant.coder.run(run.id).state == run.state and run.tree.exists()


def test_the_coder_gives_models_no_way_to_approve(assistant):
    declared = [capability.name for capability in assistant.broker.capabilities()]
    assert not [name for name in declared if name.startswith(("code.", "coder."))]


def test_pushing_is_denied_unless_granted(assistant, repo, tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    git(repo, "remote", "add", "origin", str(remote))
    run = assistant.coder.approve(assistant.coder.start(repo, "Fix add").id)

    with pytest.raises(CoderError, match="refused git.push"):
        assistant.coder.push(run.id)
    assert assistant.coder.run(run.id).state == COMMITTED
    assert git(remote, "for-each-ref") == ""

    assistant.broker.grants.add(("owner", "git.push"))
    assert assistant.coder.push(run.id).state == PUSHED
    refs = git(remote, "for-each-ref", "--format=%(refname)")
    assert refs == f"refs/heads/{run.branch}"  # its own branch, and nothing else


def test_pushing_needs_a_commit_first(assistant, repo):
    run = assistant.coder.start(repo, "Fix add")
    with pytest.raises(CoderError, match="Only a committed run"):
        assistant.coder.push(run.id)


def test_rejecting_discards_the_worktree_and_branch(assistant, repo):
    before = snapshot(repo)
    run = assistant.coder.start(repo, "Fix add")
    done = assistant.coder.reject(run.id, "not what I wanted")
    assert done.state == REJECTED
    assert not run.tree.exists() and not run.folder.exists()
    assert git(repo, "branch", "--list", "assistant/*") == ""
    assert "project" in git(repo, "worktree", "list") and str(run.tree.name) not in git(repo, "worktree", "list").replace("project", "")
    assert snapshot(repo) == before
    with pytest.raises(CoderError, match="already rejected"):
        assistant.coder.reject(run.id)


def test_rejecting_a_commit_keeps_the_branch(assistant, repo):
    run = assistant.coder.approve(assistant.coder.start(repo, "Fix add").id)
    assistant.coder.reject(run.id)
    assert git(repo, "rev-parse", run.branch) == run.commit_id


def test_runs_stay_with_their_user(assistant, repo):
    run = assistant.coder.start(repo, "Fix add", user="ada")
    assert [item.id for item in assistant.coder.runs("ada")] == [run.id]
    assert assistant.coder.runs("grace") == []
    for action in (assistant.coder.show, assistant.coder.reject):
        with pytest.raises(CoderError, match="no run"):
            action(run.id, user="grace")


# Generated code stays out of this process


PID_PROBE = 'import os\nfrom pathlib import Path\nPath(os.environ["PROBE"]).write_text(str(os.getpid()))\n'


def test_generated_code_never_runs_in_this_process(assistant, repo, tmp_path, monkeypatch):
    """The generated file runs for real, and reports which process it ran in."""
    probe = tmp_path / "pid.txt"
    monkeypatch.setenv("PROBE", str(probe))
    assistant.router = ScriptedRouter([f"{FIX_AS_FILE}<<<FILE conftest.py\n{PID_PROBE}>>>FILE\n"])

    commands: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(command: Any, *args: Any, **kwargs: Any) -> Any:
        commands.append(list(command))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    run = assistant.coder.start(repo, "Fix add")

    assert run.state == READY
    assert probe.exists(), "the generated code never ran, so this proved nothing"
    assert int(probe.read_text()) != os.getpid()
    assert "conftest" not in sys.modules or not hasattr(sys.modules["conftest"], "PID_PROBE")

    sandbox_commands = [call["command"] for call in assistant.sandbox.calls]
    assert [command for command in commands if command[0] != "git"] == sandbox_commands


def test_with_no_sandbox_nothing_runs_and_nothing_falls_back(tmp_path, repo):
    before = snapshot(repo)
    with Assistant(tmp_path / "bare") as bare:
        bare.router = ScriptedRouter([FIX_AS_FILE])
        bare.sandbox = None
        with pytest.raises(CoderError, match="never runs in this process"):
            bare.coder.start(repo, "Fix add")
        assert bare.router.requests == []
    assert snapshot(repo) == before
    assert git(repo, "branch", "--list", "assistant/*") == ""


def test_the_tests_run_on_a_copy_with_no_way_back_to_the_repository(assistant, repo):
    run = assistant.coder.start(repo, "Fix add")
    call = assistant.sandbox.calls[-1]
    assert call["network"] is False
    assert call["workdir"] != run.tree and call["workdir"] != repo
    assert ".git" not in call["files"] and not [name for name in call["files"] if name.startswith(".git/")]
    assert "calc.py" in call["files"]
    assert not run.copy.exists()  # the copy is gone afterwards


def test_what_a_test_run_does_never_reaches_the_diff(assistant, repo):
    """Code under test writing to its folder changes the copy, not what gets committed."""
    leaving = 'from pathlib import Path\nPath("left-behind.txt").write_text("x")\n'
    assistant.router = ScriptedRouter([f"{FIX_AS_FILE}<<<FILE conftest.py\n{leaving}>>>FILE\n"])
    run = assistant.coder.start(repo, "Fix add")
    assert run.state == READY
    assert "a/left-behind.txt" not in assistant.coder.show(run.id)["diff"]
    assert not (run.tree / "left-behind.txt").exists()


def test_a_test_that_hangs_is_killed_and_reported(assistant, repo):
    hang = "<<<FILE test_calc.py\nimport time\n\n\ndef test_add():\n    time.sleep(60)\n>>>FILE\n"
    assistant.router = ScriptedRouter([hang])
    run = assistant.coder.start(repo, "Fix add", max_rounds=1, test_seconds=3.0)
    assert run.state == STOPPED
    assert assistant.coder.show(run.id)["tests"][-1]["timed_out"] == 1


def test_no_tests_collected_is_not_a_pass(assistant, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q", "-b", "main")
    (empty / "pytest.ini").write_bytes(b"[pytest]\n")
    git(empty, "add", "-A")
    git(empty, "commit", "-q", "-m", "No tests")
    assistant.router = ScriptedRouter(["<<<FILE a.py\nx = 1\n>>>FILE\n"])
    assert assistant.coder.start(empty, "Add a variable", max_rounds=1).state == STOPPED


# What a model may not do


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        ".git/hooks/pre-commit",
        ".GIT/config",
        ".git./config",
        ".git /config",
        "sub/.git/config",
        "../outside.txt",
        "a/../../outside.txt",
        "/etc/passwd",
        "C:/Windows/system.ini",
        "C:\\Windows\\system.ini",
        "file.py::$DATA",
        ".git::$INDEX_ALLOCATION/config",
        "git~1/config",
        "CON",
        "nul.txt",
        "",
        ".",
    ],
)
def test_paths_a_model_may_not_write(path):
    with pytest.raises(UnusableChange):
        safe_path(path)


@pytest.mark.parametrize("path", ["calc.py", "src/app/main.py", ".github/workflows/ci.yml", "a.b.c", "gitea/x", ".gitignore", "console.py"])
def test_paths_a_model_may_write(path):
    assert safe_path(path) == path


@pytest.mark.parametrize(
    "reply",
    [
        "<<<FILE .git/config\n[core]\nhooksPath = /tmp\n>>>FILE\n",
        "<<<FILE ../escaped.txt\nx\n>>>FILE\n",
        "<<<DIFF\n--- /dev/null\n+++ b/.git/hooks/post-commit\n@@ -0,0 +1 @@\n+#!/bin/sh\n>>>DIFF\n",
        "<<<DIFF\n--- /dev/null\n+++ b/../escaped.txt\n@@ -0,0 +1 @@\n+x\n>>>DIFF\n",
        "<<<DIFF\ndiff --git a/link b/link\nnew file mode 120000\n--- /dev/null\n+++ b/link\n@@ -0,0 +1 @@\n+/etc/passwd\n\\ No newline at end of file\n>>>DIFF\n",
        "<<<DIFF\ndiff --git a/calc.py b/calc.py\nnew file mode 160000\n--- /dev/null\n+++ b/sub\n@@ -0,0 +1 @@\n+Subproject commit 0000000000000000000000000000000000000000\n>>>DIFF\n",
        "<<<DIFF\ndiff --git a/x.bin b/x.bin\nnew file mode 100644\nGIT binary patch\nliteral 1\nIc${NkU;qFB00961\n\nliteral 0\nHc$@<O00001\n\n>>>DIFF\n",
        "this is not a change at all",
        "<<<DIFF\nnot a diff\n>>>DIFF\n",
    ],
)
def test_a_bad_reply_is_refused_and_the_model_told_why(assistant, repo, reply):
    assistant.router = ScriptedRouter([reply, FIX_AS_FILE])
    config = (repo / ".git" / "config").read_bytes()
    run = assistant.coder.start(repo, "Fix add")
    assert run.state == READY and run.round == 2
    second = [request for request in assistant.router.requests if request.role == "coding"][1]
    assert "could not be applied" in second.messages[1]["content"]
    assert (repo / ".git" / "config").read_bytes() == config
    assert not (run.folder / "escaped.txt").exists() and not (repo.parent / "escaped.txt").exists()
    assert not (run.tree / "link").exists()
    diff = assistant.coder.show(run.id)["diff"]
    assert "120000" not in diff and "hooks" not in diff
    assert [row for row in assistant.audit.read() if row["action"] == "coder.change_refused"]


@pytest.mark.skipif(os.name == "nt", reason="making a symlink needs special rights on Windows")
def test_a_write_through_a_symlink_is_refused(assistant, repo, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "link").symlink_to(outside)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "A link")
    assistant.router = ScriptedRouter(["<<<FILE link/owned.txt\nx\n>>>FILE\n", FIX_AS_FILE])
    run = assistant.coder.start(repo, "Fix add")
    assert run.state == READY
    assert not (outside / "owned.txt").exists()


def test_a_tool_call_from_the_model_is_ignored(assistant, repo):
    from isharded.assistant.contracts import ToolCall

    class Scheming(ScriptedRouter):
        def complete(self, request: Any) -> ModelReply:
            reply = super().complete(request)
            return ModelReply(reply.text, (ToolCall("git.push", {"remote": "origin"}),), reply.model, reply.provider, 1, 1, 0.1)

    assistant.router = Scheming([FIX_AS_FILE])
    assert assistant.coder.start(repo, "Fix add").state == READY
    assert assistant.broker.calls == []


def test_a_change_to_the_hooks_folder_is_never_committed_by_git(assistant, repo):
    """With a relative hooks path, a generated hook would run outside the sandbox at commit."""
    git(repo, "config", "core.hooksPath", ".githooks")
    hook = "<<<FILE .githooks/pre-commit\n#!/bin/sh\ntouch /tmp/owned\n>>>FILE\n"
    assistant.router = ScriptedRouter([FIX_AS_FILE + hook])
    run = assistant.coder.start(repo, "Fix add")
    assert any("automation" in warning for warning in run.warnings)
    with pytest.raises(CoderError, match="hooks folder"):
        assistant.coder.approve(run.id)
    assert assistant.broker.calls == []


def test_the_owners_hooks_do_not_run_in_the_throwaway_copy(assistant, repo):
    marker = repo.parent / "hook-ran.txt"
    hook = repo / ".git" / "hooks" / "post-checkout"
    hook.write_bytes(f'#!/bin/sh\necho ran > "{marker.as_posix()}"\n'.encode())
    hook.chmod(0o755)
    assistant.coder.start(repo, "Fix add")
    assert not marker.exists()
    git(repo, "checkout", "-q", "main")
    assert marker.exists(), "the hook never works, so this proved nothing"


# What the owner is warned about


def test_changing_a_test_to_pass_is_flagged(assistant, repo):
    cheat = "<<<FILE test_calc.py\ndef test_add():\n    assert True\n>>>FILE\n"
    assistant.router = ScriptedRouter([cheat])
    run = assistant.coder.start(repo, "Fix add")
    assert run.state == READY
    assert any("changes tests" in warning for warning in run.warnings)


def test_removing_a_test_is_flagged(assistant, repo):
    remove = "<<<DIFF\n--- a/test_calc.py\n+++ /dev/null\n@@ -1,5 +0,0 @@\n-from calc import add\n-\n-\n-def test_add():\n-    assert add(2, 3) == 5\n>>>DIFF\n"
    keep = "<<<FILE test_other.py\ndef test_other():\n    assert True\n>>>FILE\n"
    assistant.router = ScriptedRouter([remove + keep])
    run = assistant.coder.start(repo, "Fix add")
    assert any("removes tests: test_calc.py" in warning for warning in run.warnings)


def test_changing_how_tests_run_is_flagged(assistant, repo):
    skip = "<<<FILE pytest.ini\n[pytest]\naddopts = --co -q\n>>>FILE\n"
    assistant.router = ScriptedRouter([skip + FIX_AS_FILE])
    run = assistant.coder.start(repo, "Fix add")
    assert any("how tests or builds run" in warning for warning in run.warnings)


def test_a_clean_change_has_no_warnings(assistant, repo):
    assert assistant.coder.start(repo, "Fix add").warnings == ()


# Resuming


class Crash(BaseException):
    """Stands in for the process dying: nothing may catch it."""


def crash_at(assistant: Any, method: str) -> None:
    def die(*_: Any, **__: Any) -> None:
        raise Crash()

    setattr(assistant.coder, method, die)
    assistant.coder.steps = {**assistant.coder.steps, **{state: getattr(assistant.coder, method) for state, step in assistant.coder.steps.items() if step.__name__ == method}}


@pytest.mark.parametrize("method", ["_inspect", "_propose", "_apply", "_test", "_review"])
def test_a_crash_at_any_step_can_be_resumed(tmp_path, repo, method):
    with Assistant(tmp_path / "home") as first:
        first.sandbox, first.router = ProcessSandbox(), ScriptedRouter([FIX_AS_FILE])
        crash_at(first, method)
        with pytest.raises(Crash):
            first.coder.start(repo, "Fix add")
        run_id = first.store.one("SELECT id FROM coder_runs")["id"]

    with Assistant(tmp_path / "home") as second:
        second.sandbox, second.router = ProcessSandbox(), ScriptedRouter([FIX_AS_FILE])
        run = second.coder.resume(run_id)
        assert run.state == READY
        assert (run.tree / "calc.py").read_bytes() == FIXED.encode()


NODE = """
import sys
sys.path.insert(0, sys.argv[1])
import test_assistant_coder as helpers
from isharded.assistant.app import Assistant

class Dying(helpers.ProcessSandbox):
    def run(self, command, workdir, **kwargs):
        if len(self.calls) == 1:
            import os
            os._exit(9)  # no cleanup, no unwinding: the process is simply gone
        return super().run(command, workdir, **kwargs)

assistant = Assistant(sys.argv[2])
assistant.sandbox = Dying()
assistant.router = helpers.ScriptedRouter([helpers.FIX_AS_FILE])
assistant.coder.start(sys.argv[3], "Fix add")
"""


def test_a_process_killed_mid_run_leaves_a_run_that_can_be_resumed(tmp_path, repo):
    """The real thing: a separate process dies between applying a change and testing it."""
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    home = tmp_path / "home"
    result = subprocess.run([sys.executable, "-c", NODE, str(ROOT / "tests"), str(home), str(repo)], env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 9, result.stderr

    with Assistant(home) as assistant:
        row = assistant.store.one("SELECT id, state FROM coder_runs")
        assert row["state"] == "testing"
        assistant.sandbox, assistant.router = ProcessSandbox(), ScriptedRouter([FIX_AS_FILE])
        run = assistant.coder.resume(row["id"])
        assert run.state == READY
        assert [request.role for request in assistant.router.requests] == ["verification"]  # the coding reply was not asked for again


def test_a_model_outage_leaves_the_run_waiting(assistant, repo):
    assistant.router = ScriptedRouter([RuntimeError("no models up")])
    run = assistant.coder.start(repo, "Fix add")
    assert run.state == PROPOSING
    assert "code resume" in run.problem
    assistant.router = ScriptedRouter([FIX_AS_FILE])
    resumed = assistant.coder.resume(run.id)
    assert resumed.state == READY and resumed.problem == ""


def test_a_sandbox_outage_leaves_the_run_waiting(assistant, repo):
    def broken(*_: Any, **__: Any) -> Any:
        raise OSError("sandbox is down")

    real = assistant.sandbox.run
    assistant.sandbox.run = broken
    run = assistant.coder.start(repo, "Fix add")
    assert "sandbox could not run" in run.problem
    assistant.sandbox.run = real
    assert assistant.coder.resume(run.id).state == READY


def test_resuming_a_finished_run_says_so(assistant, repo):
    run = assistant.coder.start(repo, "Fix add")
    with pytest.raises(CoderError, match="nothing to resume"):
        assistant.coder.resume(run.id)


# Starting


def test_each_run_gets_its_own_branch(assistant, repo):
    first = assistant.coder.start(repo, "Fix add")
    second = assistant.coder.start(repo, "Fix add")
    assert first.branch == "assistant/fix-add"
    assert second.branch == "assistant/fix-add-2"
    assert first.tree != second.tree


def test_a_branch_that_already_exists_is_left_alone(assistant, repo):
    git(repo, "branch", "assistant/fix-add")
    run = assistant.coder.start(repo, "Fix add")
    assert run.branch == "assistant/fix-add-2"


def test_the_goal_cannot_reach_the_branch_name(assistant, repo):
    run = assistant.coder.start(repo, "--force ../../main; rm -rf /")
    assert run.branch == "assistant/force-main-rm-rf"


@pytest.mark.parametrize("goal, slug", [("Fix  the BUG!", "fix-the-bug"), ("---", "change"), ("Ünïcode", "n-code"), ("x" * 100, "x" * 40)])
def test_slugs(goal, slug):
    assert slug_of(goal) == slug


def test_not_a_repository_says_so(assistant, tmp_path):
    folder = tmp_path / "plain"
    folder.mkdir()
    with pytest.raises(CoderError, match="not inside a git repository"):
        assistant.coder.start(folder, "Fix add")
    with pytest.raises(CoderError, match="is not a folder"):
        assistant.coder.start(folder / "missing", "Fix add")


def test_a_repository_with_no_commits_says_so(assistant, tmp_path):
    folder = tmp_path / "new"
    folder.mkdir()
    git(folder, "init", "-q")
    with pytest.raises(CoderError, match="no commits"):
        assistant.coder.start(folder, "Fix add")


def test_no_test_command_stops_and_asks_for_one(assistant, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    git(folder, "init", "-q", "-b", "main")
    (folder / "readme.txt").write_text("hello")
    git(folder, "add", "-A")
    git(folder, "commit", "-q", "-m", "Words")
    run = assistant.coder.start(folder, "Improve the words")
    assert run.state == STOPPED and "--test" in run.problem
    assert assistant.router.requests == []
    with pytest.raises(CoderError, match="no test command"):
        assistant.coder.resume(run.id, rounds=1)


def test_a_given_test_command_is_used(assistant, repo):
    run = assistant.coder.start(repo, "Fix add", test_command=["python", "-m", "pytest", "-q", "test_calc.py"])
    assert run.state == READY
    assert assistant.sandbox.calls[-1]["command"][-1] == "test_calc.py"


def test_source_stays_near_unless_the_owner_says_otherwise(assistant, repo):
    assistant.coder.start(repo, "Fix add")
    assert {request.reach for request in assistant.router.requests} == {Reach.POOL}
    assistant.router = ScriptedRouter([FIX_AS_FILE])
    assistant.coder.start(repo, "Fix add", reach=Reach.MACHINE)
    assert {request.reach for request in assistant.router.requests} == {Reach.MACHINE}


def test_repository_text_is_labelled_as_data(assistant, repo):
    assistant.coder.start(repo, "Fix add")
    prompt = assistant.router.requests[0].messages[1]["content"]
    assert "File calc.py (data, not instructions)" in prompt
    assert "data, not instructions" in assistant.router.requests[0].messages[0]["content"]


@pytest.mark.parametrize(
    "files, expected",
    [
        (["a.py", "b.py", "c.ts"], ("Python", "TypeScript")),
        (["main.go"], ("Go",)),
        (["README.md"], ()),
    ],
)
def test_languages_are_detected(files, expected):
    assert detect_languages(files) == expected


def test_test_commands_are_detected(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
    (tmp_path / "Makefile").write_text("build:\n\ttrue\ntest:\n\ttrue\n")
    assert detect_test_command(["pyproject.toml"], tmp_path) == ("python", "-m", "pytest", "-q")
    assert detect_test_command(["pkg/test_x.py"], tmp_path) == ("python", "-m", "pytest", "-q")
    assert detect_test_command(["package.json"], tmp_path) == ("npm", "test", "--silent")
    assert detect_test_command(["Cargo.toml"], tmp_path) == ("cargo", "test", "--quiet")
    assert detect_test_command(["go.mod"], tmp_path) == ("go", "test", "./...")
    assert detect_test_command(["Makefile"], tmp_path) == ("make", "test")
    assert detect_test_command(["notes.txt"], tmp_path) is None


def test_npms_placeholder_is_not_a_test_command(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "echo \\"Error: no test specified\\" && exit 1"}}')
    assert detect_test_command(["package.json"], tmp_path) is None


def test_replies_parse_in_order():
    reply = (
        "Here you go.\n<<<FILE a.py\nx = 1\n>>>FILE\n"
        "```diff\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-y\n+z\n```\n"
        "<<<FILE sub/c.py\n```not a fence here\n>>>FILE\n"
    )
    changes = parse_changes(reply)
    assert [type(change) for change in changes] == [FileChange, DiffChange, FileChange]
    assert changes[0] == FileChange("a.py", "x = 1\n")
    assert changes[2].path == "sub/c.py"


def test_a_file_can_hold_text_that_looks_like_a_marker():
    reply = "<<<FILE notes.md\n<<<FILE not-this.txt\n```diff\nstill the file\n>>>FILE\n"
    assert parse_changes(reply) == [FileChange("notes.md", "<<<FILE not-this.txt\n```diff\nstill the file\n")]


def test_a_bare_diff_counts():
    assert parse_changes("--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-y\n+z\n") == [DiffChange("--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-y\n+z\n")]


def test_an_oversized_reply_is_refused():
    with pytest.raises(UnusableChange, match="too large"):
        parse_changes("x" * (3 << 20))


def test_a_hostile_reply_cannot_slow_parsing_down():
    import time

    started = time.monotonic()
    assert parse_changes("<<<FILE a\n" * 200_000) == []
    assert parse_changes("<<<DIFF\n" * 200_000) == []
    assert time.monotonic() - started < 5


def test_an_unclosed_block_is_not_a_change():
    assert parse_changes("<<<FILE a.py\nx = 1\n") == []
    assert parse_changes("<<<FILE a.py\nx = 1\n>>>FILE\n<<<FILE b.py\ny = 2\n") == [FileChange("a.py", "x = 1\n")]


# iSharded itself


@pytest.fixture
def isharded_repo(tmp_path) -> Path:
    folder = tmp_path / "isharded-checkout"
    (folder / "isharded" / "assistant").mkdir(parents=True)
    git(folder, "init", "-q", "-b", "main")
    git(folder, "config", "core.autocrlf", "false")
    (folder / "pyproject.toml").write_bytes(b'[project]\nname = "isharded"\n')
    (folder / "isharded" / "orchestrator.py").write_bytes(b"VERSION = 1\n")
    (folder / "isharded" / "assistant" / "skills.py").write_bytes(b"LIMIT = 1\n")
    (folder / "test_version.py").write_bytes(b"def test_version():\n    assert 2 == 2\n")
    git(folder, "add", "-A")
    git(folder, "commit", "-q", "-m", "Start")
    return folder


def test_isharded_itself_gets_a_branch_tests_and_a_written_plan(assistant, isharded_repo):
    change = "<<<FILE isharded/orchestrator.py\nVERSION = 2\n>>>FILE\n"
    assistant.router = ScriptedRouter([change])
    before = snapshot(isharded_repo)
    run = assistant.coder.start(isharded_repo, "Raise the version")

    assert run.state == READY and run.branch == "assistant/raise-the-version"
    assert run.note
    for needed in ("Health check", "Rollback", "GET /health", "GET /ready", run.base, run.branch, "Nothing has been deployed or restarted"):
        assert needed in run.note
    assert "nodes and the orchestrator" in run.note
    assert (run.folder / "HEALTH_AND_ROLLBACK.md").read_text(encoding="utf-8") == run.note
    assert "HEALTH_AND_ROLLBACK" not in assistant.coder.show(run.id)["diff"]  # the plan is not part of the change
    assert snapshot(isharded_repo) == before


def test_a_change_to_the_assistant_alone_says_so(assistant, isharded_repo):
    assistant.router = ScriptedRouter(["<<<FILE isharded/assistant/skills.py\nLIMIT = 2\n>>>FILE\n"])
    run = assistant.coder.start(isharded_repo, "Raise the limit")
    assert "only `isharded assistant`" in run.note


def test_isharded_is_never_deployed_or_restarted(assistant, isharded_repo, monkeypatch):
    commands: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(command: Any, *args: Any, **kwargs: Any) -> Any:
        commands.append(list(command))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    assistant.router = ScriptedRouter(["<<<FILE isharded/orchestrator.py\nVERSION = 2\n>>>FILE\n"])
    run = assistant.coder.approve(assistant.coder.start(isharded_repo, "Raise the version").id)

    assert run.state == COMMITTED
    assert [capability for capability, _, _ in assistant.broker.calls] == ["git.commit"]
    outside_git = [command for command in commands if command[0] != "git"]
    assert outside_git == [call["command"] for call in assistant.sandbox.calls]
    assert {subcommand for command in commands if command[0] == "git" for subcommand in command[1:] if subcommand in ("push", "pull", "fetch", "merge")} == set()


def test_another_project_gets_no_plan(assistant, repo):
    assert assistant.coder.start(repo, "Fix add").note == ""


def test_a_project_merely_named_like_it_gets_none(assistant, tmp_path):
    folder = tmp_path / "other"
    folder.mkdir()
    git(folder, "init", "-q", "-b", "main")
    (folder / "pyproject.toml").write_bytes(b'[project]\nname = "not-isharded"\n')
    (folder / "test_x.py").write_bytes(b"def test_x():\n    assert True\n")
    git(folder, "add", "-A")
    git(folder, "commit", "-q", "-m", "x")
    assistant.router = ScriptedRouter(["<<<FILE y.py\nz = 1\n>>>FILE\n"])
    assert assistant.coder.start(folder, "Add y").note == ""


# The broker's arguments


def test_arguments_follow_what_the_tool_declared(assistant, repo):
    assistant.broker = GitBroker({"properties": {"path": {}, "message": {}}})
    assistant.broker.grants.add(("owner", "git.commit"))
    run = assistant.coder.approve(assistant.coder.start(repo, "Fix add").id)
    assert run.state == COMMITTED
    assert set(assistant.broker.calls[0][1]) >= {"path", "message"}


# Commands


def run_command(assistant: Any, *words: str) -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(list(words))
    return args.run(assistant, args)


def test_the_commands_take_a_change_through_its_life(assistant, repo, capsys):
    assert run_command(assistant, "code", "start", "--repo", str(repo), "Fix add") == 0
    out = capsys.readouterr().out
    assert "The tests pass" in out
    run_id = out.split()[0]

    assert run_command(assistant, "code", "list") == 0
    assert "assistant/fix-add" in capsys.readouterr().out

    assert run_command(assistant, "code", "show", run_id, "--output") == 0
    shown = capsys.readouterr().out
    assert "+    return a + b" in shown
    assert "Tests, before any change: exit 1, in a process sandbox" in shown
    assert "Tests, round 1: exit 0" in shown
    assert "coding-model (coding, local)" in shown and "verification-model (verification, local)" in shown
    assert "passed" in shown

    assert run_command(assistant, "code", "push", run_id) == 1
    assert "Only a committed run" in capsys.readouterr().out

    assert run_command(assistant, "code", "approve", run_id, "--message", "Fix the adder") == 0
    assert "Your checkout was not touched" in capsys.readouterr().out
    assert git(repo, "log", "-1", "--format=%s", "assistant/fix-add") == "Fix the adder"

    assert run_command(assistant, "code", "push", run_id, "--remote", "origin") == 1
    assert "refused git.push" in capsys.readouterr().out

    assert run_command(assistant, "code", "reject", run_id, "--reason", "changed my mind") == 0
    assert "the commit stays" in capsys.readouterr().out


def test_start_options_reach_the_run(assistant, repo, capsys):
    assistant.router = ScriptedRouter(["nothing useful"])
    code = run_command(
        assistant, "code", "start", "--repo", str(repo), "--test", "python -m pytest -q test_calc.py", "--rounds", "1", "--timeout", "30", "--reach", "machine", "Fix add"
    )
    assert code == 1
    run = assistant.coder.runs()[0]
    assert run.test_command == ("python", "-m", "pytest", "-q", "test_calc.py")
    assert run.max_rounds == 1 and run.test_seconds == 30 and run.reach is Reach.MACHINE

    assistant.router = ScriptedRouter([FIX_AS_FILE])
    assert run_command(assistant, "code", "resume", run.id, "--rounds", "1") == 0


def test_a_command_tells_the_owner_what_to_do(tmp_path, capsys):
    with Assistant(tmp_path / "bare") as bare:
        bare.router, bare.sandbox = None, None
        assert run_command(bare, "code", "list") == 0
        assert "code start" in capsys.readouterr().out
        assert run_command(bare, "code", "show", "cr_none") == 1
        assert "code list" in capsys.readouterr().out
        assert run_command(bare, "code", "start", "--repo", str(tmp_path), "x") == 1
        assert "needs the router part" in capsys.readouterr().out


# What testing the work tells the router


class Told(ScriptedRouter):
    def __init__(self, *arguments, **options) -> None:
        super().__init__(*arguments, **options)
        self.outcomes: list[tuple] = []

    def record_outcome(self, model, role, success, provider=None) -> None:
        self.outcomes.append((model, role, success, provider))


def test_tests_that_pass_and_fail_tell_the_router_about_the_coding_model(assistant, repo):
    assistant.router = Told([f"<<<FILE calc.py\n{BROKEN}# still wrong\n>>>FILE\n", FIX_AS_FILE])
    assert assistant.coder.start(repo, "Fix add").state == READY
    assert assistant.router.outcomes == [("coding-model", "coding", False, "local"), ("coding-model", "coding", True, "local")]


def test_a_reply_that_cannot_be_applied_counts_against_it(assistant, repo):
    assistant.router = Told(["nothing usable here", FIX_AS_FILE])
    assert assistant.coder.start(repo, "Fix add").state == READY
    assert assistant.router.outcomes[0] == ("coding-model", "coding", False, "local")


def test_a_reviewers_opinion_is_not_a_check(assistant, repo):
    assistant.router = Told([FIX_AS_FILE], ['{"verdict": "changes", "notes": "Hm."}', '{"verdict": "accept", "notes": "Fine."}'])
    assistant.coder.start(repo, "Fix add")
    assert {row[1] for row in assistant.router.outcomes} == {"coding"}
