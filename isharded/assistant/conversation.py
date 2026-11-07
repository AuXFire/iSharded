"""One turn: context, a model, tools through the broker, and what is kept after.

The loop asks the model, runs what it asked for, and asks again, for a set number
of rounds. It holds no authority of its own. Whether a tool may run is the broker's
answer, and this code only passes that answer on, along with whether untrusted
text could have steered the request. Once a tool has answered, the rest of the
turn is tainted, so a poisoned result cannot use a standing grant to act.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
import re
from typing import Any
import uuid

from . import gateway_cli
from .context import ContextTooSmall, TurnNotFound, merged
from .contracts import CallContext, Capability, ContextItem, Effect, ModelReply, ModelRequest, Provenance, Reach, ToolCall, Trust, Verdict
from .memory import KINDS
from .mind import actor, add_user_option, emit, plainly
from .providers import estimate_tokens, loose_json

ROLE = "conversation"
MAX_ROUNDS = 6
BUDGET_TOKENS = 6144
REPLY_TOKENS = 1024
TOOLS_SHARE = 4  # tool descriptions may take a quarter of the budget, and no more
TOOL_OUTPUT_SHARE = 4  # so may any one tool's output; all of it is still kept in the record
CHARS_PER_TOKEN = 3
MAX_LEARNED = 3
MAX_LEARNED_CHARS = 500
LEARN_MIN_WORDS = 4

LEARN_PROMPT = (
    "From the exchange below, list at most three lasting facts worth remembering about this person or their work. "
    'Reply with only a JSON list of objects like {"kind": "user", "text": "...", "subject": "short topic"}. '
    'Use the kind "user" for preferences and personal context, "semantic" for facts, and "episodic" for events. '
    "Reply with [] if nothing is worth keeping. Most exchanges have nothing worth keeping."
)
DECIDE_PROMPT = (
    "Is this request one quick answer, or a piece of work with several steps that should run as a task? "
    "Reply with exactly one word: small or full."
)

FORCED_FULL = re.compile(r"^/task\b\s*", re.IGNORECASE)
FORCED_SMALL = re.compile(r"^/now\b\s*", re.IGNORECASE)
STEP_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+\S", re.MULTILINE)
LATER = re.compile(r"\b(every (?:day|week|month|morning|night|hour)|overnight|in the background|on a schedule|when (?:it|that) finishes)\b", re.IGNORECASE)
BIG_VERB = re.compile(r"\b(build|implement|refactor|migrate|rewrite|port|develop|deploy|audit|set up|create a)\b", re.IGNORECASE)
QUESTION = re.compile(r"^\s*(what|why|how|when|where|who|which|can|could|should|would|is|are|does|do)\b", re.IGNORECASE)


def classify(text: str) -> str:
    """small, full, or unclear. Only what software cannot tell goes to a model."""
    stripped = text.strip()
    if FORCED_FULL.match(stripped):
        return "full"
    if FORCED_SMALL.match(stripped):
        return "small"
    if LATER.search(stripped) or len(STEP_LINE.findall(stripped)) >= 3:
        return "full"
    words = len(stripped.split())
    if QUESTION.match(stripped) or stripped.endswith("?") or words < 8:
        return "small"
    return "unclear" if BIG_VERB.search(stripped) else "small"


def without_prefix(text: str) -> str:
    return FORCED_SMALL.sub("", FORCED_FULL.sub("", text.strip())) or text


def decide_approval(assistant: Any, approval_id: str, approved: bool, by: str) -> bool:
    """Tell the broker what a person decided. False means it was no longer waiting."""
    broker = getattr(assistant, "broker", None)
    return broker is not None and broker.decide(approval_id, approved, by)


def tool_tokens(tool: Capability) -> int:
    return estimate_tokens(tool.name + tool.summary + str(tool.input_schema))


SCHEMA = [
    """
    CREATE TABLE paused_turns (
        turn_id TEXT PRIMARY KEY,
        user TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        approval_id TEXT,
        capability TEXT NOT NULL,
        arguments TEXT NOT NULL,
        reason TEXT NOT NULL,
        state TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE INDEX paused_turns_user ON paused_turns (user, status);
    CREATE TABLE handed_tasks (
        task_id TEXT PRIMARY KEY,
        user TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE INDEX handed_tasks_user ON handed_tasks (user, created_at);
    """,
]


@dataclass(frozen=True)
class TurnResult:
    status: str  # done, awaiting_approval, task, refused, or failed
    text: str
    user: str
    conversation_id: str
    turn_id: str
    model: str | None = None
    provider: str | None = None
    tainted: bool = False
    reach: Reach = Reach.POOL
    pending: dict[str, Any] | None = None
    task_id: str | None = None
    tools: tuple[str, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0


def final_result(items: Iterable[str | TurnResult]) -> TurnResult:
    """Run a turn to its end and keep only the result."""
    result = None
    for item in items:
        if isinstance(item, TurnResult):
            result = item
    assert result is not None
    return result


@dataclass
class Turn:
    user: str
    query: str
    conversation_id: str
    turn_id: str
    built: Any = None
    offered: tuple[Capability, ...] = ()  # the tools chosen when the context was built
    tainted: bool = False
    reach: Reach = Reach.POOL
    transcript: list[dict[str, Any]] = field(default_factory=list)  # what happened after the context was built
    said: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    rounds: int = 0
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def messages(self, budget: int) -> list[dict[str, Any]]:
        """Everything so far. If tools have said too much, the oldest outputs give way."""
        room = budget - self.built.tokens
        newest_first: list[dict[str, Any]] = []
        for entry in reversed(self.transcript):
            text = entry["text"]
            cost = estimate_tokens(text)
            if entry["kind"] == "tool" and cost > room:
                text, cost = "(This output was removed to save space. It is still in the record.)", 20
            room -= cost
            newest_first.append(self._message(entry, text))
        system, *rest = self.built.messages
        return [system, *merged([*rest, *reversed(newest_first)])]

    def _message(self, entry: dict[str, Any], text: str) -> dict[str, Any]:
        if entry["kind"] == "assistant":
            return {"role": "assistant", "content": text}
        if entry["kind"] == "tool":
            return {"role": "user", "content": self.built.wrap(text, f"tool: {entry['capability']}")}
        return {"role": "user", "content": f"[From the assistant's own software, not the person] {text}"}


class Conversation:
    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.store.migrate("conversation", SCHEMA)
        self.budget_tokens = BUDGET_TOKENS
        self.reply_tokens = REPLY_TOKENS
        self.max_rounds = MAX_ROUNDS
        self.learner: ThreadPoolExecutor | None = None
        # A turn taken by a process that died can be taken again. Nothing runs
        # twice, because the broker's approvals are good for one use.
        self.store.execute("UPDATE paused_turns SET status = 'waiting' WHERE status = 'running'")

    # The public turn

    def say(self, user: str, text: str, conversation_id: str | None = None, learn: str = "now") -> TurnResult:
        """One turn, start to finish. ``learn`` is now, background, or off."""
        result = final_result(self._run(user, text, conversation_id, streaming=False))
        self._learn_after(result, text, learn)
        return result

    def stream(self, user: str, text: str, conversation_id: str | None = None, learn: str = "now") -> Iterator[str | TurnResult]:
        """The same turn, yielding text as it comes and the TurnResult last."""
        for item in self._run(user, text, conversation_id, streaming=True):
            yield item
            if isinstance(item, TurnResult):
                self._learn_after(item, text, learn)

    def _run(self, user: str, text: str, conversation_id: str | None, streaming: bool) -> Iterator[str | TurnResult]:
        text = text.strip()
        if not text:
            raise ValueError("Say something first")
        turn = Turn(user, without_prefix(text), conversation_id or uuid.uuid4().hex, uuid.uuid4().hex)
        turn.reach = self.assistant.context.default_reach(user)
        # What a person said is kept before anything else can go wrong.
        self.assistant.memory.record_message(user, turn.conversation_id, "user", text, turn_id=turn.turn_id, reach=turn.reach)
        self.assistant.audit.record(actor(user), "conversation.turn", f"turn:{turn.turn_id}", {"conversation": turn.conversation_id})
        handed = self._hand_off(turn, classify(text))
        if handed is not None:
            yield handed
            return
        try:
            self._build(turn)
        except ContextTooSmall as error:
            yield self._finish(turn, "failed", str(error))
            return
        yield from self._advance(turn, streaming)

    def _build(self, turn: Turn) -> None:
        turn.offered = tuple(self._tools(turn.user))
        reserve = sum(tool_tokens(tool) for tool in turn.offered)
        built = self.assistant.context.build(
            turn.user, turn.query, ROLE, self.budget_tokens, turn.conversation_id, turn.turn_id, reserve_tokens=reserve, exclude_turn=turn.turn_id,
        )
        turn.built = built
        turn.tainted = turn.tainted or built.tainted
        turn.reach = min(turn.reach, built.reach)

    def _tools(self, user: str) -> list[Capability]:
        """What this person could really use, harmless tools first, within a share of the budget.

        Describing what nobody granted would only earn a refusal. With every part
        attached it also filled the share, so a granted shell.run was never offered,
        and a small model does worse the more it has to read.
        """
        broker = getattr(self.assistant, "broker", None)
        if broker is None:
            return []
        granted = {row["capability"] for row in broker.grants(user)}
        usable = [tool for tool in broker.capabilities() if tool.name in granted or Effect.PHYSICAL in tool.effects]
        offered, spent = [], 0
        for tool in sorted(usable, key=lambda item: (item.consequential, item.name)):
            spent += tool_tokens(tool)
            if spent > self.budget_tokens // TOOLS_SHARE:
                break
            offered.append(tool)
        return offered

    # The loop

    def _advance(self, turn: Turn, streaming: bool, pending: Sequence[ToolCall] = ()) -> Iterator[str | TurnResult]:
        calls = pending
        while True:
            paused = self._run_calls(turn, calls)
            if paused is not None:
                yield paused
                return
            final_round = turn.rounds >= self.max_rounds
            if final_round:
                turn.transcript.append({"kind": "note", "text": "That was the last tool you can use this turn. Answer now with what you have."})
            turn.rounds += 1
            request = ModelRequest(
                role=ROLE, messages=turn.messages(self.budget_tokens), tools=() if final_round else turn.offered, reach=turn.reach, max_tokens=self.reply_tokens
            )
            partial: list[str] = []
            try:
                reply = yield from self._ask(request, streaming, partial)
            except Exception as error:
                yield self._failed(turn, error, "".join(partial))
                return
            self._note_reply(turn, reply)
            if not reply.tool_calls or final_round:
                yield self._finish(turn, "done", "")
                return
            calls = reply.tool_calls

    def _ask(self, request: ModelRequest, streaming: bool, partial: list[str]) -> Iterator[str]:
        router = self.assistant.router
        stream = getattr(router, "stream", None) if streaming else None
        if stream is None:
            reply = router.complete(request)
            if streaming and reply.text:
                yield reply.text
            return reply
        reply = None
        for item in stream(request):
            if isinstance(item, ModelReply):
                reply = item
            else:
                partial.append(item)
                yield item
        if reply is None:
            raise RuntimeError("The model stopped before finishing its reply")
        return reply

    def _note_reply(self, turn: Turn, reply: ModelReply) -> None:
        turn.model, turn.provider = reply.model, reply.provider
        turn.prompt_tokens += reply.prompt_tokens
        turn.completion_tokens += reply.completion_tokens
        if reply.text:
            turn.said.append(reply.text)
            turn.transcript.append({"kind": "assistant", "text": reply.text})
            self.assistant.memory.record_message(
                turn.user, turn.conversation_id, "assistant", reply.text, turn_id=turn.turn_id, model=reply.model, provider=reply.provider,
                prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens, tainted=turn.tainted, reach=turn.reach,
            )

    def _failed(self, turn: Turn, error: Exception, partial: str) -> TurnResult:
        if partial:
            # What did arrive is kept, and marked, because a person has already read it.
            self.assistant.memory.record_message(
                turn.user, turn.conversation_id, "assistant", partial, turn_id=turn.turn_id, tainted=turn.tainted, reach=turn.reach
            )
            turn.said.append(partial)
        return self._finish(turn, "refused" if getattr(error, "refusal", False) else "failed", str(error))

    def _finish(self, turn: Turn, status: str, notice: str, pending: dict[str, Any] | None = None) -> TurnResult:
        if notice:
            self.assistant.memory.record_message(turn.user, turn.conversation_id, "notice", notice, turn_id=turn.turn_id, reach=turn.reach)
        self.assistant.audit.record(
            actor(turn.user), f"conversation.{status}", f"turn:{turn.turn_id}",
            {"model": turn.model, "provider": turn.provider, "tainted": turn.tainted, "tools": turn.tools},
        )
        return TurnResult(
            status, "\n\n".join([*turn.said, *([notice] if notice else [])]), turn.user, turn.conversation_id, turn.turn_id, turn.model, turn.provider,
            turn.tainted, turn.reach, pending, None, tuple(turn.tools), turn.prompt_tokens, turn.completion_tokens,
        )

    # Tools

    def _run_calls(self, turn: Turn, calls: Sequence[ToolCall]) -> TurnResult | None:
        """Run what the broker allows. Returns a result if the turn has to wait for a person."""
        broker = getattr(self.assistant, "broker", None)
        for index, call in enumerate(calls):
            if broker is None:
                turn.transcript.append({"kind": "note", "text": f"{call.capability} is not available: no tools are installed."})
                continue
            context = CallContext(user=turn.user, actor=ROLE, tainted=turn.tainted, reach=turn.reach)
            decision = broker.check(call, context)
            if decision.verdict is Verdict.ASK:
                return self._pause(turn, calls[index:], decision)
            if decision.verdict is Verdict.ALLOW:
                self._invoke(turn, broker, call, context)
            else:
                turn.transcript.append({"kind": "note", "text": f"{call.capability} was refused: {decision.reason}"})
        return None

    def _invoke(self, turn: Turn, broker: Any, call: ToolCall, context: CallContext) -> None:
        now = self.assistant.clock()
        reach = turn.reach
        try:
            result = broker.invoke(call, context)
            output, reach = result.output, result.provenance.reach
        except Exception as error:  # what a tool says when it fails is still its own text
            output = f"{call.capability} failed: {error}"
        turn.tools.append(call.capability)
        turn.tainted = True
        turn.reach = min(turn.reach, reach)
        limit = self.budget_tokens // TOOL_OUTPUT_SHARE * CHARS_PER_TOKEN
        shown = output if len(output) <= limit else f"{output[:limit]}\n(Cut here: {len(output) - limit} more characters are in the record.)"
        turn.transcript.append({"kind": "tool", "capability": call.capability, "text": shown})
        message_id = self.assistant.memory.record_message(
            turn.user, turn.conversation_id, "tool", output, turn_id=turn.turn_id, tool=call.capability, tainted=True, reach=turn.reach
        )
        origin = Provenance("tool", f"message:{message_id}", call.capability, Trust.EXTERNAL, now, reach)
        self.assistant.context.note_tool(turn.user, turn.turn_id, ContextItem(output, origin))

    # Waiting for a person

    def _pause(self, turn: Turn, calls: Sequence[ToolCall], decision: Any) -> TurnResult:
        first = calls[0]
        state = {
            "query": turn.query,
            "transcript": turn.transcript,
            "tools": turn.tools,
            "rounds": turn.rounds,
            "tainted": turn.tainted,
            "reach": int(turn.reach),
            "model": turn.model,
            "provider": turn.provider,
            "calls": [{"capability": call.capability, "arguments": call.arguments, "call_id": call.call_id} for call in calls],
        }
        self.store.execute(
            "INSERT INTO paused_turns (turn_id, user, conversation_id, approval_id, capability, arguments, reason, state, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'waiting', ?) "
            "ON CONFLICT(turn_id) DO UPDATE SET approval_id = excluded.approval_id, capability = excluded.capability, arguments = excluded.arguments, "
            "reason = excluded.reason, state = excluded.state, status = 'waiting'",
            (
                turn.turn_id, turn.user, turn.conversation_id, decision.approval_id, first.capability, json.dumps(first.arguments),
                decision.reason, json.dumps(state), self.assistant.clock(),
            ),
        )
        pending = {
            "approval_id": decision.approval_id,
            "turn_id": turn.turn_id,
            "capability": first.capability,
            "arguments": first.arguments,
            "reason": decision.reason,
        }
        notice = f"This needs your approval before it can continue: {first.capability} ({decision.reason})."
        return self._finish(turn, "awaiting_approval", notice, pending)

    def pending(self, user: str) -> list[dict[str, Any]]:
        rows = self.store.query(
            "SELECT turn_id, conversation_id, approval_id, capability, arguments, reason, created_at FROM paused_turns "
            "WHERE user = ? AND status = 'waiting' ORDER BY created_at",
            (user,),
        )
        return [{**dict(row), "arguments": json.loads(row["arguments"])} for row in rows]

    def turn_for_approval(self, user: str, approval_id: str) -> str | None:
        row = self.store.one("SELECT turn_id FROM paused_turns WHERE user = ? AND approval_id = ? AND status = 'waiting'", (user, approval_id))
        return row["turn_id"] if row else None

    def resume(self, user: str, turn_id: str, approved: bool = True, learn: str = "now") -> TurnResult:
        """Carry on after a person decided. The broker is asked again, not assumed.

        Taken by one caller only, so two clicks cannot run an action twice.
        """
        with self.store.transaction() as connection:
            cursor = connection.execute("UPDATE paused_turns SET status = 'running' WHERE turn_id = ? AND user = ? AND status = 'waiting'", (turn_id, user))
        if cursor.rowcount == 0:
            raise TurnNotFound(f"No turn {turn_id} is waiting for a decision")
        row = self.store.one("SELECT * FROM paused_turns WHERE turn_id = ? AND user = ?", (turn_id, user))
        state = json.loads(row["state"])
        turn = Turn(
            user, state["query"], row["conversation_id"], turn_id, tainted=state["tainted"], reach=Reach(state["reach"]),
            transcript=state["transcript"], said=[], tools=state["tools"], rounds=state["rounds"], model=state["model"], provider=state["provider"],
        )
        try:
            result = self._carry_on(turn, state, row, approved)
        except BaseException:
            # Put it back, so a person can decide again. Nothing runs twice: see __init__.
            self.store.execute("UPDATE paused_turns SET status = 'waiting' WHERE turn_id = ? AND user = ?", (turn_id, user))
            raise
        if result.status != "awaiting_approval":
            self.store.execute("DELETE FROM paused_turns WHERE turn_id = ? AND user = ?", (turn_id, user))
            self._learn_after(result, state["query"], learn)
        return result

    def _carry_on(self, turn: Turn, state: dict[str, Any], paused: Any, approved: bool) -> TurnResult:
        if not approved:
            if paused["approval_id"]:  # or the question stays open in `approvals list` for nobody
                decide_approval(self.assistant, paused["approval_id"], False, turn.user)
            return self._finish(turn, "done", f"Stopped. You declined {paused['capability']}, so it did not run.")
        try:
            self._build(turn)  # afresh, so anything forgotten while waiting stays forgotten
        except ContextTooSmall as error:
            return self._finish(turn, "failed", str(error))
        calls = [ToolCall(item["capability"], item["arguments"], item["call_id"]) for item in state["calls"]]
        return final_result(self._advance(turn, False, calls))

    # Small or full

    def _decide(self, turn: Turn) -> str:
        """Ask a model what software could not tell. Staying small is the safe answer."""
        request = ModelRequest(
            role="decision", messages=[{"role": "system", "content": DECIDE_PROMPT}, {"role": "user", "content": turn.query}],
            reach=turn.reach, max_tokens=8, temperature=0.0,
        )
        try:
            answer = self.assistant.router.complete(request).text.strip().lower()
        except Exception:
            return "small"
        return "full" if answer.startswith("full") else "small"

    def _plan(self, turn: Turn) -> dict[str, Any]:
        planner = getattr(self.assistant, "planner", None)
        if planner is None:
            raise ValueError("Handing work to a task needs the planner part")
        # The goal is a person's own words, so it goes no further than theirs may.
        return planner.plan(turn.user, turn.query, reach=turn.reach)

    def _hand_off(self, turn: Turn, kind: str) -> TurnResult | None:
        tasks = getattr(self.assistant, "tasks", None)
        if tasks is None:
            return None
        if kind == "unclear":
            kind = self._decide(turn)
        if kind != "full":
            return None
        try:
            plan = self._plan(turn)
            task_id = str(tasks.submit(turn.user, plan))
        except Exception as error:
            # Answering directly beats dropping the request.
            self.assistant.audit.record("conversation", "conversation.handoff_failed", f"turn:{turn.turn_id}", {"error": type(error).__name__})
            return None
        self.store.execute(
            "INSERT INTO handed_tasks (task_id, user, conversation_id, turn_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, turn.user, turn.conversation_id, turn.turn_id, self.assistant.clock()),
        )
        return replace(self._finish(turn, "task", self._handed_notice(tasks, task_id, plan)), task_id=task_id)

    @staticmethod
    def _handed_notice(tasks: Any, task_id: str, plan: dict[str, Any]) -> str:
        """What a person has to know, since a task can otherwise just sit there."""
        words = [f"This is a larger piece of work, so it is now task {task_id}."]
        needed = [entry.get("command") or entry.get("capability") for entry in plan.get("missing_grants", [])]
        if needed:
            words.append("It will stop at any step nothing allows yet. To allow them: " + "; ".join(needed) + ".")
        if tasks.runner() is None:
            words.append("Nothing is running tasks yet. `isharded assistant serve` or `task work` will run it.")
        return " ".join(words)

    # Learning

    def _learn_after(self, result: TurnResult, said: str, learn: str) -> None:
        if learn == "off" or result.status != "done" or len(said.split()) < LEARN_MIN_WORDS:
            return
        if learn == "background":
            if self.learner is None:
                self.learner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="assistant-learn")
            self.learner.submit(self.learn, result, said)
        else:
            self.learn(result, said)

    def learn(self, result: TurnResult, said: str) -> list[str]:
        """Let a model propose what to keep. Its proposals are claims, and are stored as such.

        The call goes no further than the turn did, since the reply may hold what the
        turn's text allowed. From a tainted turn the proposals are EXTERNAL, or a
        poisoned page could write itself into the memory as the model's own thought.
        """
        request = ModelRequest(
            role="worker",
            messages=[
                {"role": "system", "content": LEARN_PROMPT},
                {"role": "user", "content": f"The person said:\n{said}\n\nThe reply was:\n{result.text}"},
            ],
            reach=result.reach,
            max_tokens=300,
            temperature=0.0,
        )
        try:
            reply = self.assistant.router.complete(request)
            proposals = loose_json(reply.text)
        except Exception as error:
            self.assistant.audit.record("conversation", "conversation.learn_failed", f"turn:{result.turn_id}", {"error": type(error).__name__})
            return []
        if not isinstance(proposals, list):
            return []
        trust = Trust.EXTERNAL if result.tainted else Trust.MODEL
        kept = []
        for proposal in proposals[:MAX_LEARNED]:
            memory_id = self._keep(result, proposal, reply.model, trust)
            if memory_id:
                kept.append(memory_id)
        return kept

    def _keep(self, result: TurnResult, proposal: Any, model: str, trust: Trust) -> str | None:
        memory = self.assistant.memory
        if not isinstance(proposal, dict):
            return None
        text = str(proposal.get("text") or "").strip()[:MAX_LEARNED_CHARS]
        kind = proposal.get("kind") if proposal.get("kind") in KINDS else "user"
        subject = proposal.get("subject")
        if not text or memory.holds(result.user, text):
            return None
        origin = Provenance("conversation", result.turn_id, model, trust, self.assistant.clock(), result.reach)
        return memory.remember(result.user, kind, text, origin, subject=str(subject) if subject else None)

    def wait_for_learning(self) -> None:
        if self.learner is not None:
            self.learner.shutdown(wait=True)
            self.learner = None


def attach(assistant: Any) -> None:
    assistant.conversation = Conversation(assistant)


def detach(assistant: Any) -> None:
    assistant.conversation.wait_for_learning()


def add_commands(commands: Any) -> None:
    chat = commands.add_parser("chat", help="talk with the assistant here in the terminal")
    chat.add_argument("--conversation", default=None, help="continue this conversation")
    add_user_option(chat)
    chat.set_defaults(run=plainly(run_chat))
    say = commands.add_parser("say", help="say one thing and print the reply")
    say.add_argument("text", help="what to say")
    say.add_argument("--conversation", default=None, help="continue this conversation")
    say.add_argument("--no-learn", action="store_true", help="do not propose memories from this turn")
    add_user_option(say)
    say.set_defaults(run=plainly(run_say))
    resume = commands.add_parser("resume", help="carry on a turn that was waiting for your approval")
    resume.add_argument("turn", help="the turn's id")
    resume.add_argument("--decline", action="store_true", help="do not do what it asked; end the turn")
    add_user_option(resume)
    resume.set_defaults(run=plainly(run_resume))
    gateway_cli.add_commands(commands)  # the server is not a part, so its commands join here


def footer(result: TurnResult) -> str:
    parts = [f"turn {result.turn_id}"]
    if result.model:
        parts.insert(0, f"{result.provider}/{result.model}")
    if result.tools:  # a small model can invent what a tool would have said, and it reads just the same
        parts.append("used " + " ".join(result.tools))
    if result.tainted:
        parts.append("tainted")
    return "[" + ", ".join(parts) + "]"


def speak(assistant: Any, user: str, text: str, conversation_id: str | None, learn: str) -> TurnResult:
    """Print a turn as it arrives."""
    result = None
    printed = False
    for item in assistant.conversation.stream(user, text, conversation_id, learn="off"):
        if isinstance(item, TurnResult):
            result = item
        else:
            printed = True
            emit(item, end="")
    assert result is not None
    if not printed:
        emit(result.text)
    else:
        emit()
        if result.status != "done":
            emit(result.text.rsplit("\n\n", 1)[-1])
    emit(footer(result))
    assistant.conversation._learn_after(result, text, learn)
    return result


def run_say(assistant: Any, args: argparse.Namespace) -> int:
    result = speak(assistant, args.user, args.text, args.conversation, "off" if args.no_learn else "now")
    if result.status == "awaiting_approval":
        emit(f"When decided: isharded assistant resume {result.turn_id}")
    return 0 if result.status in ("done", "task", "awaiting_approval") else 1


def run_resume(assistant: Any, args: argparse.Namespace) -> int:
    result = assistant.conversation.resume(args.user, args.turn, approved=not args.decline)
    emit(result.text)
    emit(footer(result))
    return 0 if result.status in ("done", "awaiting_approval") else 1


def run_chat(assistant: Any, args: argparse.Namespace) -> int:
    conversation_id = args.conversation or uuid.uuid4().hex
    emit(f"Talking with {assistant.identity.name}. /new starts over, /quit leaves.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            emit()
            return 0
        if line in ("/quit", "/exit"):
            return 0
        if line == "/new":
            conversation_id = uuid.uuid4().hex
            continue
        if not line:
            continue
        result = speak(assistant, args.user, line, conversation_id, "now")
        while result.status == "awaiting_approval" and result.pending:
            result = ask_person(assistant, args.user, result)


def ask_person(assistant: Any, user: str, result: TurnResult) -> TurnResult:
    pending = result.pending or {}
    emit(f"Allow {pending.get('capability')} with {json.dumps(pending.get('arguments'))}? [y/N] ", end="")
    try:
        approved = input().strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        approved = False
    if pending.get("approval_id") and not decide_approval(assistant, pending["approval_id"], approved, user) and approved:
        emit("That approval was no longer waiting, so the action was not run.")
        approved = False
    outcome = assistant.conversation.resume(user, result.turn_id, approved=approved)
    emit(outcome.text)
    emit(footer(outcome))
    return outcome
