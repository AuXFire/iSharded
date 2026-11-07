"""A model proposes a plan. Software decides whether it may run.

Nothing a planning model writes is believed. Its plan must parse, pass the same
checks the engine applies, and name only capabilities that exist. It gets two
chances to fix what is wrong, and after that the plan is refused. A plan changes
nothing about permissions: the broker still judges every step.

Asking for a list of missing grants up front means a person answers once at the
start, instead of being stopped halfway through a task that already did half of
its work.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .contracts import Capability, Effect, ModelRequest, Reach
from .mind import LOCAL_USER
from .providers import loose_json
from .router import AllFailed, Refused
from .tasks import count_steps, plan_errors, references_in, step_inputs, steps_referenced

MAX_PLAN_STEPS = 50
MAX_ASKS = 3  # the first try and two re-asks
PLAN_TOKENS = 4096
ROLES = {"conversation", "planning", "coding", "worker", "decision", "vision", "embedding", "verification"}
EXIT_CODE_TOOLS = {"shell.run", "git.branch", "git.commit", "git.push"}
GRANT_SECONDS = 24 * 3600.0
SMALL_GOAL_WORDS = 24
SEQUENCE_WORDS = re.compile(
    r"\b(then|after|afterwards|before|finally|first|second|next|once|until|while|each|every|all|daily|weekly|hourly|"
    r"schedule|repeat|refactor|implement|migrate|deploy|develop|design|investigate|plan|project|feature)\b",
    re.IGNORECASE,
)
LIST_MARK = re.compile(r"^\s*(?:[-*]|\d+[.)])\s", re.MULTILINE)

PROMPT = """You turn a goal into a JSON plan. Reply with one JSON object and nothing else.

{"title": "short name", "steps": [ ... ]}

Every step has a unique "id" (letters, digits, dash, underscore) and a "kind". It may have "depends_on": ["id"], "retries": 0-20, "on_failure": "stop", "continue", or "escalate".

Kinds:
- {"id": "a", "kind": "tool", "capability": "NAME", "arguments": {...}} calls one capability from the list below. Use no others.
- {"id": "b", "kind": "model", "role": "worker", "prompt": "..."} asks a model. Roles: worker, coding, planning, decision, verification.
- {"id": "c", "kind": "check", "check": {...}} is a mechanical test, one of:
  {"type": "exit_code", "step": "a", "equals": 0}
  {"type": "ok", "step": "a"}
  {"type": "regex", "text": "{{steps.a.output}}", "pattern": "..."}
  {"type": "json_schema", "text": "{{steps.a.output}}", "schema": {...}}
  {"type": "file_exists", "path": "..."}
- {"id": "d", "kind": "approval", "message": "..."} waits for a person.
- {"id": "e", "kind": "wait", "seconds": 60} waits.
- {"id": "f", "kind": "subtask", "inputs": {"x": "{{steps.a.output}}"}, "plan": {"steps": [ ... ]}} runs a nested plan, which reads {{inputs.x}}.

A step uses an earlier result by writing {{steps.ID.output}} or {{steps.ID.data.FIELD}}. Use at most 50 steps. Paths must be absolute. Steps with no dependency between them run at the same time.

Capabilities:
"""


class PlanRejected(ValueError):
    """Every try was refused. ``errors`` is what was wrong with the last one."""

    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message)
        self.errors = errors


@dataclass(frozen=True)
class Triage:
    direct: bool  # small enough to run without a plan
    reason: str


def triage(goal: str) -> Triage:
    """Whether something is small enough to run directly. No model, so always the same answer."""
    text = goal.strip()
    if not text:
        return Triage(False, "There is no goal yet.")
    if LIST_MARK.search(text) or "\n" in text:
        return Triage(False, "It has several lines or a list.")
    if len(text.split()) > SMALL_GOAL_WORDS:
        return Triage(False, f"It is longer than {SMALL_GOAL_WORDS} words.")
    if len(re.findall(r"[.!?;](?:\s|$)", text.rstrip(".!? "))) >= 1:
        return Triage(False, "It has more than one sentence.")
    found = SEQUENCE_WORDS.search(text)
    if found:
        return Triage(False, f"It says {found.group(0)!r}, which suggests several stages.")
    return Triage(True, "It is one short request.")


def parse_reply(text: str) -> Any:
    """The model's JSON, allowing for the fences and words around it that models add anyway.

    Whatever comes out still has to pass every check, so being generous here costs nothing.
    """
    found = loose_json(text)
    if found is None:
        raise ValueError("no complete JSON object was found in it")
    return found


def walk_steps(plan: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for step in plan.get("steps", []):
        if isinstance(step, Mapping):
            yield step
            if step.get("kind") == "subtask" and isinstance(step.get("plan"), Mapping):
                yield from walk_steps(step["plan"])


def only_steps(plan: Any) -> Any:
    """Keep what a model may decide. When and how far a plan runs is not up to it."""
    if not isinstance(plan, Mapping):
        return plan
    kept: dict[str, Any] = {key: plan[key] for key in ("title", "steps") if key in plan}
    if isinstance(kept.get("steps"), list):
        kept["steps"] = [
            {**step, "plan": only_steps(step["plan"])} if isinstance(step, Mapping) and step.get("kind") == "subtask" and "plan" in step else step
            for step in kept["steps"]
        ]
    return kept


def plan_problems(plan: Any, known: Mapping[str, Capability], capacity: Mapping[str, int]) -> list[str]:
    problems = plan_errors(plan, capacity)
    if problems:
        return problems  # the checks below rely on the plan having the right shape
    if count_steps(plan) > MAX_PLAN_STEPS:
        problems.append(f"A plan can have at most {MAX_PLAN_STEPS} steps.")
    for step in walk_steps(plan):
        if step.get("kind") == "tool" and step.get("capability") not in known:
            problems.append(f"Step {step.get('id')} uses {step.get('capability')!r}, which is not a capability here.")
        if step.get("kind") == "model":
            for role in [step.get("role", "worker"), *step.get("escalate_to", [])]:
                if role not in ROLES:
                    problems.append(f"Step {step.get('id')} uses role {role!r}. Roles are: {', '.join(sorted(ROLES))}.")
    return problems


def describe(capability: Capability) -> str:
    schema = capability.input_schema if isinstance(capability.input_schema, dict) else {}
    required = set(schema.get("required", []))
    fields = ", ".join(f"{name}{'' if name in required else '?'}" for name in schema.get("properties", {}))
    return f"- {capability.name}({fields}): {capability.summary}"


def has_a_check(step: Mapping[str, Any], steps: list[dict[str, Any]]) -> bool:
    if "expect" in step:
        return True
    for other in steps:
        if other.get("kind") != "check":
            continue
        check = other.get("check", {})
        if check.get("step") == step["id"] or step["id"] in steps_referenced(check) or step["id"] in other.get("depends_on", []):
            return True
    return False


def check_for(step: Mapping[str, Any]) -> dict[str, Any]:
    name = step.get("capability", "")
    if name == "files.write" and isinstance(step.get("arguments", {}).get("path"), str):
        return {"type": "file_exists", "path": step["arguments"]["path"]}
    return {"type": "exit_code" if name in EXIT_CODE_TOOLS else "ok", "step": step["id"]}


def free_id(base: str, steps: list[dict[str, Any]]) -> str:
    taken = {step["id"] for step in steps}
    name, number = base[:60], 2
    while name in taken:
        name, number = f"{base[:56]}-{number}", number + 1
    return name


def add_checks(plan: dict[str, Any], known: Mapping[str, Capability]) -> list[str]:
    """Give every consequential step a mechanical check, and make what follows wait for it."""
    added: list[str] = []
    steps = plan["steps"]
    for step in list(steps):
        if step.get("kind") == "subtask":
            added += add_checks(step["plan"], known)
            continue
        capability = known.get(step.get("capability", ""))
        if step.get("kind") != "tool" or capability is None or not capability.consequential or has_a_check(step, steps):
            continue
        check_id = free_id(f"check-{step['id']}", steps)
        followers = [other for other in steps if other is not step and needs(other, step["id"])]
        steps.insert(steps.index(step) + 1, {"id": check_id, "kind": "check", "depends_on": [step["id"]], "check": check_for(step)})
        for follower in followers:
            follower["depends_on"] = [*follower.get("depends_on", []), check_id]
        added.append(check_id)
    return added


def needs(step: Mapping[str, Any], name: str) -> bool:
    return name in step.get("depends_on", []) or name in steps_referenced(step_inputs(step))


def is_template(value: Any) -> bool:
    return isinstance(value, str) and bool(references_in(value))


class Planner:
    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant

    def triage(self, goal: str) -> Triage:
        return triage(goal)

    def _known(self) -> dict[str, Capability]:
        broker = getattr(self.assistant, "broker", None)
        return {capability.name: capability for capability in broker.capabilities()} if broker else {}

    def _capacity(self) -> Mapping[str, int]:
        tasks = getattr(self.assistant, "tasks", None)
        return getattr(tasks, "capacity", {})

    def plan(self, user: str, goal: str, reach: Reach = Reach.POOL, tainted: bool = False) -> dict[str, Any]:
        """A plan that passed every check, or PlanRejected. It does not run anything."""
        router = getattr(self.assistant, "router", None)
        if router is None:
            raise ValueError("Planning needs the router part, which is not installed.")
        known = self._known()
        catalog = "\n".join(describe(capability) for capability in known.values())
        messages: list[dict[str, Any]] = [{"role": "system", "content": PROMPT + catalog}, {"role": "user", "content": goal}]
        problems: list[str] = []
        for attempt in range(1, MAX_ASKS + 1):
            reply = router.complete(ModelRequest(role="planning", messages=list(messages), reach=reach, max_tokens=PLAN_TOKENS, temperature=0.1))
            candidate, problems = self._judge(reply.text, known)
            self.assistant.audit.record("planner", "plan.attempt", user, {"attempt": attempt, "accepted": not problems, "problems": len(problems)})
            if not problems:
                return self._finish(user, goal, candidate, known, reach, tainted)
            messages += [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": "That plan was refused. Fix these and reply with JSON only:\n" + "\n".join(f"- {problem}" for problem in problems[:20])},
            ]
        raise PlanRejected(f"The plan was refused {MAX_ASKS} times. The last problems: {' '.join(problems[:5])}", problems)

    def _judge(self, text: str, known: Mapping[str, Capability]) -> tuple[Any, list[str]]:
        try:
            parsed = parse_reply(text)
        except ValueError as error:
            return None, [f"The reply was not JSON: {error}."]
        candidate = only_steps(parsed)
        return candidate, plan_problems(candidate, known, self._capacity())

    def _finish(self, user: str, goal: str, plan: dict[str, Any], known: Mapping[str, Capability], reach: Reach, tainted: bool) -> dict[str, Any]:
        finished = json.loads(json.dumps(plan))  # its own copy, since checks are added to it
        finished["title"] = str(finished.get("title") or goal[:80])
        finished["goal"] = goal
        finished["reach"] = reach.name.lower()
        if tainted:
            finished["tainted"] = True
        finished["checks_added"] = add_checks(finished, known)
        finished["missing_grants"] = self.missing_grants(user, finished)
        problems = plan_problems(finished, known, self._capacity())
        if problems:  # our own additions must never produce something the engine would refuse
            raise PlanRejected("The plan broke after its checks were added. " + " ".join(problems), problems)
        return finished

    def missing_grants(self, user: str, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
        """What a person would have to allow, gathered so they are asked once."""
        known = self._known()
        broker = getattr(self.assistant, "broker", None)
        wanted: dict[tuple[str, str | None], dict[str, Any]] = {}
        for step in walk_steps(plan):
            capability = known.get(step.get("capability", ""))
            if step.get("kind") != "tool" or capability is None:  # with no broker, nothing is known
                continue
            arguments = step.get("arguments", {})
            scope = arguments.get(capability.scope_field) if capability.scope_field else None
            physical = Effect.PHYSICAL in capability.effects
            if not physical and not is_template(scope) and broker.covers(user, capability.name, arguments):
                continue
            entry = wanted.setdefault((capability.name, scope if isinstance(scope, str) else None), self._entry(user, capability, scope, physical))
            entry["steps"].append(step["id"])
            entry["uses"] += 1 + step.get("retries", 0)  # each try spends a use, so a retry must have one
        return list(wanted.values())

    @staticmethod
    def _entry(user: str, capability: Capability, scope: Any, physical: bool) -> dict[str, Any]:
        unknown = is_template(scope)
        needs_scope = capability.scope_field is not None
        target = "SCOPE" if unknown else str(scope) if needs_scope else ""
        command = f"isharded assistant grant add {capability.name} {target} --user {user}".replace("  ", " ")
        return {
            "capability": capability.name,
            "scope": None if unknown or not needs_scope else scope,
            "effects": sorted(effect.value for effect in capability.effects),
            "steps": [],
            "uses": 0,
            "note": (
                "Moves or powers hardware, so a person approves each action when it runs."
                if physical
                else "The scope is known only when it runs, so choose one that covers it."
                if unknown
                else ""
            ),
            "command": "" if physical else command,
        }

    def allow_missing(self, user: str, plan: Mapping[str, Any], by: str, seconds: float = GRANT_SECONDS) -> dict[str, list[str]]:
        """One yes from a person for what a plan needs: exact scopes, a limited time, limited uses."""
        broker = self.assistant.broker
        now = self.assistant.clock()
        known = self._known()
        granted, left = [], []
        for entry in plan.get("missing_grants", []):
            capability = known.get(entry["capability"])
            if capability is None or Effect.PHYSICAL in capability.effects or (capability.scope_field and entry["scope"] is None):
                left.append(entry["capability"])
                continue
            granted.append(broker.add_grant(user, entry["capability"], entry["scope"], by, now + seconds, entry["uses"]))
        return {"granted": granted, "left_for_a_person": left}


def attach(assistant: Any) -> None:
    assistant.planner = Planner(assistant)


def add_commands(commands: Any) -> None:
    plan = commands.add_parser("plan", help="ask the planning model for a plan, and check it; runs nothing")
    plan.add_argument("goal", help="what you want done")
    plan.add_argument("--user", default=LOCAL_USER, help="who this is for; decides which grants count as missing")
    plan.add_argument("--out", default=None, help="write the plan here, for task run; default prints it")
    plan.set_defaults(run=run_plan)


def run_plan(assistant: Any, args: Any) -> int:
    judged = triage(args.goal)
    print(f"{'Small enough to run directly' if judged.direct else 'Needs a plan'}: {judged.reason}")
    try:
        plan = assistant.planner.plan(args.user, args.goal)
    except (ValueError, Refused, AllFailed) as error:
        print(error)
        return 1
    text = json.dumps(plan, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}.")
    else:
        print(text)
    for entry in plan["missing_grants"]:
        print(f"Missing: {entry['command'] or entry['capability']}  {entry['note']}".rstrip())
    return 0
