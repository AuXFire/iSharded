"""What a person manages, over HTTP: the commands of the other parts, behind the gateway's keys.

Kept apart from gateway.py so the server's own file stays about talking to the assistant.
FastAPI reads route annotations from module globals, so the bodies live here too.

Three rules decide who may do what. A key reads its own data, and only the owner may
name someone else's. Anything that reaches this machine, spends what others share, or
gives a permission is for the owner. And anything that decides is done in the key
holder's own name, which never comes from a request, by someone who is a person.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .broker import is_person
from .coder import CoderError, Run
from .federation import FederationError, grouped, INVITE_SECONDS, Peer
from .mind import MACHINE_ACTORS
from .providers import ProviderRefused, READ_SECONDS
from .router import AllFailed, Refused
from .skills import SkillError, SkillVersion

MOST_ROWS = 500


def me(request: Request) -> str:
    return request.state.user


def owner_only(request: Request) -> None:
    if request.state.role != "owner":
        raise HTTPException(403, "Only the owner can do this")


def person(request: Request) -> None:
    """For a route that decides. A name the broker would not take for a person decides nothing, whoever holds its key."""
    owner_only(request)
    if not is_person(me(request)) or me(request).startswith(MACHINE_ACTORS):
        raise HTTPException(403, f"{me(request)} is not a person, and only a person can decide this")


def whose(request: Request, user: str | None) -> str:
    """Whose data a request means. A key means its own unless the owner names another."""
    if user is None or user == me(request):
        return me(request)
    owner_only(request)
    return user


def found(lookup: Callable[..., Any], *arguments: Any) -> Any:
    """A part refuses in its own words. Here that means there is no such thing."""
    try:
        return lookup(*arguments)
    except (CoderError, FederationError, SkillError) as error:
        raise HTTPException(404, str(error)) from None


def most(limit: int) -> int:
    return max(1, min(limit, MOST_ROWS))


OWNER = [Depends(owner_only)]
PERSON = [Depends(person)]


class GrantRequest(BaseModel):
    capability: str
    scope: str | None = None
    for_user: str | None = None
    expires_in: float | None = None
    uses: int | None = None


class ProviderRequest(BaseModel):
    name: str
    url: str
    mesh: bool = False
    key_env: str | None = None
    key_file: str | None = None
    reach: str | None = None
    tools: str = "auto"
    daily_tokens: int | None = None
    read_seconds: float = READ_SECONDS
    model: str | None = None


class RouteRequest(BaseModel):
    candidates: list[str] = Field(min_length=1)  # an empty route would quietly fall back to every provider


class PluginRequest(BaseModel):
    manifest: str
    effects: list[str] | None = None
    tool_effects: dict[str, list[str]] = {}
    env: list[str] = []


class ReasonRequest(BaseModel):
    reason: str = ""


class CommitRequest(BaseModel):
    message: str | None = None


class InviteRequest(BaseModel):
    name: str | None = None
    address: str | None = None
    minutes: float = Field(default=INVITE_SECONDS / 60, gt=0)


class AcceptRequest(BaseModel):
    code: str
    name: str | None = None
    address: str | None = None


class PeerApproval(BaseModel):
    fingerprint: str | None = None


class PlanRequest(BaseModel):
    goal: str


class AllowRequest(BaseModel):
    plan: dict[str, Any]
    for_user: str | None = None


def public_provider(provider: Any) -> dict[str, Any]:
    """Never the key, or where it is kept."""
    return {
        "name": provider.name,
        "kind": provider.kind,
        "reach": provider.reach.name.lower(),
        "url": getattr(provider, "base_url", ""),
        "model": getattr(provider, "default_model", None),
        "tools": getattr(provider, "tool_mode", None),
        "daily_tokens": getattr(provider, "daily_tokens", None),
    }


def public_version(version: SkillVersion) -> dict[str, Any]:
    return {
        "id": version.id,
        "label": version.label,
        "name": version.name,
        "number": version.number,
        "user": version.user,
        "task_type": version.task_type,
        "state": version.state,
        "trust": version.trust.value,
        "capabilities": list(version.capabilities),
        "instructions": version.instructions,
        "applicability": version.applicability,
        "stop_conditions": version.stop_conditions,
        "tests": list(version.tests),
        "author": version.author,
        "source_task": version.source_task,
        "source_tainted": version.source_tainted,
        "review": version.review,
        "reason": version.reason,
    }


def public_run(run: Run) -> dict[str, Any]:
    """The diff and test output are shown by asking for one run."""
    return {
        "id": run.id,
        "user": run.user,
        "repo": str(run.repo),
        "goal": run.goal,
        "branch": run.branch,
        "base": run.base,
        "state": run.state,
        "round": run.round,
        "max_rounds": run.max_rounds,
        "test_command": list(run.test_command),
        "languages": list(run.languages),
        "reach": run.reach.name.lower(),
        "review": run.review,
        "warnings": list(run.warnings),
        "note": run.note,
        "problem": run.problem,
        "commit": run.commit_id,
        "approval_id": run.approval_id,
    }


def public_peer(peer: Peer, federation: Any) -> dict[str, Any]:
    return {
        "name": peer.name,
        "state": peer.state,
        "fingerprint": grouped(peer.fingerprint),
        "address": peer.address,
        "offered": federation.offers_to(peer),
        "their_offers": list(peer.their_offers),
    }


def routes(assistant: Any) -> APIRouter:
    api = APIRouter(prefix="/assistant")

    def part(name: str) -> Any:
        installed = getattr(assistant, name, None)
        if installed is None:
            raise HTTPException(404, f"The {name} part is not installed")
        return installed

    # Who the assistant is

    @api.get("/identity/history")
    def identity_history(limit: int = 50) -> dict[str, Any]:
        return {"history": assistant.identity.history(most(limit))}

    @api.get("/identity/relationship")
    def relationship_history(request: Request, user: str | None = None, limit: int = 50) -> dict[str, Any]:
        return {"history": assistant.identity.relationship_history(whose(request, user), most(limit))}

    # Permissions

    @api.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "capabilities": [
                {"name": item.name, "summary": item.summary, "effects": sorted(effect.value for effect in item.effects), "scope": item.scope_field}
                for item in part("broker").capabilities()
            ]
        }

    @api.get("/grants")
    def grants(request: Request, user: str | None = None, ended: bool = False) -> dict[str, Any]:
        # Like the audit trail: the owner reads everyone's unless they name one person.
        everyone = user is None and request.state.role == "owner"
        return {"grants": part("broker").grants(None if everyone else whose(request, user), ended)}

    @api.post("/grants", dependencies=PERSON)
    def grant_add(body: GrantRequest, request: Request) -> dict[str, Any]:
        ends = assistant.clock() + body.expires_in if body.expires_in is not None else None
        return {"id": part("broker").add_grant(body.for_user or me(request), body.capability, body.scope, me(request), ends, body.uses)}

    @api.post("/grants/{grant_id}/revoke", dependencies=PERSON)
    def grant_revoke(grant_id: str, request: Request) -> dict[str, Any]:
        if not part("broker").revoke_grant(grant_id, me(request)):
            raise HTTPException(404, f"{grant_id} is not a live grant")
        return {"revoked": True}

    # Planning

    @api.post("/plan")
    def plan(body: PlanRequest, request: Request) -> dict[str, Any]:
        user = me(request)
        try:
            return part("planner").plan(user, body.goal, reach=assistant.context.default_reach(user))  # a person's words go no further than theirs may
        except (AllFailed, Refused) as error:
            raise HTTPException(503, str(error)) from None

    @api.post("/plan/allow", dependencies=PERSON)
    def plan_allow(body: AllowRequest, request: Request) -> dict[str, list[str]]:
        return part("planner").allow_missing(body.for_user or me(request), body.plan, me(request))

    # Models

    @api.get("/providers", dependencies=OWNER)
    def providers(check: bool = False) -> dict[str, Any]:
        rows = []
        for provider in part("providers").all():
            rows.append({**public_provider(provider), **({"health": provider.health()} if check else {})})
        return {"providers": rows}

    @api.post("/providers", dependencies=PERSON)
    def provider_add(body: ProviderRequest, request: Request) -> dict[str, Any]:
        if body.key_env and body.key_file:
            raise HTTPException(400, "Give one of key_env and key_file")
        key = f"env:{body.key_env}" if body.key_env else f"file:{body.key_file}" if body.key_file else None
        added = part("providers").add(
            body.name, body.url, mesh=body.mesh, key=key, reach=body.reach, tool_mode=body.tools,
            daily_tokens=body.daily_tokens, read_seconds=body.read_seconds, model=body.model, by=me(request),
        )
        return public_provider(added)

    def named_provider(name: str) -> Any:
        provider = part("providers").get(name)
        if provider is None:
            raise HTTPException(404, f"There is no provider {name}")
        return provider

    @api.delete("/providers/{name}", dependencies=PERSON)
    def provider_remove(name: str, request: Request) -> dict[str, Any]:
        named_provider(name)
        try:
            assistant.providers.remove(name, by=me(request))
        except ProviderRefused as error:  # one a part registered, which is not this install's to remove
            raise HTTPException(409, str(error)) from None
        return {"removed": True}

    @api.post("/providers/{name}/test", dependencies=OWNER)
    def provider_test(name: str) -> dict[str, Any]:
        return named_provider(name).health()

    @api.get("/routes")
    def model_routes() -> dict[str, Any]:
        return {"routes": assistant.router.routes()}

    @api.put("/routes/{role}", dependencies=PERSON)
    def route_set(role: str, body: RouteRequest, request: Request) -> dict[str, Any]:
        assistant.router.set_route(role, body.candidates, by=me(request))
        return {"routes": assistant.router.routes()}

    # Skills

    def own_version(request: Request, reference: str, user: str | None) -> SkillVersion:
        # An id finds a version whoever it belongs to, so that has to be asked here.
        who = whose(request, user)
        version = found(part("skills").version, reference, who)
        if version.user != who:
            raise HTTPException(404, f"There is no skill version {reference}")
        return version

    @api.get("/skills")
    def skills(request: Request, user: str | None = None) -> dict[str, Any]:
        return {"skills": [public_version(version) for version in part("skills").versions(whose(request, user))]}

    @api.get("/skills/{reference}")
    def skill_show(reference: str, request: Request, user: str | None = None) -> dict[str, Any]:
        version = own_version(request, reference, user)
        rate, runs = assistant.skills.rate(version.id, assistant.skills.window)
        return {**public_version(version), "success_rate": rate, "runs": runs, "evaluation": assistant.skills.latest_evaluation(version.id)}

    @api.post("/skills/{reference}/evaluate", dependencies=PERSON)
    def skill_evaluate(reference: str, request: Request, user: str | None = None) -> dict[str, Any]:
        return assistant.skills.evaluate(own_version(request, reference, user).id)

    @api.post("/skills/{reference}/promote", dependencies=PERSON)
    def skill_promote(reference: str, request: Request, user: str | None = None) -> dict[str, Any]:
        return public_version(assistant.skills.promote(own_version(request, reference, user).id, by=me(request)))

    @api.post("/skills/{reference}/reject", dependencies=PERSON)
    def skill_reject(reference: str, request: Request, body: ReasonRequest = ReasonRequest(), user: str | None = None) -> dict[str, Any]:
        return public_version(assistant.skills.reject(own_version(request, reference, user).id, body.reason, by=me(request)))

    # Plugins

    @api.get("/plugins", dependencies=OWNER)
    def plugins() -> dict[str, Any]:
        return {"plugins": part("plugins").list()}

    @api.post("/plugins", dependencies=PERSON)
    def plugin_add(body: PluginRequest, request: Request) -> dict[str, Any]:
        # Its programs run as this process, so installing one is trusting its code.
        return {"capabilities": part("plugins").add(body.manifest, me(request), body.effects, body.tool_effects, body.env)}

    @api.delete("/plugins/{name}", dependencies=PERSON)
    def plugin_remove(name: str, request: Request) -> dict[str, Any]:
        if not part("plugins").remove(name, me(request)):
            raise HTTPException(404, f"There is no plugin {name}")
        return {"removed": True}

    # The coder

    @api.get("/code/runs")
    def code_runs(request: Request, user: str | None = None) -> dict[str, Any]:
        return {"runs": [public_run(run) for run in part("coder").runs(whose(request, user))]}

    @api.get("/code/runs/{run_id}")
    def code_show(run_id: str, request: Request, user: str | None = None, output: bool = False) -> dict[str, Any]:
        shown = found(part("coder").show, run_id, whose(request, user))
        tests = shown["tests"] if output else [{key: value for key, value in row.items() if key != "output"} for row in shown["tests"]]
        return {**public_run(shown["run"]), "diff": shown["diff"], "tests": tests, "models": shown["models"]}

    @api.post("/code/runs/{run_id}/approve", dependencies=PERSON)
    def code_approve(run_id: str, request: Request, body: CommitRequest = CommitRequest(), user: str | None = None) -> dict[str, Any]:
        who = whose(request, user)
        found(part("coder").run, run_id, who)
        return public_run(assistant.coder.approve(run_id, body.message, by=me(request), user=who))

    @api.post("/code/runs/{run_id}/reject", dependencies=PERSON)
    def code_reject(run_id: str, request: Request, body: ReasonRequest = ReasonRequest(), user: str | None = None) -> dict[str, Any]:
        who = whose(request, user)
        found(part("coder").run, run_id, who)
        return public_run(assistant.coder.reject(run_id, body.reason, by=me(request), user=who))

    # Peers. A pairing is between two owners, so a member has no part in it.

    @api.get("/peers", dependencies=OWNER)
    def peers() -> dict[str, Any]:
        federation = part("federation")
        return {"fingerprint": grouped(federation.identity.fingerprint), "peers": [public_peer(peer, federation) for peer in federation.peers()]}

    @api.post("/peers/invite", dependencies=PERSON)
    def peer_invite(body: InviteRequest) -> dict[str, Any]:
        federation = part("federation")
        return {"code": federation.invite(body.name, body.address, body.minutes * 60), "fingerprint": grouped(federation.identity.fingerprint)}

    @api.post("/peers/accept", dependencies=PERSON)
    def peer_accept(body: AcceptRequest) -> dict[str, Any]:
        federation = part("federation")
        peer = federation.accept(body.code, body.name, body.address)
        return {**public_peer(peer, federation), "own_fingerprint": grouped(federation.identity.fingerprint)}

    @api.post("/peers/{name}/approve", dependencies=PERSON)
    def peer_approve(name: str, body: PeerApproval = PeerApproval()) -> dict[str, Any]:
        federation = part("federation")
        found(federation.named, name)
        return public_peer(federation.approve(name, body.fingerprint), federation)

    @api.post("/peers/{name}/revoke", dependencies=PERSON)
    def peer_revoke(name: str) -> dict[str, Any]:
        federation = part("federation")
        found(federation.named, name)
        return public_peer(federation.revoke(name), federation)

    return api
