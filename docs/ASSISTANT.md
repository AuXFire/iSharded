# The assistant

ShardMesh runs one model across many machines. The assistant runs on top of that: a persistent identity that remembers, learns tested skills, runs durable tasks, uses tools under strict permissions, and works with other owners' installs without merging with them.

It is a separate process that each person runs for themselves, `shardmesh assistant serve`. The mesh is one of its model providers, reached through the OpenAI-compatible API, so the mesh core never depends on it and a person's memories never live on the mesh.

## The rule everything follows

The assistant owns its state, decisions, permissions, and task history. Models, tools, and peers provide capabilities and hold no authority. Any of them can be replaced without touching the rest.

```
Interfaces     command line, web page, any OpenAI client, voice
Gateway        sessions, per-user keys, separated users
Core           identity, permissions, audit                 identity broker store
Kernel         tasks, planning, context, memory, scheduling tasks planner context memory knowledge
Broker         the only way anything uses a tool            broker
Providers      mesh models, other models, tools, peers      providers router tools plugins federation voice
Hardware       this machine, the mesh, other owners' installs by agreement
```

## Parts

A part is a module in `shardmesh/assistant/` with `attach(assistant)`, and optionally `add_commands(subparsers)` and `routes(assistant)`. `app.py` loads whichever parts exist, in the order of `PARTS`. Parts meet only through `contracts.py`. Each part owns its tables through `store.migrate`, and writes to the audit trail through `assistant.audit`.

| Part | Attaches | Does |
|---|---|---|
| `identity` | `assistant.identity` | Name, voice, style, commitments, and a relationship per user. The same across any model. Presentation only: changing it cannot change a permission |
| `memory` | `assistant.memory` | Conversation, episodic, semantic, procedural, user, and system memory, with provenance and trust on every row |
| `knowledge` | `assistant.library` | Indexes folders the owner authorized, incrementally, and searches them with citations |
| `broker` | `assistant.broker` | Deny by default. Grants, approvals, taint, audit |
| `tools` | | The built-in capabilities |
| `plugins` | `assistant.plugins` | Out-of-process tools from a manifest, and MCP servers, all behind the broker |
| `sandbox` | `assistant.sandbox` | Where generated code runs, never in this process |
| `providers` | `assistant.providers` | Model backends: the mesh, any OpenAI-compatible server, a peer |
| `router` | `assistant.router` | Picks a model for a role from measured results, reach, budget, and what is up |
| `context` | `assistant.context` | Decides what a model sees, keeps provenance, labels untrusted text, and reports taint |
| `conversation` | `assistant.conversation` | One turn: context, model, tool calls through the broker, attribution, memory |
| `tasks` | `assistant.tasks` | The durable task engine |
| `planner` | `assistant.planner` | A model proposes a plan. Software validates it and the engine runs it |
| `skills` | `assistant.skills` | Versioned skills, promoted only by passing tests |
| `coder` | `assistant.coder` | Supervised coding on a branch in an isolated copy |
| `federation` | `assistant.federation` | Narrow, signed, revocable work between independent installs |
| `voice` | `assistant.voice` | Speech in and out, local and optional |
| `gateway` | | The HTTP server and web page, holding every part's routes |

## What keeps it safe

These hold for every part, and each has a test.

- **Deny by default.** A capability does nothing until a grant covers this user, this capability, and this scope.
- **Declared effects.** Every tool states what it can do, what it reads, what it needs, and whether it can run twice. Read-only is treated differently from anything consequential.
- **Approval is bound to one action.** An approval covers the exact arguments, by hash, and is used once.
- **Taint.** Text from a file, page, tool result, or peer is data, labeled as such, and marks the context as tainted. A tainted context cannot use a standing grant for a consequential effect: a person has to approve that exact action. So a poisoned web page can be read, but it cannot send anything or run anything by itself.
- **Reach.** Every item records how far it may travel: this machine, the owner's mesh, a peer, or a cloud. A prompt goes no further than its strictest item, and the router refuses a provider beyond that.
- **Learning grants nothing.** A skill can only use capabilities the user has already granted.
- **Users are separated.** Every read of a user's data is scoped to that user in the query itself.
- **Generated code runs in the sandbox,** on a copy, never in this process.
- **Every decision is audited,** refusals included, in a hash-chained record that shows tampering.
- **Remote systems gain nothing by connecting.** A peer can call only what was offered to it and granted, and its text is always tainted.
