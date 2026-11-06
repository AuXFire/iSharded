# The assistant

ShardMesh runs one model across many machines. The assistant runs on top of that: a persistent identity that remembers, learns tested skills, runs durable tasks, uses tools under strict permissions, and works with other owners' installs without merging with them.

It is a separate process that each person runs for themselves, `shardmesh assistant serve`. The mesh is one of its model providers, reached through the OpenAI-compatible API, so the mesh core never depends on it and a person's memories never live on the mesh.

## Quick start

Steps 2 to 7 were run in this order on Windows, from outside the repository, against a real model on a real mesh. It needs a model to talk to, and any OpenAI-compatible server will do in place of the mesh.

**1. Install it,** from a checkout of this repository. It fetches `setuptools` 69 or later to build, if you have an older one. This step alone was not run as written: the machine had an older `setuptools` and no downloading was allowed, so the same files were copied by the rules in `pyproject.toml`.

```sh
python -m pip install .
```

**2. Start a model.** This makes a mesh of one on this machine, with a folder of GGUF files. It prints where it is and keeps running, so leave it open and use a second terminal. `shardmesh up` keeps the owner's key in `admin.key`, in the `up` folder under `ShardMesh` in your data folder.

```sh
shardmesh up --gguf-dir FOLDER --no-browser
```

Once it says it is up, load a model by its file name, from the second terminal or from the chat page `up` printed. Point `SHARDMESH_POOL_TOKEN_FILE` at `admin.key` first, which keeps the key off the command line. In PowerShell that is `$env:SHARDMESH_POOL_TOKEN_FILE = "ADMIN_KEY_FILE"` on a line of its own. A 0.6B model took about 20 seconds.

```sh
SHARDMESH_POOL_TOKEN_FILE=ADMIN_KEY_FILE shardmesh load Qwen3-0.6B-Q8_0.gguf --url http://127.0.0.1:8209
```

Do not use `up --model` for this yet. It asks once while the node is still starting, is told `no coordinator yet`, and does not ask again, so no model loads.

**3. Add it as a provider.** The assistant holds only the path to the key, never the key. The mesh's address is the one `up` printed.

```sh
shardmesh assistant provider add mesh http://127.0.0.1:8209 --mesh --key-file ADMIN_KEY_FILE
shardmesh assistant provider list
```

**4. Talk to it.** A chat in the terminal is `chat`, and one line is `say`. It keeps what is worth keeping, and each reply ends with who answered and which tools really ran.

```sh
shardmesh assistant say "My name is Sam and I keep bees on my roof."
shardmesh assistant recall bees
shardmesh assistant say "What do I keep on my roof?"
```

A model proposes what to remember, and a small one often proposes nothing useful. What you say yourself with `remember` is kept as your own words.

```sh
shardmesh assistant remember "I keep bees on my roof." --subject bees
```

**5. Grant a tool.** Until now the assistant could do nothing but talk. A grant says who, what, and how far. Here the assistant may read one folder, and nothing else.

```sh
shardmesh assistant grant add files.read NOTES_FOLDER
shardmesh assistant say "Use the files.read tool to read NOTES_FOLDER/notes.txt, then tell me what it says."
shardmesh assistant why
shardmesh assistant audit --limit 10
```

Look for `used files.read` at the end of the reply. A small model sometimes skips the tool and makes up what the file says, in the same confident words, and that line is how you tell. [Here is how often.](#how-a-small-model-does)

`why` shows what the model saw, and `audit` shows every decision the broker made. What it read is marked as outside text, so from then on that conversation cannot use a grant for anything with consequences without asking you. `approvals list` shows what is waiting, and `approvals approve ID` then `resume TURN` carries on.

**6. Run a task.** A task is a plan written as data, which survives a crash. This one reads a file, has a model summarize it, and then a mechanical check decides whether that worked. Save it as `plan.json`, with the path to your file:

```json
{
  "title": "Summarise my notes",
  "steps": [
    {"id": "read", "kind": "tool", "capability": "files.read", "arguments": {"path": "NOTES_FOLDER/notes.txt"}},
    {"id": "summary", "kind": "model", "prompt": "In one line, what does this say? {{steps.read.output}}", "max_tokens": 80},
    {"id": "said", "kind": "check", "check": {"type": "regex", "text": "{{steps.summary.output}}", "pattern": "(?i)key"}}
  ]
}
```

```sh
shardmesh assistant task run plan.json
shardmesh assistant task list
```

**7. Use the page.** Each person has their own key, in a file and never on the screen. Name the first user `owner`, since commands on this machine act for `owner`.

```sh
shardmesh assistant user add owner
shardmesh assistant serve
```

Open `http://127.0.0.1:8470/`, and give it the key from the file `user add` named. While it runs, `serve` also runs tasks. Anything OpenAI-compatible can use `http://127.0.0.1:8470/v1` with that key.

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

Every part attaches to one `Assistant`, which owns one folder. Commands take `--home` before the command to pick that folder, and it defaults to `ShardMesh/assistant` in your data folder, or `SHARDMESH_ASSISTANT_HOME`.

| Part | Attaches | Does |
|---|---|---|
| `identity` | `assistant.identity` | Name, voice, style, commitments, and a relationship per user. The same across any model. Presentation only: changing it cannot change a permission |
| `memory` | `assistant.memory` | Conversation, episodic, semantic, procedural, user, and system memory, with provenance and trust on every row |
| `knowledge` | `assistant.library` | Indexes folders the owner authorized, incrementally, and searches them with citations |
| `broker` | `assistant.broker` | Deny by default. Grants, approvals, taint, audit |
| `tools` | `assistant.builtins` | The built-in capabilities |
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

These hold for every part, and each has a test. `tests/test_assistant_system.py` runs them with every part attached and nothing faked but a model.

- **Deny by default.** A capability does nothing until a grant covers this user, this capability, and this scope.
- **Declared effects.** Every tool states what it can do, what it reads, what it needs, and whether it can run twice. Read-only is treated differently from anything consequential.
- **Approval is bound to one action.** An approval covers the exact arguments, by hash, and is used once.
- **Taint.** Text from a file, page, tool result, or peer is data, labeled as such, and marks the context as tainted. A tainted context cannot use a standing grant for a consequential effect: a person has to approve that exact action. So a poisoned web page can be read, but it cannot send anything or run anything by itself. A memory proposed on a tainted turn is stored as EXTERNAL, not MODEL, so a poisoned page cannot write itself into memory as the model's own thought, and whoever recalls that memory later is tainted too.
- **Reach.** Every item records how far it may travel: this machine, the owner's mesh, a peer, or a cloud. A prompt goes no further than its strictest item, and the router refuses a provider beyond that. A tool call carries that limit too, so a model cannot send or remember text past it.
- **Learning grants nothing.** A skill can only use capabilities the user has already granted.
- **Users are separated.** Every read of a user's data is scoped to that user in the query itself.
- **Generated code runs in the sandbox,** on a copy, never in this process. How strong that wall is depends on the machine: see [what the `process` level cannot stop](#what-is-not-built-or-proven).
- **Every decision is audited,** refusals included, in a hash-chained record that shows tampering.
- **Remote systems gain nothing by connecting.** A peer can call only what was offered to it and granted, and its text is always tainted.

Every command below goes after `shardmesh assistant`. Commands that act for a person take `--user NAME`, which defaults to `owner`. Over HTTP a key says who is calling, and `--user` does not exist there.

## Who the assistant is

The default name is Mesh, and the owner can change it. The same words reach every model, so changing models changes nobody's assistant. This is presentation only: the module does not import or touch the broker, so no change here can change a permission, and a test holds it to that.

| Command | Does |
|---|---|
| `identity show` | Shows the name, style, promises, and relationship. `--prompt` also prints the exact text models receive |
| `identity set` | `--name`, `--style`, `--voice`, `--commit TEXT` adds a promise to everyone, or with `--to-user` to `--user` alone, `--withdraw NUMBER` withdraws one, `--relationship TEXT` sets the note for `--user` |

Every change is kept. A withdrawn promise leaves the prompt and stays in the record.

## Memory

`messages` holds what was said, word for word, with the model that produced it. `memories` holds what was kept, with its kind (`episodic`, `semantic`, `procedural`, `user`, `system`), author, source, trust, reach, and when it was captured. Each query names the user in its own SQL, and a test traces the statements SQLite really ran to prove it.

A fact with a subject replaces the current fact on that subject only if it is at least as trusted (VERIFIED, then OWNER, then MODEL, then EXTERNAL). If it is not, it is kept for history already superseded, so a model's guess can never overwrite what a person said. Superseded rows stay for history and leave search. An expired fact does not block anything.

Search ranks by bm25, trust, and recency, with a half-life of 30 days that never takes a memory below half its score.

| Command | Does |
|---|---|
| `remember TEXT` | Keeps a fact as something you said. `--kind`, `--subject`, `--reach`, `--expires-days` |
| `recall QUERY` | Shows matches with where each came from. `--limit`, and `--said` to also search what was actually said |
| `forget ID` | Deletes a memory and its older versions. `--conversation ID` deletes a whole conversation instead |
| `confirm ID` | Vouches for a model's inference, which promotes it to OWNER. It cannot promote EXTERNAL text: say that in your own words with `remember` |
| `export FILE` | Writes everything held for one person as JSON |

`forget` is a hard delete in the sense a disk would agree with. A deleted row leaves its words in old index segments, free pages, and the write-ahead log, so it also uses `secure_delete`, optimizes the index, and checkpoints the log. A test scans the raw database files for the word afterwards. It does not reach a backup made earlier, or a conversation where the same words were said: `forget --conversation` covers that.

## The library

Folders the owner authorized, each with a reach. Everything from them is EXTERNAL, because a file can say anything.

| Command | Does |
|---|---|
| `library add PATH` | Authorizes a folder. `--reach` says how far its text may travel |
| `library index [ROOT]` | Brings the index up to date. Only changed files are read |
| `library search QUERY` | Searches with a citation for each result. `--limit` |
| `library list` | Shows the folders |
| `library remove ROOT` | Stops using a folder and deletes its index. The files are not touched |

Indexing compares size and modification time, then a sha256 of the content, and removes what was deleted. An unplugged drive, an unreadable folder, or a locked file is never mistaken for a deletion.

Markdown is chunked by heading, code by blank-line blocks, plain text by paragraph, and PDF by page if `pypdf` is installed. A citation is `path:line` or `path#page=N`, and a test opens the file to check it.

The walk never follows a symlink or a Windows junction, and never leaves the root through `..`. It always skips `.git`, `.env*`, `*.key`, `*.pem`, `id_rsa*`, `credentials*`, other private keys, `.ssh`, `.gnupg`, `.aws`, and binary files, and also `node_modules`, `__pycache__`, and virtual environments as noise. Names are matched in any case. Files over 32 MB are skipped.

If a provider offers the `embedding` role, search merges bm25 with vectors by reciprocal rank fusion. Vectors are made only by a provider within the folder's reach, and the query's reach limits who may see it. Otherwise search is bm25 alone.

Measured on this machine with 5,000 mixed files (26.5 MB, 27,257 chunks), while other work was running: a first index took 7.4 s and 10.3 s in two runs, and 29.7 s in one earlier run that did not repeat. Write batches stayed flat, so it does not slow as the index grows. With nothing changed it took 0.77 s, with 50 files changed 0.63 s, and with 50 deleted 2.06 s. Search had a median of 12 ms.

## Providers

| Command | Does |
|---|---|
| `provider add NAME URL` | `--mesh` for a ShardMesh pool, `--key-env NAME` or `--key-file PATH`, `--reach`, `--tools auto\|native\|prompt`, `--daily-tokens`, `--read-seconds`, `--model` |
| `provider list` | Shows providers and checks each is up. `--no-check` does not contact them |
| `provider remove NAME` | Removes one |

Reach defaults to `cloud`, the worst case, and to `pool` for a mesh. `machine` is refused for anything not on loopback. A person's words may go only as far as `pool` until they say otherwise, so a server added with no `--reach` is refused every request until either its reach or theirs changes.

The database holds the name of a variable or a file, never a key, and a key is read when it is used. Something that looks like a pasted key is refused. A bare UPPER_CASE name is a variable, and any other variable can be given as `env:name` once it is set. A key in an address is refused.

The mesh's chat API takes plain strings and has no tool calling, so tools go in the system prompt, as short signatures with an example built from a real tool, and calls are read from the reply. [How that wording was chosen](#how-a-small-model-does) matters more than it looks. Thinking is on only for `planning`, `coding`, and `verification`. A reply from the mesh is always streamed, even when nobody is watching it, because only a stream can tell a slow pool from a lost link. Other servers use native `tools` where they accept them, falling back to the prompt if they refuse, with capability names mapped since a function name cannot contain a dot.

Models write tool calls in many shapes, and all of these are understood: a ` ```tool ` or ` ```json ` fence, `<tool_call>` tags, no fence at all, single quotes, trailing commas, and arguments as a string. Only tools that were offered count. Text in reasoning never counts, and neither does code. A call with no fence must end the reply. A call cut off by the token limit is never run, because a truncated argument is a different argument.

A person never watches a call arrive. While tools are offered, a stream holds anything that could still be a call, from its opening `{`, and releases it as soon as it cannot be one. So a `{` that never closes shows only when the reply ends.

Silence is handled as failure. Every read has a limit, a failure sent inside a stream is seen, and a stream that ends without finishing is a failure and not a short answer.

A pool with its own authority is verified against `SHARDMESH_CA_FILE`, as nodes are. If a certificate cannot be verified, the error says antivirus may be replacing it, since some antivirus re-signs HTTPS even on loopback.

## The router

| Command | Does |
|---|---|
| `route set ROLE CANDIDATE...` | Sets a role's candidates, as `PROVIDER` or `PROVIDER/MODEL` |
| `route show` | Shows each role's candidates |
| `models` | Shows how each model has really done. `--role` for one role |

The roles are `conversation`, `planning`, `coding`, `worker`, `decision`, `vision`, `embedding`, and `verification`. A role with no route uses `conversation`, then every provider. `embedding` has no fallback.

A request first loses every candidate it may not use: reach, then back-off, then budget, and a refusal says why for each. What is left is tried best first by the Wilson lower bound on its success rate, then speed, then listed order. Given the same numbers it picks the same model.

A candidate counts as untested until it has 3 runs. Strictly zero would let one passing fault score nothing, and a good model would then never be tried again while anything else worked.

A failure backs off for 15 s, doubling to at most 5 minutes, and that survives a restart. A budget reserves the most a call could cost, so it is not overrun. A stream can move to another model only before any text has gone out.

`assistant.router.record_outcome(reply, role, success)` feeds back what checking the work found. A reply counts as a success when it arrives, and a failed check takes that back, so the rate means work that held up. It also takes the model name with `provider=`.

## Context

`assistant.context.build(user, query, role, budget_tokens)` returns `messages`, `items`, `dropped`, `tainted`, and `reach`. It draws on identity, recent conversation, memories, the library, and skills if `assistant.skills` has a `relevant`, `search`, or `match` method returning ContextItems.

| Command | Does |
|---|---|
| `why [TURN]` | Shows what a model saw on a turn, what was left out, and who answered. The default is the latest. `--json` prints it all |
| `context` | Shows how far a person's words may travel. `--reach` sets it. The default is `pool`, so to use a cloud model at all a person has to allow it |

Prompts have one system message and then alternating turns, which the strictest chat templates require. EXTERNAL text goes in the last user message between markers holding a random value that changes every turn, under a standing rule that text inside is data. When it does not fit, the lowest scored goes first, and one enormous item is left out without taking everything below it.

Taint and reach are worked out from what was really included, and both carry forward. A reply written while untrusted text was in view comes back as EXTERNAL history, and keeps the reach of what it was written from. A memory proposed on a tainted turn is stored as EXTERNAL, not MODEL. Otherwise a poisoned page would only need to survive one turn to lose its label, and a private file would only need to be summarized to leave the machine. This means a conversation stays tainted while such a message is in its history, so consequential actions in it need a person.

The record of a turn holds no text, so a forgotten memory does not survive in a log of when it was used. `why` looks the words up, and says so if they are gone.

## A turn

| Command | Does |
|---|---|
| `chat` | Talks in the terminal, and asks you about approvals. `--conversation` continues one. `/new` starts over and `/quit` leaves |
| `say TEXT` | Says one thing and prints the reply, then who answered, which tools really ran as `used files.read`, and `tainted` if it read outside text. `--conversation`, and `--no-learn` proposes no memories |
| `resume TURN` | Carries on a turn that was waiting. `--decline` ends it without acting |

What a person said is stored before anything else can fail. A model is offered only the tools this person holds a grant for, read-only ones first, within a quarter of the budget, and a tool call for anything else is just words. The loop runs at most 6 rounds of tools, then the model has to answer. Tool output is EXTERNAL whatever the tool claims, makes the rest of the turn tainted, and tightens its reach. It is cut for the model at a quarter of the budget and kept whole in the record.

When the broker answers ASK, the turn stops and is stored, and survives a restart. An answer given on the command line carries on when `resume` runs, or when the same approval is answered on the page. Declining with `resume --decline` also closes the question. Resuming asks the broker again and does not take the caller's word for it, one caller only can take a turn, and memory is rebuilt, so what was forgotten while waiting stays forgotten. A turn taken by a process that died can be taken again.

Small or full is decided by software where it can: `/task` and `/now`, scheduling words, and three or more listed steps decide it, and a short message or a question stays small. Only what is left goes to the `decision` role, and staying small is the answer if that fails. Full work goes through `assistant.planner` to `assistant.tasks`, with the plan held to the person's reach. The reply says which grants the plan still needs, and says so if nothing is running tasks yet. If the planner or the engine refuses, the request is answered directly.

After a finished turn the `worker` role may propose up to 3 memories. They go no further than the turn did, since the reply may hold what the turn's text allowed, and they are stored as MODEL, or EXTERNAL from a tainted turn.

## Permissions

Nothing runs until a grant covers this user, this capability, and this scope. An unregistered capability is denied.

| What is asked | Result |
|---|---|
| Nothing covers it | Denied, with the command that would allow it |
| A grant covers it, and it only reads | Allowed, even in a tainted context |
| A grant covers it, it has consequences, untainted | Allowed |
| A grant covers it, it has consequences, tainted | Asks: a person approves this exact action |
| It moves or powers hardware (`physical`) | Always asks, and can have no grant |
| The actor starts `peer:` | Treated as tainted, whatever the call says |
| The actor starts `skill:` and no grant covers it | Denied. A skill cannot even ask |

**Scope** is compared by kind, chosen from `Capability.scope_field` or from `format` in its input schema:

- **Paths** compare the fully resolved location, so `..`, symlinks, and junctions cannot climb out, and `/work-private` is not inside `/work`. A relative path is refused, because its meaning depends on the working directory. A grant settles its own path when it is made, so moving a link later cannot widen it. The tool receives the path that was judged.
- **URLs** compare the host. `example.com` covers that host, `*.example.com` covers its subdomains, and `localhost:9000` covers that port only.
- **Anything else** is a case-sensitive glob.

`*` covers everything in any kind, and you must say it on purpose. A capability with no `scope_field` takes no scope. A list argument needs every item covered.

**An approval** covers one exact action, by the SHA-256 of its canonical arguments, with a path counted as where it really leads. So a person is shown the real place, and moving a link after they said yes makes it a different action. It is bound to the user, capability, actor, and task, it works once, and changed arguments need a new one. Saying no withdraws a yes that was not used yet. Only a person can grant or approve: names starting `skill:`, `peer:`, `task:`, `model:`, `plugin:`, or `conversation` are refused.

`check` asks what would happen and can be repeated. It spends no use and no approval. It does write to the audit trail, and an ASK puts a question to a person, once: asking again while that is unanswered returns the same approval, and asking after a person said no puts a new question.

Every decision is audited, refusals included, with the argument hash and never the values. A result is trust EXTERNAL unless a tool proved otherwise, and only `verified` counts as proof.

### `grant add CAPABILITY [SCOPE]`

| Option | Meaning |
|---|---|
| `--user` | Who this is for. Default `owner` |
| `--by` | The person granting it, for the record. Default `owner` |
| `--expires-in` | Seconds until it ends. Default never |
| `--uses` | How many times it works. Default no limit |

### `grant list`

| Option | Meaning |
|---|---|
| `--user` | Only this user |
| `--all` | Include ended grants |
| `--capabilities` | Show the capabilities a grant can name, with their effects |

### `grant revoke ID`

Takes effect at once. `--by` is the person revoking it.

### `approvals list`

Shows what is waiting, with each action's arguments and hash. Arguments whose names look secret show as `***`.

| Option | Meaning |
|---|---|
| `--user` | Only this user |
| `--all` | Include decided and used approvals |

### `approvals approve ID` and `approvals deny ID`

`--by` is the person deciding.

### `audit`

Shows the newest entries, and checks that the record is intact. Exits 1 if it was changed.

| Option | Meaning |
|---|---|
| `--limit` | How many to show. Default 50 |
| `--actor` | Only this actor, such as `conversation`, `user:owner`, or `task:ID` |

## Tools

| Capability | Effects | Scope | Runs twice safely |
|---|---|---|---|
| `files.read` | read | `path` | yes |
| `files.list` | read | `path` | yes |
| `files.search` | read | `path` | yes |
| `files.write` | write_local | `path` | yes |
| `shell.run` | execute, write_local, network | `cwd` | no |
| `web.fetch` | network | `url` | yes |
| `memory.search` | read | none | yes |
| `memory.remember` | write_local | none | no |
| `library.search` | read | none | yes |
| `system.info` | read | none | yes |
| `mesh.status` | read | none | yes |
| `git.status` | read | `repo` | yes |
| `git.diff` | read | `repo` | yes |
| `git.log` | read | `repo` | yes |
| `git.branch` | write_local | `repo` | no |
| `git.commit` | write_local | `repo` | no |
| `git.push` | external | `repo` | no |
| `peer.send` | external | `peer` | no |
| `models.complete` | read | none | yes |
| `voice.transcribe` | read | `audio_path` | yes |
| `voice.speak` | write_local | none | no |

Effects are stated for the worst a call can do. `web.fetch` is network and not read, because a GET can carry data out in its URL, so it asks in a tainted context. `shell.run` admits to writing and the network, because the weakest sandbox stops neither.

**Files.** Paths must be absolute. No tool reaches the assistant's own folder, whatever a grant says, because it holds every user's memories and the grants. A part that keeps ordinary work under that folder adds it to `assistant.open_folders`. `files.search` looks for plain text only, because a regular expression from a model could hang the process. `files.list` and `files.search` do not follow links out of the folder.

**Writing and rollback.** `files.write` writes to a temporary file, syncs it, and renames it into place, so a crash leaves the old file or the new one. It first copies the old version to `undo/` in the assistant's folder, keeping the newest 500 writes. The result's `data.undo` names the copy.

**`web.fetch`.** GET only, http and https only, no user name or password in the URL, 2 MiB by default up to 20 MiB, 30 seconds by default up to 120. Every address a name resolves to must be public, unless a grant names the host outright: a wildcard does not count. Five redirects at most, and each is checked again against the grants and the address rule. It connects to the address it judged, so a name that changes its answer gains nothing, while TLS still checks the real name. Proxy settings in the environment are ignored. A call allowed by an approval alone cannot reach a private address, and its redirects are not followed.

**`shell.run`.** Takes `command` as a list of words and `cwd`, with optional `timeout` (120 seconds, up to 3600), `network`, and `env`. It runs only in the sandbox. A task's `shell.run` stops when the task is cancelled. Its scope is the working folder, which limits where a command starts and not what it can reach. So at the `process` level, a grant for `shell.run` in any folder is close to a grant for everything this user can do. Check `sandbox` before giving one.

**Git.** Hooks do not run, and `core.fsmonitor`, external diffs, and commit signing are pinned off on the command line. Anything that would be read as a flag is refused: `git diff --output=` would otherwise write a file from a read tool. If the repository's own config sets something no flag can turn off, the tool refuses and names the setting: filters, diff drivers, merge drivers, credential helpers, `core.sshCommand`, `core.gitProxy`, `core.askPass`, `core.worktree`, remote programs, and includes. The owner's global config is trusted. This matters because sandboxed code can write to the folder it works in, and would otherwise be able to plant a command for a later read to run outside the sandbox. `git.push` takes only the name of a configured remote and a plain branch, and never forces. It uses your own git credentials.

**`mesh.status`** reads `/health` and `/v1/models` from `SHARDMESH_ORCHESTRATOR_URL`, or `http://127.0.0.1:8000`.

**`memory.remember`** keeps what it is given as a model's claim, or EXTERNAL from a tainted context, and no freer than the reach of the call. It never counts as a person saying it.

**`system.info`** reports GPUs through `nvidia-smi` where it exists, and temperatures where the system offers them, which Windows does not. Other GPUs are not listed.

### `undo list`

`--user` shows only writes made for this user.

### `undo restore ID`

Puts a file back as it was, or removes it if that write created it. It refuses if the file changed again.

| Option | Meaning |
|---|---|
| `--force` | Restore even if the file changed after that write |
| `--by` | The person doing this, for the record |

## Sandbox

Generated code runs here, never in the assistant's process. The strongest walls the machine really has are used, and `level` is only ever what happened.

| Level | How | What it gives |
|---|---|---|
| `container` | docker or podman | No network, read-only root, memory and process limits, no capabilities. The folder is mounted at `/work` |
| `namespace` | bubblewrap | No network, separate processes, read-only system, home folders hidden |
| `namespace` | `unshare` on Linux | No network, separate processes. No file system wall, and the result says so |
| `process` | Windows Job Object, or a new session and rlimits on Linux | A memory limit, and a tree that dies together. Runs as this user |

Every level has a scrubbed environment built from a short allow-list, so a secret added tomorrow stays out too. Each has a time limit that kills everything the command started, and capped output. Whatever a command leaves running is killed when it ends.

Nothing is ever downloaded: a container level needs its image on the machine already. If you demand a level with `SHARDMESH_SANDBOX` and it is missing, nothing runs. Otherwise a weaker level is used and `sandbox` says why.

**What the `process` level does not do.** It cannot block the network, and each result says so in `network_blocked`, `notes`, and a line on stderr. It gives no file wall: the command can read and change whatever this user can. Because it runs as this user, a program that means harm can read other processes of this user, including this one's environment. On Linux a child that leaves its session escapes unless its parent is still alive, and if the assistant is killed its children keep running. On Windows the job kills them.

| Setting | Meaning | Default |
|---|---|---|
| `SHARDMESH_SANDBOX` | `auto`, `container`, `namespace`, or `process` | `auto` |
| `SHARDMESH_SANDBOX_IMAGE` | The local container image | `python:3.12-slim` |
| `SHARDMESH_SANDBOX_MEMORY_MB` | Memory limit per run | `2048` |
| `SHARDMESH_SANDBOX_OUTPUT_BYTES` | Output kept per stream | `1048576` |

### `sandbox`

Shows the level this machine really gets, and why nothing stronger was used.

## Tasks

A plan is data. A pure function, `next_actions`, takes a task's state and returns what happens next, so a model never decides the flow.

**Crash recovery.** An attempt is written as started and committed before anything happens. After a crash, an attempt with no result may or may not have happened:

- If its capability is idempotent, it runs again. Whether it counts is what was recorded when it started, not what a tool claims now.
- Otherwise the task blocks with "this may or may not have happened; a person must decide". It is never silently run again. You then choose: `task retry ID --step S` to run it again, `task resume ID --step S --as done` if it did happen, or `--as failed` to give it up.

Model and check steps can always run again. Only one process runs tasks at a time, through a lease naming that exact process, so looking at a task from another process never disturbs it. `serve` runs them, or `task work` does. Nothing else does: a task from `say` waits until one of those is up.

### Plan format

```json
{
  "title": "Check the build and report",
  "priority": 5,
  "reach": "pool",
  "inputs": {"repo": "/home/me/project"},
  "steps": [
    {"id": "tests", "kind": "tool", "capability": "shell.run",
     "arguments": {"command": ["python", "-m", "pytest", "-q"], "cwd": "{{inputs.repo}}"},
     "retries": 2, "backoff_seconds": 5, "on_failure": "continue", "resources": ["gpu:1"]},
    {"id": "passed", "kind": "check",
     "check": {"type": "exit_code", "step": "tests", "equals": 0}},
    {"id": "summary", "kind": "model", "role": "worker",
     "prompt": "Summarise this test run: {{steps.tests.output}}",
     "on_failure": "escalate", "escalate_to": ["planning"],
     "expect": {"type": "regex", "pattern": "passed|failed"}},
    {"id": "review", "kind": "model", "role": "verification",
     "prompt": "Is this summary fair? {{steps.summary.output}}"},
    {"id": "gate", "kind": "approval", "message": "Publish: {{steps.summary.output}}",
     "depends_on": ["review"]},
    {"id": "later", "kind": "wait", "seconds": 60, "depends_on": ["gate"]},
    {"id": "notes", "kind": "subtask", "inputs": {"text": "{{steps.summary.output}}"},
     "depends_on": ["later"],
     "plan": {"steps": [
       {"id": "save", "kind": "tool", "capability": "files.write",
        "arguments": {"path": "/home/me/notes.txt", "content": "{{inputs.text}}"}}
     ]}}
  ]
}
```

| Plan field | Meaning |
|---|---|
| `steps` | Required. A list of steps |
| `title` | A name for lists |
| `priority` | A whole number, higher first. Default 0 |
| `reach` | How far this plan's text may travel: `machine`, `pool`, `peer`, or `cloud`. Default `pool` |
| `inputs` | Values that steps read as `{{inputs.name}}` |
| `tainted` | `true` treats every step as tainted. It cannot be turned off |
| `run_at` | When to start: seconds since 1970, or text such as `2026-11-05T09:00:00Z` |
| `every_seconds` | Repeat this often, at least 1 |
| `on_missed` | `run_once` or `skip`. Default `run_once` |

| Step field | Meaning |
|---|---|
| `id` | Required. Letters, digits, dash, or underscore |
| `kind` | Required. `tool`, `model`, `check`, `approval`, `subtask`, or `wait` |
| `depends_on` | Steps that must finish first. Steps with no link between them run at the same time |
| `retries` | How many more tries, 0 to 20. Default 0 |
| `backoff_seconds` | Wait before the first retry, doubling each time, up to an hour. Default 2 |
| `on_failure` | `stop`, `continue`, or `escalate`. Default `stop` |
| `priority` | Order among this task's steps |
| `resources` | Slots it holds while running, such as `gpu:1` |
| `expect` | A mechanical check on this step's own result. Failing it fails the attempt, so retries and escalation apply |

| Kind | Fields |
|---|---|
| `tool` | `capability`, `arguments`. Goes through the broker |
| `model` | `prompt`, and optionally `role` (default `worker`), `system`, `max_tokens`, `temperature`, `escalate_to`, `review`. Goes through the router. Tool calls in the reply are ignored |
| `check` | `check`, one of the forms below |
| `approval` | `message`. Waits for `approvals approve` |
| `wait` | `seconds`, or `until` as a time |
| `subtask` | `plan`, and optionally `inputs`. The plan cannot have a schedule of its own |

| Check | Fields | Passes when |
|---|---|---|
| `exit_code` | `step`, `equals` (default 0) | That step's `data.exit_code` matches |
| `ok` | `step` | That step reported success |
| `regex` | `text`, `pattern`, `matches` (default true) | The pattern is found, or not. It runs in a child process for at most 5 seconds, so a bad pattern cannot hang the engine |
| `json_schema` | `text`, `schema` | The text is JSON matching the schema |
| `file_exists` | `path`, `exists` (default true) | The file is there, or not. This reveals only whether a path exists, so it needs no grant |

In `expect`, `step` and `text` default to the step's own result.

**References.** `{{steps.ID.output}}`, `{{steps.ID.ok}}`, `{{steps.ID.data.FIELD}}`, and `{{inputs.NAME}}`. A reference makes a dependency by itself. A string that is only a reference keeps the value's type. Substitution is one pass, so an output cannot smuggle in a reference. Other double braces are left alone.

**Taint.** A tool result is tainted unless it proved otherwise. A step that consumes a tainted output is tainted, and so is its output, through models and subtasks in both directions. So a consequential step fed by a web page waits for a person even under a standing grant. Text marked as outside is fenced in a model's prompt, and the model is told it is data. A step that merely runs after a tainted one, without reading it, is not tainted.

**Reach** goes down to the strictest input, including into subtasks.

**Failure.** `stop` skips what has not started, lets what is running finish, and ends the task as `failed`. `continue` skips only what depends on the failed step, and the task ends as `partial`. `escalate` moves a model step up its roles, then asks a person. A refusal by the broker or a person is not retried.

**Checks come first.** A model step with `role: verification` or `review: true` waits for every check in the plan, even if the plan forgot to say so, so no model reviews what software can already reject.

**Slots.** `gpu:1` holds one of a limited resource while the step runs. A name with no configured size is a lock of size 1, such as `repo`. A capability that declares a configured resource uses that slot by itself. Every tool, model, and check step takes a `worker` slot. A plan asking for more than exists is refused.

**Priority and aging.** Tasks go in priority order, and every 300 seconds of waiting is worth one point, so new urgent work cannot starve what is older.

**Schedules.** The next run of a repeating job exists only once the last one ends, so it can never overlap itself. Runs stay on the job's original grid and never catch up. A run later than 60 seconds is missed: `run_once` runs it once now and moves to the next slot, and `skip` moves on without running it. A one-off that is missed with `skip` ends as `missed`. Cancelling a run ends the series. A failed run does not.

**Statuses.** A task is `pending`, `running`, `paused`, `blocked`, `done`, `partial`, `failed`, `cancelled`, or `missed`. A step is `pending`, `running`, `waiting`, `blocked`, `done`, `failed`, `skipped`, or `cancelled`.

**Not interrupted by cancel:** a tool or model call already running, other than `shell.run`, finishes before the task ends. There is no general step time limit: tools and providers keep their own.

| Setting | Meaning | Default |
|---|---|---|
| `SHARDMESH_TASK_SLOTS` | Slot sizes, such as `gpu:2,cpu:8,worker:4` | `worker:8`, `gpu:1`, `cpu:` this machine's count |

### `task run PLAN`

Runs a plan file here until it ends or needs a person. Exits 0 for `done` or `partial`.

| Option | Meaning |
|---|---|
| `--user` | Who this runs for. Default `owner` |
| `--detach` | Queue it and return, for an engine already running |

### `task list`

| Option | Meaning |
|---|---|
| `--user` | Only this user |
| `--active` | Only tasks that have not ended |
| `--limit` | How many to show. Default 50 |

### `task show ID`

Shows the steps, attempts, and history. `--json` prints everything.

### `task pause ID` and `task cancel ID`

Pause starts no new steps. Cancel stops the task, including anything running in the sandbox, and withdraws what a person was asked. Cancel works with no engine running. `--by` is the person doing this.

### `task resume ID`

Carries on after a pause, or settles a blocked step.

| Option | Meaning |
|---|---|
| `--step` | The blocked step a person has decided |
| `--as` | `done` or `failed`: what really happened to it |
| `--by` | The person deciding |

### `task retry ID`

Runs failed or undecided steps again.

| Option | Meaning |
|---|---|
| `--step` | Only this step, and what was skipped because of it |
| `--by` | The person deciding |

### `task work`

Runs tasks and schedules here until stopped. `--once` stops when nothing can move.

## Planning

`assistant.planner.plan(user, goal)` asks the `planning` role for JSON only. Software then validates it: acyclic, known capabilities and roles, references that resolve, at most 50 steps. On rejection it re-asks at most twice with the errors, then raises `PlanRejected`. Only `title` and `steps` are read from a model, at every level, so a model cannot schedule a plan, widen its reach, or clear its taint.

A mechanical check is added after any consequential step that lacks one, and what followed that step waits for the check. The result lists `checks_added` and `missing_grants`, each with the command that would allow it. `assistant.planner.allow_missing(user, plan, by)` is one yes from a person: exact scopes, 24 hours, and only as many uses as the plan has tries, retries included. A step run again after a crash needs a use of its own, so a person may have to allow that one. A scope known only at run time, and anything physical, is left for a person.

`triage(goal)` is deterministic and uses no model. Something short, in one sentence, with no words suggesting stages, is small enough to run directly.

### `plan GOAL`

Asks for a plan and checks it. It runs nothing.

| Option | Meaning |
|---|---|
| `--user` | Who this is for, which decides what counts as missing. Default `owner` |
| `--out` | Write the plan here for `task run`. Default prints it |

## Plugins

A plugin is someone else's code. It runs as a separate process with a scrubbed environment and a time limit, behind the broker. Its capabilities are always prefixed, as `plugin.NAME.TOOL` or `mcp.NAME.TOOL`, so it cannot take a built-in's place. Its output is always EXTERNAL, whatever it claims. Installing one starts nothing until it is called, and no capability does anything until a grant covers it.

```json
{
  "name": "weather",
  "command": ["/usr/bin/python3", "weather.py"],
  "timeout_seconds": 30,
  "memory_mb": 512,
  "capabilities": [
    {"name": "forecast", "summary": "Tomorrow's forecast.", "effects": ["network"], "reads": ["location"],
     "resources": ["network"], "scope_field": "city", "idempotent": true,
     "input_schema": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}}}
  ]
}
```

The command runs in the manifest's folder. Each call starts it, writes one line of JSON to its stdin, and reads one JSON object from its stdout:

```text
{"capability": "forecast", "arguments": {"city": "Oslo"}, "user": "owner"}
{"ok": true, "output": "Rain.", "data": {"mm": 4}}
```

**MCP servers** use `"protocol": "mcp"`, and the tools come from the server over stdio: `initialize`, then `tools/list`, then `tools/call`. A server's own hints are untrusted and ignored, so what you state at install is what counts, and the default is execute plus external. You can state effects with `--effects` and `--tool-effects`, or in the manifest under `tools`, which is your note and not the server's:

```json
{
  "name": "search",
  "protocol": "mcp",
  "command": ["/usr/bin/python3", "server.py"],
  "tools": {"find": {"effects": ["read"], "scope_field": "folder", "idempotent": true}}
}
```

Each tool's name, description, and input schema are fingerprinted at install. If a server later changes one, the plugin is turned off until a person installs it again. A new tool is not offered until then. A description is shown as the server's own unverified words, cut to 200 characters. The server gets nothing back: no sampling, no roots.

**Failures.** A crash, a hang, or a reply that is not valid counts as a failure, and three in a row turn the plugin off. A proper reply saying `ok: false`, or an MCP error, is not a failure. The rest of the assistant carries on. Installing it again turns it back on and keeps its grants. If what it declares changed, its old grants end. Removing a plugin ends its grants.

**What this does not do.** Plugins run with `process` walls as this user, whatever level the sandbox has, because their programs are on this machine and not in an image. So installing a plugin is trusting its code. A running plugin call is not stopped by `task cancel`: it ends at its own time limit.

### `plugin add MANIFEST`

Installs a plugin or MCP server, or turns it back on.

| Option | Meaning |
|---|---|
| `--effects` | What every tool in it can do, comma separated, overriding the manifest |
| `--tool-effects` | `TOOL=EFFECTS` for one tool. Can repeat |
| `--env` | An environment variable to pass through, by name. Can repeat. The value is never stored |
| `--by` | The person installing it, for the record |

### `plugin list`

Shows each plugin, and why it is off if it is.

### `plugin remove NAME`

`--by` is the person removing it.

## Federation

Two independent installs, with different owners, cooperate without merging anything. No users, memories, or grants ever cross. Either owner can end it at once, and each install keeps working alone.

### Pairing

Both owners must agree, and nothing works until they have.

```
first$   shardmesh assistant peer invite --name bob --address http://100.64.0.2:9300
second$  shardmesh assistant peer accept CODE --name alice --address http://100.64.0.3:9300
         (the two owners read their fingerprints to each other, by phone or in person)
first$   shardmesh assistant peer approve bob
```

The code works once and expires. It holds the first install's address and fingerprint, and a secret. The second install proves it holds the code, and the first proves it made it, so nobody on the path can answer with their own keys. Comparing fingerprints is what catches someone who stole the code.

Each side's name for the other is its own choice. A peer never picks its own name here, and without `--name` it is `peer-` and the start of its fingerprint.

A peer moves through these states:

| State | Means |
|---|---|
| `awaiting_owner` | They accepted. This owner has not approved. Only `peer approve` moves it on |
| `awaiting_peer` | This owner accepted. The other owner has not approved |
| `paired` | Both agreed |
| `revoked` | Ended by either side. Offers and queued work are gone, so pairing again needs a new invite and starts from nothing |

A peer can never approve for the other owner. An install with no address cannot be called, so it finds out by asking: `peer approve NAME` on that side, or its next send.

### Offers

`peer offer NAME CAPABILITY` is one act by the owner: it lists one thing this install lets that peer call, and makes the broker grant that peer needs to use it. Withdrawing the offer, or revoking the peer, ends that grant, so the two cannot drift apart. A capability with a scope needs `--scope`, and `*` allows everything, on purpose.

On this install a peer is its own user, `peer:<fingerprint>`, as well as its own actor, and it is always tainted. So the owner's grants never apply to a peer, and anything scoped to a user sees no data of the owner's. Being always tainted means every call with consequences waits for the owner's approval: `approvals list` and the page show it, and a yes runs that one call when the peer collects its answer. A no is final: the peer's next `peer collect` gets a refusal, and no new question is put to the owner.

Offering `models.complete` lets a peer use this install's models without seeing any memory. The prompt goes straight to the router, with no tools and nothing added. Its text is held to reach `POOL`, so it stays on this owner's machines.

Some things can never be offered: anything named `peer.`, `federation.`, `memory.`, `identity.`, or `skills.`, and any capability that declares it reads memories, identities, users, or grants. These are refused when offered, and again on every request.

A peer can read its own offers and nothing else, through the built-in `federation.offers`. Revoking removes every offer.

### Sending

`assistant.federation.send(peer, capability, arguments, items, context=...)` is how a part asks a peer for something:

1. Any `ContextItem` with a reach below `Reach.PEER` refuses the whole send, with a `FederationError`. Nothing is dropped silently.
2. It calls the local broker as `peer.send`, declared with `Effect.EXTERNAL` and scoped by `peer`. The items are part of the arguments, so an approval covers exactly what will be disclosed. EXTERNAL text among the items marks the call tainted, so a person has to approve it.
3. Only the text of each item and its source kind travel. Paths, row ids, and authors stay here.

Text placed in `arguments` has no provenance, so the reach check cannot see it. Put anything with a reach limit in `items`. A call from a conversation or a task carries the reach of what it was written from, so a model asking for `peer.send` in a turn that may go no further than the pool is refused whatever the arguments say.

Two calls carry only a hash or nothing, and do not go through `peer.send`: `collect` and `discover`. A model cannot call either.

A result comes back as a `ToolResult` with trust `EXTERNAL` whatever the peer claims, reach `PEER`, and `data` holding:

| Key | Holds |
|---|---|
| `status` | `ok`, `pending`, `refused`, `failed`, or, from this side, `needs_approval`, `unknown`, `unreachable`, `invalid` |
| `responder` | The fingerprint that signed it |
| `request_hash` | The hash of the signed request |
| `produced_by` | `{"kind": "tool", "name": ...}` or `{"kind": "model", "name": ..., "provider": ...}` |
| `result` | The tool's own data |

`unknown` means no reply came in time. The work may have run, so never send it again: use `peer collect`. The peer keeps every answer for seven days for this reason.

### Protocol

Version 1. All messages are JSON over HTTP POST, and the HTTP layer is not trusted for anything.

Each install has an Ed25519 signing key and an X25519 key, made on first use and stored in `federation/identity.key` in its home. They are kept out of the database so that a copy of it carries no private key, which means a backup needs this file too: without it, this install is a stranger to its peers and has to pair again. Its fingerprint is the first 32 hex digits of `SHA-256("shardmesh-peer-identity-v1\0" || signing public || exchange public)`, so it covers both keys.

**A sealed message.** `body` is the JSON text of the message. The signature covers those exact bytes, prefixed by a domain string, so a signature for one kind of message means nothing as another:

```
signature = Ed25519(sender, "shardmesh-peer-<kind>-v1\0" || body)
inner     = {"body": body, "signature": base64url(signature)}
key       = HKDF-SHA256(X25519(ephemeral, recipient), info = "shardmesh-peer-seal-v1|<kind>|" || ephemeral public || recipient public)
sealed    = {"v": 1, "ephemeral": ..., "nonce": ..., "ciphertext": AES-256-GCM(key, nonce, inner, aad = <kind>)}
```

`kind` is `request`, `result`, `notice`, or `notice-reply`. The ephemeral key is used for one message.

**A request body:**

```json
{"v": 1, "kind": "request", "from": "<fingerprint>", "to": "<fingerprint>", "nonce": "<hex>",
 "issued_at": 0.0, "expires_at": 0.0, "capability": "models.complete", "arguments": {}}
```

**The receiver's checks, in order.** Each refusal is audited.

| Check | Refused when |
|---|---|
| Opening | It cannot be decrypted, or is not valid JSON |
| Signature | The sender is unknown, or the signature fails |
| Audience | `to` is not this install, which stops a peer forwarding what it was sent |
| Expiry | It has expired, is dated more than 60 seconds ahead, or claims to live over 600 seconds |
| Nonce | This peer already used it. Nonces are in the database until the message expires, so a restart does not help. Taking one is atomic, so a replay racing the original loses |
| Pairing | The peer is not `paired` |
| Offer | The capability is not offered to this peer, or can never be offered |
| Rate limit | The peer has used its calls in the last 24 hours |
| Broker | The broker denies it, or asks a person |

If the broker asks, the reply is `pending`, and the owner decides through the broker. The peer then fetches the answer with `peer collect`, which checks everything again, so withdrawing the offer stops it.

**A result body** adds `status`, `output` or `reason`, `data`, `produced_by`, `request_hash`, `issued_at`, and `revoked`. It is signed and sealed to the requester, who checks the signature, that `to` is them, and that `request_hash` is what they sent, so an old answer cannot be substituted. When a refusal cannot be tied to a known key it is plain JSON, `{"refused": "..."}`, and the sender reports it as unproven.

HTTP status mirrors the result (200, 202 for pending, 403 for refused), but only the signed body counts.

**Notices** have `kind` `notice` and a `notice` of `approved`, `revoked`, or `state`. They pass the same audience, expiry, and nonce checks. A `notice-reply` carries the sender's view: `paired`, `waiting`, or `revoked`.

**Pairing** is the one exchange before keys are known. The acceptor posts `{v, invite, signing, exchange, address, signature, proof}`, where `invite` identifies the code, `signature` is by the key it is offering, and `proof` is an HMAC keyed from the code's secret. The inviter answers `{signing, exchange, proof}`, and the acceptor checks that those keys hash to the fingerprint in the code. Taking an invite is atomic, so two people cannot use one code.

### Revoking

`peer revoke NAME` works here at once, then tells them if it can, waiting at most three seconds. If they cannot be told, every later request still fails, and the signed refusal tells them on their next contact. An unsigned claim of revocation changes nothing.

### Commands

| Command | Does |
|---|---|
| `peer invite [--name NAME] [--address URL] [--minutes N]` | Prints a one-time code. `--name` is what to call them here. `--address` is where they reach this install, and is remembered from `peer serve` otherwise. `--minutes` is how long the code works, 60 by default |
| `peer accept CODE [--name NAME] [--address URL]` | Agrees to pair. Without `--address`, they cannot call this install, though it can call them |
| `peer approve NAME [--fingerprint F]` | Agrees to pair, after comparing fingerprints. With `--fingerprint`, refuses if it does not match. On the accepting side, it asks whether they have approved yet |
| `peer list` | This install's fingerprint, then each peer's name, state, fingerprint, address, and offers |
| `peer offer NAME [CAPABILITY] [--per-day N] [--scope S] [--withdraw]` | Offers a capability and grants it to that peer's user, at most N calls in any 24 hours, 200 by default. `--scope` limits a capability that has a scope. `--withdraw` stops offering it and ends the grant. With no capability, shows the peer's offers |
| `peer revoke NAME` | Ends the pairing |
| `peer ask NAME CAPABILITY [--text T] [--args JSON] [--user U]` | Calls something a peer offers, through this install's broker. `--text` is sent as the `prompt` argument, `--args` is a JSON object, and `--user` says who is asking. Exits 0 on success, 3 when the answer may come later, 1 otherwise |
| `peer collect NAME REQUEST` | Fetches an answer that was pending, or lost on the way |
| `peer offers NAME` | Asks a peer what it offers this install, and tells the router about its models |
| `peer serve [--host H] [--port P] [--address URL]` | Answers peers without the gateway. The port defaults to 9300, and 0 picks a free one. `--address` is where peers reach this install, if not where it listens |

### Routes

`routes(assistant)` gives these, on `peer serve` and on the gateway alike. They need no key, because every message proves itself, and the part lists them in `PUBLIC_ROUTES` so a gateway leaves exactly these open. There is deliberately no route for listing peers or offers.

| Route | Does |
|---|---|
| `GET /federation/identity` | This install's fingerprint and protocol version |
| `POST /federation/pair` | An owner accepting an invite |
| `POST /federation/request` | A sealed request |
| `POST /federation/notice` | A sealed notice |

Bodies over 1 MiB are refused before anything is parsed. Strangers get at most 30 audit rows a minute, and the next row says how many were not recorded, so junk cannot fill the record.

`FederationServer(assistant, host, port)` runs the routes alone, with `start()` returning the URL and `stop()`.

### Using a peer's models

`PeerProvider` is a full provider, named `peer:NAME`, so `provider list` and `route set` treat it like any other. It refuses a request whose reach is below `PEER` even if the router sent it. Its replies never carry tool calls, the reply's `provider` is `peer:<fingerprint>`, and a peer answers whole, so a stream arrives as one piece. Treat its text as `EXTERNAL`.

A provider is registered for each paired peer known to offer `models.complete`. `peer offers NAME` finds that out, and it is remembered across restarts. Three things must line up before a turn reaches a peer's model: `peer offers NAME` has been run, this install holds a `peer.send` grant for that peer, and the person's reach allows it (`context --reach peer`), since a person's words stay on the pool otherwise.

### Settings

| Setting | Means |
|---|---|
| `SHARDMESH_ASSISTANT_USER` | Who `peer ask` acts as, if the identity part does not name an owner. Default `owner` |

## Skills

Learning improves what the assistant does and can never add to what it may do. `assistant.skills` keeps versioned procedures per user, because a skill grows out of that user's private tasks.

### A version

A skill version holds instructions, required capabilities, applicability rules, stop conditions, and verification tests. Its content is fixed and hashed, so a change behind the store is noticed and refused. Its text is trust `MODEL` until it passes its tests, even if the owner wrote it.

| State | Means |
|---|---|
| `candidate` | Waiting to prove itself. It can be evaluated, and can never run for real |
| `active` | In use. A skill has at most one, enforced by the database |
| `retired` | Replaced by a newer version. It is what a rollback restores |
| `rejected` | Turned down, or rolled back. This is final, so a bad version cannot come back |

**Applicability** has `task_types`, `all_of`, `any_of`, and `none_of`, matched mechanically against a task. It is never empty: with no rules given, a skill applies only to its own task type.

**Stop conditions** are `max_steps` (default 8, at most 50), `max_seconds` (default 300), and `stop_phrases`.

**A verification test** is `{"name": ..., "input": ..., "checks": [...]}`. Each check has a `kind` and a `value`:

| Kind | Passes when |
|---|---|
| `contains`, `not_contains` | The answer has, or lacks, the text, ignoring case |
| `equals`, `starts_with` | The answer matches, ignoring surrounding space |
| `min_length`, `max_length` | The answer is at least, or at most, that long |
| `json_field` | The answer is JSON and `path` (such as `a.b.0`) equals `value` |
| `calls`, `not_calls` | The model did, or did not, ask for that capability |

There are no regular expressions, because a model writes these tests and a bad pattern could hang this process.

### How a skill is learned

`reflect(task_id)` needs a finished task, and records what worked, what failed, and what needed a person. Reflecting twice counts once.

From a success it can draft a candidate. The `worker` role writes it, and the `verification` role reviews it, or `planning` if that is down. A review can turn a draft down, but cannot approve anything. The capabilities come from what the task actually used, whatever the draft says.

It reads `status`, `user`, `title`, `tainted`, `reach`, and `steps` from `tasks.status(task_id)`, each step with `id`, `status`, `capability`, `error`, `tainted`, and `approval_id`. A task that read a file or a page is tainted, as nearly all do, so a candidate from one can be promoted only by a person. No skill is drafted from a task whose text may not leave this machine, because a skill's text is shown to models on the pool. Nothing reflects by itself: run `skill reflect`.

### How a candidate proves itself

`evaluate(version)` runs the candidate and the active version on the same cases, at temperature 0, and checks the answers mechanically.

- The cases are the candidate's tests plus the active version's, so a candidate cannot drop a test it would fail.
- It is eligible only if all its own tests pass, and no case passes on the active version and fails on the candidate.
- A test that accepts an empty answer or a nonsense one proves nothing, and counts against it.
- Tools are declared to the model and never run, so evaluating has no effect on anything. `calls` checks see what it asked for.
- If no model answers, nothing is recorded, since that says nothing about the skill.

Passing all its tests makes the text trust `VERIFIED`. Failing later takes that away.

`promote` needs the owner. It holds only for the exact version and active version that were compared, so if the active version has changed, evaluate again. There is deliberately no capability for promoting a skill, so no model, task, or peer can reach it.

With `SHARDMESH_ASSISTANT_SKILL_AUTO_PROMOTE` on, an eligible candidate promotes itself only if every capability it needs is declared and read-only, and it did not come from a tainted task. A capability nobody declared counts as consequential. A skill cannot use this to hide what it does, as it can only call what it declared.

### In use

`execute(name, text, user)` runs the active version under the actor `skill:<id>` for that user.

- It can call only what it declared, even if the user has granted more.
- Each call goes to the broker, so a capability the user never granted fails there.
- What a tool returned taints the rest of the run, so a consequential step after it needs a person.

Each run is counted, or reported with `record_run(version, ok)`. Once a new version has 10 runs, and its predecessor at least 3, it is rolled back by itself if its success rate is more than 0.1 lower. The reason is kept with the version and in the audit record. A run waiting on a person, or a model outage, does not count. With no baseline for its predecessor there is no verdict, so check `skill show`.

`relevant(user, text, task_type, limit)` returns active skills as `ContextItem`s, source `skill`, with their trust, at reach `POOL`. Every turn asks for these, so an active skill's instructions are in front of the model as a note, labeled with its trust. Nothing runs a skill by itself: `execute` is for a part or a test to call, and no command does. `gaps(user)` reports repeated failures by task type over 30 days: a type shows after two failures, with recent reasons, how often it needed a person, and whether a skill exists.

### Commands

`skill` takes `--user U` before the subcommand, for another user's skills. A version is `name@number` or its id.

| Command | Does |
|---|---|
| `skill list` | Every version, with state and trust |
| `skill show VERSION` | Its text, rules, tests, review, reason, success rate, and last evaluation |
| `skill evaluate VERSION` | Evaluates a candidate. Exits 1 if it cannot be promoted |
| `skill promote VERSION` | Puts an eligible candidate into use |
| `skill reject VERSION [--reason R]` | Turns a version down for good. On an active version with a predecessor, this is a rollback |
| `skill rollback NAME [--reason R]` | Restores the previous version |
| `skill gaps` | Repeated failures by task type |
| `skill reflect TASK [--no-draft]` | Records a finished task. `--no-draft` skips drafting a skill |

### Settings

| Setting | Means |
|---|---|
| `SHARDMESH_ASSISTANT_SKILL_AUTO_PROMOTE` | `1`, `true`, `yes`, or `on` allows automatic promotion as above. Off by default |

## Coding

`assistant.coder` makes supervised code changes. A run works on a new branch, `assistant/<slug>`, in a git worktree under the assistant's home at `coder/<slug>/tree`. It never touches the owner's checkout or any existing branch, and git hooks, line-ending rewriting, and `GIT_*` settings are switched off for its own git commands.

### A run

```
shardmesh assistant code start --repo PATH "Fix the rounding in invoices"
shardmesh assistant code show ID
shardmesh assistant code approve ID
```

| State | Means |
|---|---|
| `created`, `inspecting` | Making the worktree, finding languages and the test command, and running the tests once as a baseline |
| `proposing` | Asking the `coding` role for changes |
| `applying` | The reply is saved. Applying it in the worktree |
| `testing` | Running the tests in the sandbox |
| `reviewing` | The tests pass. The `verification` role reads the diff |
| `ready` | Waiting for the owner |
| `committed`, `pushed` | Done, through the broker |
| `stopped` | It reached its limit, or found no test command. The reason is shown |
| `rejected` | Thrown away |

The state is in the database, and each step saves what it needs before moving on. After a crash or outage, `code resume ID` carries on: a coding reply already saved is not asked for again, and the tree is rebuilt from the base and the last applied diff, so a half-finished step cannot leave anything behind. If a model or the sandbox is down, the run waits where it is.

Failures go back to the `coding` role, up to `--rounds` times, then it stops and reports. A reviewer asking for changes also uses a round. A review is shown to the owner and decides nothing: with no rounds left, the run is `ready` with the reviewer's objection on it.

### What a model can and cannot do

A model replies in text, with no tools. Changes come as a whole file or a unified diff:

```
<<<FILE path/from/the/root
the whole new content
>>>FILE

<<<DIFF
a unified diff
>>>DIFF
```

A fenced `diff` block, or a reply that is only a diff, also works. A change is refused, and the model told why, if it would touch `.git` in any spelling, leave the tree, go through a symbolic link, use a reserved Windows name or a hidden stream, or create a symbolic link, submodule, or binary file. These are checked on the paths git itself reads from the diff, and again on what git says actually changed.

Repository text and test output are labelled as data. By default they stay at reach `POOL`, and `--reach` lets the owner say otherwise.

### Where code runs

Generated code never runs in the assistant's process. The coder only reads and writes files as text and runs git. With no sandbox it refuses to start, and never falls back to running anything itself.

Tests run through `assistant.sandbox`, with the network off, on a further copy at `coder/<slug>/copy` that has no `.git`. So nothing that runs there can reach the repository or change what gets committed, and the copy is removed afterwards. A test command that needs git cannot work there.

A pass means the command exited 0. A test run that finds no tests, or times out, is not a pass.

### Decisions

`code show` gives the diff, each test run with how strong its sandbox was, the review, and which models contributed. It also warns when a change touched what decides whether tests pass, or what runs by itself: tests changed or removed, test or build configuration, and automation such as `.github`.

`code approve` commits through the broker's `git.commit`. It refuses unless the tree still matches, by hash, what was tested. It also refuses if the change writes into a relative `core.hooksPath`, as git would run that outside the sandbox at commit. The call is not tainted, because a person read this exact diff and asked for this exact commit, but it still needs a grant. If the broker asks for approval, the run says so: approve it through the broker, then run `code approve` again. Afterwards it compares what landed with what was tested, and says so loudly if they differ.

`git.commit` runs in the worktree, outside the sandbox, so it runs with hooks off: `core.hooksPath` is pinned to an empty folder on the command line, so a `pre-commit` hook cannot run generated code on this machine. A test commits in a repository holding such a hook and checks that it never ran.

`code push` uses `git.push`, which is denied unless granted, and only ever pushes the run's own branch. `code reject` removes the worktree and the branch. If the run was already committed, the branch stays, as that was the owner's decision.

There is deliberately no capability for approving or pushing, so no model, task, or peer can reach them. Calls name their arguments from what `git.commit` and `git.push` declare in `input_schema`, and send only what the tool takes. The worktree is the repository named, so a scoped grant must cover the assistant's `coder` folder: `grant add git.commit HOME/coder`, where `HOME` is the assistant's folder. That folder is listed in `assistant.open_folders`, so file and git tools may reach it, and nothing else of the assistant's own.

### Working on ShardMesh itself

The same path applies, with nothing extra allowed: a candidate branch, tests, and approval. When the repository is ShardMesh, `coder/<slug>/HEALTH_AND_ROLLBACK.md` is written beside the worktree and shown by `code show`. It holds the base commit, what the change affects, a health check, and how to roll back. It is not part of the diff.

The coder never deploys or restarts anything. That stays a person's decision.

### Commands

`code` takes `--user U` before the subcommand, for another user's runs.

| Command | Does |
|---|---|
| `code start --repo PATH GOAL [--test CMD] [--rounds N] [--timeout S] [--reach R]` | Starts a run and works until it is ready or stops. `--test` overrides the detected test command. `--rounds` is 3 by default. `--timeout` is seconds for the tests, 600 by default. `--reach` is `machine`, `pool`, `peer`, or `cloud`, and is `pool` by default |
| `code list` | Every run |
| `code show ID [--output]` | The diff, tests, models, review, and warnings. `--output` adds the last test output |
| `code resume ID [--rounds N]` | Carries on. `--rounds` gives a stopped run more tries |
| `code approve ID [--message M]` | Commits this exact diff. `--message` replaces the default, which names the goal, tests, and models |
| `code push ID [--remote R]` | Pushes the run's branch. `--remote` is `origin` by default |
| `code reject ID [--reason R]` | Discards the work |

Test commands are detected from `pyproject.toml`, `pytest.ini`, `tox.ini`, `setup.py`, `setup.cfg`, or `test_*.py` files (`python -m pytest -q`), a real `test` script in `package.json` (`npm test --silent`), `Cargo.toml` (`cargo test --quiet`), `go.mod` (`go test ./...`), and a `test` target in a `Makefile` (`make test`).

## Voice

`assistant.voice` is speech in and out. It is optional and local: nothing leaves this machine, and nothing needs the internet. `voice status` says what works here, and when an engine is missing, the error names the package to install.

### Transcribing

`transcribe(audio_path)` returns the words in an audio file.

1. It uses `faster-whisper` if installed, on the CPU, as the GPU belongs to the mesh. The model comes from local files only and is never downloaded by itself.
2. Otherwise it uses a whisper.cpp binary if `SHARDMESH_ASSISTANT_WHISPER_CPP` is set. That needs 16 kHz WAV input.

Transcribed text is trust `EXTERNAL`: what a recording says is data, whoever seems to be speaking in it.

### Speaking

`speak(text, voice=None, out_path=None)` writes a WAV file and returns `{"path", "engine", "voice"}`. It does not play anything: the interface decides whether and where to.

1. It uses piper if installed, or `SHARDMESH_ASSISTANT_PIPER` if set, and only with a voice model in `SHARDMESH_ASSISTANT_PIPER_MODEL`.
2. Otherwise it uses the system voice: Windows SAPI through PowerShell, `say` on macOS, or `espeak-ng`.

The text always travels through a file or a pipe, never inside a command, so nothing it says can run. Speech left in `voice/speech` is removed after an hour. The caller keeps what it needs.

### Personas

If `assistant.identity` has a `voice`, as a string, a mapping with `name`, or a callable, that picks how speech sounds. It changes presentation only: no capability, grant, or decision depends on it. It is a name and never a path. Letters, digits, spaces, dots, dashes, and underscores are accepted, anything else is ignored, and for piper it picks a `.onnx` file beside the configured model.

### The broker

If a broker is present, these are declared to it, and the routes and commands go through it:

| Capability | Effect | Notes |
|---|---|---|
| `voice.transcribe` | `READ` | Scoped by `audio_path` |
| `voice.speak` | `WRITE_LOCAL` | So text from a tainted context needs a person before it is spoken |

With no broker, they are local tools on the owner's own machine.

### Routes

`routes(assistant)` gives these. They take the user from `request.state.user`, set by a gateway, or treat a caller on this machine as the owner, and refuse anyone else with 401. A path from outside is never opened.

| Route | Takes | Gives |
|---|---|---|
| `POST /assistant/voice/transcribe` | Audio as the body, at most 64 MiB, with an `audio/*` or `application/octet-stream` `Content-Type` | `{"text": ..., "trust": "external"}` |
| `POST /assistant/voice/speak` | `{"text": ..., "voice": optional}` as `application/json` | `audio/wav`, with the engine in `X-Voice-Engine` |

Those content types are required because a web page cannot send them without asking first, so a page the owner visits cannot call these behind their back. A missing engine is 503, a broker refusal is 403 with `approval_id` if a person can allow it, and a wrong type is 415.

### Commands

`voice` takes `--user U` before the subcommand, for who is asking.

| Command | Does |
|---|---|
| `voice status` | Which engines work here, and what to install |
| `voice transcribe AUDIO` | Prints the words in an audio file |
| `voice speak TEXT [--voice NAME] [--out PATH]` | Writes speech to a WAV file. `--voice` is a voice name, and the persona's by default. `--out` says where |

### Settings

| Setting | Means |
|---|---|
| `SHARDMESH_ASSISTANT_WHISPER_MODEL` | A faster-whisper model name found locally, or a folder holding a converted model. Default `base` |
| `SHARDMESH_ASSISTANT_WHISPER_CPP` | The whisper.cpp binary |
| `SHARDMESH_ASSISTANT_WHISPER_CPP_MODEL` | Its model file |
| `SHARDMESH_ASSISTANT_PIPER` | The piper binary, if not on the path |
| `SHARDMESH_ASSISTANT_PIPER_MODEL` | Its voice model, a `.onnx` file |

## The gateway

| Command | Does |
|---|---|
| `serve` | Runs the page and API. `--host`, `--port` (default 8470), `--tls-cert`, `--tls-key`, `--trusted-network` |
| `user add NAME` | Makes a user and writes their key to `keys/NAME.key`, or `--key-file`. `--role owner\|member`, and `--reclaim` gives a removed name and its data to this person. The first user is the owner |
| `user list` | Shows who may connect |
| `user remove NAME` | Ends access at once. Their data stays, and the name cannot be reused without `--reclaim` |
| `backup FILE --passphrase-file PATH` | Writes an encrypted copy of everything |
| `restore FILE --passphrase-file PATH` | Replaces this install's state. `--force` if it already holds data |

`serve` binds to loopback, and refuses any other address without TLS or `--trusted-network`, since keys cross the network with every request. It also refuses to start when nobody could connect. While it runs it also runs tasks, so `task work` is not needed beside it, and only one process runs tasks at a time. It remembers its own address for `peer invite`, unless the owner gave another.

Commands on this machine act for `owner` unless given `--user`, so the first user is best named `owner`. `user add` says so if not.

A new key goes to a private file and never to the screen, and only a hash is stored. Keys go in `Authorization: Bearer`.

### Routes

Every path needs a key except `GET /`, `GET /assistant`, `GET /assistant/`, `GET /healthz`, and the federation routes, which prove themselves. That is enforced before any route runs, so it covers routes other parts add. Such a route reads the caller from `request.state.user` and `request.state.role`, and a part leaves a route open only by listing it in its module's `PUBLIC_ROUTES`. A path the gateway already serves keeps its first owner, and the clash goes in the audit trail.

| Route | Does |
|---|---|
| `GET /v1/models` | The assistant, under its own name |
| `POST /v1/chat/completions` | Any OpenAI client. `stream` works, and `conversation_id` or an `X-Conversation-Id` header names a conversation |
| `GET /assistant/whoami` | Who this key is |
| `GET /assistant/identity` | The persona, as it applies to this person |
| `POST /assistant/say` | One turn in the assistant's own shape |
| `GET /assistant/conversations` | This person's conversations |
| `GET /assistant/conversations/{conversation_id}` | One conversation's messages |
| `GET /assistant/memory` | Memories with provenance. `q` searches, `limit`, and `history=true` includes superseded rows |
| `POST /assistant/memory` | Keeps a fact. It is always the caller's own words, and a request cannot claim more trust |
| `GET /assistant/memory/export` | Everything held for this person |
| `GET /assistant/memory/{memory_id}` | A memory and what it replaced |
| `DELETE /assistant/memory/{memory_id}` | Forgets it |
| `POST /assistant/memory/{memory_id}/confirm` | Vouches for it |
| `GET /assistant/library` | This person's folders |
| `POST /assistant/library` | Authorizes a folder. Owner only, since it reads this machine's disk. `for_user` adds it for someone else |
| `DELETE /assistant/library/{root}` | Removes a folder |
| `POST /assistant/library/index` | Brings the index up to date |
| `GET /assistant/library/search` | Searches with citations. `q`, `limit` |
| `GET /assistant/approvals` | What is waiting on this person, from turns, tasks, and the coder. The owner also sees what peers are asking, since a peer has no key |
| `POST /assistant/approvals/{approval_id}` | Decides one with `approve`, and carries a waiting turn on. A task carries on by itself. Something no longer waiting is a 409 |
| `GET /assistant/tasks` | This person's tasks, newest first, and whether the engine is installed. `limit` |
| `POST /assistant/tasks` | Submits a plan |
| `GET /assistant/tasks/{task_id}` | One task |
| `POST /assistant/tasks/{task_id}/cancel` | Cancels one |
| `GET /assistant/audit` | The owner sees everything and a member sees their own actions. It holds ids and decisions, never what anyone said. `since`, `limit`, and `newest=true` for the last rows rather than the first |
| `GET /assistant/why/{turn}` | What a model saw on a turn |
| `GET /assistant/turns` | Recent turns. `limit` |
| `GET /assistant/models` | Measured results and routes |
| `GET /healthz` | Whether the server is up |
| `GET /federation/identity`, `POST /federation/pair`, `POST /federation/request`, `POST /federation/notice` | The peer protocol, under [Federation](#federation) |
| `POST /assistant/voice/transcribe`, `POST /assistant/voice/speak` | Speech, under [Voice](#voice) |
| `GET /`, `GET /assistant`, `GET /assistant/` | The page |

Only the newest user message in a chat request is new. A client's earlier messages and system prompt are not believed, since the assistant keeps its own record and its own identity. The OpenAI `user` field changes nothing. The real model is in the `assistant` object on the reply, or in the last event of a stream.

A turn runs on its own thread and belongs to the assistant, so a client that disconnects does not lose the reply. A failure partway keeps what arrived and says so.

The page is `assistant.html`, built on the same `ui.css` as the mesh's chat page and dashboard, which is inlined when it is served. It has no external resources, and a content security policy with a fresh nonce on each load. Text only ever reaches it as text. A key is kept in the tab's session storage and goes when the tab closes. It shows the conversation, what is waiting for you, memory with where each piece came from, tasks, earlier conversations, and the audit record, and for any reply what the model saw. Signing out, or a key that stops working, clears the page, so the next person to sign in sees nothing of the last. `tests/test_assistant_page.py` checks these in real Chrome, and skips itself without it.

### Backups

A backup holds the one database, which is nearly everything. It does not hold `federation/identity.key`, so keep that too or pair again, nor `undo/`, `keys/`, or the coder's worktrees. `backup` takes a consistent copy with sqlite's backup API, then encrypts it with scrypt (N 2^15, r 8, p 1) and AES-256-GCM in 1 MiB chunks. Each chunk's position and whether it is last are in its nonce, and the header is authenticated, so a file that was changed, reordered, or cut short is refused. The clear copy sits beside the database, as protected as the database is, and is gone when the command returns. `restore` checks the copy's integrity and the audit chain, and anything using the folder has to be restarted afterwards.

## Settings

These are constants, with the file that holds each. `conversation.budget_tokens`, `reply_tokens`, and `max_rounds` can be changed on the object.

| Setting | Value | In |
|---|---|---|
| `DEFAULT_PORT` | 8470 | `gateway_cli.py` |
| `MAX_REQUEST_BYTES` | 8 MiB | `gateway.py` |
| `BUDGET_TOKENS` | 6144 | `conversation.py` |
| `REPLY_TOKENS` | 1024 | `conversation.py` |
| `MAX_ROUNDS` | 6 | `conversation.py` |
| `MAX_LEARNED` | 3 | `conversation.py` |
| `HISTORY_MESSAGES` | 40 | `context.py` |
| `READ_SECONDS` | 120 | `providers.py` |
| `WHOLE_REPLY_SECONDS` | 300 | `providers.py` |
| `MEASURED_RUNS` | 3 | `router.py` |
| `BACKOFF_SECONDS` | 15 | `router.py` |
| `MAX_BACKOFF_SECONDS` | 300 | `router.py` |
| `MAX_FILE_BYTES` | 32 MiB | `knowledge.py` |
| `RECENCY_HALF_LIFE_DAYS` | 30 | `memory.py` |

## For people writing a part

A part is a module in `shardmesh/assistant/` with `attach(assistant)`, and optionally `detach(assistant)`, `add_commands(subparsers)`, `routes(assistant)`, and `PUBLIC_ROUTES`. `app.py` loads whichever parts exist, in the order of `PARTS`, so a later part can use an earlier one. Parts meet through `contracts.py`, each owns its tables through `store.migrate`, and each writes to the audit trail through `assistant.audit`.

| A part reaches | Which holds |
|---|---|
| `assistant.broker` | `contracts.Broker`: `register`, `capabilities`, `check`, `invoke`, `covers`, `add_grant`, `revoke_grant`, `grants`, `approval`, `approvals`, `decide`. It also offers `capability(name)`, `names_host(user, capability, url)`, `request_approval`, `withdraw`, `revoke_capability`, and `unregister`. A refused `invoke` returns `data` with `refused`, `verdict`, and `approval_id` |
| `assistant.tasks` | `contracts.TaskEngine`: `submit`, `status`, `cancel`, `start`, `stop`, `list_tasks`. It also offers `pause`, `resume`, `retry`, `events`, `attempts`, `tick`, `runner`, `run_until_idle`, and `run_until_settled`. `status(task_id, user)` is scoped to a user in the query |
| `assistant.providers` | `register(provider)`, for a part to add a `contracts.Provider` while running, such as a peer's. `add`, `remove`, `get`, and `all` |
| `assistant.router` | `complete(request)`, `stream(request)`, and `record_outcome(reply, role, success)` |
| `assistant.planner` | `plan(user, goal, reach=..., tainted=...)`, `missing_grants`, `allow_missing`, `triage` |
| `assistant.skills` | `relevant(user, text, task_type, limit)`, `propose`, `evaluate`, `promote`, `reject`, `rollback`, `execute`, `reflect`, `record_run`, `gaps` |
| `assistant.memory` | `search`, `remember`, `forget`, `record_message`, `recent`, `conversations` |
| `assistant.sandbox` | `run`, which also takes `label`, `stop(label)` to end those runs, and `describe()`, which reports the level and why it is not stronger. A result is a `SandboxRun`: a `SandboxResult` with `network_blocked`, `stopped`, and `notes` |
| `assistant.builtins` | `undo_list` and `undo_restore` |
| `assistant.plugins` | `add`, `remove`, and `list` |
| `assistant.open_folders` | Folders under the assistant's home that file and git tools may reach. The coder adds its own. Nothing else of home is reachable, whatever a grant says |

A tool that stores or sends text reads `CallContext.reach`, which says how far the text behind the call may go.

## What is not built or proven

**Not built.** These are ideas, with no code behind them:

- **Avatars.** There is a name, a style, and a voice name. There is no face, no animation, and nothing that draws one.
- **A wake word.** Nothing listens. `voice` works on an audio file or a request someone chose to send.
- **Vision on the mesh.** `vision` is a role the router knows, and that is all. No part sends an image, and the mesh's chat API takes plain strings. The `vision` route would need a provider that can see, and a part that sends it something.
- **Home or robot control.** `physical` is an effect the broker enforces: such a capability can have no grant, and every action needs a person. No such capability exists. No plugin for one has been written, and no hardware has ever been moved by this.
- **Running a skill by itself.** A skill's instructions reach the model on every turn it applies to, but `execute` is called by no command and no part.
- **Reflecting by itself.** `skill reflect` is a person's command. Nothing watches for finished tasks.
- **Most of the assistant over HTTP.** Grants, providers, routes, skills, plugins, the coder, and peers have commands and no routes or page. A member of an install cannot manage any of these, and the owner uses a terminal.

### How a small model does

This is Qwen3-0.6B at Q8, on 2 CPU threads, on a mesh of one, with everything real from the assistant down. No larger model has been through the assistant, so how much better one does is not known.

| Tried | What happened |
|---|---|
| Plain conversation | Works, at about 20 tokens a second. Over five turns, 0.6 to 6 s a reply, median 2.3 s. It got simple facts and 7 times 8 right, and kept its name and manner. It mixes up "I" and "you" |
| Proposing memories | From one sentence it proposed one to three. "Sam keeps bees on his roof" was right. "Sam's bees are a good way to keep his roof healthy and clean" was invented. That is why these are kept as MODEL, and not as what you said |
| Recalling in a new session | 4 of 5, saying "What I keep on my roof are bees". The fifth invented plants and flowers |
| Reading a file, offered one tool | 8 of 10 read the right file through the broker and answered from it, at a median of 6.1 s |
| Reading a file, offered two tools | 5 of 10, from the same request with `files.write` granted as well |
| What it said when it did not read | Every miss, in both tests, said it had read the file and made up what was in it. A miss shows no `used files.read`, and that line is the only sign |
| Reading, then writing, in one request | 0 of 5. It says what it will do, and stops. No tool ran, so nothing was held for approval either |
| Writing a plan | 2 of 3 passed software's checks, at 171 and 199 s. The third was refused after 376 s, as it never wrote complete JSON. An earlier run had all three refused. Passing means well formed, and not good: both plans used `shell.run`, which was not granted, and one had the command `shell.run` and a path. They would stop at `missing_grants`, and nothing ran |
| Running a plan a person wrote | Finished in 1.2 s, including a model step and a mechanical check that passed |

So a model this small can talk, remember, and run plans a person writes. It cannot be trusted to use a tool, or to say when it did not. Use one for a demo, and check the `used` line.

How tools are described matters a great deal. In an earlier run, shown each tool's JSON schema, it called nothing in 10 tries. Shown a placeholder example, it copied the placeholder word for word. Shown short signatures and an example built from a real tool, it made the right call 10 times in 10, on most wordings, and 1 in 10 on one wording.

On a model with one slot, background work competes with you. A question asked straight after a turn showed its first text at 3.2 s, because memories were being proposed, and the same kind of question after a 15 s pause at 0.6 s.

**What the `process` sandbox cannot stop.** On Windows this is the only level, as there is no docker, podman, or bubblewrap here to use, and on any machine it is what is left when nothing stronger is present. It is a memory limit and a process tree that dies together, and it is not a wall:

- It cannot block the network. `network: false` means nothing here, and each result says so in `network_blocked` and `notes`.
- It cannot stop a file being read or changed. A command can reach whatever this user can, including the assistant's own folder, its keys, and its database, which file tools never touch.
- It cannot stop a command reading other processes of this user, including this one's environment.
- So a grant for `shell.run` at this level is close to a grant for everything this user can do, and the coder's tests are run the same way: running a generated test is running generated code as you. `code show` says which level a test ran at. Check `sandbox` before relying on it.

**Proven only at small scale.**

- Tool calling through the mesh was run with one small model, Qwen3-0.6B, on CPU. [What it managed is written up above.](#how-a-small-model-does) No larger model has been tried through the assistant.
- The mesh's reach is taken on trust: `pool` means text goes to your own nodes, and nothing here checks where a node really is.
- Federation was run between two installs on one machine over loopback, and in tests. It has not crossed a real network between two real owners.
- The `container` and `namespace` sandbox levels have only had their command lines checked in tests. Neither has ever run a command for real, as this was built on Windows with no container service running.
- Two processes on one install are safe to use, but only the task lease has been proven under a kill. Everything else relies on SQLite.
- Speech has been run for real only with Windows' own voice, through `voice speak`. Transcribing has not been run on real speech: `faster-whisper` is installed here with no model, and nothing downloads one. `voice status` names the engine either way, so the first `voice transcribe` is what tells you a model is missing.

**Known limits.**

- Writers in one process take turns in the order they came, so a busy one cannot starve the rest. With six conversations, ten tasks, and an index of 3,000 files all at once, the longest wait was 165 ms and nothing failed. Before that queue it was 11 s, with two writes failing as locked. Between processes SQLite still gives nobody a turn: a second process writing through all of that waited 3.4 s at worst, inside its 10 s limit. A command says so plainly if it ever runs out.
- bm25 is computed across all users, so the order of one person's results can shift slightly with what others hold. Scores are not sent over HTTP for that reason, and an order alone tells very little, but it is not zero.
- A reply streamed from a model that puts its reasoning before a bare `</think>`, with no opening tag, shows that reasoning while it streams. The stored reply is clean.
- Token counts are estimated at a character count over 3 when a server reports none, so a budget can be off for those servers.
- A running plugin call or a running tool other than `shell.run` is not stopped by `task cancel`.
- A proposed memory carries what the model wrote, and a model can be wrong. MODEL trust says so, and `confirm` is how a person vouches for it.
