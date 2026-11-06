# The hands

These parts are how the assistant does things: `broker`, `tools`, `plugins`, `sandbox`, `tasks`, and `planner`. A model can ask for anything. These parts decide what happens, write it down, and survive a crash.

Every command below goes after `shardmesh assistant`, and takes `--home` before the command to pick the assistant's folder.

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

**An approval** covers one exact action, by the SHA-256 of its canonical arguments. It is bound to the user, capability, actor, and task, it works once, and changed arguments need a new one. Saying no withdraws a yes that was not used yet. Only a person can grant or approve: names starting `skill:`, `peer:`, `task:`, `model:`, `plugin:`, or `conversation` are refused.

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

Effects are stated for the worst a call can do. `web.fetch` is network and not read, because a GET can carry data out in its URL, so it asks in a tainted context. `shell.run` admits to writing and the network, because the weakest sandbox stops neither.

**Files.** Paths must be absolute. No tool reaches the assistant's own folder, whatever a grant says, because it holds every user's memories and the grants. A part that keeps ordinary work under that folder adds it to `assistant.open_folders`. `files.search` looks for plain text only, because a regular expression from a model could hang the process. `files.list` and `files.search` do not follow links out of the folder.

**Writing and rollback.** `files.write` writes to a temporary file, syncs it, and renames it into place, so a crash leaves the old file or the new one. It first copies the old version to `undo/` in the assistant's folder, keeping the newest 500 writes. The result's `data.undo` names the copy.

**`web.fetch`.** GET only, http and https only, no user name or password in the URL, 2 MiB by default up to 20 MiB, 30 seconds by default up to 120. Every address a name resolves to must be public, unless a grant names the host outright: a wildcard does not count. Five redirects at most, and each is checked again against the grants and the address rule. It connects to the address it judged, so a name that changes its answer gains nothing, while TLS still checks the real name. Proxy settings in the environment are ignored. A call allowed by an approval alone cannot reach a private address, and its redirects are not followed.

**`shell.run`.** Takes `command` as a list of words and `cwd`, with optional `timeout` (120 seconds, up to 3600), `network`, and `env`. It runs only in the sandbox. A task's `shell.run` stops when the task is cancelled.

**Git.** Hooks do not run, and `core.fsmonitor`, external diffs, and commit signing are pinned off on the command line. Anything that would be read as a flag is refused: `git diff --output=` would otherwise write a file from a read tool. If the repository's own config sets something no flag can turn off, the tool refuses and names the setting: filters, diff drivers, merge drivers, credential helpers, `core.sshCommand`, `core.gitProxy`, `core.askPass`, `core.worktree`, remote programs, and includes. The owner's global config is trusted. This matters because sandboxed code can write to the folder it works in, and would otherwise be able to plant a command for a later read to run outside the sandbox. `git.push` takes only the name of a configured remote and a plain branch, and never forces. It uses your own git credentials.

**`mesh.status`** reads `/health` and `/v1/models` from `SHARDMESH_ORCHESTRATOR_URL`, or `http://127.0.0.1:8000`.

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

Model and check steps can always run again. Only one process runs tasks at a time, through a lease naming that exact process, so looking at a task from another process never disturbs it. Whoever hosts the assistant calls `assistant.tasks.start()`, or runs `task work`. Nothing runs tasks on its own.

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

A mechanical check is added after any consequential step that lacks one, and what followed that step waits for the check. The result lists `checks_added` and `missing_grants`, each with the command that would allow it. `assistant.planner.allow_missing(user, plan, by)` is one yes from a person: exact scopes, 24 hours, and only as many uses as the plan has steps. A scope known only at run time, and anything physical, is left for a person.

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

## For other parts

`assistant.broker` also offers `capability(name)`, `covers(user, capability, arguments)`, `names_host(user, capability, url)`, `add_grant`, `revoke_grant`, `revoke_capability`, `grants`, `request_approval`, `approval`, `approvals`, `decide`, `withdraw`, and `unregister`. A refused `invoke` returns `data` with `refused`, `verdict`, and `approval_id`.

`assistant.sandbox.run` also takes `label`, and `stop(label)` ends those runs. Its result is a `SandboxRun`, a `SandboxResult` with `network_blocked`, `stopped`, and `notes`. `describe()` reports the level and why it is not stronger.

`assistant.tasks` also offers `pause`, `resume`, `retry`, `list_tasks`, `events`, `attempts`, `start`, `stop`, `tick`, `run_until_idle`, and `run_until_settled`. `status(task_id, user)` is scoped to a user when one is given.

`assistant.builtins` offers `undo_list` and `undo_restore`. `assistant.plugins` offers `add`, `remove`, and `list`.
