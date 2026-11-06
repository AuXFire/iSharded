# Growth: federation, skills, coding, voice

These four parts let the assistant reach beyond one install and improve over time. Each is a module in `shardmesh/assistant/` with `attach(assistant)`. See [the design](../ASSISTANT.md) for the rules all parts follow.

Every command below runs under `shardmesh assistant`, which takes `--home` for the assistant's folder. Each part works with whichever others are present, and says so plainly when something it needs is missing.

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

`peer offer NAME CAPABILITY` lists one thing this install lets that peer call. An offer does not grant anything: the peer must be offered the capability and also pass this install's broker.

On this install a peer is its own user, `peer:<fingerprint>`, as well as its own actor, and it is always tainted. So the owner's grants never apply to a peer, a peer needs its own grant, and anything scoped to a user sees no data of the owner's.

Offering `models.complete` lets a peer use this install's models without seeing any memory. The prompt goes straight to the router, with no tools and nothing added. Its text is held to reach `POOL`, so it stays on this owner's machines.

Some things can never be offered: anything named `peer.`, `federation.`, `memory.`, `identity.`, or `skills.`, and any capability that declares it reads memories, identities, users, or grants. These are refused when offered, and again on every request.

A peer can read its own offers and nothing else, through the built-in `federation.offers`. Revoking removes every offer.

### Sending

`assistant.federation.send(peer, capability, arguments, items, context=...)` is how a part asks a peer for something:

1. Any `ContextItem` with a reach below `Reach.PEER` refuses the whole send, with a `FederationError`. Nothing is dropped silently.
2. It calls the local broker as `peer.send`, declared with `Effect.EXTERNAL` and scoped by `peer`. The items are part of the arguments, so an approval covers exactly what will be disclosed. EXTERNAL text among the items marks the call tainted, so a person has to approve it.
3. Only the text of each item and its source kind travel. Paths, row ids, and authors stay here.

Text placed in `arguments` has no provenance, so the reach check cannot see it. Put anything with a reach limit in `items`.

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

Each install has an Ed25519 signing key and an X25519 key, made on first use and stored in `federation/identity.key` in its home. Its fingerprint is the first 32 hex digits of `SHA-256("shardmesh-peer-identity-v1\0" || signing public || exchange public)`, so it covers both keys.

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
| `peer offer NAME [CAPABILITY] [--per-day N] [--withdraw]` | Offers a capability, at most N calls in any 24 hours, 200 by default. `--withdraw` stops offering it. With no capability, shows the peer's offers |
| `peer revoke NAME` | Ends the pairing |
| `peer ask NAME CAPABILITY [--text T] [--args JSON] [--user U]` | Calls something a peer offers, through this install's broker. `--text` is sent as the `prompt` argument, `--args` is a JSON object, and `--user` says who is asking. Exits 0 on success, 3 when the answer may come later, 1 otherwise |
| `peer collect NAME REQUEST` | Fetches an answer that was pending, or lost on the way |
| `peer offers NAME` | Asks a peer what it offers this install, and tells the router about its models |
| `peer serve [--host H] [--port P] [--address URL]` | Answers peers without the gateway. The port defaults to 9300, and 0 picks a free one. `--address` is where peers reach this install, if not where it listens |

### Routes

`routes(assistant)` gives these. A gateway must leave them open to strangers, because every message proves itself. There is deliberately no route for listing peers or offers.

| Route | Does |
|---|---|
| `GET /federation/identity` | This install's fingerprint and protocol version |
| `POST /federation/pair` | An owner accepting an invite |
| `POST /federation/request` | A sealed request |
| `POST /federation/notice` | A sealed notice |

Bodies over 1 MiB are refused before anything is parsed. Strangers get at most 30 audit rows a minute, and the next row says how many were not recorded, so junk cannot fill the record.

`FederationServer(assistant, host, port)` runs the routes alone, with `start()` returning the URL and `stop()`.

### Using a peer's models

`PeerProvider` lets the router use a peer's model at reach `PEER`. It has `name`, `reach`, and `complete(ModelRequest) -> ModelReply`. It refuses a request whose reach is below `PEER` even if the router sent it. Its replies never carry tool calls, and `provider` is `peer:<fingerprint>`: treat that text as `EXTERNAL`.

If `assistant.providers` has `register` or `add`, a provider is registered for each paired peer known to offer `models.complete`. `peer offers NAME` finds that out, and it is remembered across restarts.

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

It reads these keys from `tasks.status(task_id)`, and tolerates others: `state` or `status`, `user`, `task_type`, `type`, or `kind`, `goal`, `title`, or `summary`, `error`, `tainted`, and `steps`, each with `name`, `title`, or `id`, `state` or `status`, `capability`, `error`, `tainted`, and `approval_id` or `needs_person`. A task that does not say `tainted: false` counts as tainted.

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

`relevant(user, text, task_type)` returns active skills as `ContextItem`s, source `skill`, with their trust, at reach `POOL`. `gaps(user)` reports repeated failures by task type over 30 days: a type shows after two failures, with recent reasons, how often it needed a person, and whether a skill exists.

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

`git.commit` runs in the worktree, outside the sandbox, so it should commit with hooks off: a `pre-commit` hook that runs tests would run generated code on this machine.

`code push` uses `git.push`, which is denied unless granted, and only ever pushes the run's own branch. `code reject` removes the worktree and the branch. If the run was already committed, the branch stays, as that was the owner's decision.

There is deliberately no capability for approving or pushing, so no model, task, or peer can reach them. Calls name their arguments from what `git.commit` and `git.push` declare in `input_schema`: `repo` (or `repository`, `path`, `cwd`, `workdir`), `message` (or `msg`), `branch` (or `ref`), and `remote`. The worktree is the repository named, so a scoped grant must cover the assistant's `coder` folder.

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
