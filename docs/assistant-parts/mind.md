# Memory, models, and conversation

These are the parts that decide who the assistant is, what it knows, what a model sees, and how a turn runs: `identity`, `memory`, `knowledge`, `providers`, `router`, `context`, `conversation`, and the `gateway` that serves them. `docs/ASSISTANT.md` has the rules they follow. `tests/test_assistant_mind_docs.py` fails if a command, option, or route here goes undocumented.

Every command below is run as `shardmesh assistant [--home FOLDER] COMMAND`. Commands that act for a person take `--user NAME`, which defaults to `owner`. Over HTTP a key says who is calling, and `--user` does not exist there.

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

Reach defaults to `cloud`, the worst case, and to `pool` for a mesh. `machine` is refused for anything not on loopback.

The database holds the name of a variable or a file, never a key, and a key is read when it is used. Something that looks like a pasted key is refused. A bare UPPER_CASE name is a variable, and any other variable can be given as `env:name` once it is set. A key in an address is refused.

The mesh's chat API takes plain strings and has no tool calling, so tools go in the system prompt and calls are read from a fenced block. Thinking is on only for `planning`, `coding`, and `verification`. A reply from the mesh is always streamed, even when nobody is watching it, because only a stream can tell a slow pool from a lost link. Other servers use native `tools` where they accept them, falling back to the prompt if they refuse, with capability names mapped since a function name cannot contain a dot.

Models write tool calls in many shapes, and all of these are understood: a ` ```tool ` or ` ```json ` fence, `<tool_call>` tags, no fence at all, single quotes, trailing commas, and arguments as a string. Only tools that were offered count. Text in reasoning never counts, and neither does code. A call with no fence must end the reply. A call cut off by the token limit is never run, because a truncated argument is a different argument.

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
| `say TEXT` | Says one thing and prints the reply. `--conversation`, and `--no-learn` proposes no memories |
| `resume TURN` | Carries on a turn that was waiting. `--decline` ends it without acting |

What a person said is stored before anything else can fail. The loop runs at most 6 rounds of tools, then the model has to answer. Tool output is EXTERNAL whatever the tool claims, makes the rest of the turn tainted, and tightens its reach. It is cut for the model at a quarter of the budget and kept whole in the record.

When the broker answers ASK, the turn stops and is stored, and survives a restart. Resuming asks the broker again and does not take the caller's word for it, one caller only can take a turn, and memory is rebuilt, so what was forgotten while waiting stays forgotten. A turn taken by a process that died can be taken again.

Small or full is decided by software where it can: `/task` and `/now`, scheduling words, and three or more listed steps decide it, and a short message or a question stays small. Only what is left goes to the `decision` role, and staying small is the answer if that fails. Full work goes to `assistant.tasks` if attached, through `assistant.planner` if it has a `plan` or `propose` method. If the engine refuses a plan, the request is answered directly.

After a finished turn the `worker` role may propose up to 3 memories. They go no further than the turn did, since the reply may hold what the turn's text allowed, and they are stored as MODEL, or EXTERNAL from a tainted turn.

## The gateway

| Command | Does |
|---|---|
| `serve` | Runs the page and API. `--host`, `--port` (default 8470), `--tls-cert`, `--tls-key`, `--trusted-network` |
| `user add NAME` | Makes a user and writes their key to `keys/NAME.key`, or `--key-file`. `--role owner\|member`, and `--reclaim` gives a removed name and its data to this person. The first user is the owner |
| `user list` | Shows who may connect |
| `user remove NAME` | Ends access at once. Their data stays, and the name cannot be reused without `--reclaim` |
| `backup FILE --passphrase-file PATH` | Writes an encrypted copy of everything |
| `restore FILE --passphrase-file PATH` | Replaces this install's state. `--force` if it already holds data |

`serve` binds to loopback, and refuses any other address without TLS or `--trusted-network`, since keys cross the network with every request. It also refuses to start when nobody could connect.

A new key goes to a private file and never to the screen, and only a hash is stored. Keys go in `Authorization: Bearer`.

### Routes

Every path needs a key except `GET /`, `GET /assistant`, `GET /assistant/`, and `GET /healthz`. That is enforced before any route runs, so it covers routes other parts add. Such a route reads the caller from `request.state.user` and `request.state.role`. A path the gateway already serves keeps its first owner, and the clash goes in the audit trail.

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
| `GET /assistant/approvals` | What is waiting on this person |
| `POST /assistant/approvals/{approval_id}` | Decides one with `approve`, and carries the turn on |
| `GET /assistant/tasks` | This person's tasks, and whether the engine is installed |
| `POST /assistant/tasks` | Submits a plan |
| `GET /assistant/tasks/{task_id}` | One task |
| `POST /assistant/tasks/{task_id}/cancel` | Cancels one |
| `GET /assistant/audit` | The owner sees everything and a member sees their own actions. It holds ids and decisions, never what anyone said. `since`, `limit` |
| `GET /assistant/why/{turn}` | What a model saw on a turn |
| `GET /assistant/turns` | Recent turns. `limit` |
| `GET /assistant/models` | Measured results and routes |
| `GET /healthz` | Whether the server is up |
| `GET /`, `GET /assistant`, `GET /assistant/` | The page |

Only the newest user message in a chat request is new. A client's earlier messages and system prompt are not believed, since the assistant keeps its own record and its own identity. The OpenAI `user` field changes nothing. The real model is in the `assistant` object on the reply, or in the last event of a stream.

A turn runs on its own thread and belongs to the assistant, so a client that disconnects does not lose the reply. A failure partway keeps what arrived and says so.

The page is `assistant.html`: one file, no external resources, and a content security policy with a fresh nonce on each load. Text only ever reaches it as text. A key is kept in the tab's session storage and goes when the tab closes.

### Backups

`backup` takes a consistent copy with sqlite's backup API, then encrypts it with scrypt (N 2^15, r 8, p 1) and AES-256-GCM in 1 MiB chunks. Each chunk's position and whether it is last are in its nonce, and the header is authenticated, so a file that was changed, reordered, or cut short is refused. The clear copy sits beside the database, as protected as the database is, and is gone when the command returns. `restore` checks the copy's integrity and the audit chain, and anything using the folder has to be restarted afterwards.

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

## Meeting the other parts

These parts find the others with `getattr`, and work when they are absent.

| Looks for | Uses |
|---|---|
| `assistant.broker` | `capabilities`, `check`, `invoke` from the contract. To record a person's decision it tries `approve(id, by)` and `deny`, `reject`, or `decline`, then `decide(id, approved, by)`. For approvals it did not start from a turn it tries `pending(user)` or `pending_approvals(user)` |
| `assistant.tasks` | `submit`, `status`, `cancel` from the contract, and `list_for(user)` or `list(user)` if present. A task must say `user` in its status, or have been handed off from a conversation, or it stays hidden |
| `assistant.planner` | `plan(user, goal)` or `propose(user, goal)` |
| `assistant.skills` | `relevant`, `search`, or `match`, called with `(user, query, limit)` |
| `assistant.providers.register(provider)` | For a part to add a provider while running, such as a peer. It needs `name`, `kind`, `reach`, `complete(request, model)`, `stream(request, model)`, `current_model()`, and `health()` |

## What is not proven

- Nothing here has been run against the live pool. The mesh's API is held to its real `ChatCompletionRequest` schema in the tests, through a fake server.
- bm25 is computed across all users, so the order of one person's results can shift slightly with what others hold. Scores are not sent over HTTP for that reason, and an order alone tells very little, but it is not zero.
- SQLite gives a waiting writer no turn, so something writing back to back with no gap can hold the lock past the 10 s timeout. No part does that, and a command says so plainly if it happens.
- A reply streamed from a model that puts its reasoning before a bare `</think>`, with no opening tag, shows that reasoning while it streams. The stored reply is clean.
- Token counts are estimated at a character count over 3 when a server reports none, so a budget can be off for those servers.
