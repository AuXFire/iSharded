# Operating iSharded

iSharded has no central server. Every node runs the same control plane, the nodes elect one of themselves to coordinate, and any node can serve the API. This guide covers running a pool, what it does on its own when something fails, and what to watch.

## Start a pool

On the first node:

```powershell
isharded node --node-id node-a --gguf-dir D:\models
```

On every other node, name any node that is already running:

```powershell
isharded node --node-id node-b --join http://10.0.0.5:8209
```

The address and control port are worked out for you: `--advertise-host` defaults to `auto`, which prefers a Tailscale address and then the address on the route out, and the control plane listens on `--base-port` plus 9. Set `--advertise-host` when a machine has several networks and peers should use a particular one. Once a node has joined, it learns the rest of the mesh by gossip.

Node names decide who coordinates when nobody does yet (the lowest name wins), so they should be stable.

Load a model from any node:

```powershell
isharded load "ggml-org/Qwen3.8-27B-GGUF:Qwen3.8-27B-Q4_K_M.gguf" --url http://10.0.0.6:8209
```

`/chat` and `/dashboard` work on every node.

To start a node when its machine boots, use `scripts\install_node_task.ps1` on Windows, which registers a scheduled task, or `packaging/isharded-node.service` on Linux. A node restarts its own services, so these only have to start it once. The systemd unit has not been run on a real machine yet.

A node will not start on a network without protection. See `docs/SECURITY.md` for setting up credentials.

## Deployment options

| Network | What to set |
|---|---|
| Closed network, no internet | Put GGUF files on any node with `--gguf-dir`, and load by file name: `isharded load Qwen3.6-27B-Q4_K_M.gguf`. Give each node a llama.cpp build with `--llama-dir`, because automatic downloads need GitHub. |
| LAN | Use TLS, or pass `--trusted-network` if nobody else can read the network. Without either a node refuses to start, because the pool token would cross the LAN unencrypted. |
| VLANs | As for a LAN. Peers must be able to route to each other's ports. `--join` names a node directly, so it works across VLANs. The search `isharded up` makes on a LAN is a broadcast, which stays on one, so use an invite there. Open the control port and worker ports between them. |
| Tailscale | Nothing special. Links are encrypted, and addresses work from anywhere on the tailnet. Add certificates with tickets for per-node identity. This works with antivirus intercepting TLS. |
| Remote machines over the internet | Use a pool authority and TLS on every node (`--tls-cert`, `--tls-key`, `--ca-file`). Expect each extra stage to cost the round trip between those machines. |

A pool can mix these, as long as every node can reach every other node's advertised address.

### Models on a closed network

A file on one node is enough. Other nodes read the parts they need from it, the header and tokenizer included, and fall back to other holders if one fails. Files are identified by a fingerprint of their content, so every node agrees on what a file is whatever its path. Speculative decoding needs its draft file on some node too; without it, the model runs without a draft.

A CPU worker normally runs llama.cpp's CPU build and never touches a GPU. With `--llama-dir` it runs whatever build you give it, and a GPU build borrows the GPU for prompts even on a stage with no GPU layers. On Qwen3-0.6B Q8_0 with a CUDA build on 4 threads, that took 518 MiB of VRAM that no budget counts, and read 512 tokens in 0.15 s instead of 2.3 s. Leave room for it in `--gpu-memory-gb`, or give that worker a CPU build.

## What happens when something fails

The rule is that a failure or a pool change must not drop a reply, and that weights move only when they must.

| Event | What happens | Effect on a reply in progress | Measured |
|---|---|---|---|
| The coordinator dies | The lowest-named live node takes over and adopts the running route | None: replies go straight to the entry worker | Longest pause 0.22 s; new coordinator in 16 s; no weights moved |
| A stage process crashes | The worker starts a new one in place, which finds its saved states on disk. The worker is not penalized | Pauses, restores, reads only what came after the last saved state, carries on | 2.6 s on CPU with a small model, reading 149 tokens where it used to read 1,414; text identical |
| A worker process or whole node restarts | The node's services restart, and the coordinator gives it its stage back. Stages elsewhere stay loaded, and the node restores from its own disk | Pauses, restores, carries on | 11 s on CPU with a small model; back in 25 s with cached shards on a 27B model |
| A whole node dies, and a spare can take its layers | A spare takes those layers and nothing else moves. It fetches the copy of the lost stage's states | Pauses for as long as loading those layers takes, then restores | 23 s on CPU with a small model, reading 149 tokens; text identical |
| A whole node dies, with no spare | After 30 s away, the coordinator replans on the survivors, and the layers change, so no saved state applies | Pauses for as long as moving those layers takes, and reads the prompt | 36 s on CPU with a small model; 22 s when survivors had the layers |
| A node blips for under 30 s | Nothing moves | Waits, or replays if a request failed | Not measured on real hardware |
| A quarantined node recovers | It returns after three healthy checks, more if it keeps failing, and the coordinator takes it back while idle | None | Not measured on real hardware |
| A model loads or the layout changes | Nodes fetch in the background while the old layout serves. Entry workers let replies finish first. Stages whose job did not change stay loaded | Finishes first, up to 45 s | An unchanged layout re-applies in 3 s |
| A live node fails to load its stage | Two retries. If it still fails, a replan may go ahead only if it moves under 2 GiB. Otherwise the load stops and says why | Depends on the failure | Tested with simulated failures only |
| Every node restarts | Each node saved the route, so the coordinator loads the model again by itself | Reply lost; new requests wait | Observed; about a minute with cached shards |

A reply is lost when the node a client is connected to dies, because that node holds the connection. The chat page restarts the reply through another mesh node when that happens, and says so. API clients should do the same.

## Saved prompt states

Stages save what they computed for a prompt, at the end of a chat's system prompt and at the end of its settled history. [Conversation reuse](GUIDE.md#conversation-reuse) says what that does. This is how to run it.

**Reading the saving.** `metrics.cache` in `/admin/status` has `saved_percent`, and `/metrics` has `prompt_tokens_computed_total` and `prompt_tokens_reused_total`. `worker_tokens_reused_total` says where each token came from: `prefix` or `history`, then `ram`, `disk`, or `copy`. A `copy` means a stage took over from one that was lost.

**Sizing it.** A state holds every token up to where it was saved, for one stage's layers:

| Model | Per token | Fixed | State at 4,000 tokens, all stages |
|---|---|---|---|
| Qwen3-0.6B, measured | 114,700 bytes | None | 459 MB |
| Qwen3.5-0.8B, measured | 12,312 bytes | 20 MB | 69 MB |
| Qwen3.8-27B, worked out from its file, not measured | 65,536 bytes | 157 MB | 419 MB |

A conversation keeps about one state, as each turn replaces the last, and a system prompt shared by many conversations is one more. `--state-ram-gb` and `--state-disk-gb` are per worker. A node also keeps up to `--state-disk-gb` of copies for other stages, so count that twice. States leave 2 GiB of a disk free, as a model's cache does, and past that they stay in RAM only.

**What a copy costs.** Each turn that saves a state sends all of it to one other node, at up to `--state-copy-mb`. Sending only what changed is not built. At the default 10 MB per second, a stage's share of a 400 MB state takes about 20 s, so turns faster than that fall behind: a newer state replaces a queued one, and recovery uses whatever arrived. Lower the rate if a link is shared with tokens. `--state-copy-mb 0` sends none, and a lost node then costs a full reread. Copy traffic is `worker_state_copy_bytes_total`.

**When a copy does not help.** A stage only fetches a copy when it took over the same layers, so that needs a spare: a node outside the route that can hold them. With no spare, the survivors split the layers differently, and the prompt is read in full. On a slow link, fetching a copy can also take longer than reading the prompt on a fast GPU. A reply waits up to 30 s for a fetch and then reads the prompt instead.

**When the holder fails.** Replies do not notice. With the node holding the copies gone silent on real processes, time to the first token stayed between 2.7 and 3.4 s, and no reply failed or waited for it, while the copies failed in the background and showed in `worker_state_copies_failed_total`. The next repair names another holder, and it gets the next state saved, not the old ones.

**When a state is missing.** A stage says so, and the entry worker plans again from the next longest state, or from nothing. With `--state-disk-gb 0`, a killed stage process lost its states, and the reply recovered that way in 14.5 s with the same text, where restoring from disk took 2.6 s.

**What gets deleted.** Stages delete states when the model or their layers change, and copies go when no stage runs those layers. Nothing deletes a conversation's states when it ends. See [Saved conversations](SECURITY.md#saved-conversations).

## Planned maintenance

To take a node out of service:

```bash
isharded cordon node-b --url http://10.0.0.5:8209
```

Replies in progress finish, then the node's layers move to the rest of the pool. The node stays in the mesh and can still serve the API. When it is safe to work on, nothing in the route depends on it.

```bash
isharded uncordon node-b --url http://10.0.0.5:8209
```

This puts it back, and stops a move still under way. A cordon is shared by every node and survives restarts.

A cordon is refused if the pool could not hold the model without that node, and if it would transfer more than 2 GiB between nodes, which can take a long time on a slow link. `--force` allows the large transfer.

On the test pool, cordoning the laptop during a reply with Qwen3.6-27B: the reply finished with a longest pause of 0.67 s, the model was on the desktop alone 26 s later, and uncordoning restored the split in 21 s.

### Split networks

If the network splits, each side elects a coordinator and carries on with the workers it can reach. With two nodes there is no majority to tell which side should stop, so this favors staying up. It cannot corrupt anything: a worker obeys only the highest leadership term it has seen, and stale requests fail its epoch check. When the sides reconnect, the higher term wins and the other coordinator steps down. Keys and revocations made on either side are merged, so neither side loses its changes.

A lost link usually gives no error: packets just stop. A stage that gets no answer gives up after 30 s plus 1 s per token, and each hop waits longer than the one after it, so the right stage takes the blame. A reply waiting on an entry worker ends once that worker stops checking in. Either way the reply carries on with what is left, from a saved state if the stages can reach one.

Measured with real processes, a small model that fits on either side, and links that went silent rather than resetting:

| | |
|---|---|
| New coordinator on the cut-off side | 14 s |
| Each side serving alone | 45 s to 65 s after the cut |
| Reply in progress during the cut | Finished, identical text, longest pause 21 s to 37 s over two runs |
| One coordinator after healing | 0.7 s |
| Keys from both sides working on both | 0.1 s after that |
| Links lost in one direction only, for 60 s | No change of leader, both sides kept answering |

After healing, the pool stays on the layout the winning side settled on if the model fits there, since weights move only when they must. `POST /admin/rebalance` spreads it out again.

Failures, repairs, and leadership changes also go to each node's output with timestamps, so the reason for a failure survives the process that saw it. Use `--log-file` to keep it. A log over 64 MiB is set aside as `.1` when the node starts, so restart nodes now and then or rotate the file yourself.

## Several models

Each [lane](GUIDE.md#several-models-at-once) is a full control plane and a worker on each node, so everything here applies to a lane as it does to the main route, with two differences. Its ports are 1000 per lane above the node's own, so a firewall between nodes must allow them. Cordons pass to every lane, but `load`, `plan`, and rebalancing act on one route at a time.

A lane's memory share is fixed when it starts, from what each node had spare. To change it, remove the lane and load it again.

## Soak test

The last run: 21 minutes on the test pool with Qwen3.6-27B, four conversations running continuously with merged passes on, and a fault every two to three minutes. Faults were stage processes, workers, and control planes killed on both machines, cordons, and malformed or oversized requests.

| | |
|---|---|
| Replies finished | 405 |
| Replies lost for good | 0 |
| Replies that needed the client to retry | 11, because the node they were connected to had been killed or was restarting |
| Longest wait for a token, median reply | 0.6 s |
| Longest wait for a token, 99th percentile | 12.5 s |
| Longest wait for a token, worst | 24.5 s |

Three earlier runs each found stalls of about 3 minutes, which turned out to be four bugs: drained workers never released, route changes reported as stage errors, a dead stage restarted twice, and replies turned away during a move. Those are fixed. `GET /internal/diag/tasks` shows where everything in a process is waiting, for any future stall.

At temperature 0, some replies were not word-for-word the same as an earlier reply to the same question. In a separate test, the two differences inspected were sensible rewordings partway through a reply, which fits small numeric differences between batch shapes, though that cause was not proven. Do not rely on exact repeats under load.

These runs total under two hours on two machines. Leaks that take days to show would not appear here.

## Scale harness

Before a pool grows past the machines it has run on, the scale harness shows what its control plane will cost. It starts N emulated nodes on one machine. Each is the real control plane (gossip, elections, check-ins, health probes, status) and a real worker's check-in loop, with no model and a fixed stand-in for the hardware probe, so many fit on one box. Then it measures, for each node: calls and bytes per second by kind, how many others each node calls, CPU and memory, the status call's latency, what one pass of link timing costs, and how long the mesh takes to settle after a join, a leave, a coordinator kill, a frozen coordinator and a partition. [Results](RESULTS.md#control-plane-at-scale) has the numbers so far.

```bash
# One size, the default phases, results as JSON (about 5 minutes at N=10):
python tests/scale/harness.py --nodes 50 --out scale-50.json

# Every phase, including an unplugged node, a frozen coordinator and a partition (about 7 minutes at N=20):
python tests/scale/harness.py --nodes 20 --phases all --out scale-20.json

# N=100 and N=200 on a big CPU box (a 96-core pod), one process a node, growth table at the end:
python tests/scale/harness.py --sweep 10,50,100,200 --phases all --per-process 1 --normal-priority --out-dir scale-results

# The table again, from files written earlier:
python tests/scale/report.py scale-results/scale-*.json
```

Run it from a checkout with the package installed (`pip install -e .`). It listens on 127.0.0.1 only, on free ports it picks, and keeps each run's files and logs in `--run-dir` (a temporary folder by default). It lowers its own processes' priority; `--normal-priority` turns that off.

- **Sizes.** By default nodes share processes, about N divided by the core count to each, which keeps memory to about 55-75 MB a node. Anything a phase kills or freezes always runs alone. What limits a run is CPU: a node's control plane used 0.04 cores at N=10, 0.08 at N=20 and 0.15 at N=30, and the need grows faster than N. Keep the machine's CPU, which each steady window records, well under 90%; past that the nodes slow each other and the numbers describe the box, not the code. A 16-thread laptop shared with other work kept up to about N=20. `--per-process 1` gives each node a process of its own, as a real node has, at about 95 MB a node: N=200 that way needs about 19 GB and, by the CPU trend, a core or more per node, so a 96-core pod should take N=100 cleanly and N=200 only as a measure of where it breaks. On Linux the harness raises its own open-file limit, since every node keeps a connection to every other.
- **Phases.** `--phases` picks them: `steady`, `status`, `links` (the coordinator has every worker time its link to every other, as it does before planning a route), `join`, `leave` (the process killed), `silent-leave` (the node unplugged), `kill-leader`, `freeze-leader` (suspended, then resumed) and `partition` (two halves that cannot hear each other, then healed). A phase named twice runs twice; the default runs `steady` before and after a leave. A steady window in which any process's event loop stopped for 2 s or more (paging, or a busy machine) is measured again, `--stall-retries` times (1 by default), and the result says how many windows it threw away and whether the last one stalled too.
- **Network.** `--delay-ms`, `--jitter-ms` and `--loss` emulate the network inside each node's HTTP client: every call waits the delay each way, and a lost call goes silent. A silent call, whether lost, unplugged or partitioned, fails after as long as a real one would: one read timeout over a connection that was open, or every connect attempt and its retry back-off with none. No root or netem is needed, so it runs the same on Windows, macOS and Linux.
- **Smoke test.** `tests/test_scale.py` runs five nodes through `steady`, `links`, `join`, `leave` and `kill-leader` in 60 to 100 seconds as part of the normal suite, and checks the counts against what the code's own periods say: gossip to every peer every 3 s, a check-in with every control plane every 5 s, a health probe of every worker every 10 s, 20 link probes of 5 echoes each, and one election when the coordinator dies.
- **Nightly.** The `scale` workflow runs N=20 with every phase each night on a 4-core Linux runner, puts the table in the run's summary and keeps the JSON and the logs. N=50 needs about four times the cores a standard runner has, so it is not the nightly size; started by hand on a larger runner, the workflow takes another size or other phases.

What it does not show: model traffic, a route being planned or repaired (the `links` phase times links the way planning does, but plans nothing), TLS, and real wide-area networks. Linux network namespaces with `tc netem` would put delay and loss in the kernel rather than in each node's client, but need root on every box; the client-side emulation gives the same timeouts and runs anywhere. Every node is on one machine, so CPU figures include the harness's counting, and a busy machine slows every node at once: the machine CPU and loop lag figures say when that happened.

## What to watch

Every node serves these for itself.

| Path | Who | Purpose |
|---|---|---|
| `/health` | Anyone | The process is up. |
| `/ready` | Anyone | This node can serve a reply now. Use it behind a load balancer. |
| `/metrics` | Admin | Prometheus format. |
| `/internal/mesh` | Nodes | This node's own view of the mesh, for comparing nodes. |

Metrics worth alerting on:

| Metric | Meaning |
|---|---|
| `isharded_stream_fallbacks_total` rising | Replies are failing and being replayed. Fallbacks keep replies alive, so nothing else may look wrong. |
| `isharded_worker_stage_restarts_total` rising | A stage keeps crashing. |
| `isharded_workers_quarantined` above zero | A worker is held out of the route. |
| `isharded_mesh_members` below the node count | A node cannot see part of the mesh. |
| `isharded_serving` at zero | This node cannot serve replies. |
| `isharded_worker_draft_tokens_per_pass` near 1 | The draft is not helping. |
| `isharded_stage_errors_total` rising | Stages are refusing requests. Nobody is quarantined for this, so a real fault in a stage shows up here and in failed replies. |
| `isharded_worker_state_misses_total` rising | Stages are evicting states the entry worker still counts on. Each one costs a reread and nothing else, so raise `--state-ram-gb` or `--state-disk-gb` if it keeps climbing. |
| `isharded_worker_state_copies_failed_total` rising | Copies of saved states are not arriving. Replies are fine, but a lost node would cost a full reread. |

## Files each node keeps

| What | Default | Notes |
|---|---|---|
| Route state | `orchestrator-state.json` in `%LOCALAPPDATA%\iSharded`, or `~/.cache/iSharded` off Windows | Copied to every node by gossip. |
| Audit log | `audit.jsonl` in the same folder | Rotates at 64 MiB, keeping one old file. |
| Node log | `--log-file` | Nothing is kept unless you set this. `isharded up` sets it to `node.log` in its folder. |
| Fetched shards | `--cache-dir` | Evicted by age past `--cache-budget-gb`. |
| Stage files | Inside the cache | Rebuilt from shards when a stage moves. Count double the node's share. |
| Saved prompt states | `states` inside the cache | People's conversations, unencrypted, readable only by the node's own user. Evicted past `--state-disk-gb`, and deleted when the model or the stage's layers change. |

The dashboard's "Memory for weights" counts weights only. Real use is higher: KV cache, compute buffers, a draft model, and each worker's own GPU context.

Stages read their weights once rather than mapping the file, unless some of them are on the disk tier. A mapped file stays resident in RAM even when the tensors live in VRAM: with Qwen3.6-27B on the 7900 XT, mapping cost 15.1 GiB of RAM and reading cost 1.0 GiB, at the same 35.6 tok/s. Windows reports "private" memory near the VRAM size in both cases, which is GPU allocation and not RAM.

Request logging is off, because nodes poll each other constantly. Set `ISHARDED_ACCESS_LOG=1` to turn it on while debugging.

## Upgrading

Nodes refuse peers on a different `PROTOCOL_VERSION`, and the refusal names both versions. Upgrade all nodes together when that number changes. Otherwise, restart nodes one at a time, and the mesh carries on around each one.

`isharded/llamacpp.py` is written against one llama.cpp build, `PINNED_BUILD`. A different build fails a check at load rather than misreading memory. Check `llama.h` before changing it.
