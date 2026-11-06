# Operating ShardMesh

ShardMesh has no central server. Every node runs the same control plane, the nodes elect one of themselves to coordinate, and any node can serve the API. This guide covers running a pool, what it does on its own when something fails, and what to watch.

## Start a pool

On the first node:

```powershell
shardmesh node --node-id node-a --gguf-dir D:\models
```

On every other node, name any node that is already running:

```powershell
shardmesh node --node-id node-b --join http://10.0.0.5:8209
```

The address and control port are worked out for you: `--advertise-host` defaults to `auto`, which prefers a Tailscale address and then the address on the route out, and the control plane listens on `--base-port` plus 9. Set `--advertise-host` when a machine has several networks and peers should use a particular one. Once a node has joined, it learns the rest of the mesh by gossip.

Node names decide who coordinates when nobody does yet (the lowest name wins), so they should be stable.

Load a model from any node:

```powershell
shardmesh load "ggml-org/Qwen3.8-27B-GGUF:Qwen3.8-27B-Q4_K_M.gguf" --url http://10.0.0.6:8209
```

`/chat` and `/dashboard` work on every node.

To start a node when its machine boots, use `scripts\install_node_task.ps1` on Windows, which registers a scheduled task, or `packaging/shardmesh-node.service` on Linux. A node restarts its own services, so these only have to start it once. The systemd unit has not been run on a real machine yet.

A node will not start on a network without protection. See `docs/SECURITY.md` for setting up credentials.

## Deployment options

| Network | What to set |
|---|---|
| Closed network, no internet | Put GGUF files on any node with `--gguf-dir`, and load by file name: `shardmesh load Qwen3.6-27B-Q4_K_M.gguf`. Give each node a llama.cpp build with `--llama-dir`, because automatic downloads need GitHub. |
| LAN | Use TLS, or pass `--trusted-network` if nobody else can read the network. Without either a node refuses to start, because the pool token would cross the LAN unencrypted. |
| VLANs | As for a LAN. Peers must be able to route to each other's ports. Discovery does not use broadcast, so `--join` works across VLANs. Open the control port and worker ports between them. |
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
| A stage process crashes | The worker starts a new one in place. The worker is not penalized | Pauses, replays the prompt, carries on | 10 s on CPU with a small model; text identical |
| A worker process or whole node restarts | The node's services restart, and the coordinator gives it its stage back. Stages elsewhere stay loaded | Pauses, replays, carries on | Back in 25 s with cached shards, same layout, 27B model |
| A whole node dies | After 30 s away, the coordinator replans on the survivors | Pauses for as long as moving those layers takes | 22 s when survivors had the layers |
| A node blips for under 30 s | Nothing moves | Waits, or replays if a request failed | Not measured on real hardware |
| A quarantined node recovers | It returns after three healthy checks, more if it keeps failing, and the coordinator takes it back while idle | None | Not measured on real hardware |
| A model loads or the layout changes | Nodes fetch in the background while the old layout serves. Entry workers let replies finish first. Stages whose job did not change stay loaded | Finishes first, up to 45 s | An unchanged layout re-applies in 3 s |
| A live node fails to load its stage | Two retries. If it still fails, a replan may go ahead only if it moves under 2 GiB. Otherwise the load stops and says why | Depends on the failure | Tested with simulated failures only |
| Every node restarts | Each node saved the route, so the coordinator loads the model again by itself | Reply lost; new requests wait | Observed; about a minute with cached shards |

A reply is lost when the node a client is connected to dies, because that node holds the connection. The chat page restarts the reply through another mesh node when that happens, and says so. API clients should do the same.

## Planned maintenance

To take a node out of service:

```bash
shardmesh cordon node-b --url http://10.0.0.5:8000
```

Replies in progress finish, then the node's layers move to the rest of the pool. The node stays in the mesh and can still serve the API. When it is safe to work on, nothing in the route depends on it.

```bash
shardmesh uncordon node-b --url http://10.0.0.5:8000
```

This puts it back, and stops a move still under way. A cordon is shared by every node and survives restarts.

A cordon is refused if the pool could not hold the model without that node, and if it would transfer more than 2 GiB between nodes, which can take a long time on a slow link. `--force` allows the large transfer.

On the test pool, cordoning the laptop during a reply with Qwen3.6-27B: the reply finished with a longest pause of 0.67 s, the model was on the desktop alone 26 s later, and uncordoning restored the split in 21 s.

### Split networks

If the network splits, each side elects a coordinator and carries on with the workers it can reach. With two nodes there is no majority to tell which side should stop, so this favors staying up. It cannot corrupt anything: a worker obeys only the highest leadership term it has seen, and stale requests fail its epoch check. When the sides reconnect, the higher term wins and the other coordinator steps down. Keys and revocations made on either side are merged, so neither side loses its changes.

A lost link usually gives no error: packets just stop. A stage that gets no answer gives up after 30 s plus 1 s per token, and each hop waits longer than the one after it, so the right stage takes the blame. A reply waiting on an entry worker ends once that worker stops checking in. Either way the reply replays on what is left.

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

Each [lane](../README.md#several-models-at-once) is a full control plane and set of workers, so everything here applies to a lane as it does to the main route, with two differences. Its ports are 1000 per lane above the node's own, so a firewall between nodes must allow them. Cordons pass to every lane, but `load`, `plan`, and rebalancing act on one route at a time.

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
| `shardmesh_stream_fallbacks_total` rising | Replies are failing and being replayed. Fallbacks keep replies alive, so nothing else may look wrong. |
| `shardmesh_worker_stage_restarts_total` rising | A stage keeps crashing. |
| `shardmesh_workers_quarantined` above zero | A worker is held out of the route. |
| `shardmesh_mesh_members` below the node count | A node cannot see part of the mesh. |
| `shardmesh_serving` at zero | This node cannot serve replies. |
| `shardmesh_worker_draft_tokens_per_pass` near 1 | The draft is not helping. |
| `shardmesh_stage_errors_total` rising | Stages are refusing requests. Nobody is quarantined for this, so a real fault in a stage shows up here and in failed replies. |

## Files each node keeps

| What | Default | Notes |
|---|---|---|
| Route state | `%LOCALAPPDATA%\ShardMesh\orchestrator-state.json` | Copied to every node by gossip. |
| Audit log | `%LOCALAPPDATA%\ShardMesh\audit.jsonl` | Rotates at 64 MiB, keeping one old file. |
| Node log | `--log-file` | Nothing is kept unless you set this. |
| Fetched shards | `--cache-dir` | Evicted by age past `--cache-budget-gb`. |
| Stage files | Inside the cache | Rebuilt from shards when a stage moves. Count double the node's share. |

The dashboard's "Memory for weights" counts weights only. Real use is higher: KV cache, compute buffers, a draft model, and each worker's own GPU context.

Stages read their weights once rather than mapping the file, unless some of them are on the disk tier. A mapped file stays resident in RAM even when the tensors live in VRAM: with Qwen3.6-27B on the 7900 XT, mapping cost 15.1 GiB of RAM and reading cost 1.0 GiB, at the same 35.6 tok/s. Windows reports "private" memory near the VRAM size in both cases, which is GPU allocation and not RAM.

Request logging is off, because nodes poll each other constantly. Set `SHARDMESH_ACCESS_LOG=1` to turn it on while debugging.

## Upgrading

Nodes refuse peers on a different `PROTOCOL_VERSION`, and the refusal names both versions. Upgrade all nodes together when that number changes. Otherwise, restart nodes one at a time, and the mesh carries on around each one.

`shardmesh/llamacpp.py` is written against one llama.cpp build, `PINNED_BUILD`. A different build fails a check at load rather than misreading memory. Check `llama.h` before changing it.
