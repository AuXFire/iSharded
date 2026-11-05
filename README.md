# ShardMesh

ShardMesh runs one language model across multiple independently managed computers, using whatever memory each one has: VRAM, unified memory, system RAM, or disk. It has no central server. Every node runs the control plane, the nodes elect one to coordinate, and any node can serve the API. Each worker downloads only the tensors it will run, keeps its own per-layer state, and forwards activations directly to the next worker.

It runs on a closed network, a LAN, across VLANs, over Tailscale, or across remote machines with TLS. See [Operating ShardMesh](docs/OPERATIONS.md) and [ShardMesh security](docs/SECURITY.md).

This is execution sharding, not simulated work distribution. In a two-worker run of the included offline test model, worker 1 owns layer 0 and worker 2 owns layer 1. Token generation crosses both processes on every target-model pass.

## What works

- A streaming chat page at `/chat` that shows where each layer of the model is running.
- Qwen2, Qwen2.5, Qwen3, Qwen3.5, Qwen3.6, and Qwen3.8 text models, dense and mixture-of-experts, plus GPT-2.
- Decentralized control: an elected coordinator, gossip membership, state copied to every node, and the API on any node.
- Replies that survive failures: losing the coordinator does not touch a reply in progress, and a crashed stage restarts in place and replays.
- Closed networks: nodes serve models to each other, so a pool needs no internet.
- GGUF models split across nodes on llama.cpp's own kernels, with each node fetching only its own layers.
- Speculative decoding across the pipeline, on by default for GGUF models with a matching draft.
- Conversation reuse: later chat turns process only what is new, not the whole history.
- Shard-only loading: each worker pulls just its tensors with HTTP range requests, straight from the model host.
- Per-node memory pools. Any worker can mix VRAM, unified memory, RAM, and disk, and every node can differ.
- BF16 by default, with FP32 available per node, and FP8 checkpoints for half the download and memory.
- Contiguous transformer-layer sharding across one or more workers.
- Direct worker-to-worker activation transfer using safetensors rather than pickle.
- Per-worker, per-layer KV caches.
- Bounded RAM contributors that retain incremental KV checkpoints for recovery.
- One-command nodes that can contribute GPU inference, CPU inference, RAM, or any combination.
- CUDA, ROCm, MPS, and CPU capability discovery with stage-boundary dtype conversion.
- Worker heartbeats, epoch-versioned pipelines, and explicit rebalancing.
- Automatic failover: quarantine a failed stage, repartition, replay the committed prefix, and resume.
- Safe node joins: a worker registers as standby until `POST /admin/rebalance` activates a new epoch.
- Nodes refuse to start exposed. Certificates or a pool token for nodes, named API keys with roles and rate limits for people, per-node revocation, TLS, and an audit log.
- Planned maintenance: cordon a node and its share moves elsewhere while replies finish.
- Prometheus metrics and readiness checks on every node.
- OpenAI-shaped `/v1/completions`, `/v1/chat/completions`, and `/v1/models` endpoints, with streaming.
- LAN, Tailscale IP, MagicDNS, or Tailscale Serve URLs.

## Quick start on one machine

Python 3.10+ is required.

```powershell
cd D:\project
python -m pip install -r requirements.txt
python -m shardmesh demo --workers 2
```

When those dependencies are already installed, the checkout runs directly and the install command can be skipped. `python -m pip install -e .` additionally installs the three console commands, but is not required.

The default model is `distilbert/distilgpt2`. The demo starts an orchestrator on port 8000 and workers on ports 8100 and 8101.

Open [http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat), choose a model, and start chatting. The picker shows where a model would run and how fast before anything downloads, and the rail beside the conversation colors each layer by the memory holding it.

Open the live control dashboard at [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard). It visualizes the active layer pipeline, node states, hardware backends, control and compute latency, throughput, memory, recovery count, and control-plane activity. Click any stage or worker for detailed telemetry; **Rebalance** safely activates a new pipeline epoch.

For a network-free functional test, build the deterministic two-layer model and run it on CPU:

```powershell
python scripts\build_tiny_model.py .smoke-model
python -m shardmesh demo --workers 2 --model "$PWD\.smoke-model" --device cpu
```

The offline `.smoke-model` is randomly initialized with a tiny vocabulary. Its output is intentionally not meaningful—repetition such as `time time time` only verifies that distributed and unsharded execution agree. Use a trained checkpoint such as DistilGPT-2 to evaluate generation quality. The dashboard labels this checkpoint as a **protocol test model**.

Then send a request from another terminal:

```powershell
$body = @{
  model = "distilbert/distilgpt2"
  prompt = "Once upon a time"
  max_tokens = 24
  temperature = 0
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/completions `
  -ContentType application/json `
  -Body $body
```

PowerShell users should prefer `ConvertTo-Json` plus `Invoke-RestMethod`. If using `curl.exe -d` directly, Windows paths must contain doubled backslashes inside the JSON string, and native-command quoting varies between PowerShell versions.

Inspect the virtual pool and exact layer placement at `http://127.0.0.1:8000/admin/status`.

## Supported models

| Family | `model_type` | Examples | Notes |
|---|---|---|---|
| GPT-2 | `gpt2` | `distilbert/distilgpt2`, `openai-community/gpt2-xl` | |
| Qwen2, Qwen2.5 | `qwen2` | `Qwen/Qwen2.5-0.5B-Instruct` | |
| Qwen1.5/2 MoE | `qwen2_moe` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | Routed and shared experts |
| Qwen3 | `qwen3` | `Qwen/Qwen3-8B` | |
| Qwen3 MoE | `qwen3_moe` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | |
| Qwen3.5, 3.6, 3.8 | `qwen3_5` | `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.8-27B` | Hybrid Gated DeltaNet and attention |
| Qwen3.5, 3.6 MoE | `qwen3_5_moe` | `Qwen/Qwen3.6-35B-A3B` | Hybrid, with fused experts |

Multimodal releases load as text models: the vision tower and multi-token-prediction head are skipped and never downloaded.

Block-scaled FP8 releases of these models work too, such as `Qwen/Qwen3.8-27B-FP8`. Not supported yet: AWQ, GPTQ, Qwen3-Next, and the experimental `qwen4_exp` in Qwen3.8-Flash-Next. An unsupported model is refused with the reason before anything downloads.

Workers run these models with their own forward pass instead of `transformers` classes, so a node's library versions do not limit which models it can run. Each adapter is tested against the reference `transformers` implementation.

## GGUF models

GGUF is the fast path. A quantized model runs on llama.cpp's kernels, which is what makes a 27B model practical on consumer GPUs, while ShardMesh still plans, fetches, and routes it.

```powershell
shardmesh load "ggml-org/Qwen3.8-27B-GGUF:Qwen3.8-27B-Q4_K_M.gguf"
```

A model id is `owner/repo:file.gguf`, or a path to a local `.gguf` file. Supported architectures are `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, and `llama`. Real weights have been checked for `qwen3`, `qwen35`, and `qwen35moe`, where two stages matched the whole model; `qwen35` has also run across real machines. The others share that code but are untested. Tokenizers are built from the GGUF itself, for `qwen2` and `qwen35` vocabularies. Split GGUF files (`-00001-of-00003`) are refused for now.

How it works:

- A GGUF header lists every tensor's byte offset, so each worker fetches just its layers with range requests. The cache, peer sharing, and cached-shard planning are the same as for safetensors. Nothing about llama.cpp's own RPC is used, because that needs the whole file on one machine.
- A worker assembles a **stage file** from its cached units: its layers renumbered from zero, with placeholders for anything it never evaluates. To llama.cpp this is a small complete model, so stock release builds work, with no patches.
- Hidden states enter through `llama_batch.embd` and leave through the eval callback at the stage's last layer. Measured on Qwen3.6-27B, a two-stage split gives the same tokens and logits as the whole model, and the boundary costs under 3 ms a token.
- Each stage runs in its own process. PyTorch and llama.cpp each load an OpenMP runtime and abort when they share one, and this way a native crash cannot take the worker down and unloading frees everything.
- The planner's placement decides what sits in VRAM. Layers placed in RAM or on disk are pinned to the CPU, and llama.cpp memory-maps the file, so a node can run part of its stage from a GPU and part from RAM. Experts are placed separately from the rest of a layer, so a MoE can keep attention on the GPU and its experts in RAM.

| Worker flag | Purpose |
|---|---|
| `--gguf-dir PATH` | Search a folder for GGUF files already on this machine, such as LM Studio's. A match on name and size is read locally instead of downloaded. Repeatable. |
| `--llama-dir PATH` | Use this llama.cpp build. Otherwise a pinned, hash-checked Windows build for the node's backend is downloaded to `%LOCALAPPDATA%\ShardMesh\llama`. Linux and macOS need this flag. |

The Windows ROCm release leaves out `hipblas.dll`. ShardMesh finds it in the node's PyTorch ROCm install, so the node needs that environment.

### Speculative decoding

A token's trip through the pipeline costs nearly the same whether it carries one token or six, so the entry worker runs a small draft model that guesses ahead, and the real model checks every guess in one pass. Each pass yields the guesses that held, plus one more token for free.

For `qwen35` models the draft is chosen automatically (`Qwen3.5-0.8B`, which shares their vocabulary). Name another with `shardmesh load MODEL --draft owner/repo:file.gguf`, or turn this off with `--no-draft`. The same switches are `draft_model` and `draft_tokens` on `POST /admin/load`. A draft that cannot load leaves the model running without one. A named draft must have the same vocabulary as the model.

The number of guesses adapts as a reply goes, up to `draft_tokens`, because code rewards long guesses and open-ended prose does not. Sampling keeps the model's distribution exactly: a guess `g` is accepted with probability `q(g)`, and a rejection draws from `q` with `g` removed. At temperature 0, batched arithmetic rounds slightly differently from single-token arithmetic, so output can differ from a run without a draft where two tokens are nearly tied.

### Conversation reuse

Every stage saves its state where a chat's settled history ends. If the next prompt starts with exactly those tokens, stages restore the checkpoint and process only the new text, so a long conversation stops costing more with every turn. Anything else, such as an edited history or a new route, falls back to reading the whole prompt. A restored stage matches a stage that never stopped, bit for bit. Processing a prompt in different pieces rounds slightly differently, so a reply can differ at a near-tie from one produced in a single pass.

### Several conversations at once

`shardmesh load MODEL --parallel 3` gives every stage three independent sequences, each with its own context memory, which the planner counts. Replies from any node queue at the entry worker, and each conversation returns to the sequence holding its checkpoint. While one reply is away at the next stage, this stage works on another, so more of the pool is busy at once. Stages run sequences in turn, not merged into one batch, so each weight read still serves a single reply.

### Measured performance

Qwen3.6-27B Q4_K_M, split across an RX 7900 XT desktop and an RTX 3060 laptop, both on Wi-Fi with a 15 ms ping:

| Setup | Tokens per second |
|---|---:|
| Desktop alone, llama.cpp, no split (ceiling) | 35.2 |
| Both nodes, llama.cpp's own RPC, whole file on one node | 13.6 |
| Both nodes, ShardMesh, no draft | 14.6 |
| Both nodes, ShardMesh with a draft, prose and lists | 19–21 |
| Both nodes, ShardMesh with a draft, code | 33 |
| Both nodes, ShardMesh with a draft, chat page settings, four-turn chat | 21.6 |

Models that need or suit the split, on the same two machines at chat page settings with a draft:

| Model | Placement | Tokens per second |
|---|---|---:|
| Qwen3.8-27B Q8_0, 26.6 GiB, too big for either machine | Desktop 14 GiB VRAM + 3.9 GiB RAM, laptop 3.1 GiB VRAM + 5.7 GiB RAM | 4.2–6.7 |
| Qwen3.6-35B-A3B IQ2_XXS, a MoE | Laptop 1.2 GiB VRAM, desktop 8.8 GiB VRAM | 18–43 |

Estimates from the planner leave out the draft and are rough. The figures above were measured.

Three conversations at once on the same pair, `--parallel 3`, through both nodes' APIs: 24–29 tokens/s in total over five runs, against 13–14 for one conversation in the same layout. One earlier run reached only 13, unexplained. Each conversation slows to about 10–12 tokens/s.

A 1,860-token first message took about 6.5 s to the first token. Later turns of that chat took 0.7 to 1.3 s, with over 2,000 tokens of history behind them. Downloads share the link: chat dropped to about 5 tokens/s while a new model downloaded over the desktop's Wi-Fi.

## How a model loads

1. The orchestrator reads `config.json`, the tokenizer, and each safetensors header. This costs a few megabytes and pins one commit for every node.
2. It plans which worker runs which layers, and which memory holds each unit.
3. Each worker receives an allowlist of tensors and pulls those byte ranges directly from the model host, into a local cache of small per-unit files.
4. The current model keeps serving through all of this. The switch happens only when every stage has its shards, and waits for a request boundary.
5. Each worker moves units into their pools one tensor at a time, so a checkpoint never has to fit in RAM to reach a GPU.

If a worker fails during a load, the orchestrator replans around it and finished units stay cached. A rebalance fetches only what moved. Interrupted downloads resume from the last byte received, and `POST /admin/load/cancel` stops a load.

Two behaviors keep downloads down, and both are on by default:

- **Shards in place count.** Each plan is charged its download time spread over 20,000 tokens, so a near-tie stays where the shards already are, while a real speedup still earns its download.
- **Workers share shards.** A worker asks other workers for a unit before going to the model host, and falls back to the host if a peer fails or sends the wrong size or header.

| To turn off | For one load | As an orchestrator default | On one node |
|---|---|---|---|
| Weighing cached shards | `shardmesh load MODEL --no-reuse-cached` | `--no-reuse-cached` | |
| Fetching from peers | `shardmesh load MODEL --no-peer-fetch` | `--no-peer-fetch` | `--no-peer-fetch` |
| Serving shards to peers | | | `--no-peer-serve` |

`shardmesh load` loads a model on a running pool and also takes `--entry`, `--exit`, `--placement`, and `--context-tokens`. The same switches are `reuse_cached` and `peer_fetch` on `POST /admin/load`. Peers authenticate with the pool token, so set one if the network is shared.

Gated models need `HF_TOKEN` set on each worker. A worker can also read from a local folder through `worker_model_ids`.

The Hub only publishes checksums for whole files, which a partial reader cannot verify. ShardMesh pins the commit and checks every unit's exact size, which catches truncation and mismatched revisions but not corruption inside a range.

## Memory pools

Every worker reports its own pools, and nodes do not have to match.

| Kind | Examples | Executes on | Budget flag |
|---|---|---|---|
| VRAM | NVIDIA, AMD ROCm, Intel XPU | That GPU | `--gpu-memory-gb` |
| Unified | Apple MPS, integrated GPUs | That GPU, sharing RAM | `--gpu-memory-gb` |
| RAM | System memory | CPU | `--weights-ram-gb` (a size or `auto`) |
| Disk | NVMe or SSD, memory-mapped | CPU, paged in by the OS | `--weights-disk-gb` |

Each pool reports its budget, free space, precision, and measured bandwidth. The planner then:

- Treats a layer as separately placeable units: attention and other dense weights, routed experts, the embedding, and the output head.
- Keeps the most frequently read weights in the fastest memory. Every token reads a layer's dense weights but only `top_k / num_experts` of its experts, so experts move to slower memory first.
- Never spends the same physical memory twice, whether two workers share a host's RAM or a GPU shares it through unified memory.
- Promotes standby workers only when a model does not fit without them.

The ends of the pipeline are not equal: the exit reads the whole output head for every token, while the entry reads one embedding row. So the planner tries every order through the workers and keeps the fastest (up to four workers exhaustively, then a sample of orders). Pin the ends with `entry_worker` and `exit_worker` on a load, or in the chat page's picker, for example to keep prompts and replies on a machine you trust. Only the entry sees prompt tokens and only the exit sees generated tokens, though activations passing through other stages can still reveal information. One worker pinned to both ends runs the whole model, because a worker holds one stage. If a pinned worker is lost, recovery drops the pin and logs it.

`placement: spread` (the default) uses every preferred worker and minimizes total token time, because a request crosses stages one after another. `compact` may skip workers, which is usually faster for one user because each hop costs a network round trip.

An explicit `--gpu-memory-gb` is a hard allocator limit on CUDA and ROCm. RAM and disk budgets are enforced by placement. Disk is a capacity tier: expect it to be far slower than RAM once a model outgrows the OS page cache, and use `POST /admin/plan` or the chat page's picker to see the estimate first.

Put `--cache-dir` on a fast drive with room for the worker's share of the model. Every fetched unit stays there so that rebalances are cheap, whichever pool it runs from. `--cache-budget-gb` evicts the least recently used units from other models past that size.

## Precision

Workers run in BF16 by default. Add `--dtype float32` to a node to run its VRAM and RAM at full precision, which doubles the memory each layer takes. The planner accounts for that, so BF16 and FP32 nodes can share a route.

Published Qwen checkpoints are BF16, so FP32 raises the precision of computation and not of the stored weights. The disk tier maps files as they are and always runs in the checkpoint's own dtype.

### FP8 checkpoints

An FP8 release halves the download. In memory, the planner chooses per unit between two forms:

| Form | Memory | Speed measured on an RTX 3060 and a laptop CPU |
|---|---|---|
| Expanded to BF16 at load | Twice the FP8 size | Full speed: 143 Gparam/s on GPU, 14.5 on CPU |
| Compact, expanded on each use | FP8 size | 20 Gparam/s on GPU, 0.65 on CPU |

Compact is several times slower because PyTorch has no fused FP8 kernel, and neither RTX 30 nor RDNA3 GPUs compute in FP8 natively. So the planner expands what fits and keeps only the overflow compact, preferring the GPU for it, since compact on a GPU still beats BF16 in RAM. On a CPU it is slower than disk reads of BF16, so it is the last resort.

The gain comes from avoiding disk and from fitting models that BF16 cannot, not from faster reads. A model that fits comfortably in BF16 runs at the same speed from either release.

The expansion rates are fixed constants in `planner.py`, measured on one machine, so plans for other hardware are estimates.

## Contribute CPU, RAM, and GPU

The `node` launcher turns one physical computer into independent ShardMesh services. Start the orchestrator first:

```powershell
python -m shardmesh.orchestrator --host 0.0.0.0 --port 8000
```

Then run this on a contributing computer, using its LAN or Tailscale IP or MagicDNS name for `--advertise-host`:

```powershell
python -m shardmesh node `
  --node-id workstation-01 `
  --orchestrator-url http://100.80.10.1:8000 `
  --advertise-host 100.80.10.21 `
  --base-port 8200 `
  --gpu-device cuda:0 `
  --gpu-memory-gb 4.5 `
  --cpu-threads 8 `
  --weights-ram-gb 12 `
  --weights-disk-gb 100 `
  --cache-dir D:\shardmesh `
  --ram-budget-gb 2
```

This starts up to three isolated services on consecutive ports:

| Contribution | Default behavior |
|---|---|
| GPU | Preferred inference executor on `--base-port`. With `--weights-ram-gb` or `--weights-disk-gb`, this one worker spans all of them and its stage holds layers in each. |
| CPU | Without weight memory on a GPU worker, a standby inference executor on the next port, promoted if preferred executors fail. Add `--cpu-active` to let the planner place it on normal routes. |
| KV cache | `--ram-budget-gb` runs a bounded LRU KV-checkpoint service. This is separate from `--weights-ram-gb`: it protects sessions and does not hold weights. |

If the project is on a network path or an RDP redirected drive, the node copies the `shardmesh` package to local disk and runs workers from there. Importing from such a path can add over 500 ms to every HTTP request. Turn this off with `--no-local-copy`. A worker started directly with `python -m shardmesh.worker` skips the copy, so start it from a local checkout.

Any subset is valid. Use `--gpu-device none` for CPU/RAM-only nodes. `--gpu-device auto` uses every accelerator on the host, each as its own pool. A node that contributes nothing is rejected.

The checkpoint budget is a hard process-local LRU limit, and the CPU thread count is applied to PyTorch.

When a compute stage fails, the orchestrator first tries to restore the committed per-layer KV tensors from the RAM service into the replacement route. It replays the token prefix only when no usable checkpoint exists. The dashboard exposes contribution budgets, cache occupancy, snapshot traffic, restore counts, and the active cache service.

## A real LAN or Tailscale pool

Install the project on every machine. Make the pool's credentials once, and give every machine the same token. A node will not listen on a network without them, and will not send them across a plain LAN unless you pass `--trusted-network`. [ShardMesh security](docs/SECURITY.md) covers certificates for pools that need them.

```powershell
shardmesh pool init --dir pool-secrets
$env:SHARDMESH_POOL_TOKEN = Get-Content pool-secrets\pool.token
```

On the orchestration machine, bind to its LAN or Tailscale address:

```powershell
python -m shardmesh.orchestrator --host 100.80.10.1 --port 8000
```

On each worker, bind to that machine's own reachable address and advertise the same address:

```powershell
python -m shardmesh.worker `
  --host 100.80.10.21 `
  --port 8100 `
  --worker-id office-3060 `
  --orchestrator-url http://100.80.10.1:8000 `
  --advertise-url http://100.80.10.21:8100 `
  --device auto
```

Use a unique `--worker-id` and the correct local address on every node. MagicDNS names can replace the `100.x` addresses.

After the workers appear in `/admin/status`, load and place a model:

```powershell
$headers = @{ Authorization = "Bearer $env:SHARDMESH_POOL_TOKEN" }
Invoke-RestMethod `
  -Method Post `
  -Uri http://100.80.10.1:8000/admin/load `
  -Headers $headers `
  -ContentType application/json `
  -Body '{"model_id":"distilbert/distilgpt2"}'
```

When another worker joins, it remains standby. Activate it at a safe request boundary:

```powershell
Invoke-RestMethod -Method Post -Uri http://100.80.10.1:8000/admin/rebalance -Headers $headers
```

Tailscale encrypts transport and supplies device identity. ShardMesh still owns pool membership, placement, cache state, and recovery. Apply tailnet grants so only pool members can reach the orchestration and worker ports. Tailscale recommends grants for new policy configurations; see its [policy syntax](https://tailscale.com/docs/reference/syntax/policy-file) and [device tags](https://tailscale.com/docs/features/tags).

An alternative is to bind each process to `127.0.0.1` and expose it to the tailnet with [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve). Advertise the resulting HTTPS MagicDNS URL. Do not use Funnel; these endpoints are intended to remain private.

## Mixed hardware and drivers

Workers do not use NCCL, CUDA IPC, or remote device pointers. Each process loads a PyTorch build compatible with its own host, then the wire protocol carries CPU safetensors activations. The receiving stage converts the tensor to its local device and execution dtype. Inside one worker, activations cross between pools the same way, so a layer's attention can run on the GPU while its experts run on the CPU.

| Worker | Prototype support | Notes |
|---|---:|---|
| NVIDIA CUDA | Yes | CUDA toolkit and driver versions may differ by node. |
| AMD ROCm | Yes | Install a ROCm-compatible PyTorch build on that node; PyTorch still exposes the device as `cuda`. Tested on Windows with an RX 7900 XT in BF16 and with GGUF. |
| CPU | Yes | Slow, but useful as a stage or for testing. |
| Apple MPS | Experimental | Reported as unified memory. Not covered by the current Windows test suite. |
| Intel XPU | Experimental | Discovered as VRAM. Not covered by the current test suite. |

Cross-backend numerical differences can change sampling when logits are nearly tied. Greedy inference should normally agree, but bit-for-bit identity is not promised between different kernel implementations.

## Execution and recovery

```text
client -> orchestrator -> entry worker -> compute worker(s) -> exit worker
             control       embeddings     transformer blocks   norm/head
                              |                  |                  |
                              +---------- local KV caches ----------+
                                                 |
                                      incremental checkpoints
                                                 |
                                          RAM contributor
```

Whichever node a client asks runs the reply itself, straight to the entry worker, and the entry worker runs the token loop. The coordinator only plans, so it is on no reply's path. Each request carries a pipeline epoch. If a worker disappears:

1. The upstream stage reports the failed worker.
2. The orchestrator quarantines that worker and computes a new contiguous placement.
3. Surviving stages fetch any layers they are missing, load their new assignments, and atomically advance the epoch.
4. The new stages restore committed KV tensors from the RAM contributor when available.
5. If restore is unavailable or incomplete, the orchestrator replays the committed token prefix. Hybrid Qwen models always take this path, because their recurrent state changes whole on every token and cannot be checkpointed incrementally.
6. Generation resumes. Stale responses from the old epoch are rejected.

The planner estimates token time from each pool's bandwidth and the bytes a token reads there. It never inserts more workers into a latency-critical route than there are layers.

Recovery has to download whatever the survivors lack, so with a large model a failure can pause generation for as long as that takes.

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `GET` | `/chat` | Chat page. |
| `GET` | `/dashboard` | Mesh telemetry and controls. |
| `GET` | `/admin/status` | Workers, memory pools, route, epoch, and load progress. |
| `POST` | `/admin/plan` | Show where a model would run, without downloading it. |
| `POST` | `/admin/load` | Load a model. `wait: false` returns at once; progress is in `/admin/status`. |
| `POST` | `/admin/load/cancel` | Stop the running load. Finished units stay cached. |
| `GET` | `/admin/catalog` | Suggested models and supported architectures. |
| `POST` | `/admin/rebalance` | Reconsider all live workers and activate a new epoch. |
| `POST` | `/admin/nodes/{name}/cordon` | Move a node's share elsewhere for maintenance. `uncordon` puts it back. |
| `POST` | `/admin/nodes/{name}/revoke` | Throw a node out for good. |
| `GET`, `POST` | `/admin/keys` | List or create API keys. `DELETE /admin/keys/{name}` removes one. |
| `GET` | `/ready`, `/metrics` | Readiness for load balancers, and Prometheus metrics. |
| `GET` | `/v1/models` | OpenAI-style model listing. |
| `POST` | `/v1/completions` | Text completion, streaming with `stream: true`. |
| `POST` | `/v1/chat/completions` | Chat completion, streaming with `stream: true`. `enable_thinking` switches reasoning on models that support it. |

Send `Authorization: Bearer <key>` to every endpoint except `/health`, `/ready`, and the pages. `/health` tells strangers only that the node is up. Requests are limited to 16 MiB.

## Verification

```powershell
python -m pip install "transformers>=4.42" pytest
python -m pytest -q
```

Adapter tests compare every family with the reference `transformers` implementation, and skip families newer than the installed release. Run them under a current `transformers` as well to cover all of them.

To compare any running local checkpoint with an unsharded reference using the same execution dtype:

```powershell
python scripts\verify_model.py --model D:\project\models\distilgpt2 --dtype float32
```

## Current limits

- Conversation reuse covers GGUF chat only. Other models reprocess the whole conversation each turn.
- Prompts do not overlap across stages: one stage finishes before the next begins, so a long first message costs the sum of them.
- Hybrid Qwen models on the PyTorch path process prompts token by token. This is exact, but slower than a chunked kernel on long prompts.
- Long-context rope scaling (YaRN) and sliding-window attention are refused rather than run incorrectly.
- Expert placement is per layer, not per expert: frequently used experts are not promoted to faster memory.
- The planner does not reserve VRAM for a draft model. It loads into what is left on the entry node, and if it does not fit the model runs without it.
- Multi-token batches cross between GGUF stages as BF16, and single tokens as FP32. A node set to `--dtype float32` keeps everything in FP32.
- A stage file duplicates that node's cached units on disk.
- GGUF files are parsed by native code in the stage process, so load them only from sources you trust.
- The planner uses measured memory bandwidth and a fixed cost per network hop, but not per-layer benchmarks or pairwise network telemetry. Its disk estimate assumes cold reads, so warm runs beat it.
- Workers on one host split shared RAM evenly during planning, which can refuse a plan that an uneven split would allow.
- There is one active RAM checkpoint service per route. Snapshots are sent in the background and are not yet replicated or compressed. If the store is behind when a stage fails, recovery replays the prompt instead.
- With two nodes a split network leaves two coordinators until it heals. Nothing can be corrupted, but neither side can know it is the minority.
- A reply is lost if the node its client is connected to dies. The chat page restarts it through another node, and API clients should do the same.
- Concurrent replies take turns on a stage by default, so total speed stops growing once the slowest stage is always busy. `SHARDMESH_MERGE_PASSES=1` lets single-token steps share a pass. Its answers matched in testing, and one model on one GPU went from 34 to 93 tok/s at four conversations, but a two-node pool measured slower with it and the cause is not known, so it is off. Past `--parallel` they queue at the entry worker, and past 32 waiting they are refused.
- Multi-turn reuse keeps one conversation per sequence, so more conversations than sequences push each other out and reread their history.
- Mesh membership is meant for tens of nodes: every worker checks in with every control plane.
- Prompt contents are visible to participating compute workers, and every node is fully trusted. See [ShardMesh security](docs/SECURITY.md).

These boundaries are deliberate: the prototype validates exact layer execution, direct activation routing, heterogeneous memory and runtimes, KV ownership, joins, and failure recovery before adding more aggressive scheduling.
