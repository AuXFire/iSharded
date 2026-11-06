# ShardMesh

> ShardMesh is an active project, changing quickly. Commands and the network protocol can change between versions, so upgrade all of a mesh's nodes together. Nothing here should be treated as finished or production-proven yet.

ShardMesh runs language models across a mesh of many computers, using whatever memory each one has: VRAM, unified memory, system RAM, or disk. A model too big for any one of them can run across all of them. There is no central server. Every node runs the same software, the nodes elect one of themselves to coordinate, and any node can serve the API.

It works on a closed network, a LAN, across VLANs, over Tailscale, or between remote machines with TLS. The API follows OpenAI's, so existing clients only need a new URL and key.

This is execution sharding, not simulated work distribution: each token really crosses every machine in the route. It is built to grow to many nodes. So far it has been tested on two machines, and [Scaling to many nodes](#scaling-to-many-nodes) separates what is proven from what comes next. Its security has been reviewed by AI only, not by a person. [Current limits](#current-limits) says what that means for you.

## Contents

- [What it does](#what-it-does)
- [Install](#install)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Scaling to many nodes](#scaling-to-many-nodes)
- [Using a pool](#using-a-pool)
- [Recommended configuration](#recommended-configuration)
- [Command reference](#command-reference)
- [Environment variables](#environment-variables)
- [HTTP API](#http-api)
- [Models](#models)
- [Embeddings](#embeddings)
- [Memory and placement](#memory-and-placement)
- [Precision](#precision)
- [Networks and security](#networks-and-security)
- [Checking answers](#checking-answers)
- [Running a pool](#running-a-pool)
- [Troubleshooting](#troubleshooting)
- [Measured performance](#measured-performance)
- [Development](#development)
- [Current limits](#current-limits)
- [License](#license)

More detail lives in [Operating ShardMesh](docs/OPERATIONS.md) and [ShardMesh security](docs/SECURITY.md).

## What it does

- Starts with one command, and adds a machine with an invite code. Hardware, addresses, and certificates are worked out for you.
- Splits a model's layers across machines, each holding only its share. Weights are never gathered on one machine.
- Spans memory types: one node can hold layers in VRAM, RAM, and disk at once, and nodes do not have to match.
- Uses any GPU with a Vulkan driver for GGUF models, so nodes need no CUDA or ROCm.
- Runs GGUF models on stock llama.cpp, and safetensors models on its own PyTorch code. Qwen and GPT-2 families, including mixture-of-experts models.
- Speeds up replies with a small draft model, reuses earlier conversation state, and lets conversations share a pass through each stage.
- Runs several models at once on the same mesh, and users choose between them.
- Serves OpenAI-shaped `/v1/chat/completions`, `/v1/completions`, and `/v1/models`, with streaming, plus a chat page and a live dashboard on every node.
- Keeps replies going through failures: a crashed stage, a dead node, a dead coordinator, a silent link, or a split network.
- Moves weights only when it must, shares them between nodes, and works with no internet at all.
- Runs as a network one person owns, or as one nobody owns, where each node decides who it accepts and what it hosts.
- Refuses to start exposed. Offers per-node certificates, signed membership, API keys with roles and rate limits, TLS, and an audit log.
- Supports planned maintenance: cordon a node and its share moves elsewhere while replies finish.
- Reports each node's disks, GPU temperature and power, load, and battery, where the machine exposes them.

## Install

You need Python 3.10 or newer. These install ShardMesh for your user only, and pick a PyTorch build for your hardware.

On Windows, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/AuXFire/shardmesh/main/install.ps1 | iex
```

On Linux or macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/AuXFire/shardmesh/main/install.sh | sh
```

Then `shardmesh doctor` checks the machine, and says what to fix if anything is missing.

The installers pick a CUDA build of PyTorch where they find an NVIDIA GPU, and otherwise a CPU build on Windows. A GPU PyTorch cannot see, such as an AMD card on Windows, still works for GGUF models: the node finds it through llama.cpp's Vulkan build. See [Any GPU](#any-gpu).

### By hand

You need PyTorch built for your hardware. Install PyTorch first, because the default package may not match your GPU. See [pytorch.org](https://pytorch.org/get-started/locally/) for CUDA, ROCm, and CPU builds.

```bash
git clone https://github.com/AuXFire/shardmesh.git
```

```bash
cd shardmesh
```

```bash
python -m pip install -e .
```

The `test` extra adds what the test suite needs.

This installs `shardmesh`, `shardmesh-node`, `shardmesh-orchestrator`, and `shardmesh-worker`. From a checkout with the dependencies in place, `python -m shardmesh` works the same way without installing.

For GGUF models, nodes download a pinned, hash-checked llama.cpp build for their hardware to `%LOCALAPPDATA%\ShardMesh\llama` on Windows or `~/.cache/shardmesh/llama` elsewhere. On a minimal Linux install, add the one system library it needs with `sudo apt-get install libgomp1`. On a closed network, give each node a build with `--llama-dir`.

On a machine where antivirus intercepts HTTPS, `pip` may fail with a certificate error. Add `--use-feature=truststore` so it uses the system's trusted certificates.

## Quick start

### Start a mesh

On the first machine:

```bash
shardmesh up
```

That is all it needs. It finds the hardware, picks an address, makes the mesh's certificates, and opens the chat page signed in as the owner, where you choose a model. Running it again starts the same node with the same settings.

### Add a machine

On the machine that made the mesh:

```bash
shardmesh invite
```

It prints one command to run on the new machine, with a code that works once and lasts 24 hours. The dashboard's "Add a machine" button does the same. The new machine makes its own key, proves it holds the code, and gets a certificate signed by the owner's machine, which has to be running at that moment. Nothing needs copying. Run `shardmesh up --invite CODE` on as many machines as you invite.

The new machine gets its own key for the chat page, with the user role. The owner's key stays the only admin key.

Or skip the code. A new machine running `shardmesh up` looks for a mesh on its LAN and its tailnet, and offers to join what it finds. The machine then waits for someone to approve it, on the dashboard or with `shardmesh approve NAME`, and both sides show the same key fingerprint to check. If a firewall blocks the search, which uses UDP port 8208 on a LAN, use an invite.

### What it works out

| Setting | How `shardmesh up` decides |
|---|---|
| Address | Tailscale if this machine has it, otherwise the LAN |
| Encryption | None added on Tailscale, which encrypts already. Otherwise TLS with the mesh's own certificate, so a browser warns once. `--trusted-network` skips it |
| GPU, RAM, CPU | Every accelerator, a share of free RAM, and half the cores |
| Models on disk | LM Studio's folders and `~/models`, so nothing downloads twice |
| Name | This machine's name |

Everything it decides is in `settings.json` in its folder, `%LOCALAPPDATA%\ShardMesh\up` on Windows or `~/.shardmesh/up` elsewhere. That folder also holds the mesh's authority key on the owner's machine, so keep it safe. To start a node at login, run `shardmesh service install`.

The rest of this section sets things up by hand, which gives you every option in the [command reference](#command-reference).

### One machine

```bash
python -m shardmesh demo --workers 2
```

This starts a control plane on port 8000 and two workers on ports 8100 and 8101, then loads `distilbert/distilgpt2` across them. Open [http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat) to chat, or [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) to watch it work.

With no network at all, build a tiny test model first:

```bash
python scripts/build_tiny_model.py .smoke-model
```

```bash
python -m shardmesh demo --workers 2 --model .smoke-model --device cpu
```

That model is random, so its output means nothing. It proves that sharded and unsharded runs agree.

### Several machines on Tailscale

Tailscale encrypts the link, so a shared token is enough here. On the first machine, make the pool's credentials:

```bash
shardmesh pool init --dir pool-secrets
```

Copy `pool-secrets/pool.token` to `%LOCALAPPDATA%\ShardMesh\pool.token` on each Windows machine, or `~/.shardmesh/pool.token` elsewhere. Anyone holding it has full control of the pool. Then start the first node:

```bash
shardmesh node --node-id node-a --gguf-dir D:\models --weights-ram-gb auto
```

It prints the address it advertises, and its control plane listens on port 8209. Start each further node the same way, naming any node already running:

```bash
shardmesh node --node-id node-b --join http://100.64.0.1:8209 --weights-ram-gb auto
```

Load a model from either machine. The `--url` matters, because `load` otherwise assumes the demo's port:

```bash
shardmesh load "unsloth/Qwen3.6-27B-GGUF:Qwen3.6-27B-Q4_K_M.gguf" --url http://100.64.0.1:8209
```

Open `http://100.64.0.1:8209/chat` and enter the token. A file already in a `--gguf-dir` folder is read locally, and a node without it reads only its share from the node that has it.

### A pool with certificates

Certificates give each node its own identity, so one can be thrown out without touching the others. On your own machine, which keeps the authority key:

```bash
shardmesh pool init --dir pool-secrets
```

```bash
shardmesh key create root --role admin --file access.json
```

```bash
shardmesh pool add-node --dir pool-secrets node-a --address 100.64.0.1
```

The second command prints an admin key, once. Copy `ca.pem`, `node-a.pem`, `node-a.key`, and `access.json` to node-a, and keep `ca.key` where it is. Start the node:

```bash
shardmesh node --node-id node-a --ca-file ca.pem --identity-cert node-a.pem --identity-key node-a.key --keys-file access.json
```

Other nodes ask to join, which keeps their private keys on their own machines:

```bash
shardmesh pool request-join node-b --dir node-identity --address 100.64.0.2 --url http://100.64.0.1:8209 --wait
```

```bash
shardmesh pool requests --url http://100.64.0.1:8209 --token YOUR_ADMIN_KEY
```

```bash
shardmesh pool approve node-b --dir pool-secrets --url http://100.64.0.1:8209 --token YOUR_ADMIN_KEY
```

Check the fingerprint with whoever is asking before you approve. Once approved, the waiting command saves the certificate. Start that node with its own files and `--join http://100.64.0.1:8209`.

With certificates, the pool token stops working. Use your admin key, in the chat page and with `--token` on commands.

## How it works

The diagrams use the same line colors throughout: solid blue-green for a reply's data, dashed violet for control messages, and pink for anything signed by the owner.

These diagrams show what is built. Measured figures come from the test pool, which is two machines so far.

### Two networks in one

Every node runs the same software. Control messages spread between all nodes, while a reply's data runs in a line through the stages.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/structure-dark.png">
  <img alt="Three nodes. Control planes gossip with each other, while a reply's data runs in a line through the workers." src="docs/diagrams/structure-light.png">
</picture>

Layers are split by what each machine can hold, so one model can span a GPU on one node and RAM and disk on another. On the test pool, a laptop RTX 3060 holds layers 0 to 7 and a desktop RX 7900 XT holds layers 8 to 63, over Tailscale. Workers check in with every control plane, so any of them can take over.

### One reply, from message to text

The node you ask runs your reply itself. The coordinator isn't involved, so losing it doesn't interrupt anything.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/reply-dark.png">
  <img alt="A reply goes from any control plane to the entry worker, then loops through the stages. The coordinator is not on this path." src="docs/diagrams/reply-light.png">
</picture>

Only hidden states and tokens cross the network during a reply, never weights. A small draft model guesses ahead so one pass can produce several tokens. Conversations sharing a stage also share a pass when they can, which took four conversations from 12 to 32 tok/s on one stage.

### Any node can coordinate

Nodes agree on a coordinator without voting: they all apply the same rule to the same member list.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/coordinating-dark.png">
  <img alt="Every node runs the same code. Coordinating is a role that moves between them." src="docs/diagrams/coordinating-light.png">
</picture>

The term only goes up, and the higher term wins, so two coordinators from a split settle themselves when they reconnect. Workers obey only the highest term they have seen.

### Loading a model

Replies keep running through the old layout until the new one is ready, and only then do they switch.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/loading-dark.png">
  <img alt="A model loads in the background while the old layout keeps serving, and weights move only when they must." src="docs/diagrams/loading-light.png">
</picture>

A closed network works the same way: a GGUF file on one node is enough, and the others read the parts they need from it. The 2 GiB limit applies to a node that is alive but failed to load. If a node has died, the pool replans whatever that moves.

### What happens when something fails

The rule is that a failure must not drop a reply, and weights move only when they must.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/failures-dark.png">
  <img alt="Seven kinds of failure keep the reply going. Only losing the node you are connected to restarts it." src="docs/diagrams/failures-light.png">
</picture>

Times are measured on the test pool, except the stage crash, measured on a small model. In the last soak test, 405 replies finished and none were lost while processes were killed on both machines for 21 minutes. [Operating ShardMesh](docs/OPERATIONS.md) has the details.

### Joining an owned network

A node makes its own key and asks to join through any member. The owner approves it from their own machine.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/joining-dark.png">
  <img alt="A node asks through any member, the owner signs the answer offline, and neither private key ever crosses the network." src="docs/diagrams/joining-light.png">
</picture>

Asking grants nothing. The new node's private key never leaves it, and the owner's key never touches the network, so a compromised node can't add a key or revoke a peer. Tickets prove a node's name without TLS, so antivirus that intercepts TLS can't break them. The link still needs Tailscale or another private network, because tickets don't encrypt anything. [ShardMesh security](docs/SECURITY.md) has the commands.

### Owned or ownerless

An owned network runs without a central server, but one person decides who belongs. Removing the owner removes the signing arrows, and everything they did is done another way. Both kinds work for private or public networks.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/ownership-dark.png">
  <img alt="The difference between the two kinds of network is the owner and its signing arrows, which disappear in the ownerless version." src="docs/diagrams/ownership-light.png">
</picture>

| Question | Owned | Ownerless |
|---|---|---|
| Who lets a node in? | The owner approves it | Each node decides: anyone, or nodes vouched for by those it already accepts |
| Who removes a node? | The owner, for everyone | Nobody can. Each node blocks whoever it chooses, for itself |
| Who picks the models? | The owner, or anyone with an admin key | Anyone can propose one beside what runs. Each node chooses what it hosts, and users pick from what's running |
| What is still unsolved? | A member can read prompts and return wrong answers | The same, plus one person could run thousands of nodes |

See [A network nobody owns](#a-network-nobody-owns) to run one. To redraw these diagrams, run `python scripts/build_diagrams.py`.

## Scaling to many nodes

ShardMesh is meant to run on many machines, not two. Two is only how many it has been tested on so far. Every node runs the same software, any node can coordinate, and a new node joins by naming any node already running, so nothing in how a mesh forms stops at two.

What more nodes give you:

| More nodes give you | How | Status |
|---|---|---|
| Bigger models | Each node holds only its share, so the limit is the mesh's total memory | Built |
| More models at once | Each model runs in its own [lane](#several-models-at-once) | Built |
| Survival | Any node can coordinate, serve the API, or take over a lost stage | Built, tested on two machines |
| More replies at once | Several routes for one model, across separate groups of nodes | Next |
| A faster single reply | It does not: every token crosses every stage, so a longer route is a slower one | Keep routes short with `--placement compact` |

So a large mesh should run many short routes side by side, not one long one. That is the direction, and these are the parts that stop it getting there today:

| What stops it | Today | Where it needs to go |
|---|---|---|
| Gossip | Every node contacts every other every 3 s, so messages grow with the square of the node count | Contact a few peers per round, with liveness spreading through them |
| Worker check-ins | Each worker checks in with every control plane every 5 s | Check in with a few, and let gossip carry the rest |
| Routes | One route per lane, so one model's capacity is one route's | Several routes per model, with requests spread across them |
| Placement | A fixed cost per hop | Use measured latency between nodes, so a route stays among close neighbours. Each node already measures this |
| Splits | Each side of a split elects a coordinator | With three or more nodes, let only the majority side coordinate |
| Memory | A lane's share is fixed when it starts | Share memory between lanes as load changes |

None of this has been measured past two machines, so the first milestone is a real mesh of more. If you have spare machines, running a node is the most useful test you can give it.

## Using a pool

### Chat page and dashboard

Every node serves a chat page at `/chat` and a console at `/dashboard`. These screenshots show the real pages with simulated data, because no mesh this size has run yet. Each page shows what it does today, and none of these figures are measurements.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/dashboard-dark.png">
  <img alt="The dashboard: the model's layers across machines, pool metrics, the route's stages, and what needs attention." src="docs/screenshots/dashboard-light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/chat-dark.png">
  <img alt="The chat page: a reply with its speed, and the stages it ran on." src="docs/screenshots/chat-light.png">
</picture>

The console is built to stay readable at 60 workers, grouping them by state so what needs you comes first:

![The dashboard with 60 simulated workers, marked as simulated at the top of the page.](docs/screenshots/dashboard-60-workers.png)


Every node serves both, so any node's address works.

| Page | What it shows |
|---|---|
| `/chat` | A chat window. It asks for a key, shows where a model would run and how fast before anything downloads, and colors each layer by the memory holding it. |
| `/dashboard` | The active pipeline, each node's state and hardware, latency, throughput, memory, recoveries, and events. Click a stage or worker for detail. |

If the node serving your chat dies mid-reply, the page asks before sending your key and conversation to another member, then restarts the reply there.

### Calling the API

Any node's control address works. With curl:

```bash
curl http://100.64.0.1:8209/v1/chat/completions -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":64}"
```

With the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://100.64.0.1:8209/v1", api_key="YOUR_KEY")
model = client.models.list().data[0].id
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hello."}],
    max_tokens=64,
    stream=True,
)
for part in stream:
    print(part.choices[0].delta.content or "", end="", flush=True)
```

If the connection to a node drops, retry against another node. Replies survive almost every other failure, but nothing can save a connection to a node that died. `/ready` answers 200 when a node can serve, which suits load balancers.

### Loading a model

`shardmesh load` starts a load, prints progress, and returns when the model is ready. The old model keeps serving until then.

| A model id can be | Example | Comes from |
|---|---|---|
| A Hugging Face repo | `Qwen/Qwen3-8B` | Each worker fetches only its share |
| A GGUF file in a repo | `owner/repo:file.gguf` | Each worker fetches only its share |
| A local GGUF path | `D:\models\file.gguf` | That machine, then peers read from it |
| A bare GGUF file name | `file.gguf` | Whichever node holds it, with no internet needed |
| A local safetensors folder | `D:\models\qwen3-8b` | That machine. In a secured pool it must be in `--model-dir` |

Loading a model replaces the main one. To run another beside it, see [Several models at once](#several-models-at-once). To see where a model would go without moving anything, use `POST /admin/plan`, or the picker on the chat page.

Gated models need `HF_TOKEN` set on each worker.

### Several models at once

Load another model beside the main one by giving it a lane:

```bash
shardmesh load Qwen3-0.6B-Q8_0.gguf --lane small --url http://node-a:8669
```

Each node then starts a control plane and workers for that lane, on ports 1000 above its own for the first lane, 2000 for the second, and so on. Open those ports wherever a firewall sits between nodes. The lane gets only what each node has spare under the limits its owner set, and `--lane-gpu-gb` and `--lane-ram-gb` lower that. A node started with `--no-lanes` hosts only the main model.

`GET /v1/models` lists every running model on every node, and a request goes to the model it names. The chat page shows a picker once two models run. A request that names no model goes to the main one. Keys stay with each node's main control plane, which checks the key and the rate limit before passing a request on.

Lanes are separate on purpose: a failure in one stays there, and each has its own coordinator. See or stop them with:

```bash
shardmesh lane list --url http://node-a:8669
```

```bash
shardmesh lane remove small --url http://node-a:8669
```

Removing a lane stops its processes on every node and frees their memory. Lanes come back after a restart, like the main model.

### A network nobody owns

Each node makes its own identity. Its name comes from its key, so nobody hands names out and nobody can take one:

```bash
shardmesh net init --dir ./identity --joining vouched --trust n-1a2b3c4d5e6f70819203
```

`--joining open` accepts anyone. `--joining vouched` accepts only nodes vouched for by nodes this one already accepts, starting from itself and whoever `--trust` names, and `--quorum` sets how many vouches it takes. Each node works this out for itself, so two nodes can disagree and neither can overrule the other.

Give your node's name to someone already in the network, and have them vouch for it:

```bash
shardmesh net vouch n-9f8e7d6c5b4a39281706 --dir ./identity --url http://127.0.0.1:8009
```

Start the node with `--ownerless ./identity`. Make yourself a key for your own node with `shardmesh key create me --role admin --file keys.json`, and pass `--keys-file keys.json`. That key is an admin of your node and means nothing anywhere else. Your users' keys live on your node too, and you set their limits.

What an operator can do on the rest of the network is propose:

| Action | Your node | The rest of the network |
|---|---|---|
| Load the first model | Yes | Yes, anyone can |
| Replace what is running | | Nobody can |
| Load a model beside it | Yes, with `--lane` | Yes, and each node hosts it only if its owner allows |
| Remove a lane | Yes, if your node proposed it | Nobody else can |
| Take a node out | Your own | Nobody can take out another's |
| Block a node | `shardmesh net block NAME`, for your node only | Nobody can block for anyone else |

`--host-models` lists the models your node will host, as patterns such as `*Qwen3*`, in any kind of network. A worker refuses anything else, whatever a coordinator tells it.

Nodes still pick a coordinator by lowest name, to keep a route in order. It can place work and hold a route up, but it cannot make a node host anything or change who that node accepts. Read [Current limits](#current-limits) before opening one to strangers.

## Recommended configuration

### Start from your situation

| Situation | Network | Identity | Load with |
|---|---|---|---|
| You and a friend, machines anywhere | Tailscale | Shared token | Defaults |
| A few people sharing a pool | Tailscale or wired LAN | Certificates with tickets | `--parallel 4 --no-draft` |
| Closed network, no internet | Wired LAN with `--trusted-network`, or TLS | Certificates | Bare file names, `--hashes` |
| Remote machines across the internet | Tailscale, or TLS | Certificates | Defaults |
| Machines running antivirus that scans HTTPS | Tailscale | Certificates with tickets | Defaults |
| A model bigger than any one machine | Fastest link you have | Any | `--weights-ram-gb auto` on every node |

### Node settings

| Setting | Recommendation | Why |
|---|---|---|
| `--node-id` | A stable name. Give your most reliable machine the lowest | The lowest name alive coordinates |
| `--gpu-memory-gb` | Card size minus about 1.5 GB on small cards, or 20% on large ones. 4.5 on a 6 GB card, 16 on a 20 GB card | It budgets weights only. KV cache, buffers, and a GPU context of about 0.7 GiB come on top |
| `--weights-ram-gb` | `auto`, or a fixed size on a machine you also use | Lets a node hold layers beyond its VRAM |
| `--weights-disk-gb` | 0 unless a model truly cannot fit otherwise | Disk is far slower than RAM |
| `--cache-dir` | A fast SSD with room for twice the node's share | Units and stage files both live there |
| `--cache-budget-gb` | Set it on a small disk | Evicts old models past this size |
| `--ram-budget-gb` | 0 for GGUF hybrid models, a few GB otherwise | KV checkpoints speed up recovery where a model supports them. Hybrid Qwen models replay instead |
| `--log-file` | Always | Nothing is kept otherwise |
| `--gguf-dir` | Point it at LM Studio's or your own model folder | Files there are read locally, not downloaded |

The entry node also carries the draft model, about 1.7 GiB of VRAM for the default one, and the planner does not reserve room for it. If you want it on a particular machine, pin that machine with `--entry`.

### Load settings

| Setting | Recommendation | Why |
|---|---|---|
| `--context-tokens` | The smallest that fits your use. 8192 is a good chat default | Each reply's context costs memory on every stage |
| `--parallel` | Your expected concurrent conversations, up to about 4 | Each one costs its own context memory |
| Draft model | Keep it for one or two users. Use `--no-draft` for three or more | See the table below |
| `--placement` | Leave at `spread` | `compact` may skip workers, which only helps when a model fits on fewer of them |
| `--hashes` | Use it when any node or mirror is less than fully trusted | Every node then checks every unit |

Measured on the test pool with Qwen3.6-27B Q4_K_M, total tokens per second across all conversations, at `--parallel 4`:

| Conversations at once | No draft, merged passes | With a draft |
|---:|---:|---:|
| 1 | 14 | 10 |
| 2 | 27 | 15 |
| 3 | 32 | 20 |
| 4 | 37 | 21 |

A draft does far better at `--parallel 1`, where one conversation reached 19 to 33 tokens per second, because every stage keeps more memory and the draft's guesses are not competing.

### Network

Use wired links if you can: every token crosses every stage, so a pass costs at least one round trip per stage. Both test machines were on Wi-Fi with a 15 ms ping, and the desktop's 144 Mbps link limited model transfers to about 8 MiB/s. Fewer stages are faster, so do not add a slow machine unless the model needs its memory.

### A worked example

The test pool: a laptop with a 6 GB RTX 3060 and a desktop with a 20 GB RX 7900 XT, over Tailscale, running Qwen3.6-27B Q4_K_M. The planner gave the laptop layers 0 to 7 and the desktop layers 8 to 63.

```bash
shardmesh node --node-id laptop --join http://100.64.0.2:8300 --control-port 8000 --gpu-device cuda --gpu-memory-gb 4.5 --cpu-threads 4 --weights-ram-gb 6 --cache-dir D:\shardmesh-cache --ram-budget-gb 0.5 --gguf-dir D:\models --log-file D:\shardmesh-cache\node.log --ca-file ca.pem --identity-cert laptop.pem --identity-key laptop.key
```

```bash
shardmesh node --node-id desktop --join http://100.64.0.1:8000 --control-port 8300 --base-port 8311 --gpu-memory-gb 16 --cpu-threads 8 --weights-ram-gb auto --gguf-dir D:\models --ca-file ca.pem --identity-cert desktop.pem --identity-key desktop.key
```

```bash
shardmesh load lmstudio-community/Qwen3.6-27B-GGUF:Qwen3.6-27B-Q4_K_M.gguf --url http://100.64.0.2:8300 --context-tokens 8192 --parallel 2 --token YOUR_ADMIN_KEY
```

A third node and every one after it joins the same way, with `--join` naming any node already running.

## Command reference

Every `--url` takes any node's control address. Commands default to `http://127.0.0.1:8000`, which is the demo's port. A node's control port is `--base-port` plus 9, so 8209 unless you change it.

### shardmesh up

Starts a node with everything worked out, and is the easy way in. See [Quick start](#quick-start).

| Option | Default | What it does |
|---|---|---|
| `--invite` | None | Join a mesh, with a code from `shardmesh invite` |
| `--name` | This machine's name | This node's name |
| `--port` | 8209 | Port for the chat page and API |
| `--address` | Found for you | The address other nodes reach this one on |
| `--tls` | Off on Tailscale | Use TLS even on Tailscale |
| `--trusted-network` | Off | Skip TLS, because nobody else can read this network |
| `--no-gpu` | Off | Contribute CPU and RAM only |
| `--gpu-gb` | A share of what is free | VRAM to give |
| `--ram-gb` | A share of what is free | RAM to give for weights |
| `--cpu-threads` | Half the cores | Threads to give |
| `--gguf-dir` | LM Studio's folders | A folder of GGUF files. Repeatable |
| `--model` | None | Load this model once the node is up |
| `--no-browser` | Off | Do not open the chat page |
| `--find` | Off | Ask to join the one mesh on this network, without asking you first |
| `--new-mesh` | Off | Start a new mesh even if one is on this network |
| `--no-discovery` | Off | Do not look for a mesh, and do not answer machines looking for this one |
| `--ownerless` | Off | Make or join a [network nobody owns](#a-network-nobody-owns) |
| `--joining` | `vouched` | With `--ownerless`: who this node accepts, `open` or `vouched` |
| `--join` | None | With `--ownerless`: any node already in the network |

### shardmesh invite

Makes a code that adds a machine to this mesh, on the machine that made it. `--uses` sets how many machines it can add, 1 by default, and `--hours` how long it lasts, 24 by default.

### shardmesh approve and deny

`shardmesh approve NAME` lets in a machine that asked to join, and `shardmesh deny NAME` turns it down. Run them on the machine that made the mesh. The dashboard does the same.

### shardmesh doctor

Checks this machine and says what to fix: Python, addresses, ports, memory, disk, and local models.

### shardmesh service

`shardmesh service install` starts this node when you log in, for this user only, and `shardmesh service remove` stops that. It uses a scheduled task on Windows, a systemd user unit on Linux, and a launch agent on macOS.

### shardmesh demo

Starts a control plane and workers on this machine, and loads a model.

| Option | Default | What it does |
|---|---|---|
| `--workers` | 2 | How many workers to start |
| `--model` | `distilbert/distilgpt2` | Model to load once they are up |
| `--device` | `auto` | `auto`, `cpu`, `cuda:0`, or another device |
| `--port` | 8000 | Control plane port. Workers use the next hundred up |
| `--pool-token` | `SHARDMESH_POOL_TOKEN` | Require this token |

### shardmesh node

Runs one machine's share of a pool: a control plane, plus workers for what it contributes, restarted if they stop.

Joining and naming:

| Option | Default | What it does |
|---|---|---|
| `--node-id` | Computer name | This node's name. Keep it stable, because the lowest name coordinates |
| `--join` | None | Any existing nodes, comma separated. Omit on the first node |
| `--orchestrator-url` | None | Old name for `--join` |
| `--control-port` | `--base-port` + 9 | Port for this node's control plane |
| `--no-control` | Off | Worker only: run no control plane here. Needs `--join` |

Network:

| Option | Default | What it does |
|---|---|---|
| `--bind-host` | `0.0.0.0` | Address to listen on |
| `--advertise-host` | `auto` | Address other nodes reach this one on. `auto` prefers Tailscale, then the LAN |
| `--advertise-scheme` | `http` | `http` or `https`. Set to `https` for you by `--tls-cert` |
| `--base-port` | 8200 | First worker's port. Later workers take the next ones |

Security:

| Option | Default | What it does |
|---|---|---|
| `--pool-token-file` | `pool.token` in the ShardMesh folder | File holding the pool token |
| `--ca-file` | None | Authority that signed the nodes' certificates. Turns on certificate identity |
| `--identity-cert`, `--identity-key` | None | This node's certificate and key, to prove its name with tickets, without TLS |
| `--tls-cert`, `--tls-key` | None | Certificate and key for this node's servers, for TLS |
| `--keys-file` | `access.json` in the ShardMesh folder | Where this node keeps the pool's API keys |
| `--open` | Off | Serve a network with no credentials, trusting everyone on it |
| `--trusted-network` | Off | Send credentials unencrypted, because nobody else can read this network |

Hardware and memory:

| Option | Default | What it does |
|---|---|---|
| `--gpu-device` | `auto` | `auto` (every accelerator), `cuda:0`, `cuda:0,cuda:1`, `mps`, `xpu:0`, or `none`. ROCm uses PyTorch's cuda device names |
| `--gpu-memory-gb` | 0 (auto) | VRAM budget per accelerator. A hard limit on CUDA and ROCm |
| `--dtype`, `--gpu-dtype` | `auto` | `auto` (BF16), `bfloat16`, `float32`, or `float16`, for this node's VRAM and RAM |
| `--cpu-threads` | 0 | Threads for CPU work. With no GPU, setting this adds a CPU worker |
| `--cpu-active` | Off | Place a CPU worker on normal routes, instead of keeping it on standby |
| `--weights-ram-gb` | None | System RAM for model weights, a size or `auto`. With a GPU, one worker spans both |
| `--weights-disk-gb` | 0 | Disk for memory-mapped model weights |
| `--ram-budget-gb` | 0 | RAM for KV-cache checkpoints, not weights. Adds a memory worker |

Models and cache:

| Option | Default | What it does |
|---|---|---|
| `--gguf-dir` | None | Folder to search for GGUF files already here, such as LM Studio's. Repeatable |
| `--model-dir` | None | A folder of local models peers may ask this node to load. Repeatable |
| `--llama-dir` | Downloaded | A llama.cpp build for GGUF models. Needed on closed networks |
| `--cache-dir` | `~/.cache/shardmesh/units` | Where fetched weights are kept. Put it on a fast disk |
| `--cache-budget-gb` | 0 (no limit) | Evict old models past this size |
| `--no-peer-fetch` | Off | Always download from the model host |
| `--no-peer-serve` | Off | Do not send shards to other workers |

Other:

| Option | Default | What it does |
|---|---|---|
| `--log-file` | None | Write this node's output here. Over 64 MiB, the last one is kept as `.1` at the next start |
| `--no-local-copy` | Off | Run workers from a network path as it is, instead of copying the code to local disk |
| `--no-lanes` | Off | Host only the main model here, never one loaded with `load --lane` |
| `--host-models` | Any | Comma-separated patterns such as `*Qwen3*`. Host only models that match |
| `--ownerless` | Off | Join a network nobody owns, using the identity folder from `shardmesh net init` |

A node contributes whatever you give it. With a GPU and weight memory, one worker spans them all. With no GPU, `--cpu-threads` or a weight budget adds a CPU worker. `--ram-budget-gb` adds a memory worker on the next port. A node that contributes nothing is refused.

### shardmesh load

Loads a model on a running pool and shows progress. `model` is any [model id](#loading-a-model).

| Option | Default | What it does |
|---|---|---|
| `--url` | `http://127.0.0.1:8000` | Any node's control address |
| `--token`, `--pool-token` | This machine's pool token | An admin key |
| `--context-tokens` | 4096 | Tokens each reply can hold |
| `--parallel` | 1 | Replies to run at once. Each costs its own context memory |
| `--placement` | `spread` | `spread` uses every worker, `compact` the fewest that fit |
| `--entry`, `--exit` | Chosen for speed | Worker to hold the entry or exit stage |
| `--draft` | Chosen automatically | Small GGUF for speculative decoding |
| `--no-draft` | Off | Turn speculative decoding off |
| `--no-merge-passes` | Off | Make conversations take turns on each stage |
| `--hashes` | None | File from `shardmesh model hash`. Every node checks every unit against it |
| `--no-reuse-cached` | Off | Plan as if no worker held any shards |
| `--no-peer-fetch` | Off | Download every shard from the model host |
| `--lane` | None | Load beside the running model, in a lane of this name. See [Several models at once](#several-models-at-once) |
| `--lane-gpu-gb` | What is spare | Most VRAM per node for this lane |
| `--lane-ram-gb` | What is spare | Most RAM per node for this lane |
| `--lane-apart` | Off | Keep this lane off every node another route uses, so routes can check each other |
| `--embedding` | Off | This is an embedding model: one stage, for `/v1/embeddings` and not chat. See [Embeddings](#embeddings) |

### shardmesh lane

`shardmesh lane list` shows the pool's lanes, and `shardmesh lane remove NAME` stops one and frees its memory on every node. Both take `--url` and `--token` (`--pool-token`), as `load` does.

### shardmesh net

For a [network nobody owns](#a-network-nobody-owns).

| Command | What it does |
|---|---|
| `shardmesh net init --dir DIR` | Make this node's identity. `--joining` is `open` or `vouched`, `--quorum` is the vouches a node needs, and `--trust NAME` accepts a node from the start |
| `shardmesh net vouch NAME --dir DIR --url URL` | Say this node accepts another |
| `shardmesh net show --url URL --token KEY` | This node's own view: who it accepts, blocks, and can see |
| `shardmesh net block NAME --url URL --token KEY` | Stop this node dealing with another. Only this node is affected |
| `shardmesh net unblock NAME --url URL --token KEY` | Undo that |

### shardmesh audit

Asks every route serving a model the same questions, and compares the answers. See [Checking answers](#checking-answers).

| Option | Default | What it does |
|---|---|---|
| `--model` | The main model | The model to check |
| `--samples` | 8 | Questions to ask each route |
| `--every` | Off | Keep checking, this many seconds apart |
| `--url`, `--token` (`--pool-token`) | | As for `load` |

It exits with 1 if routes disagree, so it can sit in a script.

### shardmesh cordon and uncordon

`cordon` moves a node's share elsewhere for maintenance, and `uncordon` puts the node back. Both take the node's name.

| Option | Default | What it does |
|---|---|---|
| `--url` | `http://127.0.0.1:8000` | Any node's control address |
| `--token` | This machine's pool token | An admin key |
| `--force` | Off | `cordon` only: go ahead even if this moves over 2 GiB between nodes |

### shardmesh model hash

Hashes a model on a copy you trust, for `load --hashes`. It reads the whole model. `source` is a GGUF file or a safetensors folder.

| Option | Default | What it does |
|---|---|---|
| `--out` | Required | The file to write |

### shardmesh key

`key create NAME` makes an API key and shows it once. `key remove NAME` removes one for good, and needs the authority.

| Option | Applies to | Default | What it does |
|---|---|---|---|
| `--role` | create | `user` | `user` or `admin` |
| `--requests-per-minute` | create | 0 (no limit) | Rate limit for this key |
| `--file` | create | None | Add it to a key file, for nodes to read with `--keys-file` |
| `--dir` | both | None | The pool's secrets folder. Sign the change with its authority and send it to `--url` |
| `--url` | both | `http://127.0.0.1:8000` | Any node's control address, with `--dir` |

### shardmesh pool

Commands for a pool's credentials and members.

| Command | What it does |
|---|---|
| `pool init --dir DIR` | Makes a pool token and authority in a new folder. Keep this folder off the nodes |
| `pool add-node NAME --dir DIR --address ADDRESS` | Issues a node its certificate and key directly |
| `pool request-join NAME --dir DIR --address ADDRESS --url URL` | Asks to join. The node's key stays in `--dir`. Add `--wait` to wait for the answer |
| `pool requests --token KEY --url URL` | Shows who is asking to join |
| `pool approve NAME --dir DIR --token KEY --url URL` | Lets a node in |
| `pool deny NAME --dir DIR --token KEY --url URL` | Turns a node down. That key stays denied |
| `pool revoke NAME --dir DIR --url URL` | Throws a node out for good. Bring a machine back under a new name |

`--address` is one an address peers use for that node, and is repeatable. `--token` is an admin key.

### Running the pieces by hand

`shardmesh node` starts these for you. Start them directly only for unusual layouts, such as a control plane on a machine with no workers.

`python -m shardmesh.orchestrator` runs a control plane:

| Option | Default | What it does |
|---|---|---|
| `--host` | `0.0.0.0` | Address to listen on |
| `--port` | 8000 | Port to listen on |
| `--node-id` | None | This node's name in the mesh |
| `--advertise-url` | None | How other nodes reach this one. Setting it joins the mesh |
| `--join` | None | Any existing nodes, comma separated |
| `--state-file` | `orchestrator-state.json` in the state folder | Where the running route is saved |
| `--keys-file` | `access.json` in the ShardMesh folder | Where this node keeps the pool's API keys |
| `--audit-file` | `audit.jsonl` in the state folder | Where to record who did what |
| `--no-reuse-cached` | Off | Plan as if no worker held any shards |
| `--no-peer-fetch` | Off | Always download from the model host |
| `--open` | Off | Serve a network with no credentials |
| `--trusted-network` | Off | Send credentials unencrypted, on a network nobody else can read |

The state folder is `%LOCALAPPDATA%\ShardMesh` on Windows and `~/.cache/ShardMesh` elsewhere. The ShardMesh folder is `%LOCALAPPDATA%\ShardMesh` on Windows and `~/.shardmesh` elsewhere.

`python -m shardmesh.worker` runs one worker:

| Option | Default | What it does |
|---|---|---|
| `--host` | `0.0.0.0` | Address to listen on |
| `--port` | 8100 | Port to listen on |
| `--worker-id` | Computer name | Unique name. Use `node-gpu`, `node-cpu`, or `node-memory` so certificates match |
| `--orchestrator-url` | `http://127.0.0.1:8000` | One or more control planes, comma separated. The rest are found from them |
| `--advertise-url` | `http://127.0.0.1:PORT` | How other nodes reach this worker |
| `--device` | `auto` | `auto`, `cpu`, `cuda:0`, `mps`, or `xpu:0` |
| `--roles` | `inference` | `inference`, or `kv-cache,checkpoint` for a memory worker |
| `--dtype` | `auto` | `auto` is BF16. `float32` runs this worker at full precision |
| `--cpu-threads` | 0 | CPU threads. 0 uses half the cores for a stage |
| `--gpu-memory-gb` | 0 (auto) | VRAM budget per accelerator |
| `--weights-ram-gb` | None | System RAM for model weights, or `auto` |
| `--weights-disk-gb` | 0 | Disk for memory-mapped model weights |
| `--ram-budget-gb` | 0 | RAM for KV checkpoints, not weights |
| `--cache-dir` | `~/.cache/shardmesh/units` | Where fetched weights live |
| `--cache-budget-gb` | 0 (no limit) | Evict old models past this size |
| `--gguf-dir` | None | Folder holding GGUF files. Repeatable |
| `--llama-dir` | Downloaded | A llama.cpp build for GGUF models |
| `--no-peer-fetch` | Off | Always download from the model host |
| `--no-peer-serve` | Off | Do not send shards to other workers |
| `--standby` | Off | Only use this worker when a model does not fit without it |
| `--open` | Off | Serve a network with no credentials |
| `--trusted-network` | Off | Send credentials unencrypted, on a network nobody else can read |

## Environment variables

A flag takes priority over its variable.

| Variable | Same as, or what it does |
|---|---|
| `SHARDMESH_POOL_TOKEN` | The pool token, ahead of any token file |
| `SHARDMESH_POOL_TOKEN_FILE` | `--pool-token-file` |
| `SHARDMESH_CA_FILE` | `--ca-file` |
| `SHARDMESH_IDENTITY_CERT`, `SHARDMESH_IDENTITY_KEY` | `--identity-cert`, `--identity-key` |
| `SHARDMESH_TLS_CERT`, `SHARDMESH_TLS_KEY` | `--tls-cert`, `--tls-key` |
| `SHARDMESH_KEYS_FILE` | `--keys-file` |
| `SHARDMESH_STATE_FILE` | `--state-file` |
| `SHARDMESH_AUDIT_FILE` | `--audit-file` |
| `SHARDMESH_HOST`, `SHARDMESH_PORT` | `--host`, `--port` |
| `SHARDMESH_NODE_ID`, `SHARDMESH_NODE_URL`, `SHARDMESH_JOIN` | The control plane's `--node-id`, `--advertise-url`, `--join` |
| `SHARDMESH_WORKER_ID`, `SHARDMESH_ORCHESTRATOR_URL`, `SHARDMESH_ADVERTISE_URL` | The worker's `--worker-id`, `--orchestrator-url`, `--advertise-url` |
| `SHARDMESH_DEVICE`, `SHARDMESH_ROLES`, `SHARDMESH_CPU_THREADS` | `--device`, `--roles`, `--cpu-threads` |
| `SHARDMESH_GPU_MEMORY_GB`, `SHARDMESH_RAM_BUDGET_GB` | `--gpu-memory-gb`, `--ram-budget-gb` |
| `SHARDMESH_CACHE_DIR`, `SHARDMESH_LLAMA_DIR` | `--cache-dir`, `--llama-dir` |
| `SHARDMESH_LANE`, `SHARDMESH_LANES_FILE` | Set by `shardmesh node`: which lane a process is, and where a node keeps the lanes it should run |
| `SHARDMESH_OWNERLESS` | `--ownerless` |
| `SHARDMESH_UP_HOME` | Where `shardmesh up` keeps its settings and keys |
| `SHARDMESH_DISCOVERABLE` | Set by `shardmesh up`: answer machines looking for a mesh |
| `SHARDMESH_HOST_MODELS` | `--host-models` |
| `SHARDMESH_MODEL_DIRS` | `--model-dir`, separated by the system's path separator |
| `SHARDMESH_REUSE_CACHED`, `SHARDMESH_PEER_FETCH` | Set to `0` for the control plane's `--no-reuse-cached`, `--no-peer-fetch` |
| `SHARDMESH_HOST_ID` | Which physical machine a worker is on. Defaults to the computer name, so workers sharing a host do not count its RAM twice |
| `SHARDMESH_FETCH_CONCURRENCY` | Units a worker fetches at once. Default 4 |
| `SHARDMESH_ACCESS_LOG` | Set to `1` to log every request while debugging. Off by default, because nodes poll each other constantly |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Hugging Face token, for gated models. Needed on each worker |
| `HF_ENDPOINT` | Another Hugging Face address, such as a mirror |

A worker's launcher also passes some flags through `SHARDMESH_WEIGHTS_RAM_GB`, `SHARDMESH_WEIGHTS_DISK_GB`, `SHARDMESH_CACHE_BUDGET_GB`, `SHARDMESH_PEER_SERVE`, `SHARDMESH_GGUF_DIRS`, `SHARDMESH_DTYPE`, and `SHARDMESH_STANDBY`. Use the flags for those, because they overwrite the variables. `SHARDMESH_STAGE_KEY` is internal.

## HTTP API

Send `Authorization: Bearer KEY`, except where it says none. Requests are limited to 16 MiB. Each node serves all of this, and passes `/admin` requests to the coordinator for you.

### Endpoints

| Method | Path | Needs | Purpose |
|---|---|---|---|
| `POST` | `/v1/chat/completions` | User key | Chat, streaming with `stream: true` |
| `POST` | `/v1/completions` | User key | Text completion, one prompt per request |
| `POST` | `/v1/embeddings` | User key | Vectors for text, from an embedding model. See [Embeddings](#embeddings) |
| `GET` | `/v1/models` | User key | Every running model, and what each is for |
| `GET` | `/admin/status` | User key | Workers, memory pools, route, epoch, mesh, and load progress |
| `GET` | `/admin/catalog` | User key | Suggested models and supported architectures |
| `POST` | `/admin/plan` | Admin key | Show where a model would run, without moving anything |
| `POST` | `/admin/load` | Admin key | Load a model |
| `POST` | `/admin/load/cancel` | Admin key | Stop a load. Finished units stay cached |
| `DELETE` | `/admin/lanes/{name}` | Admin key | Stop a lane and free its memory on every node |
| `POST` | `/admin/join/{name}/{answer}` | Admin key | Answer a machine asking to join, with `approve` or `deny`, on the machine that made the mesh |
| `POST` | `/admin/audit` | Admin key | Ask routes serving `model` the same `samples` questions, and compare. Each node runs this itself |
| `POST` | `/admin/invites` | Admin key | Make a code that adds a machine. Takes `uses` and `hours`, and only the machine that made the mesh can |
| `GET` | `/join/authority` | None | The mesh's public certificate, for a new machine to check against its invite |
| `GET` | `/admin/network` | User key | In a network nobody owns, this node's own view. `/admin/status` shows the coordinator's |
| `GET` | `/admin/inventory` | User key | Each node's system, disks, GPU temperature and power, load, battery, and round trip from this node, where the machine reports them |
| `DELETE` | `/admin/nodes/{name}/revoke` | Admin key | In a network nobody owns, undo this node's block on another |
| `POST` | `/admin/rebalance` | Admin key | Reconsider all live workers and apply a new layout |
| `POST` | `/admin/nodes/{name}/cordon` | Admin key | Move a node's share elsewhere. `?force=true` allows a large transfer |
| `POST` | `/admin/nodes/{name}/uncordon` | Admin key | Put a node back in service |
| `POST` | `/admin/nodes/{name}/revoke` | Admin key | Throw a node out, in a pool without an authority |
| `GET`, `POST` | `/admin/keys` | Admin key | List or create keys, in a pool without an authority |
| `DELETE` | `/admin/keys/{name}` | Admin key | Remove a key, in a pool without an authority |
| `POST` | `/access/entries` | A valid signature | Apply a change signed by the pool authority |
| `POST` | `/join/requests` | None | Ask to join. Grants nothing |
| `GET` | `/join/requests` | Admin key | List who is waiting |
| `GET` | `/join/certificate/{name}` | None | Collect an approved node's certificate |
| `GET` | `/ready` | None | 200 when this node can serve a reply, for load balancers |
| `GET` | `/health` | None | The process is up. Says nothing else to strangers |
| `GET` | `/metrics` | Admin key | Prometheus metrics |
| `GET` | `/chat`, `/dashboard` | None | The pages. They ask for a key themselves |

Routes under `/internal` are for nodes only. `GET /internal/mesh` shows a node's own view of the mesh, and `GET /internal/diag/tasks` shows where everything in a process is waiting, which is the place to look when a reply stalls.

### Chat and completion requests

| Field | Default | What it does |
|---|---|---|
| `messages` | Required for chat | List of `role` and `content` |
| `prompt` | Required for completions | A string, or a list holding one string |
| `model` | The running model | Must match it if given |
| `max_tokens` | 32 | 1 to 32768 |
| `temperature` | 0 | 0 to 5. 0 picks the most likely token |
| `top_p` | 1 | Above 0, up to 1 |
| `top_k` | 0 | 0 is off |
| `seed` | None | For repeatable sampling, below 2^63 |
| `stream` | `false` | Send tokens as server-sent events, ending with `data: [DONE]` |
| `stop` | None | A string or list that ends the reply |
| `enable_thinking` | None | Chat only: switches reasoning on models that support it |

Prompts longer than the loaded context are refused with a 400. A reply that would overrun it is cut short.

### Load requests

`POST /admin/load` and `/admin/plan` take these fields. `shardmesh load` sets most of them for you.

| Field | Default | What it does |
|---|---|---|
| `model_id` | `distilbert/distilgpt2` | Any [model id](#loading-a-model) |
| `lane` | None | Load beside the main model, in a lane of this name |
| `lane_gpu_gb`, `lane_ram_gb` | What is spare | Most VRAM and RAM per node for a new lane |
| `lane_apart` | `false` | Keep a new lane off every node another route uses |
| `revision` | `main` | Branch, tag, or commit on the model host |
| `control_model_id` | None | Read the description from here instead of `model_id` |
| `worker_model_ids` | None | A local source per worker, as worker name to path |
| `placement` | `spread` | `spread` or `compact` |
| `context_tokens` | 4096 | 128 to 1048576 |
| `parallel` | 1 | 1 to 32 |
| `entry_worker`, `exit_worker` | None | Pin the ends of the route |
| `draft_model` | `auto` | A GGUF id, `auto`, or `none` |
| `draft_tokens` | 6 | Most guesses per pass, 1 to 16 |
| `merge_passes` | `true` | Let conversations share a pass |
| `model_hashes` | None | The contents of a `shardmesh model hash` file |
| `embedding` | `false` | This is an embedding model. See [Embeddings](#embeddings) |
| `reuse_cached`, `peer_fetch` | Pool default | Weigh cached shards, and copy from peers |
| `wait` | `true` | `false` returns at once. Progress is in `/admin/status` |

### Status codes

| Code | Meaning |
|---|---|
| 400 | Bad request, or a prompt too long for the loaded context |
| 401 | No valid key |
| 403 | The key is valid but lacks the role, or a node is speaking for another |
| 404 | No such model, key, or route |
| 409 | No model is ready, or a change refused as unsafe. The message says which |
| 411 | The request has no length |
| 413 | The request is too large |
| 426 | A node is on a different protocol version. Upgrade it |
| 429 | A key's rate limit, or too many replies waiting. `Retry-After` says when |
| 502 | A stage could not process this request |
| 503 | No coordinator yet, or a stage restarting. Try again shortly |
| 507 | The model does not fit in the pool |

## Models

### Supported families

| Family | `model_type` | Examples | Notes |
|---|---|---|---|
| GPT-2 | `gpt2` | `distilbert/distilgpt2`, `openai-community/gpt2-xl` | |
| Qwen2, Qwen2.5 | `qwen2` | `Qwen/Qwen2.5-0.5B-Instruct` | |
| Qwen1.5/2 MoE | `qwen2_moe` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | Routed and shared experts |
| Qwen3 | `qwen3` | `Qwen/Qwen3-8B` | |
| Qwen3 MoE | `qwen3_moe` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | |
| Qwen3.5, 3.6, 3.8 | `qwen3_5` | `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.8-27B` | Hybrid Gated DeltaNet and attention |
| Qwen3.5, 3.6 MoE | `qwen3_5_moe` | `Qwen/Qwen3.6-35B-A3B` | Hybrid, with fused experts |

Multimodal releases load as text models: the vision tower and multi-token-prediction head are skipped and never downloaded. Block-scaled FP8 releases work too, such as `Qwen/Qwen3.8-27B-FP8`. Not supported yet: AWQ, GPTQ, Qwen3-Next, and the experimental `qwen4_exp`. An unsupported model is refused with the reason before anything downloads.

Workers run these with their own forward pass instead of `transformers` classes, so a node's library versions do not limit which models it can run. Each adapter is tested against the reference `transformers` implementation.

### GGUF models

GGUF is the fast path. A quantized model runs on llama.cpp's kernels, which is what makes a 27B model practical on consumer GPUs, while ShardMesh still plans, fetches, and routes it.

Supported architectures are `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, and `llama`. Real weights have been checked for `qwen3`, `qwen35`, and `qwen35moe`, where two stages matched the whole model, and `qwen35` and `qwen35moe` have run across real machines. The others share that code but are untested. Tokenizers are built from the GGUF itself, for `qwen2` and `qwen35` vocabularies. Split GGUF files (`-00001-of-00003`) are refused for now.

- A GGUF header lists every tensor's byte offset, so each worker fetches just its layers with range requests. llama.cpp's own RPC is not used, because that needs the whole file on one machine.
- A worker assembles a stage file from its cached units: its layers renumbered from zero, with placeholders for anything it never evaluates. To llama.cpp this is a small complete model, so stock release builds work with no patches.
- Hidden states enter through `llama_batch.embd` and leave through the eval callback at the stage's last layer. On Qwen3.6-27B, a two-stage split gives the same tokens and logits as the whole model, and the boundary costs under 3 ms a token.
- Each stage runs in its own process. PyTorch and llama.cpp each load an OpenMP runtime and abort when they share one, and this way a native crash cannot take the worker down.
- Stages read their weights once, unless some are on the disk tier, which needs the file mapped. Mapping kept the whole file in RAM even with the tensors in VRAM: 15.1 GiB against 1.0 GiB on a 15 GiB model, at the same speed.
- Experts are placed separately from the rest of a layer, so a MoE can keep attention on the GPU and its experts in RAM.

The Windows ROCm release of llama.cpp leaves out `hipblas.dll`. ShardMesh finds it in the node's PyTorch ROCm install, so the node needs that environment.

### Speculative decoding

A trip through the pipeline costs nearly the same whether it carries one token or six, so the entry worker runs a small draft model that guesses ahead, and the real model checks every guess in one pass. Each pass yields the guesses that held, plus one more token.

For `qwen35` models the draft is chosen automatically (`Qwen3.5-0.8B`, which shares their vocabulary). Name another with `--draft`, or turn this off with `--no-draft`. A named draft must have the same vocabulary as the model, and a draft that cannot load leaves the model running without one.

The number of guesses adapts as a reply goes, up to `draft_tokens`, because code rewards long guesses and open-ended prose does not. Sampling keeps the model's distribution exactly: a guess `g` is accepted with probability `q(g)`, and a rejection draws from `q` with `g` removed. At temperature 0, batched arithmetic rounds slightly differently from single-token arithmetic, so output can differ from a run without a draft where two tokens are nearly tied.

### Conversation reuse

Every stage saves its state where a chat's settled history ends. If the next prompt starts with exactly those tokens, stages restore the checkpoint and process only the new text, so a long conversation stops costing more with every turn. Anything else, such as an edited history or a new route, falls back to reading the whole prompt. This covers GGUF chat.

### Several conversations at once

`--parallel 3` gives every stage three independent sequences, each with its own context memory, which the planner counts. Replies from any node queue at the entry worker, and each conversation returns to the sequence holding its checkpoint. While one reply is away at the next stage, this stage works on another.

Single-token steps from different conversations also share a pass through a stage, because reading the weights is what a pass costs. On one stage, four conversations went from 12 to 32 tokens per second. Across two stages the same load gained only 10%, from 34 to 37, because the stages stagger the conversations. Steps with speculative guesses do not merge, so a draft limits this. `--no-merge-passes` turns it off.

Past `--parallel`, replies queue at the entry worker, and past 32 waiting they are refused with a 429.

### How a model loads

1. The coordinator reads the model's description: a GGUF header, or `config.json`, the tokenizer, and each safetensors header. This costs a few megabytes and pins one revision for every node.
2. It plans which worker runs which layers, and which memory holds each unit.
3. Each worker fetches only its units into a local cache, trying its own disk, then peers, then a holder of the file or the model host. Interrupted transfers retry and resume.
4. The current model keeps serving through all of this. Entry workers let replies finish, then stages load. A stage whose job did not change keeps its weights.
5. The new route goes to every node by gossip.

If a live node fails to load, it gets two retries. After that, a replan goes ahead only if it moves under 2 GiB. Otherwise the load stops, says why, and the old model keeps serving. A dead node is replanned around whatever that moves.

Two behaviors keep transfers down, both on by default. Shards in place count, because each plan is charged its download time spread over 20,000 tokens, so a near-tie stays where the shards already are. And workers share shards, falling back to the host if a peer fails or sends the wrong size or header.

| To turn off | For one load | For a control plane | On one node |
|---|---|---|---|
| Weighing cached shards | `load --no-reuse-cached` | `--no-reuse-cached` | |
| Fetching from peers | `load --no-peer-fetch` | `--no-peer-fetch` | `--no-peer-fetch` |
| Serving shards to peers | | | `--no-peer-serve` |

To check every byte against a list you trust, see [Models in the security guide](docs/SECURITY.md#models).

## Embeddings

`POST /v1/embeddings` takes OpenAI's request and runs a GGUF embedding model on llama.cpp, behind the same keys and rate limits as chat. Load one with `--embedding`, in a lane or as the main model:

```bash
shardmesh load Qwen3-Embedding-0.6B-Q8_0.gguf --lane vectors --embedding --url http://node-a:8669
```

```bash
curl http://node-a:8669/v1/embeddings -H "Authorization: Bearer KEY" -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-Embedding-0.6B-Q8_0.gguf", "input": ["first text", "second text"]}'
```

| Field | Default | What it does |
|---|---|---|
| `input` | Required | A string, or a list of up to 2048 strings. Token arrays are not accepted |
| `model` | The main model | Name a lane's model here |
| `encoding_format` | `float` | `float`, or `base64` for little-endian float32 |
| `dimensions` | The model's width | Shorter vectors, on models trained for it |
| `user` | None | Accepted for OpenAI clients, and not used |

The reply holds one `embedding` per input, in order, and `usage` counts `prompt_tokens` and `total_tokens`.

- Vectors come from llama.cpp's embedding mode, with the pooling the GGUF declares, scaled to unit length. A GGUF that declares mean, CLS, or last-token pooling has to be loaded with `--embedding`, and one that declares none is refused with it.
- An embedding model takes one stage on one worker, whatever `--placement` says, because pooling happens at the end of the model over whole sequences. If it would need more than one stage, the load is refused with a 507.
- The control plane packs inputs into passes of up to 64 inputs and 1024 tokens. An input longer than that gets a pass of its own.
- An input longer than `--context-tokens` is refused with a 400 that says which one. Nothing is cut silently, because a cut text gives a different vector.
- `dimensions` keeps the leading values and scales again. A GGUF cannot say whether a model was trained for this, so it works for the Qwen3-Embedding family, down to 32, and any other model is refused unless the value is its full width.
- A chat model asked for embeddings, and an embedding model asked to chat, both get a 400 that says so.
- `GET /v1/models` gives each model a `kind` of `chat` or `embedding`, and `dimensions` for an embedding model. The chat page leaves embedding models out of its picker.
- On a stage with no GPU layers, embedding mode turns off llama.cpp's operation offload. Left on, it took 1.2 GiB of VRAM on a node given no GPU, at the default 4096 tokens of context.

Measured with Qwen3-Embedding-0.6B Q8_0 on four CPU threads, on a machine busy with other jobs. The reference is `llama-server --embedding --pooling last --no-op-offload`, because the release has no `llama-embedding`.

| Measurement | Result |
|---|---|
| 64 texts one at a time, against llama-server | Cosine 1.0, largest difference in any value 1.5e-8 |
| 64 texts in one request, against llama-server the same way | Cosine 0.9996 at worst |
| llama-server batched, against itself one at a time | Cosine 0.9996 at worst, so batching alone moves vectors this much |
| Three pairs of sentences meaning the same thing | Cosine 0.77 to 0.80 |
| Most similar unrelated pair | Cosine 0.38 |
| 256 short inputs, 3,884 tokens | 239 tokens/s, 15.8 inputs/s |
| 8 long inputs, 4,496 tokens | 248 tokens/s |
| llama-server, 64 short inputs on the same settings | About 270 tokens/s |

An embedding model has not been run on a GPU stage. BERT-family GGUFs are refused, because only the architectures in [GGUF models](#gguf-models) can load.

## Memory and placement

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

The ends of the pipeline are not equal: the exit reads the whole output head for every token, while the entry reads one embedding row. So the planner tries every order through the workers and keeps the fastest (up to four workers exhaustively, then a sample of orders). Pin the ends with `--entry` and `--exit`, or in the chat page's picker, for example to keep prompts and replies on a machine you trust. Only the entry sees prompt tokens and only the exit sees generated tokens, though activations passing through other stages can still reveal information. One worker pinned to both ends runs the whole model. If a pinned worker is lost, recovery drops the pin and logs it.

An explicit `--gpu-memory-gb` is a hard allocator limit on CUDA and ROCm. RAM and disk budgets are enforced by placement. Disk is a capacity tier: expect it to be far slower than RAM once a model outgrows the OS page cache.

The dashboard's "Memory for weights" counts weights only. Real use is higher, because KV cache, compute buffers, a draft model, and each worker's GPU context come on top.

### Mixed hardware

Workers do not use NCCL, CUDA IPC, or remote device pointers. Each process loads a PyTorch build that suits its own host, and activations cross the network as plain tensors, so backends and driver versions can differ between nodes.

| Hardware | Status | Notes |
|---|---|---|
| NVIDIA CUDA | Tested | Toolkit and driver versions may differ by node |
| AMD ROCm | Tested on Windows | With an RX 7900 XT, in BF16 and with GGUF. PyTorch exposes the device as `cuda` |
| CPU | Tested | Slow, but useful as a stage or for testing |
| Apple MPS | Experimental | Reported as unified memory. Not covered by the test suite |
| Intel XPU | Experimental | Discovered as VRAM. Not covered by the test suite |

Differences between backends can change sampling when logits are nearly tied. Bit-for-bit identity is not promised between different kernels.

### Any GPU

A node uses PyTorch to find GPUs. If PyTorch sees none, the node asks llama.cpp's Vulkan build instead, in a process of its own, so a GPU with a working Vulkan driver takes part whatever its maker. `--gpu-device vulkan` chooses this on purpose. It downloads a pinned build the first time, as other builds do.

Vulkan runs GGUF models only, because safetensors models run on PyTorch. The planner keeps those off such a node. It also cannot time the GPU's memory, so it assumes 300 GB/s when placing layers.

Measured with small models, one reply at a time. Each pair gave the same text:

| GPU and model | Backend | Tokens per second |
|---|---|---|
| RX 7900 XT, Qwen3.5-0.8B Q8_0 | Vulkan | 185.1 |
| RX 7900 XT, Qwen3.5-0.8B Q8_0 | ROCm | 161.9 |
| RTX 3060 Laptop, Qwen3-0.6B Q8_0 | CUDA | 104.1 |
| RTX 3060 Laptop, Qwen3-0.6B Q8_0 | Vulkan | 102.6 |
| Laptop CPU, 4 threads, Qwen3-0.6B Q8_0 | CPU | 21.9 |

So an AMD card needs no ROCm for GGUF models. These are small models, and a large one may differ.

## Precision

Workers run in BF16 by default. Add `--dtype float32` to a node to run its VRAM and RAM at full precision, which doubles the memory each layer takes. The planner accounts for that, so BF16 and FP32 nodes can share a route. Published Qwen checkpoints are BF16, so FP32 raises the precision of computation and not of the stored weights. The disk tier always runs in the checkpoint's own dtype.

An FP8 release halves the download. In memory, the planner chooses per unit between two forms:

| Form | Memory | Speed measured on an RTX 3060 and a laptop CPU |
|---|---|---|
| Expanded to BF16 at load | Twice the FP8 size | Full speed: 143 Gparam/s on GPU, 14.5 on CPU |
| Compact, expanded on each use | FP8 size | 20 Gparam/s on GPU, 0.65 on CPU |

Compact is several times slower because PyTorch has no fused FP8 kernel, and neither RTX 30 nor RDNA3 GPUs compute in FP8 natively. So the planner expands what fits and keeps only the overflow compact. The gain comes from avoiding disk and fitting models BF16 cannot, not from faster reads. The expansion rates are constants measured on one machine, so plans for other hardware are estimates.

## Networks and security

The full guide is [ShardMesh security](docs/SECURITY.md).

| Network | What to set |
|---|---|
| Closed network, no internet | Put GGUF files on any node with `--gguf-dir` and load by file name. Give each node a llama.cpp build with `--llama-dir`, because automatic downloads need GitHub |
| LAN | Use TLS, or pass `--trusted-network` if nobody else can read the network. Without either, a node refuses to start, because credentials would cross the LAN unencrypted |
| VLANs | As for a LAN. Discovery does not use broadcast, so `--join` works across VLANs. Open the control port and worker ports between them |
| Tailscale | Nothing special. Links are encrypted, and addresses work from anywhere on the tailnet. Restrict the ports to pool members with tailnet grants |
| Remote machines over the internet | Use a pool authority and TLS on every node. Expect each extra stage to cost the round trip between those machines |

A pool can mix these, as long as every node can reach every other node's advertised address.

Who can do what:

| Caller | Credential | Can do |
|---|---|---|
| Node, with a pool authority | Certificate, over TLS or as a ticket | Node routes only |
| Node, without one | Pool token | Everything |
| Admin | API key, role `admin` | Load models, cordon, read metrics, and all user actions. Manage keys and revoke nodes in a pool without an authority |
| User | API key, role `user` | Chat, completions, read status |
| Owner | The authority's private key | Add, remove, and revoke, in a pool with an authority |

Antivirus that scans HTTPS replaces certificates, which ends mutual TLS. Nodes refuse to trust it, so the failure is safe, and a node reports it plainly. Use Tailscale with tickets, which need no TLS at all, or exclude the pool's ports from scanning.

## Checking answers

Any node in a route can return anything, and nothing in a reply proves otherwise. What can be measured is agreement: identical prompts, with no randomness, on routes built from different machines. They hold the same weights, so honest routes run together, and a broken or tampering one parts from them.

Load the same model again on other machines, then ask:

```bash
shardmesh load Qwen3-0.6B-Q8_0.gguf --lane check --lane-apart --url http://node-a:8209
```

```bash
shardmesh audit --url http://node-a:8209
```

With two routes, a disagreement says something is wrong without saying where. With three or more, the route out of line with the others is named, along with the machines it runs on. What you do about that is yours: cordon or revoke it in an owned network, or `shardmesh net block` it for your own node in one nobody owns.

The questions are made up and look like ordinary chat, so nobody's real words travel anywhere extra, and a node cannot easily tell it is being tested. This is a smoke alarm, not a proof. It catches a stage that is broken, or that tampers with what it handles. It does not catch one that behaves whenever it might be watched, or that only touches certain topics. It costs a second copy of the model, which a large mesh can spare and a small one may not.

## Running a pool

The full guide is [Operating ShardMesh](docs/OPERATIONS.md).

**Start at boot.** A node started with `shardmesh up` needs only `shardmesh service install`, which starts it when you log in. To start at boot instead, on Windows, `scripts\install_node_task.ps1` registers a scheduled task. On Linux, use `packaging/shardmesh-node.service`, which has not been run on a real machine yet. A node restarts its own services, so these only have to start it once.

**Maintenance.** `shardmesh cordon NAME` lets replies finish and moves that node's layers away. The node stays in the mesh. `shardmesh uncordon NAME` brings it back. A cordon is shared by every node and survives restarts. It is refused if the pool could not hold the model without that node, and if it would transfer over 2 GiB, unless you add `--force`.

**Upgrades.** Nodes refuse peers on a different `PROTOCOL_VERSION`, and the refusal names both versions. Upgrade all nodes together when that number changes. Otherwise, restart them one at a time, and the mesh carries on around each one.

**Backups.** Keep the authority folder safe and offline. Losing `ca.key` means reissuing every certificate, and anyone who gets it controls the pool. Route state is copied to every node, so it needs no backup.

**Files.**

| What | Where | Notes |
|---|---|---|
| Route state | `orchestrator-state.json` in the state folder | Copied to every node by gossip |
| Keys and revocations | `access.json` in the ShardMesh folder | Copied to every node by gossip |
| Audit log | `audit.jsonl` in the state folder | Rotates at 64 MiB, keeping one old file. Each node logs only what it served |
| Node log | `--log-file` | Nothing is kept unless you set this |
| Fetched shards | `--cache-dir` | Evicted past `--cache-budget-gb` |
| Stage files | Inside the cache | Rebuilt from shards when a stage moves. Count double the node's share |

**Metrics.** `/metrics` serves Prometheus format on every node, to an admin key. All names start with `shardmesh_`.

| Metric | Meaning |
|---|---|
| `serving` | Whether this node can serve replies. Alert at 0 |
| `workers_quarantined` | Workers held out of the route. Alert above 0 |
| `workers_live`, `worker_live` | Workers heard from recently, overall and per worker |
| `mesh_members` | Control planes reachable from here. Alert below the node count |
| `mesh_term`, `coordinator`, `epoch` | Leadership term, whether this node coordinates, and the pipeline epoch |
| `stream_fallbacks_total` | Replies that fell back to replay. Should stay near zero, because fallbacks hide trouble |
| `stage_errors_total` | Requests a stage refused. Nobody is quarantined for these, so a real fault shows here |
| `stage_restarts_total`, `worker_stage_restarts_total` | Replays after a stage restart, and stage processes restarted |
| `recoveries_total` | Pipeline repairs during a reply |
| `requests_total`, `tokens_generated_total`, `embedding_tokens_total`, `replies_in_flight` | Traffic through this node |
| `token_latency_ms`, `worker_compute_ms` | Time per token, and compute per pass |
| `worker_stages_kept_total` | Reconfigures that moved no weights |
| `worker_draft_tokens_per_pass` | Tokens per speculative pass. Near 1 means the draft is not helping |

## Troubleshooting

| What you see | Cause | What to do |
|---|---|---|
| `reachable by other machines and this pool has no credentials` | No token or authority on a node listening on the network | Run `shardmesh pool init`, or pass `--open` on a network you trust completely |
| `would carry the pool's credentials unencrypted` | The node would send credentials across a plain network | Use Tailscale or TLS, or pass `--trusted-network` |
| The chat page keeps asking for a token | A wrong key, or a pool token in a pool with certificates | Use an API key. With certificates, the pool token does nothing |
| `no model is ready; load one first` | Nothing is loaded, or a load is still running | Load a model, or check progress on the dashboard |
| `shardmesh load` cannot connect | It assumes the demo's port | Pass `--url http://NODE:8209` |
| A node shows as standby | Its share moved away after a failure, and the model still fits without it | `POST /admin/rebalance` when you want the layout back. It returns by itself when idle |
| `failed to load, and replanning would move` | A live node could not load its stage, and moving its share would be large | Read the error after the colon. The old model keeps serving meanwhile |
| `no node could send bytes` | A peer transfer failed after retries. The message ends with the reason | Check the link, and that the node holding the file is up |
| `CERTIFICATE_VERIFY_FAILED`, or an event mentioning antivirus | Something replaced a certificate | Exclude the pool's ports from scanning, or use tickets over Tailscale |
| A node reports HTTP 426 | Nodes are on different versions | Upgrade them together |
| The ROCm build sees no GPU | `hipblas.dll` is missing | Use the node's PyTorch ROCm environment, which carries it |
| Far slower than usual | VRAM spilled to shared memory, or a stray process holds the GPU | Check GPU memory per process. A stage with a large "shared" figure needs a reload |
| A reply went quiet | A stage or link is stuck | Read `GET /internal/diag/tasks` on each node to see what everything is waiting on |
| `the rest of the pool cannot hold this model` | A cordon would leave nowhere for that node's layers | Add memory, or load a smaller model first |
| `libgomp.so.1: cannot open shared object file` | A minimal Linux install lacks a library llama.cpp needs | `sudo apt-get install libgomp1` |
| A browser warns about the certificate | `shardmesh up` used TLS with the mesh's own certificate, off Tailscale | Accept it once, or use Tailscale, which needs no warning |
| `nobody signed this request` | The owner's machine is off, or the invite ran out | Start `shardmesh up` on the owner's machine, and make a new invite |
| `is not the mesh this invite is for` | The address in the invite answers with a different mesh | Stop, and ask the owner for a new code. Something else is at that address |
| `another machine already uses the name` | Two machines have the same name | Run `shardmesh up --invite CODE --name another-name` |
| `This machine already belongs to a mesh` | `shardmesh up` has settings from an earlier mesh | Remove the folder it names to join another |
| A new machine does not find the mesh | A firewall blocks UDP port 8208, or the mesh was started with `--no-discovery` | Use `shardmesh invite` |
| `only the machine that made this mesh can invite` | The authority's key is on the owner's machine | Run `shardmesh invite` or open the page there |
| A lane stays on `starting` | A firewall blocks its ports, which are 1000 per lane above the node's own | Open them between nodes, then remove the lane and load it again |
| `no node has memory left under the limits its owner set` | The main model uses all that owners gave | Give nodes more with `--gpu-memory-gb` or `--weights-ram-gb`, or load a smaller main model |
| In a network nobody owns, a node never appears | The others do not accept it yet | Have a node they accept run `shardmesh net vouch`, or use `--joining open` |
| `nobody can replace what this network runs` | In a network nobody owns, the running model stays | Load yours beside it with `--lane` |

## Measured performance

Qwen3.6-27B Q4_K_M, split across an RX 7900 XT desktop and an RTX 3060 laptop, both on Wi-Fi with a 15 ms ping:

| Setup | Tokens per second |
|---|---:|
| Desktop alone, llama.cpp, no split (ceiling) | 35.2 |
| Both nodes, llama.cpp's own RPC, whole file on one node | 13.6 |
| Both nodes, ShardMesh, no draft | 14.6 |
| Both nodes, ShardMesh with a draft, prose and lists | 19–21 |
| Both nodes, ShardMesh with a draft, code | 33 |
| Both nodes, ShardMesh with a draft, chat page settings, four-turn chat | 21.6 |

Models that need or suit the split, on the same test pool:

| Model | Placement | Tokens per second |
|---|---|---:|
| Qwen3.8-27B Q8_0, 26.6 GiB, too big for either machine | Desktop 14 GiB VRAM + 3.9 GiB RAM, laptop 3.1 GiB VRAM + 5.7 GiB RAM | 4.2–6.7 |
| Qwen3.6-35B-A3B IQ2_XXS, a MoE with 256 experts | Laptop 1.2 GiB VRAM, desktop 8.8 GiB VRAM | 27 for one conversation, 48 in total for three |

A 1,860-token first message took about 6.5 s to the first token. Later turns of that chat took 0.7 to 1.3 s, with over 2,000 tokens of history behind them. Downloads share the link: chat dropped to about 5 tokens per second while a new model downloaded over the desktop's Wi-Fi.

Planner estimates leave out the draft and are rough. The figures above were measured. For failure timings, see [What happens when something fails](#what-happens-when-something-fails).

## Development

```bash
python -m pip install -e ".[test]"
```

```bash
python -m pytest tests -q
```

Adapter tests compare every family with the reference `transformers` implementation, and skip families newer than the installed release, so run them under a current `transformers` as well. TLS tests skip on a machine where antivirus intercepts loopback. `tests/test_readme.py` fails if an option, setting, endpoint, or metric is missing from this file.

`scripts/ci_local.sh` runs what CI runs from a clean clone, which catches anything only your machine has.

| Script | What it does |
|---|---|
| `install.sh`, `install.ps1` | Install for one user, picking a PyTorch build |
| `scripts/build_tiny_model.py` | Builds the offline test model |
| `scripts/verify_model.py` | Compares a running checkpoint with an unsharded reference |
| `scripts/build_diagrams.py` | Redraws the diagrams in this file |
| `scripts/ci_local.sh` | Runs the tests from a clean clone |
| `scripts/install_node_task.ps1` | Starts a node at boot on Windows |
| `scripts/launch_lan_node_55.py`, `scripts/launch_lan_node_55_rocm.py`, `scripts/bootstrap_remote_node_55.py`, `scripts/bootstrap_remote_rocm_node_55.py` | Example launchers. They read `SHARDMESH_NODE_ID`, `SHARDMESH_JOIN`, `SHARDMESH_ADVERTISE_HOST`, and `SHARDMESH_GGUF_DIR` |

| Code | What it holds |
|---|---|
| `shardmesh/orchestrator.py` | The control plane: API, planning, repair, gossip handling |
| `shardmesh/worker.py` | A worker: stages, the token loop, merged passes, fetching |
| `shardmesh/up.py`, `shardmesh/discovery.py` | One-command setup: settings, invites, finding a mesh, `doctor`, and `service` |
| `shardmesh/node.py` | The launcher that supervises a node's services |
| `shardmesh/lanes.py` | Several models at once: the front door and each node's lanes |
| `shardmesh/commons.py` | A network nobody owns: self-made identities, vouches, and each node's view |
| `shardmesh/inventory.py`, `shardmesh/gpu_probe.py` | What a node reports about itself, and how it finds GPUs PyTorch cannot see |
| `shardmesh/audit.py` | Comparing routes that serve the same model |
| `shardmesh/mesh.py` | Membership and leader election, as pure logic |
| `shardmesh/planner.py`, `shardmesh/memory.py` | Placement, and memory pool discovery |
| `shardmesh/gguf.py`, `shardmesh/gguf_stage.py`, `shardmesh/llamacpp.py`, `shardmesh/llama_host.py` | GGUF parsing, stage files, the llama.cpp binding, and the stage process |
| `shardmesh/weights.py` | Readers, the unit cache, and hash checks |
| `shardmesh/access.py`, `shardmesh/pool.py`, `shardmesh/tickets.py`, `shardmesh/tls.py` | Keys, certificates, signed entries, tickets, and TLS |

## Current limits

Trust and status:

- Two machines is the most this has run on, so everything about larger meshes is design, not measurement. See [Scaling to many nodes](#scaling-to-many-nodes). The longest test ran under an hour. No human has audited it, and reviews by AI do not show it is free of flaws.
- A node in the route can read prompts. It can return wrong answers too: [checking answers](#checking-answers) catches a route that disagrees with others, but not one that only misbehaves when it is not being tested. Certificates limit who is in a pool. They do not make a hostile member safe.
- An owned network has an owner.
- A network nobody owns has run on three processes on one machine, not between strangers. Identities are free, so `--joining open` lets one person run many nodes, and a low name can be ground out to win coordination. A coordinator can stall a route. Traffic is not encrypted, so run it over a network that is, such as Tailscale. Use `--joining vouched` with a quorum unless you accept all of that.
- Several models run in lanes with fixed memory shares. The main model's planner does not see a lane's memory, so moving layers after a failure can run a node short. Workers started with `--no-control` do not join lanes.
- GitHub Actions has not run yet, because the account's jobs do not start. The same steps pass locally on Linux and Windows.
- Linux has run real nodes on CPU only, in WSL. A Linux GPU, and macOS, have not been run at all, although their builds are pinned and download by themselves.
- Adding a machine with an invite needs the owner's machine running at that moment. The mesh keeps working without it.

Behavior:

- Prompts do not overlap across stages: one stage finishes before the next begins, so a long first message costs the sum of them.
- Hybrid Qwen models on the PyTorch path process prompts token by token. This is exact, but slower than a chunked kernel on long prompts.
- Long-context rope scaling (YaRN) and sliding-window attention are refused rather than run incorrectly.
- Expert placement is per layer, not per expert: frequently used experts are not promoted to faster memory.
- The planner does not reserve VRAM for a draft model. It loads into what is left on the entry node, and if it does not fit the model runs without it.
- Multi-token batches cross between GGUF stages as BF16, and single tokens as FP32. A node set to `--dtype float32` keeps everything in FP32.
- A stage file duplicates that node's cached units on disk.
- GGUF files are parsed by native code in the stage process, so load them only from sources you trust.
- The planner uses measured memory bandwidth and a fixed cost per network hop, but not per-layer benchmarks or pairwise network telemetry. Its disk estimate assumes cold reads, so warm runs beat it.
- Workers on one host split shared RAM evenly during planning, which can refuse a plan an uneven split would allow.
- There is one active RAM checkpoint service per route. Snapshots are not yet replicated or compressed. If the store is behind when a stage fails, recovery replays the prompt.
- With two nodes, a split network leaves two coordinators until it heals. Nothing can be corrupted, but neither side can know it is the minority.
- A reply is lost if the node its client is connected to dies.
- Multi-turn reuse keeps one conversation per sequence, so more conversations than sequences push each other out and reread their history.
- Mesh membership is meant for tens of nodes: every worker checks in with every control plane.

## License

ShardMesh is licensed under the [Apache License 2.0](LICENSE). Copyright 2026 AuXFire.
