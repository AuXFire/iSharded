# iSharded security

iSharded runs code and holds prompts on every machine in a pool, so a pool is only as trustworthy as its least trusted node. Read this before connecting machines you do not fully control.

## Trust model

A node in the route sees prompts and replies in plain text, as hidden states or tokens, and can return anything it likes. A node you cannot trust should not be in your pool. Certificates and revocation limit what a stolen credential can do and let you throw a node out. They do not make a hostile node safe to include.

## Who owns a pool

An ordinary pool is decentralized in how it runs, not in who owns it. No server is special, any node can coordinate, and losing any machine loses nothing but what it was running. But it has an owner: whoever holds the authority key, or the shared token in a pool without one. That person decides who joins, who is thrown out, and who may load models.

This suits a pool whose members trust each other. It is not a public mesh. A public mesh, where anyone can join, would need answers iSharded does not have today: a way to detect wrong results that a careful node cannot dodge, a way to stop one person running thousands of nodes, and a clear warning that node operators can read prompts. [Checking answers](../README.md#checking-answers) catches a route that disagrees with others, and no more. [A network nobody owns](#a-network-nobody-owns) covers what is built for a public mesh, and what stays unanswered.

## Invites

`isharded up` makes an owned network without anyone handling certificates. It does not weaken anything: the authority's key stays on the owner's machine, and only the process there can sign.

- A code carries the mesh's addresses, a hash of its authority, and a secret. It works a set number of times, one by default, for a set time, 24 hours by default.
- A new machine fetches the authority and checks it against the hash, so a stranger cannot pose as the mesh. Its key never leaves it.
- It proves it holds the secret with an HMAC over its exact request and key hash. The proof travels by gossip, so a member who sees it pass cannot reuse it on another request.
- The owner's machine checks the proof, signs a certificate, and signs a user key for the new machine. Nothing else in the mesh can admit anyone, so if that machine is off, an invite waits.
- Anyone holding a live code can join until it is used. Treat it like a password, and keep the time short.
- `invites.json` on the owner's machine holds live secrets beside `ca.key`, and has the same protection.

## Finding a mesh

A mesh started with `isharded up` answers a UDP broadcast on port 8208, and serves its public certificate at `/join/authority`, so anyone on the network can learn that it exists, its name, and its address. `--no-discovery` turns the broadcast off.

An answer proves nothing, and anyone can send one. A machine that finds a mesh only asks to join. A person on the owner's side approves it after comparing the key fingerprint both machines show, so nothing gets in by being found. Going the other way, a rogue answer can lure a new machine to ask a stranger, so a new machine shows the authority's hash and asks you before it does. `--find` skips that question, so use it only on a network you trust.

## A network nobody owns

In this kind of network there is no pool authority and no owner. [The README shows how to run one](../README.md#a-network-nobody-owns). What changes:

- A node makes its own key, and its name is a hash of that key. A certificate is accepted only if it signed itself and carries the name its key gives, so nobody can claim another node's name.
- Each node decides who it accepts: anyone, or nodes vouched for by nodes it already accepts, starting from itself and whoever it trusts. Nodes can disagree, and nothing makes them agree.
- Gossip carries signed vouches and nothing else. Keys, blocks, and cordons never travel between nodes, and a signature on any other kind of entry counts for nothing.
- An operator's key is an admin of their own node only. To everyone else, including on the coordinator's own node, they are a member: they can propose a model beside what runs, remove a lane their node proposed, and take their own node out.
- A block is one node's own choice. Nobody can remove a node for everyone, which also means a bad node stays in the network for anyone who has not blocked it.

What this does not give you:

- Identities cost nothing. With open joining one person can run many nodes. With vouched joining, the protection is only as good as the vouchers.
- A name is a hash, so someone can grind out a low one to win coordination. A coordinator can stall a route, or place work badly. It cannot make a node host a model that node refuses, or change who it accepts.
- A member in the route can read prompts and return wrong answers, as in any pool, and here you did not choose that member.
- Traffic is not encrypted: certificates here prove names, not privacy. Run it over a network that encrypts, such as Tailscale.

## Who can do what

| Caller | Credential | Can do |
|---|---|---|
| Node, with a pool authority | Certificate, over TLS or as a ticket | Node routes only: gossip, registration, stage traffic |
| Node, without one | Pool token | Everything, including the node protocol |
| Admin | API key, role `admin` | Load models, rebalance, cordon, read metrics, and all user actions. Manage keys and revoke nodes in a pool without an authority |
| User | API key, role `user` | Chat, completions, read status |
| Owner | The authority's private key | Add, remove, and revoke, in a pool with an authority |

A node starts refusing to serve a network if it would be exposed:

- With no credentials, it will not listen beyond loopback. Run `isharded pool init`, or pass `--open` if the network itself is the protection.
- With credentials but no encryption, it will not advertise an ordinary LAN address, because every gossip round carries the pool token. Use TLS or a Tailscale address, or pass `--trusted-network` if nobody else can read that network.

## Setting up a pool

For a pool on Tailscale or a closed network, a shared token is enough:

```bash
isharded pool init --dir pool-secrets
```

Copy `pool.token` to each node as `%LOCALAPPDATA%\iSharded\pool.token` (Windows) or `~/.isharded/pool.token`. Anyone holding it can act as any node.

For machines you may need to throw out individually, or remotes across the internet, give each node a certificate.

```bash
isharded pool add-node --dir pool-secrets node-b --address 10.0.0.6
```

Keep `ca.key` off the nodes. The name in the certificate must match `--node-id`, because a node may only speak for itself and its own workers.

A node proves its name in one of two ways, using the same certificate:

| | Use it when | How |
|---|---|---|
| Tickets | The link is already private, as on Tailscale, or antivirus intercepts TLS | `--ca-file ca.pem --identity-cert node-b.pem --identity-key node-b.key` |
| Mutual TLS | The link needs encrypting, and nothing intercepts TLS | `--ca-file ca.pem --tls-cert node-b.pem --tls-key node-b.key` |

A ticket is a short signed statement in an ordinary header. It names the node it is for and expires within two minutes, so a peer that receives one cannot use it elsewhere. Tickets encrypt nothing, so a node using them on an ordinary LAN refuses to start without `--trusted-network`. Clocks must agree to within a minute.

With a pool authority the pool token counts for nothing, on control planes and workers alike, so do not copy it to nodes.

### Antivirus on nodes

Many antivirus products intercept TLS on every port, even on loopback, and replace the certificates in it. Nodes only trust the pool authority, so this fails closed: mutual TLS stops working rather than becoming insecure. AVG does this on the development laptop. There are three ways to run such a node:

- Use Tailscale and tickets. Nothing depends on TLS, so antivirus has nothing to intercept. This is what the test pool runs, with antivirus on.
- Exclude the pool's ports or `python.exe` from HTTPS scanning, and use mutual TLS.
- On a private network, use tickets with `--trusted-network`, accepting that traffic is unencrypted.

## Adding members

An owner can issue a certificate directly with `isharded pool add-node`, or a node can ask. Asking keeps the node's private key on the node, and the owner's key off the network.

```bash
isharded pool request-join node-c --dir node-identity --address 100.64.0.9 --url http://10.0.0.5:8209 --wait
```

The node prints a key fingerprint. Any member passes the request on. The owner lists what is waiting, checks the fingerprint with whoever is asking, and answers:

```bash
isharded pool requests --url http://10.0.0.5:8209 --token <admin key>
```

```bash
isharded pool approve node-c --dir pool-secrets --url http://10.0.0.5:8209 --token <admin key>
```

```bash
isharded pool deny node-c --dir pool-secrets --url http://10.0.0.5:8209 --token <admin key>
```

Asking grants nothing. An approved node collects its certificate from any member and starts with `--ca-file`, `--identity-cert`, and `--identity-key`. A denied key stays denied, and a revoked name cannot ask again. Up to 64 requests wait at once. Check `ca.pem` against the owner's copy before trusting it, because the node that hands it over could be lying.

## Managing keys and nodes

With a pool authority, keys and revocations are signed with `ca.key` and sent to any node, and gossip carries them to the rest. Nodes ignore anything unsigned, so a compromised node cannot add a key, remove one, or revoke a peer.

```bash
isharded key create alice --role user --requests-per-minute 60 --dir pool-secrets --url http://10.0.0.5:8209
```

```bash
isharded key remove alice --dir pool-secrets --url http://10.0.0.5:8209
```

```bash
isharded pool revoke node-b --dir pool-secrets --url http://10.0.0.5:8209
```

A key is shown once and stored as a SHA-256 hash. `/admin/keys` and `/admin/nodes/{name}/revoke` refuse these changes in such a pool, and say to use these commands.

Without a pool authority, an admin key or the pool token changes these through any node: `POST /admin/keys`, `DELETE /admin/keys/{name}`, `POST /admin/nodes/{name}/revoke`. To seed a key before a pool starts, in either kind of pool, use `isharded key create root --role admin --file access.json`, and give each node the file with `--keys-file`. A key file is trusted by the node that reads it, and nothing else.

Revocation stops every node accepting that certificate, takes its workers out of the route, and stops gossip with it. A revoked name cannot register new workers.

Revocation is permanent for that name, and so is removing a key. Both are kept for good and merged entry by entry, so neither a network split nor a forged document can undo them. To bring a machine back, issue it a certificate under a new name. A removed key's name cannot be used again in a pool with an authority.

## Encryption

| Network | Encrypted by |
|---|---|
| Tailscale | Tailscale |
| LAN, VLAN | Nothing, unless you enable TLS |
| Internet | You must enable TLS |

Certificates need the address each node advertises in their subject alternative names.

Antivirus web shields can intercept TLS even on loopback, replacing the node's certificate with their own. AVG does this on the development laptop, which is why the TLS tests skip there. On such a machine, exclude the pool's ports from inspection, or connections will fail verification.

## Limits

Requests above 16 MiB are refused from their headers before any of the body is read, and bodies must declare a length. Stage traffic may reach 1 GiB, and only `/internal/forward` and the bandwidth probe accept that much. A copy of a saved state may be larger. Only `/internal/state/copy` accepts one, it checks the sender before reading any of it, and it writes to disk as the bytes arrive, up to the node's `--state-disk-gb`.

A request whose content a stage cannot process fails with an error and costs nobody their place in the route. Only losing contact with a stage can quarantine it, so a user cannot empty the pool one request at a time.

Peers name model sources, so a worker in a secured pool only loads local models from its `--gguf-dir` folders, its cache, and any `--model-dir`. File names from peers cannot leave a model's folder, and a draft model can only be a `.gguf`.

## Saved conversations

A saved prompt state is what a stage worked out from someone's conversation. Treat it as the conversation itself: nobody should assume the text cannot be recovered from it.

- Each stage keeps states in RAM, and writes them to `states` in its `--cache-dir`, unencrypted. On Linux and macOS the folder and files are readable only by the node's own user. On Windows the permission calls do little, so the files have whatever the folder gives them. Keep the cache somewhere private, and on an encrypted disk if the machine could be stolen.
- States stay on disk after a reply ends, until they are evicted or the model or the stage's layers change. Nothing deletes a conversation's states on request.
- One other node keeps a copy of each state, for its own stage's layers only. That is the next stage along the route, which already sees what this stage sees, unless a node offers `--ram-budget-gb` for this. A node like that may be outside the route, and it then holds text it would otherwise never see.
- Only pool members can send or fetch a state. A copy is checked for damage, but a pool member could send a false one, in the same way a stage can return false results. llama.cpp parses a state in the stage process, so a bad one costs that process and nothing else.
- Stages share states between conversations by their exact tokens. A person who can guess the whole start of someone else's prompt can tell from timing whether it was sent recently.

`--state-disk-gb 0` keeps states in RAM only, `--state-copy-mb 0` keeps them on their own node, and `--no-prompt-cache` on a load saves none at all.

## Audit log

Each node records who loaded models, changed the pool, and asked for replies: time, node, caller, path, status, and client address. It never records prompts or replies. Every node that handles a request records it, so a forwarded request appears twice, and you need the files from every node for the full picture.

## Models

GGUF files are data, but they are parsed by native code, in a separate stage process with the node's privileges. Load files only from sources you trust. Safetensors weights avoid pickle, and pickle checkpoints are refused. Chat templates run in a Jinja sandbox, and need Jinja2 3.1.6 or newer.

Fetched ranges are checked for size, and pinned to a commit or a content fingerprint. The fingerprint covers the file's first megabyte and its size, which identifies a file but would not detect tampering deeper in it.

To check every byte, hash the model on a copy you trust and load with the result:

```bash
isharded model hash D:\models\model.gguf --out model.hashes
```

```bash
isharded load model.gguf --hashes model.hashes
```

Every node then checks every unit against that list, whether it came from a peer, a mirror, or its own disk. A peer serving altered weights is passed over, and if no source matches, the load stops. The list also covers the model's description: config, tensor shapes, and offsets. It does not cover the tokenizer's source or a draft model. A wrong draft can only slow replies, because the main model checks every guess.

## Reviews and known gaps

The code has had two security reviews, both by an AI model reading the source. The first found one critical and five high findings, and the second, of the signing, ticket, hashing, and merging code added after it, found four high. Those and most medium and low findings are fixed, with regression tests in `tests/test_security.py`. No human has reviewed or penetration tested it, and reviews like these do not show the code is free of other flaws.

- A node in the route sees prompts and can return wrong results. Nothing detects it reading them. Comparing routes with `isharded audit` catches a route that disagrees with others, but not one that behaves whenever it might be watched. Certificates limit who is in the pool. They do not make a hostile member safe.
- An unrevoked node can still claim leadership, publish a route, or cordon nodes through gossip, and a node that holds a verified identity can act for a caller it claims to have checked. It cannot add keys, revoke peers, or learn anyone's key, because in a pool with an authority a key never leaves the node it was given to.
- Tickets can be replayed for up to two minutes by anyone who can read the link, which is why they need a private one.
- The hash list covers weights and the model's description, not the tokenizer or chat template, and it reaches workers through the coordinator, so a dishonest coordinator could leave it out.
- If a node stops answering, the chat page asks before sending your key and conversation to another member, because members declare their own addresses.
- Without a pool authority, the shared token is everyone's identity. Nodes cannot be told apart, and revocation only blocks a name.
- Weight bytes are only checked when you load with `--hashes`.
- Prompts are in memory on every stage, and hidden states are readable by anyone who can read the link if it has no TLS or Tailscale.
- Certificates expire after 825 days and renewal is by hand. Revocation lives in the access list, not a certificate revocation list.
- `ca.key` is not protected by a passphrase, and whoever holds it controls the pool.
- The chat page keeps its key in the browser's local storage.
