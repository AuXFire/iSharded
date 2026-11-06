# ShardMesh security

ShardMesh runs code and holds prompts on every machine in a pool, so a pool is only as trustworthy as its least trusted node. Read this before connecting machines you do not fully control.

## Trust model

A node in the route sees prompts and replies in plain text, as hidden states or tokens, and can return anything it likes. A node you cannot trust should not be in your pool. Certificates and revocation limit what a stolen credential can do and let you throw a node out. They do not make a hostile node safe to include.

## Who owns a pool

ShardMesh is decentralized in how it runs, not in who owns it. No server is special, any node can coordinate, and losing any machine loses nothing but what it was running. But a pool has an owner: whoever holds the authority key, or the shared token in a pool without one. That person decides who joins, who is thrown out, and who may load models.

This suits a pool whose members trust each other. It is not a public mesh. A public mesh, where anyone can join, would need answers ShardMesh does not have today: a way to detect wrong results, a way to stop one person running thousands of nodes, and a clear warning that node operators can read prompts.

## Who can do what

| Caller | Credential | Can do |
|---|---|---|
| Node, with a pool authority | Certificate, over TLS or as a ticket | Node routes only: gossip, registration, stage traffic |
| Node, without one | Pool token | Everything, including the node protocol |
| Admin | API key, role `admin` | Load models, rebalance, cordon, manage keys, revoke nodes, read metrics, and all user actions |
| User | API key, role `user` | Chat, completions, read status |

A node starts refusing to serve a network if it would be exposed:

- With no credentials, it will not listen beyond loopback. Run `shardmesh pool init`, or pass `--open` if the network itself is the protection.
- With credentials but no encryption, it will not advertise an ordinary LAN address, because every gossip round carries the pool token. Use TLS or a Tailscale address, or pass `--trusted-network` if nobody else can read that network.

## Setting up a pool

For a pool on Tailscale or a closed network, a shared token is enough:

```bash
shardmesh pool init --dir pool-secrets
```

Copy `pool.token` to each node as `%LOCALAPPDATA%\ShardMesh\pool.token` (Windows) or `~/.shardmesh/pool.token`. Anyone holding it can act as any node.

For machines you may need to throw out individually, or remotes across the internet, give each node a certificate. Every node then needs the `pool` extra (`pip install shardmesh[pool]`).

```bash
shardmesh pool add-node --dir pool-secrets node-b --address 10.0.0.6
```

Keep `ca.key` off the nodes. The name in the certificate must match `--node-id`, because a node may only speak for itself and its own workers.

A node proves its name in one of two ways, using the same certificate:

| | Use it when | How |
|---|---|---|
| Tickets | The link is already private, as on Tailscale, or antivirus intercepts TLS | `--ca-file ca.pem --identity-cert node-b.pem --identity-key node-b.key` |
| Mutual TLS | The link needs encrypting, and nothing intercepts TLS | `--ca-file ca.pem --tls-cert node-b.pem --tls-key node-b.key` |

A ticket is a short signed statement in an ordinary header. It names the node it is for and expires within five minutes, so a peer that receives one cannot use it elsewhere. Tickets encrypt nothing, so a node using them on an ordinary LAN refuses to start without `--trusted-network`. Clocks must agree to within a minute.

With a pool authority the pool token counts for nothing, on control planes and workers alike, so do not copy it to nodes.

### Antivirus on nodes

Many antivirus products intercept TLS on every port, even on loopback, and replace the certificates in it. Nodes only trust the pool authority, so this fails closed: mutual TLS stops working rather than becoming insecure. AVG does this on the development laptop. There are three ways to run such a node:

- Use Tailscale and tickets. Nothing depends on TLS, so antivirus has nothing to intercept. This is what the two-machine test pool runs, with AVG on.
- Exclude the pool's ports or `python.exe` from HTTPS scanning, and use mutual TLS.
- On a private network, use tickets with `--trusted-network`, accepting that traffic is unencrypted.

## Managing keys and nodes

With a pool authority, keys and revocations are signed with `ca.key` and sent to any node, and gossip carries them to the rest. Nodes ignore anything unsigned, so a compromised node cannot add a key, remove one, or revoke a peer.

```bash
shardmesh key create alice --role user --requests-per-minute 60 --dir pool-secrets --url http://10.0.0.5:8000
```

```bash
shardmesh key remove alice --dir pool-secrets --url http://10.0.0.5:8000
```

```bash
shardmesh pool revoke node-b --dir pool-secrets --url http://10.0.0.5:8000
```

A key is shown once and stored as a SHA-256 hash. `/admin/keys` and `/admin/nodes/{name}/revoke` refuse these changes in such a pool, and say to use these commands.

Without a pool authority, an admin key or the pool token changes these through any node: `POST /admin/keys`, `DELETE /admin/keys/{name}`, `POST /admin/nodes/{name}/revoke`. To seed a key before a pool starts, in either kind of pool, use `shardmesh key create root --role admin --file access.json`, and give each node the file with `--keys-file`. A key file is trusted by the node that reads it, and nothing else.

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

Requests above 16 MiB are refused from their headers before any of the body is read, and bodies must declare a length. Stage traffic may reach 1 GiB, and only `/internal/forward` and the bandwidth probe accept that much.

A request whose content a stage cannot process fails with an error and costs nobody their place in the route. Only losing contact with a stage can quarantine it, so a user cannot empty the pool one request at a time.

Peers name model sources, so a worker in a secured pool only loads local models from its `--gguf-dir` folders, its cache, and any `--model-dir`. File names from peers cannot leave a model's folder, and a draft model can only be a `.gguf`.

## Audit log

Each node records who loaded models, changed the pool, and asked for replies: time, node, caller, path, status, and client address. It never records prompts or replies. Every node that handles a request records it, so a forwarded request appears twice, and you need the files from every node for the full picture.

## Models

GGUF files are data, but they are parsed by native code, in a separate stage process with the node's privileges. Load files only from sources you trust. Safetensors weights avoid pickle, and pickle checkpoints are refused. Chat templates run in a Jinja sandbox, and need Jinja2 3.1.6 or newer.

Fetched ranges are checked for size, and pinned to a commit or a content fingerprint. The fingerprint covers the file's first megabyte and its size, which identifies a file but would not detect tampering deeper in it.

To check every byte, hash the model on a copy you trust and load with the result:

```bash
shardmesh model hash D:\models\model.gguf --out model.hashes
```

```bash
shardmesh load model.gguf --hashes model.hashes
```

Every node then checks every unit against that list, whether it came from a peer, a mirror, or its own disk. A peer serving altered weights is passed over, and if no source matches, the load stops. The list also covers the model's description: config, tensor shapes, and offsets. It does not cover the tokenizer's source or a draft model. A wrong draft can only slow replies, because the main model checks every guess.

## Reviews and known gaps

The code has had two security reviews, both by an AI model reading the source. The first found one critical and five high findings, and the second, of the signing, ticket, hashing, and merging code added after it, found four high. Those and most medium and low findings are fixed, with regression tests in `tests/test_security.py`. No human has reviewed or penetration tested it, and reviews like these do not show the code is free of other flaws.

- A node in the route sees prompts and can return wrong results, and nothing detects it. Certificates limit who is in the pool. They do not make a hostile member safe.
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
