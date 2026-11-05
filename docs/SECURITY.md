# ShardMesh security

ShardMesh runs code and holds prompts on every machine in a pool, so a pool is only as trustworthy as its least trusted node. Read this before connecting machines you do not fully control.

## Trust model

A node in the route sees prompts and replies in plain text, as hidden states or tokens, and can return anything it likes. A node you cannot trust should not be in your pool. Certificates and revocation limit what a stolen credential can do and let you throw a node out. They do not make a hostile node safe to include.

## Who can do what

| Caller | Credential | Can do |
|---|---|---|
| Node, with a pool authority | Certificate | Node routes only: gossip, registration, stage traffic |
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

For machines you may need to throw out individually, or remotes across the internet, give each node a certificate. This needs the `pool` extra (`pip install shardmesh[pool]`) on the machine that issues them, and nothing on the nodes:

```bash
shardmesh pool add-node --dir pool-secrets node-b --address 10.0.0.6
```

```bash
shardmesh node --node-id node-b --tls-cert node-b.pem --tls-key node-b.key --ca-file ca.pem --keys-file access.json --join https://10.0.0.5:8000
```

Keep `ca.key` off the nodes. The name in the certificate must match `--node-id`, because a node may only speak for itself and its own workers.

With a pool authority the pool token counts for nothing, on control planes and workers alike, so do not copy it to nodes. Admins use API keys. Make the first one before starting the pool, and give every node the file once:

```bash
shardmesh key create root --role admin --file access.json
```

After that, keys are managed through any node and gossip carries them to the rest:

```bash
shardmesh key create alice --role user --requests-per-minute 60 --file keys.json
```

A key is shown once and stored as a SHA-256 hash. `POST /admin/keys` does the same on a running pool, and `DELETE /admin/keys/{name}` removes one.

## Revoking a node

`POST /admin/nodes/{name}/revoke` with an admin key stops every node accepting that certificate, takes its workers out of the route, and stops gossip with it. A revoked name cannot register new workers.

Revocation is permanent for that name, and so is removing a key. Both are kept for good and merged entry by entry, so neither a network split nor a forged document can undo them. To bring a machine back, issue it a certificate under a new name.

If a compromised node might have added an API key before it was revoked, review `GET /admin/keys` afterwards. See the gaps below.

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

## Reviews and known gaps

The code has had one security review, by an AI model reading the source, which found one critical and five high findings. Those and the medium and low findings are fixed, with a regression test for each in `tests/test_security.py`. The attacks on the critical and certificate findings were also repeated against real control planes over real mutual TLS. No human has reviewed or penetration tested it, and a review like this does not show the code is free of other flaws.

- An unrevoked node is still trusted by the pool. It can claim leadership, publish a route, or add an API key through gossip, because access documents are not signed. Signing them with a key held by admins would close this.
- Without a pool authority, the shared token is everyone's identity. Nodes cannot be told apart, and revocation only blocks a name.
- A node can return wrong results, and nothing detects it.
- Weight units are not verified by hash, so a peer serving a file can serve wrong bytes.
- Prompts are in memory on every stage, and hidden states are readable by anyone who can read the link if it has no TLS or Tailscale.
- Certificates expire after 825 days and renewal is by hand. There is no certificate revocation list, so revocation lives in the pool's access document.
- The chat page keeps its key in the browser's local storage.
