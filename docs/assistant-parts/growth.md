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
