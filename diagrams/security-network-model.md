# Security & Network Model

How OpenDiPaCo nodes find each other, authenticate, move weights/gradients, and
defend the shared module bank. Sources: `schedule/{wire,reactor,identity,tls,tracker,guard,compress,sharded}.py`,
`schedule/p2p.py`, and the design docs under `docs/` (W1 NAT, W2 bandwidth,
Phase 2–4).

## 1. Roles & topology

```mermaid
graph TB
    subgraph CTRL["Control plane (rendezvous)"]
        TR["Tracker<br/><i>signed peer directory</i><br/>TTL liveness · enrollment<br/>bootstrap-from-cache"]
        RL["Relay(s)<br/><i>libp2p Circuit Relay v2</i><br/>reaches NAT'd owners (k≥2)"]
    end

    subgraph DATA["Data / training plane"]
        SCH["Scheduler<br/><i>mints + signs commit grants</i><br/>leases · task sizing (W5)"]
        PS1["Parameter Server shard A<br/><i>holds part of module bank</i>"]
        PS2["Parameter Server shard B<br/><i>holds part of module bank</i>"]
        ING["Ingest<br/><i>shard recipe / token counts</i>"]
    end

    subgraph WORKERS["Workers (volunteers — dial-out only / NAT)"]
        W1["Worker · path i<br/>inner AdamW steps"]
        W2["Worker · path j"]
        W3["Worker · path k"]
    end

    W1 & W2 & W3 -. "register signed record<br/>fetch directory" .-> TR
    SCH -. register .-> TR
    PS1 & PS2 -. register .-> TR

    SCH -- "lease (unique token)" --> W1
    W1 -- "pull weights · push pseudo-grad+grant" --> PS1
    W1 -- "push private weights+grant" --> PS2
    SCH -- "grant_key / scheduler_pub<br/>(sign grants)" --> PS1 & PS2

    ING -- "materialize_shard recipe" --> W1

    W2 -. "NAT'd owner reached via" .-> RL
    RL -. "relayed → DCUtR hole-punch" .-> W3

    classDef ctrl fill:#fde68a,stroke:#b45309,color:#000;
    classDef data fill:#bfdbfe,stroke:#1e40af,color:#000;
    classDef wrk fill:#bbf7d0,stroke:#15803d,color:#000;
    class TR,RL ctrl;
    class SCH,PS1,PS2,ING data;
    class W1,W2,W3 wrk;
```

`world_size` must divide `num_paths`. The scheduler can be removed entirely in
`schedule.mode: decentralized` (Phase 4): each path's **primary owner** mints
its own grants and the tracker becomes a pure bootstrap seed.

## 2. Identity, authentication & channel security

```mermaid
flowchart TB
    subgraph ID["Identity (identity.py)"]
        K["Ed25519 keypair<br/>PEM private key (mode 0600)"]
        PID["peer_id = sha256(pubkey)<br/><i>self-derived, collision-resistant</i>"]
        K --> PID
    end

    subgraph AUTHN["Client→Server auth handshake (wire.py / reactor.py)"]
        direction TB
        N["Server sends 32-byte nonce<br/>(challenge)"]
        HMAC["<b>HMAC mode</b><br/>HMAC-SHA256(shared_secret, nonce)<br/>constant-time verify · key rotation · per-worker keys"]
        SIG["<b>Identity mode</b><br/>Ed25519 sign(AUTH_CONTEXT‖nonce)<br/>server checks pubkey ∈ admitted_peers"]
        N --> HMAC
        N --> SIG
        HMAC & SIG --> WEL["Server → welcome"]
    end

    subgraph CHAN["Channel confidentiality (tls.py)"]
        PT["Default: PLAINTEXT<br/><i>auth proves possession, does NOT encrypt</i><br/>trusted net / SSH tunnel only"]
        TLS["TLS 1.2+ context<br/>server cert/key · optional CA + mutual TLS<br/>(require_client_cert) · self-signed for dev"]
        NOISE["libp2p Noise stream (p2p.py)<br/>host key derived from SAME Ed25519 seed<br/>relay sees only ciphertext"]
    end

    PID -.-> SIG
    AUTHN --> CHAN

    note1["⚠ Handshake authenticates CLIENT→SERVER only.<br/>Server authenticity + channel binding come from TLS/Noise.<br/>Without them a MITM can relay the challenge response."]
    AUTHN -.-> note1

    classDef warn fill:#fee2e2,stroke:#b91c1c,color:#000;
    class note1,PT warn;
```

Key facts:
- **peer_id = `sha256(raw pubkey)`** — `verify_record` rejects a record whose
  `peer_id` isn't honestly derived from its embedded `pub`, so you can't sign
  someone else's id with your own key.
- HMAC keys normalize to a *set* on the server (`acceptable_keys`): rotate by
  listing old+new; revoke by dropping a key. The node's own `auth_key` doubles
  as its client identity.
- TLS is **off by default**. The wire format already unpickles nothing, so the
  threat TLS closes is **confidentiality** (on-path reading of weights/grads),
  not RCE.

## 3. Self-certifying records & the rendezvous tracker

```mermaid
flowchart LR
    subgraph SR["Signed record (identity.sign_record)"]
        BODY["body: kind, reachability,<br/>addr, roles, capabilities, issued_at"]
        CANON["canonical JSON<br/>(sort_keys, RECORD_CONTEXT prefix)"]
        S["+ peer_id + pub + sig"]
        BODY --> CANON --> S
    end

    subgraph TRK["Tracker (tracker.py)"]
        ENR{"enrollment?"}
        OPEN["open_enrollment:<br/>any valid sig accepted"]
        CLOSED["closed:<br/>only enrolled pubkeys<br/>enroll()/expel()"]
        DIR["Directory<br/>ordered by signed issued_at<br/>TTL liveness · tombstones on deregister"]
        ENR --> OPEN & CLOSED --> DIR
    end

    S -->|register| ENR
    DIR -->|"anyone fetches"| CACHE["Client cache"]
    CACHE -->|"import into fresh tracker"| TRK

    note["Records verify on their own → trustworthy when relayed.<br/>Stale copy can't displace newer (issued_at).<br/>Losing the tracker degrades, doesn't halt: bootstrap from any cache."]
    DIR -.-> note

    REACH["Reachability tiers:<br/>public = advertises addr (hosts P2P plane)<br/>nat = dial-out only (+ optional /p2p-circuit relay addrs)"]
    DIR -.-> REACH
```

## 4. Data movement: lease → train → grant → commit

The core transport invariant: **a PS push requires the scheduler's single-use
commit grant**, and a **lease token** fences zombie workers.

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker (path i)
    participant S as Scheduler
    participant P as Parameter Server(s)

    Note over W,P: handshake (§2) + optional TLS already done on every connection

    W->>S: request task
    S->>S: size task — W5 shrinks batch then inner_steps for slow workers, parks if too slow
    S-->>W: LEASE [path, unique token, task size]

    W->>P: pull current module weights (version = (epoch, counter))
    P-->>W: weights (bf16 if compress≠none)

    loop inner_steps × AdamW on own data shard
        W->>W: train (optimizer state NEVER leaves the worker)
    end

    W->>S: submit (echo lease token) → request commit
    S->>S: verify token, sign grant
    Note right of S: make_grant(path, keys, weight, token)<br/>HMAC(grant_key) OR Ed25519(scheduler identity)
    S-->>W: GRANT [weight, allowed keys, signature]

    W->>P: PUSH pseudo-grad (shared) + private weights + GRANT
    P->>P: verify_grant (grant_key / scheduler_pub pin)
    P->>P: guard.py — reject non-finite, optional max_update_norm clip
    P->>P: apply: weighted Σ pseudo-grads → outer Nesterov → bank
    P-->>W: ack

    Note over S,P: heartbeats echo the lease token — a stale/zombie<br/>worker's push is rejected (token fenced)
```

Invariants enforced on this path:
- **Lease token** — unique per lease; echoed on submit/nack/heartbeat. Fences a
  zombie worker whose lease was reassigned.
- **Commit grant** — single-use, carries the allowed keys + weight.
  - `grant_key` set → **HMAC-SHA256** signed (kept secret from workers).
  - scheduler `identity=` + servers pin `scheduler_pub=` → **Ed25519** signed;
    this **refuses HMAC/unsigned grants outright** (no downgrade).
  - `schedule.mode: decentralized` → grant signed by the path's **primary
    owner**; co-owners verify the signer against the epoch record (`grant_signed_by`).
- **Optimizer state never crosses the wire** (`bytes_opt` metric stays 0).
- **Version = `(epoch, counter)`** must always identify identical bytes: owner
  banks built with a shared `bank_seed` so `(0,0)` matches everywhere;
  replication pulls ship exact uncompressed state (never bf16 a replication payload).

## 5. Server-side defense of the bank

```mermaid
flowchart TB
    IN["Authenticated, granted contribution"] --> WIRE

    subgraph WIRE["1 · Wire decode (wire.py)"]
        NP["Pickle-free typed codec<br/>JSON structure + raw tensor bytes"]
        AL["dtype ALLOWLIST · declared shape<br/>tensors reconstructed, never executed"]
        CAP["size cap (max_bytes, default 4 GiB)<br/>bounds a garbage length prefix"]
    end

    WIRE --> GUARD

    subgraph GUARD["2 · Sanity guard (guard.py) — always on"]
        FIN["Reject ANY non-finite<br/>(pseudo-grad / private weights / loss)<br/>bounds bit-flips on consumer HW"]
        CLIP["Optional L2 norm CLIP (max_update_norm)<br/>bounds single contribution's influence"]
    end

    GUARD --> ROB

    subgraph ROB["3 · Adversarial (Phase 3, robustness.mode: on)"]
        AGG["Robust aggregation across sharing paths<br/>(aggregate.py)"]
        REP["Version-pinned redundant execution<br/>→ per-peer reputation (reputation.py)<br/>gates owner eligibility · lease priority · rate limit"]
        RL2["Rate limiting (ratelimit.py)"]
        PROP["Private-module proposal-gating"]
    end

    ROB --> APPLY["Apply to module bank"]

    note["Layered: codec bounds RCE/DoS surface · guard bounds DAMAGE<br/>(faulty hardware) · Phase 3 bounds MALICE (finite, norm-bounded,<br/>wrong-direction gradients)."]
    ROB -.-> note

    classDef def fill:#e9d5ff,stroke:#7c3aed,color:#000;
    class WIRE,GUARD,ROB def;
```

## 6. Wire compression (changes bytes, not trust)

All off by default; the "off" path is byte-identical. Self-describing payloads,
decode boundary Byzantine-hardened.

```mermaid
flowchart LR
    subgraph DOWN["Weights DOWN (server→worker)"]
        D1["bf16 cast (~2×)"]
        D2["delta: int8/int4 current−keyframe<br/>vs owner version ring + full fallback (W2a)"]
    end
    subgraph UP["Pseudo-grads UP (worker→server)"]
        U1["int8 symmetric per-tensor (~4×)<br/>+ worker-side ERROR FEEDBACK"]
        U2["top-k per-row (up_density<1)<br/>error-feed dropped mass (W2b)"]
        U3["int4 per-group (W2c)"]
    end
    LEVER["Free lever: traffic ∝ 1/inner_steps"]

    note["Quantized tensor travels as {q:int8, s:scale}.<br/>Replication payloads are EXEMPT — always exact uncompressed."]
    UP -.-> note
```

## 7. libp2p / NAT traversal (W1, `transport.kind: libp2p`)

```mermaid
flowchart TB
    PI["PeerIdentity (Ed25519 seed)"] -->|"derive host key (D4)"| HOST["libp2p host<br/>id 12D3KooW…  (app id stays sha256(pubkey))"]
    HOST --> NOISE["Noise-secured stream<br/>our wire frames inside"]

    subgraph NAT["NAT'd owner reachability"]
        REL["Circuit Relay v2 (k≥2 relays)<br/>advertise /p2p-circuit addrs"]
        DC["DCUtR best-effort<br/>relayed → direct hole-punch"]
        REL --> DC
    end

    NOISE --> NAT
    BRIDGE["trio ↔ threads bridge (Libp2pTransport)<br/>synchronous facade; _handle off the trio loop"]
    HOST --- BRIDGE

    ANCHOR["TCP reactor stays the byte-identical anchor;<br/>serve_over_libp2p runs a PARALLEL host"]
    HOST -.-> ANCHOR
```

---

### Threat-model summary

| Layer | Mechanism | Defends against |
|---|---|---|
| Serialization | pickle-free typed codec, dtype allowlist, size cap | RCE via deserialization, oversized-prefix DoS |
| AuthN | HMAC challenge **or** Ed25519 challenge (`admitted_peers`) | unauthenticated clients |
| Channel | TLS 1.2+ (opt mutual) / libp2p Noise | on-path eavesdropping, server impersonation (with verify) |
| Directory | self-certifying signed records, TTL, tombstones | forged/replayed/stale peer records, tracker as SPOF |
| Authorization | lease token (fence) + single-use commit grant (HMAC/Ed25519/owner) | zombie workers, unauthorized PS writes, grant downgrade |
| Bank integrity | non-finite reject (always) + optional norm clip | faulty-hardware poisoning, oversized contributions |
| Adversarial | robust aggregation + reputation + rate limit + proposal-gating | Byzantine wrong-direction gradients, Sybil influence |
| Secret hygiene | optimizer state never shipped; PEM key 0600; `grant_key` secret from workers | state leakage, key theft |
