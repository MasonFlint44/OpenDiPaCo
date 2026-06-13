# W1 design — NAT traversal / relay tier (libp2p substrate)

Status: **design; no slices landed yet.** W1 (from [viability-roadmap.md](viability-roadmap.md))
removes the biggest *practical* wall to consumer-hardware training: today the
**owner** tier (and owner↔owner replication/gossip) requires public
reachability, but almost no consumer machine behind home NAT/CGNAT is reachable,
so "owner" effectively means "volunteer with a public IP." W1 lets a **NAT'd
consumer machine serve as an owner**, reached through relays. Three operator
calls fixed the approach (§5): **adopt libp2p** (pure py-libp2p, not hivemind)
as the transport+NAT substrate, **end-to-end encryption** of relayed traffic,
and **k≥2 relays** per NAT'd peer.

> **Feasibility spike (done before this doc — go/no-go on the substrate).**
> `py-libp2p 0.6.0` installs cleanly via uv and ships the make-or-break pieces as
> real modules, not stubs: **Circuit Relay v2** (`libp2p.relay.circuit_v2`:
> `CircuitV2Protocol(allow_hop=)`, `CircuitV2Transport.reserve/dial_peer_info`,
> `RelayConfig(enable_hop/enable_client/roles)`), **DCUtR** hole-punching
> (`circuit_v2.dcutr`), **AutoNAT** (`host.autonat`), and **Noise**
> (`security.noise`). Verified by running, not just importing: (1) an Ed25519
> host key derived from a 32-byte seed yields a deterministic peer id — so it
> reconciles with our `PeerIdentity`; (2) two hosts exchanged our bytes over a
> **Noise-secured stream** (`echo:hello-over-libp2p`) — the integration boundary
> works. Verdict: **GO on pure py-libp2p; the go-libp2p sidecar fallback is not
> needed.** Not yet run in the spike (→ W1b acceptance): a full 3-node *relayed*
> round-trip. The one real integration cost: py-libp2p is **trio-async** and our
> stack is threads + a custom event-loop, so a trio↔threads bridge is required
> (D3) — confined to the transport seam.

## 1. Goal and trust model

**Goal.** A peer with no public address participates fully — including as a
module **owner** — by being reachable *through* relays. Public volunteers run
relays (and owners); NAT'd volunteers become reachable owners over them.

**Trust model — extends Phase 4, doesn't loosen it.** A **relay** is a
semi-trusted volunteer that forwards others' traffic. With libp2p Noise applied
**end-to-end through the circuit** (D7), a relay sees only ciphertext and cannot
read or tamper (a tampered frame fails the Noise MAC and drops the connection);
it can still *drop/withhold* (an availability attack), which k≥2 relays (D6)
and the existing quorum/replication machinery bound. Byzantine *owners* remain
defended exactly as in Phase 4 (quorum reads, digest agreement) — relaying
changes *how* an owner is reached, not what it's trusted to compute. A malicious
relay is therefore strictly less powerful than a malicious owner.

## 2. Shape of the result

```
              tracker (rendezvous: signed records now carry libp2p id + multiaddrs)
                 ▲ register / discover relays                ▲
   public peers ─┘  (run CircuitV2 allow_hop = RELAY role)   │
        ▲   ▲ reserve slots (k>=2)                           │
        │   └──────────────── NAT'd owner ──────────────┐    │
        │                     (reachable via /p2p/<relay>/p2p-circuit/p2p/<self>)
   dial through relay (Noise e2e: relay sees only ciphertext; DCUtR may upgrade to direct)
        │                                                │
   any peer ─────────────────────────────────────────────┘
   ── all of it carries our existing wire frames + handlers over a libp2p stream;
      our PeerIdentity / signed records / grants / HRW / reputation are unchanged ──
```

libp2p owns **connection establishment, NAT traversal, and the secure channel**;
everything above the stream — wire codec, `_handle` dispatch, tracker directory,
identity records, owner/scheduler/quorum/robustness logic — is unchanged.

## 3. Decisions

### D1. Substrate: pure py-libp2p, transport+NAT only (spike-verified)
libp2p is the connection/traversal/security layer; our application stack rides
on libp2p **streams** instead of raw sockets. We do **not** adopt libp2p's DHT
discovery, pubsub, or peer-id namespace for the app layer (D4/D11) — only the
parts that solve NAT. go-libp2p-as-sidecar stays the fallback only if a py-libp2p
limitation surfaces in a later slice.

### D2. The seam: a transport abstraction; TCP stays the default
Introduce a small **transport interface** — `rpc(target, msg) -> reply` (client)
and `serve(handler)` (server), framing our wire codec over a connection. Two
implementations: the existing **raw-TCP/reactor** path (today, the default and
the deterministic anchor) and the new **libp2p** path. The wire codec
(`wire.py`) and the message handlers (`_handle`) are shared verbatim; only the
byte pipe differs. `transport: tcp` (default) is bit-identical to today;
`transport: libp2p` opts in.

### D3. Trio↔threads bridge (the integration cost, confined to the seam)
py-libp2p runs on **trio**; our stack is threads + a custom event-loop with no
async. So the libp2p host runs in a **dedicated trio thread**, and the seam
exposes a synchronous facade:
- outbound `rpc()` submits the coroutine to the trio loop via
  `trio.from_thread.run(..., trio_token=token)` and blocks the calling thread for
  the reply;
- inbound stream handlers (trio) dispatch our synchronous `_handle` via
  `trio.to_thread.run_sync` so a handler taking a lock / doing compute never
  stalls the trio loop.
This is the entire async surface — nothing else in the codebase becomes async.

### D4. Identity reconciliation: derive the libp2p key, keep our peer_id
Our `PeerIdentity` is Ed25519 and our app-layer peer id is `sha256(pubkey)`,
baked into HRW placement, reputation, grants, and signed records across Phases
1–4. libp2p's peer id is a different encoding (`12D3KooW…`). Rather than churn
the app layer, we **derive the libp2p host key from our `PeerIdentity`'s bytes**
(`Ed25519PrivateKey.from_bytes` — spike-verified deterministic) and **keep our
`sha256` peer id** for everything above the transport. The **signed directory
record binds the two** (our peer id ↔ libp2p peer id ↔ multiaddrs, all under the
identity's signature), so the transport resolves our peer id → libp2p
PeerInfo to dial, and the binding is self-certifying.

### D5. Addressing & directory: records carry multiaddrs; tracker stays rendezvous
A directory record gains a `libp2p` peer id and **multiaddrs**: a public peer
advertises direct addrs (`/ip4/…/tcp/…`); a NAT'd peer advertises **circuit-relay
addrs** (`/p2p/<relay>/p2p-circuit/p2p/<self>`), one per reserved relay. The
**tracker remains the rendezvous** — peers discover each other and relays from
it (and from owner gossip, Phase 4). We do *not* use libp2p auto-relay/DHT
discovery: tracker-driven selection is controllable and matches the existing
model.

### D6. Relay tier + k≥2 reservations
A public peer offering the `relay` role (the reserved Phase 1 role) runs
`CircuitV2Protocol(allow_hop=True)`. A NAT'd peer discovers relays from the
directory, **reserves on k≥2** of them (`CircuitV2Transport.reserve`), advertises
all the circuit addrs, and **re-homes** when a relay dies or its reservation
lapses. k≥2 means no single relay is a per-peer SPOF or a clean eclipse vector;
the cost is k keepalive connections + k addrs in the record.

### D7. End-to-end security via libp2p Noise (subsumes per-frame envelopes)
When A dials B through relay R, A and B run the **Noise handshake over the
relayed connection**, so R relays only ciphertext and cannot read or alter it
(tampering breaks the MAC). This satisfies the **e2e-encryption** decision with
**no crypto for us to build**, and it **subsumes the per-frame signed-envelope**
work that Phases 1–2 deferred "until relayed data-plane messages need them" — the
secure channel provides transport integrity + confidentiality between endpoints.
Our app-layer signed records/grants stay, but for **authorization**, not
transport integrity. (W1b's acceptance test verifies a relay sees only
ciphertext.)

### D8. NAT'd owners — the payoff
`owner_eligible` today requires `reachability == "public"`. W1 extends it: a
`reachability == "nat"` peer with **≥k valid, live relay reservations** (verified
circuit addrs in its signed record) is **owner-eligible**. HRW placement,
pull-replication, gossip, and digest-audit then operate over relay-reachable
owners unchanged (they dial via the resolved circuit addrs); multi-relay
redundancy gives the owner its own reachability failover, independent of Phase
4's ownership failover.

### D9. DCUtR hole-punch upgrade (the bandwidth optimization, ~free)
With DCUtR enabled, libp2p attempts to **upgrade a relayed connection to a
direct one** after it's established (the classic hole-punch), transparently
falling back to the relay if the NAT won't allow it. We get the bandwidth win of
direct P2P where possible without designing it — it rides on the same connection
the relay bootstrapped.

### D10. Compatibility and the deterministic anchor
`transport: tcp` (default) and the whole central/rendezvous TCP path,
`LocalBackend`, `AsyncScheduler`, `CoordinatorServer`, and the synchronous engine
stay **bit-identical**. libp2p is an **optional `[nat]` dependency extra** (the
default `cpu`/`cu130` installs are untouched), and its tests `importorskip`
libp2p so the baseline suite stays green without the extra. CI gains a job that
installs the extra and runs the libp2p path.

### D11. Explicitly deferred / out of scope
- **libp2p DHT discovery & pubsub** — the tracker + Phase 4 owner gossip already
  cover rendezvous and directory; revisit only if the tracker becomes limiting.
- **QUIC transport** — TCP + relay first; QUIC (present in py-libp2p) is a later
  perf option.
- **Production hardening of py-libp2p at scale** (throughput, Windows, large
  fan-out) — measured during the 0f-systems WAN run, not assumed here.
- **Per-frame signed envelopes** — subsumed by D7; stays retired.

## 4. Implementation slices

Each lands green on its own; the foundation (the seam + bridge) comes before the
relay, and NAT'd owners (the payoff) before the hole-punch optimization.

| Slice | Contents | Key tests |
|---|---|---|
| **W1a** | The transport seam (D2) + trio↔threads bridge (D3) + identity derivation (D4): a `Libp2pTransport` running our wire frames over **direct** Noise streams, behind the `transport:` config seam; TCP default untouched. | Two transports exchange our wire frames (round-trip); libp2p key derives deterministically from a `PeerIdentity`; a sharded cluster runs end-to-end over libp2p **direct** streams in-process; TCP-path parity unbroken. |
| **W1b** | Relay tier (D6) + relayed reachability (D5) + e2e-Noise verification (D7): `relay` role runs `allow_hop`; a NAT'd peer reserves on k relays and advertises circuit addrs; a dialer reaches it through a relay. | The 3-node **relayed round-trip** (the spike's loose end); a relay observes only ciphertext (can't read/tamper); reservation re-home on relay loss. |
| **W1c** | NAT'd owners (D8): `owner_eligible` accepts relay-reachable NAT peers; placement/replication/gossip/digest-audit over circuit addrs; multi-relay failover. | A NAT'd peer serves as an owner (holds a shard, serves fetches, replicates) through relays; killing one of its relays fails over to another with no data loss. |
| **W1d** | DCUtR upgrade (D9) + launch wiring (`transport: libp2p`, `relay`/`nat` roles, `[nat]` extra) + a validation script + docs/roadmap status. | Relayed connection upgrades to direct where NAT permits (falls back otherwise); CLI smoke over libp2p; `validate_*` shows a relayed owner serving; roadmap W1 status updated. |

Rough sizing: W1a L (the bridge + seam is the foundation), W1b L (relay), W1c
M–L, W1d M. XL overall — comparable to Phase 4.

## 5. Operator decisions (resolved)
1. **Substrate (D1): adopt libp2p (pure py-libp2p, not hivemind).** Spike
   confirmed py-libp2p 0.6 has working Circuit Relay v2 + DCUtR + AutoNAT +
   Noise and reconciles with our Ed25519 identity, so no go-libp2p sidecar.
2. **Relayed-traffic security (D7): end-to-end encryption.** Satisfied for free
   by libp2p Noise through the circuit; subsumes the deferred per-frame envelopes.
3. **Relay redundancy (D6): k≥2 relays per NAT'd peer** — no single-relay SPOF
   or eclipse vector, at the cost of k keepalives + k advertised addrs.
