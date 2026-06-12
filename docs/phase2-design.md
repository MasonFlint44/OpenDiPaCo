# Phase 2 design — distribute the module bank (PS → replicated owner peers)

Status: **slices 2a + 2b landed.** 2a: `schedule/ownership.py` (HRW + epoch
records), Ed25519 grants beside HMAC, scheduler `epoch` RPC + `publish_epoch`,
tracker epoch cache with signer pinning. 2b: `(epoch, counter)` version pairs,
dynamic owner key sets with the `syncing → active` lifecycle
(`apply_epoch`/`bootstrap=`), pull replication (`include_state`, owner-session
gated, momentum + private counters, **exact bytes** — see the D4 amendment),
rendezvous-mode scheduler routing, worker fetch-any/push-primary with replica
fallback, and seeded bank builds (`bank_seed`, see D5 amendment). 2c:
`EpochManager` (grace/flap/rate-limit hysteresis) + `Scheduler.watch_tracker`
(tracker liveness → signed epoch bumps, cached back to the tracker), owners
poll the scheduler for epochs from their replication loop
(`scheduler_addr=`), **bootstrap-flagged first epoch** (a fresh cluster's
owners boot-serve their seeded banks exactly once — without it a polled first
epoch deadlocks with everyone syncing), zombie write fencing + one-epoch lame
ducks that double as **fallback pull sources** (a wholesale remap can still
cold-sync from last epoch's owners), and `start_tracker_heartbeat`. 2d:
per-key checkpoint files (`module_<hash>.pt`, remap-proof, skip-unchanged;
warm restarts resume at the saved version and **delta-sync, never cold-sync**),
restart reconciliation (resumed keys stay syncing until an **exhaustive**
pull adopts the max across all replicas — closing 2c's restart-over-lost-disk
hole; owner-authorized pulls see syncing keys so a full-cluster restart
reconciles instead of deadlocking; sole owners self-activate), the
scheduler-signed **recovery manifest** + readiness gate (`fit(resume=True)`
refuses to serve until live owners hold ≥ the manifest version for every
key), and launch wiring (`ownership` config section, rendezvous
scheduler/owner roles with tracker heartbeats, `run_local` rendezvous smoke).
**All four slices have landed — Phase 2 is complete.** Still open within the
phase's scope notes: replication-path compression and delta-encoding (D10
bandwidth items), per-epoch-0 number reuse across scheduler restarts is
disambiguated only by `issued_at`, and Byzantine behavior everywhere is
Phase 3. This expands the Phase 2 sketch in
[internet-scale-plan.md](internet-scale-plan.md) into concrete decisions before
code. Each decision below (D1–D10) states the options considered and the
recommendation; "open questions" at the end are the ones genuinely worth a
human call.

## 1. Goal and trust model

Today the bank lives on K statically-assigned `ParameterServer`s
(`assign_shards`: round-robin over sorted keys, fixed at launch). Lose one PS
and its keys are gone until an operator restarts it from a checkpoint; add a
node and nothing changes. Phase 2 makes module ownership **dynamic and
replicated**: each module key maps to *k* owner peers drawn from the live
`public` tier, ownership re-maps automatically on churn, and weight-fetch /
gradient-push bandwidth spreads across all owners (finding §1.8).

**Trust model — unchanged from Phase 0/1.** Owners are *trusted but
unreliable*: enrolled peers that may crash, disappear, or lag, but are not
adversarial. Byzantine owners (lying about versions, serving poisoned
weights) are Phase 3 scope — robust aggregation and reputation gate owner
eligibility there. Phase 2 buys availability and bandwidth, not
adversary-resistance, and the doc says so wherever the two could be confused.

## 2. Shape of the result

```
                 tracker (directory: signed peer records + cached epoch record)
                    ▲  register/heartbeat        ▲ publish epoch
   owners (public) ─┘                            │
        ▲  ▲  pull-replicate (owner↔owner)   scheduler (leases, grants, epochs)
        │  └────────────────────────────┐       ▲
   fetch any replica / push primary     │       │ lease / commit → grant
        │                               │       │
   workers (nat, dial-out only) ────────┴───────┘
```

The `Scheduler` stays the single control-plane authority for the run (Phase 4
is where that decentralizes). What changes is *where weights live* and *how
workers find them*.

## 3. Decisions

### D1. Ownership authority: scheduler-signed epochs (not view-based rendezvous)

Pure rendezvous hashing over "the live peer set" is only deterministic if
everyone agrees on the set — with each peer reading the tracker at different
times, two workers can disagree about who owns a key (split-brain writes).

**Decision:** membership changes are batched into **epochs**. The scheduler
watches the tracker directory, and when the owner-eligible set changes
(subject to hysteresis, D5) it emits an **epoch record**:

```json
{"kind": "epoch", "epoch": 7, "owners": [{"peer_id": "...", "addr": ["h", p]}, ...],
 "k": 3, "issued_at": ...}
```

signed with the scheduler's `PeerIdentity` (`sign_record` — the Phase 1
self-certifying record machinery, a new `kind`). Everyone derives the same
key→owner mapping from the same epoch record (D2). Distribution:

- **workers** never see the epoch directly — the scheduler already ships
  per-task `routing`; it keeps doing that, computed from the current epoch
  (worker protocol stays almost unchanged, D8);
- **owners** poll the scheduler with a small `{"type": "epoch"}` RPC (and
  receive the current epoch in the reply to any push/replication refusal, so
  a remapped owner learns immediately);
- the **tracker** caches the latest epoch record as a convenience for
  bootstrap — it's self-certifying, so relaying it is safe. The tracker is
  *not* the authority; eligibility lives where grants already live.

Rejected alternative: tracker-computed epochs. It would split run authority
across two nodes (tracker decides ownership, scheduler decides staleness) and
gives the tracker a signing role it doesn't need. The tracker stays a pure
directory.

### D2. Placement: highest-random-weight (rendezvous) hashing, primary = rank 0

For each module key, score every owner in the epoch by
`sha256(epoch_salt ‖ key ‖ peer_id)` and take the top *k* (default **k = 3**).
Rank 0 is the **primary**, ranks 1..k−1 are backups.

- HRW's minimal-disruption property is the point: an owner leaving moves only
  its own keys; an owner joining steals ~1/n of each key's probability. No
  ring maintenance, no virtual nodes needed at our key counts (typically
  dozens of module keys).
- Deterministic rank order doubles as the **succession order** — no election
  protocol; on primary loss the next epoch's rank 0 for that key is (almost
  always) the old rank 1, which already holds a recent replica (D5).
- Eligibility filter (input to the epoch): registered in the tracker with
  `reachability="public"`, role offer `"owner"`, enrolled, and alive (TTL).
  Phase 3 adds reputation to this filter; the seam is one predicate.
- `assign_shards` (static round-robin) **stays** as the default for trusted
  clusters (D9); rendezvous is opt-in via config.

### D3. Write path: primary-only writes; grants become Ed25519-signed

All pushes for a key go to its **primary**; backups refuse pushes (with the
current epoch in the refusal). Reason: the outer optimizer is stateful
(Nesterov momentum), so replicas stay consistent only if they apply the same
update sequence in the same order — letting workers push to all k replicas
independently would interleave concurrent pushes differently per replica and
silently diverge them. The primary serializes; backups copy state (D4).

**Grant signing changes.** Today grants are HMAC-signed with a `grant_key`
shared between scheduler and PSs — fine when PSs are operator-run, untenable
when any volunteer can become an owner (every owner could then *forge*
grants). Phase 2 adds **Ed25519 grants**: the scheduler signs the grant
payload with its `PeerIdentity`; owners verify against the scheduler's public
key, which is public information (carried in the epoch record). HMAC
`grant_key` mode remains for trusted clusters; the two are config-selected,
and `verify_grant` grows a signature branch.

Owner-side validation (guard.py: non-finite rejection, norm clipping,
allowed-keys from the grant, single-use grant tokens) is unchanged — it just
runs on the primary.

### D4. Replication: pull-based delta sync, reusing the fetch protocol

Options considered:

1. *Push log-shipping* — primary streams each (weight, grad) in order;
   backups replay. Bit-identical replicas, tiny bandwidth (int8 grads), but
   needs per-backup positions, ordered delivery, and a full-resync path on
   any gap. Most machinery for the least slack.
2. *Push snapshots* — primary sends state after each push. Idempotent but the
   primary now tracks backup liveness and buffers for slow backups.
3. **Pull snapshots (chosen)** — each backup periodically runs the *existing
   fetch protocol* against the primary: `have: {key: version}`, primary
   returns only stale keys. Extended with `include_state: true`, which adds
   the **outer-optimizer momentum** to the reply and is honored only when the
   requesting session's `peer_id` (the reactor already tracks it) is an owner
   of that key in the current epoch.

Pull wins on simplicity and reuse: it is naturally coalescing (poll interval
= loss window), idempotent (version-gated, last-writer-wins), needs zero
primary-side bookkeeping, survives reconnects with no special path, and —
decisively — **cold-sync is the same mechanism** (D6). The replication
interval (`replicate_interval`, default ~10 s) is an explicit
durability/bandwidth dial: on failover you lose at most the pushes accepted
within one interval. Log-shipping stays on the books as a bandwidth
optimization if measurement demands it.

**Momentum replicates with weights.** A promoted backup with weights but
stale momentum changes the outer dynamics silently. Replication payload per
key = weights + outer momentum + version. *(Amended during 2b: replication
ships **exact bytes**, never wire-compressed — bf16 round-trips are lossy, and
a version stamp must always identify identical content or the whole gate
lies. Replication-path compression is deferred with the other bandwidth
optimizations in D10.)*

**Private modules** get a version counter too (today `_versions` covers only
shared keys; private stores bump nothing, so a puller can't tell stale from
current). A per-key store counter, bumped on every private store, makes them
pull-syncable exactly like shared keys. Private state still never leaves the
owner set + the leasing worker.

### D5. Failover and churn: epoch bump + promotion + two-part versions

- **Detection:** tracker TTL expiry is the membership signal; failed pushes
  reported by workers (or by the scheduler's own epoch poll) can prompt an
  early liveness check but don't unilaterally evict.
- **Hysteresis:** an owner must be missing for a grace period
  (`owner_grace`, default ≥ 2× tracker TTL) before an epoch bump; bumps are
  rate-limited (`min_epoch_interval`); multiple changes batch into one bump.
  A flapping owner must not thrash ownership.
- **Promotion:** new epoch → HRW re-ranks → old rank-1 is the new rank-0,
  already holding a replica ≤ one `replicate_interval` stale. It serves from
  its highest held version. Pushes in flight against the old primary fail and
  the contribution is dropped — the same semantics as a rejected commit
  today, and the error-feedback residual was never committed for it.
- **Version ambiguity fix:** after promotion, the new primary may be behind
  the dead one (the unreplicated window). If versions were a bare counter, it
  could re-issue version 100 with different bytes, and worker `have`-caches
  would silently keep the wrong tensor. **Versions become `(epoch, counter)`
  pairs**; the counter resets per epoch and the pair is totally ordered.
  Every `have`/`versions` field and the checkpoint format carries the pair.
  (This is the one wire-visible breaking change; the static-ownership path
  encodes it as `(0, counter)` so the formats stay uniform.)
- **Accepted loss window:** failover discards ≤ `replicate_interval` worth of
  accepted pushes on the failed key's shard. That is the same class of
  perturbation as the async staleness dynamics — documented, dialable, and
  validated like every other dynamics change (§1.4 discipline).
- *(Amended during 2b: **seeded bank builds**. `build_module_bank` used the
  ambient RNG, so "every owner boots the identical `(0, 0)` bank" was false —
  two owners' fresh builds differed, and a never-written key would silently
  serve different bytes per replica. `build_module_bank(config, seed=…)` is
  now a pure function of (config, seed) and owners pass a shared `bank_seed`
  (default 0); version `(0, 0)` therefore means the same bytes on every
  replica, which is the premise the whole version gate rests on.)*

### D6. Owner lifecycle and cold-sync

Per-key owner state machine: `syncing → active`. A peer that gains a key in a
new epoch (join or remap) pulls that key from the current replicas (same D4
fetch, any replica will do for catch-up; final catch-up pull from the
primary) and only then marks it `active`. Fetch/push for a `syncing` key gets
`{"type": "not_ready"}` and the worker falls back to the next replica in its
routing list (D8). An owner that *loses* a key keeps serving fetches for it
until the next epoch's owners are active (lame-duck), then drops it; its
on-disk copy stays as a warm-start cache (D7).

### D7. Persistence and the signed manifest

- **Per-key checkpoint files** replace the per-shard blob: today's
  `save_shard` writes one file named by the hash of the *key set*, which
  dynamic ownership invalidates on every remap. New layout:
  `dir/module_<sha256(key)[:16]>.pt` containing weights + momentum +
  `(epoch, counter)` version, written atomically as now. An owner restarted
  (or re-acquiring a key it held before) loads the local file and
  delta-syncs forward instead of cold-syncing from zero.
- **Manifest:** on each cluster checkpoint the scheduler collects
  `key → (version, owner peer_ids)` from the owners' checkpoint acks, signs
  the manifest with its identity, and writes it alongside `scheduler.pt`.
  Recovery = relaunch owners (each loads what it has locally), relaunch the
  scheduler with the manifest; the scheduler refuses to serve until, for
  every key, some active owner holds ≥ the manifest version.
- **Honesty about the invariant:** the *engine* checkpoint invariant
  (bit-for-bit resumable) holds per node. A *cluster* recovery point under
  async + replication is consistent-within-bounded-staleness, not bit-exact
  across keys — same as the existing sharded cluster checkpoint, now stated
  explicitly here and in the plan doc.

### D8. Worker protocol changes (small by design)

- Task `routing` becomes `key → [[addr, peer_id], …]` — the k replicas in
  rank order, primary first — plus the task carries `epoch`.
- **Fetch from any replica** (prefer an already-connected owner, else try in
  rank order); replies carry `(epoch, counter)` versions so the worker's
  `have` cache stays correct across failovers. Reading a slightly-stale
  backup is equivalent to a small extra staleness — inside the existing
  async-dynamics envelope, and it's what spreads the fetch bandwidth.
- **Push to the primary only.** A refused push (`not_primary` / `not_ready`)
  drops the contribution exactly like a rejected commit today; the worker
  moves on and the next task carries fresh routing. No worker-side epoch
  reasoning, no retries against backups.
- TLS note: workers dial owner addrs taken from a signed epoch record; with
  TLS enabled, server certs authenticate the owner as today. Without TLS the
  trust level is unchanged from Phase 1 (server authenticity = TLS; client
  authenticity = HMAC/identity handshake — see the `identity.py` scope note).

### D9. Compatibility and the deterministic anchor

`ownership: static` (today's `assign_shards`, k = 1, HMAC grants) remains the
default and is untouched, as are `LocalBackend`, `AsyncScheduler`, and
`CoordinatorServer` — the deterministic anchor chain stays bit-identical.
`ownership: rendezvous` opts into everything above. CI exercises both; the
parity test pins `rendezvous` with one owner and no churn against `static`
end-state equality.

### D10. Explicitly deferred (and why)

- **Relay role / NAT'd owners** — unused in Phase 2: workers are dial-out,
  owners are `public`, owner↔owner is public↔public, so no relayed data-plane
  path exists yet. The `relay` role offer stays reserved. This also keeps
  **per-frame signed envelopes** deferred: replication is authenticated by
  the session identity (reactor `peer_id`) since the puller dials the replica
  directly — no relayed frames to sign. Both revive if/when NAT'd owners land.
- **Autonomous directory gossip** — owners *can* now carry it (they run
  servers), but nothing needs it while the scheduler is the epoch authority;
  it's a Phase 4 dependency, not a Phase 2 one.
- **Log-shipping replication & weight delta-encoding** — bandwidth
  optimizations, by measurement only.
- **Byzantine owners, quorum reads, reputation-gated eligibility** — Phase 3.

## 4. Implementation slices

Each slice lands green on its own; the order front-loads the pieces that
don't change behavior.

| Slice | Contents | Key tests |
|---|---|---|
| **2a** | `schedule/ownership.py`: HRW placement, epoch records (build/sign/verify), eligibility filter; Ed25519 grant signing/verification beside HMAC; scheduler epoch RPC + tracker caching of the epoch record. No default-path behavior change. | HRW determinism + minimal-disruption property; epoch record round-trip + tamper rejection; signed-grant accept/forge/replay; static path untouched (parity). |
| **2b** | Two-part versions everywhere; owner dynamic key sets + `syncing/active` lifecycle; pull replication (`include_state`, owner-session gating, momentum + private counters); worker fetch-any/push-primary routing. | Replica convergence (weights *and* momentum bit-equal after a pull); non-owner denied `include_state`; non-primary push refused; fetch falls back across replicas; version-pair ordering across an epoch bump. |
| **2c** | Failover: scheduler-side liveness → hysteresis → epoch bump; promotion; lame-duck handoff; cold-sync on join. | Kill-primary-mid-run completes with bounded loss; flapping owner causes ≤1 bump; joiner serves only after sync; zombie old-primary pushes fenced by epoch. |
| **2d** | Per-key persistence + signed manifest + recovery gate; launch wiring (`owner` role, `ownership`/`k`/`replicate_interval`/`owner_grace` config); docs + plan-doc status update. | Restart-from-manifest resumes ≥ manifest versions; local-file warm start delta-syncs (not cold); CLI smoke with `ownership: rendezvous`. |

Rough sizing: 2a S–M, 2b L (the heart), 2c M, 2d M.

## 5. Open questions (recommendation first)

1. **k default** — recommend **k = 3** (tolerates one loss while a
   replacement syncs); k = 2 halves replication bandwidth if owner churn
   proves rare in 0f-style runs. Config-exposed either way.
2. **Read-from-backup staleness bound** — recommend *unbounded within
   `replicate_interval`* (it's small); the alternative is workers preferring
   the primary for reads, which re-concentrates fetch bandwidth and defeats
   half the point.
3. **Scheduler identity availability** — Ed25519 grants make the scheduler's
   key a single point of compromise (as `grant_key` already is). Recommend
   accepting this for Phase 2 (scheduler is already the run's SPOF until
   Phase 4) rather than designing key rotation now.
