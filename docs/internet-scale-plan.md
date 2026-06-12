# opendipaco — internet-scale gaps & the peer-to-peer transition plan

`remaining-gaps.md` maps the project against "reproduce DiPaCo on a cluster". This
document maps it against a different goal: **training over the public internet on
volunteer consumer hardware**, and lays out the plan to get there — including the
transition from today's hub-and-spoke transport to a **peer-to-peer topology**.

The one-line summary: **the distributed-systems machinery is the right shape, but it
assumes every participant is honest and well-connected — the opposite of the target
environment. Trust and bandwidth are the two walls; neither is addressed anywhere in
the codebase yet. Nothing needs to be thrown away: the engine, wire format, reactor,
and scheduler skeleton all evolve into the P2P design rather than being replaced.**

Severity tags are relative to *this* goal (they differ from `remaining-gaps.md`):
- **P0** — blocks volunteer-internet training outright
- **P1** — survivable at small scale, breaks as the swarm grows
- **P2** — hardening / hygiene

Each finding is also tagged **[design]** (a gap in the architecture) or **[bug]**
(incorrect behavior in the code as written, even on a trusted cluster).

---

## 1. Prioritized findings

### P0 — blockers

#### 1.1 The trust model is "authenticated = trusted" · [design]

HMAC/TLS proves a worker holds a key; it says nothing about what the worker *does*.
Once authenticated:

- ~~`CoordinatorServer._receive` applies whatever pseudo-gradient arrives — no
  NaN/Inf check, no norm cap, no plausibility check against the reported loss.~~
  **✅ The cheap half is done (`schedule/guard.py`)**: every server now rejects
  non-finite contributions outright (pseudo-gradient, private weights, *and*
  reported loss — always on, no knob), an optional `max_update_norm` clips
  oversized pseudo-gradients per module (plumbed through
  `TransportCfg.max_update_norm`), the sharded scheduler refuses a push grant for
  a non-finite loss at commit, and bogus/mistyped module keys are filtered (an
  unknown key can't crash the server; a "private" payload can't overwrite a
  shared module). Rejections/clips are counted in `TransportMetrics`
  (`invalid_rejected`, `norm_clipped`). Verified by `tests/test_update_guard.py`.
  This bounds *damage from faulty hardware*, not malice.
- **Private modules (embedding, head) are still loaded verbatim from worker
  pushes** (now finite-checked and type-checked, but a volunteer can still
  overwrite the embedding table with arbitrary *finite* values; no aggregation
  softens it).
- No redundant execution, no cross-checking, no reputation, no outlier rejection.
  A path is computed by exactly one worker per generation, so a single bad actor
  owns that path's update entirely. Volunteer systems (BOINC, Petals-style swarms)
  all need at least replicated tasks + agreement. **This — Phase 3 — remains the
  real trust wall.**

#### 1.2 Bandwidth: full fp32 tensors both ways, every generation, no compression · [design] · ◑ compression done

**✅ Wire compression is in (`schedule/compress.py`)**: `compress="int8"` (a server
policy stamped on each task; `TransportCfg.compress`) ships weights down as bf16,
shards as int32, and pseudo-gradients up as per-tensor symmetric **int8 with
worker-side error feedback** (the quantization residual is carried per
(path, module) and folded into the next generation's delta, reset on cold start;
in the sharded path the residual is updated only after the commit is accepted, so
it always reflects an update that was actually pushed). Payloads are
self-describing (`{"q", "s"}` markers; bf16 state dicts auto-cast on
`load_state_dict`), so receivers need no mode config and reject malformed
encodings via the §0b guards. Measured on the CLI smoke run: **down 68.6 → 34.3 MB
(2×), up 68.3 → 17.1 MB (4×)**. Verified by `tests/test_compress.py`, including a
loss-tracking test that int8 training stays in the fp32 run's ballpark.

**Still open:** the "ship only stale versions" cache (`distributed.py`) remains
structurally defeated in async mode — every accepted contribution bumps
`_versions` for modules shared across paths, so full (now bf16) weights still
re-ship nearly every task. Closing the rest of the gap needs **delta encoding
against the worker's held version** and/or sparsification, and the convergence
impact of quantization must be part of the §0f WAN validation. For a paper-scale
150M-param path the per-generation traffic is now roughly ~300 MB down + ~150 MB
up — better, but a 20 Mbps uplink still spends ~10 min/round uploading.

#### 1.3 The data plane is fully centralized and in-RAM · [design] · ◑ shard shipping done

**✅ Servers can now ship recipes instead of bytes (`data/spec.py`)**: with
`data.ship: spec`, the coordinator/scheduler holds a `SpecCorpus` — a shard
*spec* (document source + k-means centroids + the deterministic featurizer's
parameters + packing rules, a few KB) plus per-path token counts (the alpha
basis) — and **no sequence tensors at all** (`SpecCorpus.shard()` refuses).
Workers materialize their own shards locally: regenerate (synthetic) or stream
(C4) the documents, route them with the shipped router, keep their path's, pack
— bounded memory, optional on-disk cache (`data_dir=` /
`data.shard_cache_dir`). Materialization is **bit-identical** to the shards the
bytes path would have shipped (parity-tested against `ShardedCorpus`), and the
end-to-end tests assert `bytes_shard == 0`. Verified by `tests/test_data_spec.py`.

**Still open:** the server still touches the corpus *once at startup* to fit the
k-means router and compute token counts (a streaming pass; it keeps nothing);
EM re-sharding remains central; spec mode has no per-path validation split (so
per-path early stopping is off); and a cold C4 worker re-streams the corpus
prefix rather than reusing `ingest_c4_shard`'s resumable bulk path. Worker-side
`shard_cache` is still unbounded in RAM (the disk cache bounds *re-streaming*,
not memory).

#### 1.4 Async convergence is unvalidated — and it's the only mode that fits the internet · [design]

Already the project's own P0: per-contribution Nesterov steps, inverse-staleness
damping, dropped √P rescale. Everything validated on GPU is the *synchronous* path.
Whether the async coordinator converges at scale is an open empirical question, and
internet training cannot use the synchronous path. If the async dynamics don't
converge, the transport work is moot — validate early.

#### 1.5 No lease fencing — a zombie worker can hijack a re-leased path · [bug] · ✅ fixed

Every lease now carries a unique token that the worker must echo on
submit/nack/heartbeat; a mismatched token is dropped, and reclaim invalidates the
token. Verified by `test_zombie_submit_fenced_after_release` and
`test_sharded_zombie_commit_fenced`. *(Original bug: `gen_id` was echoed but
ignored, and `_owner` was checked for heartbeats but not at submit/commit — worker
A leases path P, stalls past the heartbeat timeout, P is reclaimed and re-leased
to B; A's stale submit was accepted with understated staleness and B's fresher
result silently dropped.)*

#### 1.6 Parameter-server pushes are decoupled from commit verdicts · [bug] · ✅ fixed

The scheduler's commit now returns a **grant** (path, single-use lease token,
damped weight, allowed keys) that the PS requires on every push: the weight and
key allow-list come from the grant, replays are refused, and with `grant_key=` set
on the scheduler + PSs (kept secret from workers) the grant is HMAC-signed so a
worker can't forge one. Verified by `test_ps_push_requires_valid_grant`.
*(Original bug: `_push` applied any authenticated push with whatever `weight` the
worker claimed — the staleness bound was unenforceable where the weights live.)*

#### 1.7 Async coordinator checkpoints lose optimizer + clock state · [bug] · ✅ fixed

`CoordinatorServer` checkpoints now include the async clock `_T`, per-path
`_completed` (the inner-LR schedule position), `_versions`, and the per-key outer
Nesterov momentum, snapshotted consistently under the lock; PS `save_shard` /
`load_shard` round-trip the outer momentum too. Verified by
`test_async_checkpoint_restores_clock_momentum_and_schedule` and
`test_ps_checkpoint_restores_outer_momentum`. *(Original bug: only the engine was
saved — `engine.outer_opt` is unused on the async path — so resume reset the
momentum and restarted every path's cosine schedule at generation 0.)*

### P1 — breaks as the swarm grows

#### 1.8 Star topology with a heavyweight center · [design]

All weight traffic flows through one coordinator (or K parameter servers). Central
egress scales O(workers) — with the payloads of §1.2, even ~50 volunteers saturate a
10 Gbps hub. No peer-to-peer aggregation or relay tree. Workers only dial out
(NAT-friendly — keep this property), but the center must be real infrastructure, it
bounds swarm size, and it remains a SPOF with checkpoint/restart as the only
recovery. **This is the finding the P2P plan (§2) addresses.**

#### 1.9 Work granularity doesn't fit a swarm · [design] · ◑ idle pacing done

**✅ Server-driven idle backoff is in**: with `idle_backoff` set
(`TransportCfg.idle_backoff`), idle replies carry `retry_in` and workers wait
that long instead of hammering at their own `poll_interval` — the poll-storm
half of this finding is closed. **Still open (the structural half):** at most
`num_paths` leases can be in flight (one lease per path), so surplus workers
have nothing to do but wait. Redundant execution (§1.1 / Phase 3) absorbs the
oversupply and provides the agreement signal at the same time.

#### 1.10 No accommodation for heterogeneous hardware · [design] · ◑ basics done

**✅ The basics are in**: `DiLoCoConfig.inner_autocast` runs the inner
forward/backward under **bf16 autocast** (params/grads/pseudo-gradients stay
fp32; no loss scaling — ~2× throughput/memory on consumer GPUs), and workers
advertise a **capability profile** on every request (`device`, plus an optional
`max_batch_size=` / `run.worker_max_batch` cap that the server clamps task batch
sizes to — a small-VRAM volunteer trains smaller batches instead of
OOM-crash-looping; note the documented caveat that a capped worker's paths see
different batch statistics). Verified by `tests/test_heterogeneity.py`, including
bit-identity of the default fp32 path. **Still open:** task sizing from
*measured* throughput (the capability profile is a place to put it), and
per-worker `inner_steps` (deliberately untouched — it changes the inner LR
schedule semantics).

### P2 — hardening / hygiene

- **1.11 `torch.load(weights_only=False)`** · [bug-adjacent] · ✅ fixed — every
  checkpoint/shard/cache load (engine ranks, coordinator extra, PS shards,
  scheduler clock, shard caches, document caches) now uses `weights_only=True`;
  the corpus classes are the only allow-listed non-primitives
  (`torch.serialization.add_safe_globals`). Checkpoint files are no longer a
  code-execution vector.
- **1.12 Stale, alarming docstring** · [bug] · ✅ fixed — `schedule/distributed.py`'s
  module docstring claimed the wire format was `torch.save`/unpickles; it now
  describes the pickle-free codec (and the lease fence).
- **1.13 Identity & enrollment** · [design] · ✅ done — Phase 1 landed per-peer
  Ed25519 identities (`schedule/identity.py`, `opendipaco gen-identity`),
  identity-signature auth in the reactor handshake with runtime
  admit/revoke, and tracker-level enrollment (allowlist or open, token-gated
  via composed HMAC). A managed secret store remains out of scope.
- **1.14 No rate limiting / abuse protection** beyond `max_connections` and the
  message-size cap; an authenticated worker can loop `request` to force large
  payloads (bandwidth amplification). · [design]

---

## 2. The peer-to-peer transition plan

### Design stance

Two observations shape the plan:

1. **DiPaCo's structure is unusually P2P-friendly.** Each shared module is owned by
   the subset of paths that share it — communication is *naturally* partitioned into
   small per-module groups (the same insight behind `TorchDistBackend`'s subgroup
   all-reduce). A P2P design doesn't need global all-to-all; it needs many small,
   mostly-disjoint aggregation neighborhoods.
2. **Full decentralization is not the first milestone.** The pragmatic target is the
   BitTorrent shape: a **light tracker** (control plane: membership, leases, clock —
   tiny messages) with the **heavy bytes peer-to-peer** (weights, pseudo-gradients,
   data). The tracker is what `Scheduler` already almost is — it holds no weights.
   Full P2P scheduling is the optional endgame (Phase 4), not the prerequisite.

What evolves rather than being replaced:

| Today | Becomes |
|---|---|
| `ParameterServer` (K fixed shards) | **Module-owner peer role** — same fetch/push surface, replicated, assigned by rendezvous hashing over reliable peers |
| `Scheduler` (queue + clock, no weights) | **Tracker** — membership, lease grants, reputation; eventually optional |
| `wire.py` codec | Unchanged; gains a **signed envelope** (per-peer keys) and **compressed tensor dtypes** |
| `reactor.py` | Unchanged; every peer that accepts inbound runs one |
| `assign_shards` (static) | **Rendezvous hashing + owner-set churn protocol** (the old "P3 · deliberate" static-assignment gap becomes required) |
| HMAC shared secrets | **Per-peer Ed25519 identities**; HMAC stays for the tracker bootstrap |

### Phase 0 — topology-agnostic hardening (do first; pays off in any topology)

Everything here is needed regardless of P2P and most of it is independent work:

- **Fix the protocol bugs**: lease tokens checked at submit/commit (§1.5);
  PS pushes bound to scheduler-signed commit grants (§1.6); persist the async
  clock, `_versions`, `_completed`, and per-key outer momentum (§1.7).
- **Update validation at the aggregation point** (§1.1, the cheap half): reject
  NaN/Inf, cap outer-grad norms, sanity-check reported loss against history.
  Protects against faulty hardware before adversaries.
- **Compression** (§1.2): bf16 weights on the wire; int8-quantized pseudo-gradients
  with error feedback held worker-side; ship weight *deltas* against the worker's
  held version where cheaper. The wire dtypes already exist. Target ≥8× reduction.
- **Decentralize data** (§1.3): ship shard *ids*; wire `data/streaming.py` ingestion
  into the worker so volunteers pull C4 (or any public corpus) directly; ship the
  router (it's tiny — k-means centroids) so peers route documents locally.
- **Heterogeneity basics** (§1.10): bf16/AMP inner loop; worker advertises a
  capability profile (VRAM, tokens/sec) on `request`; tracker sizes
  `batch_size`/`inner_steps` per worker. Server-driven idle backoff (§1.9).
- **Hygiene**: `weights_only=True` loads with a tensor-only payload (§1.11); fix the
  stale docstring (§1.12).

**Exit criterion:** a trusted-but-distributed run over real WAN links (e.g. three
homes + one VPS) completes a multi-day training with restarts, at <1/8 today's
bytes, with validated convergence vs. the synchronous anchor. This also retires
§1.4 at small scale before P2P amplifies the dynamics.

### Phase 1 — identity, membership, and reachability · ✅ done

- **Per-peer identity** — ✅ `schedule/identity.py`: Ed25519 keypair per peer
  (`PeerIdentity`, PEM key file mode 0600, `opendipaco gen-identity`); peer id =
  sha256(pubkey). The reactor handshake accepts an identity **signature over the
  challenge nonce** alongside HMAC (`admitted_peers=` on every server /
  `transport.admitted_peers`; workers present `transport.identity_key`);
  enrollment = admit a pubkey (manual, or token-gated by composing HMAC
  underneath), revocation = `revoke_peer()` (new handshakes refused).
  *Scope note:* instead of a per-frame wire envelope, signing landed where
  trust is actually needed today — the challenge handshake (session identity)
  and **self-certifying signed records** (`sign_record`/`verify_record`,
  canonical-JSON + embedded pubkey + key-derived id). Per-frame envelopes wait
  until Phase 2 introduces relayed data-plane messages that need them.
- **Tracker as rendezvous** — ✅ `schedule/tracker.py` (+ `opendipaco tracker`
  role, `tracker` config section): peers register signed records
  (peer id, reachability, optional addr, role offers, capability profile);
  TTL-based liveness (heartbeat = re-register; `issued_at` ordering so stale
  relayed copies never displace newer ones); signed deregistration with
  tombstones; enrollment allowlist (`enroll`/`expel`) or `open_enrollment`.
  *Gossip-lite:* records verify independently of who serves them, and
  `fetch_directory` → `import_records` bootstraps a **replacement tracker from
  any peer's cache** (tested) — tracker loss degrades rather than halts.
  Autonomous peer-to-peer directory gossip is deferred to Phase 2 (when peers
  run servers that could carry it).
- **Reachability tiers** — ✅ `"nat"` peers register addressless (dial-out-only,
  today's workers); `"public"` peers advertise an addr + roles
  (`owner`/`relay`), self-nominating to host the Phase 2 P2P plane. Validation
  enforced at signing and at the tracker. UDP hole-punching remains a later
  optimization, not a dependency.
- **Build-vs-adopt** — ✅ **resolved: custom on the existing TCP reactor.** The
  deciding fact: `cryptography` was already the optional `[launch]` dependency
  (TLS cert generation), so Ed25519 costs zero new deps, and the reactor/wire
  stack needed only an extra auth branch. hivemind/libp2p stay fallback options
  if Phase 2's relay tier proves heavier than expected.

### Phase 2 — distribute the module bank (PS → replicated owner peers)

> Detailed design (decisions D1–D10, slices 2a–2d): [phase2-design.md](phase2-design.md).

- **Ownership**: rendezvous hashing maps each module key to **k owner peers** (k≈3)
  drawn from the reachable, reputation-qualified tier. The owner set per key is
  published via the tracker/gossip; workers fetch from the nearest/fastest owner.
- **Replication**: per-module version counters already exist — extend to a
  primary-per-key with backups pulling deltas (owner-to-owner traffic is small:
  one module, k peers). On primary loss, highest-version backup promotes; on owner
  churn, rendezvous hashing re-maps and the new owner cold-syncs from replicas
  (this replaces static `assign_shards`).
- **Checkpointing**: each owner persists its modules (today's `save_shard`, plus
  outer momentum per §1.7); the tracker assembles a **signed manifest**
  (key → version → owner) so a cluster restart has a consistent recovery point.
- **Bandwidth effect**: weight-fetch and gradient-push load spreads across all
  owners instead of K parameter servers; combined with Phase 0 compression this
  removes the central egress wall (§1.8) for the data plane. The tracker's
  remaining traffic is control-plane-tiny.

### Phase 3 — Byzantine-robust aggregation (the trust wall, §1.1)

- **Quorum aggregation at owners**: an owner buffers contributions per module
  until a small quorum c (e.g. 3) arrives, then applies a **robust aggregate**
  (coordinate-wise trimmed mean or median, plus norm clipping) instead of applying
  each contribution individually. Bounded staleness still gates admission via the
  commit grant. This changes the outer-step dynamics — it must be validated against
  the deterministic anchor like every other dynamics change (§1.4 discipline).
- **Redundant execution**: the tracker issues each lease r-fold (r≈2–3) on a random
  sample of tasks; agreement between replicas is checked on a digest of the
  quantized pseudo-gradient. Disagreement burns reputation. This simultaneously
  absorbs worker oversupply (§1.9).
- **Reputation**: per-peer-id score from validated contributions, agreement checks,
  and uptime; gates owner eligibility (Phase 2), lease priority, and quorum weight.
  Sybil resistance comes from enrollment (Phase 1) + reputation-gated influence,
  not proof-of-work.
- **Private modules**: stop accepting verbatim overwrites. Either (a) pin each
  path's private modules to its current trusted owner and treat worker copies as
  proposals subject to the same robust aggregation, or (b) make embedding/head
  shared-with-robust-aggregation for internet runs. (a) preserves paper semantics;
  decide empirically.

### Phase 4 — decentralized scheduling (optional endgame)

- Replace tracker-issued leases with deterministic assignment: hash(path,
  generation) → eligible peer set, takeover on lease expiry via the gossip layer;
  staleness via the per-module version vectors that already exist instead of the
  global `_T` clock.
- The tracker degrades to a bootstrap node — the swarm survives without it.
- Only worth it if tracker availability actually becomes the limiting factor;
  Phases 0–3 deliver the goal without it.

---

## 3. Open decisions

1. ~~**Build vs. adopt the P2P substrate** (Phase 1)~~ — **resolved:
   custom-on-reactor** (Ed25519 via the already-optional `cryptography` dep; the
   reactor needed only an extra auth branch). Revisit only if Phase 2's relay
   tier proves heavier than expected.
2. **Private-module policy under adversaries** (Phase 3a vs 3b above).
3. **Quorum vs. throughput trade-off**: c-of-n aggregation multiplies compute cost
   by ~c on protected modules; how much of the bank needs it vs. spot-checking.
4. **Enrollment friction**: open enrollment + reputation ramp-up vs. invite-gated.

## 4. Sequencing summary

| Order | Work | Findings addressed | Size |
|---|---|---|---|
| ~~0a~~ ✅ | ~~Protocol bug fixes (fencing, commit grants, async checkpoint)~~ | **Done** — 1.5, 1.6, 1.7 (+ the 1.12 docstring) | S |
| ~~0b~~ ✅ | ~~Update validation (NaN/norm/loss)~~ | **Done** — 1.1 (the faulty-hardware half; malice needs Phase 3) | S |
| ~~0c~~ ✅ | ~~Compression (bf16 + int8 grads + error feedback)~~ | **Done** — 1.2 (2× down / 4× up measured; delta-encoding + convergence validation remain) | M |
| ~~0d~~ ✅ | ~~Data decentralization (shard ids + worker-side ingest + local routing)~~ | **Done** — 1.3 (`data.ship: spec`; router fitting + EM still central) | M |
| ~~0e~~ ✅ | ~~bf16 inner loop, capability negotiation, idle backoff, hygiene~~ | **Done** — 1.10 (autocast + batch caps), 1.9 (idle pacing), 1.11, 1.12 | S–M |
| 0f | WAN validation run of the async path | 1.4 | M (mostly wall-clock) |
| ~~1~~ ✅ | ~~Peer identity + tracker + reachability tiers~~ | **Done** — 1.13, 1.8 (prereq); per-frame envelopes + autonomous gossip deferred to Phase 2 | M |
| 2 | Replicated module owners, dynamic ownership, signed manifests — **2a, 2b done** (HRW placement, signed epochs, Ed25519 grants, version pairs, pull replication, fetch-any/push-primary); 2c–2d pending | 1.8 | L |
| 3 | Robust aggregation, redundancy, reputation, private-module policy | 1.1, 1.9, 1.14 | L |
| 4 | Decentralized scheduling (optional) | residual SPOF | L |

**Bottom line:** Phase 0 is the highest leverage-per-effort and is required no matter
what; Phases 1–2 remove the bandwidth/SPOF wall; Phase 3 removes the trust wall.
Validate training dynamics (§1.4) before and during each phase that changes them —
the synchronous engine remains the deterministic anchor throughout.
