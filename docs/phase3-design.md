# Phase 3 design — Byzantine-robust aggregation (the trust wall, §1.1)

Status: **slice 3a landed** (`schedule/aggregate.py` robust combiner +
owner-side quorum buffering with timeout flush; `robustness: off` default,
bit-identical to the Phase 2 anchor). Slices 3b–3d pending. Expands the Phase 3
sketch in [internet-scale-plan.md](internet-scale-plan.md). Decisions D1–D8
state the options and the chosen path; §5 records the four operator calls.

**3a amendments (discovered while building, like Phase 2's):**
- *Per-key buffering, not `(key, generation)`.* The design floated bucketing by
  generation; in the async sharded path a shared module has no single global
  generation (each path advances independently and contributions already apply
  against a moving base). So the buffer is **per key**, flushed at quorum or
  timeout — matching the plan's "buffers contributions per module" wording.
- *Direction/magnitude decoupling.* The robust step is `weight_sum × combine(gᵢ)`
  — an **unweighted** robust direction times the **summed** weight — not a
  combine of the weighted grads. This keeps the engine's summed-across-sharing
  scale, makes `c=1` exactly `weight·grad` (the bit-identity seam), and denies
  an adversary the leverage of scaling its own influence on the direction via
  its (e.g. low-staleness) weight. Rationale in `aggregate.py`.
- *Norm clipping stays per-contribution at push time* (counted in
  `TransportMetrics.norm_clipped` as before); the aggregator combines
  already-clipped grads.

## 1. Goal and trust model

Phases 0–2 made the swarm *correct under faults and churn*: bad floats are
rejected, the bank survives owner death, recovery is gated. But a peer that
authenticates is still **trusted with whatever it submits** — finding §1.1, the
last and hardest wall. A volunteer can present a valid identity and a valid
grant and push a finite, norm-bounded pseudo-gradient that points the wrong
way, or lie about the path it computed. Phase 3 makes *influence earned and
cross-checked* rather than granted by authentication.

**Threat model — precise, because it bounds every decision below.**

- **In scope:** a *minority* of enrolled peers behaving arbitrarily — wrong-way
  gradients, fabricated path results, attempts to gain owner influence, lease
  hogging, bandwidth amplification (§1.14). "Minority" is per protected
  module: of the contributors that reach a module, fewer than the robust
  aggregate's breakdown point are adversarial.
- **Out of scope, by design:** a *majority* coalition (no permissionless
  protocol survives >50% without external trust — we lean on enrollment to
  keep the honest fraction high, not on cryptographic miracles); network-level
  adversaries (TLS owns confidentiality + MITM since Phase 1); and a malicious
  *scheduler* (it is the run's trust root until Phase 4 — same status as its
  signing key today).
- **Sybil resistance = enrollment (Phase 1) + reputation-gated influence**, not
  proof-of-work. Identities are cheap to mint, so a *new* identity must start
  with near-zero influence and *earn* it; minting 1000 peers buys 1000
  near-useless contributors, not 1000 votes.

Phase 3 buys adversary tolerance. It does **not** change the paper's
math when adversaries are absent — the `off` path stays bit-identical to the
Phase 2 anchor (D7), because robust aggregation is a dynamics change and
dynamics changes get validated, never assumed (§1.4 discipline).

## 2. Shape of the result

Two **complementary** defenses at two nodes, plus a reputation substrate that
ties them together. They defend different parts of the bank:

```
   scheduler ── redundant execution: r-fold lease a sampled fraction of tasks,
      │         compare quantized-pseudo-grad digests → agreement signal
      │         (defends LOW-sharing / private modules; feeds reputation)
      │
      │  reputation (per peer_id): agreement + acceptance + uptime
      │     ├─ gates owner eligibility (Phase 2 EpochManager input)
      │     ├─ gates lease priority / admission + rate limit (§1.14)
      │     └─ optional weight in aggregation
      ▼
   owners ──── robust aggregation: buffer a module's contributions from its
               (naturally several) sharing paths, apply a coordinate-wise
               robust aggregate (trimmed mean / median) once a quorum arrives
               (defends HIGH-sharing modules)
```

The split is the key idea. A **high-sharing** module already receives many
honest contributions per round (DiPaCo's structure), so a statistical robust
aggregate at the owner is the natural, cheap defense — no extra compute.
A **low-sharing or private** module is computed by one or few paths, so there
is nothing to aggregate over; the defense is **redundant execution** — recompute
it on r workers and compare. Reputation is the shared currency both feed.

## 3. Decisions

### D1. Robust aggregation at owners (defends high-sharing modules)

Today (`ParameterServer._push`) each contribution is applied as its *own* outer
step (`apply_outer_grads` + `opt.step()`) the instant it arrives — the
documented async-dynamics difference from the engine, which sums sharing paths'
pseudo-gradients before one step. Phase 3 introduces a **per-(key, round)
buffer**: contributions accumulate until a quorum `c` arrives (or a timeout),
then a single robust aggregate is applied as one outer step.

- **Round identity.** Contributions buffer by `(key, generation)` so only
  same-generation pseudo-gradients (computed against the same base weights)
  aggregate together. A late contribution for an already-applied round is
  dropped (its lease's staleness gate usually catches it first).
- **Quorum `c` and timeout.** `c = min(sharing_degree(key), quorum_target)`
  (default `quorum_target = 3`). A buffer that doesn't fill within
  `quorum_timeout` applies what it has (liveness > perfect quorum) — a stalled
  path must not freeze a module forever.
- **Aggregate.** Coordinate-wise, after per-contribution norm clipping (bounds
  any single actor's magnitude *before* it can skew the statistic):
  - `c < 3`: weighted mean (can't trim without ≥3) — i.e. today's behavior;
  - `c ≥ 3`: coordinate-wise **trimmed mean** (trim the top/bottom β fraction)
    or **median**. Default trimmed mean (β≈1/c, drops one extreme per side):
    it tolerates a minority while keeping more signal than the median.
- **Weighting.** The α shard-size weight + √P rescale + staleness damp that
  the grant carries today still apply, but **per contribution before
  aggregation**; the robust step replaces the *sum* over sharing paths, not the
  weights. (This is the dynamics crux — see D7.)

### D2. Redundant execution at the scheduler (defends low-sharing/private modules; feeds reputation)

The scheduler issues a *sampled* fraction `redundancy_rate` of tasks **r-fold**
(`r≈2–3`) to distinct workers training the **same (path, generation, shard)**.
Their pseudo-gradients should agree up to nondeterminism; the scheduler compares
a **digest** (the int8-quantized pseudo-gradient hashed, reusing Phase 0c
quantization so the digest is robust to fp noise) and:

- **agree** → accept one (or the robust mean of the r), credit reputation to all;
- **disagree** → reject the odd one out, debit its reputation, accept the
  majority; if no majority (r=2 split), reject both and re-issue.

This is the *only* defense for modules a single path owns (and for private
modules under policy 3a), and it simultaneously absorbs the worker oversupply
of §1.9 — surplus workers do the redundant copies instead of idling.

- **Lease bookkeeping** changes from one lease per path to up to `r` leases per
  `(path, generation)` replica slot; commit waits for the replica set (or a
  timeout) before granting pushes. Single-replica tasks (the unsampled
  majority) are unchanged.
- **Sampling, not always-on.** `redundancy_rate` (default ~0.1) keeps the
  compute multiplier near 1 + 0.1·(r−1); a peer can't predict which of its
  tasks is checked, so cheating anywhere risks detection everywhere.

### D3. Reputation substrate (scheduler-owned)

A per-`peer_id` score the scheduler maintains from signals it already sees:
redundant-execution agreement (strong signal), contribution acceptance vs.
guard rejection (weak), and uptime/liveness. New identities start at a **low
floor**, not neutral — the Sybil defense lives here.

- **What it gates:** (a) **owner eligibility** — the Phase 2 `EpochManager`
  takes a reputation predicate alongside the existing public/owner/addr filter,
  so a low-rep peer is never placed in an epoch; (b) **lease priority +
  admission** — high-rep peers get scarce leases first, brand-new peers are
  rate-limited and over-sampled for redundancy until they earn standing;
  (c) optionally **aggregation weight** (deferred — start with binary
  accept/reject from D1/D2, add weighting only if measurement wants it).
- **Persistence.** Rides the scheduler checkpoint (signed alongside the
  manifest) so a restart doesn't reset everyone to the floor.
- **Decay.** Scores decay toward the floor over time so a peer can't bank
  reputation then defect cost-free, and a recovered peer can climb back.

### D4. Rate limiting / abuse protection (§1.14)

Per-`peer_id` token bucket on `request`/`commit`/`push` admission (the
amplification vector is a worker looping `request` to pull large task payloads).
Buckets size from reputation (new/low-rep peers get smaller buckets). Small,
scheduler+owner-side, and it shares the per-peer-id bookkeeping with D3, so it
lands in the same slice.

### D5. Private modules (open decision #2 — see §5)

Today a private embedding/head is loaded **verbatim** from its one owning path's
push — a single volunteer can overwrite the embedding table with arbitrary
finite values. Two policies:

- **3a — pin + propose (recommended).** The private module stays authoritative
  on its current owner; worker pushes become *proposals* subject to the same
  redundant-execution cross-check (D2) before they're accepted. Preserves the
  paper's per-path-private semantics.
- **3b — share + robust-aggregate.** Make embedding/head shared-with-robust-
  aggregation for internet runs (D1 covers them). Changes the model's privacy
  structure but folds them into the strong defense.

The plan says "decide empirically"; recommendation is 3a (keeps paper
semantics; 3b is a model change, not just a transport change). **Needs the
operator's call** — it changes what slice 3d builds.

### D6. Composition — which defense covers what

| Module kind | Primary defense | Why |
|---|---|---|
| High-sharing (experts shared by many paths) | Robust aggregation (D1) | Many honest contributions already arrive per round |
| Low-sharing (shared by 1–2 paths) | Redundant execution (D2) | Too few contributors to aggregate over |
| Private (embedding/head, policy 3a) | Redundant execution (D2) | One contributor; cross-check is the only check |

Reputation (D3) underlies all three; rate limiting (D4) is orthogonal and
always on.

### D7. Compatibility and the deterministic anchor (non-negotiable)

`robustness.mode: off` (default) is **exactly today's behavior, bit-identical**:
`c=1` + mean aggregate = apply-each-immediately, `redundancy_rate=0` = one lease
per path, reputation/rate-limit gates open. The static path, `LocalBackend`,
`AsyncScheduler`, and `CoordinatorServer` are untouched.

Robust aggregation **changes the outer-step dynamics** even with zero
adversaries (buffering+aggregating moves the async sharded path *toward* the
engine's batched semantics). So, per §1.4: the synchronous engine stays the
anchor; we validate that (a) `off` is bit-identical, (b) `on` with no
adversaries converges comparably to `off`, and (c) `on` with injected
adversaries (sign-flip, large-norm, fabricated) degrades gracefully where `off`
diverges. (c) is the actual acceptance test for the phase, and it overlaps the
still-pending **0f** WAN run — Phase 3 is the first phase whose *point* is a
dynamics property, so it cannot be called done on green unit tests alone.

### D8. What stays out (and why)

- **Aggregation-weight reputation, Krum/multi-Krum, secure aggregation
  (crypto MPC):** start with trimmed-mean/median + binary reputation; richer
  aggregates are a measurement-driven follow-up, MPC is a different project.
- **Majority-coalition defense, proof-of-work Sybil resistance:** out of the
  threat model (D1).
- **Decentralized scheduling (Phase 4):** reputation + redundancy assume the
  scheduler is the trust root; removing it is the next phase, not this one.

## 4. Implementation slices

Ordered so each lands green on its own and the dynamics-changing piece comes
with its validation harness, not after it.

| Slice | Contents | Key tests |
|---|---|---|
| **3a** | Owner-side robust aggregation: `(key, generation)` buffer, quorum `c` + timeout, norm-clip-then-trimmed-mean/median in a pure `aggregate.py`, `c=1`/mean pass-through identity. `robustness` config seam. | Pure-aggregator unit tests (trimmed mean/median tolerate k of n outliers; mean+c=1 == today byte-for-byte); buffered application matches summed application within fp tolerance; an injected sign-flip minority doesn't move the aggregate past a bound; anchor parity for `off`. |
| **3b** | Reputation substrate + rate limiting: per-peer-id scores (signals, decay, floor, checkpoint-persisted + signed), `EpochManager` reputation predicate, lease-priority/admission, token buckets. | New peer starts at floor and climbs on acceptance; rep gates owner eligibility (low-rep peer absent from epochs); token bucket throttles a `request` flood; reputation survives a scheduler restart. |
| **3c** | Redundant execution: r-fold sampled leasing, replica commit set, digest agreement (reuse int8 quantization), reputation feedback, oversupply absorption (§1.9). | r-fold lease issued for sampled tasks only; agreeing replicas both credited; a disagreeing replica is rejected + debited; r=2 split re-issues; surplus workers get redundant copies instead of idle. |
| **3d** | Private-module policy (per §5 #1) + launch wiring (`robustness` section: mode/quorum/redundancy/reputation knobs) + docs + plan-doc status + the adversarial dynamics validation script (the D7(c) harness, env-var driven like the other validation scripts). | Policy enforcement (3a: a lone private push is a proposal, not an overwrite); CLI smoke with `robustness: on`; the validation script runs a sign-flip-adversary scenario and reports convergence vs `off`. |

Rough sizing: 3a M–L (the aggregator + buffering + anchor parity), 3b M, 3c L
(lease-set rework is the fiddly part), 3d M.

## 5. Operator decisions (resolved)

All four taken at the recommended option:

1. **Private-module policy (D5): 3a — pin + propose.** Private modules stay
   authoritative on their owner; worker pushes are proposals cross-checked by
   redundant execution (D2) before acceptance. Preserves the paper's
   per-path-private semantics; slice 3d builds this (not the 3b model change).
2. **Enrollment posture: open enrollment + reputation ramp.** Anyone with a
   valid identity may join; new peers start at the reputation floor (D3) and
   earn influence, with redundancy over-sampling them until they do. This is
   the "train across the internet on consumer hardware" goal; reputation +
   redundancy are the safety. (Invite-gated remains available via Phase 1
   `enroll_peers` for a conservative deployment — the two compose.)
3. **Quorum scope + `c`: all shared modules, `c = min(sharing_degree, 3)`.**
   Uniform and nearly free (high-sharing modules already receive the
   contributions); redundancy (D2) spot-checks the low-sharing remainder.
4. **Robust aggregate default: coordinate-wise trimmed mean** (one extreme
   trimmed per side at `c ≥ 3`, after per-contribution norm clipping). Median
   stays available as the ultra-conservative knob; Krum deferred (D8).
