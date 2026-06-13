# Phase 4 design — decentralized scheduling (remove the central trust root)

Status: **all four slices landed — the control plane is decentralized.** 4a:
`assignment.py` (HRW`(path, gen)` self-assignment, takeover-on-expiry,
version-lag staleness, coordinator-key) + the `schedule.mode` seam. 4b: the
owner becomes the path coordinator (owner-minted/owner-verified Ed25519 grants,
version-fence, owner-local reputation + rate-limit). 4c: the Byzantine-owner
defense (`state_digest`, `quorum.py` confirm/divergence, owner cross-check →
reputation debit). 4d: deterministic signer-less epochs derived from a gossiped
directory (`derive_epoch`), owner directory import/serve + gossip, the eviction
loop closed (debit → next epoch drops the owner), launch wiring
(`schedule: decentralized`), and `examples/validate_decentralized.py`.
`schedule.mode: central` (default) stays bit-identical to Phase 3. Three
operator calls fixed the scope (§5): **full** Byzantine-owner defense,
**leaderless** HRW self-assignment, **owner-to-owner gossip**. This expands the
one-paragraph Phase 4 sketch in [internet-scale-plan.md](internet-scale-plan.md)
§2. Decisions D1–D9 state the options and the chosen path; §4 slices it 4a–4d;
§5 records the operator decisions.

**The remaining 0f-gated piece (stated plainly).** What is *not* yet runnable in
one process is the **decentralized worker loop** — self-assign → quorum-read
bases → commit to the coordinator → **push to all `k` owners** — and therefore a
single-process `run_local` for `schedule: decentralized` (it raises with a
pointer to the validation script and per-role launch). This is deliberate and
consistent with the 4c boundary: a backup defending against a Byzantine
*primary* needs that push-to-all-`k` path so each owner recomputes rather than
trusts the primary's bytes, and whether `k` independent aggregations converge
comparably to one-primary replication is a **dynamics** property unit tests
can't settle — it is a 0f WAN acceptance item, like Phase 3's convergence
verdict. Everything the defense *is built on* (assignment, grants, fence,
digests, quorum reads, divergence detection, eviction, deterministic epochs,
gossip import) is landed and unit/integration-tested; the end-to-end runtime
that ties them into a converging swarm is the final integration, owed to 0f.

**4d amendments (discovered while building):**
- *No per-owner `EpochManager` in decentralized mode — directory TTL provides
  the hysteresis.* The `EpochManager`'s grace/rate-limit timers are per-node and
  timing-dependent, which would make owners derive *different* epochs from the
  same membership. Instead owners derive directly from their TTL-pruned,
  gossiped directory via `derive_epoch` (a pure function), so identical
  directories yield identical epochs; the directory's liveness TTL is the
  "is it still here" hysteresis. The `EpochManager` stays the central/rendezvous
  mechanism (the scheduler runs one); decentralized mode replaces it.
- *Epoch numbering bumps only on a membership-hash change* (`members_sig`), and
  unchanged churn re-derives the *same* record — so the `(epoch, counter)`
  version gate stays stable across re-derivations. Cross-node numbering
  convergence under gossip lag is the bounded split-brain D6 already accepts
  (version-fence + quorum reads make a transient disagreement harmless).
- *The eviction loop is closed end-to-end:* a 4c digest-divergence debit drops a
  co-owner below `min_owner_reputation`, and the next `derive_and_apply_epoch`'s
  reputation gate excludes it (tested in `test_decentralized_epochs`).

> **Strategic note (recorded up front, honestly).** Phase 4 is the *optional
> endgame* in the plan — *"only worth it if tracker availability actually
> becomes the limiting factor; Phases 0–3 deliver the goal without it."* Two
**4c amendments (discovered while building):**
- *The deferred-from-4b audit relocation is **subsumed**, not ported.* In
  decentralized mode the ``k`` owners independently computing/serving a shared
  key **are** the redundant execution for that key — the cross-owner digest
  agreement is the agreement signal the scheduler's audits used to provide — and
  private modules already cross-check owner-side via Phase 3d proposal-gating
  (``_private_proposal`` lives on the owner). So no separate scheduler-style
  ``_audits`` machinery is ported; the redundancy lives in the replication tier
  where the cross-checking already is.
- *Where 4c draws the validated/0f line (the Phase 3 discipline).* The
  **detection** primitives are unit-validated and sound on their own:
  ``state_digest`` (fp-recompute-tolerant via int8, like ``pseudograd_digest``),
  ``confirm_version`` (majority at the highest agreed version), ``divergent_peers``
  (flags only a same-version digest contradiction, so a *lagging* honest owner is
  never blamed), ``read_quorum_versions`` (a reader trusts only the quorum
  digest), and the owner-side ``_audit_digests_once`` debit toward eviction. What
  rides the **0f** run is the *enforcement dynamics* of the full write path: a
  backup defending against a Byzantine **primary** requires workers to push to
  all ``k`` owners so each can recompute rather than trust the primary's bytes,
  and whether ``k`` independent aggregations converge comparably to one-primary
  replication is exactly the kind of dynamics property unit tests can't settle.
  4c lands the cross-check *signal* and the read defense; the push-to-all-``k``
  write path is wired with the worker loop in 4d and its convergence is a 0f
  acceptance item, called out rather than assumed.
- *One reputation store, two behaviours.* Owner-behaviour debits (digest
  divergence) and worker-behaviour debits (bad commits, 4b) share the per-
  ``peer_id`` ``Reputation`` on the owner, so a peer that is both gets one score
  and the same eviction gate (``EpochManager.is_eligible``) covers both. The
  eviction *wiring* (owners running the EpochManager over the gossiped directory)
  lands in 4d; 4c produces the debit.

**4b amendments (discovered while building):**
- *The owner-coordinator is added to* ``ParameterServer`` *as a focused
  ``commit``/``generation`` RPC pair, not a Scheduler refactor.* Sharing the
  central scheduler's commit code risked the bit-identical ``central`` path, so
  the owner reuses the **pure** helpers (``assignment.py`` HRW/version-lag,
  ``make_grant``, ``Reputation``/``RateLimiter``, ``guard.loss_ok``) and the
  central scheduler is left entirely untouched. Some logic shape repeats; the
  semantics live in the shared pure functions.
- *Staleness is version-lag over the keys the coordinator actually holds* (in
  practice the private coordinator key, whose counter advances once per accepted
  commit for that path — so it **is** the path's generation clock). The
  coordinator can't see versions of the path's other modules (they live on other
  owners), and it doesn't need to: a single-contributor private counter is a
  faithful per-path staleness, and the ``staleness_bound`` gate still applies.
- *Audit **resolution** relocates in 4c, not 4b.* The design table put redundant-
  execution audits in 4b, but they are tightly coupled to the owner-cross-check
  machinery 4c builds (pinned bases, digest agreement) — moving them with the
  cross-checks keeps the two coherent. 4b lands the load-bearing trust pieces:
  owner-minted/owner-verified grants, the version-fence, owner-local reputation,
  and per-owner rate-limit buckets. ``worker_set`` (the live directory the HRW
  assignee check reads) is injectable now and gossip-fed in 4d; absent it, the
  version-fence alone gates (graceful degradation).

> facts are true while building it: (1) the **0f WAN run still hasn't happened**,
> so we have not measured the central scheduler as a bottleneck — Phase 4
> optimizes a SPOF whose cost is so far theoretical; and (2) it **reopens the
> Phase 3 threat model**, which put a malicious scheduler explicitly out of
> scope (*"it is the run's trust root until Phase 4"*). The operator chose the
> **full** trust scope, so Phase 4 also delivers the *owner-behavior
> cross-checking* that Phase 3b deferred (*"detecting an owner serving bad
> weights needs the cross-checks of 3c; documented, not silently assumed"*).
> That makes this the largest phase, and — like Phase 3 — its convergence and
> robustness verdict rides a real run, not unit tests. `schedule.mode: central`
> (today) stays the default and bit-identical throughout.

## 1. Goal and trust model

Through Phase 3 the swarm has one central node, the `Scheduler`, that is the
**run's trust root**: it serializes work into a global clock `_T`, issues
leases, signs the commit grants owners require, holds reputation, runs the
redundant-execution audits, and is the sole signer of ownership epochs. Lose it
and the run halts; compromise it and every guarantee Phase 3 bought
evaporates. Finding §1.8's residual SPOF is *this* node (the parameter-server
SPOF was already removed in Phase 2).

Phase 4 removes it. Every job above moves to a tier that already exists and is
already replicated and reputation-gated — the **owners** (Phase 2) — or to the
workers themselves (self-assignment). The tracker, already a pure self-
certifying directory (Phase 1), degrades to a **bootstrap seed**: owners gossip
the directory among themselves, so the swarm keeps training if the tracker
dies.

**Trust model — the expansion.** Phases 0–2 trusted owners (unreliable, not
adversarial); Phase 3 trusted the scheduler (the trust root). Phase 4 trusts
**neither**:

- **In scope (new):** a *minority* of the **owners** of any given key behaving
  arbitrarily — serving poisoned weights on read, fabricating versions,
  computing a divergent aggregate, lying about a path's generation. "Minority"
  is per key: of the `k` replicated owners of a key (default `k=3`), fewer than
  `⌈(k+1)/2⌉` are adversarial. This is the cross-check Phase 3 deferred.
- **In scope (carried):** a minority of Byzantine *workers* (Phase 3) — robust
  aggregation, redundant execution, reputation all carry over, just hosted on
  owners instead of the scheduler.
- **Out of scope, by design (unchanged):** a *majority* coalition of the owners
  of a key (no permissionless protocol survives >½ without external trust — we
  lean on enrollment + reputation-gated owner eligibility to keep the honest
  fraction high); network adversaries (TLS); and a malicious **tracker** *as an
  authority* — but a malicious tracker can now only *withhold or stale* the
  bootstrap directory, because every record it serves is self-certifying and
  the live directory is reconstructable from owner gossip (it can't forge
  membership; at worst it partitions a newcomer, who can bootstrap from any
  enrolled owner instead).

Phase 4 does **not** change the paper's math when adversaries are absent — the
`central` path stays bit-identical to the Phase 3 anchor (D9), because every
decentralization here is a dynamics-and-trust change and those get validated,
never assumed (§1.4 discipline).

## 2. Shape of the result

```
   tracker (bootstrap SEED only — self-certifying directory; swarm survives its loss)
        ▲ register/heartbeat (best-effort)         ▲ seed fetch on cold start
        │                                          │
   owners (public, replicated k-per-key) ◀── gossip ──▶ owners
        │  • derive epochs deterministically from the gossiped, self-signed directory (no central signer)
        │  • per-path PRIMARY owner: tracks the path's generation, mints commit grants, runs its audits
        │  • k owners per key cross-check: replicated robust aggregation + quorum reads
        │  • owner-local reputation (worker AND owner behaviour)
        ▲  ▲ fetch-with-quorum / push-to-primary / pull-replicate
        │  └──────────────────────────────────────┐
   workers (nat, dial-out only)                    │
        • self-assign: HRW(path, generation) over the live worker set
        • take over a path on lease expiry (next-ranked worker)
        • staleness = version-vector lag of the bases they fetched
        └────────────────────────────────────────-┘
   (no Scheduler node)
```

The control plane stops being a place and becomes a **protocol**: assignment is
a deterministic function every worker computes, the clock is the
already-existing per-module version vector, and trust is sharded across the
owner tier with the owners cross-checking each other.

## 3. Decisions

### D1. Leaderless assignment: HRW(path, generation) over the live worker set

Today `_next_task` is a central queue: the scheduler hands the least-completed
free path to whoever asks. Decentralized, there is no queue — each worker
computes its own assignment:

- The **eligible worker set** is the live `nat`/`public` enrolled peers from the
  (gossiped) directory, reputation-qualified. A worker reads it the same way an
  owner reads the owner set.
- For a path `p` at its current generation `g` (D2), score every eligible worker
  by `sha256(epoch_salt ‖ p ‖ g ‖ peer_id)` (the same HRW machinery as owner
  placement, `ownership._score`); **rank 0 is the assignee** for `(p, g)`. A
  worker trains `p` exactly when it ranks 0 for some `(p, g)` it can serve.
- **Takeover on expiry.** Rank 0 has a soft lease window (`lease_ttl`, derived
  from `heartbeat_timeout`). If `g` hasn't advanced (no commit landed) by the
  deadline, rank 1 becomes the assignee, then rank 2, … — deterministic
  succession, no election, exactly like owner promotion (Phase 2 D2/D5). The
  generation counter advancing (a commit landing at the owner) is the signal
  that the slot is filled; everyone observes it by reading the owner's version.
- **Why HRW over `(p, g)` and not just `p`:** re-rolling the assignee each
  generation spreads paths across the worker pool over time (load balancing,
  the old `_completed`-based fairness) and means a slow/dead worker only stalls
  *one* generation of *one* path before succession reassigns it.

This replaces `_inflight`/`_lease`/`_owner`/`_completed` and the
least-completed heuristic. The **fence** that `_lease` provided (only the
current holder may commit) moves to the owner as a version-fence (D3).

### D2. The clock: per-path generation counters at the primary owner; staleness from version-vector lag

The global `_T` and per-path `_completed[p]` were the scheduler's. Decentralized:

- **A path's generation `g`** is owned by the path's **primary owner** — the
  rank-0 owner (HRW) of a canonical key for that path (its private module's key,
  or its lowest-sorted module key if it has none). The primary owner increments
  `g` when it accepts a commit for `(p, g)` (D3's version-fence). Workers learn
  `g` by reading it from the primary owner (carried in fetch replies). This is a
  small per-path integer co-located with state the owner already versions.
- **Staleness** stops being `_T − _issued[p]` (a global serialization order)
  and becomes **version-vector lag**: when a worker fetches its bases it records
  each module's `(epoch, counter)`; at commit the owner compares the worker's
  fetched versions against the current ones and takes the max lag across the
  path's modules. This is a *local, owner-computable* staleness that needs no
  global clock and is in fact a more honest measure of "how stale is this
  pseudo-gradient" than a global commit counter. The `staleness_bound` /
  inverse-staleness damping semantics are unchanged; only the source of the
  number changes.
- **No global `_target`.** A decentralized run ends by a per-path generation
  target (each path reaches `~num_generations`), checked locally; the
  "run complete" signal gossips like any other directory fact.

### D3. Write path under no scheduler: owner-minted, version-fenced grants

The commit grant — `(path, allowed-keys, push_weight, single-use token)`,
Ed25519-signed — is what an owner requires before applying a push (Phase 2 D3).
With no scheduler to sign it, the **path's primary owner mints it**:

- A worker that finishes `(p, g)` submits its result to the primary owner. The
  owner checks: the worker is HRW rank-0 (or the live successor) for `(p, g)`;
  `g` is the current generation (a stale `g` is a dropped commit, same as a
  stale lease today); the reported loss is finite (guard.py); rate limit (D5).
  If it passes, the owner increments `g`, computes `push_weight`
  (α·shard-weight·staleness-damp, all owner-computable), and mints a grant
  signed with **its own** `PeerIdentity`.
- The grant is consumed at the owners of the path's keys — *including the
  minting owner's co-owners*. They verify the grant against the **minting
  owner's public key**, which is public information carried in the epoch record
  (every owner's pubkey is already there). So a grant is valid iff signed by the
  legitimate primary owner of that path under the current epoch — no shared
  secret, no central signer.
- **Version-fence = the old lease fence.** "Only the current `(p, g)` assignee's
  commit counts" is enforced by the owner accepting the *first* valid commit for
  generation `g` and then advancing to `g+1`; a late/duplicate commit for `g`
  finds the generation already advanced and is dropped. This is exactly the
  semantics of `_lease` + `_completed`, now local to the owner.

HMAC `grant_key` mode is meaningless without a central signer and is **not**
offered in decentralized mode (it remains for `central`/trusted clusters). The
owner-side guard (non-finite rejection, norm clip, allowed-keys, single-use
token) is unchanged.

### D4. The Byzantine-owner defense: quorum reads + replicated deterministic aggregation

This is the slice the "full" scope buys, and the part Phase 3 deferred. Two
mechanisms, read-side and write-side, both leaning on the `k`-replication that
Phase 2 already provides.

**Read-side — quorum reads.** A worker (or a co-owner pulling replication)
fetching a key no longer trusts a single owner's bytes. It fetches the
`(version, content-digest)` from a quorum of the key's `k` replicas (the cheap
part — a digest, not the weights), takes the **majority digest** at the highest
version a majority agrees on, and downloads the *weights* from any replica whose
digest matches. A single Byzantine owner among `k=3` that serves poisoned
weights is outvoted: its digest is in the minority, so its bytes are never
accepted. If no majority exists at the top version (replicas legitimately
mid-sync), fall back to the highest version a majority *does* agree on (older
but honest) — liveness over freshness, the same trade as the replication loss
window.

**Write-side — replicated aggregation with digest agreement.** Robust
aggregation (Phase 3a) is **deterministic** given the same buffered
contributions (trimmed mean / median are order-independent). So instead of one
primary owner applying it and backups copying the result (Phase 2 pull
replication, which trusts the primary's bytes), **all `k` owners buffer the same
contributions and each computes the aggregate independently**, then exchange the
resulting-state digest:

- Workers push their contribution to **all `k` owners** of a shared key (the
  grant authorizes it at each; the per-contribution payload is small after
  int8 compression). Each owner buffers to quorum `c` (Phase 3a) and applies the
  *same* deterministic robust step → the same `(epoch, counter)` → the same
  bytes.
- Owners compare digests at each version. **Agreement** → the version is
  *confirmed* (this is what quorum reads then serve). **A divergent owner** —
  one whose digest doesn't match the majority at a confirmed version — is
  serving or computing something wrong: its **owner-behavior reputation** (D6)
  is debited, and persistent divergence drops it below the owner-eligibility
  gate so the next epoch evicts it (the EpochManager predicate already exists —
  Phase 3b wired *worker* reputation into it; D6 adds *owner* reputation to the
  same seam).

This keeps the design's "small, mostly-disjoint neighborhoods" stance — there
is **no global consensus**, only per-key agreement among that key's `k` owners.
It is *detection + eviction*, not prevention (matching Phase 3c's stance: "a bad
update lands once, then reputation tanks") — a Byzantine owner can serve bad
bytes to a reader that doesn't quorum-check, but the protocol's readers do
quorum-check, so a confirmed version is always the majority's, and the bad owner
is on its way out.

### D5. Owner-local reputation, audits, and rate limiting

Everything the scheduler did per-`peer_id` shards onto the owners:

- **Reputation** becomes owner-local: each owner scores the peers that interact
  with *its* keys — workers (commit accept/reject, audit agreement, as Phase 3b)
  **and now co-owners** (digest agreement from D4). A peer's effective standing
  is the aggregate of what its key-neighbourhood owners report; reputation
  facts gossip alongside the directory (signed per-owner observations, combined
  conservatively — a peer is demoted if *any* honest quorum of its neighbours
  demotes it; an owner can't *inflate* a peer it doesn't co-own). Owner-behavior
  reputation (D4) feeds the **owner-eligibility** predicate the EpochManager
  already takes.
- **Audits** (redundant execution, Phase 3c) move to the primary owner of the
  audited path: it samples, records the pinned base + digest it received, and
  resolves the checker set — all the `_audits` bookkeeping, owner-side. The
  version-pinned-base machinery (owner history) already lives on the owner, so
  this is a natural relocation.
- **Rate limiting** (Phase 3b/§1.14) is per-owner per-`peer_id` token buckets on
  fetch/commit/push admission — naturally distributed (each owner limits its own
  inbound), no shared state.

### D6. Epochs without a central signer: deterministic over the gossiped directory

Phase 2 D1 chose scheduler-signed epochs to avoid split-brain ("rendezvous over
'the live set' is only deterministic if everyone agrees on the set"). With no
scheduler, epochs become a **deterministic function of the self-certifying
directory** instead:

- Each peer record is already individually Ed25519-signed (Phase 1) — the
  directory's *contents* can't be forged, only its *membership view* can differ
  between peers mid-gossip.
- Every owner runs the **same `EpochManager`** (same `owner_grace` /
  `min_epoch_interval` hysteresis) over its gossiped directory view and derives
  the epoch deterministically: `epoch_id = sha256(sorted eligible owner records)`
  plus a logical `(generation-of-membership)` counter for ordering. No
  signature needed — the function is pure and its inputs are signed.
- **Residual split-brain is bounded, not eliminated** (the honest version of
  Phase 2 D1's concern): two owners briefly disagreeing on the set during gossip
  convergence place a key on transiently-different owners; a push to a
  wrong-because-stale owner is *refused with the fresher view* (exactly today's
  `not_primary` refusal + `_fresh_routing` retry), and quorum reads (D4) mean a
  reader never commits to a minority view's bytes. Gossip + hysteresis make
  disagreement windows short and self-healing; the version-fence makes them
  harmless. This is strictly more robust than today's single-signer SPOF for
  *availability*, at the cost of eventual-consistency reasoning we now make
  explicit.
- `verify_epoch_record`'s `signer_pub` pinning still applies in `central` mode;
  decentralized mode verifies the *constituent peer records* and recomputes the
  epoch locally rather than trusting a signature over the set.

### D7. Directory gossip: owner-to-owner pull, tracker as seed

Phase 1 built `fetch_directory` / `import_records` and proved a replacement
tracker can bootstrap from any peer's cache; Phase 2 D10 deferred *autonomous*
gossip to here. Phase 4 turns it on:

- Owners periodically **pull each other's directories** (reuse
  `fetch_directory` against co-owners + a few random owners) and `import_records`
  the result; `issued_at` ordering already prevents stale copies from
  displacing fresh ones, and tombstones already propagate. This is anti-entropy
  gossip, not a new protocol.
- The **tracker becomes a seed**: cold-start peers register with and fetch from
  it, but once a peer knows ≥1 owner it can bootstrap entirely from gossip. A
  dead tracker degrades newcomer onboarding (they need a live seed — any
  enrolled owner can publish its addr out-of-band), not the running swarm.
- Workers (`nat`, dial-out) don't serve a directory; they *consume* it from any
  owner they're already connected to (owners piggyback the current directory
  view + epoch on task/fetch replies), so a worker never needs the tracker after
  its first connection either.

### D8. What stays out (and why)

- **Global consensus / BFT total order (PBFT, Raft, blockchains).** Out by the
  design stance: DiPaCo's per-module owner neighbourhoods are small and mostly
  disjoint, so per-key digest agreement among `k` owners is sufficient and
  `O(k)`; a global ordered log would be the wrong tool and a different project.
- **Majority-coalition defense, proof-of-work Sybil resistance.** Out of the
  threat model (§1), unchanged from Phase 3.
- **Secure aggregation / MPC, encrypted contributions.** A different project;
  TLS covers confidentiality on the wire.
- **NAT'd owners / relay tier, per-frame signed envelopes.** Still unused —
  owners are `public`, gossip is public↔public, the data plane is unrelayed
  (Phase 2 D10). Revives only if NAT'd owners land.
- **Key rotation for owner identities.** Owners' keys are now grant signers; a
  compromised owner key is contained by quorum reads + eviction, not rotation.
  Managed-secret-store rotation stays out (Phase 1 scope note).

### D9. Compatibility and the deterministic anchor (non-negotiable)

`schedule.mode: central` (today's `Scheduler` + `ownership: static|rendezvous`,
HMAC or scheduler-Ed25519 grants, global `_T`) remains the **default and is
untouched**, as are `LocalBackend`, `AsyncScheduler`, and `CoordinatorServer` —
the deterministic anchor chain stays bit-identical. `schedule.mode:
decentralized` opts into everything above and *implies* `ownership: rendezvous`
(it's built on the replicated owner tier). CI exercises both; the parity test
pins decentralized-with-one-owner-one-worker-no-churn against the central
end-state where the dynamics coincide.

Decentralization changes training dynamics in two ways even with zero
adversaries — version-vector staleness differs from `_T` staleness, and
replicated aggregation (D4) replaces primary-applies-once. Per §1.4: validate
(a) `central` bit-identical, (b) `decentralized` with no adversaries converges
comparably, (c) `decentralized` with a Byzantine *owner* (poisoned-read,
divergent-aggregate, fabricated-version) degrades gracefully where an
unprotected read would diverge. (c) is the phase's actual acceptance test and it
overlaps the still-pending **0f** WAN run — like Phase 3, Phase 4 cannot be
called done on green unit tests alone.

## 4. Implementation slices

Ordered so each lands green on its own and the trust-critical piece (4c) comes
with the cross-check tests, not after. The pieces that only relocate behavior
(4a/4b) come before the piece that adds new trust semantics (4c).

| Slice | Contents | Key tests |
|---|---|---|
| **4a** | Leaderless assignment + version-vector clock (D1/D2): HRW(path, gen) self-assignment over the live worker set, takeover-on-expiry, per-path generation at the primary owner, version-vector-lag staleness. `schedule.mode` seam; `central` bit-identical. | HRW assignment determinism + minimal-churn; takeover advances to rank-1 on expiry; version-lag staleness == `_T` staleness in the degenerate single-owner case; central-path parity. |
| **4b** | Owner-minted grants + owner-local reputation/audits/rate-limit (D3/D5): primary owner mints Ed25519 grants signed with its own key, verified by co-owners against the epoch record; reputation/audits/buckets relocated owner-side. | Owner-minted grant accept / forge (wrong signer) / replay refused; stale-generation commit dropped (version-fence); reputation credit/debit + audit resolution owner-side. |
| **4c** | The Byzantine-owner defense (D4): quorum reads (digest-majority across `k` replicas), replicated deterministic aggregation with cross-owner digest agreement, owner-behavior reputation → eviction. | A single Byzantine owner among `k=3` serving poisoned weights is outvoted on read; `k` owners independently reach the same aggregate digest; a divergent owner is detected + debited + evicted next epoch; quorum-read fallback to the highest majority version. |
| **4d** | Owner gossip + deterministic epochs (D6/D7) + launch wiring (`schedule: decentralized`) + the Byzantine-owner validation scenario + docs (this file's status, internet-scale-plan.md Phase 4, CLAUDE.md). | Swarm trains through tracker death (gossip sustains the directory); two owners converge to the same deterministic epoch from divergent initial views; CLI smoke with `schedule: decentralized`; validation script runs a Byzantine-owner scenario and reports convergence vs `central`. |

Rough sizing: 4a L (the assignment+clock rework is the foundation), 4b M (mostly
relocation), 4c L (the new trust semantics + cross-checks), 4d M–L (gossip +
deterministic epochs + the harness). XL overall — the largest phase.

## 5. Operator decisions (resolved)

1. **Trust scope (D4): full — defend against Byzantine owners.** Phase 4
   delivers the owner-behavior cross-checking Phase 3b deferred (quorum reads +
   replicated-aggregation digest agreement), so removing the central trust root
   does not reopen a hole. The larger scope is taken deliberately.
2. **Assignment (D1): leaderless self-assignment.** HRW(path, generation) over
   the live worker set with deterministic takeover-on-expiry; no per-path
   coordinator queue. Matches the plan's `hash(path, generation) → eligible peer
   set` wording and keeps the "control plane is a protocol, not a place"
   property.
3. **Directory (D7): owner-to-owner gossip; tracker is a bootstrap seed.** The
   swarm survives tracker loss; autonomous gossip (deferred from Phase 2 D10)
   lands here.
