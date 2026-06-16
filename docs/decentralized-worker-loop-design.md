# Decentralized worker loop — completing Phase 4 / the §0f on-box half

Status: **landed (on-box).** Phase 4 landed the decentralized *control plane* —
leaderless HRW`(path, gen)` self-assignment (`assignment.py`), owner-minted
version-fenced Ed25519 grants, quorum reads + cross-owner digest agreement +
eviction (`quorum.py`), and signer-less `derive_epoch` over a gossiped directory
(`docs/phase4-design.md`). What it explicitly left for §0f was the **worker
runtime that ties those primitives into a converging swarm**, and a
single-process `run_local` for it. Both are now built (slices a–c): the worker
loop (`run_decentralized_worker`/`_serve_decentralized`), an in-process
`run_local` driver, and a `validate_dynamics.py` decentralized arm that trains
the scheduler-less write path vs. the synchronous anchor on one box. The **WAN
systems** half (real latency/NAT/bandwidth/churn) and the at-scale convergence
verdict still ride the §0f WAN run.

The §0f WAN run answers two separable questions (`validate_dynamics.py`
docstring): the **systems** half (real latency/NAT/bandwidth/churn) needs
distributed hardware; the **dynamics** half (does it *converge?*) does not. This
work delivers the dynamics half for decentralized mode — a `run_local` that runs
the real loop end-to-end and a `validate_dynamics` arm checking it converges
comparably to the central anchor — leaving only the systems half to hardware.

`schedule: central` (default) stays bit-identical. Each decision (D1–D10) states
the options and the recommendation; slices a–c land green independently.

## 1. Goal and what's already built

The owner side is **complete** (Phase 4b–d): a `ParameterServer` in
`schedule_mode="decentralized"` already serves the RPCs the worker needs —

| RPC | what it does (owner side, built) |
|---|---|
| `generation` | reports a path's current `(g, opened_at)` if this owner coordinates it |
| `fetch` (+ quorum) | serves weights/versions; `_replicate_once` already quorum-confirms in decentralized mode |
| `digest` | cheap `(version, content-digest)` for quorum reads |
| `commit` | **version-fences** `g`, computes version-lag staleness + `push_weight`, **mints** the Ed25519 grant signed with its own identity |
| `push` | applies a granted pseudo-gradient (grant verified against the minting owner's pubkey from the epoch); buffers to quorum `c` + deterministic robust step |
| `directory` | serves this owner's gossiped peer directory (tracker = seed) |

Plus the pure helpers: `derive_epoch`, `is_assignee`/`responsible_rank`,
`read_quorum_versions`/`confirm_version`/`divergent_peers`, `coordinator_key`/
`path_primary`. **The only missing piece is the worker that calls them in the
right order.**

**Goal.** A `run_decentralized_worker` that self-assigns → quorum-fetches bases →
trains → commits to the path's coordinator → pushes to all `k` owners; a
`run_local` that stands up owners + workers in one process and trains to a
per-path generation target; and the on-box convergence verdict.

## 2. The loop (one iteration)

```
refresh directory (pull from an owner; tracker seed) -> derive_epoch locally
for a path I may serve:
    (g, opened_at) <- primary-owner generation RPC
    if not is_assignee(me, path, g, workers, elapsed=now-opened_at, lease_ttl):  continue
    bases, fetched_versions <- quorum-fetch each key (read_quorum_versions -> confirmed digest -> weights)
    contrib <- _train_path(path, shard, batch, g)            # reuse the engine
    ack <- commit to primary owner {path, g, loss, fetched_versions, base, digest}
    if not ack.accepted:  continue                            # g advanced / stale / rate-limited
    push contrib to ALL k owners of each key, with ack.grant  # each aggregates independently
```

## 3. Decisions

### D1. A separate `_serve_decentralized` loop (reusing the shared mechanics)

`_serve_sharded` is a central-queue loop (request → lease → fetch-primary →
commit → push-primary). The decentralized loop differs at three steps
(self-assign vs lease, quorum-fetch vs single, push-to-all-`k` vs primary-only),
so a **separate `_serve_decentralized`** is clearer than threading a mode flag
through `_serve_sharded`. It **reuses** the transport `link` (TCP/libp2p seam),
`AsyncScheduler._train_path`, the warm caches, and `_compress_contribution` —
the shared mechanics, not the control flow. `run_decentralized_worker` is the
entry (parallel to `run_sharded_worker`); it needs an **identity** (it derives
epochs and is HRW-scored by `peer_id`).

### D2. Directory + epoch discovery: the worker pulls and derives locally

A `nat` worker is dial-out-only — it can't be *gossiped to*. So it **pulls** the
directory (an owner's `directory` RPC; the tracker as bootstrap seed) and runs
`derive_epoch` **locally** — the same signer-less, deterministic derivation the
owners run, so an identical directory yields an identical epoch (owner set +
worker set + `k` + salt). It refreshes on an interval. Bounded cross-node
disagreement under gossip lag is the **accepted D6 residual**: the version-fence
(D5) + quorum reads (D4) make a transient epoch disagreement harmless (a worker
on a stale epoch either isn't the assignee — its commit is dropped — or fetches/
pushes to owners that quorum-agree). Rejected: trust an owner's *served* epoch —
owners can momentarily disagree under lag, and trusting one re-introduces a
single point the signer-less design removed.

### D3. Self-assignment: scan paths, claim the ones I'm rank-0 (or successor) for

Per iteration the worker scans the paths and, for each, reads `(g, opened_at)`
from the path's **primary owner** (`generation` RPC) and computes
`is_assignee(me, path, g, workers, elapsed, lease_ttl)`. HRW makes each worker
rank-0 for a roughly disjoint subset, so honest workers don't collide;
**takeover-on-expiry** (`responsible_rank` from `elapsed = now − opened_at`)
hands a stalled `(p, g)` to rank 1, 2, … deterministically. The generation
*advancing* (a commit landed at the owner) is the "slot filled" signal everyone
observes by reading `g`. A worker serves one assigned path per iteration (then
re-scans), so a fast worker naturally picks up successor slots; backoff/`idle`
when it's assignee of nothing keeps the scan cheap.

### D4. Quorum-fetch bases (read-side Byzantine defense)

The worker fetches each key's base via `read_quorum_versions` over the key's `k`
replicas → `confirm_version` (majority digest at the highest majority version) →
downloads the *weights* from a replica whose digest matches. It records the
fetched `(epoch, counter)` per key and reports them at commit, so the owner
computes **version-lag staleness** (D2/phase4). A single Byzantine owner is
outvoted; if no top-version majority exists (replicas mid-sync), fall back to the
highest majority version — liveness over freshness, the replication-window trade.
Reuses the exact primitives the owner replication loop already uses.

### D5. Commit to the coordinator; D6. push to all `k` owners

- **Commit (write-fence).** The worker submits to the path's **primary owner**
  (`path_primary(coordinator_key(path), epoch)`). The owner version-fences `g`
  (first valid commit for `g` wins; a stale `g` is dropped), guards the loss,
  rate-limits, computes `push_weight`, and **mints** the grant. The worker gets
  the grant or a drop (treated like a rejected central commit — re-scan).
- **Push to all `k`.** With the grant, the worker pushes the pseudo-gradient to
  **every** owner of each key, and in decentralized mode **every active owner
  applies it independently** (`_may_write_locked`) — the same deterministic outer
  step (or, under `robustness.mode: on`, the same quorum-buffered aggregate) from
  the same base + grant weight, so all `k` reach the **same version and bytes**.
  This is load-bearing, not an optimization: the slice-c `validate_dynamics` arm
  showed that the original "only the primary applies; co-owners get it by
  replication" plan **deadlocks** — a fresh write is held by one replica, and
  decentralized replication only adopts a *quorum-confirmed* version (Byzantine-
  source safety, phase4 D4), so a single-replica version can never reach quorum,
  staleness grows unbounded, and commits stall. Independent application makes the
  write reach quorum immediately *and* is the Byzantine-**primary** defense
  (co-owners recompute rather than trust the primary). Determinism holds because
  the worker pushes to all `k` for one contribution before producing the next, so
  every owner applies the same ordered sequence. The grant is single-use **per
  server** (each owner consumes its own token, verifying it against the minting
  owner's pubkey from the epoch — `grant_signed_by`). A key that lands at **no**
  owner (the epoch moved entirely under the task) re-routes once after a local
  epoch re-derive; the bounded-loss window is unchanged.

### D7. `run_local` for decentralized

Stand up, in one process: a `Tracker` (seed), `k`+ owner `ParameterServer`s
(`schedule_mode="decentralized"`, identities, gossip + `derive_and_apply_epoch`
in their replication loops), and `run_decentralized_worker` threads. The **first
epoch is bootstrap-flagged** (owners boot-serve their seeded `(0,0)` banks, as in
Phase 2) so the swarm starts without a sync deadlock. The run ends on a **per-
path generation target** (each path reaches ~`num_generations`), checked by
reading the owners' generations — there is no global `_target`. This replaces the
`raise` in `run_local`; the launch `worker` role routes to the decentralized loop
under `schedule: decentralized`.

### D8. On-box convergence verdict: a `validate_dynamics` decentralized arm

Add a `decentralized` arm to `examples/validate_dynamics.py`: train the same
corpus/seed via the in-process decentralized path (owners + workers, push-to-
all-`k`, each owner aggregating independently) and compare best-path perplexity
to the synchronous **anchor**, same as the async/int8/W2/W3/W5 arms. This is the
**dynamics half of §0f for decentralized mode** — does `k` independent
aggregations converge comparably to one-primary replication? The systems half
(real WAN) still rides hardware. Honest caveat in the docstring, like the others.

### D9. Compatibility and the deterministic anchor (non-negotiable)

`schedule: central` is untouched and bit-identical — the decentralized loop is a
new, separate function reached only under `schedule: decentralized`. The
in-process synchronous engine remains the anchor the new arm is measured against.

### D10. Explicitly deferred (and why)

- **Real-WAN systems behavior** for decentralized (latency/NAT/bandwidth/real
  churn, multi-relay'd NAT owners, DCUtR success) — the §0f *systems* half, on
  hardware. The libp2p transport composes via the `link` seam but its
  decentralized validation rides §0f.
- **Up-sizing / W5 sizing in decentralized mode** — sizing lives on the central
  scheduler's lease path; the decentralized self-assign path has no scheduler to
  size, so it's out of scope here (a worker self-paces by what it can serve).
- **Cross-node epoch-numbering convergence proof under adversarial gossip lag** —
  the bounded split-brain D6 already accepts; this loop doesn't change it.

## 4. Implementation slices

All three slices are **landed** (a, b, c ✅). Refinements/fixes beyond the original
plan, in order of importance:

- **The write path: every active owner applies the granted push** (D6, above).
  The slice-c `validate_dynamics` arm caught that the original "primary applies,
  co-owners replicate" plan deadlocks (quorum-gated replication can't propagate a
  single-replica version) — staleness grew unbounded and commits stalled after
  ~8 generations. With independent application the arm converges to **~0.98× the
  synchronous anchor** on one box.
- **Fair scan rotation** (`_pick_assigned_path` start cursor) so a worker
  responsible for several paths round-robins them rather than monopolizing path 0
  (else the per-path target never completes).
- Reject lossy `compress` in decentralized mode (quorum reads need byte-exact
  agreement); `ParameterServer.shutdown` joins its background threads (a daemon
  repl loop mid-torch-op at teardown tripped a C++ `terminate`); a decentralized
  worker count of 1 in the dynamics arm (multi-worker HRW re-roll thrash is a
  systems/throughput property, not the convergence question).

| Slice | Contents | Key tests |
|---|---|---|
| **a** | `run_decentralized_worker` / `_serve_decentralized`: directory-pull + `derive_epoch`, self-assign (D3), quorum-fetch (D4), commit-to-coordinator (D5), push-to-all-`k` (D6). Reuses link/`_train_path`/compression. Driven in tests through a fake in-process link (`addr → owner._handle`), the same transport seam. | An assignee claims its `(p, g)` and a non-member skips; takeover-on-expiry hands a stalled slot to the successor; quorum-fetch loads the majority base (a Byzantine replica is outvoted) and raises when no quorum is reachable; a push applies at **all `k`** owners independently and they **agree** on version + bytes (the grant is single-use per server, so a replay lands nowhere); a full iteration commits and advances the generation, and generations advance fairly across paths. |
| **b** | `run_local` decentralized driver (D7): tracker + owners (gossip/derive) + worker threads, bootstrap first epoch, per-path generation target; drop the `run_local` refusal; launch `worker` role routes to the decentralized loop. | `run_local(schedule: decentralized)` trains to the per-path target and returns the merged bank; the bank moved; a Byzantine owner injected mid-run is divergence-flagged and evicted at the next epoch (liveness). |
| **c** | `validate_dynamics.py` decentralized arm (D8) + docs: close the §0f on-box half in `phase4-design.md`, `viability-roadmap.md`, `internet-scale-plan.md`, `CLAUDE.md`. | The decentralized arm learns and lands within the same tolerance as the async arms vs the anchor; status docs honest (on-box dynamics ✅, WAN systems still owed). |

Rough sizing: a L (the heart), b M, c M.

## 5. Open questions (recommendation first)

1. **Worker directory-refresh cadence** — recommend reusing `gossip_interval`
   (the owners' directory-pull cadence): the worker refreshes its derived epoch
   on the same beat the swarm gossips, so it converges to the same membership
   view within a gossip round. A staler worker is harmless (D2).
2. **Scan order / fairness across a worker's assigned paths** — recommend HRW
   order with a short backoff when assignee-of-nothing; avoids a hot-spin and
   naturally load-balances, matching the central `_completed` fairness the design
   replaced. Revisit only if a real run shows starvation.
3. **`run_local` end condition under churn-free in-process** — recommend the
   per-path generation target with a wall-clock safety timeout (a stalled owner
   shouldn't hang the harness), mirroring the central `fit` budget.
