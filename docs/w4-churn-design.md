# W4 design — churn robustness at consumer reality

Status: **design.** Phase 2 built dynamic ownership, k-replication, pull
replication, epoch-bump failover, per-key checkpoints, and a signed recovery
manifest — but timed and tested for **cluster** churn. W4 is the roadmap's
*"eng / tuning"* item (`docs/viability-roadmap.md` §W4): make those mechanisms
survive **home** churn — machines that sleep, reboot, and drop links at rates a
cluster never sees — by (1) measuring failover under injected churn, (2) adding
**graceful suspend/resume** so a closing node hands off cleanly instead of
timing out, and (3) retuning the detection/replication knobs for consumer links,
proven by the harness rather than guessed.

W4 builds **no new subsystem.** Every decision below either tightens an existing
timing, adds a clean-departure fast path beside the existing TTL-expiry slow
path, or instruments what we already have. Each decision (D1–D10) states the
options considered and the recommendation; "open questions" at the end are the
genuine human calls. Slices W4a–W4d land green independently.

## 1. Goal and trust model

**Trust model — unchanged from Phase 2/3.** Owners are trusted-but-unreliable
(Phase 2); Byzantine behavior is Phase 3's robust aggregation + reputation,
already landed and orthogonal. W4 changes nothing about *who* is trusted — only
how fast the swarm reacts when an honest node **leaves**, and how cleanly a node
can announce it's leaving. A departing node is not an adversary; a *forged*
departure (a tombstone for a peer that's still alive) is exactly the eviction
case Phase 3 already reasons about (the tombstone is signed by the leaving
peer's own identity — D3 — so it can only remove itself).

**Goal.** Under the D1 churn model, the run **survives** (never wedges, no
silent weight loss past the documented window), **converges** comparably to the
no-churn anchor (the §0f envelope), and **loses bounded work** per departure —
ideally ~zero for a *graceful* leave and ≤ one `replicate_interval` for an
abrupt one.

## 2. What exists vs. what home churn breaks

| Mechanism (Phase 2/3) | Where | Tuned for cluster | Breaks at home because… |
|---|---|---|---|
| Tracker TTL liveness | `tracker.py` `ttl=120s` | slow, stable nodes | a rebooting laptop is "alive" for 2 min after it's gone |
| Owner-drop hysteresis | `EpochManager.owner_grace=240s` | flap suppression | 4+ min to even *start* promoting a backup |
| Epoch-bump rate limit | `min_epoch_interval=60s` | batch cluster adds | batches *correct* failovers behind a flapper |
| Pull replication window | `replicate_interval=10s` | low loss window | fine — but it's the only thing standing between a dead primary and lost pushes |
| Worker lease / reclaim | `scheduler.py` `lease_ttl≈30s` | one dead worker | fine for crash; a *graceful* worker leave still waits out the lease |
| Departure | TTL expiry only | — | **no clean-leave path at all**: `owner.shutdown()` just stops heartbeating (`sharded.py:1041`); a closing node always pays full `owner_grace` |

The headline gap is the last row: there is a signed `deregister` tombstone in
the tracker (`tracker.py:182`) but **nothing wires it into a node's shutdown
path**, and the scheduler's `watch_tracker`→`EpochManager` only reacts to TTL
expiry, never to an explicit leave. W4's center of gravity is closing that.

## 3. Decisions

### D1. Churn model: a moderate home swarm, made concrete and injectable

All timings are guesses until there's a model to tune against. W4 fixes one,
parameterised so the harness (D6) can sweep it:

- **Session length**: median ~1–2 h, heavy tail (a desktop up for days; a
  laptop for 20 min). Model as exponential with a configurable mean, the
  standard memoryless churn assumption.
- **Event mix per node**: *abrupt drop* (crash / link loss / lid close with no
  warning), *graceful leave* (clean shutdown — the case D3 optimises), and
  *suspend→resume* (sleep then return within minutes — must **not** trigger a
  remap; this is what `owner_grace` flap-suppression is for).
- **Rate**: a few percent of the owner set transitions per minute at the
  target. Aggressive (BOINC-minute-sessions) and conservative
  (cluster-stable) are harness sweep endpoints, not the design point.

This is a *design and test* target, not a wire constant. Nothing in the protocol
encodes it; it parameterises D2's defaults and D6's harness.

### D2. Detection-latency retuning: home-grade defaults, evidence-driven, anchor-safe

The cluster defaults assume detection latency is cheap relative to session
length. At home it isn't. Direction (final numbers come from the D6 harness in
W4d, not this doc):

| Knob | Cluster default | Home direction | Why |
|---|---|---|---|
| `tracker.ttl` | 120 s | ~30–45 s | a reboot should be *detectable* in seconds, not minutes |
| `heartbeat_interval` | 30 s | ~10–15 s | keep ≪ ttl so one missed beat isn't an eviction |
| `owner_grace` | 240 s | ~60–90 s | covers a sleep→resume bounce but not a reboot |
| `min_epoch_interval` | 60 s | ~20–30 s | failovers shouldn't queue behind a flapper |
| `replicate_interval` | 10 s | keep ~10 s | already the loss-window dial; revisit only if D6 shows it dominates |
| `lease_ttl` | 30 s | ~20 s | a dropped worker's task re-leases sooner |

**Anchor discipline (the W3 split).** The **core** configs
(`DiLoCoConfig`/`OwnershipCfg`/`ScheduleCfg` dataclasses) keep today's
conservative defaults verbatim, so `LocalBackend`/`AsyncScheduler` and every
unit test stay bit-identical and fast. The **launch** layer
(`launch/config.py`) adopts the home-grade defaults — these govern only real
multi-node runs and are *strictly availability-improving*: faster failover loses
**fewer** contributions, never more. The one dynamics seam — faster failover
under heavy churn shifts the effective staleness distribution — rides the same
§0f envelope as every other async-dynamics knob and is exercised by the D6
harness, not asserted bit-exact.

### D3. Graceful departure: a signed leave that bypasses `owner_grace` (full handoff)

Today departure is *implicit* (stop heartbeating, wait out TTL+grace). W4 adds
an *explicit* fast path beside it. Three separable pieces, all gated on a
clean shutdown signal (D5 wiring):

1. **Signed fast-deregister.** On shutdown a node sends the tracker's existing
   `deregister` RPC — a record signed with **its own** `PeerIdentity`
   (`tracker.py:272`), so it can only remove itself (no forged eviction of a
   live peer; trust model intact). The tracker already tombstones it. *New:*
   the tombstone must reach the **scheduler's `EpochManager`** so it removes the
   peer **immediately** (skip `owner_grace`) rather than waiting out the grace.
   Mechanically: `watch_tracker`'s directory read already drops absent peers
   after grace; a tombstoned peer is treated as *grace-expired now* — one new
   predicate in `EpochManager.observe`, still `min_epoch_interval`-rate-limited
   so a burst of clean leaves still batches into one bump.
2. **Primary drain.** Before a node exits, for each key it is **primary** of, it
   pushes its highest-version state (weights + outer momentum + version) to the
   current **rank-1** backup via the existing replication path
   (`include_state`, owner-session-gated, **exact bytes** — never compressed,
   the Phase-2 D4 invariant). This collapses the failover loss window from
   ≤ `replicate_interval` to ~0 for a graceful leave: rank-1 is promoted next
   epoch already holding the *last* accepted push, not one window stale.
3. **Worker lease release.** A worker leaving mid-task `nack`s its in-flight
   lease (the existing `nack` path, `scheduler.py:323`, fenced by the lease
   token) so the path is re-leasable **immediately** instead of after
   `lease_ttl`. The partial contribution is dropped exactly as a reclaimed
   lease is today — no new dynamics.

Each piece degrades safely if it fails: if the deregister RPC is lost, you fall
back to TTL+grace; if the drain push fails, you fall back to the
`replicate_interval` window; if the nack is lost, the lease times out. W4 makes
the *common* case clean without making the *failure* case worse.

### D4. Resume: warm restart already works; W4 guarantees the suspend→resume bounce

A node returning after a brief sleep must **not** have caused a remap (else
every laptop lid-close thrashes ownership). Two layers handle this:

- *Within `owner_grace`*: the node never left the desired set (D2/EpochManager
  flap-suppression) — it resumes serving its keys with **no epoch bump**. This
  is the dominant suspend/resume case and is already the EpochManager's design;
  W4 only sizes `owner_grace` (D2) to cover a realistic sleep.
- *After a remap* (gone longer than grace, keys reassigned): the returning node
  re-acquires keys via the normal `syncing→active` cold/delta-sync (Phase 2 D6),
  warm-starting from its **per-key checkpoint** (Phase 2 D7,
  `module_<hash>.pt`) so it delta-syncs forward instead of cold-syncing from
  zero. Nothing new — W4 just adds a harness assertion that a resume after a
  remap converges and serves correct bytes (the manifest-version gate).

A node that fast-deregistered (D3) and then returns is identical to a fresh
join: it re-registers, the next epoch includes it, HRW may hand it keys, it
syncs. The tombstone blocks stale *re-imports* of the old record (tracker
semantics), not a fresh signed re-registration.

### D5. Shutdown signal wiring: one hook, every long-lived role

The fast paths (D3) need a clean-shutdown signal to fire on. Today each role's
loop exits on `self._stop`/`self._dead` or process death; there's no
"about-to-exit, do the handoff" seam.

- Add a `graceful=True` argument to the existing `shutdown()`/`close()` on the
  owner server, scheduler, and worker loop that, *before* setting the stop
  event, runs the role's departure steps (D3): owner → drain + deregister;
  worker → nack in-flight lease + (if also an owner) drain; scheduler → nothing
  new (it's the SPOF until Phase 4; a scheduler restart already resumes epoch
  numbering, Phase 2 post-completion fix).
- The **launch** layer installs a `SIGTERM`/`SIGINT` handler per role that calls
  `shutdown(graceful=True)` with a bounded deadline (a few seconds — a closing
  laptop won't wait), then exits. Past the deadline it falls back to abrupt
  (D3's safe-degrade). This is the only place process-signal handling enters;
  the library stays signal-free and testable in-process.

### D6. Measure-first: a churn stress harness (`examples/validate_churn.py`)

The W3 lesson — *don't tune what you haven't measured* — applies doubly here.
Before retuning D2's defaults, build the harness that proves a setting works.

In-process (like `validate_dynamics.py`/`validate_robustness.py`: real
`Scheduler` + owner servers + `AsyncScheduler` workers over loopback, no
hardware), it:

- drives a D1 churn process against the owner + worker sets — abrupt drop
  (kill the server thread / drop the socket), graceful leave (`shutdown(
  graceful=True)`), suspend→resume (stop heartbeating then resume within/after
  grace), and join (spin a new owner);
- runs a short training job throughout and **asserts survival** (the run
  completes, never wedges) and **convergence** (final loss tracks a no-churn
  control within the §0f tolerance);
- reports **churn metrics**: epochs bumped, keys remapped, time-to-failover
  (departure → backup `active`), contributions dropped (abrupt vs graceful —
  graceful should be ~0), and bytes moved by drains/cold-syncs.

The harness is the W4a deliverable and the evidence base for W4d's defaults.
Like the other `validate_*` scripts its *convergence verdict at scale* still
rides the §0f WAN run; on-box it proves the control plane survives the churn
process and the dynamics track at toy scale.

### D7. k and the replication window under churn

`k=3` (Phase 2 default) tolerates one loss while a replacement syncs; W4 keeps
it and lets D6 confirm it's enough at the D1 rate (raise to k=4 only if the
harness shows simultaneous double-loss stranding keys — a config change, no
code). `replicate_interval` stays the explicit durability/bandwidth dial; D3's
drain makes it irrelevant for *graceful* leaves, so it only bounds *abrupt*-loss
work, and D6 measures whether shrinking it buys anything before paying the
bandwidth.

### D8. Worker churn: already mostly handled, two gaps closed

A leasing worker vanishing mid-task is the most common home event. The existing
lease/reclaim/zombie-fence machinery (`scheduler.py`) already handles the
**abrupt** case correctly (lease times out → path re-queued → token fences the
zombie's late submit). W4 adds: (1) the **graceful** worker leave (D3 part 3 —
nack on shutdown, instant re-lease); (2) `lease_ttl`/`heartbeat_timeout` sized
for consumer links (D2) so a slow-but-alive worker isn't reclaimed mid-task
(false-positive churn). No new failure modes; the §0f dynamics are unchanged
(a reclaimed task is a reclaimed task).

### D9. Compatibility and the deterministic anchor

`ownership: static` and `schedule: central` defaults, `LocalBackend`,
`AsyncScheduler`, `CoordinatorServer` — untouched and bit-identical. Graceful
departure is opt-in behaviour that only *adds* a fast path; with no shutdown
signal it never fires, and the TTL+grace path is byte-for-byte today's. Core
config defaults stay conservative (D2); only the launch layer adopts home
timings. CI exercises the graceful paths in-process (D6 harness + unit tests);
the existing rendezvous parity test (one owner, no churn) stays green.

### D10. Explicitly deferred (and why)

- **Predictive / pre-emptive migration** (move keys off a node *before* it
  leaves, from battery/thermal signals) — needs client telemetry (W6); D3's
  on-shutdown drain is the 90% of it.
- **Per-worker churn-aware task sizing** (give flaky workers smaller tasks so
  less is lost on drop) — that's W5 (throughput-measured sizing); W4 sizes
  *timings*, W5 sizes *tasks*.
- **Scheduler HA** under churn — the scheduler is the run SPOF until Phase 4
  decentralizes it; W4 doesn't try to make the central scheduler churn-tolerant
  beyond its existing checkpoint-resume.
- **Real WAN churn measurement** — the convergence-under-churn verdict at scale
  rides the §0f WAN run (with W1's NAT and W2's bandwidth), like every other
  dynamics property; D6 is the on-box half.

## 4. Implementation slices

Measure-first ordering (the W3 discipline): the harness lands before the tuning
it justifies.

| Slice | Contents | Key tests |
|---|---|---|
| **W4a** | `examples/validate_churn.py`: in-process churn injector (abrupt/graceful/suspend/join) + churn metrics (epochs, remaps, time-to-failover, dropped contributions); minimal test hooks on the owner/worker loops to drive churn deterministically. No behavior change. | Harness runs a job through injected churn and reports metrics; abrupt-kill failover completes with bounded loss (today's behavior, now measured); suspend-within-grace causes 0 bumps. |
| **W4b** | Graceful departure mechanics (D3 parts 1+3): signed fast-deregister → `EpochManager` immediate-removal predicate; worker nack-on-leave; `shutdown(graceful=True)` seam (D5 library half). | Tombstone → next bump removes the peer skipping `owner_grace`, still rate-limited; forged tombstone for a live peer rejected (wrong signer); graceful worker leave re-leases its path immediately; non-graceful path bit-identical to today. |
| **W4c** | Primary drain on departure (D3 part 2): a leaving primary flushes highest-version state to rank-1 over the exact-bytes replication path before exit. | Drain → promoted rank-1 serves the *last* accepted push (loss window ~0) vs ≤`replicate_interval` without; drain payload is exact bytes (version identifies identical content); drain failure degrades to the window (no wedge). |
| **W4d** | Home-grade launch defaults (D2), validated by the W4a harness across the D1 model; `SIGTERM`/`SIGINT` → `shutdown(graceful=True)` launch handlers (D5 launch half); docs + roadmap/plan status. | Retuned defaults survive + converge in the harness across the churn sweep; launch signal handler runs the handoff within its deadline then exits; `viability-roadmap.md`/`internet-scale-plan.md` W4 status honest. |

Rough sizing: W4a M (the harness is the work), W4b M, W4c S–M, W4d S + tuning.

## 5. Open questions (recommendation first)

1. **`owner_grace` vs. sleep duration** — recommend ~60–90 s, sized so a brief
   laptop sleep (lid close → reopen) bounces *within* grace and causes no remap,
   while a reboot (longer) correctly fails over. If D6 shows real sleeps run
   longer (overnight suspend), those *should* remap (the node is genuinely gone)
   and resume via cold/delta-sync — so don't oversize grace to cover them.
2. **Drain deadline on a closing laptop** — recommend a small bounded budget
   (a few seconds) for the graceful handoff; past it, abrupt-degrade. A user
   closing a lid won't wait, and the abrupt path is already safe. Exact budget
   is a D6 measurement (how long does a single-key drain take on a consumer
   uplink).
3. **Tighten `min_epoch_interval` vs. flap cost** — recommend ~20–30 s: fast
   enough that a real failover isn't queued behind an unrelated flapper, slow
   enough to still batch a burst. The tension is real only if churn is bursty
   (many leaves at once); D6's sweep decides.
