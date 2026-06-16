# W5 design — throughput-measured task sizing & pacing

Status: **complete — W5a/b/c landed.** Heterogeneity *basics* existed (bf16 inner
loop, a self-declared `max_batch` memory cap clamped per task), but tasks were not
sized from **measured** speed, so a slow-but-alive volunteer held a path's lease
far longer than a fast one and straggled every module that path feeds. W5 is the
last B1 item (`docs/viability-roadmap.md` §W5; plan §1.10 "still open"): measure
each worker's effective rate and size each task so its lease completes in a
bounded wall-time.

**W5a** scheduler-observed effective-rate EMA (lease→commit ÷ task work),
dynamics-neutral. **W5b** batch-first shrink sizing to `task_seconds`
(`_size_task_locked`; per-task `inner_steps` only at the batch floor) + audit
size-pinning (a check reuses the primary's `batch`/`inner` — also fixing a latent
heterogeneous-`max_batch` false-divergence bug). **W5c** slow-worker **parking**
(too slow even for the minimum task → `idle`, one request per cooldown re-measures
so a recovered worker rejoins) + launch wiring (`run.task_seconds`/`park_factor`/
`min_task_rate`, off by default) + the `validate_dynamics.py` `het-batch` arm
(per-path batch heterogeneity converges within tolerance on-box, ~0.9× anchor).
Off by default (byte-identical anchor); convergence-at-scale + the wall-time
straggler benefit ride the §0f run.

Like W2/W3's lossy levers, task sizing **changes training dynamics** (it varies
the minibatch, and at the floor the inner-step count), so it is **off by
default** — byte-identical to the anchor — and its convergence verdict rides the
§0f run, with an on-box arm in `examples/validate_dynamics.py`. The
**measurement** half (W5a) is dynamics-neutral and always on.

Each decision (D1–D10) states the options considered and the recommendation;
"open questions" at the end are the genuine human calls. Slices W5a–W5c land
green independently.

## 1. Goal and the straggler mechanic

Async leasing is already self-pacing for **throughput**: a fast worker finishes a
task and asks again sooner, so it naturally does more tasks. The unsolved problem
is **latency** — a slow-but-alive worker keeps heartbeating, so its lease is
*never reclaimed* (that path fences zombie writes, not slow ones); the path it
holds doesn't advance until it finally commits, and in robust/quorum mode the
module that path feeds waits for it (up to `quorum_timeout`). One slow volunteer
paces a whole module.

**Goal.** Size each task's work so any worker's lease completes in ~a target
wall-time `task_seconds`, derived from that worker's **measured** effective rate.
A fast worker gets the full configured task; a slow worker gets a smaller one and
commits on the same cadence; a pathologically slow worker is parked rather than
left holding paths. Sync bandwidth/sec is *equalized* as a side effect (every
worker commits — and thus syncs the model once — every ~`task_seconds`).

**Trust model — unchanged.** Sizing is a scheduler decision from
scheduler-observed timing (D1), so a worker can't lie to dodge audits or grab
work. The Phase 3 redundant-execution check must reproduce the primary's exact
computation, so an audited task pins its size (D8).

## 2. What exists vs. what W5 adds

| | Today | W5 |
|---|---|---|
| Capability | worker advertises `device`, optional `max_batch` (memory cap) | + scheduler **measures** effective rate per worker (no new trust) |
| Batch | clamped to `min(batch_size, max_batch)` | sized to hit `task_seconds`, floor 1, ceiling = the clamp |
| inner_steps | global (`diloco.inner_steps`), not per-task | per-task field; reduced only once batch hits its floor |
| Slow worker | holds a path's lease indefinitely (heartbeating) | lease bounded to ~`task_seconds`; too-slow → parked |
| Pacing | `idle_backoff` (idle workers) | + work-sized pacing (active workers) |

## 3. Decisions

### D1. Throughput: scheduler-observed effective rate (not worker-advertised)

The scheduler already records a lease's issue time and sees its commit; it knows
the task's work (`batch_size × inner_steps × seq_len` tokens, since it sized it).
So it computes an **effective rate** = work ÷ (commit − lease) and keeps a per-
worker EMA. This measures **fetch (downlink) + train** — precisely the
**lease-hold window** (the lease is freed at commit), which is both what
straggles the path *and* what sizing can shrink. It captures a slow **download
link** as well as slow **compute**. The post-commit **push** (uplink) is
deliberately *not* in the window: the lease is already freed when it happens, so
it doesn't straggle the path, and the pseudo-gradient it ships is a fixed-size
per-module delta (independent of `batch`/`inner_steps`) that sizing can't shrink
anyway — so a push-bound worker correctly keeps full tasks (its slow uplink is a
staleness concern, damped by the existing inverse-staleness weighting, not a
sizing one). No new wire field; bootstrap = the first task (no estimate yet) uses
the configured default size.

Rejected: worker-advertised tokens/sec in the capability profile (what plan §1.10
literally suggests). Simpler and available before the first commit, but spoofable
— a worker could under-report to get tiny tasks (dodge audits) or over-report to
hog. Scheduler-observed is Byzantine-robust and needs no protocol change. The
`max_batch` memory cap stays worker-advertised (it's a safety bound on the
worker's own RAM, not a lever it gains by lying — lying only risks OOM-ing
itself).

### D2. Sizing lever: batch first, inner_steps only at the floor

Target work `W* = rate × task_seconds` (tokens). Both batch and inner_steps
reduce work-per-task equally and leave per-task bandwidth identical (one sync per
task either way), so the choice is about **dynamics**:

- **batch_size** varies minibatch noise — the gentle, well-understood change, and
  it keeps the DiLoCo communication structure (the same inner-step count, hence
  the same inner-LR schedule) intact across workers.
- **inner_steps** varies the number of local steps before a sync — it changes the
  inner-LR schedule and the pseudo-gradient magnitude per worker (the roadmap's
  "needs care"), the sharper change.

**Decision:** shrink **batch first** toward a floor of 1; only when batch is
already 1 and the task still overshoots `task_seconds` do we reduce
**inner_steps** toward 1. So `inner_steps` moves only for the genuinely slowest
workers, and never on the common path. `batch* = clamp(round(W*/(inner_steps ×
seq)), 1, cap)`; if `batch*==1` and `inner_steps × seq > W*`, then `inner* =
clamp(round(W*/seq), 1, inner_steps)`.

### D3. Objective: shrink-only toward a target wall-time; configured size is the ceiling

Sizing **only ever shrinks** a task below the configured `(batch_size,
inner_steps)`. A worker fast enough that `W* ≥ default_work` gets the *exact*
configured task — so the fast path is the anchor's task, unchanged. This makes
W5 purely a "don't let slow workers straggle" lever, not a global retuning, and
keeps the ceiling = the deterministic default. (Up-sizing fast workers to do more
per lease — fewer syncs, less bandwidth — is a real future optimization but a
bigger dynamics change; deferred, D10.)

### D4. Bounds, stability, and the first task

- `batch ∈ [1, min(batch_size, max_batch)]`, `inner_steps ∈ [1, diloco
  default]` — sizing never exceeds the configured ceiling or the worker's memory
  cap, and never drops below 1.
- EMA (recommended α ≈ 0.3) absorbs per-task jitter (a cold fetch, a GC pause)
  so one slow task doesn't collapse a worker's size; keyed by the worker's
  stable id (peer id / `worker_id`) so it survives a reconnect.
- The first task from a worker (no estimate) uses the configured default; the
  estimate refines from its first commit on.
- **Fetch overhead is in the rate, on purpose.** The measured rate is work ÷
  (fetch + train), and fetch is ~fixed regardless of task size — so a smaller
  task has a *lower* effective rate. This doesn't spiral: the sizing map has a
  stable fixed point at `work* ≈ compute_rate × (task_seconds − fetch)` (clamped
  at the `(1, 1)` floor, EMA-damped), i.e. sizing converges to the work that fits
  in `task_seconds` *after* subtracting fetch. A worker whose fetch alone exceeds
  `task_seconds` floors at the minimum task and is parked (D7) — correct, it
  genuinely can't keep the cadence.

### D5. Default off; anchor bit-identical

A single launch knob `run.task_seconds` (None = off). Off → every task is the
configured `(batch_size, inner_steps)` → byte-identical to today (and the
in-process anchor / unit tests, which never set it). On → sizing applies. Because
it changes dynamics, the convergence verdict rides §0f; an on-box
`validate_dynamics.py` arm runs a heterogeneous-speed cohort (some workers
artificially slowed) **with** vs **without** sizing and checks the sized run
converges comparably while bounding lease latency. Same discipline as W2/W3/W4.

### D6. Per-task `inner_steps` wire field

The task dict gains an optional `inner_steps`; the worker uses
`task.get("inner_steps", diloco.inner_steps)`. Absent (the default/off path) →
the worker's own `inner_steps`, byte-identical. The worker already receives
`batch_size` per task, so the batch lever needs no new field.

### D7. Slow-worker floor: park, don't strand paths

Even sized to `batch=1, inner_steps=1`, a worker whose estimated time for the
minimum task exceeds `task_seconds × slow_factor` (recommend ×3) is **parked** —
the scheduler returns `idle` instead of leasing it a path, so it never holds a
path hostage. A parked worker keeps requesting; if it speeds up (or surplus work
appears) it resumes. Optional `min_task_rate` hard floor for an absolute cutoff.
This is the dynamics-neutral half of pacing (it changes *who* computes, not the
math) and composes with the existing `idle_backoff`.

### D8. Audited tasks pin their size (Phase 3 correctness)

Redundant execution re-runs a task from its pinned base and compares digests;
the recomputation is only deterministic if the checker uses the **primary's**
`batch_size`, `inner_steps`, seed, and shard — not the checker's own sized
values. So a check task carries the audited task's size (the scheduler knows it —
it sized the primary). Without this, sizing would make every audit falsely
diverge. The pin rides the existing audit record (which already pins `base`).

### D9. Compatibility and the deterministic anchor

`run.task_seconds` unset is the default and is bit-identical; `LocalBackend`,
`AsyncScheduler`, `CoordinatorServer`, and the static path are untouched. The
existing `max_batch` clamp stays and composes (it's the per-worker ceiling).
CI exercises both; a parity test pins "sizing off ⇒ identical task to today".

### D10. Explicitly deferred (and why)

- **Up-sizing fast workers** (bigger-than-configured batch / more inner_steps to
  cut their sync frequency) — a bandwidth optimization and a larger dynamics
  change; revisit by measurement after the shrink-only lever's §0f verdict.
- **Per-worker LR compensation** for the varied batch/inner_steps — a dynamics
  refinement the §0f run would motivate; not built blind.
- **Device-class priors** (seed the rate from `device` before the first commit)
  — a small latency win on a worker's very first task; the EMA converges after
  one commit anyway.

## 4. Implementation slices

Measure-first (the W3/W4 discipline): the estimate lands before the sizing it drives.

| Slice | Contents | Key tests |
|---|---|---|
| **W5a** ✅ | Scheduler-observed effective-rate EMA per worker (lease→commit ÷ task work), exposed in metrics. No behavior change (sizing not wired yet). | Rate EMA tracks a known work/latency; survives reconnect (keyed by id); a missing/instant sample doesn't divide-by-zero or poison the EMA; default task unchanged. |
| **W5b** ✅ | Batch-first task sizing to `task_seconds` (D2/D3/D4) + per-task `inner_steps` field (D6) + audit size-pinning (D8); `run.task_seconds` off by default. | Slow worker → smaller batch, lease ≈ `task_seconds`; batch floors at 1 then inner_steps shrinks; fast worker → exact configured task; sizing off ⇒ byte-identical; audited task + its check use identical size (digests match). |
| **W5c** ✅ | Slow-worker parking (D7) + launch wiring (`run.task_seconds`, floors) + `validate_dynamics.py` heterogeneous-speed arm + docs/roadmap status. | A too-slow worker is parked (gets idle, holds no path) and resumes when it speeds up; launch config maps the knob; the dynamics arm shows sized vs fixed converge comparably while latency is bounded. |

Rough sizing: W5a S–M, W5b M (the heart), W5c S–M.

## 5. Open questions (recommendation first)

1. **`task_seconds` default when enabled** — recommend **~5 s**: long enough to
   amortize fetch/push overhead (sync is a fixed cost per task), short enough to
   bound straggle and keep modules advancing. The §0f run tunes it against real
   links; it's a config knob either way.
2. **EMA responsiveness** — recommend **α ≈ 0.3** (a handful of tasks to adapt):
   fast enough to follow a worker that throttles (thermal, contention), slow
   enough to ignore a one-off cold fetch. Re-evaluated on-box.
3. **Parking aggressiveness** — recommend parking at **3× `task_seconds`** for
   the minimum task: generous enough not to exclude merely-modest hardware,
   strict enough to stop a node that would straggle every module it touches.
