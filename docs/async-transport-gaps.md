# Async scheduler transport — gap analysis

**Status: ALL gaps (#1–11) are now implemented.** The transport is bandwidth-efficient
(stateful workers, no optimizer state on the wire), fault-tolerant (heartbeats, reclaim,
reconnect, checkpoint/restart), observable (metrics), secure (pickle-free + HMAC auth +
size cap), connection-scalable (selector reactor), genuinely **async** (bounded-staleness),
and **shardable** (a light Scheduler + K ParameterServers so no node holds the whole bank).
Each gap below is marked done with implementation notes. Remaining work is *not* transport
gaps but **out-of-scope production concerns** (TLS, scheduler HA, dynamic re-sharding) and —
the project's real open risk — **validating that the method converges on real data at scale**.

## P0 — implemented (2026 update)

The stateless ship-everything design is now **stateful workers with versioned sync**:

- **Workers keep private modules, Adam state, and shards warm** across generations; a
  task ships only the updated **shared** weights (versioned — fetched once per worker
  per generation) plus private/shard **only when a worker is cold** for a path.
  **Optimizer state never crosses the wire** (it was the 2×-weights item). Verified by
  `test_warm_shipping_drops_opt_private_shard`.
- **Locality:** the coordinator owns a path→worker map and re-assigns a path to the
  worker holding it warm; lease-timeout reclaim clears ownership (failover → cold).
- **Reset-on-failover (chosen):** a failed-over path resets its Adam state and re-fetches
  current private modules from the coordinator (which workers upload every submit, so the
  coordinator bank stays consistent). Inner-opt state is **not** checkpointed.
- **Streaming reduce:** the coordinator folds each submitted pseudo-gradient into a
  running accumulator and drops it (memory ≈ one shared-accumulator set, not Σ paths).
- **Durability:** `CoordinatorServer.fit(checkpoint_dir=, checkpoint_every=, resume=)`
  checkpoints (reusing `opendipaco/checkpoint.py`) and restarts; `run_worker(reconnect=)`
  reconnects across a coordinator restart (warm caches survive). Verified by
  `test_checkpoint_resume_continues` and `test_worker_reconnects_across_coordinator_restart`.

What remains irreducible (by design, DiLoCo low-comm): shared-module weights down +
shared pseudo-gradient up, once per generation. Residual cost above that: private modules
uploaded each submit (keeps the coordinator bank consistent; small relative to opt state).

---

### Original analysis (kept as a record — every gap below is now ✅ done)

## What exists today

`schedule/distributed.py` is a **correctness-and-fault-tolerance reference**, built
around deliberately simple choices:

- `CoordinatorServer` — a threaded TCP server wrapping an `AsyncScheduler`. It holds
  the **whole** authoritative bank + outer optimizer and the per-generation task
  queue (`_Generation`). One handler thread per worker connection.
- `run_worker` — a client that builds a scratch serial engine, and for each leased
  task loads the shipped weights into its bank, trains via
  `AsyncScheduler._train_path`, and ships back a `Contribution`.
- **Stateless, interchangeable workers**: each task ships *everything* the worker
  needs (the path's module `state_dict`s, its inner-optimizer state, its data shard,
  the seed/horizon). That is what makes a dead worker a non-event — but it is also
  the root of most gaps below.
- Wire format: length-prefixed `torch.save` payloads (`send_msg`/`recv_msg`).
- Generation-**synchronous**: the coordinator waits for all of generation *g* (or
  drops stragglers after `max_attempts`/timeout) before the outer step and *g+1*.

Validated: an over-TCP run reproduces the in-process result within float tolerance,
and the run survives workers leaving mid-job (`tests/test_scheduler_distributed.py`).

## Severity legend

- **P0 — viability-critical**: without it, a real-scale run is impossible (bandwidth,
  memory, or single-point-of-failure walls).
- **P1 — fleet-hardening**: needed before trusting it on a real, flaky fleet.
- **P2 — security/ops**: needed on untrusted/multi-tenant networks, or to operate a
  run at all.
- **P3 — architecture**: research-grade rebuild closest to the paper's actual system.

| # | Gap | Tier |
|---|-----|------|
| 1 | Full weights re-shipped every task | **P0 ✅ done** |
| 2 | Data shard re-shipped every task | **P0 ✅ done** |
| 3 | Coordinator holds all contributions in RAM before reduce | **P0 ✅ done** |
| 4 | Coordinator is a single point of failure | **P0 ✅ done** |
| 5 | No heartbeats; liveness inferred only from lease expiry | **P1 ✅ done** |
| 6 | No worker reconnect on transient network failure | **P1 ✅ done** |
| 7 | Thread-per-connection, no backpressure/flow control | **P1 ✅ done** |
| 8 | `torch.load` unpickles (RCE); no auth/TLS; unbounded framing | **P2 ✅ done** |
| 9 | No metrics / observability | **P2 ✅ done** |
| 10 | Generation-synchronous (stragglers block) | **P3 ✅ done** |
| 11 | Single coordinator holds the entire bank | **P3 ✅ done** |

---

## P0 — viability-critical

### 1. Full weights re-shipped on every task — **✅ done** (see "P0 — implemented" above)
- **Now:** `CoordinatorServer._next_task` builds `weights = {k: _state_to_cpu(bank[k].state_dict()) for k in path_module_keys}` and ships it for **every lease, every generation**.
- **Problem:** a 150M-param path ≈ 600 MB. 256 paths × ~590 generations (88k steps / τ=150) ≈ tens of **petabytes** of coordinator egress. Hard wall.
- **Needed:** stateful workers + **versioned weight sync**. Workers persist their path's modules across generations; the coordinator tracks a per-module version and ships only the **outer-step delta** (or a "still current" no-op) since the worker last synced. Pulls in **locality-aware scheduling** (prefer assigning a path to a worker that already holds it), while keeping the dead-worker-recovers property (another worker can re-fetch on demand).
- **Depends on / enables:** changes the stateless-worker model that #2, #6, #11 also touch; do these together.

### 2. Data shard re-shipped on every task — **✅ done** (see "P0 — implemented" above)
- **Now:** `_next_task` ships `"shard": corpus.shard(path_idx)` every time.
- **Problem:** redundant data movement on top of #1; shards are static across generations.
- **Needed:** ship each shard **once**, cached on the worker by content hash (or pre-distribute data to nodes and ship only a shard id). Coordinator keeps the corpus only for α-weighting / validation.

### 3. Coordinator buffers all contributions before reducing — **✅ done** (see "P0 — implemented" above)
- **Now:** `fit` calls `_apply_generation(corpus, list(gen.results.values()), g)`; `gen.results` holds every path's full `Contribution` (shared deltas + private state) until the generation closes.
- **Problem:** ~256 × 600 MB ≈ 150 GB resident at peak.
- **Needed:** **stream-reduce** — fold each contribution into the running outer-grad accumulator on arrival (in `_receive`), then discard it. Note this trades away the current sorted-by-path deterministic reduce for arrival-order accumulation (already float-noisy, but call it out).

### 4. Coordinator is a single point of failure — **✅ done** (see "P0 — implemented" above)
- **Now:** fault tolerance covers **worker** death only. `CoordinatorServer.fit` has no periodic persistence; a coordinator crash ends the run, and workers do not survive a coordinator restart (their sockets drop and `run_worker` exits).
- **Problem:** the one indispensable node has no recovery path.
- **Needed:** wire the existing `save_checkpoint`/`load_checkpoint` into the serve loop (checkpoint every N generations), support **restart-from-latest**, and have workers **auto-reconnect** and resume after a coordinator bounce. (Full HA — replicated/consensus coordinator — is out of scope; checkpoint+restart is the pragmatic bar.)

---

## P1 — fleet-hardening

### 5. No heartbeats; liveness == lease expiry — **✅ done**
- **Was:** a worker was presumed dead only when its lease timed out, so a slow task
  looked like a dead worker (forcing `lease_timeout` ≫ task time, slow detection).
- **Now:** a worker runs a background **heartbeat** thread (every `heartbeat_interval`,
  default 3s) while a task is in progress; the coordinator refreshes the lease on each
  heartbeat (`_Generation.refresh_lease`, owner-checked) and uses a short
  `CoordinatorServer(heartbeat_timeout=…)` as the reclaim deadline. So a *slow-but-alive*
  task keeps its lease while a *dead* worker is detected within `heartbeat_timeout`,
  independent of task length. Heartbeats share the socket via a send-lock and get no
  reply. Verified by `test_heartbeats_keep_a_slow_task_alive`. Also: `complete()` is now
  idempotent (a late completion of a reclaimed path removes it from the queue).

### 6. No worker reconnect — **✅ done (P0/P1)**
- **Now:** `run_worker(reconnect=, reconnect_timeout=)` reconnects on a dropped socket
  (with exponential backoff) and resumes requesting; warm caches survive. In-flight work
  is recovered by the lease/heartbeat mechanism. (Implemented with the P0 durability work;
  reconnect-across-restart is covered by `test_worker_reconnects_across_coordinator_restart`.)

### 7. Thread-per-connection, no backpressure — **✅ done**
- **Was:** one handler thread per connection — thousands of workers meant thousands of threads.
- **Now:** a **reactor** — a fixed pool of `io_threads` (default 4) **selector**-based I/O
  loops (non-blocking framed read/parse + buffered write), fed by one accept thread that
  round-robins connections via a self-pipe wakeup. The coordinator's thread/concurrency
  footprint is **bounded and independent of the worker count** (`_Conn`/`_IOThread` in
  `distributed.py`; auth handshake + framing run inline non-blocking; `run_worker` is
  unchanged, still blocking). Backpressure: a `max_connections` cap (default 1024) refuses
  excess connections, and the async coordinator already bounds concurrent *work* to the
  number of in-flight leases (≤ num_paths). Verified by
  `test_bounded_io_threads_serve_many_workers` (2 I/O threads serve 8 workers to completion)
  and `test_max_connections_cap_refuses_excess`.
- **Still open (only at extreme scale):** a separate compute pool to overlap the
  bank-serialization cost off the I/O threads (today it runs inline; fine because heavy
  task-builds are rare — bounded by num_paths leases — while the thousands of idle pollers
  get cheap "idle" replies).

---

## P2 — security & ops

### 8. Unsafe wire protocol — **✅ done**
- **Was:** `torch.load(..., weights_only=False)` unpickled every message (**RCE**); no
  auth; `_recv_n` would read from an attacker-controlled length prefix.
- **Now:** a **pickle-free** serializer in `schedule/wire.py` — the structure is JSON
  (`json.loads`, no code execution; string keys only, tensors/tuples tagged) and tensors
  are raw bytes reconstructed against an explicit **dtype allowlist** and declared shape,
  never executed. An optional **auth handshake** (HMAC-SHA256 challenge-response over a
  shared `auth_key` on `CoordinatorServer` + `run_worker`) rejects unauthorized workers
  before any task is served, and `max_msg_bytes` caps incoming messages (anti-OOM).
  Verified by `tests/test_wire.py` (round-trip incl. bfloat16/int64/bool/tuples/None/NaN,
  unsupported-type/garbage rejection, oversize rejection, auth success/failure) and the
  end-to-end `test_auth_allows_matching_key` / `test_auth_rejects_wrong_key`.
- **Still open (hardening):** no TLS (the channel isn't encrypted — pair with stunnel/an
  SSH tunnel/`ssl` wrapping if confidentiality is needed); the HMAC proves key possession
  but doesn't encrypt payloads.

### 9. No observability — **✅ done**
- **Now:** `CoordinatorServer.metrics` (a `TransportMetrics`) tracks generations, tasks,
  submits, throughput (tasks/s), reclaim/nack/dropped counts, heartbeats, and **bytes on
  the wire** by direction and component (shared weights / private / shard down;
  shared-grad / private up; control). `metrics.report()` prints a summary; `summary()`
  returns a dict. Byte tallies come from `send_msg` (returns bytes) + `recv_msg_sized` +
  `_nbytes` (tensor walker); reclaims from a `_Generation.reclaimed` counter.
- This also **measures the P0 win**: `bytes_opt` stays **0** (optimizer state never on the
  wire) and shards/private modules are charged once per warm worker, not per generation
  (verified by `test_metrics_track_bytes_and_prove_no_optimizer_on_wire`; visible in
  `examples/train_scheduled_distributed.py`'s output).
- **Still open (ops):** structured logging and a live status endpoint (the current
  metrics are an end-of-run/periodic snapshot, not a streaming dashboard).

---

## P3 — architecture (research-grade)

### 10. Generation-synchronous aggregation — **✅ done (the network coordinator is now async)**
- **Was:** the coordinator waited for all paths in generation *g* before the outer step — a
  single slow worker stalled the fleet.
- **Now:** `CoordinatorServer` is **bounded-staleness async** (it *replaced* the synchronous
  network coordinator; the in-process `AsyncScheduler` stays synchronous and is the
  deterministic correctness anchor). There is no generation barrier: workers lease the
  **least-completed** path (balances progress, bounds staleness), and each submit is applied
  as a **per-contribution outer step** on **per-module** SGD+Nesterov optimizers. A submit
  whose base weights have since advanced by more than `staleness_bound` outer steps is
  **rejected** (the path becomes re-eligible); otherwise it is applied with **inverse-staleness
  damping** `1/(1+s)` and the α shard-weight (the sync-only √P rescale is off). `fit` runs to a
  total **update target** (`num_generations·num_paths`), reclaiming dead in-flight leases via
  heartbeats; a slow/dead worker never blocks it. Metrics add `stale_rejected` /
  `accepted_updates` / `mean`+`max` staleness. Verified by `test_async_*` (completes, uneven
  per-path progress, bound enforced, stragglers don't block).
- **Caveat (documented, not hidden):** this **changes training dynamics** — per-contribution
  α-weighted steps, no √P, inverse-staleness damping — so async hyperparameters (outer LR,
  `staleness_bound`, damping) need **separate tuning** from sync, and convergence is unvalidated
  at real scale. The in-process synchronous scheduler remains for deterministic validation.

### 11. Single coordinator holds the whole bank — **✅ done (sharded, parameter-server split)**
- **Was:** one coordinator authoritative for **all** modules — can't hold 256×150M, a funnel.
- **Now:** a **parameter-server split** in `schedule/sharded.py` (added alongside the single
  `CoordinatorServer`, which stays the common case). A light **`Scheduler`** owns the task
  queue + async clock + staleness (and the corpus/α-weights) but **no model weights**; **K
  `ParameterServer`s** each own a disjoint shard of module keys (`assign_shards` round-robins
  `sorted(keys)`, stable across processes) with their weights + per-module outer optimizers +
  versions. `run_sharded_worker`: lease a path from the scheduler → **fetch** its modules from
  the owning PSs (versioned, private cold-only) → `_train_path` → **commit** to the scheduler
  (accept/reject on staleness, returns a damped `push_weight`) → **push** pseudo-gradients to
  the owning PSs (which apply the per-module outer step). Model memory *and* weight bandwidth
  are sharded; the scheduler stays light (its `bytes_up` is ~0 — the weight traffic bypasses
  it). All three roles are `_ReactorServer`s (the reactor was factored into `reactor.py`).
  Per-PS `save_shard`/`load_shard` for checkpointing. Verified by `tests/test_sharded.py`
  (disjoint shards whose union is the full bank + scheduler holds none; training completes and
  every shard moves; staleness-bound commit; a **multi-process** run with shards in separate
  process memories). See `examples/train_sharded.py`.
- **Deferred (documented):** scheduler-side HA (a light SPOF; checkpoint/restart pattern
  carries over), data pre-distribution (the scheduler still holds the corpus — a separate
  axis), dynamic re-sharding. The async-dynamics caveat is inherited.

---

## Dependencies & suggested sequencing

The P0 items are interrelated and should land as **one coherent change** ("stateful
workers + versioned sync + streaming reduce + checkpointable coordinator"): going
stateful (#1) is what makes #2 (ship-shard-once), #6 (reconnect/resume), and #11
(sharded ownership) coherent, and #3 (streaming reduce) + #4 (coordinator checkpoint)
are the memory/durability halves of the same loop.

Rough order:
1. **P0 bundle** — stateful workers, versioned weight/delta sync, ship-shard-once,
   streaming reduce, coordinator checkpoint/restart + worker reconnect.
2. **P1** — heartbeats, reconnect hardening, backpressure/connection scaling.
3. **P2** — pickle-free + authenticated wire format, metrics.
4. **P3** — bounded-staleness async, sharded coordinator.

## Strategic note: is the async path even the right one?

For **replicating the paper's numbers**, the async transport is likely *not* the
fastest route. `TorchDistBackend` (synchronous all-reduce over `nccl` on a GPU
cluster) is the standard, bandwidth-efficient path and avoids most P0 gaps by
construction (no parameter server, no per-task weight shipping). The async scheduler
earns its keep specifically when you need **elastic / preemptible** fleets (spot
instances, volunteer/heterogeneous compute) — which is the paper's deployment story,
but not a prerequisite for reproducing its results. Decide which goal we're serving
before investing in the P0 bundle:

- **Goal = reproduce paper results** → harden/validate `TorchDistBackend` on GPU first;
  treat the async transport as future work.
- **Goal = the paper's elastic async system** → the P0 bundle is the real work.
