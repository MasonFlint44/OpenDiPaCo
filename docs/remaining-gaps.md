# opendipaco — remaining architecture gaps

The async **transport** gap list (`async-transport-gaps.md`) is fully closed (all 11).
This document is broader: it captures what's still missing across the **whole project**,
including the items that were explicitly *out of scope* of the 11-gap transport list, so
we have one honest map of where this stands.

This map is written against the "reproduce DiPaCo on a cluster" goal. For the
**volunteer-internet / consumer-hardware** goal — a different lens with different
severities — see [`internet-scale-plan.md`](internet-scale-plan.md), which also holds
the peer-to-peer transition plan.

Each gap is tagged:
- **severity** — P0 (blocks the project's purpose) · P1 (needed for real use) · P2 (hardening/ops) · P3 (nice-to-have)
- whether it's a **deliberate scope decision** (a choice we made, documented) vs. genuinely **unfinished**.

The one-line summary: **the distributed-training machinery is comprehensively built and
tested at toy scale on CPU; what's missing is everything between "the mechanisms work" and
"this reproduces DiPaCo at scale" — above all, empirical validation.**

---

## 1. Method validation — the headline gap  ·  **P0 · ◑ DiPaCo's claim confirmed at modest GPU scale; paper-scale still open**

- **Validation harness exists** — `opendipaco.validation.run_comparison` trains a **dense
  baseline** (DiPaCo with one expert per level, `level_sizes=[1,1]`) and a DiPaCo model
  (K×K) on the **same backbone**, then compares **held-out perplexity at equal inference
  cost** (one path executes). `examples/validate_c4.py` runs it on **real C4** with a
  freshly-trained tokenizer (`train_tokenizer`) and a held-out test split. Verified by
  `tests/test_validation.py`. `run_comparison(device="cuda")` runs the whole thing on GPU.
- **DiPaCo's core claim is now confirmed at modest GPU scale** (full results +
  reproduction in [`docs/gpu-validation.md`](gpu-validation.md)). The **CUDA path is validated
  end-to-end** (never run on GPU before); the full train → outer-step → route → eval pipeline
  runs clean on an RTX 5070 Ti. On **real C4** at **13.4M params/path** (16 paths, hidden 384,
  30 rounds, 8k docs, ~5 GB, ~3 min), a single routed DiPaCo path **beats the matched dense
  baseline in both seeds** (dense ~486 vs DiPaCo 439 / 381). The project's first empirical
  confirmation of DiPaCo's headline claim on real data.
- **DiPaCo's scaling claim is confirmed: its advantage grows with paths when data scales too.**
  Two path-count sweeps tell the full story. *Fixed data* (10k docs): the margin peaks ~9 paths
  (+151) then **collapses to −75 at 36 paths** (more experts split a fixed corpus → data-starved
  + routing harder). *Data scaled with paths* (~2,500 docs/path, 10k→90k): the margin **grows
  monotonically +150 → +267** from 4 → 36 paths, DiPaCo's ppl improving 335 → 242 while dense
  stays ~flat. The decisive contrast is at 36 paths — **−75 (fixed data) vs +267 (scaled)**,
  same model — proving the fixed-data collapse was *starvation, not a path ceiling*: added path
  capacity (more model at fixed inference cost) buys better perplexity when it's fed. This is
  DiPaCo's central claim, on real data. Full results: [`docs/gpu-validation.md`](gpu-validation.md).
- **Open:** **paper scale** (256 paths × 150M, billions of tokens, multi-GPU/nccl) — needs a
  *cluster*, not one 16 GB card; the single-GPU trend is established, the open question is
  whether it holds ~2 orders of magnitude up. Bigger backbones / more seeds are bounded by this
  card's memory/time, not by code (`validate_c4_gpu.py` / `scale_sweep_gpu.py` /
  `data_scaled_sweep_gpu.py` env knobs).
- **The async path changes training dynamics and is unvalidated.** The async coordinator uses
  per-contribution α-weighted outer steps, inverse-staleness damping, and drops the √P rescale
  — so its hyperparameters need tuning *separate* from the synchronous path, and whether it
  converges is an open empirical question. The in-process synchronous `AsyncScheduler` is the
  deterministic anchor.

---

## 2. Transport — deferred / out-of-scope items

The 11 transport gaps are closed; these were never on that list but are real for a
production deployment. Mostly **deliberate scope decisions**, documented here.

- **TLS / payload encryption** — `P2 · done.` The HMAC auth proves key possession but
  doesn't encrypt; **optional TLS now closes that.** `opendipaco.schedule.server_context` /
  `client_context` build the `ssl.SSLContext`s; pass `tls=` to `CoordinatorServer` /
  `Scheduler` / `ParameterServer` (accepted sockets are TLS-wrapped, the handshake driven
  **non-blocking in the reactor I/O thread** so a slow peer can't stall accept) and to
  `run_worker` / `run_sharded_worker` (the scheduler takes `ps_tls=` for its checkpoint RPCs).
  Supports CA verification, hostname checking, and **mutual TLS** (`require_client_cert=True`);
  `client_context(insecure=True)` encrypts-without-verify for a trusted network;
  `generate_selfsigned_cert` is a dev/test helper. TLS composes with the HMAC auth. Verified by
  `tests/test_tls.py` (end-to-end single-node + sharded, CA verification, untrusted-server and
  plaintext-client rejection). Without `tls=` the transport is plaintext as before (fine behind
  an SSH tunnel / on a trusted network).
- **Scheduler / coordinator is a light SPOF.** `P1 · checkpoint/restart ✅, replication out of
  scope.` The sharded `Scheduler.fit(checkpoint_dir=, checkpoint_every=, resume=)` now does a
  **cluster checkpoint** — it saves its clock (`scheduler.pt`) and triggers each `ParameterServer`
  to persist its shard (stable cross-process filenames, atomic writes) — and resumes from it
  (PSs via `resume_dir=`). No automatic failover/replication (out of scope; checkpoint+restart
  is the bar, verified by `test_sharded_checkpoint_resume`).
- **Sharded worker reconnect** is now **done** (`P1 ✅`): `run_sharded_worker(reconnect=,
  reconnect_timeout=)` retries a dropped scheduler/PS connection with backoff; warm caches
  survive. Verified by `test_sharded_worker_reconnects_across_scheduler_restart` (scheduler
  crashes and restarts on the same port; the worker reconnects and the run finishes).
- **Data is not sharded/pre-distributed.** `P2 · deliberate.` The coordinator/scheduler
  still holds the **whole corpus** and ships shards. #11 sharded only the *model bank*; data
  sharding is a separate axis. Production would pre-distribute data to nodes and ship a
  shard *id*, not the bytes.
- **Static module→shard assignment; no dynamic re-sharding.** `P3 · deliberate.`
  `assign_shards` is fixed at startup; a cluster can't rebalance shards or add/remove
  parameter servers at runtime.
- **Reactor compute runs inline on I/O threads.** `P3 · deliberate.` Bank serialization for
  a task happens on the selector thread (fine now because heavy task-builds are rare —
  bounded by in-flight leases — while idle pollers get cheap replies). At extreme scale a
  separate bounded compute pool would overlap it.
- **Observability streaming** — `P2 · done.` `TransportMetrics` is no longer snapshot-only:
  it renders **Prometheus exposition** (`.prometheus()`) and tracks **per-worker liveness**
  (`record_worker` / `active_workers`, also in `.summary()`). `schedule.observability`
  adds a `MetricsExporter` (a stdlib HTTP server: `/metrics` Prometheus, `/` report,
  `/healthz`) and a `MetricsLogger` (periodic structured-JSON snapshots). A server turns
  them on with `server.start_metrics_server()` / `start_metrics_logging()` (stopped on
  `shutdown()`), so a live coordinator/scheduler/PS can be scraped and watched. Verified by
  `tests/test_observability.py`; demo `examples/observe_metrics.py`. (A push-based export to
  a remote backend / a packaged Grafana dashboard is still out of scope.)
- **Inner-optimizer state is not checkpointed in the distributed paths.** `P2 · deliberate.`
  Reset-on-failover means a failed-over path resets its Adam moments (and, in async, private
  modules can roll back to the last sync). The engine's own checkpoint keeps inner-opt state;
  the network paths deliberately don't.

---

## 3. Data pipeline

- **Tokenizer/vocab fidelity** — `P1 · addressed.` `train_tokenizer(texts, vocab_size=32000,
  model="unigram"|"bpe")` trains a fresh SentencePiece-style (Unigram) tokenizer *on the
  data*, like the paper (vs. borrowing `t5-base`), returning a `PreTrainedTokenizerFast`
  drop-in for `tokenize_documents`. Set `BackboneConfig(vocab_size=tokenizer.vocab_size)` to
  keep them from drifting. (Still not the paper's *exact* tokenizer — that's not public — but
  now the paper's *recipe*.)
- **Train/val/test discipline** — `P1 · addressed.` `split_documents(docs, val_fraction=,
  test_fraction=, seed=)` gives a deterministic, reproducible held-out split for honest eval;
  `load_c4_documents(split="validation")` loads C4's real held-out set.
- **Scaled/streaming-resumable ingestion** — `P2 · done.` `data/streaming.py` adds **sharded**
  ingestion (`shard_stream` / `stream_documents` round-robin a stream by global position so each
  of `num_shards` hosts ingests a disjoint `1/N` slice, tokenizing one doc at a time — memory is
  bounded by what you keep, not the corpus size) and **resumability** (`ShardCache` persists
  `(docs, next_index)` in one atomic file; `ingest_c4_shard` resumes a partial shard from exactly
  where it left off, re-deriving only the un-flushed tail — no dup, no loss). `stream_c4_documents`
  / `ingest_c4_shard` wire it to the real C4 stream (native `.skip()` on resume); a dependency-
  injected `source` keeps it testable offline. Verified by `tests/test_streaming_ingestion.py`;
  demo `examples/ingest_c4_sharded.py` (resumes a real-C4 shard, two disjoint shard caches).
  (`load_c4_documents`'s single-file whole-corpus cache stays for the small case. Per-path
  *training* iteration was already deterministic/resumable — seeded `_iter_batches` + the restored
  generation counter — so a resumed run re-derives the same batches.)
- **Routing/featurizer at scale** — `P2 · partly exercised.` `ModelFeaturizer` + k-means +
  discriminative routing now run **on GPU on real C4** in the validation run (§1) and produce a
  working routed shard split; EM re-sharding remains exercised on toy data only.

---

## 4. Synchronous backend (`TorchDistBackend`)

- **Untested at scale / on GPU.** `P1 · unfinished.` The synchronous all-reduce backend
  exists and passes small CPU tests, but has never run with **nccl on a GPU cluster** — and
  for *reproducing the paper's numbers* it's arguably the better path (bandwidth-efficient,
  no parameter server). It's the least-exercised major component.

---

## 5. Hardware / scale

- **GPU path validated; cluster scale still open.** `P0-for-results · ◑.` The **single-GPU
  CUDA path now works end-to-end** — the engine, outer steps, routing, and inference all run on
  an RTX 5070 Ti (16 GB), and the §1 validation run trains both models there in ~3 min / ~5 GB.
  Still untested: **multi-GPU / nccl** comms (the `TorchDistBackend`), the paper's
  150M×256-path scale, and memory behavior near the card's limit. Device handling for the
  *single-GPU* synchronous path is no longer the unknown it was.
- **No load/scale testing.** `P2 · unfinished.` Backpressure (#7) and sharding (#11) are
  structurally correct but never run at the thousands-of-workers / too-big-for-one-node
  scale where they actually matter.
- **Multiprocess tests are localhost-only.** `P2 · deliberate.` Real multi-host networking
  (latency, partitions, partial failures) is unexercised.

---

## 6. Operability / production readiness

- **Launch tooling / CLI** — `P2 · done.` `opendipaco.launch` adds a config-driven launcher:
  one file (`LaunchConfig`, loaded from YAML/TOML/JSON) describes the whole run — model,
  DiLoCo, data, transport (host/port, auth + `accept_keys`, TLS, metrics port), and schedule —
  and the `opendipaco` console script runs each role from it: `coordinator`, `scheduler`,
  `ps --shard-id N`, `worker`, `ingest --shard-id N`, plus `init-config` / `gen-cert` helpers
  and an all-in-one `run` that stands up the entire cluster in one process. It wires together
  everything built: auth/TLS, the Prometheus metrics endpoint, sharded model + resumable
  ingestion, checkpoint/resume. Verified by `tests/test_launch.py` (parsing/validation,
  builders, both modes end-to-end via `run_local`); demo `examples/launch_cluster.py`. (Docker
  images / systemd units / a cluster orchestrator are still out of scope.)
- **Auth-key rotation / per-worker identity** — `P2 · done (rotation + per-worker); secret
  store still out of scope.` A server now takes `accept_keys=` (alongside `auth_key=`) — a
  list of secrets it will accept — so **key rotation** (list old+new during the migration
  window) and **per-worker identity** (each worker holds its own secret; revoke by dropping it)
  both work. Verified constant-time (no early-out on which key matched) by
  `tests/test_polish_bundle.py`. A managed secret *store* (Vault, rotation automation) is still
  out of scope — keys are passed as constructor args.

---

## 7. Method-fidelity — deliberate deviations (documented, not bugs)

The method was audited extensively against the paper; these are **deliberate, documented
choices**, listed for completeness:

- Inner-optimizer (Adam) state **persists across outer rounds** (DiLoCo-style; the paper is
  silent). `deliberate.`
- Inner **gradient clipping on by default** (paper doesn't mention; configurable via
  `inner_grad_clip`). `deliberate.`
- **Train/eval sequence-length split** — `P3 · done.` Now a config knob:
  `DiPaCoConfig(eval_sequence_length=…)` with an `eval_seq_len` property (defaults to
  `sequence_length`); `validation.run_comparison` carries it on the config. The paper
  evaluates at a longer context than it trains on.
- The α-reweighting **denominator convention** (Eq. 2-3) is ambiguous in the paper; we use
  per-path |D_i|, shown to be immaterial (a global constant absorbed by the outer LR).
  `deliberate.`
- **Warm-start from a real pretrained dense model** (`init_from=`) — `P2 · done end-to-end
  (scale still GPU-gated).` Verified *behaviorally*, not just weight-equal: a single path
  warm-started from a real HF `LlamaForCausalLM` reproduces that model's logits to float
  tolerance (`tests/test_polish_bundle.py`) — the whole warm-start → compose → forward
  pipeline is correct. A run against a large real checkpoint at scale remains GPU-gated.

---

## 8. Testing

- **No GPU tests, no real-data tests, no multi-host tests** (see §3–5).
- **Async tests assert behavior, not determinism** — inherent (threaded/async float
  reordering), not a gap, but it means we can't catch fine numerical regressions on those
  paths the way the synchronous engine tests do.

---

## Priority summary

| Priority | Gap | Why |
|---|---|---|
| ◑ P0 | Method validation (§1) | **DiPaCo's claim confirmed at modest GPU scale** (beats matched dense on real C4, 2 seeds); paper-scale (256×150M, cluster) still open. |
| ◑ P0 | GPU validation (§5) | **Single-GPU CUDA path validated end-to-end**; multi-GPU/nccl + paper-scale memory behavior still untested. |
| **P1** | TorchDistBackend at scale (§4) | The likely *right* path for reproducing numbers. |
| ~~P1~~ ✅ | ~~Sharded scheduler checkpoint/resume + worker reconnect (§2)~~ | **Done** — durability parity for the sharded path. |
| ~~P1~~ ✅ | Tokenizer/data fidelity (§3) | **Done** — `train_tokenizer` (paper-style 32k) + `split_documents` (held-out eval) + sharded/resumable streaming ingestion. |
| ~~P2~~ ✅ | ~~TLS / payload encryption (§2)~~ | **Done** — optional TLS (incl. mutual TLS) on every transport role; composes with HMAC auth. |
| ~~P2~~ ✅ | ~~Observability streaming (§2)~~ | **Done** — Prometheus `/metrics` endpoint + structured JSON logging + per-worker liveness on every server. |
| ~~P2~~ ✅ | ~~Streaming-resumable ingestion (§3)~~ | **Done** — sharded `ingest_c4_shard` + crash-consistent `ShardCache`. |
| ~~P2~~ ✅ | ~~Auth rotation / per-worker identity + warm-start E2E (§6, §7)~~ | **Done** — `accept_keys=` accept-list; warm-start verified logit-equal to a real Llama. |
| ~~P3~~ ✅ | ~~Eval-seq-length config knob (§7)~~ | **Done** — `DiPaCoConfig(eval_sequence_length=…)` / `eval_seq_len`. |
| ~~P2~~ ✅ | ~~Launch tooling / CLI (§6)~~ | **Done** — `opendipaco` console script + config-driven roles (`run`/`coordinator`/`scheduler`/`ps`/`worker`/`ingest`). |
| **P2** | Data pre-distribution, secret store (§2, §6) | Documented non-goals / future. |
| **P3** | Dynamic re-sharding, reactor compute pool (§2) | Extreme-scale / polish. |

**Bottom line:** the distributed *systems* work is complete and well-tested, the
production-hardening backlog is **cleared** (TLS, observability streaming, sharded/resumable
ingestion, auth rotation, warm-start E2E, config-driven launch CLI), and — the headline —
**DiPaCo's core claim is now empirically confirmed at modest GPU scale**: on real C4 at
13.4M params/path, a single routed DiPaCo path beats a matched dense model in both tested seeds
(the CUDA path, never run before, works end-to-end). What's genuinely left is **paper-scale**
validation — 256 paths × 150M, billions of tokens, multi-GPU/nccl — which needs a *cluster*,
not this one 16 GB card; plus a handful of documented non-goals (data pre-distribution, dynamic
re-sharding, a managed secret store, Docker/orchestration). On *this* GPU you can still push the
comparison larger (`examples/validate_c4_gpu.py` env knobs), bounded by its memory/time.
