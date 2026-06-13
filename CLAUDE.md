# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An implementation of DiPaCo (arXiv:2403.10616): a modular LM where a **path** picks one expert per level, each path trains on its own data shard, and shared modules sync via DiLoCo (inner AdamW steps, then outer Nesterov on the pseudo-gradient `global − local`).

## Commands

```bash
# Install (uv; cpu and cu130 extras are mutually exclusive torch builds)
uv sync --extra cpu --extra data --extra launch      # CPU dev / CI
uv sync --extra cu130 --extra data --extra launch    # NVIDIA GPU

# Tests (CPU; CI runs the same with --frozen, so keep uv.lock in sync with pyproject)
uv run pytest
uv run pytest tests/test_engine.py                   # one file
uv run pytest tests/test_engine.py::test_name        # one test

# Lint (line-length 100, py310)
uv run ruff check src tests examples

# End-to-end smoke test (~5s on CPU)
uv run opendipaco init-config --out cluster.yaml && uv run opendipaco run --config cluster.yaml
```

Tests don't need an editable install — `tests/conftest.py` adds `src/` to `sys.path`.

## Architecture

The training round lives in `src/opendipaco/train/loop.py` (`DiPaCoEngine`). One round: each owned path takes `inner_steps` AdamW steps on its shard → per-module pseudo-gradients are weighted by shard size and summed across sharing paths → outer Nesterov updates the global modules → globals copied back into every path. Everything else composes around this loop.

**Backend abstraction** (`backend/`): the engine is backend-agnostic. `SyncBackend.global_reduce` is the only thing that differs — identity for `LocalBackend` (all paths in-process), an all-reduce over module-sharing subgroups for `TorchDistBackend` (torchrun; `world_size` must divide `num_paths`). The same engine code drives both.

**Layers, bottom to top:**
- `config.py` / `topology.py` — `DiPaCoConfig.level_sizes=[K₁..K_L]` defines the path grid; topology maps module keys → sharing paths; `embedding/head="private"` and `SegmentSpec(sharing="private")` mark never-communicated modules.
- `model.py` — module bank + `PathModel` (composes one path from HF `LlamaDecoderLayer` stacks).
- `routing/` + `data/` — featurizers + KMeans router assign documents to paths; `ShardedCorpus` holds per-path shards; `em.py` re-shards by lowest loss. `data/spec.py` is the decentralized alternative (`data.ship: spec`): the server keeps a `SpecCorpus` (a few-KB shard *recipe* + per-path token counts, zero sequence tensors) and workers `materialize_shard` locally — bit-identical to the bytes path, no per-path val split.
- `schedule/` — fault-tolerant async layer on top of the engine: in-process `AsyncScheduler` (the deterministic reference), TCP `CoordinatorServer`/`run_worker`, and sharded `Scheduler` + `ParameterServer`s (no node holds the full bank). Bounded staleness; optimizer state never crosses the wire. Wire protocol in `wire.py`/`reactor.py` (pickle-free); HMAC auth + TLS in `tls.py`; per-peer Ed25519 identity in `identity.py` (handshake accepts an identity signature alongside HMAC via `admitted_peers=`; self-certifying signed records); the rendezvous `Tracker` in `tracker.py` (signed peer directory, TTL liveness, enrollment, bootstrap-from-cache); server-side update validation in `guard.py` (non-finite contributions always rejected; optional `max_update_norm` clips); wire compression in `compress.py` (`compress="int8"`: bf16 weights down, int8 pseudo-gradients up with worker-side error feedback; payloads self-describing); dynamic replicated ownership in `ownership.py` + `sharded.py` (Phase 2, `ownership.mode: rendezvous`: HRW placement over scheduler-signed epoch records, primary-only writes, pull replication, tracker-liveness failover via `EpochManager`/`watch_tracker`, per-key checkpoints + signed recovery manifest — design and amendments in `docs/phase2-design.md`); Byzantine robustness in `aggregate.py` + `reputation.py` + `ratelimit.py` + `sharded.py` (Phase 3, `robustness.mode: on`: owner-side robust aggregation of shared modules, version-pinned redundant execution feeding per-peer reputation that gates owner eligibility + lease priority + rate-limit generosity, private-module proposal-gating — design and amendments in `docs/phase3-design.md`, adversarial harness `examples/validate_robustness.py`); decentralized scheduling in `assignment.py` + `quorum.py` + `sharded.py` (Phase 4, `schedule.mode: decentralized`: removes the scheduler node — leaderless HRW`(path, gen)` self-assignment, version-vector-lag staleness, each path's primary **owner** mints/version-fences Ed25519 grants (co-owners verify against the epoch record, no central signer) and hosts reputation/rate-limit, Byzantine-owner defense via quorum reads + cross-owner `state_digest` agreement → reputation debit → eviction, signer-less `derive_epoch` over the gossiped directory so the tracker is only a bootstrap seed — design and amendments in `docs/phase4-design.md`, harness `examples/validate_decentralized.py`; the decentralized **worker loop** and its convergence ride the 0f run, like Phase 3, so `run_local` refuses decentralized mode for now).
- `launch/` — the `opendipaco` CLI; one YAML config drives every role (`run`, `coordinator`, `scheduler`, `ps`, `worker`, `ingest`, `tracker`; `gen-identity`/`gen-cert` helpers).

**Transport protocol invariants (don't weaken):** every lease carries a unique token the worker must echo on submit/nack/heartbeat (fences zombie workers); in the sharded path a PS push requires the scheduler's commit **grant** (single-use, carries the weight + allowed keys; HMAC-signed when `grant_key` is set on scheduler+PSs — keep it secret from workers — or Ed25519-signed when the scheduler has `identity=` and servers pin `scheduler_pub=`, which refuses HMAC/unsigned grants outright; in `schedule.mode: decentralized` there is no scheduler, so the grant is signed by the path's **primary owner** and co-owners verify the signer against the epoch record — `grant_signed_by`); async checkpoints persist the clock/`_completed`/versions/outer momentum, not just the engine. Module versions are `(epoch, counter)` pairs and a version must always identify identical bytes: owner banks are built with a shared `bank_seed` (so `(0, 0)` matches everywhere) and replication pulls (`include_state`, owner-session gated) ship exact uncompressed state — never bf16 a replication payload.

The async path changes training dynamics (α-weighted outer steps, staleness damping); the synchronous in-process scheduler is the deterministic anchor — be careful equating results across them. Wire compression, `inner_autocast` (bf16 inner loop, fp32 master weights), and `robustness.mode: on` (robust aggregation buffers + applies one step across sharing paths instead of one-at-a-time) all change numerics; all default off, and the off paths are kept bit-identical. Robustness is the first feature whose *point* is a dynamics property — its convergence verdict rides a real run (`examples/validate_robustness.py` validates the aggregation primitive; the 0f WAN run validates end-to-end), not unit tests. `schedule.mode: decentralized` (Phase 4) is the same kind of bet: its control plane (assignment, owner-minted grants, quorum reads, eviction, gossip-derived epochs) is landed and tested, but the decentralized worker loop + the push-to-all-`k` write path's convergence ride the 0f run, so it is off by default and `run_local` refuses it.

**Invariants:** checkpoints are bit-for-bit resumable (`checkpoint.py`, atomic per-rank) and load with `weights_only=True` (allow-list new payload classes via `add_safe_globals`, never flip the flag back); routed eval/EM need the full module bank (`gather_full_bank` first under torch.dist); paper-default weightings are `shard_size_weighting` and `rescale_by_sqrt_sharing`. New `DiLoCoConfig`/`DiPaCoConfig` fields change `config_fingerprint`, so older checkpoints then need `strict=False`.

Validation scripts (`examples/validate_c4_gpu.py`, `scale_sweep_gpu.py`) are env-var driven (`HIDDEN=`, `ROUNDS=`, `NUM_DOCS=`); results and reproduction in `docs/gpu-validation.md`. Known gaps tracked in `docs/remaining-gaps.md`; the volunteer-internet goal (prioritized findings + the peer-to-peer transition plan; phases 0a–0e, 1, and 2 landed) lives in `docs/internet-scale-plan.md` — keep its per-finding status honest when closing or partially closing a gap.
