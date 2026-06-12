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
- `schedule/` — fault-tolerant async layer on top of the engine: in-process `AsyncScheduler` (the deterministic reference), TCP `CoordinatorServer`/`run_worker`, and sharded `Scheduler` + `ParameterServer`s (no node holds the full bank). Bounded staleness; optimizer state never crosses the wire. Wire protocol in `wire.py`/`reactor.py` (pickle-free); HMAC auth + TLS in `tls.py`; per-peer Ed25519 identity in `identity.py` (handshake accepts an identity signature alongside HMAC via `admitted_peers=`; self-certifying signed records); the rendezvous `Tracker` in `tracker.py` (signed peer directory, TTL liveness, enrollment, bootstrap-from-cache); server-side update validation in `guard.py` (non-finite contributions always rejected; optional `max_update_norm` clips); wire compression in `compress.py` (`compress="int8"`: bf16 weights down, int8 pseudo-gradients up with worker-side error feedback; payloads self-describing).
- `launch/` — the `opendipaco` CLI; one YAML config drives every role (`run`, `coordinator`, `scheduler`, `ps`, `worker`, `ingest`, `tracker`; `gen-identity`/`gen-cert` helpers).

**Transport protocol invariants (don't weaken):** every lease carries a unique token the worker must echo on submit/nack/heartbeat (fences zombie workers); in the sharded path a PS push requires the scheduler's commit **grant** (single-use, carries the weight + allowed keys; HMAC-signed when `grant_key` is set on scheduler+PSs — keep it secret from workers — or Ed25519-signed when the scheduler has `identity=` and servers pin `scheduler_pub=`, which refuses HMAC/unsigned grants outright); async checkpoints persist the clock/`_completed`/versions/outer momentum, not just the engine. Module versions are `(epoch, counter)` pairs and a version must always identify identical bytes: owner banks are built with a shared `bank_seed` (so `(0, 0)` matches everywhere) and replication pulls (`include_state`, owner-session gated) ship exact uncompressed state — never bf16 a replication payload.

The async path changes training dynamics (α-weighted outer steps, staleness damping); the synchronous in-process scheduler is the deterministic anchor — be careful equating results across them. Wire compression and `inner_autocast` (bf16 inner loop, fp32 master weights) also change numerics; both default off, and the off paths are kept bit-identical.

**Invariants:** checkpoints are bit-for-bit resumable (`checkpoint.py`, atomic per-rank) and load with `weights_only=True` (allow-list new payload classes via `add_safe_globals`, never flip the flag back); routed eval/EM need the full module bank (`gather_full_bank` first under torch.dist); paper-default weightings are `shard_size_weighting` and `rescale_by_sqrt_sharing`. New `DiLoCoConfig`/`DiPaCoConfig` fields change `config_fingerprint`, so older checkpoints then need `strict=False`.

Validation scripts (`examples/validate_c4_gpu.py`, `scale_sweep_gpu.py`) are env-var driven (`HIDDEN=`, `ROUNDS=`, `NUM_DOCS=`); results and reproduction in `docs/gpu-validation.md`. Known gaps tracked in `docs/remaining-gaps.md`; the volunteer-internet goal (prioritized findings + the peer-to-peer transition plan, phases 0a–0e landed) lives in `docs/internet-scale-plan.md` — keep its per-finding status honest when closing or partially closing a gap.
