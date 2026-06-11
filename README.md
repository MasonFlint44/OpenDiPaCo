# opendipaco

[![CI](https://github.com/MasonFlint44/OpenDiPaCo/actions/workflows/ci.yml/badge.svg)](https://github.com/MasonFlint44/OpenDiPaCo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![arXiv](https://img.shields.io/badge/arXiv-2403.10616-b31b1b.svg)](https://arxiv.org/abs/2403.10616)

An open implementation of **DiPaCo** ([Distributed Path Composition](https://arxiv.org/abs/2403.10616),
Douillard et al., 2024), packaged as usable code.

DiPaCo trains a *modular* language model: parameters are split into **modules**
arranged in **levels**, and a **path** picks one expert module per level. Each
path is trained almost independently on its own slice of data, and the shared
modules are kept loosely in sync with **DiLoCo**-style low-communication updates
(infrequent averaging of pseudo-gradients). At test time a router selects a
single path, so only a fraction of the parameters ever execute — no full model
is materialised.

```
input ──▶ [ embed ] ──▶ level 0: pick 1 of K₀ ──▶ level 1: pick 1 of K₁ ──▶ [ head ] ──▶ logits
                         ┌───┬───┬───┐             ┌───┬───┬───┐
                         │ E0│ E1│ …             │ E0│ E1│ …
                         └───┴───┴───┘             └───┴───┴───┘
```

This repo focuses on a faithful, readable core that runs **two ways from the same
training code**:

- **single-process simulation** (`LocalBackend`) — all paths in one process, the
  reference for validating correctness;
- **distributed** (`TorchDistBackend`) — one process per path, shared modules
  averaged over `torch.distributed` process subgroups. Launch with `torchrun`.

The DiLoCo optimizer is a clean from-scratch reimplementation (no dependency on
the unmaintained OpenDiLoCo / hivemind stack); a hivemind/DHT backend can be
added behind the same `SyncBackend` interface.

## Install

This project uses [**uv**](https://docs.astral.sh/uv/). `uv sync` creates a virtual
env and installs from the lockfile; pick a torch build via an extra:

```bash
uv sync --extra cu130              # GPU (CUDA 13.0 wheel)
uv sync --extra cpu                # CPU-only wheel
uv sync --extra cpu --extra data --extra launch   # + corpora + launch CLI/YAML

uv run pytest                      # run the tests
uv run opendipaco --help           # the launch CLI
uv run python examples/validate_c4_gpu.py
```

(`--extra cpu` and `--extra cu130` are mutually exclusive torch builds; add `--extra
data` for `datasets`/`tokenizers` and `--extra launch` for the CLI's YAML/cert deps.)
Dev tooling (pytest/ruff) is in the `dev` dependency group, installed by `uv sync`
automatically.

<details><summary>Plain pip</summary>

```bash
pip install -e .            # core (torch + transformers)
pip install -e ".[data]"    # + datasets/tokenizers for real corpora
```
(pip resolves torch from PyPI's default index — the CUDA build on Linux. Use uv for
CPU/CUDA wheel selection and a reproducible lockfile.)
</details>

## How it maps to the paper

| Paper concept | Here |
| --- | --- |
| Levels `L`, experts per level `Kₗ` | `DiPaCoConfig.level_sizes = [K₁, …, K_L]` |
| A path = one expert per level | `topology.Path`, `PathModel` |
| Path backbone (150M, Llama-style) | `BackboneConfig` → HF `LlamaDecoderLayer` stacks |
| Training corpus (C4) | `data.load_c4_documents` (streamed, tokenized, cached) |
| Coarse document routing (k-means on a prefix representation `z`) | `routing.KMeansRouter` + a `Featurizer` (`EmbeddingFeaturizer` / `HFEncoderFeaturizer`) |
| Per-path data shards (disjoint or top-k **overlapping**) | `data.ShardedCorpus.from_documents(..., top_k=2)` |
| EM re-sharding (E-step: re-assign docs to lowest-loss path) | `em.reshard_by_loss`, `em.assign_paths_by_loss` |
| Inner AdamW (`H` steps, cosine LR) / outer Nesterov | `DiLoCoConfig` (`inner_lr_schedule="cosine"`), `optim.diloco` |
| Every path initialised from the same model (`θ̄`; experts identical, diverge via data) | `DiPaCoEngine(..., init_from=...)` / `warm_start_modules`, or `identical_expert_init=True` (default) from scratch |
| Document-centric sequences (no cross-document packing) | `ShardedCorpus.from_documents(..., pack_mode="document")` |
| Average a module only over the paths that share it | `topology.paths_through_module`, backend `global_reduce` |
| Path-private / "not-communicated" params (private embedding & trunk blocks in the 16×16) | `embedding="private"`, `head="private"`, or `body=[SegmentSpec(..., sharing="private")]` |
| Shard-size reweighting `α` (Eq. 2-3), √(sharing) rescale | `shard_size_weighting`, `normalize_outer_delta`, `rescale_by_sqrt_sharing` (defaults reproduce the paper) |
| Test-time single-path execution + discriminative router (lowest-loss labels, held-out subset) | `inference.routed_perplexity`, `em.fit_discriminative_router` |
| Per-path early stopping (lowest shard-validation loss) | `ShardedCorpus(..., val_fraction=)`, `engine.compose_best(path)`, `engine.best_val_loss` |
| Sub-document re-routing every `W` tokens (Table 3) | `inference.routed_window_perplexity` |
| Eval excludes the first 32 (routing-prefix) tokens from perplexity | `prefix_len=` in both eval functions (default 32) |

## Quick start (single process)

```bash
python examples/train_synthetic.py
```

Or in code:

```python
from opendipaco import (
    BackboneConfig, DiPaCoConfig, DiLoCoConfig,
    PathTopology, LocalBackend, DiPaCoEngine,
)
from opendipaco.routing import BagOfTokensFeaturizer, KMeansRouter
from opendipaco.data import ShardedCorpus

config = DiPaCoConfig(
    backbone=BackboneConfig(layers_per_level=[6, 6]),  # 12-layer path
    level_sizes=[4, 4],                                # 16 paths
)
topo = PathTopology(tuple(config.level_sizes))

feat = BagOfTokensFeaturizer(config.backbone.vocab_size)
router = KMeansRouter(config.num_paths).fit(feat([d[:32] for d in docs]))   # docs: list[LongTensor]
corpus = ShardedCorpus.from_documents(docs, router, feat,
                                      config.num_paths, config.sequence_length)

engine = DiPaCoEngine(
    config, DiLoCoConfig(inner_steps=50), LocalBackend(topo), device="cuda",
    init_from="meta-llama/Llama-3.2-1B",   # optional: warm-start θ̄ from a pretrained dense model
)
engine.fit(corpus, num_rounds=200, batch_size=16)   # cosine inner LR schedule (default)
```

## Shared vs path-private modules

By default the embedding, head, and routing experts are all **shared** (averaged
across the paths that use them). The paper also makes some parameters
*path-private* — each path keeps its own copy that is **never communicated**. The
embedding and head toggle with one flag:

```python
config = DiPaCoConfig(level_sizes=[16, 16], embedding="private")  # private embed, as in the 16×16
```

For private *trunk blocks* interleaved with routing levels, use the body API:

```python
from opendipaco import SegmentSpec

config = DiPaCoConfig(
    embedding="private",
    body=[
        SegmentSpec(layers=1, sharing="private"),   # private block (per-path)
        SegmentSpec(layers=5, num_experts=16),      # shared routing level (1 of 16)
        SegmentSpec(layers=5, num_experts=16),      # shared routing level (1 of 16)
        SegmentSpec(layers=1, sharing="private"),   # private block (per-path)
    ],
    head="shared",
)
```

Private modules take no outer (DiLoCo) step and are excluded from cross-path
averaging; their locally-trained weights are authoritative.

## Routing featurizers

A router maps a sequence's prefix to a path; the prefix is first turned into a
representation `z` by a `Featurizer`. Four are provided (all satisfy the same
protocol, so they're interchangeable in `KMeansRouter` / `DiscriminativeRouter` /
`ShardedCorpus.from_documents`):

| Featurizer | `z` | use |
| --- | --- | --- |
| `ModelFeaturizer(bank, config)` | pooled hidden states of a reference path of the **DiPaCo model itself** | **recommended** — the paper's `z`; co-evolves with training, path-neutral at init |
| `EmbeddingFeaturizer(embedding)` | masked mean/last pool of token embeddings | cheap; pass a **pretrained** embedding |
| `HFEncoderFeaturizer(model)` | masked pool of a separate **frozen** HF model's hidden states | external encoder; must share the data's tokenizer |
| `BagOfTokensFeaturizer(vocab)` | random projection of token counts | dependency-free fallback |

```python
from opendipaco.routing import ModelFeaturizer, KMeansRouter
feat = ModelFeaturizer(engine.global_modules(), config)   # routes on the model's own features
router = KMeansRouter(config.num_paths).fit(feat([d[:32] for d in docs]))
```

`ModelFeaturizer` needs the reference path's modules in the bank (the full bank,
or a `gather_full_bank` result in distributed).

## EM re-sharding & overlapping shards

Routing is approximate EM: training the paths is the M-step; re-assigning each
document to its lowest-loss path is the E-step. Alternate them, and optionally let
each document live in its top-`k` paths (overlapping shards):

```python
from opendipaco import reshard_by_loss
from opendipaco.data import ShardedCorpus

corpus = ShardedCorpus.from_documents(docs, kmeans, feat, config.num_paths,
                                      config.sequence_length, top_k=2)   # overlap
for _ in range(num_em_rounds):
    engine.fit(corpus, num_rounds=10, batch_size=16)                    # M-step
    corpus = reshard_by_loss(docs, config, engine.global_modules(),     # E-step
                             config.sequence_length, top_k=2)
```

See `examples/train_em.py` for the full loop (ends by fitting a
`DiscriminativeRouter` on the EM assignment for test-time routing).

## Real data (C4)

The same training code runs on a real corpus. `load_c4_documents` streams
[C4](https://huggingface.co/datasets/allenai/c4), tokenizes it with a HuggingFace
tokenizer (default `t5-base`, a ~32k vocab matching the paper), and returns the
`list[LongTensor]` the rest of the stack consumes — with an optional on-disk cache
so re-runs skip the download:

```python
from opendipaco.data import load_c4_documents

docs = load_c4_documents(num_documents=50_000, max_doc_tokens=1024,
                         cache_path="c4_50k.pt")   # cached after the first run
```

Size the backbone's `vocab_size` to your tokenizer (`tokenizer.vocab_size`). The
dataset-agnostic core is `data.tokenize_documents(texts, tokenizer)`, so any text
source works the same way. Requires the data extra (`pip install -e ".[data]"`).
See `examples/train_c4.py` for the full C4 → shard → train → checkpoint → resume flow.

For a closer match to the paper, **train your own ~32k tokenizer on the data**
(`data.train_tokenizer(texts, vocab_size=32000, model="unigram")` — a SentencePiece-style
Unigram tokenizer, a drop-in for `tokenize_documents`) rather than borrowing a pretrained
vocab. For honest evaluation, `data.split_documents(docs, val_fraction=…, test_fraction=…)`
gives a deterministic held-out split (or use C4's real `split="validation"`).

**At scale**, `load_c4_documents`'s one-list-one-file approach won't hold — so
`data.ingest_c4_shard(cache_dir, shard_id=…, num_shards=N, target_docs=…, tokenizer=…)`
ingests only *this host's* `1/N` slice (round-robin by stream position), tokenizes one doc
at a time (memory bounded by what you keep), and caches `(docs, next_index)` atomically so an
interrupted ingest **resumes** from where it stopped — never duplicating or losing a document.
The lower-level `data.stream_documents` / `shard_stream` are dependency-injected on any text
source. See `examples/ingest_c4_sharded.py`.

## Checkpoint & resume

Long runs are crash-safe. A checkpoint is a **directory**: each rank writes its own
slice of the module bank plus the outer/inner optimizer state, the LR-schedule
counters, per-path early-stopping snapshots, and RNG state — enough to resume
**bit-for-bit** (verified in `tests/test_checkpoint.py`, eager and serial).

```python
from opendipaco import save_checkpoint, load_checkpoint, latest_checkpoint

save_checkpoint(engine, "ckpts/round1000", corpus=corpus)   # corpus optional
...
out = load_checkpoint(engine, latest_checkpoint("ckpts"))   # into a fresh engine
corpus = out.get("corpus")                                  # exact shards, no re-routing
engine.fit(corpus, num_rounds=..., batch_size=...)          # continues the cosine schedule
```

The engine itself exposes `state_dict()` / `load_state_dict()`; `checkpoint.py` adds
the per-rank file fan-out, atomic writes, and a `corpus.pt` blob. Single-process runs
are just one `rank0.pt`. Pass `strict=False` to override the config-fingerprint /
world-size guard.

## Distributed (one or more paths per process)

```bash
# 2x2 topology = 4 paths. Any world_size that divides num_paths works:
torchrun --nproc_per_node=4 examples/train_distributed.py   # 1 path per rank
torchrun --nproc_per_node=2 examples/train_distributed.py   # 2 paths per rank
```

`world_size` must divide `num_paths`; each rank trains a contiguous block of
`num_paths / world_size` paths and holds only those paths' modules. A shared
module is averaged within the subgroup of ranks that own a path using it (and
across the rank's own paths when several share it). Works with `gloo` (CPU) or
`nccl` (GPU).

When a rank owns many paths, pass `materialize="serial"` to the engine so it
trains them **one at a time** (peak memory ≈ the owned module bank + a single
path, with inner-optimizer state offloaded to CPU between rounds) instead of
holding all of them co-resident — the paper's many-paths-per-worker regime. It is
numerically identical to the default `"eager"` mode:

```python
engine = DiPaCoEngine(config, DiLoCoConfig(), backend, materialize="serial")
```

Because each rank holds only its own paths, routed evaluation and EM re-sharding
(which build arbitrary paths) need the **full** bank. Assemble it on every rank
with `gather_full_bank` before running them:

```python
from opendipaco import gather_full_bank
from opendipaco.inference import routed_perplexity

full = gather_full_bank(backend, engine.bank, config)   # broadcasts each module from an owner
if rank == 0:
    ppl = routed_perplexity(config, full, eval_seqs, router, featurizer)
```

## Fault-tolerant scheduler

The synchronous engine sweeps every path in lockstep. The paper instead *decouples
workers from paths*: workers lease a path-task from a queue, train it for a
generation, and submit the pseudo-gradient; a coordinator aggregates and applies
the outer step. A worker that is preempted or dies just stops submitting — its
lease expires and the path is re-queued — so the run tolerates failures.

`AsyncScheduler` realizes this **in-process with worker threads** (the
single-machine reference for the scheduler, the way `LocalBackend` is for the
distributed backend; the coordinator/lease/re-queue logic is transport-agnostic,
so a multi-node transport can slot in behind it later). It wraps a
`DiPaCoEngine`, so the outer-step math is the engine's — only the *gathering* of
contributions is queue-driven:

```python
from opendipaco import AsyncScheduler, DiPaCoEngine, LocalBackend

# Serial engine so each path's inner-optimizer state is checkpointable.
engine = DiPaCoEngine(config, DiLoCoConfig(), LocalBackend(topo), materialize="serial")
scheduler = AsyncScheduler(engine, num_workers=4, lease_timeout=30.0, max_attempts=3)
scheduler.fit(corpus, num_generations=100, batch_size=16)   # like engine.fit, but fault-tolerant
```

A generation re-queues failed/preempted path-tasks (up to `max_attempts`); set
`min_fraction < 1.0` to apply the outer step once that fraction of paths report,
tolerating permanent stragglers (dropped paths land in `scheduler.dropped`, with
the last error in `scheduler.errors`). Per-path inner-optimizer state lives on the
coordinator and is handed to whichever worker leases that path next, so the DiLoCo
inner-state persistence holds across generations — and checkpoint/resume works
exactly as for the engine. Keep `lease_timeout` well above a task's run time so a
slow-but-alive worker isn't mistaken for a dead one.

Results are independent of worker count and of any pattern of recovered failures
(up to floating-point noise from threaded execution); see
`tests/test_scheduler.py` and `examples/train_scheduled.py`.

### Across machines (socket transport)

The scheduler runs over **TCP**, with workers in other processes or on other hosts — no
extra dependencies (`schedule/distributed.py`). `CoordinatorServer` is **asynchronous
(bounded-staleness)**: there's no generation barrier, so a slow worker never stalls the
fleet. It hands out the **least-completed** path, and applies each submitted pseudo-gradient
as it arrives — a contribution whose weights are stale by more than `staleness_bound` outer
steps is rejected, the rest applied with inverse-staleness damping `1/(1+s)`. (The in-process
`AsyncScheduler` stays *synchronous and deterministic* — the correctness anchor.)

Workers are **stateful**: they keep each path's private modules, Adam state, and shard
**warm**, so a task ships only the updated *shared* weights (and private/shard the first time
a worker sees a path) — **optimizer state never crosses the wire**. The coordinator holds the
authoritative bank, per-module **versions**, and a path→worker **owner** map (locality).

The coordinator serves connections with a **fixed pool of `io_threads` selector-based I/O
loops** (not one thread per worker), so its footprint stays bounded as the fleet grows;
`max_connections` caps total connections.

```python
# Coordinator (one host):
from opendipaco import AsyncScheduler, DiPaCoEngine, LocalBackend
from opendipaco.schedule import CoordinatorServer

engine = DiPaCoEngine(config, DiLoCoConfig(), LocalBackend(topo), materialize="serial")
server = CoordinatorServer(AsyncScheduler(engine), corpus, batch_size=16, port=5555)
server.start()
server.fit(num_generations=100, checkpoint_dir="ckpts", checkpoint_every=10)
server.shutdown()
# Restart after a crash: same call with resume=True reloads the latest checkpoint.

# Worker (each other host) — point it at the coordinator:
from opendipaco.schedule import run_worker
run_worker(config, DiLoCoConfig(), coordinator_host, 5555)   # reconnects across restarts
```

Fault tolerance: a worker that dies drops its socket, its lease times out, and the path
**fails over cold** to another worker (its Adam state resets; the coordinator ships it the
current private modules). New workers connect any time (elastic membership), and a worker
whose connection blips **reconnects** (its warm caches survive). The coordinator
checkpoints/restarts (`checkpoint_dir`/`checkpoint_every`/`resume`), reusing the engine's
checkpoint format. `run_worker(..., max_tasks=N)` gives a worker a finite budget.

Liveness is tracked with **heartbeats**: a worker pings the coordinator every
`heartbeat_interval` (default 3s) while a task runs, so a short
`CoordinatorServer(heartbeat_timeout=…)` detects a dead worker quickly *without*
reclaiming a slow-but-alive task — task duration and failure detection are decoupled
(keep `heartbeat_interval` a few times below `heartbeat_timeout`).

The coordinator exposes **metrics** (`server.metrics.report()` / `.summary()`):
generations, throughput, reclaim/nack/dropped counts, **per-worker liveness**, and **bytes
on the wire** by direction and component — which makes the bandwidth profile visible and
proves the win (`optimizer-on-wire=0B`; shards shipped once per warm worker, not per
generation). For a **live run**, stream them instead of waiting for the end:
`server.start_metrics_server()` exposes a Prometheus `/metrics` endpoint (`/` report,
`/healthz`) and `server.start_metrics_logging(interval=…)` emits periodic structured-JSON
snapshots (both stop on `shutdown()`; same for the sharded `Scheduler`/`ParameterServer`).
See `examples/observe_metrics.py`.

The wire format is **pickle-free** (`schedule/wire.py`): a JSON structure (parsed without
code execution) plus raw tensor bytes rebuilt against a dtype allowlist — so a received
message can't run code. Set a shared secret to authenticate workers
(`CoordinatorServer(auth_key=…)` + `run_worker(auth_key=…)`, an HMAC challenge-response),
and `max_msg_bytes` caps message size. For **key rotation / per-worker identity**, give the
server an accept-list (`CoordinatorServer(accept_keys=[old, new])` or one secret per worker)
— any listed key authenticates, so you can rotate without downtime or revoke a single worker.
Pin `torch.set_num_threads(1)` per process when colocating many on one box.

**Encryption (optional TLS).** Auth proves key possession but doesn't encrypt; for
confidentiality on an untrusted network, turn on TLS. `opendipaco.schedule.server_context` /
`client_context` build the `ssl.SSLContext`s; pass `tls=` to the server side
(`CoordinatorServer` / `Scheduler` / `ParameterServer`) and to `run_worker` /
`run_sharded_worker` (and `ps_tls=` so the scheduler's checkpoint RPCs to the parameter
servers are encrypted too). Accepted sockets are TLS-wrapped and the handshake is driven
**non-blocking in the reactor I/O thread**, so a slow handshake can't stall the accept loop.
It supports CA verification, hostname checking, and **mutual TLS**
(`server_context(..., require_client_cert=True)`); `client_context(insecure=True)` encrypts
without verifying the server (quick start on a trusted network); `generate_selfsigned_cert`
is a dev helper. TLS composes with `auth_key`. Without `tls=` the channel is plaintext as
before — fine behind an SSH tunnel. See `examples/run_tls.py`.

A runnable multi-process demo is `examples/train_scheduled_distributed.py`;
`tests/test_scheduler_distributed.py` checks an over-TCP run completes to the update target,
ships no optimizer/private/shard once warm, enforces the staleness bound, survives a
coordinator restart, rejects an unauthorized worker, and serves many workers from a fixed
I/O-thread pool. `tests/test_tls.py` adds the encrypted-channel cases (CA verification,
mutual rejection of an untrusted server and a plaintext client, TLS + auth together).

### Sharded (model too big for one node)

When the bank won't fit on one machine, split it: a light **`Scheduler`** (task queue +
async clock; *no weights*) plus **K `ParameterServer`s** that each own a disjoint shard of
module keys. A worker leases a path from the scheduler, **fetches** that path's modules from
the parameter servers that own them, trains, **commits** to the scheduler (which accepts/
rejects on staleness and returns a damped weight), and **pushes** the pseudo-gradients to
those servers — so model memory *and* weight bandwidth are sharded and the scheduler stays
light. `assign_shards` routes keys; `run_sharded_worker` is the client.

```python
from opendipaco.schedule import ParameterServer, Scheduler, assign_shards, run_sharded_worker

key_shard = assign_shards(config.build_topology().module_keys(), num_shards=2)
pss = [ParameterServer(config, [k for k, s in key_shard.items() if s == i], DiLoCoConfig(),
                       host="0.0.0.0", port=0) for i in range(2)]
for ps in pss: ps.start()
sched = Scheduler(config, corpus, [(host_i, ps.port) for ps in pss], DiLoCoConfig(),
                  batch_size=16, port=5555); sched.start()
sched.fit(num_generations=100)
# Each worker (any host): run_sharded_worker(config, DiLoCoConfig(), (sched_host, 5555))
```

See `examples/train_sharded.py` and `tests/test_sharded.py` (which verifies the shards are
disjoint and the scheduler holds no weights — across separate processes). All transport gaps
are now closed; see [`docs/async-transport-gaps.md`](docs/async-transport-gaps.md).

> **Async caveat:** the async coordinator *changes training dynamics* (per-contribution
> α-weighted outer steps, inverse-staleness damping, no √P), so its hyperparameters need
> tuning separate from the synchronous in-process scheduler, and convergence is unvalidated
> at scale. The in-process `AsyncScheduler` remains the deterministic reference.

## Launching a cluster (CLI)

The examples above are in-process; to run across hosts there's a config-driven launcher.
One file (`LaunchConfig`, loaded from YAML/TOML/JSON) describes the whole run — model, DiLoCo,
data, transport (host/port, auth + `accept_keys`, TLS, metrics port), and schedule — and the
`opendipaco` console script runs each role from the *same* file:

```bash
opendipaco init-config --out cluster.yaml --mode sharded   # write a starter config
opendipaco gen-cert    --out certs/                        # (optional) self-signed TLS cert

# all-in-one local cluster (coordinator+workers, or scheduler+PSs+workers) in one process:
opendipaco run --config cluster.yaml

# or, across hosts (same config everywhere):
opendipaco scheduler --config cluster.yaml                 # one host
opendipaco ps        --config cluster.yaml --shard-id 0    # each parameter-server host
opendipaco worker    --config cluster.yaml                 # each worker host
opendipaco ingest    --config cluster.yaml --shard-id 0    # sharded resumable ingestion
```

It wires together everything above — auth/TLS, the Prometheus metrics endpoint, the sharded
model bank, resumable ingestion, checkpoint/resume — from config. `mode: coordinator` runs the
single-node bank; `mode: sharded` runs the scheduler + `num_shards` parameter servers. See
`examples/launch_cluster.py` and `opendipaco.launch`. (Install the extra for YAML configs:
`pip install -e ".[launch]"`.)

## Layout

```
src/opendipaco/
  config.py        Backbone / DiPaCo / DiLoCo dataclasses
  topology.py      paths, module keys, sharing maps
  modules.py       HF Llama-backed Embedding / Level / Head modules
  model.py         PathModel — compose & run one path
  optim/diloco.py  inner AdamW / outer Nesterov primitives
  backend/         SyncBackend: LocalBackend, TorchDistBackend
  routing/         BagOfTokens featurizer, KMeans + Discriminative routers
  data/sharding.py route documents -> disjoint per-path packed shards
  data/text.py     tokenize raw text -> per-document token tensors
  data/c4.py       stream + cache the C4 corpus
  train/loop.py    DiPaCoEngine — the inner/outer round (+ state_dict/load_state_dict)
  schedule/        AsyncScheduler (threads) + CoordinatorServer/run_worker (TCP, multi-node)
                   reactor.py: selector I/O base + metrics; wire.py: pickle-free framing + auth
                   sharded.py: Scheduler + ParameterServer + run_sharded_worker (sharded bank)
  checkpoint.py    save/load a run (per-rank, atomic, resume bit-for-bit)
  inference.py     single-path / routed evaluation
```

## Status & roadmap

Implemented: modular path model with **shared routing experts and path-private
(never-communicated) modules** — including a private embedding / trunk blocks as
in the paper's 16×16 — document routing + sharding, DiLoCo inner/outer
optimization with per-module sharing-aware averaging (α reweighting + √P rescale,
paper defaults), single-process and `torch.distributed` backends, routed
evaluation, tests.

Also implemented: warm-start from a pretrained dense model (`init_from=`), cosine
inner LR schedule (`inner_lr_schedule`), learned routing featurizers
(`EmbeddingFeaturizer` / `HFEncoderFeaturizer`), sub-document re-routing at eval
(`routed_window_perplexity`), EM re-sharding (`reshard_by_loss`), top-k
overlapping shards (`top_k=`), per-path early stopping on shard-validation loss
(`val_fraction` + `engine.compose_best`), a held-out split for the discriminative
router, identical-from-scratch expert init (`identical_expert_init`, the paper's
`θ̄` symmetry), and document-centric sequences (`pack_mode="document"`). The method
is now faithfully covered end to end; see `examples/train_em.py` for the full EM
loop.

Infra: a real-corpus **C4 data pipeline** (`load_c4_documents`, streamed + cached),
**checkpoint/resume** (`save_checkpoint` / `load_checkpoint`, per-rank, atomic,
bit-for-bit resume for both materialize modes), and a **fault-tolerant async
scheduler** (`AsyncScheduler` in-process, plus a **multi-node TCP transport**
`CoordinatorServer` / `run_worker` with elastic membership — custom and
hivemind-free per this project's clean-reimplementation philosophy) are all
implemented; see `examples/train_c4.py`, `examples/train_scheduled.py`, and
`examples/train_scheduled_distributed.py`.

The async transport is **complete** — all 11 gaps closed (stateful workers + versioned
sync, heartbeats, reconnect, the selector reactor, pickle-free/authenticated wire format,
metrics, bounded-staleness async, and a sharded Scheduler + ParameterServers); see
[`docs/async-transport-gaps.md`](docs/async-transport-gaps.md).

**Validation:** `opendipaco.validation.run_comparison` trains a **dense baseline** (DiPaCo
with one expert per level — same backbone, so one path equals the dense model) against a
K×K DiPaCo and compares **held-out perplexity at equal inference cost** (one path executes).
`examples/validate_c4.py` runs it on real C4 with a freshly-trained tokenizer;
`run_comparison(device="cuda")` runs the whole thing on GPU (`examples/validate_c4_gpu.py`).

**Result (modest GPU scale; full writeup in [`docs/gpu-validation.md`](docs/gpu-validation.md)):**
on real C4 at **13.4M params/path** (16 paths, hidden 384, 30 rounds; ~3 min / ~5 GB on a 16 GB
card), a single routed DiPaCo path **beats the matched dense baseline** in both tested seeds
(dense ~486 ppl vs DiPaCo 439 / 381) — the project's first empirical confirmation of DiPaCo's
core claim on real data. Two **path-count sweeps** confirm the scaling thesis: at *fixed* data
DiPaCo's margin peaks ~9 paths then collapses to **−75 ppl at 36 paths** (experts starve, routing
gets harder); but when **data scales with paths**, the margin **grows monotonically from +150 to
+267 ppl** (4 → 36 paths). Same model at 36 paths, −75 vs +267 — the only difference is whether
the capacity is fed. So **DiPaCo's edge over a same-inference-cost dense model widens with scale**,
exactly as the paper claims. (At *toy CPU* size dense wins outright — the advantage is a **scale
phenomenon**.)

What's still missing is **paper-scale** validation — 256 paths × 150M params, billions of
tokens, multi-GPU/nccl — which needs a *cluster*, not a single card; plus a few documented
non-goals. The full project-wide gap map is in
[`docs/remaining-gaps.md`](docs/remaining-gaps.md).

## References

- DiPaCo: Distributed Path Composition — https://arxiv.org/abs/2403.10616
- DiLoCo: Distributed Low-Communication Training — https://arxiv.org/abs/2311.08105
- OpenDiLoCo — https://github.com/PrimeIntellect-ai/OpenDiloco
