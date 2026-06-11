# OpenDiPaCo

[![CI](https://github.com/MasonFlint44/OpenDiPaCo/actions/workflows/ci.yml/badge.svg)](https://github.com/MasonFlint44/OpenDiPaCo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![arXiv](https://img.shields.io/badge/arXiv-2403.10616-b31b1b.svg)](https://arxiv.org/abs/2403.10616)

An open, runnable implementation of **DiPaCo** ([Distributed Path Composition](https://arxiv.org/abs/2403.10616),
Douillard et al., 2024) — train a *modular* language model where a **path** picks one
expert module per level, each path trains on its own slice of data, and the shared
modules are kept loosely in sync with **DiLoCo** low-communication updates. At test
time a router picks **one** path, so only a fraction of the parameters ever run.

```
input ─▶ [ embed ] ─▶ level 0: pick 1 of K₀ ─▶ level 1: pick 1 of K₁ ─▶ [ head ] ─▶ logits
                       ┌──┬──┬──┐               ┌──┬──┬──┐
                       │E0│E1│… │               │E0│E1│… │
                       └──┴──┴──┘               └──┴──┴──┘
```

**It works on real data:** on real C4, one routed DiPaCo path beats a same-inference-cost
dense model, and the margin *grows* as paths and data scale together (+150 → +267 ppl,
4 → 36 paths). See [Validate the method](#validate-the-method).

---

## Install

Uses [**uv**](https://docs.astral.sh/uv/). Pick a torch build with an extra — `cu130`
for an NVIDIA GPU (CUDA 13), or `cpu` — and add `data` (corpora) and `launch` (CLI):

```bash
# GPU:
uv sync --extra cu130 --extra data --extra launch
# CPU-only:
uv sync --extra cpu --extra data --extra launch

source .venv/bin/activate     # then run bare commands (or prefix each with `uv run`)
```

Now `opendipaco`, `python examples/...`, and `pytest` are on your path. (Plain `pip
install -e ".[data]"` also works; uv just gives you CPU/GPU wheel selection + a lockfile.)

---

## Quickstart — a whole cluster in two commands

The fastest way to see it train end-to-end is the CLI's all-in-one local mode:

```bash
opendipaco init-config --out cluster.yaml        # a small, fast starter config
opendipaco run --config cluster.yaml             # coordinator + workers, one process (~5s on CPU)
```

```
per-path updates: {(0, 0): 3, (0, 1): 3, (1, 0): 3, (1, 1): 3}
accepted=13 stale_rejected=0 mean_staleness=0.8 tasks=13 throughput=5.3/s
bytes  down=68.6MB  up=68.3MB  control=4.4KB   optimizer-on-wire=0B
```

The starter config is deliberately tiny so it runs in seconds — scale the model, data, and
`generations` up (and set `run.device: cuda`) for real training. Or the pure-Python demo:

```bash
python examples/train_synthetic.py
```

---

## Train from Python

The core loop: build a config, route documents into per-path shards, fit, evaluate.
`docs` is a `list[torch.LongTensor]` (one per document — see [Real data](#real-data-c4)).

```python
from opendipaco import BackboneConfig, DiPaCoConfig, DiLoCoConfig, LocalBackend, DiPaCoEngine
from opendipaco.routing import ModelFeaturizer, KMeansRouter
from opendipaco.data import ShardedCorpus

config = DiPaCoConfig(
    backbone=BackboneConfig(hidden_size=384, layers_per_level=[2, 2]),
    level_sizes=[4, 4],          # 4×4 = 16 paths
    sequence_length=128,
)

engine = DiPaCoEngine(config, DiLoCoConfig(inner_steps=20), LocalBackend(config.build_topology()),
                      device="cuda")          # or "cpu"

# route documents to paths on the model's own features (the paper's `z`), then shard:
feat = ModelFeaturizer(engine.global_modules(), config)
router = KMeansRouter(config.num_paths).fit(feat([d[:128] for d in docs]))
corpus = ShardedCorpus.from_documents(docs, router, feat, config.num_paths, config.sequence_length)

engine.fit(corpus, num_rounds=30, batch_size=24)
```

Evaluate at **equal inference cost** (one path runs) with a test-time router:

```python
from opendipaco.inference import routed_perplexity
ppl = routed_perplexity(config, engine.global_modules(), test_seqs, router, feat)
```

Useful knobs:

- **Warm-start** every path from a pretrained dense model: `DiPaCoEngine(..., init_from="meta-llama/Llama-3.2-1B")`.
- **Path-private** (never-communicated) modules: `DiPaCoConfig(embedding="private", head="private")`, or interleave private trunk blocks via the `body=[SegmentSpec(...)]` API.
- **Routing featurizers** are interchangeable: `ModelFeaturizer` (recommended), `EmbeddingFeaturizer`, `HFEncoderFeaturizer`, or the dependency-free `BagOfTokensFeaturizer`.
- **EM re-sharding** (re-assign each doc to its lowest-loss path) + overlapping top-k shards: see `examples/train_em.py`.
- **Checkpoint/resume** (bit-for-bit): `save_checkpoint(engine, "ckpt/r1000")` / `load_checkpoint(engine, latest_checkpoint("ckpt"))`.

---

## Validate the method

Does a routed DiPaCo path actually match/beat a dense model of the same size? Run the
comparison yourself (real C4, trains a dense baseline + a K×K DiPaCo, compares held-out
perplexity at equal inference cost):

```bash
python examples/validate_c4_gpu.py                       # auto-detects CUDA
python examples/scale_sweep_gpu.py                       # margin vs #paths (fixed data)
python examples/data_scaled_sweep_gpu.py                 # margin vs #paths (data scales too)

# tune scale via env vars:
HIDDEN=512 LEVELS_SWEEP=2,3,4,5,6 ROUNDS=40 NUM_DOCS=20000 python examples/scale_sweep_gpu.py
```

**Result** (16 GB GPU, real C4): a single DiPaCo path beats matched dense, and the edge
**grows with scale** — at a fixed data budget it peaks then collapses (data starvation),
but when **data scales with paths** the margin climbs monotonically from **+150 ppl (4
paths) to +267 ppl (36 paths)**. That's DiPaCo's central claim, on real data. Full
writeup and reproduction: [`docs/gpu-validation.md`](docs/gpu-validation.md).

---

## Real data (C4)

```python
from opendipaco.data import load_c4_documents, train_tokenizer, split_documents

tok  = train_tokenizer(sample_texts, vocab_size=32000, model="unigram")   # paper-style, on your data
docs = load_c4_documents(num_documents=50_000, tokenizer=tok, cache_path="c4.pt")
train, val, test = split_documents(docs, val_fraction=0.05, test_fraction=0.05)
```

Set `BackboneConfig(vocab_size=tok.vocab_size)` to match. Any text source works via
`data.tokenize_documents(texts, tok)`. For corpora too big for memory, ingest only this
host's shard, resumably:

```bash
opendipaco ingest --config cluster.yaml --shard-id 0     # or data.ingest_c4_shard(...)
```

See `examples/train_c4.py` and `examples/ingest_c4_sharded.py`.

---

## The launch CLI

One config file describes the whole run; every role (`coordinator`, `scheduler`, `ps`,
`worker`) reads the **same** file. Run `opendipaco <command> --config cluster.yaml`:

| command | what it does |
| --- | --- |
| `run` | stand up the **whole cluster in one process** (local dev / smoke test) |
| `coordinator` | single-node async coordinator (the bank lives on one host) |
| `scheduler` | sharded scheduler — task queue + clock, **holds no weights** |
| `ps --shard-id N` | one parameter server owning a disjoint shard of the model |
| `worker [--max-tasks N]` | a worker: lease a path, train, submit |
| `ingest --shard-id N` | resumably ingest this host's data shard |
| `init-config --out f.yaml [--mode sharded]` | write a starter config |
| `gen-cert --out certs/` | self-signed TLS cert for dev |

```bash
# Across hosts (model too big for one box → sharded), same cluster.yaml everywhere:
opendipaco scheduler --config cluster.yaml                 # one host
opendipaco ps        --config cluster.yaml --shard-id 0    # each parameter-server host
opendipaco ps        --config cluster.yaml --shard-id 1
opendipaco worker    --config cluster.yaml                 # each worker host
```

### The config file

`init-config` writes a small, runnable starter with every field populated. Here are the
fields you'll usually edit, shown at a more realistic scale:

```yaml
mode: sharded                 # "coordinator" (single-node bank) | "sharded" (split across PSs)

model:
  vocab_size: 8192
  hidden_size: 384
  num_attention_heads: 6
  intermediate_size: 1024
  layers_per_level: [2, 2]    # transformer layers in each level's expert
  level_sizes: [4, 4]         # experts per level → 16 paths
  sequence_length: 128
  embedding: shared           # "shared" | "private"
  head: shared

diloco:
  inner_steps: 50             # local AdamW steps between syncs
  inner_lr: 0.0004
  outer_lr: 0.7               # outer Nesterov step on the pseudo-gradient

data:
  source: c4                  # "c4" | "synthetic"
  num_documents: 10000
  tokenizer: null             # HF name/path, or null → t5-base
  routing: kmeans             # "kmeans" | "round_robin"

transport:
  host: 0.0.0.0
  port: 29500
  auth_key: null              # shared secret (HMAC); or accept_keys: [old, new] to rotate
  metrics_port: null          # set → Prometheus /metrics endpoint

tls:
  enabled: false              # true + certfile/keyfile/cafile for encryption (incl. mutual TLS)

sharded:
  num_shards: 2
  parameter_servers: [["10.0.0.1", 29501], ["10.0.0.2", 29502]]   # how workers reach each PS

run:
  generations: 30
  batch_size: 24
  device: cpu                 # "cuda" for GPU workers
  checkpoint_dir: ./ckpt
  checkpoint_every: 100
  local_workers: 3            # workers the all-in-one `run` spawns
```

See `examples/launch_cluster.py` and `opendipaco.launch`.

---

## Scaling out (lower-level APIs)

The CLI is the easy path; underneath are three composable layers.

**`torch.distributed`** — one or more paths per process, shared modules all-reduced over
process subgroups (`gloo`/`nccl`):

```bash
torchrun --nproc_per_node=4 examples/train_distributed.py   # world_size must divide num_paths
```

Pass `materialize="serial"` for many paths per rank (trains them one at a time, Adam state
offloaded between rounds — the paper's regime), and `gather_full_bank(...)` before routed
eval/EM (which need the full bank).

**Async scheduler** — workers lease path-tasks from a queue and a coordinator applies the
outer step, so preempted/dead workers don't stall the run. In-process (the deterministic
reference) or over TCP across hosts:

```python
from opendipaco import AsyncScheduler, DiPaCoEngine, LocalBackend
from opendipaco.schedule import CoordinatorServer, run_worker

engine = DiPaCoEngine(config, DiLoCoConfig(), LocalBackend(topo), materialize="serial")
server = CoordinatorServer(AsyncScheduler(engine), corpus, batch_size=16, port=5555); server.start()
server.fit(num_generations=100, checkpoint_dir="ckpt", checkpoint_every=10)
# each worker host:  run_worker(config, DiLoCoConfig(), coordinator_host, 5555)
```

It's bounded-staleness (stale contributions damped/rejected), workers are stateful
(optimizer state never crosses the wire), and it survives worker death, reconnects, and
coordinator restarts. **Sharded** mode (`Scheduler` + K `ParameterServer`s) splits the
bank so no node holds it all. Production knobs: HMAC `auth_key` / `accept_keys` (rotation),
optional **TLS** (`tls=`, incl. mutual TLS), and a **Prometheus** `/metrics` endpoint
(`server.start_metrics_server()`).

See `examples/train_scheduled.py`, `train_scheduled_distributed.py`, `train_sharded.py`,
`run_tls.py`, `observe_metrics.py`.

> **Note:** the async path changes training dynamics (per-contribution α-weighted outer
> steps, staleness damping) and is unvalidated at scale — the synchronous in-process
> `AsyncScheduler` is the deterministic anchor.

---

## Examples

| file | shows |
| --- | --- |
| `train_synthetic.py` | single-process training, no downloads |
| `train_c4.py` | real C4 → tokenize → shard → train → checkpoint → resume |
| `train_em.py` | EM re-sharding loop + a discriminative test-time router |
| `train_distributed.py` | `torchrun` multi-process (`TorchDistBackend`) |
| `train_scheduled.py` | in-process fault-tolerant `AsyncScheduler` |
| `train_scheduled_distributed.py` | the coordinator + workers over TCP |
| `train_sharded.py` | sharded `Scheduler` + `ParameterServer`s |
| `launch_cluster.py` | the launch CLI, end to end |
| `run_tls.py` / `observe_metrics.py` | encrypted transport / live Prometheus metrics |
| `validate_c4_gpu.py`, `scale_sweep_gpu.py`, `data_scaled_sweep_gpu.py` | the method-validation runs |
| `ingest_c4_sharded.py` | sharded, resumable C4 ingestion |

---

## How it maps to the paper

| Paper concept | Here |
| --- | --- |
| Levels `L`, experts per level `Kₗ` | `DiPaCoConfig.level_sizes = [K₁, …, K_L]` |
| A path = one expert per level | `topology.Path`, `model.PathModel` |
| Path backbone (Llama-style) | `BackboneConfig` → HF `LlamaDecoderLayer` stacks |
| Coarse document routing on a prefix repr `z` | `routing.KMeansRouter` + a `Featurizer` |
| Per-path shards (disjoint or top-k overlapping) | `ShardedCorpus.from_documents(..., top_k=)` |
| EM re-sharding (re-assign to lowest-loss path) | `em.reshard_by_loss` |
| Inner AdamW (cosine LR) / outer Nesterov | `DiLoCoConfig`, `optim.diloco` |
| Every path init from the same `θ̄` | `init_from=` / `identical_expert_init=True` |
| Path-private / not-communicated params | `embedding="private"`, `SegmentSpec(sharing="private")` |
| α reweighting (Eq. 2-3), √(sharing) rescale | `shard_size_weighting`, `rescale_by_sqrt_sharing` (paper defaults) |
| Test-time single-path + discriminative router | `inference.routed_perplexity`, `em.fit_discriminative_router` |
| Sub-document re-routing every `W` tokens | `inference.routed_window_perplexity` |

---

## Project layout

```
src/opendipaco/
  config.py        Backbone / DiPaCo / DiLoCo dataclasses
  topology.py      paths, module keys, sharing maps
  model.py         PathModel — compose & run one path
  optim/diloco.py  inner AdamW / outer Nesterov primitives
  backend/         LocalBackend (in-process) + TorchDistBackend
  routing/         featurizers + KMeans / Discriminative routers
  data/            text → tokens, C4 loader, sharding, streaming ingestion
  train/loop.py    DiPaCoEngine — the inner/outer round
  schedule/        AsyncScheduler + CoordinatorServer/run_worker (TCP) + sharded Scheduler/PS
  launch/          config-driven launcher behind the `opendipaco` CLI
  validation.py    dense-vs-DiPaCo comparison
  checkpoint.py    save/load a run (per-rank, atomic, bit-for-bit resume)
  inference.py     single-path / routed evaluation
```

## Status & docs

The method is implemented end to end and **validated on GPU** (DiPaCo beats matched dense
on real C4; the edge grows with scale). The distributed systems — `torch.distributed`
backend, fault-tolerant async scheduler, sharded parameter servers, TLS, metrics, launch
CLI — are complete and tested. What's open is **paper-scale** validation (256 paths × 150M,
multi-GPU/nccl), which needs a cluster.

- [`docs/gpu-validation.md`](docs/gpu-validation.md) — the GPU results + how to reproduce
- [`docs/remaining-gaps.md`](docs/remaining-gaps.md) — honest project-wide gap map
- [`docs/async-transport-gaps.md`](docs/async-transport-gaps.md) — the transport design

## References

- DiPaCo: Distributed Path Composition — https://arxiv.org/abs/2403.10616
- DiLoCo: Distributed Low-Communication Training — https://arxiv.org/abs/2311.08105
