"""Single-box dynamics sweep — the no-internet half of plan §0f.

The §0f WAN run answers two separable questions. The **systems** half (real
latency, NAT, bandwidth, churn) genuinely needs distributed nodes. The
**dynamics** half — *do the changes we made to the synchronous DiPaCo algorithm
still converge?* — does not: convergence is a property of the algorithm, not the
network, so it runs on one box. This harness is that half.

The synchronous engine is the deterministic anchor (sum sharing paths'
pseudo-gradients, one outer step per round, √P rescale). We compare its
converged held-out perplexity against the **real in-process async sharded path**
(per-contribution outer steps, inverse-staleness damping, no √P — the actual
production dynamics, driven over localhost with worker oversupply so commits
genuinely race and go stale), with two further deltas layered on:

* ``int8`` wire compression (Phase 0c) — does quantized communication still train?
* robust aggregation (Phase 3, ``robustness: on``) — owner-side quorum buffering
  applies one aggregated outer step across sharing paths instead of one-at-a-time,
  which changes the outer step even with no adversaries.
* ``delta-down`` (W2a, ``down: delta``) — the worker trains from a keyframe + int8
  delta reconstruction of the weights instead of the exact (bf16) weights; does
  the bounded reconstruction error still train?
* ``sparse-up`` (W2b, ``up_density < 1``) — the worker sends only the top fraction
  of each pseudo-gradient (error-feeding the rest); does sparsified communication
  still train?
* ``int4`` (W2c, ``compress: int4``) — int4 per-group pseudo-gradients/deltas; does
  4-bit communication still train? (plus a ``W2 stacked`` arm with all three on.)
* ``8-bit Adam`` (W3d, ``optim_8bit``) — blockwise int8 optimizer moments; does the
  quantized inner optimizer still train?
* ``dedup-private`` (W3d, ``dedup_private``) — aliasing the worker's private modules
  changes warm-round private warming; does it still converge?
* ``het-batch`` (W5, ``het_batch``) — half the workers train a smaller batch, so
  paths see mixed batch sizes round to round (the per-path batch heterogeneity
  throughput-measured task sizing introduces); does it still converge? (One box
  has no real *speed* heterogeneity, so the batch mix is injected directly; the
  straggler/wall-time benefit of sizing rides the WAN run.)

All configs train the **same** corpus/sharding/seed and are evaluated by the same
router-free metric (best-path perplexity on a held-out split), so the numbers are
directly comparable. Env-overridable:

    python examples/validate_dynamics.py
    ROUNDS=40 HIDDEN=128 LEVELS=3 NUM_DOCS=400 WORKERS=6 python examples/validate_dynamics.py

HONEST CAVEAT — what this does and does NOT cover:
  • COVERS, end-to-end, on one box: async staleness dynamics, int8 compression,
    and robust-aggregation dynamics, vs. the synchronous anchor.
  • Does NOT cover: real-WAN *systems* behavior (latency / NAT / bandwidth / real
    churn) — that still needs the multi-node run — nor Phase 4's decentralized
    push-to-all-k *write* path (the worker loop is 0f's other half, unbuilt); its
    aggregation primitive is covered by validate_decentralized.py.
  • This is a SMALL-SCALE check: read the gap-to-anchor *trend*, not absolute
    perplexity. A green run is evidence the deltas converge, not a scale proof.
"""

from __future__ import annotations

import os
import threading

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig, DiPaCoEngine
from opendipaco.backend import LocalBackend
from opendipaco.data import ShardedCorpus, pack_sequences
from opendipaco.inference import compose_path, config_path, perplexity
from opendipaco.schedule import ParameterServer, Scheduler, assign_shards, run_sharded_worker


def _i(name, d):
    return int(os.environ.get(name, d))


HIDDEN = _i("HIDDEN", 96)
HEADS = _i("HEADS", 4)
INTERMEDIATE = _i("INTERMEDIATE", 192)
LAYERS = _i("LAYERS", 2)
VOCAB = _i("VOCAB", 256)
SEQ_LEN = _i("SEQ_LEN", 32)
LEVELS = _i("LEVELS", 2)                 # K -> K x K paths
ROUNDS = _i("ROUNDS", 40)                # sync rounds; async gets ~the same per-path updates
INNER_STEPS = _i("INNER_STEPS", 10)
INNER_LR = float(os.environ.get("INNER_LR", "3e-4"))   # stable regime for the toy task
BATCH = _i("BATCH", 8)
NUM_DOCS = _i("NUM_DOCS", 240)
NUM_SHARDS = _i("NUM_SHARDS", 2)
WORKERS = _i("WORKERS", 0)               # 0 -> default to num_paths (full concurrency)
SEED = _i("SEED", 0)
DEVICE = os.environ.get("DEVICE", "cpu")
# A pass threshold: the async variants should land within this factor of the
# anchor's perplexity (dynamics differ, so we want "comparable", not "equal").
TOL = float(os.environ.get("TOL", "1.5"))


def _topic_docs(n, length, *, seed):
    """Token 'topics' so routing has real structure (a path can specialize)."""
    g = torch.Generator().manual_seed(seed)
    span = VOCAB // 4
    per = max(1, n // 4)
    return [torch.randint(t * span, (t + 1) * span, (length,), generator=g)
            for t in range(4) for _ in range(per)]


def _config():
    bb = BackboneConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_attention_heads=HEADS,
                        intermediate_size=INTERMEDIATE, layers_per_level=[1] * LAYERS,
                        max_position_embeddings=max(64, SEQ_LEN))
    return DiPaCoConfig(backbone=bb, level_sizes=[LEVELS, LEVELS], sequence_length=SEQ_LEN)


def _corpus(config):
    docs = _topic_docs(NUM_DOCS, SEQ_LEN * 2, seed=SEED)
    assign = torch.tensor([i % config.num_paths for i in range(len(docs))])
    return ShardedCorpus.from_assignments(docs, assign, config.num_paths, SEQ_LEN)


def _val_seqs():
    return pack_sequences(_topic_docs(64, SEQ_LEN * 2, seed=SEED + 9999), SEQ_LEN)


def best_path_ppl(config, modules, val_seqs) -> float:
    """Held-out perplexity under the single best-fitting path (router-free, so
    it's identical-by-construction across configs and just measures how well the
    shared+private modules learned)."""
    with torch.no_grad():
        return min(perplexity(compose_path(config, modules, config_path(config, i)), val_seqs)
                   for i in range(config.num_paths))


def train_sync(config, diloco, corpus) -> dict:
    """The deterministic anchor: synchronous DiPaCo rounds."""
    eng = DiPaCoEngine(config, diloco, LocalBackend(config.build_topology()),
                       seed=SEED, device=DEVICE)
    eng.total_rounds = ROUNDS
    eng.fit(corpus, num_rounds=ROUNDS, batch_size=BATCH, log_every=0)
    return eng.global_modules()


def train_async(config, diloco, corpus, *, compress="none", robustness="off",
                down="full", up_density=1.0, optim_8bit=False, dedup_private=False,
                het_batch=False) -> dict:
    """The real in-process async sharded path (localhost TCP, worker oversupply
    so commits race + go stale). Returns the merged authoritative bank from the
    parameter servers -- no node held it whole.

    ``het_batch`` (W5) gives half the workers a smaller advertised batch cap, so
    paths train with **mixed batch sizes round to round** -- the per-path batch
    heterogeneity that throughput-measured sizing introduces. (One box can't
    create real *speed* heterogeneity, so we inject the batch heterogeneity it
    produces directly; the wall-time/straggler benefit rides the WAN run.)"""
    import dataclasses
    # W3d worker-side levers live on the worker's diloco (8-bit inner optimizer;
    # private-copy de-dup in _train_path).
    diloco = dataclasses.replace(diloco, optim_8bit=optim_8bit, dedup_private=dedup_private)
    keys = config.build_topology().module_keys()
    shards = [[k for k, s in assign_shards(keys, NUM_SHARDS).items() if s == i]
              for i in range(NUM_SHARDS)]
    ps_kw = {"down": down}
    if robustness == "on":
        # Small quorum/flush windows so partial buffers flush inside a short run.
        ps_kw |= dict(robustness="on", quorum_target=2, quorum_timeout=0.5,
                      replicate_interval=0.2)
    pss = [ParameterServer(config, sk, diloco, host="127.0.0.1", port=0, **ps_kw)
           for sk in shards]
    for ps in pss:
        ps.start()
    sched = Scheduler(config, corpus, [("127.0.0.1", ps.port) for ps in pss], diloco,
                      batch_size=BATCH, host="127.0.0.1", port=0, compress=compress,
                      down=down, up_density=up_density)
    sched.start()
    n_workers = WORKERS or config.num_paths
    # W5: half the cohort advertises a smaller batch cap -> per-path batch varies.
    def _mb(i):
        return max(1, BATCH // 2) if (het_batch and i % 2 == 1) else None
    workers = [threading.Thread(
        target=run_sharded_worker, args=(config, diloco, ("127.0.0.1", sched.port)),
        kwargs=dict(seed=SEED, heartbeat_interval=2.0, max_batch_size=_mb(i)), daemon=True)
        for i in range(n_workers)]
    for w in workers:
        w.start()
    try:
        sched.fit(num_generations=ROUNDS, total_generations=ROUNDS)
    finally:
        sched.shutdown()
        merged = {k: m for ps in pss for k, m in ps.bank.items()}
        for ps in pss:
            ps.shutdown()
        for w in workers:
            w.join(timeout=10)
    return merged


def main() -> None:
    torch.manual_seed(SEED)
    config = _config()
    diloco = DiLoCoConfig(inner_steps=INNER_STEPS, inner_lr=INNER_LR)
    corpus = _corpus(config)
    val = _val_seqs()
    paths = config.num_paths

    print(f"model: hidden={HIDDEN} layers={LAYERS} vocab={VOCAB}  paths={paths} "
          f"({LEVELS}x{LEVELS})  shards={NUM_SHARDS}  workers={WORKERS or paths}")
    print(f"budget: ~{ROUNDS} per-path updates x {INNER_STEPS} inner steps  "
          f"docs={NUM_DOCS} seq_len={SEQ_LEN}  seed={SEED}\n")

    # Uniform-random perplexity is the no-learning reference; the topic spans
    # give a learnable optimum well below it. "Learned" = clearly under random.
    random_ppl = float(VOCAB)
    learned_ceiling = random_ppl * 0.8
    print(f"  (reference: uniform-random ppl = {random_ppl:.0f}; learning means well below it)\n")

    anchor = best_path_ppl(config, train_sync(config, diloco, corpus), val)
    print(f"  sync (anchor)          best-path val ppl = {anchor:8.3f}")

    # Guard against a vacuous verdict: if the anchor didn't learn or diverged,
    # "comparable to the anchor" means nothing. This synthetic task is sensitive
    # to over-training (push ROUNDS/INNER_LR too high and the toy LM diverges) --
    # that's the data, not the deltas. A credible large run uses real C4.
    if not (anchor < learned_ceiling):
        print(f"\nINCONCLUSIVE: the anchor didn't train stably at this scale "
              f"(ppl {anchor:.1f} vs random {random_ppl:.0f}). Lower INNER_LR / ROUNDS, "
              f"raise NUM_DOCS, or use a real corpus -- the synthetic toy task is fragile, "
              f"not a verdict on the deltas.")
        return

    variants = [
        ("async", dict(compress="none", robustness="off")),
        ("async + int8", dict(compress="int8", robustness="off")),
        ("async + robust agg", dict(compress="none", robustness="on")),
        ("async + delta-down", dict(compress="int8", down="delta")),   # W2a
        ("async + sparse-up", dict(compress="int8", up_density=0.25)),  # W2b
        ("async + int4", dict(compress="int4")),                        # W2c
        ("async + W2 stacked", dict(compress="int4", down="delta", up_density=0.25)),
        ("async + 8-bit Adam", dict(optim_8bit=True)),                   # W3d
        ("async + dedup-private", dict(dedup_private=True)),             # W3d
        ("async + het-batch (W5)", dict(het_batch=True)),                # W5: per-path batch mix
    ]
    worst, all_learned = 1.0, True
    for name, kw in variants:
        ppl = best_path_ppl(config, train_async(config, diloco, corpus, **kw), val)
        ratio = ppl / anchor
        worst = max(worst, ratio)
        learned = ppl < learned_ceiling
        all_learned &= learned
        flag = "OFF-ANCHOR" if ratio > TOL else ("DIDN'T LEARN" if not learned else "ok")
        print(f"  {name:<22} best-path val ppl = {ppl:8.3f}   "
              f"{ratio:4.2f}x anchor   [{flag}]")

    print()
    if all_learned and worst <= TOL:
        print(f"PASS: the anchor learned, and every async variant both learned and stayed "
              f"within {TOL:.2f}x of it (worst {worst:.2f}x). The async / int8 / robust-agg "
              f"deltas track the algorithm on one box.")
    else:
        print(f"WEAK: a variant didn't learn or landed >{TOL:.2f}x off the anchor "
              f"(worst {worst:.2f}x). Re-run with more ROUNDS/NUM_DOCS; if it persists, the "
              f"dynamics need a look BEFORE the WAN run.")
    print("\nCaveat: small-scale, one box. Validates async/int8/robust *dynamics*, not "
          "WAN systems behavior or the Phase 4 decentralized write path. See the module "
          "docstring.")


if __name__ == "__main__":
    main()
