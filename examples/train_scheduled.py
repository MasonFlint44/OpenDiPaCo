"""DiPaCo via the async, fault-tolerant scheduler.

Instead of sweeping every path in lockstep, workers lease path-tasks from a queue,
train them for a generation, and submit pseudo-gradients that the coordinator
aggregates into the outer step -- the paper's decoupled scheduler. Workers that
fail or are preempted have their leases reclaimed and re-queued, so the run
survives flaky workers.

This demo injects transient faults (some path attempts raise) and shows the run
still completes and lands in the same place as a clean run. The whole thing is
in-process with worker threads -- the single-machine reference for the scheduler,
mirroring how ``LocalBackend`` is the reference for the distributed backend.

(Recovery from a *truly dead / preempted* worker -- whose lease times out and is
reclaimed -- is exercised in ``tests/test_scheduler.py``. Keep ``lease_timeout``
comfortably larger than a task's run time so a slow-but-alive worker is never
mistaken for a dead one.)

    python examples/train_scheduled.py
"""

from __future__ import annotations

import torch

from opendipaco import (
    AsyncScheduler,
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
)
from opendipaco.data import ShardedCorpus, pack_sequences
from opendipaco.inference import routed_perplexity
from opendipaco.routing import KMeansRouter, ModelFeaturizer
from opendipaco.schedule import TransientFault


def make_topic_docs(vocab, num_topics=4, docs_per_topic=40, length=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = vocab // num_topics
    return [
        torch.randint(t * span, (t + 1) * span, (length,), generator=g)
        for t in range(num_topics)
        for _ in range(docs_per_topic)
    ]


def flaky(path, attempt):
    """Simulate an unreliable fleet: the first attempt of paths (0,0) and (1,1)
    raises a transient error. Both are re-queued and recover on retry."""
    if attempt == 1 and path in ((0, 0), (1, 1)):
        raise TransientFault()


def main():
    torch.manual_seed(0)
    vocab, seq_len = 128, 32
    bb = BackboneConfig(
        vocab_size=vocab, hidden_size=64, num_attention_heads=4,
        intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64,
    )
    config = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=seq_len)
    topo = config.build_topology()

    docs = make_topic_docs(vocab)

    # Build a serial engine (so per-path optimizer state is checkpointable) and
    # route documents into shards from the model's own features.
    engine = DiPaCoEngine(
        config, DiLoCoConfig(inner_steps=8, inner_lr=1e-3),
        LocalBackend(topo), seed=0, materialize="serial",
    )
    feat = ModelFeaturizer(engine.global_modules(), config)
    kmeans = KMeansRouter(config.num_paths, seed=0).fit(feat([d[:32] for d in docs]))
    corpus = ShardedCorpus.from_documents(docs, kmeans, feat, config.num_paths, seq_len)
    print("shard sizes:", {p: corpus.num_sequences(p) for p in range(config.num_paths)})

    # 3 workers for 4 paths -> at least one worker handles two paths. lease_timeout
    # is well above a task's run time so only genuine failures trigger a re-queue.
    scheduler = AsyncScheduler(engine, num_workers=3, lease_timeout=5.0, max_attempts=3)
    print("training with an unreliable fleet (faults injected)...")
    scheduler.fit(
        corpus, num_generations=12, batch_size=8, total_generations=12,
        log_every=3, fault_hook=flaky,
    )
    print("dropped paths (none expected):", scheduler.dropped or "none")

    eval_seqs = pack_sequences(make_topic_docs(vocab, docs_per_topic=10, seed=99), 64)
    ppl = routed_perplexity(config, engine.global_modules(), eval_seqs, kmeans, feat)
    print(f"routed eval perplexity after fault-tolerant training: {ppl:.2f}")


if __name__ == "__main__":
    main()
