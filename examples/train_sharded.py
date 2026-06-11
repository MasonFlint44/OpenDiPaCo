"""Sharded DiPaCo: a light Scheduler + K ParameterServers (transport gap #11).

The single coordinator holds the whole bank; for a model too big for one node, the
bank is **sharded** across K ParameterServers (each owns a disjoint subset of module
keys) coordinated by one light Scheduler (task queue + async clock; *no weights*). A
worker leases a path, fetches that path's modules from the ParameterServers that own
them, trains, commits to the scheduler, and pushes pseudo-gradients to those servers.

This demo runs all three roles + workers as threads over localhost TCP and prints the
per-server key counts — visibly demonstrating that no single node holds the full bank.
For a real run, start each ParameterServer and the Scheduler on their own hosts and
point ``run_sharded_worker`` at the scheduler's address.

    python examples/train_sharded.py
"""

from __future__ import annotations

import threading

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import ParameterServer, Scheduler, assign_shards, run_sharded_worker

VOCAB, SEQ_LEN, GENS, BATCH, NUM_SHARDS = 128, 32, 8, 8, 2


def make_topic_docs(num_topics=4, docs_per_topic=40, length=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = VOCAB // num_topics
    return [torch.randint(t * span, (t + 1) * span, (length,), generator=g)
            for t in range(num_topics) for _ in range(docs_per_topic)]


def main():
    torch.manual_seed(0)
    bb = BackboneConfig(
        vocab_size=VOCAB, hidden_size=64, num_attention_heads=4,
        intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64,
    )
    config = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=SEQ_LEN)
    diloco = DiLoCoConfig(inner_steps=8, inner_lr=1e-3)

    docs = make_topic_docs()
    assign = torch.tensor([i % config.num_paths for i in range(len(docs))])
    corpus = ShardedCorpus.from_assignments(docs, assign, config.num_paths, SEQ_LEN)

    # Shard the module keys across NUM_SHARDS parameter servers.
    key_shard = assign_shards(config.build_topology().module_keys(), NUM_SHARDS)
    shards = [[k for k, s in key_shard.items() if s == i] for i in range(NUM_SHARDS)]
    pss = [ParameterServer(config, sk, diloco, host="127.0.0.1", port=0) for sk in shards]
    for i, ps in enumerate(pss):
        ps.start()
        print(f"[ps {i}] owns {len(ps.bank)} modules: {sorted(ps.bank)}", flush=True)

    scheduler = Scheduler(config, corpus, [("127.0.0.1", ps.port) for ps in pss], diloco,
                          batch_size=BATCH, host="127.0.0.1", port=0)
    scheduler.start()
    print(f"[scheduler] holds no weights (has bank attr: {hasattr(scheduler, 'bank')}); "
          f"serving on port {scheduler.port}", flush=True)

    workers = [threading.Thread(
        target=run_sharded_worker, args=(config, diloco, ("127.0.0.1", scheduler.port)),
        kwargs=dict(seed=0, heartbeat_interval=2.0), daemon=True) for _ in range(3)]
    for w in workers:
        w.start()

    completed = scheduler.fit(num_generations=GENS, total_generations=GENS)
    scheduler.shutdown()
    for ps in pss:
        ps.shutdown()
    for w in workers:
        w.join(timeout=10)

    print("[scheduler] per-path updates (uneven = async):", completed, flush=True)
    print("[scheduler] metrics:", flush=True)
    print(scheduler.metrics.report(), flush=True)


if __name__ == "__main__":
    main()
