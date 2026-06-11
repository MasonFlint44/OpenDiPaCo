"""Multi-process DiPaCo via the socket scheduler transport.

Launches a **coordinator** process and several **worker** processes that talk
over TCP (localhost here, but the workers could be on other machines -- point
them at the coordinator's host/port). The coordinator is **asynchronous
(bounded-staleness)**: workers lease path-tasks, train, and submit pseudo-gradients
that are applied as they arrive (stale ones damped/rejected), so stragglers never
block. One worker is given a finite task budget so it leaves mid-run -- the
coordinator hands its remaining work to the others (elastic membership).

    python examples/train_scheduled_distributed.py

This is the real transport, just colocated on one machine. For a genuine
multi-node run, start the coordinator on one host and run ``run_worker(config,
diloco, COORD_HOST, COORD_PORT)`` on each other host.
"""

from __future__ import annotations

import io
import multiprocessing as mp

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
from opendipaco.schedule import CoordinatorServer, run_worker

VOCAB, SEQ_LEN, GENS, BATCH = 128, 32, 8, 8


def build_config():
    bb = BackboneConfig(
        vocab_size=VOCAB, hidden_size=64, num_attention_heads=4,
        intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=SEQ_LEN)


def diloco():
    return DiLoCoConfig(inner_steps=8, inner_lr=1e-3)


def make_topic_docs(num_topics=4, docs_per_topic=40, length=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = VOCAB // num_topics
    return [
        torch.randint(t * span, (t + 1) * span, (length,), generator=g)
        for t in range(num_topics)
        for _ in range(docs_per_topic)
    ]


def build_corpus(config):
    """Deterministic shards so the coordinator's corpus is well-defined."""
    docs = make_topic_docs()
    engine = DiPaCoEngine(config, diloco(), LocalBackend(config.build_topology()), seed=0)
    feat = ModelFeaturizer(engine.global_modules(), config)
    router = KMeansRouter(config.num_paths, seed=0).fit(feat([d[:32] for d in docs]))
    return ShardedCorpus.from_documents(docs, router, feat, config.num_paths, SEQ_LEN), router, feat


def coordinator_main(port_q, result_q):
    torch.set_num_threads(1)  # colocated procs -> avoid CPU oversubscription
    config = build_config()
    corpus, router, feat = build_corpus(config)
    engine = DiPaCoEngine(config, diloco(), LocalBackend(config.build_topology()),
                          seed=0, materialize="serial")
    scheduler = AsyncScheduler(engine, lease_timeout=30.0)
    server = CoordinatorServer(scheduler, corpus, batch_size=BATCH, host="127.0.0.1", port=0)
    server.start()
    port_q.put(server.port)
    print("[coordinator] serving on port", server.port, flush=True)
    completed = server.fit(num_generations=GENS, total_generations=GENS, log_every=0)
    server.shutdown()
    print("[coordinator] per-path updates (uneven = async):", completed, flush=True)
    print("[coordinator] transport metrics:", flush=True)
    print(server.metrics.report(), flush=True)

    # Evaluate the trained model, then send the bank back as bytes.
    eval_seqs = pack_sequences(make_topic_docs(docs_per_topic=10, seed=99), 64)
    ppl = routed_perplexity(config, engine.global_modules(), eval_seqs, router, feat)
    print(f"[coordinator] routed eval perplexity: {ppl:.2f}", flush=True)
    buf = io.BytesIO()
    torch.save({"ppl": ppl}, buf)
    result_q.put(buf.getvalue())


def worker_main(rank, port, max_tasks=None):
    torch.set_num_threads(1)
    budget = f" (budget {max_tasks} tasks)" if max_tasks else ""
    print(f"[worker {rank}] connecting to coordinator on port {port}{budget}", flush=True)
    run_worker(build_config(), diloco(), "127.0.0.1", port, seed=0, max_tasks=max_tasks)
    print(f"[worker {rank}] done", flush=True)


def main():
    ctx = mp.get_context("spawn")
    port_q, result_q = ctx.Queue(), ctx.Queue()
    coord = ctx.Process(target=coordinator_main, args=(port_q, result_q))
    coord.start()
    port = port_q.get(timeout=60)

    # Three workers; worker 2 leaves after 2 tasks to show elastic membership.
    specs = [(0, port, None), (1, port, None), (2, port, 2)]
    workers = [ctx.Process(target=worker_main, args=spec) for spec in specs]
    for w in workers:
        w.start()

    result_q.get(timeout=300)  # wait for the coordinator to finish + report
    coord.join(timeout=30)
    for w in workers:
        w.join(timeout=30)


if __name__ == "__main__":
    main()
