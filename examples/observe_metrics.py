"""Watch a live run: stream the transport metrics over HTTP + structured logs.

Same in-process async coordinator + workers as the other demos, but with
observability turned on: the coordinator exposes a Prometheus ``/metrics`` endpoint
and emits a structured JSON snapshot every second while training runs. The main
thread scrapes the endpoint mid-run so you can see the counters move in real time.

    python examples/observe_metrics.py

Point a Prometheus scraper (or ``curl http://127.0.0.1:<port>/metrics``) at the
printed port to graph throughput / staleness / active workers / bytes-on-wire of a
live coordinator or scheduler. The same `server.start_metrics_server()` works for
the sharded `Scheduler` and each `ParameterServer`.
"""

from __future__ import annotations

import logging
import threading
import time
from urllib.request import urlopen

import torch

from opendipaco import (
    AsyncScheduler,
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
)
from opendipaco.data import ShardedCorpus
from opendipaco.routing import KMeansRouter, ModelFeaturizer
from opendipaco.schedule import CoordinatorServer, run_worker

VOCAB, SEQ_LEN, GENS, BATCH = 128, 32, 12, 8


def build_config():
    bb = BackboneConfig(vocab_size=VOCAB, hidden_size=64, num_attention_heads=4,
                        intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=SEQ_LEN)


def diloco():
    return DiLoCoConfig(inner_steps=8, inner_lr=1e-3)


def topic_docs(per=40, length=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = VOCAB // 4
    return [torch.randint(t * span, (t + 1) * span, (length,), generator=g)
            for t in range(4) for _ in range(per)]


def build_corpus(config):
    docs = topic_docs()
    engine = DiPaCoEngine(config, diloco(), LocalBackend(config.build_topology()), seed=0)
    feat = ModelFeaturizer(engine.global_modules(), config)
    router = KMeansRouter(config.num_paths, seed=0).fit(feat([d[:32] for d in docs]))
    return ShardedCorpus.from_documents(docs, router, feat, config.num_paths, SEQ_LEN)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(1)
    config = build_config()
    engine = DiPaCoEngine(config, diloco(), LocalBackend(config.build_topology()),
                          seed=0, materialize="serial")
    server = CoordinatorServer(AsyncScheduler(engine, lease_timeout=30.0), build_corpus(config),
                               batch_size=BATCH, host="127.0.0.1", port=0)
    server.start()
    exporter = server.start_metrics_server(host="127.0.0.1")  # /metrics, /healthz, /
    server.start_metrics_logging(interval=1.0)                # structured JSON every 1s
    print(f"[coordinator] serving on {server.port}; metrics at "
          f"http://127.0.0.1:{exporter.port}/metrics", flush=True)

    workers = [threading.Thread(target=run_worker,
                                args=(config, diloco(), "127.0.0.1", server.port),
                                kwargs=dict(seed=0, reconnect=False), daemon=True)
               for _ in range(3)]
    for w in workers:
        w.start()

    fit = threading.Thread(
        target=lambda: server.fit(num_generations=GENS, total_generations=GENS, log_every=0))
    fit.start()

    # Scrape the live endpoint a few times while training runs.
    while fit.is_alive():
        time.sleep(1.0)
        scrape = urlopen(f"http://127.0.0.1:{exporter.port}/metrics", timeout=5).read().decode()
        accepted = next((ln.split()[-1] for ln in scrape.splitlines()
                         if ln.startswith("opendipaco_transport_accepted_updates_total ")), "0")
        active = next((ln.split()[-1] for ln in scrape.splitlines()
                       if ln.startswith("opendipaco_transport_active_workers ")), "0")
        print(f"[scrape] accepted_updates={accepted} active_workers={active}", flush=True)

    fit.join()
    print("\n[coordinator] final report:\n" + server.metrics.report(), flush=True)
    server.shutdown()  # also stops the exporter + logger
    for w in workers:
        w.join(timeout=10)


if __name__ == "__main__":
    main()
