"""The socket transport with TLS encryption (coordinator + workers over TLS).

Same multi-process DiPaCo run as ``train_scheduled_distributed.py``, but the
channel is **encrypted**: the coordinator presents a (self-signed, for the demo)
certificate and the workers verify it against that cert as their CA and check the
hostname. TLS is layered *under* the HMAC ``auth_key`` here, so the run has both
confidentiality (TLS) and worker authentication (the shared secret).

    python examples/run_tls.py

For a real deployment, replace ``generate_selfsigned_cert`` with a proper cert/key
(and a real CA), or drop verification with ``client_context(insecure=True)`` for a
quick encrypted-but-unverified channel on an otherwise-trusted network. To run
without TLS at all, just omit the ``tls=`` arguments.
"""

from __future__ import annotations

import multiprocessing as mp
import tempfile

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
from opendipaco.schedule import (
    CoordinatorServer,
    client_context,
    generate_selfsigned_cert,
    run_worker,
    server_context,
)

VOCAB, SEQ_LEN, GENS, BATCH = 128, 32, 6, 8
SECRET = "shared-cluster-secret"  # HMAC auth, on top of TLS


def build_config():
    bb = BackboneConfig(vocab_size=VOCAB, hidden_size=64, num_attention_heads=4,
                        intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=SEQ_LEN)


def diloco():
    return DiLoCoConfig(inner_steps=8, inner_lr=1e-3)


def topic_docs(num_topics=4, per=40, length=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = VOCAB // num_topics
    return [torch.randint(t * span, (t + 1) * span, (length,), generator=g)
            for t in range(num_topics) for _ in range(per)]


def build_corpus(config):
    docs = topic_docs()
    engine = DiPaCoEngine(config, diloco(), LocalBackend(config.build_topology()), seed=0)
    feat = ModelFeaturizer(engine.global_modules(), config)
    router = KMeansRouter(config.num_paths, seed=0).fit(feat([d[:32] for d in docs]))
    return ShardedCorpus.from_documents(docs, router, feat, config.num_paths, SEQ_LEN)


def coordinator_main(port_q, certfile, keyfile):
    torch.set_num_threads(1)
    config = build_config()
    engine = DiPaCoEngine(config, diloco(), LocalBackend(config.build_topology()),
                          seed=0, materialize="serial")
    server = CoordinatorServer(
        AsyncScheduler(engine, lease_timeout=30.0), build_corpus(config), batch_size=BATCH,
        host="127.0.0.1", port=0, auth_key=SECRET,
        tls=server_context(certfile, keyfile),  # <-- present the cert; encrypt the channel
    )
    server.start()
    port_q.put(server.port)
    print(f"[coordinator] serving TLS on port {server.port}", flush=True)
    completed = server.fit(num_generations=GENS, total_generations=GENS, log_every=0)
    server.shutdown()
    print("[coordinator] per-path updates (uneven = async):", completed, flush=True)
    print(f"[coordinator] accepted {server.metrics.accepted_updates} encrypted updates", flush=True)
    port_q.put("done")


def worker_main(rank, port, cafile):
    torch.set_num_threads(1)
    # Verify the coordinator's cert against it (as our CA) and check the hostname.
    ctx = client_context(cafile=cafile, check_hostname=True)
    print(f"[worker {rank}] connecting over TLS to 127.0.0.1:{port}", flush=True)
    run_worker(build_config(), diloco(), "127.0.0.1", port, seed=0, auth_key=SECRET,
               tls=ctx, tls_hostname="127.0.0.1")
    print(f"[worker {rank}] done", flush=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        certfile, keyfile = generate_selfsigned_cert(tmp)  # SAN: localhost, 127.0.0.1
        ctx = mp.get_context("spawn")
        port_q = ctx.Queue()
        coord = ctx.Process(target=coordinator_main, args=(port_q, certfile, keyfile))
        coord.start()
        port = port_q.get(timeout=60)

        workers = [ctx.Process(target=worker_main, args=(r, port, certfile)) for r in range(3)]
        for w in workers:
            w.start()
        port_q.get(timeout=300)  # wait for the coordinator to finish + report
        coord.join(timeout=30)
        for w in workers:
            w.join(timeout=30)


if __name__ == "__main__":
    main()
