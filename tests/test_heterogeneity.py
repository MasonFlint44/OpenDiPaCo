"""Tests for the heterogeneity basics (internet-scale plan §0e).

bf16 autocast inner loop, worker capability advertising + per-worker batch caps,
and server-driven idle backoff. (The ``weights_only=True`` load hygiene from the
same phase is covered by the existing checkpoint/sharded/streaming round-trip
tests, which now run through the hardened loaders.)
"""

import math
import threading

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
from opendipaco.schedule import CoordinatorServer, Scheduler, run_worker

BATCH = 8


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _diloco(**kw):
    return DiLoCoConfig(inner_steps=4, inner_lr=1e-3, **kw)


def _docs():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(8)]


def _corpus(cfg):
    assign = torch.tensor([i % cfg.num_paths for i in range(32)])
    return ShardedCorpus.from_assignments(_docs(), assign, cfg.num_paths, cfg.sequence_length)


def _snap(bank):
    return {k: {n: p.detach().clone() for n, p in m.named_parameters()}
            for k, m in bank.items()}


def _maxdiff(a, b):
    return max(float((a[k][n] - b[k][n]).abs().max()) for k in a for n in a[k])


# -- bf16 autocast inner loop ------------------------------------------------------


def test_inner_autocast_trains_and_keeps_fp32_params():
    """With ``inner_autocast`` the engine trains (finite loss, weights move) while
    parameters and the bank stay fp32 (autocast only affects compute dtype)."""
    cfg = _cfg()
    eng = DiPaCoEngine(cfg, _diloco(inner_autocast=True),
                       LocalBackend(cfg.build_topology()), seed=0, materialize="serial")
    before = _snap(eng.bank)
    history = eng.fit(_corpus(cfg), num_rounds=2, batch_size=BATCH, log_every=0)
    assert all(math.isfinite(m.inner_loss) for m in history)
    assert _maxdiff(before, _snap(eng.bank)) > 1e-4
    assert all(p.dtype == torch.float32
               for m in eng.bank.values() for p in m.parameters())


def test_inner_autocast_off_is_bit_identical_to_before():
    """The default (off) path is unchanged: two engines, one explicitly off and
    one default, produce identical weights."""
    cfg = _cfg()
    a = DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                     seed=0, materialize="serial")
    b = DiPaCoEngine(cfg, _diloco(inner_autocast=False), LocalBackend(cfg.build_topology()),
                     seed=0, materialize="serial")
    a.fit(_corpus(cfg), num_rounds=1, batch_size=BATCH, log_every=0)
    b.fit(_corpus(cfg), num_rounds=1, batch_size=BATCH, log_every=0)
    assert _maxdiff(_snap(a.bank), _snap(b.bank)) == 0.0


# -- capability advertising + batch caps -------------------------------------------


def _serving_coordinator(cfg, **kw):
    eng = DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                       seed=0, materialize="serial")
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0, **kw)
    with server._lock:
        server._serving = True
        server._target = 10 ** 9
        server._completed = {p: 0 for p in eng.topology.paths()}
    return server


def _req(wid="w", caps=None):
    return {"worker_id": wid, "warm_paths": [], "cached_shards": [],
            "have_shared": {}, "capabilities": caps or {}}


def test_coordinator_clamps_batch_to_worker_cap():
    cfg = _cfg()
    server = _serving_coordinator(cfg)
    capped = server._next_task(_req("small", caps={"device": "cuda", "max_batch": 2}))
    assert capped["batch_size"] == 2
    full = server._next_task(_req("big"))
    assert full["batch_size"] == BATCH                  # uncapped workers unchanged
    assert server._worker_caps["small"]["max_batch"] == 2
    server.shutdown()


def test_sharded_scheduler_clamps_batch_to_worker_cap():
    cfg = _cfg()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], _diloco(),
                      batch_size=BATCH, host="127.0.0.1", port=0)
    with sched._lock:
        sched._serving = True
        sched._target = 10 ** 9
        sched._completed = {p: 0 for p in sched.paths}
    task = sched._next_task({"worker_id": "w", "warm_paths": [], "cached_shards": [],
                             "capabilities": {"max_batch": 3}})
    assert task["batch_size"] == 3
    sched.shutdown()


def test_capped_worker_trains_end_to_end():
    """A worker advertising a tiny batch cap still completes a real run."""
    cfg = _cfg()
    eng = DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                       seed=0, materialize="serial")
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=10.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0,
                               heartbeat_timeout=10.0)
    before = _snap(eng.bank)
    server.start()
    w = threading.Thread(target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
                         kwargs=dict(seed=0, reconnect=False, max_batch_size=2),
                         daemon=True)
    w.start()
    server.fit(num_generations=2, total_generations=2, log_every=0)
    server.shutdown()
    w.join(timeout=15)
    assert server._T >= server._target
    assert _maxdiff(before, _snap(eng.bank)) > 1e-4


# -- server-driven idle backoff ------------------------------------------------------


def test_idle_reply_carries_server_backoff():
    cfg = _cfg()
    server = _serving_coordinator(cfg, idle_backoff=0.25)
    with server._lock:                                  # nothing eligible -> idle
        server._serving = False
    msg = server._next_task(_req())
    assert msg["type"] == "idle" and msg["retry_in"] == 0.25
    server.shutdown()

    default = _serving_coordinator(cfg)
    with default._lock:
        default._serving = False
    assert "retry_in" not in default._next_task(_req())  # unset -> worker-paced
    default.shutdown()


def test_run_completes_with_idle_backoff_set():
    """More workers than paths, server-paced idling: the run still finishes."""
    cfg = _cfg()
    eng = DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                       seed=0, materialize="serial")
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=10.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0,
                               heartbeat_timeout=10.0, idle_backoff=0.05)
    server.start()
    ws = [threading.Thread(target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
                           kwargs=dict(seed=0, reconnect=False), daemon=True)
          for _ in range(6)]                            # 6 workers, 4 paths
    for w in ws:
        w.start()
    server.fit(num_generations=2, total_generations=2, log_every=0)
    server.shutdown()
    for w in ws:
        w.join(timeout=15)
    assert server._T >= server._target
