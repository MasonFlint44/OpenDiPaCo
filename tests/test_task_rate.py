"""Tests for W5a: scheduler-observed per-worker effective rate
(docs/w5-task-sizing-design.md D1). The scheduler times lease->commit and divides
by the task's work to get a tokens/lease-second EMA per worker. Measurement only
(dynamics-neutral); W5b sizes tasks from it. Rejected / reclaimed / nacked /
empty leases must NOT record a sample.
"""

import time

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import Scheduler

BATCH = 8


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _corpus(cfg):
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(0, 40, (48,), generator=g) for _ in range(16)]
    assign = torch.tensor([i % cfg.num_paths for i in range(16)])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _serving_scheduler(cfg, **kw):
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], DiLoCoConfig(inner_steps=2),
                      batch_size=BATCH, host="127.0.0.1", port=0, **kw)
    sched.start()
    sched._completed = {p: 0 for p in sched.paths}
    sched._serving = True
    sched._target = sched._T + 100 * len(sched.paths)
    return sched


def _lease(sched, wid="w1"):
    task = sched._next_task({"worker_id": wid, "warm_paths": [], "cached_shards": []})
    assert task["type"] == "task"
    return task


# -- the EMA helper (deterministic) --------------------------------------------


def test_record_rate_ema_and_guards():
    cfg = _cfg()
    sched = _serving_scheduler(cfg)
    try:
        # First sample: rate = work / elapsed.
        sched._record_rate_locked("w", work=100, elapsed=2.0)
        assert sched.worker_rate("w") == 50.0
        # Second sample blends with alpha (0.3): 0.7*50 + 0.3*(200/2) = 35 + 30 = 65.
        sched._record_rate_locked("w", work=200, elapsed=2.0)
        assert abs(sched.worker_rate("w") - 65.0) < 1e-9
        # Guards: a zero/negative duration or missing work/worker is ignored.
        sched._record_rate_locked("w", work=100, elapsed=0.0)
        sched._record_rate_locked("w", work=0, elapsed=1.0)
        sched._record_rate_locked(None, work=100, elapsed=1.0)
        assert abs(sched.worker_rate("w") - 65.0) < 1e-9   # unchanged
        assert sched.worker_rate("never-seen") is None
    finally:
        sched.shutdown()


# -- end-to-end through lease -> commit ----------------------------------------


def test_worker_rate_recorded_on_accepted_commit():
    cfg = _cfg()
    sched = _serving_scheduler(cfg)
    try:
        task = _lease(sched, "w1")
        assert sched.worker_rate("w1") is None             # nothing committed yet
        time.sleep(0.05)
        ack = sched._commit({"type": "commit", "path": task["path"], "worker_id": "w1",
                             "lease": task["lease"], "loss": 1.0})
        assert ack["accepted"]
        r = sched.worker_rate("w1")
        work = task["batch_size"] * 2 * cfg.sequence_length   # batch * inner_steps * seq
        assert r is not None and r > 0
        # rate = work / elapsed; elapsed is in (0.01s, 2s) even under CI load.
        assert work / 2.0 < r < work / 0.01
    finally:
        sched.shutdown()


def test_empty_commit_records_no_rate():
    """An empty-shard no-op commit (no real training) must not pollute the rate."""
    cfg = _cfg()
    sched = _serving_scheduler(cfg)
    try:
        task = _lease(sched, "w1")
        ack = sched._commit({"type": "commit", "path": task["path"], "worker_id": "w1",
                             "lease": task["lease"], "loss": float("nan"), "empty": True})
        assert ack["accepted"]                              # empty no-op is accepted
        assert sched.worker_rate("w1") is None              # ...but yields no rate sample
    finally:
        sched.shutdown()


def test_reclaimed_lease_records_no_rate():
    """A lease that times out (worker presumed dead) is reclaimed, not committed,
    so it leaves no rate sample and no stale timing entry."""
    cfg = _cfg()
    sched = _serving_scheduler(cfg, heartbeat_timeout=0.05)
    try:
        task = _lease(sched, "slow")
        path = task["path"]
        assert path in sched._lease_at
        time.sleep(0.1)
        with sched._lock:
            sched._reclaim_inflight_locked()
        assert path not in sched._lease_at and path not in sched._lease_work
        assert sched.worker_rate("slow") is None
    finally:
        sched.shutdown()
