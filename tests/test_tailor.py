"""Tests for W6c: scheduler per-worker uplink tailoring (docs/w6-client-design.md D4b).

A worker advertises its bandwidth budget in `capabilities`; with
`tailor_bandwidth` on, the scheduler stamps a lighter (compress, density) on
that worker's task -- never lighter than the global, and byte-identical when off
or when no budget is advertised.
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.launch import LaunchConfig
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


def _serving(cfg, **kw):
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], DiLoCoConfig(inner_steps=2),
                      batch_size=BATCH, host="127.0.0.1", port=0, **kw)
    sched.start()
    sched._completed = {p: 0 for p in sched.paths}
    sched._serving = True
    sched._target = sched._T + 100 * len(sched.paths)
    return sched


def _task(sched, wid, caps):
    return sched._next_task({"worker_id": wid, "warm_paths": [], "cached_shards": [],
                             "capabilities": caps})


def test_tailoring_off_is_byte_identical():
    cfg = _cfg()
    sched = _serving(cfg)                                  # tailor_bandwidth defaults off
    try:
        t = _task(sched, "w", {"max_mbps": 1.0})           # tight budget, but tailoring off
        assert t["type"] == "task"
        assert t["compress"] == "none" and t["density"] == 1.0   # global, unchanged
    finally:
        sched.shutdown()


def test_tailors_a_budget_constrained_worker_when_on():
    cfg = _cfg()
    sched = _serving(cfg, tailor_bandwidth=True)
    try:
        tight = _task(sched, "tight", {"max_mbps": 2.0})
        assert tight["compress"] == "int4" and tight["density"] == 0.5
        moderate = _task(sched, "mid", {"max_mbps": 8.0})
        assert moderate["compress"] == "int8" and moderate["density"] == 1.0
        # An uncapped worker (no advertised budget) still gets the global encoding.
        ample = _task(sched, "ample", {})
        assert ample["compress"] == "none" and ample["density"] == 1.0
    finally:
        sched.shutdown()


def test_tailoring_never_undercuts_the_global_compress():
    cfg = _cfg()
    sched = _serving(cfg, compress="int4", tailor_bandwidth=True)
    try:
        # Ample budget would pick "none", but the run's base is int4 -> stays int4.
        t = _task(sched, "w", {"max_mbps": 100.0})
        assert t["compress"] == "int4"
    finally:
        sched.shutdown()


def test_config_rejects_tailor_bandwidth_outside_central_sharded():
    import pytest
    base_sharded = {"mode": "sharded",
                    "sharded": {"num_shards": 2,
                                "parameter_servers": [["127.0.0.1", 1], ["127.0.0.1", 2]]}}
    # coordinator mode: rejected.
    with pytest.raises(ValueError, match="tailor_bandwidth"):
        LaunchConfig.from_dict({"mode": "coordinator", "transport": {"tailor_bandwidth": True}})
    # decentralized: rejected (no central scheduler).
    with pytest.raises(ValueError, match="tailor_bandwidth"):
        LaunchConfig.from_dict({**base_sharded, "ownership": {"mode": "rendezvous"},
                                "schedule": {"mode": "decentralized"},
                                "transport": {"tailor_bandwidth": True}})
    # central sharded: accepted.
    ok = LaunchConfig.from_dict({**base_sharded, "transport": {"tailor_bandwidth": True}})
    assert ok.transport.tailor_bandwidth is True
