"""Tests for W5b: batch-first task sizing to a target wall-time + audit
size-pinning (docs/w5-task-sizing-design.md D2/D3/D6/D8). Sizing is off by default
(byte-identical anchor); when ``task_seconds`` is set, a slow worker (low measured
rate) gets a smaller batch, then fewer inner_steps once batch floors at 1, while a
fast worker gets the full configured task. Audited tasks pin their size so a
checker reproduces the primary's exact computation.
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import Scheduler

BATCH, INNER, SEQ = 8, 2, 16   # default_work = 8*2*16 = 256 tokens


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=SEQ)


def _corpus(cfg):
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(0, 40, (48,), generator=g) for _ in range(16)]
    assign = torch.tensor([i % cfg.num_paths for i in range(16)])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _serving(cfg, **kw):
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], DiLoCoConfig(inner_steps=INNER),
                      batch_size=BATCH, host="127.0.0.1", port=0, **kw)
    sched.start()
    sched._completed = {p: 0 for p in sched.paths}
    sched._serving = True
    sched._target = sched._T + 1000 * len(sched.paths)
    return sched


def _set_rate(sched, wid, tokens_per_sec):
    with sched._lock:
        sched._record_rate_locked(wid, work=tokens_per_sec, elapsed=1.0)  # rate = work/1s


def _lease(sched, wid):
    t = sched._next_task({"worker_id": wid, "warm_paths": [], "cached_shards": []})
    assert t["type"] == "task"
    return t


# -- the sizing function (deterministic) ---------------------------------------


def test_size_task_math():
    cfg = _cfg()
    sched = _serving(cfg, task_seconds=1.0)   # target tokens == rate
    try:
        with sched._lock:
            sched._record_rate_locked("slow", 100, 1.0)     # target 100
            sched._record_rate_locked("vslow", 20, 1.0)     # target 20
            sched._record_rate_locked("fast", 5000, 1.0)    # target 5000
            # slow: round(100/(2*16)) = 3 -> batch 3, inner 2
            assert sched._size_task_locked("slow", BATCH) == (3, 2)
            # very slow: batch floors at 1, then inner = round(20/16) = 1
            assert sched._size_task_locked("vslow", BATCH) == (1, 1)
            # fast: would be huge -> clamped to the configured ceiling
            assert sched._size_task_locked("fast", BATCH) == (8, 2)
            # no estimate yet -> full configured task (bootstrap)
            assert sched._size_task_locked("unknown", BATCH) == (8, 2)
    finally:
        sched.shutdown()


def test_sizing_off_returns_full_task():
    cfg = _cfg()
    sched = _serving(cfg)   # task_seconds None -> off
    try:
        with sched._lock:
            sched._record_rate_locked("slow", 20, 1.0)
            assert sched._size_task_locked("slow", BATCH) == (8, 2)  # off -> full ceiling
    finally:
        sched.shutdown()


# -- end-to-end through _next_task ---------------------------------------------


def test_sizing_off_is_byte_identical_task():
    """Off (the default): the task carries the configured batch and NO inner_steps
    field, so the worker uses its own default -> identical to the pre-W5 task."""
    cfg = _cfg()
    sched = _serving(cfg)
    try:
        t = _lease(sched, "w")
        assert t["batch_size"] == BATCH and "inner_steps" not in t
    finally:
        sched.shutdown()


def test_slow_worker_gets_smaller_batch_full_inner():
    cfg = _cfg()
    sched = _serving(cfg, task_seconds=1.0)
    try:
        _set_rate(sched, "slow", 100)            # target 100 -> batch 3
        t = _lease(sched, "slow")
        assert t["batch_size"] == 3 and t["inner_steps"] == 2
    finally:
        sched.shutdown()


def test_very_slow_worker_floors_batch_then_shrinks_inner():
    cfg = _cfg()
    sched = _serving(cfg, task_seconds=1.0)
    try:
        _set_rate(sched, "vslow", 20)            # batch floors at 1, inner -> 1
        t = _lease(sched, "vslow")
        assert t["batch_size"] == 1 and t["inner_steps"] == 1
    finally:
        sched.shutdown()


def test_fast_worker_gets_full_configured_task():
    cfg = _cfg()
    sched = _serving(cfg, task_seconds=1.0)
    try:
        _set_rate(sched, "fast", 100000)
        t = _lease(sched, "fast")
        assert t["batch_size"] == BATCH and t["inner_steps"] == INNER
    finally:
        sched.shutdown()


# -- audit size-pinning (D8) ---------------------------------------------------


def test_audit_record_pins_sized_batch_and_inner():
    """A sampled (audited) primary task records its *sized* batch/inner so a
    checker can reproduce it. Without this a heterogeneous checker would re-run a
    different batch and falsely flag the audit as divergent."""
    cfg = _cfg()
    sched = _serving(cfg, task_seconds=1.0, redundancy=2, redundancy_rate=1.0)
    try:
        _set_rate(sched, "slow", 100)            # -> batch 3, inner 2
        t = _lease(sched, "slow")
        key = (t["path"], t["gen_id"])
        a = sched._audits[key]
        assert a["batch"] == 3 and a["inner"] == 2
        # A reserved check returns the pinned size, regardless of the checker.
        with sched._lock:
            *_, pin_batch, pin_inner = sched._reserve_check_locked(key, "checker")
        assert pin_batch == 3 and pin_inner == 2
    finally:
        sched.shutdown()


def test_train_path_honors_per_task_inner_steps():
    """The worker half of D6: _train_path's inner_steps override actually changes
    the local-step count (it isn't ignored) and trains finitely under the cosine
    LR schedule with the reduced count -- the path the scheduler-side sizing tests
    don't exercise."""
    import math

    from opendipaco.backend import LocalBackend
    from opendipaco.schedule import AsyncScheduler
    from opendipaco.train.loop import DiPaCoEngine

    cfg = _cfg()
    eng = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=4, inner_lr=1e-2),
                       LocalBackend(cfg.build_topology()), device="cpu", seed=0,
                       materialize="serial")
    eng.total_rounds = 1
    worker = AsyncScheduler(eng, num_workers=1)
    path = cfg.build_topology().path_from_index(0)
    shard = torch.randint(0, 48, (32, cfg.sequence_length))

    sized = worker._train_path(path, shard, 4, 0, inner_steps=1)   # sized down
    eng._opt_state.pop(path, None)                                 # clean base for the default run
    default = worker._train_path(path, shard, 4, 0)                # configured 4
    assert not sized.empty and not default.empty
    assert math.isfinite(sized.loss) and math.isfinite(default.loss)
    # The override is honored: 1 step vs 4 steps from the same base -> different result.
    assert sized.loss != default.loss


def test_check_task_uses_pinned_size_not_checker_ceiling():
    """End to end: with all paths leased so only checks remain, a fast checker
    with a big cap still gets the audited primary's small batch/inner."""
    cfg = _cfg()
    sched = _serving(cfg, task_seconds=1.0, redundancy=2, redundancy_rate=1.0)
    try:
        # A slow primary leases + creates an audit, then commits so the audit
        # opens for checkers; lease the remaining paths so the checker finds no
        # primary work and is offered the check.
        _set_rate(sched, "slow", 100)            # primary sized to batch 3, inner 2
        prim = _lease(sched, "slow")
        key = (prim["path"], prim["gen_id"])
        ack = sched._commit({"type": "commit", "path": prim["path"], "worker_id": "slow",
                             "lease": prim["lease"], "loss": 1.0,
                             "base": {"k": [0]}, "digest": "D"})
        assert ack["accepted"] and sched._audits[key]["base"] is not None
        # Occupy *every* path (incl. the committed one, now eligible again) so the
        # checker finds no primary work and is offered the open (P,0) check.
        for i, _p in enumerate(sched.paths):
            _lease(sched, f"filler{i}")
        _set_rate(sched, "fast", 100000)         # the checker is fast + full cap
        chk = sched._next_task({"worker_id": "fast", "warm_paths": [], "cached_shards": [],
                                "capabilities": {"max_batch": 64}})
        assert chk.get("check_only") and tuple(chk["path"]) == prim["path"]
        assert chk["batch_size"] == 3 and chk["inner_steps"] == 2   # pinned, not the cap
    finally:
        sched.shutdown()
