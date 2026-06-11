"""Tests for the async fault-tolerant scheduler.

Note on determinism: the scheduler runs each path's inner loop in a worker
thread. PyTorch CPU execution is not bit-reproducible across threads/runs (parallel
reductions reorder), so we assert *numerical agreement* (``max|Δ|`` small), not
bit-exactness. The map and reduce steps are each bit-deterministic in isolation;
the only spread is benign floating-point noise (~1e-4 here), which is cleanly
separated from what a real fault-handling bug would cause -- a silently dropped
path removes a whole pseudo-gradient term (O(0.1+)). ``NOISE`` is the threshold.
"""

import torch

from opendipaco import (
    AsyncScheduler,
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
    load_checkpoint,
    save_checkpoint,
)
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import Preempt, TransientFault

NOISE = 1e-2  # >> observed float noise (~1e-4), << any real fault-handling bug


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _docs(seed=0, n_per=8):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(n_per)]


def _corpus(cfg):
    docs = _docs()
    assign = torch.tensor([i % cfg.num_paths for i in range(len(docs))])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _engine(cfg, seed=0):
    # Serial so per-path inner-optimizer state is checkpointable.
    return DiPaCoEngine(
        cfg, DiLoCoConfig(inner_steps=4, inner_lr=1e-3),
        LocalBackend(cfg.build_topology()), seed=seed, materialize="serial",
    )


def _snapshot(engine):
    return {k: {n: p.detach().clone() for n, p in m.named_parameters()}
            for k, m in engine.bank.items()}


def _max_diff(a, b):
    assert a.keys() == b.keys()
    return max(float((a[k][n] - b[k][n]).abs().max()) for k in a for n in a[k])


def _run(num_workers, *, fault_hook=None, gens=2, seed=0, **sched_kw):
    cfg = _cfg()
    eng = _engine(cfg, seed=seed)
    sched = AsyncScheduler(eng, num_workers=num_workers, lease_timeout=0.05, **sched_kw)
    sched.fit(_corpus(cfg), num_generations=gens, batch_size=8, total_generations=gens,
              log_every=0, fault_hook=fault_hook)
    return sched, _snapshot(eng)


def test_consistent_across_worker_counts():
    _, one = _run(1)
    _, four = _run(4)
    assert _max_diff(one, four) < NOISE


def test_runs_change_the_bank():
    cfg = _cfg()
    eng = _engine(cfg)
    before = _snapshot(eng)
    AsyncScheduler(eng, num_workers=2, lease_timeout=0.05).fit(
        _corpus(cfg), num_generations=2, batch_size=8, total_generations=2, log_every=0
    )
    assert _max_diff(before, _snapshot(eng)) > 1e-3  # training actually moved the weights


def test_transient_faults_recover():
    """First attempt of every path raises -> re-queued -> ~same final bank."""
    def hook(path, attempt):
        if attempt == 1:
            raise TransientFault()

    _, clean = _run(4)
    _, faulty = _run(4, fault_hook=hook)
    assert _max_diff(clean, faulty) < NOISE


def test_preemption_recovers():
    """First attempt 'dies' (no ack) -> lease expires -> reclaimed -> ~same bank."""
    def hook(path, attempt):
        if attempt == 1:
            raise Preempt()

    _, clean = _run(3)
    _, preempted = _run(3, fault_hook=hook)
    assert _max_diff(clean, preempted) < NOISE


def test_partial_tolerance_drops_permanently_failing_path():
    """A path that always fails is dropped; the generation still steps with the rest."""
    cfg = _cfg()
    eng = _engine(cfg)
    target = eng.topology.path_from_index(0)

    def hook(path, attempt):
        if path == target:
            raise TransientFault()  # never succeeds

    sched = AsyncScheduler(
        eng, num_workers=4, lease_timeout=0.05, max_attempts=2, min_fraction=0.5,
    )
    before = _snapshot(eng)
    sched.fit(_corpus(cfg), num_generations=1, batch_size=8, total_generations=1,
              log_every=0, fault_hook=hook)
    assert target in sched.dropped
    assert _max_diff(before, _snapshot(eng)) > 1e-3  # surviving paths still stepped


def test_scheduler_checkpoint_resume(tmp_path):
    cfg = _cfg()
    corpus = _corpus(cfg)

    # Reference: 4 generations straight.
    ref_eng = _engine(cfg)
    AsyncScheduler(ref_eng, num_workers=2, lease_timeout=0.05).fit(
        corpus, num_generations=4, batch_size=8, total_generations=4, log_every=0
    )
    ref = _snapshot(ref_eng)

    # Interrupted: 2 generations, checkpoint, fresh engine + scheduler resume 2 more.
    a_eng = _engine(cfg)
    AsyncScheduler(a_eng, num_workers=2, lease_timeout=0.05).fit(
        corpus, num_generations=2, batch_size=8, total_generations=4, log_every=0
    )
    save_checkpoint(a_eng, tmp_path / "mid")

    b_eng = _engine(cfg, seed=321)
    load_checkpoint(b_eng, tmp_path / "mid")
    AsyncScheduler(b_eng, num_workers=2, lease_timeout=0.05).fit(
        corpus, num_generations=2, batch_size=8, log_every=0
    )
    assert b_eng.total_rounds == 4
    assert b_eng._global_round == 4
    assert _max_diff(ref, _snapshot(b_eng)) < NOISE
