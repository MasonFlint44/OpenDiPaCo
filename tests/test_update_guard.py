"""Tests for server-side update validation (guard.py; internet-scale plan §0b).

Non-finite contributions (pseudo-gradient, private weights, reported loss) must be
rejected before they touch the bank -- one applied NaN poisons it permanently --
and the optional ``max_update_norm`` cap must clip oversized pseudo-gradients.
Covers all three enforcement points: the single-node ``CoordinatorServer``, the
sharded ``Scheduler`` (loss gate at commit), and the ``ParameterServer`` (tensor
checks at push). Servers are driven directly, like the staleness-bound tests.
"""

import math

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
from opendipaco.optim.diloco import make_outer_optimizer
from opendipaco.schedule import CoordinatorServer, ParameterServer, Scheduler, make_grant
from opendipaco.schedule.guard import all_finite, clip_norm_, loss_ok
from opendipaco.topology import is_private_key

BATCH = 8


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _diloco():
    return DiLoCoConfig(inner_steps=4, inner_lr=1e-3)


def _docs():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(8)]


def _corpus(cfg):
    assign = torch.tensor([i % cfg.num_paths for i in range(32)])
    return ShardedCorpus.from_assignments(_docs(), assign, cfg.num_paths, cfg.sequence_length)


def _engine(cfg, seed=0):
    return DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                        seed=seed, materialize="serial")


def _snap(bank):
    return {k: {n: p.detach().clone() for n, p in m.named_parameters()}
            for k, m in bank.items()}


def _maxdiff(a, b):
    return max(float((a[k][n] - b[k][n]).abs().max()) for k in a for n in a[k])


# -- unit: the guard helpers ---------------------------------------------------


def test_all_finite_walks_nested_structures():
    ok = {"a": [torch.ones(2)], "b": {"w": torch.zeros(3)}}
    assert all_finite(ok)
    assert all_finite({})                      # nothing to check -> fine
    bad_nan = {"a": [torch.tensor([1.0, float("nan")])]}
    bad_inf = {"a": {"w": torch.tensor(float("inf"))}}
    assert not all_finite(bad_nan)
    assert not all_finite(bad_inf)


def test_loss_ok_gates_nonfinite_except_empty():
    assert loss_ok(1.5) and loss_ok(None)
    assert not loss_ok(float("nan")) and not loss_ok(float("inf"))
    assert loss_ok(float("nan"), empty=True)   # the empty-shard no-op convention


def test_clip_norm_scales_in_place_and_reports():
    ts = [torch.full((4,), 3.0), torch.full((4,), 4.0)]   # joint norm 10
    norm = clip_norm_(ts, 1.0)
    assert math.isclose(norm, 10.0, rel_tol=1e-6)
    joint = math.sqrt(sum(float(t.pow(2).sum()) for t in ts))
    assert math.isclose(joint, 1.0, rel_tol=1e-5)         # scaled to the cap
    small = [torch.full((4,), 0.01)]
    keep = small[0].clone()
    clip_norm_(small, 1.0)
    assert torch.equal(small[0], keep)                    # under the cap -> untouched


# -- coordinator ----------------------------------------------------------------


def _serving_coordinator(cfg, eng, **kw):
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0, **kw)
    server._outer_opts = {k: make_outer_optimizer({k: eng.bank[k]}, eng.diloco)
                          for k in server._versions}
    with server._lock:
        server._serving = True
        server._target = 10 ** 9
        server._completed = {p: 0 for p in eng.topology.paths()}
    return server


def _lease(server):
    task = server._next_task(
        {"worker_id": "w", "warm_paths": [], "cached_shards": [], "have_shared": {}})
    return task["path"], task["lease"]


def _grad_for(eng, path, fill=1.0):
    keys = [k for k in eng.topology.path_module_keys(path) if not is_private_key(k)]
    return {k: [torch.full_like(p, fill) for p in eng.bank[k].parameters()] for k in keys}


def test_coordinator_rejects_nonfinite_grad():
    """A NaN pseudo-gradient is dropped before touching the bank; the path is
    re-eligible and a clean resubmission is accepted."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = _serving_coordinator(cfg, eng)
    path, lease = _lease(server)
    grad = _grad_for(eng, path)
    next(iter(grad.values()))[0][0] = float("nan")

    before = _snap(eng.bank)
    server._receive({"path": path, "lease": lease, "shared_grad": grad,
                     "private_weights": {}, "loss": 0.1})
    assert server.metrics.invalid_rejected == 1
    assert server.metrics.accepted_updates == 0
    assert _maxdiff(before, _snap(eng.bank)) == 0.0   # bank untouched

    # The lease was freed: the path can be re-leased and a finite update lands.
    path2, lease2 = _lease(server)
    assert path2 == path
    server._receive({"path": path, "lease": lease2, "shared_grad": _grad_for(eng, path),
                     "private_weights": {}, "loss": 0.1})
    assert server.metrics.accepted_updates == 1
    assert _maxdiff(before, _snap(eng.bank)) > 0.0
    server.shutdown()


def test_coordinator_rejects_nonfinite_private_and_loss():
    """Inf in shipped private weights, or a NaN loss on a non-empty contribution,
    rejects the whole contribution (private modules are loaded verbatim -- a NaN
    embedding would poison the bank with no aggregation to soften it)."""
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    # Private embedding -> the bank has per-path private modules to attack.
    cfg = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16,
                       embedding="private")
    eng = DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                       seed=0, materialize="serial")
    server = _serving_coordinator(cfg, eng)
    private_key = next(k for k in eng.bank if is_private_key(k))
    bad_private = {private_key: {
        n: torch.full_like(v, float("inf")) for n, v in eng.bank[private_key].state_dict().items()
    }}
    before = _snap(eng.bank)

    path, lease = _lease(server)
    server._receive({"path": path, "lease": lease, "shared_grad": {},
                     "private_weights": bad_private, "loss": 0.1})
    assert server.metrics.invalid_rejected == 1
    assert _maxdiff(before, _snap(eng.bank)) == 0.0

    path, lease = _lease(server)
    server._receive({"path": path, "lease": lease, "shared_grad": _grad_for(eng, path),
                     "private_weights": {}, "loss": float("nan")})
    assert server.metrics.invalid_rejected == 2
    assert server.metrics.accepted_updates == 0
    assert _maxdiff(before, _snap(eng.bank)) == 0.0
    server.shutdown()


def test_coordinator_norm_cap_clips_oversized_updates():
    """With ``max_update_norm`` set, a huge-but-finite pseudo-gradient is scaled to
    the cap (accepted, counted, and its effect on the bank stays tiny)."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = _serving_coordinator(cfg, eng, max_update_norm=1e-6)
    path, lease = _lease(server)
    before = _snap(eng.bank)
    server._receive({"path": path, "lease": lease, "shared_grad": _grad_for(eng, path, 100.0),
                     "private_weights": {}, "loss": 0.1})
    assert server.metrics.accepted_updates == 1
    assert server.metrics.norm_clipped >= 1
    moved = _maxdiff(before, _snap(eng.bank))
    assert 0.0 < moved < 1e-4     # uncapped, a fill-100 delta would move weights ~O(100)
    server.shutdown()


def test_coordinator_ignores_unknown_and_mistyped_keys():
    """An unknown shared key must not crash the server, and a 'private' payload
    must not be able to overwrite a shared module."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = _serving_coordinator(cfg, eng)
    shared_key = next(k for k in eng.bank if not is_private_key(k))
    smuggle = {shared_key: {  # a shared module smuggled through the private channel
        n: torch.zeros_like(v) for n, v in eng.bank[shared_key].state_dict().items()
    }}
    before = _snap(eng.bank)
    path, lease = _lease(server)
    server._receive({"path": path, "lease": lease,
                     "shared_grad": {"no-such-module": [torch.ones(3)]},
                     "private_weights": smuggle, "loss": 0.1})
    # Bogus keys are filtered (no crash); the smuggled shared module is untouched.
    assert _maxdiff(before, _snap(eng.bank)) == 0.0
    server.shutdown()


# -- sharded scheduler (loss gate at commit) -------------------------------------


def test_sharded_commit_rejects_nonfinite_loss():
    cfg = _cfg()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], _diloco(),
                      batch_size=BATCH, host="127.0.0.1", port=0)
    with sched._lock:
        sched._serving = True
        sched._target = 10 ** 9
        sched._completed = {p: 0 for p in sched.paths}
    req = {"worker_id": "w", "warm_paths": [], "cached_shards": []}

    task = sched._next_task(req)
    bad = sched._commit({"path": task["path"], "worker_id": "w", "lease": task["lease"],
                         "loss": float("nan")})
    assert bad["accepted"] is False              # diverged training gets no push grant
    assert sched.metrics.invalid_rejected == 1

    task = sched._next_task(req)                 # path re-eligible after the reject
    empty = sched._commit({"path": task["path"], "worker_id": "w", "lease": task["lease"],
                           "loss": float("nan"), "empty": True})
    assert empty["accepted"] is True             # empty-shard no-op stays accepted

    task = sched._next_task(req)
    good = sched._commit({"path": task["path"], "worker_id": "w", "lease": task["lease"],
                          "loss": 2.5})
    assert good["accepted"] is True
    sched.shutdown()


# -- parameter server -------------------------------------------------------------


def _ps_and_key(grant_key=None, **kw):
    cfg = _cfg()
    keys = sorted(cfg.build_topology().module_keys())
    ps = ParameterServer(cfg, keys, _diloco(), host="127.0.0.1", port=0,
                         grant_key=grant_key, **kw)
    shared = sorted(k for k in keys if not is_private_key(k))
    path = cfg.build_topology().path_from_index(0)
    return ps, keys, shared[0], path


def test_ps_push_rejects_nonfinite():
    ps, keys, k0, path = _ps_and_key()
    try:
        nan_grad = {"grad": [torch.full_like(p, float("nan"))
                             for p in ps.bank[k0].parameters()]}
        before = _snap(ps.bank)
        r = ps._push({"grant": make_grant(path, keys, 1.0, "t1"), "updates": {k0: nan_grad}})
        assert r["applied"] is False
        assert ps.metrics.invalid_rejected == 1
        assert _maxdiff(before, _snap(ps.bank)) == 0.0

        ok_grad = {"grad": [torch.ones_like(p) for p in ps.bank[k0].parameters()]}
        r = ps._push({"grant": make_grant(path, keys, 1.0, "t2"), "updates": {k0: ok_grad}})
        assert r["applied"] is True
        assert _maxdiff(before, _snap(ps.bank)) > 0.0
    finally:
        ps.shutdown()


def test_ps_push_norm_cap_clips():
    ps, keys, k0, path = _ps_and_key(max_update_norm=1e-6)
    try:
        huge = {"grad": [torch.full_like(p, 100.0) for p in ps.bank[k0].parameters()]}
        before = _snap(ps.bank)
        r = ps._push({"grant": make_grant(path, keys, 1.0, "t1"), "updates": {k0: huge}})
        assert r["applied"] is True
        assert ps.metrics.norm_clipped == 1
        moved = _maxdiff(before, _snap(ps.bank))
        assert 0.0 < moved < 1e-4
    finally:
        ps.shutdown()
