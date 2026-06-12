"""Tests for the sharded coordinator (Scheduler + K ParameterServers).

The in-process tests run all three server roles + workers as threads over real
localhost TCP. A multi-process test runs them as separate `spawn` processes so the
shards genuinely live in different process memories (the #11 point). The async
caveat is inherited: behavior is asserted, not a deterministic reference.
"""

import multiprocessing as mp
import threading

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import (
    ParameterServer,
    Scheduler,
    assign_shards,
    make_grant,
    run_sharded_worker,
)
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


def _key_shards(cfg, num_shards):
    ks = assign_shards(cfg.build_topology().module_keys(), num_shards)
    return [[k for k, s in ks.items() if s == i] for i in range(num_shards)]


def _ps_snap(ps):
    return {k: {n: p.detach().clone() for n, p in m.named_parameters()}
            for k, m in ps.bank.items()}


def _maxdiff(a, b):
    return max(float((a[k][n] - b[k][n]).abs().max()) for k in a for n in a[k])


def test_sharded_no_node_holds_full_bank():
    """Two parameter servers own disjoint key shards whose union is the full bank;
    the scheduler holds no model weights at all."""
    cfg = _cfg()
    all_keys = set(cfg.build_topology().module_keys())
    s0, s1 = _key_shards(cfg, 2)
    ps0 = ParameterServer(cfg, s0, _diloco(), host="127.0.0.1", port=0)
    ps1 = ParameterServer(cfg, s1, _diloco(), host="127.0.0.1", port=0)
    try:
        assert set(s0).isdisjoint(s1)
        assert set(s0) | set(s1) == all_keys
        assert set(ps0.bank) == set(s0) and set(ps1.bank) == set(s1)  # each holds only its shard
        sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", ps0.port), ("127.0.0.1", ps1.port)],
                          _diloco(), batch_size=BATCH, host="127.0.0.1", port=0)
        assert not hasattr(sched, "bank")  # the scheduler holds no weights
        sched.shutdown()
    finally:
        ps0.shutdown()
        ps1.shutdown()


def _run_sharded_inproc(num_workers, *, gens=3, num_shards=2, hook=None):
    cfg = _cfg()
    dl = _diloco()
    shards = _key_shards(cfg, num_shards)
    pss = [ParameterServer(cfg, sk, dl, host="127.0.0.1", port=0) for sk in shards]
    for ps in pss:
        ps.start()
    befores = [_ps_snap(ps) for ps in pss]
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", ps.port) for ps in pss], dl,
                      batch_size=BATCH, host="127.0.0.1", port=0)
    sched.start()
    ws = [threading.Thread(
        target=run_sharded_worker,
        args=(cfg, dl, ("127.0.0.1", sched.port)),
        kwargs=dict(seed=0, heartbeat_interval=1.0, fault_hook=hook), daemon=True)
        for _ in range(num_workers)]
    for w in ws:
        w.start()
    completed = sched.fit(num_generations=gens, total_generations=gens)
    sched.shutdown()
    for ps in pss:
        ps.shutdown()
    for w in ws:
        w.join(timeout=10)
    return sched, pss, befores, completed


def test_sharded_training_completes_and_moves_bank():
    """Scheduler + 2 PS + 2 workers run to the update target; every shard's weights
    move and per-path progress is uneven (async)."""
    sched, pss, befores, completed = _run_sharded_inproc(2, gens=3)
    assert sched._T >= sched._target
    # The fit-returned snapshot did at least the target's worth of updates (a few
    # late in-flight commits may land after the snapshot, so compare with >=).
    assert sum(completed.values()) >= sched._target
    for ps, before in zip(pss, befores):
        assert _maxdiff(before, _ps_snap(ps)) > 1e-4  # this shard was trained
    assert sched.metrics.accepted_updates >= sched._target


def test_sharded_staleness_bound():
    """The scheduler rejects a commit stale by more than the bound; accepts one
    within it (returning a damped push weight). Driven directly for determinism."""
    cfg = _cfg()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1), ("127.0.0.1", 2)], _diloco(),
                      batch_size=BATCH, host="127.0.0.1", port=0, staleness_bound=1)
    with sched._lock:
        sched._serving = True
        sched._target = 10 ** 9
        sched._completed = {p: 0 for p in sched.paths}
    req = {"worker_id": "w", "warm_paths": [], "cached_shards": []}

    task = sched._next_task(req)                # issued at _T = 0
    path = task["path"]
    with sched._lock:
        sched._T = 5                            # 5 commits since -> staleness 5
    rej = sched._commit({"path": path, "worker_id": "w", "lease": task["lease"]})
    assert rej["accepted"] is False
    assert sched.metrics.stale_rejected == 1

    task2 = sched._next_task(req)               # re-lease at _T = 5 -> staleness 0
    acc = sched._commit({"path": path, "worker_id": "w", "lease": task2["lease"]})
    assert acc["accepted"] is True and acc["push_weight"] > 0
    # The grant carries the verdict to the PSs: same weight, single-use lease token.
    assert acc["grant"]["weight"] == acc["push_weight"]
    assert acc["grant"]["token"] == task2["lease"]
    assert sched.metrics.accepted_updates == 1
    sched.shutdown()


def test_sharded_zombie_commit_fenced():
    """A commit from an expired (reclaimed + re-leased) lease is rejected; the
    current lease holder's commit is the one accepted."""
    cfg = _cfg()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], _diloco(),
                      batch_size=BATCH, host="127.0.0.1", port=0)
    with sched._lock:
        sched._serving = True
        sched._target = 10 ** 9
        sched._completed = {p: 0 for p in sched.paths}

    task_a = sched._next_task({"worker_id": "A", "warm_paths": [], "cached_shards": []})
    path = task_a["path"]
    with sched._lock:
        sched._inflight[path] = 0.0             # worker A goes silent; lease expires
        sched._reclaim_inflight_locked()
    task_b = sched._next_task({"worker_id": "B", "warm_paths": [], "cached_shards": []})
    assert task_b["path"] == path and task_b["lease"] != task_a["lease"]

    zombie = sched._commit({"path": path, "worker_id": "A", "lease": task_a["lease"]})
    assert zombie["accepted"] is False           # the dead lease can't commit
    live = sched._commit({"path": path, "worker_id": "B", "lease": task_b["lease"]})
    assert live["accepted"] is True
    assert sched.metrics.accepted_updates == 1
    sched.shutdown()


def test_ps_push_requires_valid_grant():
    """Pushes are gated by the scheduler's commit grant: refused without one, with a
    forged/unsigned grant when a signature is required, or on replay; the applied
    weight and allowed keys come from the grant, not the worker."""
    cfg = _cfg()
    keys = _key_shards(cfg, 1)[0]
    ps = ParameterServer(cfg, keys, _diloco(), host="127.0.0.1", port=0,
                         grant_key="ps-secret")
    try:
        shared = sorted(k for k in keys if not is_private_key(k))
        k0 = shared[0]
        grad = {"grad": [torch.ones_like(p) for p in ps.bank[k0].parameters()]}
        path = cfg.build_topology().path_from_index(0)
        before = _ps_snap(ps)

        # No grant -> refused, weights untouched.
        r = ps._push({"updates": {k0: grad}})
        assert r["applied"] is False and _maxdiff(before, _ps_snap(ps)) == 0.0

        # Grant signed with the wrong key -> refused.
        forged = make_grant(path, keys, 100.0, "tok-forged", grant_key="wrong-key")
        r = ps._push({"grant": forged, "updates": {k0: grad}})
        assert r["applied"] is False and _maxdiff(before, _ps_snap(ps)) == 0.0

        # Unsigned grant while the PS requires a signature -> refused.
        unsigned = make_grant(path, keys, 1.0, "tok-unsigned")
        r = ps._push({"grant": unsigned, "updates": {k0: grad}})
        assert r["applied"] is False and _maxdiff(before, _ps_snap(ps)) == 0.0

        # Properly signed grant -> applied.
        good = make_grant(path, keys, 1.0, "tok-good", grant_key="ps-secret")
        r = ps._push({"grant": good, "updates": {k0: grad}})
        assert r["applied"] is True and _maxdiff(before, _ps_snap(ps)) > 0.0

        # Replay of the same grant -> refused, weights unchanged.
        after = _ps_snap(ps)
        r = ps._push({"grant": good, "updates": {k0: grad}})
        assert r["applied"] is False and _maxdiff(after, _ps_snap(ps)) == 0.0

        # A key outside the grant's allow-list is skipped even on a valid grant.
        k1 = shared[1]
        grad1 = {"grad": [torch.ones_like(p) for p in ps.bank[k1].parameters()]}
        only_k0 = make_grant(path, [k0], 1.0, "tok-narrow", grant_key="ps-secret")
        snap = _ps_snap(ps)
        r = ps._push({"grant": only_k0, "updates": {k1: grad1}})
        assert r["applied"] is True                   # the push is processed...
        assert _maxdiff(snap, _ps_snap(ps)) == 0.0    # ...but k1 wasn't touched
    finally:
        ps.shutdown()


def test_ps_checkpoint_restores_outer_momentum(tmp_path):
    """``save_shard``/``load_shard`` round-trip the per-module outer Nesterov
    momentum, not just weights and versions."""
    cfg = _cfg()
    keys = _key_shards(cfg, 1)[0]
    dl = _diloco()
    ps = ParameterServer(cfg, keys, dl, host="127.0.0.1", port=0)
    shared = [k for k in keys if not is_private_key(k)]
    path = cfg.build_topology().path_from_index(0)
    updates = {k: {"grad": [torch.ones_like(p) for p in ps.bank[k].parameters()]}
               for k in shared}
    ps._push({"grant": make_grant(path, keys, 1.0, "tok"), "updates": updates})
    ckpt = str(tmp_path / "ck")
    ps.save_shard(ckpt)

    ps2 = ParameterServer(cfg, keys, dl, host="127.0.0.1", port=0, resume_dir=ckpt)
    try:
        for k in shared:
            sa = ps._outer_opts[k].state_dict()["state"]
            sb = ps2._outer_opts[k].state_dict()["state"]
            assert sa and set(sa) == set(sb)   # momentum exists and was restored
            for i in sa:
                assert torch.equal(sa[i]["momentum_buffer"], sb[i]["momentum_buffer"])
    finally:
        ps.shutdown()
        ps2.shutdown()


def _combined_snap(pss):
    out = {}
    for ps in pss:
        for k, m in ps.bank.items():
            out[k] = {n: p.detach().clone() for n, p in m.named_parameters()}
    return out


def _run_ckpt(ckpt_dir, *, gens=3, resume=False, num_shards=2):
    cfg = _cfg()
    dl = _diloco()
    shards = _key_shards(cfg, num_shards)
    pss = [ParameterServer(cfg, sk, dl, host="127.0.0.1", port=0,
                           resume_dir=ckpt_dir if resume else None) for sk in shards]
    for ps in pss:
        ps.start()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", ps.port) for ps in pss], dl,
                      batch_size=BATCH, host="127.0.0.1", port=0)
    sched.start()
    ws = [threading.Thread(target=run_sharded_worker,
                           args=(cfg, dl, ("127.0.0.1", sched.port)),
                           kwargs=dict(seed=0, heartbeat_interval=1.0), daemon=True)
          for _ in range(2)]
    for w in ws:
        w.start()
    sched.fit(num_generations=gens, total_generations=gens,
              checkpoint_dir=ckpt_dir, checkpoint_every=2, resume=resume)
    sched.shutdown()
    for ps in pss:
        ps.shutdown()
    for w in ws:
        w.join(timeout=10)
    return _combined_snap(pss)


def test_sharded_checkpoint_resume(tmp_path):
    """A cluster checkpoint (scheduler clock + each PS's shard) lets a fresh cluster
    restore the *exact* checkpointed weights and clock rather than starting over.

    We assert the durable, deterministic property -- resume reloads the checkpoint
    bit-for-bit -- not a drift-from-init comparison: DiLoCo's outer (Nesterov) steps
    don't move weights monotonically away from init, so "more training => farther"
    is a fragile premise that async reordering makes flaky."""
    cfg = _cfg()
    shards = _key_shards(cfg, 2)
    init_pss = [ParameterServer(cfg, sk, _diloco(), host="127.0.0.1", port=0) for sk in shards]
    init = _combined_snap(init_pss)
    for ps in init_pss:
        ps.shutdown()

    ckpt = str(tmp_path / "ck")
    trained = _run_ckpt(ckpt, gens=3)  # trains + checkpoints (shards + scheduler clock)
    assert _maxdiff(init, trained) > 1e-4  # training actually moved the bank

    # A fresh cluster pointed at the checkpoint loads *trained* weights (not init) and
    # does so deterministically: two independent resumes are bit-for-bit identical.
    # (We compare resumes to each other, not to ``trained`` -- under async a few worker
    # pushes can land *after* the final checkpoint, so the on-disk checkpoint and the
    # final in-memory bank legitimately differ; the durable guarantee is that the file
    # restores exactly and reproducibly.)
    def _resume_snap():
        pss = [ParameterServer(cfg, sk, _diloco(), host="127.0.0.1", port=0, resume_dir=ckpt)
               for sk in shards]
        try:
            return _combined_snap(pss), [max(ps._versions.values()) for ps in pss]
        finally:
            for ps in pss:
                ps.shutdown()

    loaded, versions = _resume_snap()
    loaded2, _ = _resume_snap()
    assert _maxdiff(init, loaded) > 1e-4   # restored trained weights, not init...
    assert _maxdiff(loaded, loaded2) == 0  # ...exactly and reproducibly from the file
    assert all(v > 0 for v in versions)    # versions restored too

    # The scheduler's clock resumes (a fresh scheduler restores _T > 0 from the file).
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1), ("127.0.0.1", 2)], _diloco(),
                      batch_size=BATCH, host="127.0.0.1", port=0)
    try:
        sched._load_state(ckpt)
        assert sched._T > 0 and sum(sched._completed.values()) > 0
    finally:
        sched.shutdown()


def test_sharded_worker_reconnects_across_scheduler_restart(tmp_path):
    """The worker survives a scheduler crash: it reconnects to the restarted
    scheduler (same port, resumed) while the parameter servers stay up."""
    cfg = _cfg()
    dl = _diloco()
    ckpt = str(tmp_path / "ck")
    shards = _key_shards(cfg, 2)
    pss = [ParameterServer(cfg, sk, dl, host="127.0.0.1", port=0) for sk in shards]
    for ps in pss:
        ps.start()
    addrs = [("127.0.0.1", ps.port) for ps in pss]

    schA = Scheduler(cfg, _corpus(cfg), addrs, dl, batch_size=BATCH, host="127.0.0.1", port=0)
    schA.start()
    sport = schA.port
    w = threading.Thread(target=run_sharded_worker, args=(cfg, dl, ("127.0.0.1", sport)),
                         kwargs=dict(seed=0, reconnect=True, reconnect_timeout=20.0,
                                     heartbeat_interval=1.0), daemon=True)
    w.start()
    schA.fit(num_generations=1, checkpoint_dir=ckpt, checkpoint_every=1)
    schA.simulate_crash()  # scheduler dies; the parameter servers stay up

    schB = Scheduler(cfg, _corpus(cfg), addrs, dl, batch_size=BATCH,
                     host="127.0.0.1", port=sport)
    schB.start()
    finished = {}

    def run_b():
        schB.fit(num_generations=1, checkpoint_dir=ckpt, checkpoint_every=1, resume=True)
        finished["ok"] = True

    tb = threading.Thread(target=run_b, daemon=True)
    tb.start()
    tb.join(timeout=40)
    schB.shutdown()
    for ps in pss:
        ps.shutdown()
    w.join(timeout=20)

    assert finished.get("ok")          # B reached its target -> the worker reconnected
    assert schB._T >= schB._target


# -- multi-process: shards live in separate process memories -----------------
def _ps_proc(shard_keys, port_q, stop_ev):
    torch.set_num_threads(1)
    ps = ParameterServer(_cfg(), shard_keys, _diloco(), host="127.0.0.1", port=0)
    ps.start()
    port_q.put(ps.port)
    stop_ev.wait(timeout=120)
    ps.shutdown()


def _sched_proc(ps_ports, port_q, result_q, gens, stop_ev):
    torch.set_num_threads(1)
    addrs = [("127.0.0.1", p) for p in ps_ports]
    sched = Scheduler(_cfg(), _corpus(_cfg()), addrs, _diloco(), batch_size=BATCH,
                      host="127.0.0.1", port=0, heartbeat_timeout=30.0)
    sched.start()
    port_q.put(sched.port)
    completed = sched.fit(num_generations=gens, total_generations=gens)
    sched.shutdown()
    result_q.put({"T": sched._T, "target": sched._target, "completed_total": sum(completed.values())})
    stop_ev.set()  # let the PS processes exit


def _sharded_worker_proc(sched_port):
    torch.set_num_threads(1)
    run_sharded_worker(_cfg(), _diloco(), ("127.0.0.1", sched_port), seed=0, heartbeat_interval=2.0)


def test_sharded_multiprocess_end_to_end():
    ctx = mp.get_context("spawn")
    stop_ev = ctx.Event()
    s0, s1 = _key_shards(_cfg(), 2)
    pq0, pq1 = ctx.Queue(), ctx.Queue()
    ps0 = ctx.Process(target=_ps_proc, args=(s0, pq0, stop_ev))
    ps1 = ctx.Process(target=_ps_proc, args=(s1, pq1, stop_ev))
    ps0.start()
    ps1.start()
    p0, p1 = pq0.get(timeout=30), pq1.get(timeout=30)

    sport_q, result_q = ctx.Queue(), ctx.Queue()
    sched = ctx.Process(target=_sched_proc, args=([p0, p1], sport_q, result_q, 3, stop_ev))
    sched.start()
    sport = sport_q.get(timeout=30)
    workers = [ctx.Process(target=_sharded_worker_proc, args=(sport,)) for _ in range(2)]
    for w in workers:
        w.start()
    try:
        result = result_q.get(timeout=120)
    finally:
        stop_ev.set()
        for p in [sched, ps0, ps1, *workers]:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
    assert result["T"] >= result["target"]
    assert result["completed_total"] == result["T"]
