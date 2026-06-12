"""Tests for the multi-node socket transport (bounded-staleness async coordinator).

The network ``CoordinatorServer`` is asynchronous: workers run ahead out of
lockstep, stale contributions are damped/rejected, and stragglers never block.
So we assert *behavior* (the run completes, the bank moves, the staleness bound is
enforced, mechanics like warm-shipping / auth / reconnect still hold) rather than
matching a deterministic reference -- the in-process ``AsyncScheduler`` remains the
deterministic correctness anchor (``test_scheduler.py``).

Multi-process tests spawn a coordinator + workers over real TCP; the rest run in
one process with worker threads so they can inspect payloads / metrics / restart.
"""

import io
import multiprocessing as mp
import socket
import threading
import time

import torch

from opendipaco import (
    AsyncScheduler,
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
)
from opendipaco.checkpoint import latest_checkpoint
from opendipaco.data import ShardedCorpus
from opendipaco.optim.diloco import make_outer_optimizer
from opendipaco.schedule import CoordinatorServer, run_worker
from opendipaco.topology import is_private_key

GENS = 3
BATCH = 8


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _cfg_private():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    # Private embedding -> per-path private modules, exercising private shipping.
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16, embedding="private")


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


def _snap(engine):
    return {k: {n: p.detach().clone() for n, p in m.named_parameters()}
            for k, m in engine.bank.items()}


def _maxdiff(a, b):
    return max(float((a[k][n] - b[k][n]).abs().max()) for k in a for n in a[k])


def _to_bytes(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _from_bytes(data: bytes):
    return torch.load(io.BytesIO(data), weights_only=False)


# -- in-process driver (worker threads; lets tests read metrics directly) ----
def _serve(num_workers, *, gens=GENS, cfg=None, staleness_bound=None,
           heartbeat_timeout=5.0, heartbeat_interval=1.0, hook=None, spy=None):
    cfg = cfg or _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(
        AsyncScheduler(eng, lease_timeout=heartbeat_timeout), _corpus(cfg),
        batch_size=BATCH, host="127.0.0.1", port=0,
        heartbeat_timeout=heartbeat_timeout, staleness_bound=staleness_bound,
    )
    if spy is not None:
        orig = server._next_task
        server._next_task = lambda req: spy(orig(req))
    before = _snap(eng)
    server.start()
    ws = [
        threading.Thread(
            target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
            kwargs=dict(seed=0, reconnect=False, heartbeat_interval=heartbeat_interval,
                        fault_hook=hook),
            daemon=True,
        )
        for _ in range(num_workers)
    ]
    for w in ws:
        w.start()
    completed = server.fit(num_generations=gens, total_generations=gens, log_every=0)
    server.shutdown()
    for w in ws:
        w.join(timeout=15)
    return server, eng, before, completed


# -- multi-process harness ---------------------------------------------------
def _coordinator_main(port_q, result_q, gens, ckpt_dir, ckpt_every, resume):
    torch.set_num_threads(1)
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=30.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0, heartbeat_timeout=30.0)
    server.start()
    port_q.put(server.port)
    before = _snap(eng)
    completed = server.fit(num_generations=gens, total_generations=gens, log_every=0,
                           checkpoint_dir=ckpt_dir, checkpoint_every=ckpt_every, resume=resume)
    server.shutdown()
    result_q.put(_to_bytes({
        "bank": _snap(eng),
        "completed": completed,
        "T": server._T,
        "target": server._target,
        "moved": _maxdiff(before, _snap(eng)) > 1e-4,
    }))


def _worker_main(port, max_tasks=None):
    torch.set_num_threads(1)
    run_worker(_cfg(), _diloco(), "127.0.0.1", port, seed=0, max_tasks=max_tasks, reconnect=False)


def _run_distributed(worker_specs, *, gens=GENS, ckpt_dir=None, ckpt_every=0,
                     resume=False, timeout=120):
    ctx = mp.get_context("spawn")
    port_q, result_q = ctx.Queue(), ctx.Queue()
    coord = ctx.Process(target=_coordinator_main,
                        args=(port_q, result_q, gens, ckpt_dir, ckpt_every, resume))
    coord.start()
    port = port_q.get(timeout=30)
    workers = [ctx.Process(target=_worker_main, args=(port,), kwargs=kw) for kw in worker_specs]
    for w in workers:
        w.start()
    try:
        result = result_q.get(timeout=timeout)
    finally:
        coord.join(timeout=10)
        for w in workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
        if coord.is_alive():
            coord.terminate()
    return _from_bytes(result)


# -- async behavior ----------------------------------------------------------
def test_async_run_completes_and_moves_bank():
    """A coordinator + 2 worker processes over TCP run to the update target."""
    got = _run_distributed([{}, {}])
    assert got["T"] >= got["target"]            # reached the async update budget
    assert got["moved"]                          # training changed the weights
    # The run met its budget. (Not ``== T``: ``completed`` is snapshotted when fit
    # returns, but late in-flight commits during shutdown's grace can still bump T,
    # so exact equality is an async race -- the budget is the meaningful invariant.)
    assert sum(got["completed"].values()) >= got["target"]


def test_async_stragglers_dont_block():
    """One slow worker holding a path doesn't stop the others; progress is uneven."""
    slow = _cfg().build_topology().path_from_index(0)
    # The straggler's first task is far slower than the whole rest of the run, so
    # the other worker reaches the target while it's still in flight -> the slow
    # path stays well behind (robustly uneven), proving it didn't block the run.
    server, eng, before, completed = _serve(
        2, gens=GENS, hook=_slow_first_hook(slow, 3.0),
        heartbeat_timeout=5.0, heartbeat_interval=0.1,
    )
    assert server._T >= server._target          # reached target despite the straggler
    # Async -> paths advance at different rates (sync would be perfectly even).
    assert max(completed.values()) > min(completed.values())
    assert _maxdiff(before, _snap(eng)) > 1e-4


def test_async_staleness_bound_enforced():
    """A contribution stale by more than the bound is rejected (bank unchanged);
    one within the bound is applied. Driven directly for determinism."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0, staleness_bound=1)
    server._outer_opts = {k: make_outer_optimizer({k: eng.bank[k]}, eng.diloco)
                          for k in server._versions}
    with server._lock:
        server._serving = True
        server._target = 10 ** 9
        server._completed = {p: 0 for p in eng.topology.paths()}
    req = {"worker_id": "w", "warm_paths": [], "cached_shards": [], "have_shared": {}}

    # Lease a path (issued at T=0), then advance the clock past the bound.
    task = server._next_task(req)
    path = task["path"]
    shared_keys = [k for k in eng.topology.path_module_keys(path) if not is_private_key(k)]
    grad = {k: [torch.ones_like(p) for p in eng.bank[k].parameters()] for k in shared_keys}
    with server._lock:
        server._T = 5  # 5 outer steps happened since this path was issued -> staleness 5

    before = _snap(eng)
    server._receive({"path": path, "lease": task["lease"], "shared_grad": grad,
                     "private_weights": {}, "loss": 0.1})
    assert server.metrics.stale_rejected == 1
    assert _maxdiff(before, _snap(eng)) == 0.0       # rejected -> bank untouched

    # Re-lease (issued at the current T) and submit -> staleness 0 -> accepted.
    task2 = server._next_task(req)
    server._receive({"path": path, "lease": task2["lease"], "shared_grad": grad,
                     "private_weights": {}, "loss": 0.1})
    assert server.metrics.accepted_updates == 1
    assert server.metrics.max_staleness <= 1
    assert _maxdiff(before, _snap(eng)) > 0.0        # accepted -> the outer step moved weights
    server.shutdown()


def test_zombie_submit_fenced_after_release():
    """A submission from an expired (reclaimed + re-leased) lease is dropped; only
    the current lease holder's result is applied. Without the lease-token fence the
    zombie's stale result would be accepted (with understated staleness) and the
    live worker's fresher result silently discarded."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0)
    server._outer_opts = {k: make_outer_optimizer({k: eng.bank[k]}, eng.diloco)
                          for k in server._versions}
    with server._lock:
        server._serving = True
        server._target = 10 ** 9
        server._completed = {p: 0 for p in eng.topology.paths()}

    task_a = server._next_task(
        {"worker_id": "A", "warm_paths": [], "cached_shards": [], "have_shared": {}})
    path = task_a["path"]
    # Worker A goes silent; its lease expires and is reclaimed.
    with server._lock:
        server._inflight[path] = 0.0  # force the deadline into the past
        server._reclaim_inflight_locked()
    # The path is re-leased to worker B under a fresh token.
    task_b = server._next_task(
        {"worker_id": "B", "warm_paths": [], "cached_shards": [], "have_shared": {}})
    assert task_b["path"] == path and task_b["lease"] != task_a["lease"]

    shared_keys = [k for k in eng.topology.path_module_keys(path) if not is_private_key(k)]
    grad = {k: [torch.ones_like(p) for p in eng.bank[k].parameters()] for k in shared_keys}
    before = _snap(eng)

    # Zombie A submits against its dead lease -> dropped, bank untouched.
    server._receive({"path": path, "lease": task_a["lease"], "shared_grad": grad,
                     "private_weights": {}, "loss": 0.1})
    assert server.metrics.accepted_updates == 0
    assert _maxdiff(before, _snap(eng)) == 0.0

    # B, the live lease holder, submits -> applied.
    server._receive({"path": path, "lease": task_b["lease"], "shared_grad": grad,
                     "private_weights": {}, "loss": 0.1})
    assert server.metrics.accepted_updates == 1
    assert _maxdiff(before, _snap(eng)) > 0.0
    server.shutdown()


def test_async_checkpoint_restores_clock_momentum_and_schedule(tmp_path):
    """A resumed coordinator restores the async clock, per-path completion counts
    (the inner-LR schedule position), shared-module versions, and the per-key outer
    Nesterov momentum -- not just the engine bank. (Previously only the engine was
    checkpointed, so resume reset the momentum and restarted every path's cosine
    schedule at generation 0.)"""
    cfg = _cfg()
    eng = _engine(cfg)
    A = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                          batch_size=BATCH, host="127.0.0.1", port=0)
    A._outer_opts = {k: make_outer_optimizer({k: eng.bank[k]}, eng.diloco)
                     for k in A._versions}
    with A._lock:
        A._serving = True
        A._target = 10 ** 9
        A._completed = {p: 0 for p in eng.topology.paths()}
    task = A._next_task(
        {"worker_id": "w", "warm_paths": [], "cached_shards": [], "have_shared": {}})
    path = task["path"]
    shared_keys = [k for k in eng.topology.path_module_keys(path) if not is_private_key(k)]
    grad = {k: [torch.ones_like(p) for p in eng.bank[k].parameters()] for k in shared_keys}
    A._receive({"path": path, "lease": task["lease"], "shared_grad": grad,
                "private_weights": {}, "loss": 0.1})
    assert A._T == 1
    ckpt = str(tmp_path / "ck")
    A._save_cluster_checkpoint(ckpt)
    A.shutdown()

    eng_b = _engine(cfg, seed=7)  # different init; resume must overwrite it
    B = CoordinatorServer(AsyncScheduler(eng_b, lease_timeout=5.0), _corpus(cfg),
                          batch_size=BATCH, host="127.0.0.1", port=0)
    # num_generations=0 -> fit restores state and returns at the (already met) target.
    B.fit(num_generations=0, log_every=0, checkpoint_dir=ckpt, resume=True)

    assert B._T == 1
    assert B._completed[path] == 1        # the inner-LR schedule continues, not restarts
    assert B._versions == A._versions     # warm workers' version caches stay meaningful
    touched = [k for k in A._outer_opts if A._outer_opts[k].state_dict()["state"]]
    assert touched                        # the outer step left momentum to restore
    for k in touched:
        sa = A._outer_opts[k].state_dict()["state"]
        sb = B._outer_opts[k].state_dict()["state"]
        assert set(sa) == set(sb)
        for i in sa:
            assert torch.equal(sa[i]["momentum_buffer"], sb[i]["momentum_buffer"])
    B.shutdown()


def test_distributed_tolerates_workers_leaving():
    """Workers with a finite budget leave mid-run; the run still reaches target."""
    got = _run_distributed([{"max_tasks": 1}, {"max_tasks": 1}, {}])
    assert got["T"] >= got["target"]
    assert got["moved"]


# -- backpressure / connection scaling ---------------------------------------
def test_bounded_io_threads_serve_many_workers():
    """A fixed pool of I/O threads serves many more connections than threads."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0,
                               io_threads=2, heartbeat_timeout=5.0)
    server.start()
    ws = [threading.Thread(target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
                           kwargs=dict(seed=0, reconnect=False, heartbeat_interval=1.0),
                           daemon=True)
          for _ in range(8)]
    for w in ws:
        w.start()
    server.fit(num_generations=2, log_every=0)
    server.shutdown()
    for w in ws:
        w.join(timeout=10)
    assert len(server._io) == 2        # fixed I/O threads, independent of the 8 workers
    assert server._T >= server._target  # the reactor served them all to completion


def test_max_connections_cap_refuses_excess():
    """A connection beyond ``max_connections`` is refused (closed) by the server."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0, max_connections=1)
    server.start()
    a = socket.create_connection(("127.0.0.1", server.port))  # accepted (open=1)
    time.sleep(0.2)
    b = socket.create_connection(("127.0.0.1", server.port))  # over the cap -> refused
    b.settimeout(2.0)
    try:
        assert b.recv(64) == b""       # server closed the over-cap connection
    finally:
        a.close()
        b.close()
        server.shutdown()


# -- transport mechanics (carry over from the sync coordinator) --------------
def test_warm_shipping_drops_opt_private_shard():
    """After a worker is warm on a path, its tasks ship only the changed shared
    weights -- no optimizer state, private modules, or shard."""
    tasks = []
    _serve(1, gens=GENS, cfg=_cfg_private(),
           spy=lambda t: (tasks.append(t) if t.get("type") == "task" else None, t)[1])

    by_path: dict = {}
    for t in tasks:
        by_path.setdefault(t["path"], []).append(t)
    assert by_path
    for ts in by_path.values():
        assert ts[0]["private_weights"] is not None and ts[0]["shard"] is not None  # cold
        for t in ts[1:]:
            assert t["private_weights"] is None and t["shard"] is None              # warm
    assert all("opt_state" not in t for t in tasks)
    assert any(t["shared_weights"] for t in tasks[1:])  # shared modules still sync


def test_metrics_track_bytes_and_prove_no_optimizer_on_wire():
    """Metrics measure the P0 win: optimizer state never on the wire; shard/private
    shipped once per path (cold)."""
    server, _eng, _before, _completed = _serve(1, gens=GENS, cfg=_cfg_private())
    m = server.metrics.summary()
    cfg = _cfg_private()
    assert m["accepted_updates"] >= cfg.num_paths * GENS        # reached the update budget
    assert m["bytes_opt"] == 0                                  # never shipped (the win)
    assert m["tasks_with_shard"] == cfg.num_paths               # shard down once per path
    assert m["tasks_with_private_down"] == cfg.num_paths        # private down once (cold)
    assert m["bytes_down"] > 0 and m["bytes_up"] > 0
    assert m["bytes_shared_grad"] > 0 and m["bytes_private_up"] > 0


def test_auth_allows_matching_key():
    """A worker that proves the shared secret authenticates and does work."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0, auth_key="s3cret")
    server.start()
    w = threading.Thread(target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
                         kwargs=dict(seed=0, reconnect=False, auth_key="s3cret"), daemon=True)
    w.start()
    server.fit(num_generations=2, total_generations=2, log_every=0)
    server.shutdown()
    w.join(timeout=10)
    assert server._T >= server._target
    assert server.metrics.accepted_updates >= cfg.num_paths * 2  # the worker really worked


def test_auth_rejects_wrong_key():
    """A worker with the wrong key is rejected at the handshake (no work served)."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0, auth_key="right")
    server.start()
    err = {}

    def run():
        try:
            run_worker(cfg, _diloco(), "127.0.0.1", server.port, seed=0,
                       reconnect=False, auth_key="wrong")
        except Exception as e:  # noqa: BLE001
            err["e"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=10)
    server.shutdown()
    assert isinstance(err.get("e"), PermissionError)
    assert server.metrics.tasks_sent == 0


def _slow_first_hook(slow_path, seconds=1.0):
    """Hook whose *first* execution of ``slow_path`` (across all workers) blocks."""
    st = {"n": 0}
    lk = threading.Lock()

    def hook(path, attempt):
        if path == slow_path:
            with lk:
                st["n"] += 1
                first = st["n"] == 1
            if first:
                time.sleep(seconds)

    return hook


def test_heartbeat_refreshes_lease_and_reclaim_frees_dead():
    """Heartbeats keep an in-flight lease alive; without them, the deadline passes
    and reclaim frees the path (decoupling liveness from task duration)."""
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=0.3), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0, heartbeat_timeout=0.3)
    # Put the coordinator in a serving state without launching a real fit.
    with server._lock:
        server._serving = True
        server._target = 10 ** 9
        server._completed = {p: 0 for p in eng.topology.paths()}
    req = {"worker_id": "w", "warm_paths": [], "cached_shards": [], "have_shared": {}}
    task = server._next_task(req)
    path = task["path"]
    assert path in server._inflight
    d0 = server._inflight[path]

    time.sleep(0.05)
    server._heartbeat({"path": path, "worker_id": "w", "lease": task["lease"]})
    assert server._inflight[path] > d0           # heartbeat pushed the deadline out

    time.sleep(0.4)                              # now let it lapse without heartbeats
    with server._lock:
        server._reclaim_inflight_locked()
    assert path not in server._inflight          # dead lease reclaimed
    assert server.metrics.reclaims >= 1
    server.shutdown()


def test_checkpoint_resume_continues(tmp_path):
    """A resumed coordinator continues from the checkpoint, not from scratch.

    ``scratch`` trains GENS from init; ``resumed`` loads a GENS-trained checkpoint
    and trains GENS more (≈ 2·GENS from init), so it ends *farther* from the
    deterministic init -- proving it continued rather than restarting.
    """
    init = _snap(_engine(_cfg()))  # deterministic untrained init (seed 0)
    ckpt = str(tmp_path / "ck")
    r1 = _run_distributed([{}, {}], gens=GENS, ckpt_dir=ckpt, ckpt_every=2)
    assert r1["T"] >= r1["target"] and latest_checkpoint(ckpt)

    resumed = _run_distributed([{}, {}], gens=GENS, ckpt_dir=ckpt, ckpt_every=2, resume=True)
    scratch = _run_distributed([{}, {}], gens=GENS)  # fresh init, no resume

    assert _maxdiff(init, resumed["bank"]) > _maxdiff(init, scratch["bank"])


def test_worker_reconnects_across_coordinator_restart(tmp_path):
    """A worker survives a coordinator crash: it reconnects to the restarted server
    (same port, resumed from checkpoint) and the run finishes."""
    cfg = _cfg()
    ckpt = str(tmp_path / "ck")

    engA = _engine(cfg)
    A = CoordinatorServer(AsyncScheduler(engA, lease_timeout=5.0), _corpus(cfg),
                          batch_size=BATCH, host="127.0.0.1", port=0)
    A.start()
    port = A.port
    w = threading.Thread(
        target=run_worker, args=(cfg, _diloco(), "127.0.0.1", port),
        kwargs=dict(seed=0, reconnect=True, reconnect_timeout=20.0), daemon=True,
    )
    w.start()
    A.fit(num_generations=1, log_every=0, checkpoint_dir=ckpt, checkpoint_every=1)
    A.simulate_crash()  # abrupt drop (no graceful "stop") -> worker reconnects

    engB = _engine(cfg, seed=5)
    B = CoordinatorServer(AsyncScheduler(engB, lease_timeout=5.0), _corpus(cfg),
                          batch_size=BATCH, host="127.0.0.1", port=port)
    B.start()
    finished = {}

    def run_b():
        B.fit(num_generations=1, log_every=0, checkpoint_dir=ckpt, checkpoint_every=1, resume=True)
        finished["ok"] = True

    tb = threading.Thread(target=run_b, daemon=True)
    tb.start()
    tb.join(timeout=40)
    B.shutdown()
    w.join(timeout=20)

    assert finished.get("ok")          # B reached its target -> the worker reconnected
    assert B._T >= B._target
