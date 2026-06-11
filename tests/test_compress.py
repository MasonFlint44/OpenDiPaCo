"""Tests for wire compression (compress.py; internet-scale plan §0c).

Units cover the int8 quantizer + error-feedback identity and the self-describing
decode; integration runs a real coordinator + workers (and the sharded trio) with
``compress="int8"`` and asserts training completes, the bank moves, and the bytes
on the wire actually shrink (bf16 weights down, int8 pseudo-gradients up).
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
from opendipaco.schedule import (
    CoordinatorServer,
    ParameterServer,
    Scheduler,
    assign_shards,
    run_sharded_worker,
    run_worker,
)
from opendipaco.schedule.compress import (
    compress_delta,
    compress_shard,
    compress_state,
    maybe_dequantize,
    restore_shard,
)

BATCH = 8
GENS = 3


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


# -- units ---------------------------------------------------------------------


def test_int8_roundtrip_error_bounded_and_ef_exact():
    """Dequantized int8 is within half a quantization step of the input, and the
    error-feedback identity holds exactly: dequant(payload) + residual == input."""
    g = torch.Generator().manual_seed(0)
    t = torch.randn(64, 33, generator=g) * 5.0
    payload, residual = compress_delta([t], "int8")
    (deq,) = maybe_dequantize(payload)
    step = float(t.abs().max()) / 127.0
    assert float((deq - t).abs().max()) <= step / 2 + 1e-6
    assert torch.allclose(deq + residual[0], t, atol=1e-6)   # nothing is lost


def test_bf16_roundtrip_and_ef_exact():
    g = torch.Generator().manual_seed(1)
    t = torch.randn(31, generator=g)
    payload, residual = compress_delta([t], "bf16")
    assert payload[0].dtype == torch.bfloat16
    (deq,) = maybe_dequantize(payload)
    assert torch.allclose(deq + residual[0], t, atol=1e-7)


def test_error_feedback_carries_into_next_round():
    """A second-round delta of zero still ships (approximately) the first round's
    quantization residual — the error is deferred, not dropped."""
    t = torch.linspace(-1.0, 1.0, 101)               # values that don't quantize cleanly
    _, res1 = compress_delta([t], "int8")
    assert float(res1[0].abs().max()) > 0            # round 1 really lost something
    payload2, _ = compress_delta([torch.zeros_like(t)], "int8", carry=res1)
    (deq2,) = maybe_dequantize(payload2)
    step2 = float(res1[0].abs().max()) / 127.0
    assert torch.allclose(deq2, res1[0], atol=step2 / 2 + 1e-6)


def test_zero_and_nonfinite_tensors():
    payload, residual = compress_delta([torch.zeros(5)], "int8")
    (deq,) = maybe_dequantize(payload)
    assert torch.equal(deq, torch.zeros(5))          # zero-scale path is exact
    bad = [torch.tensor([float("nan")])]
    (deq_bad,) = maybe_dequantize(compress_delta(bad, "int8")[0])
    assert not torch.isfinite(deq_bad).all()         # NaN survives to be guard-rejected


def test_compress_state_casts_floats_only():
    sd = {"w": torch.randn(3), "ids": torch.arange(4)}
    out = compress_state(sd, "int8")                 # weights are bf16 in any mode
    assert out["w"].dtype == torch.bfloat16
    assert out["ids"].dtype == torch.int64           # integer buffers untouched
    assert compress_state(sd, "none") is sd


def test_shard_roundtrip():
    shard = torch.randint(0, 48, (10, 16))
    assert compress_shard(shard, "none") is shard
    small = compress_shard(shard, "int8")
    assert small.dtype == torch.int32
    assert torch.equal(restore_shard(small), shard)  # lossless
    assert restore_shard(None) is None


def test_quantized_payload_survives_the_wire():
    from opendipaco.schedule.wire import decode, encode
    t = torch.randn(8, 3)
    payload, _ = compress_delta([t], "int8")
    out = decode(encode({"shared_grad": {"m": payload}}))
    (deq,) = maybe_dequantize(out["shared_grad"]["m"])
    (ref,) = maybe_dequantize(payload)
    assert torch.equal(deq, ref)


# -- end to end: coordinator -----------------------------------------------------


def _serve_compressed(compress, num_workers=2, gens=GENS):
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=10.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0,
                               heartbeat_timeout=10.0, compress=compress)
    tasks, submits = [], []
    orig_task, orig_recv = server._next_task, server._receive
    server._next_task = lambda req: (lambda t: (tasks.append(t) if t.get("type") == "task"
                                                else None, t)[1])(orig_task(req))
    server._receive = lambda m: (submits.append(m), orig_recv(m))[1]
    before = _snap(eng.bank)
    server.start()
    ws = [threading.Thread(target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
                           kwargs=dict(seed=0, reconnect=False, heartbeat_interval=1.0),
                           daemon=True)
          for _ in range(num_workers)]
    for w in ws:
        w.start()
    server.fit(num_generations=gens, total_generations=gens, log_every=0)
    server.shutdown()
    for w in ws:
        w.join(timeout=15)
    return server, eng, before, tasks, submits


def test_coordinator_int8_run_completes_with_compressed_wire():
    """An int8 run reaches its target with bf16 weights + int32 shards down and
    int8 pseudo-gradients up, and the bank still trains."""
    server, eng, before, tasks, submits = _serve_compressed("int8")
    assert server._T >= server._target
    assert _maxdiff(before, _snap(eng.bank)) > 1e-4

    for t in tasks:                                  # downlink: bf16 weights, int32 shard
        for sd in t["shared_weights"].values():
            assert all(v.dtype == torch.bfloat16 for v in sd.values()
                       if v.is_floating_point())
        if t["shard"] is not None:
            assert t["shard"].dtype == torch.int32
        assert t["compress"] == "int8"
    graded = [s for s in submits if s.get("shared_grad")]
    assert graded                                    # uplink: {"q": int8, "s": scale}
    for s in graded:
        for items in s["shared_grad"].values():
            assert all(isinstance(it, dict) and it["q"].dtype == torch.int8
                       for it in items)
    assert server.metrics.accepted_updates >= server._target


def test_int8_bytes_are_about_4x_smaller_per_submit():
    """The measured per-submit pseudo-gradient bytes drop ~4x vs fp32."""
    s_none, *_ = _serve_compressed("none", num_workers=1, gens=2)
    s_int8, *_ = _serve_compressed("int8", num_workers=1, gens=2)
    per_none = s_none.metrics.bytes_shared_grad / max(s_none.metrics.submits, 1)
    per_int8 = s_int8.metrics.bytes_shared_grad / max(s_int8.metrics.submits, 1)
    assert per_int8 < per_none * 0.3                 # int8 payload ≈ fp32/4


# -- end to end: sharded ----------------------------------------------------------


def test_sharded_int8_run_completes_and_trains():
    cfg = _cfg()
    dl = _diloco()
    ks = assign_shards(cfg.build_topology().module_keys(), 2)
    shards = [[k for k, s in ks.items() if s == i] for i in range(2)]
    pss = [ParameterServer(cfg, sk, dl, host="127.0.0.1", port=0, compress="int8")
           for sk in shards]
    for ps in pss:
        ps.start()
    befores = [_snap(ps.bank) for ps in pss]
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", ps.port) for ps in pss], dl,
                      batch_size=BATCH, host="127.0.0.1", port=0, compress="int8")
    sched.start()
    ws = [threading.Thread(target=run_sharded_worker,
                           args=(cfg, dl, ("127.0.0.1", sched.port)),
                           kwargs=dict(seed=0, heartbeat_interval=1.0), daemon=True)
          for _ in range(2)]
    for w in ws:
        w.start()
    sched.fit(num_generations=GENS, total_generations=GENS)
    sched.shutdown()
    for ps in pss:
        ps.shutdown()
    for w in ws:
        w.join(timeout=10)
    assert sched._T >= sched._target
    assert sched.metrics.invalid_rejected == 0       # quantized pushes decode cleanly
    for ps, before in zip(pss, befores):
        assert ps.metrics.invalid_rejected == 0
        assert _maxdiff(before, _snap(ps.bank)) > 1e-4   # every shard still trains


def test_submit_ack_reports_applied_and_residuals_stay_pure():
    """The coordinator's submit ack carries ``applied`` (True only when the update
    reached the bank), and ``_compress_contribution`` no longer mutates the
    residual store — the worker commits it only on a positive ack, so a
    rejected/stale submit can neither discard the previous error-feedback carry
    nor leak its own rounding error into a later accepted update."""
    from opendipaco.optim.diloco import make_outer_optimizer
    from opendipaco.schedule.scheduler import Contribution
    from opendipaco.schedule.distributed import (
        _commit_residuals,
        _compress_contribution,
    )
    from opendipaco.topology import is_private_key

    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0,
                               compress="int8")
    server._outer_opts = {k: make_outer_optimizer({k: eng.bank[k]}, eng.diloco)
                          for k in server._versions}
    with server._lock:
        server._serving = True
        server._target = 10 ** 9
        server._completed = {p: 0 for p in eng.topology.paths()}
    req = {"worker_id": "w", "warm_paths": [], "cached_shards": [], "have_shared": {}}

    # Build a worker-side contribution and encode it (residual store untouched).
    task = server._next_task(req)
    path = task["path"]
    shared_keys = [k for k in eng.topology.path_module_keys(path) if not is_private_key(k)]
    delta = {k: [torch.full_like(p, 0.123) for p in eng.bank[k].parameters()]
             for k in shared_keys}
    contrib = Contribution(path, eng.topology.path_index(path), 0.1, delta, {}, {})
    residuals: dict = {}
    payload, _priv, pending = _compress_contribution(contrib, "int8", residuals, path)
    assert residuals == {}                       # encoding is pure: nothing committed
    assert pending and all(pending[k] for k in shared_keys)

    # A zombie submit (wrong lease) is refused: ack says not applied.
    ack = server._handle({"type": "submit", "path": path, "lease": "dead-token",
                          "shared_grad": payload, "private_weights": {}, "loss": 0.1}, 0)
    assert ack == {"type": "ack", "applied": False}

    # The live lease's submit is applied: ack says so; the worker then commits.
    ack = server._handle({"type": "submit", "path": path, "lease": task["lease"],
                          "shared_grad": payload, "private_weights": {}, "loss": 0.1}, 0)
    assert ack == {"type": "ack", "applied": True}
    _commit_residuals(residuals, path, pending)
    assert set(residuals[path]) == set(shared_keys)

    # An invalid (NaN) submit on a fresh lease is also reported as not applied.
    task2 = server._next_task(req)
    bad = {k: [torch.full_like(p, float("nan")) for p in eng.bank[k].parameters()]
           for k in shared_keys}
    ack = server._handle({"type": "submit", "path": task2["path"], "lease": task2["lease"],
                          "shared_grad": bad, "private_weights": {}, "loss": 0.1}, 0)
    assert ack == {"type": "ack", "applied": False}

    # An empty pending commit must not wipe an existing carry.
    before = {k: [t.clone() for t in v] for k, v in residuals[path].items()}
    _commit_residuals(residuals, path, {})
    assert all(torch.equal(a, b) for k in before
               for a, b in zip(before[k], residuals[path][k]))
    server.shutdown()


def test_compress_mode_is_validated():
    import pytest
    cfg = _cfg()
    with pytest.raises(ValueError):
        CoordinatorServer(AsyncScheduler(_engine(cfg), lease_timeout=5.0), _corpus(cfg),
                          batch_size=BATCH, host="127.0.0.1", port=0, compress="zip")


def test_int8_training_tracks_uncompressed_loss():
    """Compression must not wreck the toy run: the int8 run's final mean inner loss
    stays close to the fp32 run's (loose tolerance; async runs aren't bit-stable)."""
    def final_loss(compress):
        cfg = _cfg()
        eng = _engine(cfg)
        server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=10.0), _corpus(cfg),
                                   batch_size=BATCH, host="127.0.0.1", port=0,
                                   heartbeat_timeout=10.0, compress=compress)
        losses = []
        orig = server._receive
        server._receive = lambda m: (losses.append(m.get("loss")), orig(m))[1]
        server.start()
        w = threading.Thread(target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
                             kwargs=dict(seed=0, reconnect=False), daemon=True)
        w.start()
        server.fit(num_generations=4, total_generations=4, log_every=0)
        server.shutdown()
        w.join(timeout=15)
        tail = [x for x in losses if x is not None and math.isfinite(x)][-4:]
        return sum(tail) / len(tail)

    base, quant = final_loss("none"), final_loss("int8")
    assert math.isfinite(base) and math.isfinite(quant)
    assert quant < base * 1.5 + 0.5                  # same ballpark, not divergence
