"""Tests for W2b structured sparsification of up pseudo-gradients
(docs/w2-bandwidth-design.md D6).

The worker keeps each pseudo-gradient's top ``density`` fraction (per output-row
for 2-D weights) and error-feeds the dropped mass. These check the codec, per-row
structure, mass conservation (error feedback loses nothing), byte-identity when
off, malformed-input refusal, and an end-to-end run.
"""

import pytest
import torch

from opendipaco.schedule.compress import compress_delta, maybe_dequantize


def test_sparsify_keeps_top_k_per_row_and_conserves_mass():
    """Per-row top-k for a 2-D weight; the dropped mass goes *entirely* into the
    residual (error feedback loses nothing): reconstruction + residual == input."""
    torch.manual_seed(0)
    d = torch.randn(4, 10)
    payload, residual = compress_delta([d], "none", density=0.3)  # keep ceil(0.3*10)=3/row
    recon = maybe_dequantize(payload)[0]
    assert recon.shape == d.shape
    # Exactly 3 kept per row (structured), zeros elsewhere.
    assert (recon != 0).sum(dim=1).tolist() == [3, 3, 3, 3]
    # Error feedback is lossless bookkeeping: nothing vanishes.
    assert torch.allclose(recon + residual[0], d, atol=1e-6)
    # The kept entries are the largest-magnitude ones (per row).
    for r in range(4):
        kept = d[r].abs().topk(3).values.sort().values
        got = recon[r][recon[r] != 0].abs().sort().values
        assert torch.allclose(got, kept, atol=1e-6)


def test_sparsify_flat_for_non_2d():
    d = torch.randn(20)
    payload, _ = compress_delta([d], "none", density=0.25)        # keep 5 of 20
    recon = maybe_dequantize(payload)[0]
    assert (recon != 0).sum().item() == 5


def test_sparse_error_feedback_eventually_sends_dropped_mass():
    """A constant gradient that's mostly dropped each round still gets delivered:
    the carried residual grows until those entries become top-k. Over many rounds
    the cumulative reconstruction approaches the cumulative input (no systematic
    bias toward the always-largest entries)."""
    g = torch.tensor([1.0, 1.0, 1.0, 1.0])   # uniform -> each round only some kept
    carry, sent_sum = None, torch.zeros(4)
    rounds = 40
    for _ in range(rounds):
        payload, residual = compress_delta([g], "none", carry=carry, density=0.25)  # 1 of 4
        sent_sum += maybe_dequantize(payload)[0]
        carry = residual                              # the per-tensor residual list
    # Exact mass conservation (mode none): everything sent plus what's still
    # carried equals the cumulative input -- nothing is lost or invented.
    assert torch.allclose(sent_sum + carry[0], torch.full((4,), float(rounds)), atol=1e-5)
    assert sent_sum.min() > 0                        # every coordinate served (no starve)


def test_density_one_is_byte_identical_dense():
    """density=1.0 takes the existing dense path: no sparse payloads."""
    d = torch.randn(3, 3)
    # mode none + dense -> raw passthrough, no residual.
    payload, res = compress_delta([d], "none", density=1.0)
    assert res is None and torch.equal(payload[0], d)
    # int8 + dense -> {"q","s"}, never {"sp"}.
    payload, _ = compress_delta([d], "int8", density=1.0)
    assert "q" in payload[0] and "sp" not in payload[0]


def test_sparse_int8_values_roundtrip():
    """Sparsification composes with int8 value encoding (a step toward W2c): kept
    values are int8-quantized and dequantized on receipt."""
    torch.manual_seed(1)
    d = torch.randn(4, 8)
    payload, _ = compress_delta([d], "int8", density=0.5)
    assert "sp" in payload[0] and "q" in payload[0]["v"]      # sparse + int8 values
    recon = maybe_dequantize(payload)[0]
    assert (recon != 0).sum(dim=1).tolist() == [4, 4, 4, 4]   # top-4 of 8 per row


def test_sparsify_stays_on_input_device():
    """The payload + reconstruction stay on the pseudo-gradient's device (a GPU
    worker's gradient is on cuda; a CPU arange/zeros would device-mismatch the
    GPU topk indices). Pins device-consistency -- also runs on cuda if present."""
    from opendipaco.schedule.compress import _sparsify

    for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        d = torch.randn(4, 8, device=dev)
        payload, recon = _sparsify(d, 0.5, "int8")
        assert recon.device.type == d.device.type
        assert payload["i"].device.type == d.device.type
        assert payload["v"]["q"].device.type == d.device.type


def test_malformed_sparse_payload_refused():
    """A Byzantine peer (valid grant) must not crash the owner with a crafted
    sparse payload: bad value type, out-of-bounds / mismatched indices, or a
    negative shape all raise a *caught* TypeError/ValueError, not an uncaught
    IndexError/RuntimeError from the scatter (the push path treats these as an
    invalid push)."""
    with pytest.raises(TypeError):
        maybe_dequantize([{"sp": [2, 2], "i": torch.tensor([0]), "v": "not a tensor"}])
    with pytest.raises(ValueError):                                  # index >= numel
        maybe_dequantize([{"sp": [2, 2], "i": torch.tensor([99]), "v": torch.tensor([1.0])}])
    with pytest.raises(ValueError):                                  # negative index
        maybe_dequantize([{"sp": [2, 2], "i": torch.tensor([-1]), "v": torch.tensor([1.0])}])
    with pytest.raises(ValueError):                                  # length mismatch
        maybe_dequantize([{"sp": [2, 2], "i": torch.tensor([0, 1]), "v": torch.tensor([1.0])}])
    with pytest.raises(ValueError):                                  # negative shape dim
        maybe_dequantize([{"sp": [-1, 2], "i": torch.tensor([0]), "v": torch.tensor([1.0])}])


def test_owner_rejects_malformed_sparse_push_without_crashing():
    """End to end: an owner served a crafted out-of-bounds sparse push (with a
    valid grant) rejects it (applied=False) and keeps serving -- the decode
    ValueError is caught by the push path, not an uncaught scatter crash."""
    from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
    from opendipaco.schedule import ParameterServer, make_grant
    from opendipaco.topology import is_private_key

    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    cfg = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)
    keys = sorted(cfg.build_topology().module_keys())
    shared = next(k for k in keys if not is_private_key(k))
    path = cfg.build_topology().path_from_index(0)
    ps = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=2), host="127.0.0.1",
                         port=0, grant_key="s")
    try:
        nparams = len(list(ps.bank[shared].parameters()))
        bad = [{"sp": [4], "i": torch.tensor([9999]), "v": torch.tensor([1.0])}
               for _ in range(nparams)]
        grant = make_grant(path, [shared], 1.0, "tok", grant_key="s")
        r = ps._push({"grant": grant, "updates": {shared: {"grad": bad}}})
        assert r["applied"] is False              # rejected, not crashed
        # The owner still serves a normal fetch afterward.
        assert ps._fetch({"keys": [shared], "have": {}})["versions"][shared] == (0, 0)
    finally:
        ps.shutdown()


def test_owner_rejects_huge_declared_sparse_shape_no_oom():
    """A tiny sparse push that *claims* a huge dense shape must be refused before
    maybe_dequantize allocates math.prod(shape) -- max_msg_bytes bounds the
    encoded frame, not the densified tensor, so this would OOM the owner. The
    declared shape is validated against the target param pre-decode."""
    from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
    from opendipaco.schedule import ParameterServer, make_grant
    from opendipaco.topology import is_private_key

    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    cfg = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)
    keys = sorted(cfg.build_topology().module_keys())
    shared = next(k for k in keys if not is_private_key(k))
    path = cfg.build_topology().path_from_index(0)
    ps = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=2), host="127.0.0.1",
                         port=0, grant_key="s")
    try:
        nparams = len(list(ps.bank[shared].parameters()))
        # 1 index/value but a declared dense shape of a billion -> ~4 GB if densified.
        huge = [{"sp": [10 ** 9], "i": torch.tensor([0], dtype=torch.int32),
                 "v": torch.tensor([1.0])} for _ in range(nparams)]
        grant = make_grant(path, [shared], 1.0, "tok", grant_key="s")
        r = ps._push({"grant": grant, "updates": {shared: {"grad": huge}}})
        assert r["applied"] is False                       # rejected pre-decode, no alloc
        assert ps._fetch({"keys": [shared], "have": {}})["versions"][shared] == (0, 0)
    finally:
        ps.shutdown()


def test_run_local_sharded_trains_with_sparse_up():
    """A full sharded cluster with transport.up_density < 1 trains to budget."""
    from opendipaco.launch import run_local
    from opendipaco.launch.config import LaunchConfig

    cfg = LaunchConfig.from_dict({
        "mode": "sharded",
        "model": {"vocab_size": 64, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "max_position_embeddings": 64,
                  "layers_per_level": [1, 1], "level_sizes": [2, 2], "sequence_length": 16},
        "diloco": {"inner_steps": 4, "inner_lr": 1e-3},
        "data": {"source": "synthetic", "num_documents": 64},
        "transport": {"compress": "int8", "up_density": 0.25},
        "sharded": {"num_shards": 2, "parameter_servers": [["127.0.0.1", 0], ["127.0.0.1", 0]]},
        "run": {"generations": 2, "batch_size": 8, "local_workers": 2},
    })
    server, completed = run_local(cfg)
    assert sum(completed.values()) >= 2 * cfg.model.level_sizes[0] * cfg.model.level_sizes[1]
    assert server.metrics.accepted_updates > 0


def test_delta_down_and_sparse_up_compose():
    """Both bandwidth levers on at once (the real deployment config): delta-down
    on the weights AND sparsified pseudo-gradients up. They are orthogonal
    directions sharing no state, but train end to end together."""
    from opendipaco.launch import run_local
    from opendipaco.launch.config import LaunchConfig

    cfg = LaunchConfig.from_dict({
        "mode": "sharded",
        "model": {"vocab_size": 64, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "max_position_embeddings": 64,
                  "layers_per_level": [1, 1], "level_sizes": [2, 2], "sequence_length": 16},
        "diloco": {"inner_steps": 4, "inner_lr": 1e-3},
        "data": {"source": "synthetic", "num_documents": 64},
        "transport": {"compress": "int8", "down": "delta", "up_density": 0.25},
        "sharded": {"num_shards": 2, "parameter_servers": [["127.0.0.1", 0], ["127.0.0.1", 0]]},
        "run": {"generations": 2, "batch_size": 8, "local_workers": 2},
    })
    server, completed = run_local(cfg)
    assert sum(completed.values()) >= 2 * cfg.model.level_sizes[0] * cfg.model.level_sizes[1]
    assert server.metrics.accepted_updates > 0
