"""Tests for W2c sub-int8 quantization (int4 + per-group scale)
(docs/w2-bandwidth-design.md D7).

int4 stacks on the W2a delta-down and W2b sparsification paths: it's the value
encoding for dense pseudo-gradients, sparse kept values, and down deltas. These
check the per-group quantizer, packing (~0.5 B/elem), the round-trips through
each path, malformed-input refusal, and an end-to-end run.
"""

import pytest
import torch

from opendipaco.schedule.compress import (
    _dequant_int4,
    _quantize_int4,
    apply_state_delta,
    compress_delta,
    encode_state_delta,
    maybe_dequantize,
)


def test_int4_per_group_roundtrip_and_packing():
    """int4 reconstructs within half a per-group quant step, packs two values per
    byte (~0.5 B/elem), and the residual is exactly input − reconstruction."""
    torch.manual_seed(0)
    t = torch.randn(256)                                  # 2 groups of 128
    payload, residual = _quantize_int4(t, group_size=128)
    deq = _dequant_int4(payload)
    assert payload["q4"].numel() == 128                   # 256 nibbles -> 128 bytes
    assert payload["s"].numel() == 2                      # one scale per group
    # Per-group error bound: step = group_absmax / 7, error <= step/2.
    step = t.reshape(2, 128).abs().amax(dim=1) / 7.0
    bound = (step / 2 * 1.001).repeat_interleave(128)
    assert (deq - t).abs().le(bound + 1e-6).all()
    assert torch.allclose(residual.reshape(-1), t - deq, atol=1e-5)


def test_int4_per_group_scale_adapts():
    """A per-group scale lets a small group and a large group both reconstruct
    well -- a single per-tensor scale would swamp the small group."""
    torch.manual_seed(1)
    t = torch.cat([torch.randn(128) * 0.01, torch.randn(128) * 100.0])
    deq = _dequant_int4(_quantize_int4(t)[0])
    for lo, hi in ((0, 128), (128, 256)):
        rel = (deq[lo:hi] - t[lo:hi]).abs().max() / t[lo:hi].abs().max()
        assert rel < 0.15                                 # ~1/7 worst case, both groups


def test_int4_through_dense_sparse_and_delta_paths():
    """int4 round-trips through all three W2 consumers."""
    torch.manual_seed(2)
    # dense pseudo-gradient (compress_delta + maybe_dequantize)
    d = torch.randn(4, 64)
    payload, _ = compress_delta([d], "int4")
    assert "q4" in payload[0]
    recon = maybe_dequantize(payload)[0]
    assert recon.shape == d.shape and (recon - d).abs().max() < d.abs().max()
    # sparse kept values encoded int4
    payload, _ = compress_delta([d], "int4", density=0.5)
    assert "sp" in payload[0] and "q4" in payload[0]["v"]
    assert (maybe_dequantize(payload)[0] != 0).sum(dim=1).tolist() == [32, 32, 32, 32]
    # down delta encoded int4
    base = {"w": torch.randn(2, 130)}
    cur = {"w": base["w"] + 0.01 * torch.randn(2, 130)}
    tensors = encode_state_delta(cur, base, mode="int4")
    assert "q4" in tensors["w"]
    out = apply_state_delta(base, tensors)
    assert out["w"].shape == cur["w"].shape
    assert (out["w"] - cur["w"]).abs().max() < (base["w"] - cur["w"]).abs().max()


def test_malformed_int4_payload_refused():
    """A crafted int4 payload (length mismatch / bad fields) raises a caught
    ValueError, not an uncaught reshape crash."""
    good, _ = _quantize_int4(torch.randn(128))
    with pytest.raises(ValueError):
        _dequant_int4({**good, "s": torch.zeros(99)})          # wrong scale count
    with pytest.raises(ValueError):
        _dequant_int4({**good, "q4": torch.zeros(7, dtype=torch.uint8)})  # wrong byte count
    with pytest.raises(ValueError):
        _dequant_int4({"q4": good["q4"], "s": good["s"], "g": 0, "n": 128})  # bad group
    # And it composes with the push decode path (maybe_dequantize).
    with pytest.raises(ValueError):
        maybe_dequantize([{**good, "shape": [128], "s": torch.zeros(99)}])
    # A valid int4 payload but a shape whose product != element count would
    # reshape-crash (uncaught RuntimeError); refuse as ValueError instead.
    with pytest.raises(ValueError):
        maybe_dequantize([{**good, "shape": [7, 7]}])           # 49 != 128
    # Same for an int4 down-delta whose element count != the base tensor.
    delta = encode_state_delta({"w": torch.randn(2, 130)}, {"w": torch.zeros(2, 130)},
                               mode="int4")
    with pytest.raises(ValueError):
        apply_state_delta({"w": torch.zeros(4, 4)}, delta)      # base 16 != delta 260
    # And an int8 down-delta whose tensor shape != the base.
    with pytest.raises(ValueError):
        apply_state_delta({"w": torch.zeros(4, 4)},
                          encode_state_delta({"w": torch.randn(2, 8)},
                                             {"w": torch.zeros(2, 8)}))


def test_run_local_sharded_trains_with_int4():
    """A full sharded cluster with compress="int4" trains to budget -- int4
    pseudo-gradients/deltas still carry the signal."""
    from opendipaco.launch import run_local
    from opendipaco.launch.config import LaunchConfig

    cfg = LaunchConfig.from_dict({
        "mode": "sharded",
        "model": {"vocab_size": 64, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "max_position_embeddings": 64,
                  "layers_per_level": [1, 1], "level_sizes": [2, 2], "sequence_length": 16},
        "diloco": {"inner_steps": 4, "inner_lr": 1e-3},
        "data": {"source": "synthetic", "num_documents": 64},
        "transport": {"compress": "int4", "down": "delta", "up_density": 0.5},
        "sharded": {"num_shards": 2, "parameter_servers": [["127.0.0.1", 0], ["127.0.0.1", 0]]},
        "run": {"generations": 2, "batch_size": 8, "local_workers": 2},
    })
    server, completed = run_local(cfg)
    assert sum(completed.values()) >= 2 * cfg.model.level_sizes[0] * cfg.model.level_sizes[1]
    assert server.metrics.accepted_updates > 0
