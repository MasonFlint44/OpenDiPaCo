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
    with pytest.raises(TypeError):
        maybe_dequantize([{"sp": [2, 2], "i": torch.tensor([0]), "v": "not a tensor"}])


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
