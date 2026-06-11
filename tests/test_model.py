import torch

from opendipaco.config import BackboneConfig, DiPaCoConfig
from opendipaco.model import build_module_bank, build_path_model


def tiny_config():
    bb = BackboneConfig(
        vocab_size=64, hidden_size=32, num_attention_heads=4,
        intermediate_size=64, layers_per_level=[1, 1], max_position_embeddings=32,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def test_forward_and_loss_shapes():
    cfg = tiny_config()
    bank = build_module_bank(cfg)
    pm = build_path_model(cfg, (1, 0), bank)
    ids = torch.randint(0, 64, (3, 16))
    logits, loss = pm(ids, labels=ids)
    assert logits.shape == (3, 16, 64)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_causal_no_future_leak():
    cfg = tiny_config()
    bank = build_module_bank(cfg)
    pm = build_path_model(cfg, (0, 0), bank).eval()
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        logits_a, _ = pm(ids)
        ids2 = ids.clone()
        ids2[0, -1] = (ids2[0, -1] + 1) % 64
        logits_b, _ = pm(ids2)
    # Changing the last token must not affect earlier-position logits.
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_modules_by_key_matches_path():
    cfg = tiny_config()
    bank = build_module_bank(cfg)
    pm = build_path_model(cfg, (1, 1), bank)
    keys = set(pm.modules_by_key())
    assert keys == {"embed", "head", "L0E1", "L1E1"}


def test_deepcopy_independent_from_bank():
    cfg = tiny_config()
    bank = build_module_bank(cfg)
    pm = build_path_model(cfg, (0, 0), bank, deepcopy=True)
    # Mutating the path model's expert must not change the bank's copy.
    with torch.no_grad():
        next(pm.modules_by_key()["L0E0"].parameters()).add_(1.0)
    bank_p = next(bank["L0E0"].parameters())
    path_p = next(pm.modules_by_key()["L0E0"].parameters())
    assert not torch.allclose(bank_p, path_p)
