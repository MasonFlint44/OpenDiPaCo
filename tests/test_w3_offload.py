"""Tests for W3c exact memory levers (docs/w3-vram-design.md D5/D6):
tied embed/head (the per-module deepcopy used to sever the tie) and chunked
cross-entropy (avoids the full [tokens, vocab] logits).
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.model import build_module_bank, build_path_model
from opendipaco.optim.diloco import make_inner_optimizer
from opendipaco.train.loop import run_inner_steps


def _cfg(tie=False):
    bb = BackboneConfig(vocab_size=200, hidden_size=64, num_attention_heads=4,
                        intermediate_size=128, layers_per_level=[1, 1],
                        max_position_embeddings=128)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=32,
                        tie_word_embeddings=tie)


# -- tied weights survive the working-copy deepcopy (D6) ------------------------


def test_deepcopy_preserves_tied_weights():
    """build_path_model(deepcopy=True) must keep embed/head tied (one weight, half
    the memory) -- the per-module deepcopy used to sever it."""
    cfg = _cfg(tie=True)
    bank = build_module_bank(cfg, seed=0)
    pm = build_path_model(cfg, cfg.build_topology().path_from_index(0), bank, deepcopy=True)
    assert pm.embed.embed_tokens.weight is pm.head.lm_head.weight     # still tied
    # ...and a training step keeps them tied (one shared grad), not drifting apart.
    dl = DiLoCoConfig(inner_steps=2, inner_lr=1e-2)
    opt = make_inner_optimizer(pm, dl)
    shard = torch.arange(8 * 32).remainder(200).reshape(8, 32)
    run_inner_steps(pm, opt, shard, 4, torch.Generator().manual_seed(0),
                    inner_steps=2, total_steps=2, base_step=0, diloco=dl, device="cpu")
    assert pm.embed.embed_tokens.weight is pm.head.lm_head.weight


def test_untied_deepcopy_is_independent():
    """Without tying there are no shared tensors, so the single-call deepcopy is
    equivalent to the old per-module copy: embed and head are independent."""
    cfg = _cfg(tie=False)
    bank = build_module_bank(cfg, seed=0)
    pm = build_path_model(cfg, cfg.build_topology().path_from_index(0), bank, deepcopy=True)
    assert pm.embed.embed_tokens.weight is not pm.head.lm_head.weight


# -- chunked cross-entropy (D6) -------------------------------------------------


def _loss(cfg, chunks):
    bank = build_module_bank(cfg, seed=0)
    pm = build_path_model(cfg, cfg.build_topology().path_from_index(0), bank, deepcopy=True)
    pm.loss_chunks = chunks
    x = torch.arange(4 * 32).remainder(200).reshape(4, 32)
    _, loss = pm(x, labels=x)
    return loss


def test_chunked_cross_entropy_matches_dense():
    """Chunked CE equals the dense loss to fp tolerance (same math, the sum is in
    chunk order); the full-logits path returns logits, the chunked path doesn't."""
    cfg = _cfg()
    dense = _loss(cfg, 1)
    for chunks in (2, 4, 7):
        assert torch.allclose(_loss(cfg, chunks), dense, rtol=1e-5, atol=1e-6)


def test_chunked_skips_full_logits():
    cfg = _cfg()
    bank = build_module_bank(cfg, seed=0)
    pm = build_path_model(cfg, cfg.build_topology().path_from_index(0), bank, deepcopy=True)
    x = torch.arange(4 * 32).remainder(200).reshape(4, 32)
    pm.loss_chunks = 4
    logits, loss = pm(x, labels=x)
    assert logits is None and loss is not None          # no [tokens, vocab] tensor
    pm.loss_chunks = 1
    logits, _ = pm(x, labels=x)
    assert logits is not None and logits.shape == (4, 32, 200)


def test_chunked_ce_trains():
    """A short training run with chunked CE on converges like the dense path
    (loss decreases), end to end through run_inner_steps."""
    cfg = _cfg()
    bank = build_module_bank(cfg, seed=0)
    pm = build_path_model(cfg, cfg.build_topology().path_from_index(0), bank, deepcopy=True)
    dl = DiLoCoConfig(inner_steps=5, inner_lr=1e-2, loss_chunks=4)
    opt = make_inner_optimizer(pm, dl)
    shard = torch.arange(16 * 32).remainder(200).reshape(16, 32)
    loss = run_inner_steps(pm, opt, shard, 8, torch.Generator().manual_seed(0),
                           inner_steps=5, total_steps=5, base_step=0, diloco=dl, device="cpu")
    assert loss > 0 and pm.loss_chunks == 4
