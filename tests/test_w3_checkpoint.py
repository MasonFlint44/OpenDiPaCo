"""Tests for W3b activation checkpointing (docs/w3-vram-design.md D3).

Checkpointing recomputes body-block activations in backward instead of storing
them -- a memory cut that is **bit-exact** (only changes what is stored). These
verify the exactness, the flag wiring, and that it's inert outside training.
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.model import build_module_bank, build_path_model
from opendipaco.optim.diloco import make_inner_optimizer
from opendipaco.train.loop import run_inner_steps


def _cfg():
    bb = BackboneConfig(vocab_size=200, hidden_size=64, num_attention_heads=4,
                        intermediate_size=128, layers_per_level=[2, 2],
                        max_position_embeddings=128)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=32)


def _train(cfg, checkpoint, autocast=False):
    path = cfg.build_topology().path_from_index(0)
    bank = build_module_bank(cfg, seed=0)
    pm = build_path_model(cfg, path, bank, deepcopy=True)
    dl = DiLoCoConfig(inner_steps=3, inner_lr=1e-3, inner_autocast=autocast,
                      activation_checkpoint=checkpoint)
    opt = make_inner_optimizer(pm, dl)
    shard = torch.arange(16 * 32).remainder(200).reshape(16, 32)
    g = torch.Generator().manual_seed(0)
    run_inner_steps(pm, opt, shard, 4, g, inner_steps=3, total_steps=3, base_step=0,
                    diloco=dl, device="cpu")
    return [p.detach().clone() for p in pm.parameters()], pm


def test_checkpointing_is_bit_exact():
    """Training with checkpointing on vs off yields bit-identical weights -- it
    changes memory, not the math (use_reentrant=False preserves RNG/autocast)."""
    cfg = _cfg()
    off, _ = _train(cfg, checkpoint=False)
    on, pm = _train(cfg, checkpoint=True)
    assert all(torch.equal(a, b) for a, b in zip(off, on))
    assert pm.checkpoint is True                        # flag set from diloco during training


def test_checkpointing_composes_with_autocast():
    """The recompute runs under the same bf16 autocast context (use_reentrant=
    False), so checkpointing + inner_autocast is still bit-exact."""
    cfg = _cfg()
    off, _ = _train(cfg, checkpoint=False, autocast=True)
    on, _ = _train(cfg, checkpoint=True, autocast=True)
    assert all(torch.equal(a, b) for a, b in zip(off, on))


def test_checkpoint_inert_outside_training():
    """A PathModel checkpoints only when training with grad on; encode()/no_grad
    must not attempt to checkpoint (it would be a wasteful no-op or error)."""
    cfg = _cfg()
    bank = build_module_bank(cfg, seed=0)
    pm = build_path_model(cfg, cfg.build_topology().path_from_index(0), bank, deepcopy=False)
    pm.checkpoint = True
    pm.eval()
    x = torch.randint(0, 200, (2, 32))
    with torch.no_grad():
        logits, _ = pm(x)                               # runs the direct (non-checkpoint) path
    assert logits.shape == (2, 32, 200)


def test_launch_default_enables_checkpointing():
    """Real runs default checkpointing on (it's exact); the core DiLoCoConfig
    default stays off so the in-process anchor/tests are fast + byte-identical."""
    from opendipaco.launch.config import DiLoCoCfg, diloco_config

    assert diloco_config(DiLoCoCfg()).activation_checkpoint is True
    assert DiLoCoConfig().activation_checkpoint is False
