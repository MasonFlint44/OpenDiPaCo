"""Tests for W3d lossy VRAM levers (docs/w3-vram-design.md D7/D4):
blockwise 8-bit AdamW moments and the private-copy de-dup / warming. Both off by
default, §0f-gated; here we check the mechanics + on-box convergence.
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.model import build_module_bank, build_path_model
from opendipaco.optim.adam8bit import Adam8bit, _dequantize_blockwise, _quantize_blockwise
from opendipaco.optim.diloco import make_inner_optimizer
from opendipaco.topology import is_private_key
from opendipaco.train.loop import run_inner_steps


def _cfg():
    bb = BackboneConfig(vocab_size=200, hidden_size=64, num_attention_heads=4,
                        intermediate_size=128, layers_per_level=[1, 1],
                        max_position_embeddings=128)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=32)


# -- blockwise 8-bit quantization ----------------------------------------------


def test_blockwise_quant_roundtrip_within_bound():
    torch.manual_seed(0)
    t = torch.randn(1000)
    q, absmax = _quantize_blockwise(t, 256)
    deq = _dequantize_blockwise(q, absmax, t.numel(), t.shape)
    assert deq.shape == t.shape
    # Per-block error <= half a quant step (absmax/127).
    nblocks = (t.numel() + 255) // 256
    assert q.shape == (nblocks, 256) and absmax.shape == (nblocks,)
    step = (absmax / 127.0).repeat_interleave(256)[: t.numel()]
    assert (deq - t).abs().le(step / 2 + 1e-6).all()


# -- Adam8bit ------------------------------------------------------------------


def test_adam8bit_tracks_fp32_adamw():
    """8-bit AdamW optimizes a quadratic close to fp32 AdamW (the moments are
    int8, the param update is full precision)."""
    torch.manual_seed(0)
    target = torch.randn(512)

    def run(opt_cls):
        x = torch.zeros(512, requires_grad=True)
        opt = opt_cls([x], lr=0.1)
        for _ in range(100):
            opt.zero_grad()
            ((x - target) ** 2).sum().backward()
            opt.step()
        return x.detach()

    ref = run(lambda p, lr: torch.optim.AdamW(p, lr=lr))
    q8 = run(lambda p, lr: Adam8bit(p, lr=lr))
    # Both converge toward the target; 8-bit stays close to fp32.
    assert (q8 - target).abs().mean() < 0.05
    assert (q8 - ref).abs().mean() < 0.02


def test_adam8bit_state_is_small_and_serializable():
    """Moments are stored int8 (~2 B/param vs 8) and survive the CPU offload
    round-trip the worker does between tasks."""
    x = torch.zeros(600, requires_grad=True)
    opt = Adam8bit([x], lr=0.1)
    ((x - 1) ** 2).sum().backward()
    opt.step()
    st = opt.state[x]
    assert st["m_q"].dtype == torch.int8 and st["v_q"].dtype == torch.uint8
    sd = opt.state_dict()                                # offload/checkpoint path
    opt2 = Adam8bit([x], lr=0.1)
    opt2.load_state_dict(sd)
    # int8 must survive the load (Optimizer.load_state_dict casts to the param's
    # fp32; the override re-casts so the memory win isn't lost on resume).
    assert opt2.state[x]["m_q"].dtype == torch.int8
    assert torch.equal(opt2.state[x]["m_q"], st["m_q"])


def test_adam8bit_survives_worker_offload_cycle():
    """The worker offloads optimizer state to CPU between tasks via
    _optimizer_state_to_cpu and resumes it into a fresh optimizer each task. The
    8-bit state must round-trip that exact cycle (stay int8/uint8, keep training)
    -- the plain state_dict test doesn't exercise the worker's helper."""
    from opendipaco.train.loop import _optimizer_state_to_cpu

    def step(x, opt):
        opt.zero_grad()
        ((x - 1) ** 2).sum().backward()
        opt.step()

    # No-offload reference: 8 straight steps.
    xs = torch.zeros(600, requires_grad=True)
    os_ = Adam8bit([xs], lr=0.1)
    for _ in range(8):
        step(xs, os_)

    # Offloaded: 3 steps, offload to CPU, resume into a fresh optimizer, 5 more.
    xr = torch.zeros(600, requires_grad=True)
    opt = Adam8bit([xr], lr=0.1)
    for _ in range(3):
        step(xr, opt)
    opt2 = Adam8bit([xr], lr=0.1)
    opt2.load_state_dict(_optimizer_state_to_cpu(opt.state_dict()))
    assert opt2.state[xr]["m_q"].dtype == torch.int8 and opt2.state[xr]["v_q"].dtype == torch.uint8
    assert int(opt2.state[xr]["step"]) == 3                 # step counter resumed
    for _ in range(5):
        step(xr, opt2)
    # Resume is exact: 3+offload+5 lands bit-identically on 8 straight steps.
    assert torch.equal(xr, xs)


def test_make_inner_optimizer_selects_8bit():
    cfg = _cfg()
    pm = build_path_model(cfg, cfg.build_topology().path_from_index(0),
                          build_module_bank(cfg, seed=0), deepcopy=True)
    assert isinstance(make_inner_optimizer(pm, DiLoCoConfig(optim_8bit=True)), Adam8bit)
    assert not isinstance(make_inner_optimizer(pm, DiLoCoConfig()), Adam8bit)


def test_optim_8bit_trains():
    cfg = _cfg()
    pm = build_path_model(cfg, cfg.build_topology().path_from_index(0),
                          build_module_bank(cfg, seed=0), deepcopy=True)
    dl = DiLoCoConfig(inner_steps=5, inner_lr=1e-2, optim_8bit=True)
    opt = make_inner_optimizer(pm, dl)
    shard = torch.arange(16 * 32).remainder(200).reshape(16, 32)
    loss = run_inner_steps(pm, opt, shard, 8, torch.Generator().manual_seed(0),
                           inner_steps=5, total_steps=5, base_step=0, diloco=dl, device="cpu")
    assert loss > 0


# -- private de-dup ------------------------------------------------------------


def test_dedup_private_aliases_private_copies_shared():
    """dedup_private aliases the private modules from the bank (the memory win)
    but still deep-copies the shared ones (needed for the global−local delta)."""
    cfg = _cfg()
    bank = build_module_bank(cfg, seed=0)
    path = cfg.build_topology().path_from_index(0)
    pm = build_path_model(cfg, path, bank, deepcopy=True, dedup_private=True)
    by_key = pm.modules_by_key()
    for k, mod in by_key.items():
        for p_pm, p_bank in zip(mod.parameters(), bank[k].parameters()):
            if is_private_key(k):
                assert p_pm is p_bank          # private aliased (no copy)
            else:
                assert p_pm is not p_bank      # shared deep-copied
