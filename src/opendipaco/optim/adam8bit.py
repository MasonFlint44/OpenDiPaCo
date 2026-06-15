"""Blockwise 8-bit AdamW (W3d; docs/w3-vram-design.md D7).

The inner AdamW keeps two fp32 moments per parameter (``2P`` = 8 bytes/param) --
often the largest single VRAM term during a worker round. This stores them in
**blockwise int8** (~2 bytes/param, a ~4x cut): each moment is split into blocks
of ``block_size`` elements, each quantized symmetrically by its own absmax. The
moments are dequantized to fp32 for the step (so the parameter update is full-
precision) and requantized for storage; the quantization error enters only via
the carried-forward moment.

This is **lossy** -- a dynamics change, off by default, validated on-box
(``examples/validate_dynamics.py``) with the WAN §0f run as the final verdict,
exactly like W2's compression. A portable, CPU-testable alternative to a CUDA-
only ``bitsandbytes`` dependency; bitsandbytes is the drop-in for production
CUDA throughput.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

_BLOCK = 256
# Max elements dequantized to fp32 at once in step() -- bounds the transient
# moment buffers so a huge embed/head doesn't materialize a full fp32 copy
# (Codex review). Small params (< this) are one chunk, so no extra overhead.
_MAX_CHUNK_ELEMS = 1 << 22


def _to_blocks(t: torch.Tensor, block_size: int):
    flat = t.reshape(-1)
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.reshape(-1, block_size), n


def _quantize_blockwise(t: torch.Tensor, block_size: int):
    """Symmetric int8 per block (for the first moment ``m``, which is signed and
    may be ~0): returns (``q`` [nblocks, block_size] int8, ``absmax`` [nblocks])."""
    blocks, _ = _to_blocks(t, block_size)
    absmax = blocks.abs().amax(dim=1)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax)).unsqueeze(1)
    q = torch.round(blocks / scale).clamp_(-127, 127).to(torch.int8)
    return q, absmax


def _dequantize_blockwise(q: torch.Tensor, absmax: torch.Tensor, numel: int,
                          shape) -> torch.Tensor:
    scale = (absmax / 127.0).unsqueeze(1)
    return (q.to(torch.float32) * scale).reshape(-1)[:numel].reshape(shape)


# The second moment v >= 0 spans orders of magnitude, and a *linear* int8 zeros
# small values in a mixed-magnitude block -> denom = sqrt(v)+eps collapses to eps
# -> exploding step. Quantize it in the **log domain** (relative precision) so the
# smallest value in a block maps to a level, never to zero.
_V_FLOOR = 1e-20


def _quantize_v(v: torch.Tensor, block_size: int):
    """Log-domain uint8 per block: returns (q uint8, lo [nblocks], scale [nblocks])."""
    blocks, _ = _to_blocks(v, block_size)
    lv = torch.log(blocks.clamp_min(_V_FLOOR))
    lo = lv.amin(dim=1, keepdim=True)
    scale = ((lv.amax(dim=1, keepdim=True) - lo) / 255.0).clamp_min(1e-12)
    q = torch.round((lv - lo) / scale).clamp_(0, 255).to(torch.uint8)
    return q, lo.squeeze(1), scale.squeeze(1)


def _dequantize_v(q: torch.Tensor, lo: torch.Tensor, scale: torch.Tensor, numel: int,
                  shape) -> torch.Tensor:
    lv = q.to(torch.float32) * scale.unsqueeze(1) + lo.unsqueeze(1)
    return torch.exp(lv).reshape(-1)[:numel].reshape(shape)


class Adam8bit(Optimizer):
    """AdamW whose moments are stored in blockwise int8 (W3d). The math matches
    ``torch.optim.AdamW`` (decoupled weight decay, bias correction); only the
    moment *storage* is quantized."""

    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, block_size: int = _BLOCK):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, block_size=block_size))

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict casts non-step state to the param's dtype
        # (fp32), which would turn the quantized moments back into fp32 -- losing
        # the int8 memory win at the first post-resume step (the warm-task peak).
        # Re-cast them to their integer dtypes; the values are integer-valued, so
        # the fp32 round-trip is lossless.
        super().load_state_dict(state_dict)
        for st in self.state.values():
            if "m_q" in st:
                st["m_q"] = st["m_q"].to(torch.int8)
                st["v_q"] = st["v_q"].to(torch.uint8)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd, bs = group["lr"], group["eps"], group["weight_decay"], group["block_size"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    nblocks = (p.numel() + bs - 1) // bs
                    state["step"] = 0
                    state["m_q"] = torch.zeros((nblocks, bs), dtype=torch.int8, device=p.device)
                    state["m_absmax"] = torch.zeros(nblocks, device=p.device)
                    state["v_q"] = torch.zeros((nblocks, bs), dtype=torch.uint8, device=p.device)
                    state["v_lo"] = torch.zeros(nblocks, device=p.device)
                    state["v_scale"] = torch.zeros(nblocks, device=p.device)
                first = state["step"] == 0
                state["step"] += 1
                t = state["step"]
                bc1, bc2_sqrt = 1 - b1 ** t, (1 - b2 ** t) ** 0.5
                self._step_param(p, state, b1, b2, lr, eps, wd, bs, bc1, bc2_sqrt, first)
        return loss

    @staticmethod
    def _step_param(p, state, b1, b2, lr, eps, wd, bs, bc1, bc2_sqrt, first):
        """Update one parameter in **block-chunks** so the dequantized fp32
        moments (m, v, denom) are only ever a bounded transient -- never a full-
        size copy of the (possibly huge) embed/head this lever is meant to fit.
        The per-element AdamW is independent, so chunking is bit-identical to
        processing the whole tensor at once."""
        g, pflat = p.grad.reshape(-1), p.reshape(-1)
        nblocks = state["m_q"].shape[0]
        per_chunk = max(1, _MAX_CHUNK_ELEMS // bs)       # whole blocks per chunk
        n = p.numel()
        for c0 in range(0, nblocks, per_chunk):
            c1 = min(c0 + per_chunk, nblocks)
            e0, e1 = c0 * bs, min(c1 * bs, n)            # real elements in this chunk
            gc = g[e0:e1]
            if gc.numel() < (c1 - c0) * bs:              # pad the last block's tail
                gc = torch.cat([gc, gc.new_zeros((c1 - c0) * bs - gc.numel())])
            m = _dequantize_blockwise(state["m_q"][c0:c1], state["m_absmax"][c0:c1],
                                      (c1 - c0) * bs, (-1,))
            v = (torch.zeros_like(gc) if first else
                 _dequantize_v(state["v_q"][c0:c1], state["v_lo"][c0:c1],
                               state["v_scale"][c0:c1], (c1 - c0) * bs, (-1,)))
            m.mul_(b1).add_(gc, alpha=1 - b1)
            v.mul_(b2).addcmul_(gc, gc, value=1 - b2)
            state["m_q"][c0:c1], state["m_absmax"][c0:c1] = _quantize_blockwise(m, bs)
            state["v_q"][c0:c1], state["v_lo"][c0:c1], state["v_scale"][c0:c1] = _quantize_v(v, bs)
            real = e1 - e0
            denom = (v[:real].sqrt() / bc2_sqrt).add_(eps)
            pflat[e0:e1].mul_(1 - lr * wd).addcdiv_(m[:real], denom, value=-lr / bc1)
