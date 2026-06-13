"""Wire compression for the DiLoCo traffic (internet-scale plan §0c / finding 1.2).

The transport ships everything as fp32, which doesn't fit consumer uplinks: a
150M-param path is ~600 MB each way per generation. This module compresses the
three heavy payloads:

* **Weights down** (coordinator/PS -> worker): floating tensors cast to
  **bfloat16** (2x). The receiver needs no decode step — ``load_state_dict``
  casts back into the fp32 modules on load. Integer buffers are untouched.
* **Pseudo-gradients up** (worker -> coordinator/PS): **int8** per-tensor
  symmetric quantization (~4x) with **error feedback**: the worker carries the
  quantization residual per (path, module) and adds it into the next
  generation's delta, so quantization error accumulates into later updates
  instead of being lost. ``"bf16"`` mode casts instead (2x, residual likewise
  carried).
* **Shards down**: token ids cast int64 -> int32 (2x; ids are vocab-bounded),
  restored with ``.long()`` on receipt.

The mode is **server policy**: the coordinator/scheduler stamps ``compress`` on
each task and workers follow it. Payloads are nevertheless *self-describing* —
a quantized tensor travels as ``{"q": int8, "s": scale}`` which
:func:`maybe_dequantize` detects — so receivers accept any mode without
configuration, and ``"none"`` keeps today's bytes bit-identical.

Compression changes training numerics (the worker trains from bf16-rounded
weights; the outer step sees quantized deltas). Mechanics are tested here;
convergence impact belongs to the plan's §0f WAN validation.
"""

from __future__ import annotations

import hashlib
import math
import struct

import torch

MODES = ("none", "bf16", "int8")


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"compress must be one of {MODES}, got {mode!r}")
    return mode


# -- weights (state dicts) -----------------------------------------------------


def compress_state(sd: dict, mode: str) -> dict:
    """Cast a state_dict's floating tensors to bf16 (any non-"none" mode).

    Weights stay bf16 even in "int8" mode — int8 is for *deltas*, whose scale a
    symmetric quantizer captures well; quantizing raw weights to int8 is far
    lossier. ``load_state_dict`` on the receiver casts back to the module dtype.
    """
    if mode == "none":
        return sd
    return {n: (v.to(torch.bfloat16) if v.is_floating_point() else v)
            for n, v in sd.items()}


# -- shards (token ids) ----------------------------------------------------------


def compress_shard(shard, mode: str):
    if shard is None or mode == "none":
        return shard
    return shard.to(torch.int32)  # vocab ids are far below 2^31


def restore_shard(shard):
    return None if shard is None else shard.long()


# -- pseudo-gradients ------------------------------------------------------------


def _quantize_int8(t: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Symmetric per-tensor int8: returns (``{"q", "s"}`` payload, residual)."""
    t = t.detach().float()
    absmax = float(t.abs().max()) if t.numel() else 0.0
    scale = absmax / 127.0
    if scale == 0.0:
        q = torch.zeros(t.shape, dtype=torch.int8)
        return {"q": q, "s": 0.0}, t.clone()
    q = torch.round(t / scale).clamp_(-127, 127).to(torch.int8)
    return {"q": q, "s": scale}, t - q.float() * scale


def pseudograd_digest(shared_delta: dict) -> str:
    """A stable hash of a contribution's shared pseudo-gradients, for redundant-
    execution agreement (Phase 3c).

    Two workers training the same (path, generation, shard) from the same base
    weights produce the *same* update; the digest lets the scheduler check that
    cheaply without holding the tensors. Quantizing each tensor to symmetric
    int8 before hashing collapses the low-order fp differences that
    nondeterministic kernels introduce across heterogeneous hardware, so honest
    replicas agree while a materially different (e.g. fabricated or sign-flipped)
    update lands on a different digest. Keys are hashed in sorted order so the
    digest is independent of dict iteration order.
    """
    h = hashlib.sha256()
    for key in sorted(shared_delta):
        h.update(key.encode("utf-8"))
        for t in shared_delta[key]:
            payload, _ = _quantize_int8(t)
            q, s = payload["q"], payload["s"]
            # Bucket the scale coarsely: tiny absmax jitter shouldn't flip the
            # digest, but an order-of-magnitude difference should.
            h.update(struct.pack(">q", round(math.log(s) * 1e3) if s > 0 else 0))
            h.update(q.numpy().tobytes())
    return h.hexdigest()


def state_digest(state: dict) -> str:
    """A stable hash of a module's weights (Phase 4c: owner cross-checking).

    The decentralized swarm has no trusted owner, so a reader cross-checks a
    key's ``(version, digest)`` across its ``k`` replicas and trusts only the
    bytes a majority agrees on (quorum reads), and co-owners flag a replica
    whose digest at a confirmed version diverges. Like
    :func:`pseudograd_digest`, each tensor is symmetric-int8-quantized before
    hashing so honest replicas that *recomputed* an aggregate (rather than
    copied exact bytes) agree despite low-order fp noise, while materially
    different weights land on a different digest. Keys hash in sorted order.
    """
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode("utf-8"))
        t = state[name]
        if not torch.is_floating_point(t):
            h.update(t.cpu().numpy().tobytes())  # int buffers hash exactly
            continue
        payload, _ = _quantize_int8(t)
        q, s = payload["q"], payload["s"]
        h.update(struct.pack(">q", round(math.log(s) * 1e3) if s > 0 else 0))
        h.update(q.numpy().tobytes())
    return h.hexdigest()


def compress_delta(delta, mode: str, carry=None):
    """Compress one module's pseudo-gradient (a list of tensors).

    ``carry`` is the previous round's residual list for this (path, module) —
    error feedback adds it in before encoding. Returns ``(payload, residual)``;
    residual is ``None`` in "none" mode (nothing is lost, nothing to carry).
    """
    if mode == "none":
        return list(delta), None
    if carry is not None:
        delta = [d + c for d, c in zip(delta, carry)]
    payload, residual = [], []
    for d in delta:
        if mode == "bf16":
            c = d.detach().to(torch.bfloat16)
            payload.append(c)
            residual.append(d.detach().float() - c.float())
        else:  # int8
            q, r = _quantize_int8(d)
            payload.append(q)
            residual.append(r)
    return payload, residual


def maybe_dequantize(items) -> list[torch.Tensor]:
    """Restore a pseudo-gradient list to fp32 whatever encoding it arrived in.

    Accepts fp32 (passthrough), bf16 (cast), or ``{"q", "s"}`` int8 payloads;
    raises ``TypeError`` on anything else so the server can refuse a malformed
    push instead of crashing.
    """
    out = []
    for it in items:
        if torch.is_tensor(it):
            out.append(it if it.dtype == torch.float32 else it.to(torch.float32))
        elif isinstance(it, dict) and "q" in it and torch.is_tensor(it["q"]):
            out.append(it["q"].to(torch.float32) * float(it["s"]))
        else:
            raise TypeError(f"not a pseudo-gradient tensor: {type(it)}")
    return out
