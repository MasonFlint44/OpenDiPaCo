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

MODES = ("none", "bf16", "int8", "int4")
_INT4_GROUP = 128   # elements per int4 scale (per-tensor would be too lossy at 4 bits)


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


def _quantize_int4(t: torch.Tensor, group_size: int = _INT4_GROUP):
    """Symmetric int4 with a **per-group scale** (W2c/D7): values in [-7, 7], one
    scale per ``group_size`` elements (a single per-tensor scale is far too lossy
    at 4 bits), two nibbles packed per byte. Returns
    (``{"q4","s","g","n","shape"}`` payload, residual). Stays on ``t``'s device."""
    flat = t.detach().reshape(-1).float()
    n = flat.numel()
    pad = (-n) % group_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    grp = flat.reshape(-1, group_size)
    scale = grp.abs().amax(dim=1, keepdim=True) / 7.0
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(grp / safe).clamp_(-7, 7)
    recon = (q * scale).reshape(-1)[:n].reshape(t.shape)
    qu = (q + 8).to(torch.uint8).reshape(-1)            # [-7,7] -> [1,15], fits a nibble
    packed = ((qu[0::2] << 4) | qu[1::2]).contiguous()  # two values per byte (g even)
    payload = {"q4": packed, "s": scale.reshape(-1).to(torch.float32),
               "g": group_size, "n": n, "shape": list(t.shape)}
    return payload, t.detach().float() - recon


def _dequant_int4(p: dict) -> torch.Tensor:
    """Inverse of :func:`_quantize_int4`; returns a flat fp32 tensor of length
    ``n``. Validates lengths so a malformed payload raises ``ValueError`` (a
    caught, rejected push) instead of an uncaught reshape/index crash."""
    packed, scale = p.get("q4"), p.get("s")
    g, n = int(p.get("g", 0)), int(p.get("n", -1))
    if not (torch.is_tensor(packed) and torch.is_tensor(scale)) or g <= 0 or n < 0:
        raise ValueError("malformed int4 payload")
    ngroups = (n + g - 1) // g
    if scale.numel() != ngroups or packed.numel() != ngroups * g // 2:
        raise ValueError("malformed int4 payload: length mismatch")
    pk = packed.to(torch.int16)
    qu = torch.stack([(pk >> 4) & 0xF, pk & 0xF], dim=1).reshape(-1)
    q = (qu.to(torch.float32) - 8.0).reshape(-1, g)
    return (q * scale.reshape(-1, 1).to(torch.float32)).reshape(-1)[:n]


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


def _sparsify(d: torch.Tensor, density: float, mode: str):
    """Structured top-k of one tensor (W2b): keep the largest-|·| ``density``
    fraction -- **per output-row** for a 2-D weight (so each row keeps its own
    top entries), flat otherwise -- encode the kept values via ``mode``, and
    return ``(payload, dense_reconstruction)``. The dropped entries are carried by
    the caller as error-feedback (``original − reconstruction`` is the residual).
    The payload is self-describing: ``{"sp": shape, "i": flat int64 indices,
    "v": <encoded kept values>}``."""
    d = d.detach()
    # Stay on d's device throughout (a worker's pseudo-gradient lives on its
    # training device, often GPU); arange/zeros default to CPU, which would
    # device-mismatch the GPU topk indices. The payload rides the wire from
    # whatever device, exactly like the int8 path.
    if d.dim() == 2:
        rows, cols = d.shape
        k = max(1, math.ceil(density * cols))
        idx = d.abs().topk(k, dim=1).indices                 # [rows, k] per-row
        kept = torch.gather(d, 1, idx).reshape(-1)
        rid = torch.arange(rows, device=d.device).unsqueeze(1)
        flat_idx = (rid * cols + idx).reshape(-1)
    else:
        n = d.numel()
        k = max(1, math.ceil(density * n))
        flat = d.reshape(-1)
        flat_idx = flat.abs().topk(k).indices
        kept = flat[flat_idx]
    if mode == "int8":
        enc, _ = _quantize_int8(kept)
        recon_vals = enc["q"].float() * enc["s"]
    elif mode == "int4":
        enc, _ = _quantize_int4(kept)
        recon_vals = _dequant_int4(enc)
    elif mode == "bf16":
        enc = kept.to(torch.bfloat16)
        recon_vals = enc.float()
    else:  # "none": keep fp32 values, sparsify only
        enc = kept.float()
        recon_vals = enc
    dense = torch.zeros(d.numel(), dtype=torch.float32, device=d.device)
    dense[flat_idx] = recon_vals
    # int32 indices halve the index wire cost vs int64 (a single module tensor is
    # always far under 2^31 elements) -- the index is the dominant overhead, so
    # this roughly doubles the density at which sparsification beats a dense ship.
    return {"sp": list(d.shape), "i": flat_idx.to(torch.int32), "v": enc}, dense.reshape(d.shape)


def compress_delta(delta, mode: str, carry=None, density: float = 1.0):
    """Compress one module's pseudo-gradient (a list of tensors).

    ``carry`` is the previous round's residual list for this (path, module) —
    error feedback adds it in before encoding. ``density`` < 1.0 additionally
    **sparsifies** each tensor to its top-k (W2b), error-feeding the dropped mass
    via the same residual. Returns ``(payload, residual)``; residual is ``None``
    only in "none" mode with no sparsification (nothing is lost, nothing to carry).
    """
    if mode == "none" and density >= 1.0:
        return list(delta), None
    if carry is not None:
        delta = [d + c for d, c in zip(delta, carry)]
    payload, residual = [], []
    for d in delta:
        if density < 1.0:
            p, recon = _sparsify(d, density, mode)
            payload.append(p)
            residual.append(d.detach().float() - recon)   # dropped mass + kept quant error
        elif mode == "bf16":
            c = d.detach().to(torch.bfloat16)
            payload.append(c)
            residual.append(d.detach().float() - c.float())
        elif mode == "int4":
            p, r = _quantize_int4(d)
            payload.append(p)
            residual.append(r)
        else:  # int8
            q, r = _quantize_int8(d)
            payload.append(q)
            residual.append(r)
    return payload, residual


# -- weight deltas (W2a delta-down; docs/w2-bandwidth-design.md) -----------------


def encode_state_delta(cur: dict, base: dict, mode: str = "int8") -> dict:
    """Encode a state_dict as a **delta against ``base``** (a prior version the
    receiver holds exactly -- a keyframe). Floating tensors become a small-
    magnitude ``cur - base`` quantized to ``mode`` (int8, or int4 per-group for
    W2c) -- deltas have far smaller range than raw weights, so low-bit captures
    them at pseudo-gradient quality (raw weights would need bf16). Non-floating
    (or shape-changed) tensors ship verbatim. ``base`` is the *same bytes the
    receiver holds* (the caller passes ``compress_state``-d history), so the only
    reconstruction error is one bounded quant step."""
    out = {}
    for n, c in cur.items():
        b = base.get(n)
        if c.is_floating_point() and b is not None and b.shape == c.shape:
            d = c.float() - b.float()
            out[n] = _quantize_int4(d)[0] if mode == "int4" else _quantize_int8(d)[0]
        else:
            out[n] = c                      # non-float / shape change -> verbatim
    return out


def apply_state_delta(base: dict, tensors: dict) -> dict:
    """Reconstruct a state_dict from a keyframe ``base`` + an
    :func:`encode_state_delta` payload: ``base + dequant(delta)`` for the int8 /
    int4 entries, verbatim for the rest. Raises ``TypeError``/``ValueError`` on a
    malformed entry so a receiver refuses a bad delta instead of crashing."""
    out = {}
    for n, t in tensors.items():
        if isinstance(t, dict) and "q" in t and torch.is_tensor(t["q"]):
            b = base[n]
            if t["q"].shape != b.shape:   # a mismatched delta would broadcast-crash
                raise ValueError(f"int8 delta shape {tuple(t['q'].shape)} != base {tuple(b.shape)}")
            out[n] = b.float() + t["q"].to(torch.float32) * float(t["s"])
        elif isinstance(t, dict) and "q4" in t:
            b = base[n]
            deq = _dequant_int4(t)
            if deq.numel() != b.numel():   # a mismatched delta would reshape-crash
                raise ValueError(f"int4 delta has {deq.numel()} elems != base {b.numel()}")
            out[n] = b.float() + deq.reshape(b.shape)
        elif torch.is_tensor(t):
            out[n] = t                      # verbatim (non-float / shape change)
        else:
            raise TypeError(f"not a state-delta entry: {type(t)}")
    return out


def maybe_dequantize(items) -> list[torch.Tensor]:
    """Restore a pseudo-gradient list to fp32 whatever encoding it arrived in.

    Accepts fp32 (passthrough), bf16 (cast), ``{"q", "s"}`` int8 payloads,
    ``{"q4", ...}`` int4 per-group payloads (W2c), or a ``{"sp", "i", "v"}``
    sparse top-k payload (W2b, scattered back to a dense tensor); raises
    ``TypeError``/``ValueError`` on anything else so the server can refuse a
    malformed push instead of crashing.
    """
    out = []
    for it in items:
        if torch.is_tensor(it):
            out.append(it if it.dtype == torch.float32 else it.to(torch.float32))
        elif isinstance(it, dict) and "q" in it and torch.is_tensor(it["q"]):
            out.append(it["q"].to(torch.float32) * float(it["s"]))
        elif isinstance(it, dict) and "q4" in it:       # int4 per-group (dense)
            deq = _dequant_int4(it)
            shape = [int(s) for s in it["shape"]]
            if any(s < 0 for s in shape) or math.prod(shape) != deq.numel():
                raise ValueError("int4 payload: shape does not match element count")
            out.append(deq.reshape(shape))
        elif isinstance(it, dict) and "sp" in it and torch.is_tensor(it.get("i")):
            shape = [int(s) for s in it["sp"]]
            v = it["v"]
            if isinstance(v, dict) and "q" in v:        # int8-encoded kept values
                vals = v["q"].to(torch.float32) * float(v["s"])
            elif isinstance(v, dict) and "q4" in v:     # int4-encoded kept values
                vals = _dequant_int4(v)
            elif torch.is_tensor(v):                    # fp32 / bf16 kept values
                vals = v.to(torch.float32)
            else:
                raise TypeError(f"not a sparse-payload value: {type(v)}")
            # Validate before scattering: a Byzantine peer (valid grant) could craft
            # out-of-bounds / mismatched indices, whose scatter would raise an
            # *uncaught* IndexError/RuntimeError and crash the handler. Refuse as a
            # ValueError (which the push path treats as an invalid push).
            if any(s < 0 for s in shape):
                raise ValueError("sparse payload: negative shape dim")
            n = math.prod(shape)
            idx, vals = it["i"].to(torch.int64).reshape(-1), vals.reshape(-1)
            if idx.numel() != vals.numel():
                raise ValueError("sparse payload: index/value length mismatch")
            if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n):
                raise ValueError("sparse payload: index out of range")
            dense = torch.zeros(n, dtype=torch.float32)
            dense[idx] = vals
            out.append(dense.reshape(shape))
        else:
            raise TypeError(f"not a pseudo-gradient tensor: {type(it)}")
    return out
