"""Server-side sanity checks on worker-submitted updates.

The servers apply whatever an authenticated worker sends; this module is the
cheap half of defending that surface (internet-scale plan §0b / finding 1.1).
Consumer hardware produces garbage floats even without adversaries — an
overclocked GPU or faulty RAM can flip bits mid-training — and a single
non-finite value applied to the bank poisons it permanently. So:

* **Non-finite anything rejects the contribution outright** (pseudo-gradient,
  private weights, reported loss). There is no legitimate NaN/Inf update, so
  this check has no knob and is always on.
* **An optional L2 norm cap *clips* oversized pseudo-gradients** instead of
  rejecting them (``max_update_norm=`` on the coordinator / parameter server),
  bounding any single contribution's influence on the bank. Off by default:
  a useful cap depends on model scale and DiLoCo hyper-parameters.

These checks bound *damage*, not *malice*: a worker can still submit a finite,
norm-bounded gradient pointing the wrong way. Robust aggregation and redundant
execution (plan §Phase 3) are the adversarial defense; this is the floor.
"""

from __future__ import annotations

import math

import torch


def all_finite(obj) -> bool:
    """True if every tensor anywhere in a nested dict/list/tuple structure is finite.

    Handles both wire shapes: ``{key: [tensor, ...]}`` (pseudo-gradients) and
    ``{key: state_dict}`` (private modules). Non-tensor leaves are ignored.
    """
    if torch.is_tensor(obj):
        return bool(torch.isfinite(obj).all())
    if isinstance(obj, dict):
        return all(all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_finite(v) for v in obj)
    return True


def loss_ok(loss, *, empty: bool = False) -> bool:
    """A reported inner loss must be finite — unless the shard was empty, where
    NaN is the documented no-op convention (`Contribution.empty`)."""
    if empty or loss is None:
        return True
    try:
        return math.isfinite(float(loss))
    except (TypeError, ValueError):
        return False


def contribution_ok(loss, empty, *tensor_structures) -> bool:
    """One verdict for a whole contribution: sane loss and all tensors finite."""
    return loss_ok(loss, empty=bool(empty)) and all(
        all_finite(s) for s in tensor_structures
    )


def clip_norm_(tensors, max_norm: float) -> float:
    """Scale ``tensors`` in place so their joint L2 norm is at most ``max_norm``.

    Returns the pre-clip norm (compare against ``max_norm`` to detect clipping).
    Mirrors ``torch.nn.utils.clip_grad_norm_`` semantics over a plain list.
    """
    total = math.sqrt(sum(float(t.detach().pow(2).sum()) for t in tensors))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for t in tensors:
            t.detach().mul_(scale)
    return total
