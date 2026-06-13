"""Robust aggregation of pseudo-gradient contributions (Phase 3a, finding 1.1).

When several sharing paths contribute a pseudo-gradient for the same module, a
plain sum lets a single Byzantine contributor poison the module. This combines a
buffer of contributions into one outer-step delta that tolerates a *minority* of
adversaries, by **decoupling direction from magnitude**:

* **direction** — a coordinate-wise robust combine of the raw pseudo-gradients:
  ``mean`` (nothing to trim), ``trimmed_mean`` (drop one extreme per coordinate
  per side, so a lone outlier can't move it), or ``median``;
* **magnitude** — the *sum* of the contributions' weights, so the applied step
  keeps the same scale as today's summed-across-sharing-paths outer step (the
  engine's batched semantics) rather than a per-contributor average.

With a single contribution the direction is that contribution and the magnitude
is its weight, so the result is exactly ``weight * grad`` — bit-identical to
applying the one push directly. That is what lets the ``robustness: off`` path
(which never calls this) and a degree-1 quorum stay identical to the Phase 2
anchor.

Why unweighted direction × summed weight, rather than aggregating the
*weighted* grads: with equal weights the two agree, but letting a single
contribution's (e.g. low-staleness) weight scale its influence on the
*direction* is exactly the leverage an adversary wants. Robustness keeps the
direction democratic and folds the weights into magnitude only.

Pure functions over plain tensor lists — no module/optimizer/lock state — so
they are cheap to unit-test and to reason about for the dynamics validation
(plan §1.4).
"""

from __future__ import annotations

import torch

AGGREGATES = ("mean", "trimmed_mean", "median")


def check_aggregate(name: str) -> str:
    if name not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {name!r}")
    return name


def _combine(stack: torch.Tensor, mode: str) -> torch.Tensor:
    """Coordinate-wise combine along dim 0 (the contribution axis)."""
    if mode == "median":
        return stack.median(dim=0).values
    if mode == "trimmed_mean" and stack.shape[0] >= 3:
        # Drop the single highest and lowest per coordinate, mean the rest: one
        # adversary can't be both extremes, so it never survives the trim.
        ordered, _ = stack.sort(dim=0)
        return ordered[1:-1].mean(dim=0)
    return stack.mean(dim=0)  # mean, or trimmed_mean with too few to trim


def robust_delta(contributions, *, aggregate: str = "trimmed_mean"):
    """Combine buffered contributions into one outer-step delta.

    ``contributions`` is a list of ``(weight, grad)``; ``grad`` is a list of
    tensors (one per module parameter), already norm-clipped by the caller if a
    cap is configured. Returns ``(delta, weight_sum)``: ``delta`` is the
    per-parameter direction and ``weight_sum`` the summed weight, so the caller
    applies ``[weight_sum * d for d in delta]`` as a single outer step. With one
    contribution this is exactly ``[weight * g for g in grad]``.
    """
    check_aggregate(aggregate)
    weight_sum = float(sum(w for w, _ in contributions))
    grads = [g for _, g in contributions]
    if len(grads) == 1:
        return list(grads[0]), weight_sum  # bit-identical passthrough
    delta = [
        _combine(torch.stack([g[p] for g in grads], dim=0), aggregate)
        for p in range(len(grads[0]))
    ]
    return delta, weight_sum
