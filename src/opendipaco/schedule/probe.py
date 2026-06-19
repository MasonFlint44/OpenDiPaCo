"""Trusted-probe screening for data-poisoning defense (W8 part 1; design
``docs/w8-data-poisoning-design.md``).

Phase 3c's redundant-execution audit agrees on a *digest of the pseudo-gradient*,
which catches a worker whose computation diverges but not adversarial data
*content*: every honest checker handed the same poisoned shard reproduces the
same harmful gradient, so the digests agree and the update is applied. A poisoned
update that passed every finite/norm/agreement check still **raises loss on a
clean held-out set** -- so the audit checker (which already composes and trains
the full path) measures the update's effect on a small **trusted probe** and
reports it; a contribution that harms the probe is rejected.

This module is the primitive: :func:`probe_loss` (worker/checker side, the loss
of a composed path model on the probe) and :func:`is_harmful` (owner side, the
verdict from the before/after losses the checker reports). :class:`TrustedProbe`
carries the clean batch. Everything here is inert unless a probe is configured.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def probe_loss(model: nn.Module, sequences: torch.Tensor, *, batch_size: int = 8) -> float:
    """Mean token-level next-token cross-entropy of ``model`` over ``sequences``.

    Matches the *training* loss convention (``DiPaCoEngine._eval_val``): the whole
    sequence is scored (``logits[:, :-1]`` vs ``[:, 1:]``, no prefix exclusion), so
    the value is directly comparable to what the path was trained to minimise.
    Restores the model's prior train/eval mode. Returns NaN for an empty probe (no
    scorable tokens) -- the caller treats "couldn't measure" as "can't screen".
    """
    if sequences is None or sequences.numel() == 0:
        return float("nan")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, tokens = 0.0, 0
    try:
        for start in range(0, sequences.size(0), batch_size):
            batch = sequences[start : start + batch_size].to(device)
            if batch.size(1) < 2:
                continue  # need >= 2 tokens to form one (prediction, target) pair
            logits, _ = model(batch)
            pred, tgt = logits[:, :-1, :], batch[:, 1:]
            total += float(
                F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="sum")
            )
            tokens += tgt.numel()
    finally:
        if was_training:
            model.train()
    return total / tokens if tokens else float("nan")


def is_harmful(before: float, after: float, *, abs_margin: float = 0.05,
               rel_margin: float = 0.02) -> bool:
    """Owner-side verdict: did the update **raise** trusted-probe loss beyond the
    margin? ``before``/``after`` are the probe losses the checker reports for the
    pinned base and the reproduced (trained-local) model.

    Harmful when ``after - before > abs_margin + rel_margin * |before|`` -- a small
    absolute floor (the probe is small, one inner loop perturbs it a little either
    way) plus a relative term (loss scale varies across runs/levels). An honest
    update drives the delta <= ~0; a poisoned one drives it clearly positive. A
    non-finite ``before``/``after`` (empty/failed probe) is **not** harmful: we
    can't screen, so we don't block (coverage caveat in the design). The default
    margins are a starting point -- the screening threshold is tuned on the 0f run.
    """
    if not (math.isfinite(before) and math.isfinite(after)):
        return False
    return after - before > abs_margin + rel_margin * abs(before)


class TrustedProbe:
    """A small, clean, operator-curated held-out batch the checker screens
    contributions against (trust rides the W6 manifest pinning). Holds the token
    ``[N, seq_len]`` tensor + the eval batch size; ``loss(model)`` is the model's
    probe loss, ``empty`` is True when there is nothing to screen with."""

    def __init__(self, sequences: torch.Tensor | None, *, batch_size: int = 8):
        self.sequences = sequences
        self.batch_size = int(batch_size)

    @property
    def empty(self) -> bool:
        return self.sequences is None or self.sequences.numel() == 0

    def loss(self, model: nn.Module) -> float:
        return probe_loss(model, self.sequences, batch_size=self.batch_size)
