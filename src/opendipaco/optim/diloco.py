"""DiLoCo inner/outer optimization primitives (clean re-implementation).

DiLoCo runs two nested optimizers:

* **inner** -- AdamW, taking ``H`` local steps on each worker's data shard.
* **outer** -- SGD with Nesterov momentum, applied to the shared (global) module
  weights using the averaged *pseudo-gradient* ``global - local``.

This module provides the small, reusable pieces; orchestration (rounds,
averaging across paths, redistribution) lives in :mod:`opendipaco.train`.

Reference: Douillard et al., "DiLoCo: Distributed Low-Communication Training of
Language Models", https://arxiv.org/abs/2311.08105
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import DiLoCoConfig


def inner_lr_at(step: int, total_steps: int | None, cfg: DiLoCoConfig) -> float:
    """Inner learning rate at global inner-step ``step`` (0-indexed).

    Linear warmup for ``inner_warmup_steps``, then (for the cosine schedule)
    cosine decay from ``inner_lr`` down to ``inner_lr * inner_min_lr_ratio`` over
    the rest of the run. Falls back to a constant LR when the schedule is
    "constant" or ``total_steps`` is unknown.
    """
    peak = cfg.inner_lr
    warm = cfg.inner_warmup_steps
    if warm > 0 and step < warm:
        return peak * (step + 1) / warm
    if cfg.inner_lr_schedule == "constant" or total_steps is None:
        return peak
    progress = (step - warm) / max(total_steps - warm, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (cfg.inner_min_lr_ratio + (1.0 - cfg.inner_min_lr_ratio) * cosine)


def make_inner_optimizer(model: nn.Module, cfg: DiLoCoConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.inner_lr,
        betas=cfg.inner_betas,
        weight_decay=cfg.inner_weight_decay,
    )


def make_outer_optimizer(
    modules: dict[str, nn.Module], cfg: DiLoCoConfig
) -> torch.optim.Optimizer:
    """SGD+Nesterov over the authoritative (global) copies of ``modules``.

    The averaged pseudo-gradient is written into each parameter's ``.grad`` and
    a single ``step()`` advances all shared modules this node owns.
    """
    params: list[nn.Parameter] = []
    for mod in modules.values():
        params.extend(mod.parameters())
    return torch.optim.SGD(
        params,
        lr=cfg.outer_lr,
        momentum=cfg.outer_momentum,
        nesterov=cfg.outer_nesterov,
        weight_decay=0.0,
    )


@torch.no_grad()
def module_delta(global_mod: nn.Module, local_mod: nn.Module) -> list[torch.Tensor]:
    """Pseudo-gradient ``global - local`` for one module, in parameter order."""
    return [
        (g.detach() - l.detach())
        for g, l in zip(global_mod.parameters(), local_mod.parameters())
    ]


@torch.no_grad()
def apply_outer_grads(global_mod: nn.Module, delta: list[torch.Tensor]) -> None:
    """Write ``delta`` into the ``.grad`` of a global module's parameters."""
    for p, d in zip(global_mod.parameters(), delta):
        p.grad = d.to(p.device, p.dtype)
