"""Helpers for evaluation / EM under distributed training.

During distributed training each rank holds only its own path's modules, so it
cannot build an arbitrary path -- which routed evaluation and EM re-sharding both
need. :func:`gather_full_bank` reassembles the complete, trained module bank on
every rank by broadcasting each module from one of its owners. It is an explicit,
opt-in call (it materialises every module on every rank), intended for an eval /
coordination step rather than the hot training loop.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import DiPaCoConfig
from .model import build_module_bank


@torch.no_grad()
def gather_full_bank(backend, owned_bank: dict[str, nn.Module], config: DiPaCoConfig):
    """Return the complete module bank, identical on every rank.

    In a single process (``LocalBackend`` / no distributed group) the owned bank
    already is the full bank, so it is returned unchanged. Under
    ``TorchDistBackend`` each module is broadcast from its lowest-ranked owner.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return owned_bank

    topo = backend.topology
    device = next(next(iter(owned_bank.values())).parameters()).device
    full = {k: m.to(device) for k, m in build_module_bank(config).items()}

    for key in topo.module_keys():  # deterministic order -> collective-safe
        # Pick a source rank that owns this module (lowest path index -> its rank).
        owner_path = min(topo.paths_through_module(key), key=topo.path_index)
        src = backend.rank_of_path(owner_path)
        if dist.get_rank() == src:
            for fp, op in zip(full[key].parameters(), owned_bank[key].parameters()):
                fp.data.copy_(op.data)
        for p in full[key].parameters():
            dist.broadcast(p.data, src=src)
    return full
