"""Distributed backend: each process (rank) owns one or more paths.

``num_paths`` must be divisible by ``world_size``; rank ``r`` trains the
contiguous block of ``num_paths / world_size`` paths starting at ``r * block``
and holds authoritative copies of exactly those paths' modules. A shared module
is averaged across the subgroup of ranks that own at least one path using it; we
build one ``torch.distributed`` process group per module key up front (a
collective that every rank must enter in the same order).

Launch with ``torchrun --nproc_per_node=<world_size> your_script.py`` for any
``world_size`` that divides ``num_paths`` (e.g. 1 rank per path, or fewer ranks
each owning several paths).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ..topology import Path, PathTopology
from .base import ReducedDelta, SyncBackend


class TorchDistBackend(SyncBackend):
    def __init__(self, topology: PathTopology, device: torch.device | None = None):
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized; call dist.init_process_group first"
            )
        world = dist.get_world_size()
        if topology.num_paths % world != 0:
            raise ValueError(
                f"num_paths ({topology.num_paths}) must be divisible by world_size "
                f"({world}); each rank owns num_paths / world_size paths"
            )
        self.topology = topology
        self.rank = dist.get_rank()
        self.group_size = topology.num_paths // world  # paths owned per rank
        self.device = device or (
            torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        base = self.rank * self.group_size
        self._paths = [topology.path_from_index(base + j) for j in range(self.group_size)]

        # Build one subgroup per module key (collective: same order on all ranks).
        # A key's subgroup is the set of ranks owning >= 1 path through it.
        self._groups: dict[str, dist.ProcessGroup] = {}
        self._member: dict[str, bool] = {}
        for key in topology.module_keys():
            ranks = sorted(
                {self.rank_of_path(p) for p in topology.paths_through_module(key)}
            )
            self._groups[key] = dist.new_group(ranks=ranks)
            self._member[key] = self.rank in ranks

    def rank_of_path(self, path: Path) -> int:
        return self.topology.path_index(path) // self.group_size

    def owned_paths(self) -> list[Path]:
        return list(self._paths)

    def owns_module(self, key: str) -> bool:
        return self._member.get(key, False)

    def global_reduce(self, key: str, local: ReducedDelta) -> ReducedDelta:
        group = self._groups[key]
        summed = [t.to(self.device) for t in local.summed]
        for t in summed:
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        w = torch.tensor([local.weight], device=self.device)
        dist.all_reduce(w, op=dist.ReduceOp.SUM, group=group)
        return ReducedDelta(summed=summed, weight=float(w.item()))

    def barrier(self) -> None:
        dist.barrier()
