"""Backend interface for cross-path module synchronization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ..topology import Path, PathTopology


@dataclass
class ReducedDelta:
    """A module's pseudo-gradient summed (weighted) over all sharing paths.

    ``summed`` holds ``sum_i w_i * (global - local_i)`` across every path that
    shares the module, ``weight`` holds ``sum_i w_i``. The engine divides to get
    the averaged delta. Keeping them separate lets distributed backends reduce
    sums with a single all-reduce.
    """

    summed: list[torch.Tensor]
    weight: float


class SyncBackend(ABC):
    """Reduces per-module pseudo-gradients across the paths that share them."""

    topology: PathTopology

    @abstractmethod
    def owned_paths(self) -> list[Path]:
        """Paths this node is responsible for training this round."""

    def rank_of_path(self, path: Path) -> int:
        """Process rank that owns ``path`` (0 in a single process)."""
        return 0

    @abstractmethod
    def owns_module(self, key: str) -> bool:
        """Whether this node holds an authoritative copy of module ``key``."""

    @abstractmethod
    def global_reduce(self, key: str, local: ReducedDelta) -> ReducedDelta:
        """Combine this node's ``local`` contribution with all other nodes that
        share module ``key``, returning the globally summed delta and weight.

        For the local backend this is the identity (all sharing paths are
        already local). For distributed backends this performs an all-reduce
        over the subgroup of processes that share ``key``.
        """

    def barrier(self) -> None:  # optional sync point
        pass
