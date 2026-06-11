"""Single-process backend: one node simulates every path.

Because all paths live in the same process, the engine has already summed each
module's contribution over the paths that share it, so :meth:`global_reduce` is
the identity. This backend is the reference implementation for validating DiPaCo
correctness without any networking.
"""

from __future__ import annotations

from ..topology import Path, PathTopology
from .base import ReducedDelta, SyncBackend


class LocalBackend(SyncBackend):
    def __init__(self, topology: PathTopology):
        self.topology = topology

    def owned_paths(self) -> list[Path]:
        return self.topology.paths()

    def owns_module(self, key: str) -> bool:
        return True

    def global_reduce(self, key: str, local: ReducedDelta) -> ReducedDelta:
        return local
