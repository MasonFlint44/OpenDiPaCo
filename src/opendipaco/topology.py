"""Path / module topology bookkeeping.

A DiPaCo model is an ordered list of **segments**: the token embedding, the body
(transformer-block groups), and the head. Each body segment is either

* a **routing level** -- ``num_experts > 1`` modules; a path picks one, and that
  expert is averaged (DiLoCo) across the ``num_paths / num_experts`` paths that
  pick it; this is what defines the set of paths, or
* a **non-routing block** -- a single module position that is either ``SHARED``
  (one instance, averaged across *all* paths) or ``PRIVATE`` (a per-path copy
  that is **never communicated**, matching the paper's "not communicated across
  paths" blocks / private embedding in the 16x16 model).

The embedding and head are non-routing segments and may likewise be SHARED or
PRIVATE. A *path* is the tuple of expert choices over the routing levels only,
so making a component private/shared does not change the set of paths.

Module keys (stable across the simple default config):
    "embed" / "embed.p{i}"   shared / private embedding
    "head"  / "head.p{i}"    shared / private head
    "L{j}E{e}"               expert ``e`` of routing level ``j`` (0-indexed)
    "B{s}"  / "B{s}.p{i}"    shared / private body block at segment ``s``
A ``.p{i}`` suffix marks a private, never-averaged copy belonging to path ``i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import prod

Path = tuple[int, ...]  # one expert index per routing level


class Sharing(str, Enum):
    SHARED = "shared"    # one instance, averaged across the sharing group
    PRIVATE = "private"  # per-path instance, never communicated/averaged


@dataclass(frozen=True)
class Segment:
    """One position in a path: embedding, a body block group, or the head."""

    role: str                       # "embed" | "body" | "head"
    layers: int = 0                 # decoder layers (0 for embed/head)
    num_experts: int = 1            # > 1 => routing level
    sharing: Sharing = Sharing.SHARED

    @property
    def is_routing(self) -> bool:
        return self.num_experts > 1


def embed_key() -> str:
    return "embed"


def head_key() -> str:
    return "head"


def expert_key(level: int, expert: int) -> str:
    return f"L{level}E{expert}"


def is_private_key(key: str) -> bool:
    """Whether ``key`` names a per-path, never-averaged module."""
    return ".p" in key


class PathTopology:
    """Enumerates paths and the module-sharing structure they induce.

    Backwards-compatible constructor: ``PathTopology(level_sizes)`` builds the
    simple all-shared topology (shared embed/head + one shared routing level per
    entry). Pass ``segments=`` for full control (private blocks, trunk, etc.).
    """

    def __init__(
        self,
        level_sizes: tuple[int, ...] | None = None,
        share_embedding: bool = True,
        share_head: bool = True,
        *,
        segments: list[Segment] | None = None,
        layers_per_level: tuple[int, ...] | None = None,
        embedding: str = "shared",
        head: str = "shared",
    ):
        if segments is None:
            if level_sizes is None:
                raise ValueError("provide either level_sizes or segments")
            level_sizes = tuple(level_sizes)
            if layers_per_level is None:
                layers_per_level = (1,) * len(level_sizes)
            emb = Sharing.PRIVATE if not share_embedding else Sharing(embedding)
            hd = Sharing.PRIVATE if not share_head else Sharing(head)
            segments = [Segment("embed", 0, 1, emb)]
            for size, layers in zip(level_sizes, layers_per_level):
                segments.append(Segment("body", layers, size, Sharing.SHARED))
            segments.append(Segment("head", 0, 1, hd))
        self.segments: tuple[Segment, ...] = tuple(segments)

        self.routing_indices = [i for i, s in enumerate(self.segments) if s.is_routing]
        self.level_sizes = tuple(self.segments[i].num_experts for i in self.routing_indices)

        # Layer offsets so each path's decoder layers get unique, contiguous indices.
        self._offset: dict[int, int] = {}
        acc = 0
        for i, s in enumerate(self.segments):
            self._offset[i] = acc
            acc += s.layers
        self.total_layers = acc

    # -- basic counts --------------------------------------------------------
    @property
    def num_levels(self) -> int:
        return len(self.routing_indices)

    @property
    def num_paths(self) -> int:
        return prod(self.level_sizes) if self.level_sizes else 1

    # -- paths ---------------------------------------------------------------
    def paths(self) -> list[Path]:
        if not self.level_sizes:
            return [()]
        return [tuple(p) for p in product(*(range(k) for k in self.level_sizes))]

    def path_index(self, path: Path) -> int:
        idx = 0
        for size, choice in zip(self.level_sizes, path):
            idx = idx * size + choice
        return idx

    def path_from_index(self, index: int) -> Path:
        out: list[int] = []
        for size in reversed(self.level_sizes):
            out.append(index % size)
            index //= size
        return tuple(reversed(out))

    # -- module keys ---------------------------------------------------------
    def _base_name(self, seg_index: int) -> str:
        role = self.segments[seg_index].role
        if role == "embed":
            return "embed"
        if role == "head":
            return "head"
        return f"B{seg_index}"

    def _segment_key(self, seg_index: int, path: Path) -> str:
        seg = self.segments[seg_index]
        if seg.is_routing:
            j = self.routing_indices.index(seg_index)
            return expert_key(j, path[j])
        base = self._base_name(seg_index)
        if seg.sharing == Sharing.PRIVATE:
            return f"{base}.p{self.path_index(path)}"
        return base

    def module_keys(self) -> list[str]:
        keys: list[str] = []
        for i, seg in enumerate(self.segments):
            if seg.is_routing:
                j = self.routing_indices.index(i)
                keys.extend(expert_key(j, e) for e in range(seg.num_experts))
            elif seg.sharing == Sharing.SHARED:
                keys.append(self._base_name(i))
            else:
                keys.extend(f"{self._base_name(i)}.p{p}" for p in range(self.num_paths))
        return keys

    def path_module_keys(self, path: Path) -> list[str]:
        """Ordered module keys realised by ``path`` (embed -> body -> head)."""
        return [self._segment_key(i, path) for i in range(len(self.segments))]

    # -- sharing structure ---------------------------------------------------
    def paths_through_module(self, key: str) -> list[Path]:
        if is_private_key(key):
            return [self.path_from_index(int(key.split(".p")[1]))]
        if key[0] == "L" and "E" in key:
            j, e = _parse_expert_key(key)
            return [p for p in self.paths() if p[j] == e]
        return self.paths()  # shared, non-routing -> every path

    def sharing_count(self, key: str) -> int:
        """How many paths share ``key`` (``P_{l,e}`` in the paper; 1 if private)."""
        if is_private_key(key):
            return 1
        if key[0] == "L" and "E" in key:
            j, _ = _parse_expert_key(key)
            return self.num_paths // self.level_sizes[j]
        return self.num_paths

    # -- module construction info (used by model.py) -------------------------
    def _segment_index_of_key(self, key: str) -> int:
        base = key.split(".p")[0]
        if base == "embed":
            return next(i for i, s in enumerate(self.segments) if s.role == "embed")
        if base == "head":
            return next(i for i, s in enumerate(self.segments) if s.role == "head")
        if base[0] == "L":
            j, _ = _parse_expert_key(base)
            return self.routing_indices[j]
        return int(base[1:])  # "B{s}"

    def role_of_key(self, key: str) -> str:
        return self.segments[self._segment_index_of_key(key)].role

    def layers_of_key(self, key: str) -> int:
        return self.segments[self._segment_index_of_key(key)].layers

    def offset_of_key(self, key: str) -> int:
        return self._offset[self._segment_index_of_key(key)]


def _parse_expert_key(key: str) -> tuple[int, int]:
    level_str, expert_str = key[1:].split("E")
    return int(level_str), int(expert_str)
