"""Partition a corpus into one disjoint shard per path.

DiPaCo routes each *document* to a path (via the generative router) and trains
that path only on its shard. The flow is:

1. featurize each document's prefix,
2. ``router.predict`` -> path index per document,
3. group documents by path and pack each group into fixed-length LM sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..routing.base import Featurizer, Router


def assign_paths(
    documents: list[torch.Tensor],
    router: Router,
    featurizer: Featurizer,
    prefix_len: int = 32,
    batch_size: int = 256,
    top_k: int = 1,
) -> torch.Tensor:
    """Path assignment per document.

    ``top_k == 1`` returns ``[num_docs]`` (one path each). ``top_k > 1`` returns
    ``[num_docs, top_k]`` (each document's best paths, for overlapping shards) and
    requires the router to implement ``predict_topk``.
    """
    assignments: list[torch.Tensor] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        prefixes = [d[:prefix_len] for d in batch]
        feats = featurizer(prefixes)
        if top_k == 1:
            assignments.append(router.predict(feats).cpu())
        else:
            if not hasattr(router, "predict_topk"):
                raise ValueError("router must implement predict_topk for top_k > 1")
            assignments.append(router.predict_topk(feats, top_k).cpu())
    if not assignments:
        shape = (0,) if top_k == 1 else (0, top_k)
        return torch.zeros(shape, dtype=torch.long)
    return torch.cat(assignments)


def pack_sequences(documents: list[torch.Tensor], seq_len: int) -> torch.Tensor:
    """Concatenate documents and chunk into ``[N, seq_len]`` (standard LM packing).

    Trailing tokens that don't fill a full window are dropped. Returns an empty
    ``[0, seq_len]`` tensor if there isn't enough data for one window. Note that a
    packed window may span multiple documents (cross-document attention).
    """
    if not documents:
        return torch.zeros(0, seq_len, dtype=torch.long)
    flat = torch.cat([d.reshape(-1) for d in documents])
    n = (flat.numel() // seq_len) * seq_len
    if n == 0:
        return torch.zeros(0, seq_len, dtype=torch.long)
    return flat[:n].reshape(-1, seq_len)


def chunk_documents(documents: list[torch.Tensor], seq_len: int) -> torch.Tensor:
    """Chunk each document *independently* into ``[*, seq_len]`` windows.

    Closer to the paper's document-centric sequences: a window never spans two
    documents (no cross-document attention). Each document's trailing partial
    window is dropped, and documents shorter than ``seq_len`` contribute nothing
    (no padding/masking here).
    """
    chunks: list[torch.Tensor] = []
    for d in documents:
        flat = d.reshape(-1)
        n = (flat.numel() // seq_len) * seq_len
        if n:
            chunks.append(flat[:n].reshape(-1, seq_len))
    if not chunks:
        return torch.zeros(0, seq_len, dtype=torch.long)
    return torch.cat(chunks, dim=0)


@dataclass
class ShardedCorpus:
    """Per-path packed token sequences plus the document/token counts behind them."""

    sequences: dict[int, torch.Tensor]  # path index -> [N_p, seq_len] (training)
    doc_counts: dict[int, int]          # path index -> number of documents
    num_paths: int
    seq_len: int
    token_counts: dict[int, int] | None = None  # path index -> total tokens (the alpha basis)
    val_sequences: dict[int, torch.Tensor] | None = None  # held-out per-path validation

    @classmethod
    def from_assignments(
        cls,
        documents: list[torch.Tensor],
        assignments: torch.Tensor,
        num_paths: int,
        seq_len: int,
        val_fraction: float = 0.0,
        seed: int = 0,
        pack_mode: str = "pack",
    ) -> "ShardedCorpus":
        """Build shards from an explicit per-document assignment.

        ``assignments`` is ``[N]`` (one path each) or ``[N, k]`` (top-k overlapping
        shards -- a document is added to every listed path). Used by both the
        router-based and loss-based (EM) sharding.

        ``pack_mode`` is ``"pack"`` (concatenate a shard's documents, then chunk --
        standard, dense) or ``"document"`` (chunk each document independently, so a
        sequence never spans two documents -- closer to the paper, but drops
        documents shorter than ``seq_len``).

        ``val_fraction > 0`` holds out that fraction of each path's sequences for
        shard-validation (used for per-path early stopping).
        """
        if pack_mode not in ("pack", "document"):
            raise ValueError(f"pack_mode must be 'pack' or 'document', got {pack_mode!r}")
        to_sequences = pack_sequences if pack_mode == "pack" else chunk_documents

        buckets: dict[int, list[torch.Tensor]] = {p: [] for p in range(num_paths)}
        for doc, a in zip(documents, assignments):
            paths = a.tolist() if a.ndim > 0 else [int(a)]
            for p in paths:
                buckets[p].append(doc)
        doc_counts = {p: len(docs) for p, docs in buckets.items()}
        token_counts = {p: sum(int(d.numel()) for d in docs) for p, docs in buckets.items()}

        sequences: dict[int, torch.Tensor] = {}
        val_sequences: dict[int, torch.Tensor] = {}
        gen = torch.Generator().manual_seed(seed)
        for p, docs in buckets.items():
            packed = to_sequences(docs, seq_len)
            if val_fraction > 0 and packed.size(0) > 1:
                n_val = max(1, int(packed.size(0) * val_fraction))
                perm = torch.randperm(packed.size(0), generator=gen)
                val_sequences[p] = packed[perm[:n_val]]
                sequences[p] = packed[perm[n_val:]]
            else:
                sequences[p] = packed
                val_sequences[p] = packed[:0]
        return cls(
            sequences, doc_counts, num_paths, seq_len, token_counts,
            val_sequences if val_fraction > 0 else None,
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[torch.Tensor],
        router: Router,
        featurizer: Featurizer,
        num_paths: int,
        seq_len: int,
        prefix_len: int = 32,
        top_k: int = 1,
        val_fraction: float = 0.0,
        seed: int = 0,
        pack_mode: str = "pack",
    ) -> "ShardedCorpus":
        assignments = assign_paths(documents, router, featurizer, prefix_len, top_k=top_k)
        return cls.from_assignments(
            documents, assignments, num_paths, seq_len, val_fraction, seed, pack_mode
        )

    def shard(self, path_index: int) -> torch.Tensor:
        return self.sequences[path_index]

    @property
    def has_validation(self) -> bool:
        return self.val_sequences is not None

    def val_shard(self, path_index: int) -> torch.Tensor | None:
        if self.val_sequences is None:
            return None
        return self.val_sequences.get(path_index)

    def num_sequences(self, path_index: int) -> int:
        return self.sequences[path_index].size(0)

    def shard_weight(self, path_index: int) -> float:
        """Relative shard size used to weight the path's pseudo-gradient (alpha).

        Based on token count (the paper's ``|D_{l,e}| / |D_total|``); falls back to
        packed-sequence count only if token counts were not recorded. Token-based
        weighting avoids a path with short documents being zero-weighted just
        because its tokens don't fill a full ``seq_len`` window.
        """
        if self.token_counts is not None:
            total = sum(self.token_counts.values())
            return self.token_counts[path_index] / max(total, 1)
        total = sum(s.size(0) for s in self.sequences.values())
        return self.sequences[path_index].size(0) / max(total, 1)
