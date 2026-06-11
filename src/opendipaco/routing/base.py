"""Routing interfaces and a default prefix featurizer.

DiPaCo uses *coarse* (document-level) routing: a sequence's prefix is mapped to
a feature vector, and a router turns that into a path index. Two routers are
provided -- a generative k-means router (used to *build* the data shards) and a
discriminative linear router (used at test time). Both consume features from a
:class:`Featurizer`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Featurizer(Protocol):
    """Maps a batch of prefix token-id sequences to feature vectors."""

    feature_dim: int

    def __call__(self, prefixes: list[torch.Tensor]) -> torch.Tensor:
        """``prefixes``: list of 1-D LongTensors. Returns ``[N, feature_dim]``."""
        ...


@runtime_checkable
class Router(Protocol):
    """Maps prefix features to path indices."""

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """``features``: ``[N, D]``. Returns ``[N]`` Long path indices."""
        ...


class BagOfTokensFeaturizer:
    """Deterministic, dependency-free featurizer.

    Builds a normalized bag-of-tokens count vector for each prefix and projects
    it through a *fixed* random matrix (a frozen random projection / hashing
    trick). Documents using similar vocabulary land near each other, which is
    enough signal for coarse clustering. Swap in sentence-embeddings or a frozen
    LM encoder for a stronger generative router.
    """

    def __init__(self, vocab_size: int, feature_dim: int = 256, seed: int = 0):
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        gen = torch.Generator().manual_seed(seed)
        # Fixed projection: [vocab_size, feature_dim].
        self.projection = torch.randn(vocab_size, feature_dim, generator=gen) / feature_dim**0.5

    def __call__(self, prefixes: list[torch.Tensor]) -> torch.Tensor:
        feats = torch.zeros(len(prefixes), self.feature_dim)
        for i, ids in enumerate(prefixes):
            if ids.numel() == 0:
                continue
            counts = torch.bincount(ids.clamp(max=self.vocab_size - 1), minlength=self.vocab_size)
            counts = counts.float()
            counts /= counts.sum().clamp(min=1.0)
            feats[i] = counts @ self.projection
        # L2-normalize so k-means uses cosine-like geometry.
        return torch.nn.functional.normalize(feats, dim=-1)
