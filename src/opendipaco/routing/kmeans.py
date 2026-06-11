"""Generative routing via k-means (used to build data shards).

Cluster ``i`` maps to path index ``i``; with ``num_paths`` clusters this yields
one disjoint shard per path, matching DiPaCo's document-level routing.
"""

from __future__ import annotations

import torch


class KMeansRouter:
    def __init__(self, num_paths: int, max_iters: int = 50, seed: int = 0, tol: float = 1e-4):
        self.num_paths = num_paths
        self.max_iters = max_iters
        self.seed = seed
        self.tol = tol
        self.centroids: torch.Tensor | None = None

    def fit(self, features: torch.Tensor) -> "KMeansRouter":
        gen = torch.Generator(device=features.device).manual_seed(self.seed)
        n = features.size(0)
        if n < self.num_paths:
            raise ValueError(f"need >= {self.num_paths} samples to fit, got {n}")
        # k-means++-ish init: random distinct points.
        idx = torch.randperm(n, generator=gen, device=features.device)[: self.num_paths]
        centroids = features[idx].clone()
        for _ in range(self.max_iters):
            dists = torch.cdist(features, centroids)  # [N, K]
            assign = dists.argmin(dim=1)
            new = centroids.clone()
            for k in range(self.num_paths):
                mask = assign == k
                if mask.any():
                    new[k] = features[mask].mean(dim=0)
                else:  # re-seed an empty cluster to the farthest point
                    new[k] = features[dists.min(dim=1).values.argmax()]
            shift = (new - centroids).norm()
            centroids = new
            if shift < self.tol:
                break
        self.centroids = centroids
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("KMeansRouter must be fit() before predict()")
        return torch.cdist(features, self.centroids.to(features.device)).argmin(dim=1)

    def predict_topk(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """The ``k`` nearest centroids per row -> ``[N, k]`` (nearest first)."""
        if self.centroids is None:
            raise RuntimeError("KMeansRouter must be fit() before predict_topk()")
        dists = torch.cdist(features, self.centroids.to(features.device))
        return dists.topk(min(k, dists.size(1)), largest=False, dim=1).indices
