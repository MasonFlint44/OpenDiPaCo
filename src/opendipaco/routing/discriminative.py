"""Discriminative routing: a linear classifier over prefix features.

At test time DiPaCo does not have access to the generative clustering used to
build shards, so it trains a cheap classifier to predict a sequence's path from
its prefix (the paper's logistic-regression router). Train it on the shard
assignments produced by the k-means router, then use it to route eval chunks.
"""

from __future__ import annotations

import torch
from torch import nn


class DiscriminativeRouter(nn.Module):
    def __init__(self, feature_dim: int, num_paths: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_paths)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.0,
        seed: int = 0,
    ) -> "DiscriminativeRouter":
        """Fit the router.

        With ``val_fraction > 0`` a held-out split is carved out (as in the paper):
        the classifier trains on the rest and the parameters with the lowest
        held-out loss are kept (early stopping), guarding against overfitting the
        path assignment.
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if val_fraction <= 0 or features.size(0) < 2:
            for _ in range(epochs):
                opt.zero_grad()
                nn.functional.cross_entropy(self(features), labels).backward()
                opt.step()
            return self

        n = features.size(0)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        n_val = max(1, int(n * val_fraction))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        x_tr, y_tr = features[tr_idx], labels[tr_idx]
        x_val, y_val = features[val_idx], labels[val_idx]

        best_loss, best_state = float("inf"), None
        for _ in range(epochs):
            opt.zero_grad()
            nn.functional.cross_entropy(self(x_tr), y_tr).backward()
            opt.step()
            with torch.no_grad():
                v = float(nn.functional.cross_entropy(self(x_val), y_val))
            if v < best_loss:
                best_loss = v
                best_state = {k: t.detach().clone() for k, t in self.state_dict().items()}
        if best_state is not None:
            self.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).argmax(dim=-1)

    @torch.no_grad()
    def predict_topk(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """The ``k`` highest-scoring paths per row -> ``[N, k]`` (best first)."""
        logits = self(features)
        return logits.topk(min(k, logits.size(-1)), dim=-1).indices
