"""Learned prefix featurizers for routing.

DiPaCo routes on a representation ``z`` of a sequence's first ~32 tokens. These
featurizers produce ``z`` from frozen models (no gradients), which is far
stronger than the bag-of-tokens baseline:

* :class:`EmbeddingFeaturizer`  -- masked mean/last pooling of token embeddings
  (cheap; pass a *pretrained* embedding table for a meaningful signal).
* :class:`HFEncoderFeaturizer`  -- masked pooling of a frozen HuggingFace model's
  hidden states (the paper's "representation from a transformer"; the model must
  share the data's tokenizer / vocab).

Both satisfy the :class:`~opendipaco.routing.base.Featurizer` protocol, so they
drop straight into ``KMeansRouter``, ``DiscriminativeRouter`` and
``ShardedCorpus.from_documents``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _pad_batch(
    prefixes: list[torch.Tensor], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length prefixes into ``ids[N, L]`` + ``mask[N, L]`` (1=token)."""
    lengths = [int(p.numel()) for p in prefixes]
    max_len = max(lengths) if lengths else 0
    max_len = max(max_len, 1)  # avoid zero-width tensors
    ids = torch.full((len(prefixes), max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(prefixes), max_len), dtype=torch.bool, device=device)
    for i, p in enumerate(prefixes):
        n = lengths[i]
        if n:
            ids[i, :n] = p.to(device=device, dtype=torch.long)
            mask[i, :n] = True
    return ids, mask


def _pool(hidden: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool ``hidden[N, L, D]`` over valid positions -> ``[N, D]``."""
    m = mask.unsqueeze(-1).to(hidden.dtype)  # [N, L, 1]
    if pooling == "mean":
        counts = m.sum(dim=1).clamp(min=1.0)
        return (hidden * m).sum(dim=1) / counts
    if pooling == "last":
        # index of the last valid position per row (0 for empty rows)
        last = mask.float().cumsum(dim=1).argmax(dim=1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last]
    raise ValueError(f"unknown pooling {pooling!r} (use 'mean' or 'last')")


class EmbeddingFeaturizer:
    """Masked pooling of a (frozen) token-embedding table.

    Pass any ``nn.Embedding`` or a ``[vocab, dim]`` weight tensor -- ideally a
    *pretrained* embedding so similar contexts land near each other. Lives on CPU
    by default; features are returned on CPU.
    """

    def __init__(
        self,
        embedding: nn.Embedding | torch.Tensor,
        *,
        pooling: str = "mean",
        normalize: bool = True,
        pad_id: int = 0,
        device: str | torch.device = "cpu",
        batch_size: int = 256,
    ):
        weight = embedding.weight if isinstance(embedding, nn.Embedding) else embedding
        self.weight = weight.detach().to(device)
        self.feature_dim = self.weight.size(-1)
        self.pooling = pooling
        self.normalize = normalize
        self.pad_id = pad_id
        self.device = torch.device(device)
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, prefixes: list[torch.Tensor]) -> torch.Tensor:
        out: list[torch.Tensor] = []
        for start in range(0, len(prefixes), self.batch_size):
            batch = prefixes[start : start + self.batch_size]
            ids, mask = _pad_batch(batch, self.pad_id, self.device)
            hidden = F.embedding(ids, self.weight)
            out.append(_pool(hidden, mask, self.pooling).cpu())
        feats = torch.cat(out) if out else torch.zeros(0, self.feature_dim)
        return F.normalize(feats, dim=-1) if self.normalize else feats


class ModelFeaturizer:
    """Routing features from the DiPaCo model itself (the paper's ``z``).

    Runs each prefix through a *reference path* and pools its final hidden states.
    With identical-expert init / warm-start, every path is identical at the start
    of training, so the reference-path features equal the pretrained dense model's
    features -- exactly the representation the paper uses for the initial k-means
    sharding. As paths diverge, a single fixed reference path gives a consistent
    (if path-biased) representation for the discriminative / EM step.

    ``reference_path`` defaults to path index 0. The bank must contain that path's
    modules (the full bank, or a gathered bank in distributed).
    """

    def __init__(
        self,
        bank: dict[str, nn.Module],
        config,
        reference_path=None,
        pooling: str = "mean",
        normalize: bool = True,
        pad_id: int = 0,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
    ):
        from ..model import build_path_model  # local import avoids any cycle

        topo = config.build_topology()
        if reference_path is None:
            reference_path = topo.path_from_index(0)
        # Alias the bank (no copy) so features track the current weights. We must
        # NOT call ``.requires_grad_(False)`` here: that would mutate the shared
        # bank modules for the reference path and break their training (the path
        # is then materialized with grad disabled). Featurization is already done
        # under ``@torch.no_grad()`` in ``__call__``, so no graph is built anyway.
        self.model = build_path_model(config, reference_path, bank, deepcopy=False).to(device)
        self.feature_dim = config.backbone.hidden_size
        self.pooling = pooling
        self.normalize = normalize
        self.pad_id = pad_id
        self.device = torch.device(device)
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, prefixes: list[torch.Tensor]) -> torch.Tensor:
        out: list[torch.Tensor] = []
        for start in range(0, len(prefixes), self.batch_size):
            batch = prefixes[start : start + self.batch_size]
            ids, mask = _pad_batch(batch, self.pad_id, self.device)
            # Right-padding + causal attention -> real-token states are unaffected
            # by padding; pooling masks the padded positions out.
            hidden = self.model.encode(ids)
            out.append(_pool(hidden, mask, self.pooling).float().cpu())
        feats = torch.cat(out) if out else torch.zeros(0, self.feature_dim)
        return F.normalize(feats, dim=-1) if self.normalize else feats


class HFEncoderFeaturizer:
    """Masked pooling of a frozen HuggingFace model's hidden states.

    ``model`` may be a model name (loaded via ``AutoModel``) or an ``nn.Module``
    that returns ``last_hidden_state`` (or a tensor) given ``input_ids`` and
    ``attention_mask``. The encoder is frozen and run under ``no_grad``. Its
    tokenizer/vocab must match the data's.
    """

    def __init__(
        self,
        model: str | nn.Module,
        *,
        pooling: str = "mean",
        normalize: bool = True,
        pad_id: int = 0,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
    ):
        if isinstance(model, str):
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model)
        self.model = model.to(device).eval().requires_grad_(False)
        cfg = getattr(model, "config", None)
        self.feature_dim = getattr(cfg, "hidden_size", None)
        self.pooling = pooling
        self.normalize = normalize
        self.pad_id = getattr(cfg, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = pad_id
        self.device = torch.device(device)
        self.batch_size = batch_size

    @staticmethod
    def _hidden(outputs) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return outputs
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]

    @torch.no_grad()
    def __call__(self, prefixes: list[torch.Tensor]) -> torch.Tensor:
        out: list[torch.Tensor] = []
        for start in range(0, len(prefixes), self.batch_size):
            batch = prefixes[start : start + self.batch_size]
            ids, mask = _pad_batch(batch, self.pad_id, self.device)
            hidden = self._hidden(self.model(input_ids=ids, attention_mask=mask.long()))
            pooled = _pool(hidden, mask, self.pooling).float().cpu()
            out.append(pooled)
            if self.feature_dim is None:
                self.feature_dim = pooled.size(-1)
        feats = torch.cat(out) if out else torch.zeros(0, self.feature_dim or 1)
        return F.normalize(feats, dim=-1) if self.normalize else feats
