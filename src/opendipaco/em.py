"""EM re-sharding: re-assign documents to their lowest-loss path.

DiPaCo's routing is an approximate EM: the **M-step** trains the paths on the
current shards (the normal training loop), and the **E-step** re-assigns each
document to the path that best explains it (lowest LM loss). Running this once or
a few times during training lets shards follow the model as paths specialise.

After re-sharding you typically fit a :class:`DiscriminativeRouter` on the new
assignments so test-time routing matches (the E-step's argmin needs every path's
loss, which is too expensive at inference).

Scoring all ``P`` paths for every document is ``O(N * P)`` forwards; restrict
``candidates`` (e.g. to a router's top-k) to cut that cost.

These functions need the **full** module bank (they score arbitrary paths). Under
distributed training each rank holds only its own path, so first assemble the
complete bank with :func:`opendipaco.gather_full_bank` and run the E-step on a
coordinator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import DiPaCoConfig
from .data.sharding import ShardedCorpus
from .inference import compose_path
from .routing.base import Featurizer
from .routing.discriminative import DiscriminativeRouter
from .routing.featurizers import _pad_batch


@torch.no_grad()
def path_losses(
    documents: list[torch.Tensor],
    config: DiPaCoConfig,
    bank: dict[str, nn.Module],
    seq_len: int,
    candidates: list[int] | None = None,
    batch_size: int = 16,
    pad_id: int = 0,
) -> torch.Tensor:
    """Per-document LM loss under each candidate path -> ``[N, num_paths]``.

    Non-candidate paths are left as ``+inf`` so they're never chosen by argmin/topk.
    """
    topo = config.build_topology()
    if candidates is None:
        candidates = list(range(topo.num_paths))
    device = next(next(iter(bank.values())).parameters()).device
    docs = [d[:seq_len] for d in documents]
    losses = torch.full((len(docs), topo.num_paths), float("inf"))

    for p in candidates:
        model = compose_path(config, bank, topo.path_from_index(p)).eval()
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            ids, mask = _pad_batch(batch, pad_id, device)
            logits, _ = model(ids)
            pred = logits[:, :-1, :]
            tgt = ids[:, 1:]
            tgt_mask = mask[:, 1:].to(pred.dtype)
            ce = F.cross_entropy(
                pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="none"
            ).reshape(tgt.shape)
            per_doc = (ce * tgt_mask).sum(dim=1) / tgt_mask.sum(dim=1).clamp(min=1.0)
            losses[start : start + len(batch), p] = per_doc.cpu()
    return losses


def assign_paths_by_loss(
    documents: list[torch.Tensor],
    config: DiPaCoConfig,
    bank: dict[str, nn.Module],
    seq_len: int,
    top_k: int = 1,
    candidates: list[int] | None = None,
    batch_size: int = 16,
    pad_id: int = 0,
) -> torch.Tensor:
    """EM E-step assignment: ``[N]`` (top-1) or ``[N, top_k]`` lowest-loss paths."""
    losses = path_losses(documents, config, bank, seq_len, candidates, batch_size, pad_id)
    if top_k == 1:
        return losses.argmin(dim=1)
    return losses.topk(top_k, largest=False, dim=1).indices


def reshard_by_loss(
    documents: list[torch.Tensor],
    config: DiPaCoConfig,
    bank: dict[str, nn.Module],
    seq_len: int,
    top_k: int = 1,
    candidates: list[int] | None = None,
    batch_size: int = 16,
    pad_id: int = 0,
) -> ShardedCorpus:
    """Run the E-step and rebuild a :class:`ShardedCorpus` from the new assignment."""
    assignments = assign_paths_by_loss(
        documents, config, bank, seq_len, top_k, candidates, batch_size, pad_id
    )
    return ShardedCorpus.from_assignments(documents, assignments, config.num_paths, seq_len)


def fit_discriminative_router(
    documents: list[torch.Tensor],
    config: DiPaCoConfig,
    bank: dict[str, nn.Module],
    featurizer: Featurizer,
    seq_len: int,
    prefix_len: int = 32,
    candidates: list[int] | None = None,
    batch_size: int = 16,
    **fit_kwargs,
) -> DiscriminativeRouter:
    """Train a discriminative router to predict each document's **lowest-loss path**.

    This is the paper's amortized E-step: label each sequence by ``argmax_j s_j``
    (the path with the highest likelihood = lowest loss) and fit a logistic
    regression on the prefix features. Pass a **held-out** subset of documents
    (the paper's reserved "second part") so the router isn't trained on the same
    data the paths were sharded on. Extra ``fit_kwargs`` (e.g. ``val_fraction``)
    are forwarded to :meth:`DiscriminativeRouter.fit`.
    """
    labels = assign_paths_by_loss(
        documents, config, bank, seq_len, top_k=1, candidates=candidates, batch_size=batch_size
    )
    feats = featurizer([d[:prefix_len] for d in documents])
    router = DiscriminativeRouter(featurizer.feature_dim, config.num_paths)
    router.fit(feats, labels, **fit_kwargs)
    return router
