"""Test-time path selection and evaluation.

At inference DiPaCo executes a *single* path: the router picks one path from the
input prefix, and only that path's modules run -- no full model is ever
materialised. With the (optional) re-routing window, the path can be re-selected
every ``W`` tokens based on recent context.

Routing may pick any path, so these functions need the **full** module bank.
Under distributed training (one path per rank) gather it first with
:func:`opendipaco.gather_full_bank`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import DiPaCoConfig
from .model import PathModel, build_path_model
from .routing.base import Featurizer, Router
from .topology import Path


def compose_path(config: DiPaCoConfig, bank: dict[str, nn.Module], path: Path) -> PathModel:
    """Assemble a runnable :class:`PathModel` that *aliases* the trained global
    modules (no copy), so it reflects the latest learned weights."""
    return build_path_model(config, path, bank, deepcopy=False)


def route(prefixes: list[torch.Tensor], router: Router, featurizer: Featurizer) -> torch.Tensor:
    """Path index for each prefix sequence."""
    return router.predict(featurizer(prefixes))


@torch.no_grad()
def perplexity(model: PathModel, sequences: torch.Tensor, batch_size: int = 8) -> float:
    """Token-level perplexity of ``model`` on packed ``[N, seq_len]`` sequences."""
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_batches = 0.0, 0
    for start in range(0, sequences.size(0), batch_size):
        batch = sequences[start : start + batch_size].to(device)
        _, loss = model(batch, labels=batch)
        total_loss += float(loss)
        total_batches += 1
    if total_batches == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_loss / total_batches)))


@torch.no_grad()
def routed_perplexity(
    config: DiPaCoConfig,
    bank: dict[str, nn.Module],
    sequences: torch.Tensor,
    router: Router,
    featurizer: Featurizer,
    prefix_len: int = 32,
    batch_size: int = 8,
    compose_fn=None,
) -> float:
    """Perplexity when each sequence is routed to its own path by the router.

    Groups sequences by predicted path so each path runs as a batch. The first
    ``prefix_len`` tokens (used for the routing decision) are **excluded** from the
    perplexity, matching the paper's fair-comparison protocol ("perplexity using
    all but the first 32 tokens of each sequence"). Pass ``compose_fn(path) ->
    PathModel`` (e.g. ``engine.compose_best``) to evaluate each path with its
    early-stopped checkpoint instead of the live ``bank``.
    """
    prefixes = [seq[:prefix_len] for seq in sequences]
    paths = route(prefixes, router, featurizer)
    start_pos = max(prefix_len, 1)  # first scored target position
    total_loss, total_tokens = 0.0, 0
    for path_idx in paths.unique().tolist():
        mask = paths == path_idx
        path = config_path(config, path_idx)
        model = compose_fn(path) if compose_fn is not None else compose_path(config, bank, path)
        seqs = sequences[mask]
        for start in range(0, seqs.size(0), batch_size):
            batch = seqs[start : start + batch_size].to(next(model.parameters()).device)
            if batch.size(1) <= start_pos:
                continue  # nothing left to score after the excluded prefix
            logits, _ = model(batch)
            pred = logits[:, start_pos - 1 : -1, :]
            tgt = batch[:, start_pos:]
            total_loss += float(
                F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="sum")
            )
            total_tokens += tgt.numel()
    if total_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))


def config_path(config: DiPaCoConfig, path_index: int) -> Path:
    return config.build_topology().path_from_index(path_index)


@torch.no_grad()
def routed_window_perplexity(
    config: DiPaCoConfig,
    bank: dict[str, nn.Module],
    sequences: torch.Tensor,
    router: Router,
    featurizer: Featurizer,
    window: int = 64,
    batch_size: int = 8,
    compose_fn=None,
    prefix_len: int = 32,
) -> float:
    """Perplexity with **sub-document re-routing every ``window`` tokens**.

    Each sequence is split into ``window``-token chunks. The path for chunk ``i``
    is predicted from the *previous* chunk's representation (chunk 0 routes from
    itself), then that chunk's tokens are scored under the chosen path conditioned
    on the full preceding context. This is the paper's frequent-re-routing eval
    (Table 3: e.g. 11.38 @ W=64 vs 12.39 routing once per sequence). The first
    ``prefix_len`` tokens of each sequence are excluded from the score (the paper's
    fair-comparison protocol).

    Note: rather than carry a mixed-path KV cache across path switches, the
    context is recomputed under each chunk's active path -- exact for "score this
    chunk under its routed path given the full history", just not cached.
    """
    if sequences.numel() == 0:
        return float("nan")
    device = next(next(iter(bank.values())).parameters()).device
    topo = config.build_topology()
    num_seq, seqlen = sequences.shape
    num_windows = math.ceil(seqlen / window)
    model_cache: dict[int, PathModel] = {}

    total_loss, total_tokens = 0.0, 0
    for w in range(num_windows):
        win_start, win_end = w * window, min((w + 1) * window, seqlen)
        # Exclude the first `prefix_len` tokens (and the position-0 token, which has
        # no context) from the score.
        target_start = max(win_start, prefix_len, 1)
        if target_start >= win_end:
            continue
        # Route the next chunk from the previous chunk (chunk 0 from itself).
        rc = w - 1 if w > 0 else 0
        chunks = [sequences[n, rc * window : min((rc + 1) * window, seqlen)] for n in range(num_seq)]
        paths = route(chunks, router, featurizer)

        for path_idx in paths.unique().tolist():
            rows = (paths == path_idx).nonzero(as_tuple=True)[0]
            model = model_cache.get(path_idx)
            if model is None:
                path = topo.path_from_index(path_idx)
                model = (compose_fn(path) if compose_fn is not None
                         else compose_path(config, bank, path)).eval()
                model_cache[path_idx] = model
            for s in range(0, rows.numel(), batch_size):
                idx = rows[s : s + batch_size]
                batch = sequences[idx, :win_end].to(device)
                logits, _ = model(batch)
                # logits[:, t] predicts token t+1, so target token positions
                # [target_start, win_end) are predicted from [target_start-1, win_end-1).
                pred = logits[:, target_start - 1 : win_end - 1, :]
                tgt = batch[:, target_start:win_end]
                total_loss += float(
                    F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="sum")
                )
                total_tokens += tgt.numel()

    if total_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))
