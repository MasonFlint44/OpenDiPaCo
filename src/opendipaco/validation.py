"""Validation harness: DiPaCo vs. a matched dense baseline (the paper's claim).

DiPaCo's headline is that a modular model — at test time only **one path** executes
— matches or beats a **dense** model of that same per-path size, having distilled
more from the data into the shared modules. This harness runs that comparison.

The key simplification: **a dense baseline is just DiPaCo with one expert per
level** (``level_sizes=[1, 1]`` → a single path = a plain dense transformer). So the
whole comparison reuses the engine — same backbone, so one DiPaCo path is exactly
the dense model's size. We train:

* the **dense** model (1 path) on *all* the data, and
* the **DiPaCo** model (K×K paths) on routed shards (each path on its slice),

then evaluate **both** on the same held-out test set with `routed_perplexity`
(equal inference cost: one path executes). DiPaCo uses more *total* training compute
— that's the point: a bigger model that's cheap to serve.

**Caveat:** DiPaCo's advantage is a *scale* phenomenon. At toy size (a few-M-param
backbone, a small corpus, CPU) the comparison may be inconclusive or even favor
dense. The harness is correct and ready to run at scale; it does not promise the
paper's result at toy scale. Read both numbers honestly.
"""

from __future__ import annotations

import torch

from .config import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from .backend import LocalBackend
from .data import ShardedCorpus, pack_sequences
from .em import fit_discriminative_router
from .inference import routed_perplexity
from .routing import BagOfTokensFeaturizer, KMeansRouter, ModelFeaturizer
from .train import DiPaCoEngine


class _ZeroRouter:
    """Routes everything to path 0 (for the single-path dense baseline)."""

    def predict(self, features) -> torch.Tensor:
        return torch.zeros(features.shape[0], dtype=torch.long)

    def predict_topk(self, features, k) -> torch.Tensor:
        return torch.zeros(features.shape[0], k, dtype=torch.long)


def _params(engine) -> int:
    return sum(p.numel() for m in engine.bank.values() for p in m.parameters())


def run_comparison(
    backbone: BackboneConfig,
    train_docs: list,
    test_docs: list,
    diloco: DiLoCoConfig,
    *,
    dipaco_levels=(2, 2),
    dense_levels=(1, 1),
    rounds: int = 20,
    batch_size: int = 8,
    seq_len: int = 64,
    eval_len: int = 128,
    seed: int = 0,
    router_seed: int = 0,
    router_holdout: float = 0.15,
    device: str | torch.device = "cpu",
) -> dict:
    """Train a dense baseline + a DiPaCo model on ``backbone`` and compare held-out
    perplexity. Returns dense vs. DiPaCo perplexity and the model sizes.

    Both models train on the same documents; a small reserved slice is held out to
    fit DiPaCo's discriminative test-time router (the paper's "second part").
    ``device`` (e.g. ``"cuda"``) runs the whole comparison on that device.
    """
    dense_cfg = DiPaCoConfig(backbone=backbone, level_sizes=list(dense_levels),
                             sequence_length=seq_len, eval_sequence_length=eval_len)
    dipaco_cfg = DiPaCoConfig(backbone=backbone, level_sizes=list(dipaco_levels),
                              sequence_length=seq_len, eval_sequence_length=eval_len)
    test_seqs = pack_sequences(test_docs, dense_cfg.eval_seq_len)

    n_hold = max(1, int(len(train_docs) * router_holdout))
    router_docs, core_docs = train_docs[:n_hold], train_docs[n_hold:]

    # --- dense baseline: one path, trained on all the (core) data ---
    dense_eng = DiPaCoEngine(dense_cfg, diloco, LocalBackend(dense_cfg.build_topology()),
                             seed=seed, device=device)
    dense_corpus = ShardedCorpus.from_assignments(
        core_docs, torch.zeros(len(core_docs), dtype=torch.long), dense_cfg.num_paths, seq_len
    )
    dense_eng.total_rounds = rounds
    dense_eng.fit(dense_corpus, num_rounds=rounds, batch_size=batch_size, log_every=0)
    dense_ppl = routed_perplexity(
        dense_cfg, dense_eng.global_modules(), test_seqs,
        _ZeroRouter(), BagOfTokensFeaturizer(backbone.vocab_size),
    )

    # --- DiPaCo: K x K paths, each trained on its routed shard ---
    dipaco_eng = DiPaCoEngine(dipaco_cfg, diloco, LocalBackend(dipaco_cfg.build_topology()),
                              seed=seed, device=device)
    feat = ModelFeaturizer(dipaco_eng.global_modules(), dipaco_cfg, device=device)
    kmeans = KMeansRouter(dipaco_cfg.num_paths, seed=router_seed).fit(
        feat([d[:seq_len] for d in core_docs])
    )
    dipaco_corpus = ShardedCorpus.from_documents(
        core_docs, kmeans, feat, dipaco_cfg.num_paths, seq_len
    )
    dipaco_eng.total_rounds = rounds
    dipaco_eng.fit(dipaco_corpus, num_rounds=rounds, batch_size=batch_size, log_every=0)
    # Test-time router: predict each doc's lowest-loss path (the paper's amortized
    # E-step), fit on the reserved held-out slice.
    disc = fit_discriminative_router(router_docs, dipaco_cfg, dipaco_eng.global_modules(),
                                     feat, seq_len)
    dipaco_ppl = routed_perplexity(dipaco_cfg, dipaco_eng.global_modules(), test_seqs, disc, feat)

    return {
        "dense_ppl": dense_ppl,
        "dipaco_ppl": dipaco_ppl,
        "params_per_path": _params(dense_eng),       # one path == the dense model
        "dipaco_paths": dipaco_cfg.num_paths,
        "dipaco_total_params": _params(dipaco_eng),  # the full modular model (bigger)
    }
