import pytest
import torch

from opendipaco import (
    BackboneConfig,
    DiPaCoConfig,
    LocalBackend,
    assign_paths_by_loss,
    build_module_bank,
    fit_discriminative_router,
    gather_full_bank,
    path_losses,
    reshard_by_loss,
)
from opendipaco.data import ShardedCorpus
from opendipaco.model import build_path_model
from opendipaco.routing import BagOfTokensFeaturizer, DiscriminativeRouter, KMeansRouter


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    # Independent experts so an untrained bank gives distinguishable per-path
    # losses (the default identical init would make all paths score the same).
    return DiPaCoConfig(
        backbone=bb, level_sizes=[2, 2], sequence_length=32, identical_expert_init=False
    )


def _docs(seed=0, n_per=10):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(t * 12, t * 12 + 12, (40,), generator=g)
            for t in range(4) for _ in range(n_per)]


# --- #8: top-k / overlapping shards ----------------------------------------
def test_kmeans_predict_topk_nearest_first():
    feats = torch.randn(20, 16)
    router = KMeansRouter(4, seed=0).fit(feats)
    top = router.predict_topk(feats, 3)
    assert top.shape == (20, 3)
    assert torch.equal(top[:, 0], router.predict(feats))  # nearest centroid first


def test_discriminative_predict_topk_best_first():
    router = DiscriminativeRouter(8, 4)
    feats = torch.randn(15, 8)
    top = router.predict_topk(feats, 2)
    assert top.shape == (15, 2)
    assert torch.equal(top[:, 0], router.predict(feats))


def test_overlapping_shards_duplicate_documents():
    cfg = _cfg()
    docs = _docs()
    feat = BagOfTokensFeaturizer(48, feature_dim=32)
    router = KMeansRouter(cfg.num_paths, seed=0).fit(feat([d[:32] for d in docs]))
    overlap = ShardedCorpus.from_documents(docs, router, feat, cfg.num_paths, 32, top_k=2)
    disjoint = ShardedCorpus.from_documents(docs, router, feat, cfg.num_paths, 32, top_k=1)
    # Each document lands in exactly 2 shards -> total doc placements double.
    assert sum(overlap.doc_counts.values()) == 2 * len(docs)
    assert sum(disjoint.doc_counts.values()) == len(docs)


# --- #7: EM re-sharding (loss-based assignment) ----------------------------
def test_path_losses_shape_and_finiteness():
    cfg = _cfg()
    bank = build_module_bank(cfg)
    docs = _docs(n_per=4)
    losses = path_losses(docs, cfg, bank, seq_len=32)
    assert losses.shape == (len(docs), cfg.num_paths)
    assert torch.isfinite(losses).all()


def test_loss_assignment_matches_argmin():
    cfg = _cfg()
    bank = build_module_bank(cfg)
    docs = _docs(n_per=4)
    losses = path_losses(docs, cfg, bank, seq_len=32)
    top1 = assign_paths_by_loss(docs, cfg, bank, seq_len=32)
    assert torch.equal(top1, losses.argmin(dim=1))
    top2 = assign_paths_by_loss(docs, cfg, bank, seq_len=32, top_k=2)
    assert top2.shape == (len(docs), 2)
    assert torch.equal(top2[:, 0], losses.argmin(dim=1))  # best path first


def test_candidate_restriction():
    cfg = _cfg()
    bank = build_module_bank(cfg)
    docs = _docs(n_per=4)
    losses = path_losses(docs, cfg, bank, seq_len=32, candidates=[1])
    assert torch.isinf(losses[:, [0, 2, 3]]).all()   # non-candidates stay +inf
    assert torch.isfinite(losses[:, 1]).all()
    assign = assign_paths_by_loss(docs, cfg, bank, seq_len=32, candidates=[1])
    assert (assign == 1).all()


def test_reshard_by_loss_rebuilds_corpus():
    cfg = _cfg()
    bank = build_module_bank(cfg)
    docs = _docs(n_per=5)
    corpus = reshard_by_loss(docs, cfg, bank, seq_len=32)
    assert isinstance(corpus, ShardedCorpus)
    assert sum(corpus.doc_counts.values()) == len(docs)
    overlap = reshard_by_loss(docs, cfg, bank, seq_len=32, top_k=2)
    assert sum(overlap.doc_counts.values()) == 2 * len(docs)


# --- #3: token-based alpha weighting ---------------------------------------
def test_shard_weight_uses_token_counts():
    cfg = _cfg()
    long_doc = torch.zeros(100, dtype=torch.long)
    short_doc = torch.zeros(5, dtype=torch.long)   # fewer tokens than seq_len
    corpus = ShardedCorpus.from_assignments(
        [long_doc, short_doc], torch.tensor([0, 1]), num_paths=cfg.num_paths, seq_len=32
    )
    assert corpus.token_counts[0] == 100 and corpus.token_counts[1] == 5
    # The short shard packs to zero sequences but still carries a nonzero alpha.
    assert corpus.num_sequences(1) == 0
    assert corpus.shard_weight(1) > 0
    assert abs(corpus.shard_weight(0) - 100 / 105) < 1e-6
    assert abs(corpus.shard_weight(1) - 5 / 105) < 1e-6


# --- #4: full-bank requirement for eval/EM ---------------------------------
def test_missing_module_raises_helpful_error():
    cfg = _cfg()
    bank = build_module_bank(cfg)
    del bank["L0E0"]  # simulate a partial (per-rank) distributed bank
    with pytest.raises(KeyError, match="gather_full_bank"):
        build_path_model(cfg, (0, 0), bank)


def test_gather_full_bank_single_process_is_identity():
    cfg = _cfg()
    bank = build_module_bank(cfg)
    out = gather_full_bank(LocalBackend(cfg.build_topology()), bank, cfg)
    assert out is bank  # no distributed group -> unchanged


def test_fit_discriminative_router_uses_lowest_loss_labels():
    """The router is trained to predict each document's argmin-loss path."""
    cfg = _cfg()
    bank = build_module_bank(cfg)
    docs = _docs(n_per=6)
    feat = BagOfTokensFeaturizer(48, feature_dim=32)
    router = fit_discriminative_router(docs, cfg, bank, feat, seq_len=32, epochs=300)
    # Labels the router was fit on = the lowest-loss path per document.
    expected = assign_paths_by_loss(docs, cfg, bank, seq_len=32)
    # It should classify the held-out documents largely in line with those labels.
    feats = feat([d[:32] for d in docs])
    acc = (router.predict(feats) == expected).float().mean().item()
    assert acc > 0.5
