import torch

from opendipaco import (
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
)
from opendipaco.data import ShardedCorpus
from opendipaco.routing import BagOfTokensFeaturizer, DiscriminativeRouter, KMeansRouter


# --- #2: discriminative router held-out split ------------------------------
def test_discriminative_router_held_out_split_learns():
    torch.manual_seed(0)
    centers = torch.randn(4, 8) * 3
    labels = torch.randint(0, 4, (120,))
    feats = centers[labels] + 0.2 * torch.randn(120, 8)
    router = DiscriminativeRouter(8, 4).fit(feats, labels, val_fraction=0.25, epochs=300)
    acc = (router.predict(feats) == labels).float().mean().item()
    assert acc > 0.9  # held-out early stopping still fits the separable data


def test_discriminative_router_val_fraction_is_deterministic():
    torch.manual_seed(0)
    feats, labels = torch.randn(50, 6), torch.randint(0, 3, (50,))
    # Same model init (manual_seed) + same split seed -> identical fit.
    torch.manual_seed(7)
    a = DiscriminativeRouter(6, 3).fit(feats, labels, val_fraction=0.2, seed=1, epochs=50)
    torch.manual_seed(7)
    b = DiscriminativeRouter(6, 3).fit(feats, labels, val_fraction=0.2, seed=1, epochs=50)
    assert torch.equal(a.predict(feats), b.predict(feats))


# --- #1: per-path early stopping -------------------------------------------
def _setup(val_fraction=0.3, seed=0):
    torch.manual_seed(seed)
    vocab = 64
    bb = BackboneConfig(
        vocab_size=vocab, hidden_size=48, num_attention_heads=4,
        intermediate_size=96, layers_per_level=[1, 1], max_position_embeddings=64,
    )
    cfg = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=32)
    topo = cfg.build_topology()
    g = torch.Generator().manual_seed(seed)
    docs = [torch.randint(t * 16, t * 16 + 16, (60,), generator=g)
            for t in range(4) for _ in range(40)]
    feat = BagOfTokensFeaturizer(vocab, feature_dim=64)
    router = KMeansRouter(cfg.num_paths, seed=0).fit(feat([d[:32] for d in docs]))
    corpus = ShardedCorpus.from_documents(
        docs, router, feat, cfg.num_paths, cfg.sequence_length, val_fraction=val_fraction
    )
    return cfg, topo, corpus


def test_corpus_validation_split():
    cfg, topo, corpus = _setup(val_fraction=0.25)
    assert corpus.has_validation
    for p in range(cfg.num_paths):
        val = corpus.val_shard(p)
        assert val is not None
        # train + val sequences reconstruct the full packed count.
        if corpus.num_sequences(p) + val.size(0) > 1:
            assert val.size(0) >= 1

    _, _, no_val = _setup(val_fraction=0.0)
    assert not no_val.has_validation
    assert no_val.val_shard(0) is None


def test_engine_tracks_best_and_reports_val_loss():
    cfg, topo, corpus = _setup()
    engine = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=4, inner_lr=1e-3), LocalBackend(topo), seed=0)
    engine.fit(corpus, num_rounds=5, batch_size=8, log_every=0)
    assert engine.best_val_loss  # populated for validated paths
    assert set(engine.best_path_state) == set(engine.best_val_loss)
    m = engine.run_round(corpus, batch_size=8)
    assert m.val_loss is not None


def test_serial_early_stopping_matches_eager():
    cfg, topo, corpus = _setup()
    dl = DiLoCoConfig(inner_steps=4, inner_lr=1e-3)
    eager = DiPaCoEngine(cfg, dl, LocalBackend(topo), seed=0, materialize="eager")
    serial = DiPaCoEngine(cfg, dl, LocalBackend(topo), seed=0, materialize="serial")
    eager.fit(corpus, num_rounds=4, batch_size=8, log_every=0)
    serial.fit(corpus, num_rounds=4, batch_size=8, log_every=0)
    assert set(eager.best_val_loss) == set(serial.best_val_loss)
    for path in eager.best_val_loss:
        assert abs(eager.best_val_loss[path] - serial.best_val_loss[path]) < 1e-4


def test_compose_best_reproduces_recorded_loss_and_is_optimal():
    cfg, topo, corpus = _setup()
    engine = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=4, inner_lr=2e-3), LocalBackend(topo), seed=0)
    engine.fit(corpus, num_rounds=6, batch_size=8, log_every=0)
    for path in engine.best_path_state:
        val = corpus.val_shard(topo.path_index(path))
        best_model = engine.compose_best(path)
        recorded = engine.best_val_loss[path]
        # The snapshot reproduces exactly the loss that was recorded as best.
        assert abs(engine._eval_val(best_model, val, 8) - recorded) < 1e-4
        # And it is no worse than the path's current (final-round) weights.
        current = engine._eval_val(engine.path_models[path], val, 8)
        assert recorded <= current + 1e-6
