import torch

from opendipaco import (
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
)
from opendipaco.data import ShardedCorpus
from opendipaco.routing import BagOfTokensFeaturizer, KMeansRouter
from opendipaco.topology import embed_key, expert_key
from opendipaco.train.loop import _iter_batches


def test_iter_batches_without_replacement_within_epoch():
    shard = torch.arange(12).reshape(12, 1)
    gen = torch.Generator().manual_seed(0)
    batches = list(_iter_batches(shard, batch_size=4, steps=3, gen=gen))  # one full epoch
    assert len(batches) == 3
    assert all(b.shape[0] == 4 for b in batches)
    seen = torch.cat([b.reshape(-1) for b in batches]).tolist()
    assert sorted(seen) == list(range(12))  # every example exactly once per epoch


def test_iter_batches_continues_across_epoch_boundary():
    shard = torch.arange(6).reshape(6, 1)
    gen = torch.Generator().manual_seed(0)
    batches = list(_iter_batches(shard, batch_size=4, steps=5, gen=gen))
    assert len(batches) == 5  # keeps yielding past the epoch boundary
    assert all(b.shape[0] == 4 for b in batches)


def _setup(seed=0, embedding="shared"):
    torch.manual_seed(seed)
    vocab = 64
    bb = BackboneConfig(
        vocab_size=vocab, hidden_size=48, num_attention_heads=4,
        intermediate_size=96, layers_per_level=[1, 1], max_position_embeddings=64,
    )
    cfg = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=32, embedding=embedding)
    topo = cfg.build_topology()
    docs = []
    g = torch.Generator().manual_seed(seed)
    for t in range(4):
        lo = t * 16
        for _ in range(40):
            docs.append(torch.randint(lo, lo + 16, (50,), generator=g))
    feat = BagOfTokensFeaturizer(vocab, feature_dim=64)
    router = KMeansRouter(cfg.num_paths, seed=0).fit(feat([d[:32] for d in docs]))
    corpus = ShardedCorpus.from_documents(docs, router, feat, cfg.num_paths, cfg.sequence_length)
    return cfg, topo, corpus


def test_training_reduces_loss():
    cfg, topo, corpus = _setup()
    engine = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=5, inner_lr=1e-3), LocalBackend(topo), seed=0)
    hist = engine.fit(corpus, num_rounds=10, batch_size=8, log_every=0)
    assert hist[-1].inner_loss < hist[0].inner_loss


def test_serial_materialize_matches_eager():
    """Serial (one path at a time) must be numerically identical to eager."""
    cfg, topo, corpus = _setup()
    dl = DiLoCoConfig(inner_steps=5, inner_lr=1e-3)
    eager = DiPaCoEngine(cfg, dl, LocalBackend(topo), seed=0, materialize="eager")
    serial = DiPaCoEngine(cfg, dl, LocalBackend(topo), seed=0, materialize="serial")
    eager.fit(corpus, num_rounds=4, batch_size=8, log_every=0)
    serial.fit(corpus, num_rounds=4, batch_size=8, log_every=0)

    assert not serial.path_models   # serial holds no co-resident working models
    assert serial._opt_state        # but it does persist per-path optimizer state
    for key in eager.bank:
        for a, b in zip(eager.bank[key].parameters(), serial.bank[key].parameters()):
            assert torch.allclose(a, b, atol=1e-5), key


def test_shared_modules_stay_synced_across_paths():
    """After an outer step, every path's copy of a shared module must be identical."""
    cfg, topo, corpus = _setup()
    engine = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=3), LocalBackend(topo), seed=0)
    engine.run_round(corpus, batch_size=8)
    # embed is shared by all paths -> all working copies equal the global bank.
    bank_embed = next(engine.bank[embed_key()].parameters())
    for path, pm in engine.path_models.items():
        p = next(pm.modules_by_key()[embed_key()].parameters())
        assert torch.allclose(p, bank_embed)


def test_expert_only_updated_by_its_paths():
    """An expert's two working copies (paths (0,*)) must match the global bank."""
    cfg, topo, corpus = _setup()
    engine = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=3), LocalBackend(topo), seed=0)
    engine.run_round(corpus, batch_size=8)
    key = expert_key(0, 0)  # used by paths (0,0) and (0,1)
    bank_p = next(engine.bank[key].parameters())
    for path in [(0, 0), (0, 1)]:
        p = next(engine.path_models[path].modules_by_key()[key].parameters())
        assert torch.allclose(p, bank_p)


def test_private_embedding_diverges_across_paths():
    """Private modules are never averaged, so each path's embedding differs."""
    cfg, topo, corpus = _setup(embedding="private")
    engine = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=5, inner_lr=1e-3), LocalBackend(topo), seed=0)
    # Sanity: the topology really made the embedding private (one key per path).
    assert "embed" not in engine.bank
    assert "embed.p0" in engine.bank

    engine.run_round(corpus, batch_size=8)

    # Each path's private embedding equals its own bank entry (authoritative)...
    for path, pm in engine.path_models.items():
        idx = topo.path_index(path)
        local = next(pm.modules_by_key()[f"embed.p{idx}"].parameters())
        bank = next(engine.bank[f"embed.p{idx}"].parameters())
        assert torch.allclose(local, bank)
    # ...but different paths' embeddings have diverged (no cross-path averaging).
    e0 = next(engine.bank["embed.p0"].parameters())
    e3 = next(engine.bank["embed.p3"].parameters())
    assert not torch.allclose(e0, e3)


def test_private_embedding_excluded_from_outer_optimizer():
    from opendipaco.topology import is_private_key

    cfg, topo, corpus = _setup(embedding="private")
    engine = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=2), LocalBackend(topo), seed=0)
    outer_params = {id(p) for g in engine.outer_opt.param_groups for p in g["params"]}
    for key, mod in engine.bank.items():
        in_outer = any(id(p) in outer_params for p in mod.parameters())
        # Private modules must NOT be in the outer optimizer; shared ones must.
        assert in_outer == (not is_private_key(key))
