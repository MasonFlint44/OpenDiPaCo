import pytest
import torch

from opendipaco import (
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from opendipaco.data import ShardedCorpus


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _docs(seed=0, n_per=8):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(n_per)]


def _corpus(cfg):
    docs = _docs()
    assign = torch.tensor([i % cfg.num_paths for i in range(len(docs))])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _engine(cfg, materialize="eager", seed=0):
    return DiPaCoEngine(
        cfg, DiLoCoConfig(inner_steps=4, inner_lr=1e-3),
        LocalBackend(cfg.build_topology()), seed=seed, materialize=materialize,
    )


def _banks_equal(a, b):
    for key in a:
        pa = dict(a[key].named_parameters())
        pb = dict(b[key].named_parameters())
        assert pa.keys() == pb.keys()
        for name in pa:
            if not torch.equal(pa[name], pb[name]):
                return False
    return True


def test_save_load_roundtrip_matches_bank(tmp_path):
    cfg = _cfg()
    corpus = _corpus(cfg)
    eng = _engine(cfg)
    eng.fit(corpus, num_rounds=3, batch_size=8, log_every=0)

    save_checkpoint(eng, tmp_path / "ckpt", corpus=corpus)

    fresh = _engine(cfg, seed=999)  # different seed -> different init
    assert not _banks_equal(eng.bank, fresh.bank)
    out = load_checkpoint(fresh, tmp_path / "ckpt")
    assert _banks_equal(eng.bank, fresh.bank)
    assert fresh._global_round == eng._global_round == 3
    assert "corpus" in out  # corpus was saved alongside


@pytest.mark.parametrize("materialize", ["eager", "serial"])
def test_resume_equals_uninterrupted(tmp_path, materialize):
    cfg = _cfg()
    corpus = _corpus(cfg)

    # Reference: 6 rounds straight through.
    ref = _engine(cfg, materialize=materialize)
    ref.fit(corpus, num_rounds=6, batch_size=8, log_every=0, total_rounds=6)

    # Interrupted: 3 rounds, checkpoint, fresh engine resumes for 3 more.
    a = _engine(cfg, materialize=materialize)
    a.fit(corpus, num_rounds=3, batch_size=8, log_every=0, total_rounds=6)
    save_checkpoint(a, tmp_path / "mid")

    b = _engine(cfg, materialize=materialize, seed=123)
    load_checkpoint(b, tmp_path / "mid")
    b.fit(corpus, num_rounds=3, batch_size=8, log_every=0)

    assert b.total_rounds == 6  # restored horizon -> same cosine schedule
    assert _banks_equal(ref.bank, b.bank)


def test_fingerprint_mismatch_raises(tmp_path):
    cfg = _cfg()
    corpus = _corpus(cfg)
    eng = _engine(cfg)
    eng.fit(corpus, num_rounds=1, batch_size=8, log_every=0)
    save_checkpoint(eng, tmp_path / "ckpt")

    # A different config -> different fingerprint -> strict load must refuse.
    other = DiPaCoConfig(backbone=cfg.backbone, level_sizes=[2, 2], sequence_length=32)
    bad = DiPaCoEngine(
        other, DiLoCoConfig(inner_steps=4), LocalBackend(other.build_topology()),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(bad, tmp_path / "ckpt")


def test_latest_checkpoint_picks_highest_round(tmp_path):
    cfg = _cfg()
    corpus = _corpus(cfg)
    eng = _engine(cfg)
    eng.fit(corpus, num_rounds=2, batch_size=8, log_every=0)
    save_checkpoint(eng, tmp_path / "round2")
    eng.fit(corpus, num_rounds=3, batch_size=8, log_every=0)
    save_checkpoint(eng, tmp_path / "round5")
    assert latest_checkpoint(tmp_path).endswith("round5")
