import torch

from opendipaco import (
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
    build_module_bank,
    warm_start_modules,
)
from opendipaco.optim import inner_lr_at


# --- #4: inner LR schedule -------------------------------------------------
def _cfg(**kw):
    return DiLoCoConfig(inner_lr=1.0, **kw)


def test_constant_schedule_is_flat():
    cfg = _cfg(inner_lr_schedule="constant")
    assert inner_lr_at(0, 100, cfg) == 1.0
    assert inner_lr_at(99, 100, cfg) == 1.0


def test_cosine_without_total_is_constant():
    cfg = _cfg(inner_lr_schedule="cosine")
    assert inner_lr_at(5, None, cfg) == 1.0


def test_cosine_decays_from_peak_to_floor():
    cfg = _cfg(inner_lr_schedule="cosine", inner_min_lr_ratio=0.1)
    start = inner_lr_at(0, 100, cfg)
    mid = inner_lr_at(50, 100, cfg)
    end = inner_lr_at(100, 100, cfg)
    assert abs(start - 1.0) < 1e-6
    assert abs(mid - 0.55) < 1e-6        # 0.1 + 0.9 * 0.5
    assert abs(end - 0.1) < 1e-6         # floor = peak * min_ratio
    assert start > mid > end


def test_warmup_then_decay():
    cfg = _cfg(inner_lr_schedule="cosine", inner_warmup_steps=10)
    assert abs(inner_lr_at(0, 100, cfg) - 0.1) < 1e-6     # 1/10 of peak
    assert abs(inner_lr_at(9, 100, cfg) - 1.0) < 1e-6     # end of warmup
    assert inner_lr_at(10, 100, cfg) <= 1.0               # cosine begins
    assert inner_lr_at(50, 100, cfg) > inner_lr_at(90, 100, cfg)


def test_schedule_drives_optimizer_lr():
    cfg, topo, _ = _toy()
    diloco = DiLoCoConfig(inner_steps=4, inner_lr=1e-3, inner_lr_schedule="cosine")
    engine = DiPaCoEngine(_full_cfg(), diloco, LocalBackend(topo), seed=0)
    engine.total_rounds = 5
    corpus = _toy()[2]
    engine.run_round(corpus, batch_size=8, round_idx=4)  # last round -> near the floor
    lr = engine.inner_opts[(0, 0)].param_groups[0]["lr"]
    assert lr < 1e-3   # decayed below the peak


def test_schedule_is_continuous_across_fit_calls():
    """The cosine must span the whole run, not restart each fit() call."""
    cfg, topo, corpus = _toy()
    diloco = DiLoCoConfig(inner_steps=4, inner_lr=1e-3, inner_lr_schedule="cosine", inner_min_lr_ratio=0.1)
    engine = DiPaCoEngine(_full_cfg(), diloco, LocalBackend(topo), seed=0)
    engine.total_rounds = 10  # full horizon

    engine.fit(corpus, num_rounds=5, batch_size=8, log_every=0)
    assert engine._global_round == 5
    assert engine.total_rounds == 10  # not reset by fit

    engine.fit(corpus, num_rounds=5, batch_size=8, log_every=0)
    assert engine._global_round == 10
    # By the end of the full run the LR sits at the cosine floor (not back at peak).
    lr = engine.inner_opts[(0, 0)].param_groups[0]["lr"]
    assert abs(lr - 1e-4) < 2e-4


def test_fit_does_not_reset_horizon():
    """A second fit() without an explicit horizon keeps the first one (no sawtooth)."""
    cfg, topo, corpus = _toy()
    engine = DiPaCoEngine(_full_cfg(), DiLoCoConfig(inner_steps=2), LocalBackend(topo), seed=0)
    engine.fit(corpus, num_rounds=3, batch_size=8, log_every=0)
    assert engine.total_rounds == 3
    engine.fit(corpus, num_rounds=3, batch_size=8, log_every=0)
    assert engine.total_rounds == 3        # horizon unchanged
    assert engine._global_round == 6       # but rounds keep accumulating


# --- #3: warm-start from a pretrained dense model --------------------------
def _pretrained(bb):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=bb.vocab_size, hidden_size=bb.hidden_size,
        intermediate_size=bb.intermediate_size, num_hidden_layers=bb.total_layers,
        num_attention_heads=bb.num_attention_heads, num_key_value_heads=bb.kv_heads(),
        max_position_embeddings=bb.max_position_embeddings,
        rms_norm_eps=bb.rms_norm_eps, tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg)


def _backbone():
    return BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )


def _full_cfg():
    return DiPaCoConfig(backbone=_backbone(), level_sizes=[2, 2], sequence_length=32)


def _toy(seed=0):
    from opendipaco.data import ShardedCorpus
    from opendipaco.routing import BagOfTokensFeaturizer, KMeansRouter

    cfg = _full_cfg()
    topo = cfg.build_topology()
    g = torch.Generator().manual_seed(seed)
    docs = [torch.randint(t * 12, t * 12 + 12, (40,), generator=g)
            for t in range(4) for _ in range(20)]
    feat = BagOfTokensFeaturizer(48, feature_dim=32)
    router = KMeansRouter(cfg.num_paths, seed=0).fit(feat([d[:32] for d in docs]))
    corpus = ShardedCorpus.from_documents(docs, router, feat, cfg.num_paths, cfg.sequence_length)
    return cfg, topo, corpus


def test_warm_start_copies_pretrained_weights():
    cfg = _full_cfg()
    topo = cfg.build_topology()
    model = _pretrained(cfg.backbone)
    bank = build_module_bank(cfg)
    warm_start_modules(bank, topo, model)

    assert torch.equal(bank["embed"].embed_tokens.weight, model.model.embed_tokens.weight)
    assert torch.equal(bank["head"].norm.weight, model.model.norm.weight)
    assert torch.equal(bank["head"].lm_head.weight, model.lm_head.weight)
    # Body block "L0E0" is layer offset 0 -> pretrained layer 0.
    got = bank["L0E0"].layers[0].self_attn.q_proj.weight
    want = model.model.layers[0].self_attn.q_proj.weight
    assert torch.equal(got, want)


def test_warm_start_makes_level_experts_identical():
    cfg = _full_cfg()
    topo = cfg.build_topology()
    bank = build_module_bank(cfg)
    warm_start_modules(bank, topo, _pretrained(cfg.backbone))
    # Both experts of level 0 copy the same pretrained layer -> identical at init.
    a = bank["L0E0"].layers[0].mlp.gate_proj.weight
    b = bank["L0E1"].layers[0].mlp.gate_proj.weight
    assert torch.equal(a, b)


def test_engine_init_from_warm_starts_path_models():
    cfg = _full_cfg()
    topo = cfg.build_topology()
    model = _pretrained(cfg.backbone)
    engine = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=1), LocalBackend(topo), init_from=model)
    pm = engine.path_models[(0, 0)]
    assert torch.equal(
        pm.modules_by_key()["embed"].embed_tokens.weight, model.model.embed_tokens.weight
    )
