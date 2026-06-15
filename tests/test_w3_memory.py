"""Tests for the W3a VRAM profiler (docs/w3-vram-design.md D1).

The parameter/optimizer terms are exact (counted on the meta device); the
activation term is a coarse estimate that the levers should visibly reduce.
"""

import torch
from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.train.memory import _count_params, fits, measure_peak, vram_breakdown


def _cfg(vocab=2000, hidden=128, layers=2, levels=2):
    bb = BackboneConfig(vocab_size=vocab, hidden_size=hidden, num_attention_heads=4,
                        intermediate_size=256, layers_per_level=[layers] * levels,
                        max_position_embeddings=512)
    return DiPaCoConfig(backbone=bb, level_sizes=[2] * levels, sequence_length=128)


def test_param_counts_are_exact_and_categorized():
    """Counted on meta (no allocation); embed/head = vocab x hidden; private split
    follows the config's embedding/head sharing."""
    cfg = _cfg(vocab=2000, hidden=128)
    total, private, embed, head = _count_params(cfg)
    assert embed == 2000 * 128                              # vocab x hidden exactly
    assert 2000 * 128 <= head <= 2000 * 128 + 128          # + a final norm
    assert total > embed and total > 0
    assert private == 0                                     # default embedding/head = shared
    # Private when the config marks them so.
    bb = BackboneConfig(vocab_size=2000, hidden_size=128, num_attention_heads=4,
                        intermediate_size=256, layers_per_level=[2, 2],
                        max_position_embeddings=512)
    priv_cfg = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=128,
                            embedding="private", head="private")
    _t, p, _e, _h = _count_params(priv_cfg)
    assert p >= 2 * 2000 * 128                              # embed + head now private


def test_breakdown_terms_are_exact():
    cfg = _cfg()
    total = _count_params(cfg)[0]
    b = vram_breakdown(cfg, batch_size=8, seq_len=128)
    assert b["params"] == total * 4                         # fp32 master
    assert b["adam"] == 2 * total * 4                       # m, v
    assert b["grads"] == total * 4
    assert b["total"] == (b["params"] + b["global"] + b["adam"]
                          + b["grads"] + b["activations"])


def test_levers_reduce_the_estimate():
    cfg = _cfg()
    base = vram_breakdown(cfg, batch_size=8, seq_len=128)
    # autocast halves activation bytes; checkpointing cuts the block term; chunked
    # logits drops the [tokens, vocab] term entirely.
    ac = vram_breakdown(cfg, batch_size=8, seq_len=128, autocast=True)
    assert ac["activations"] < base["activations"]
    ckpt = vram_breakdown(cfg, batch_size=8, seq_len=128, autocast=True, checkpoint=True)
    assert ckpt["activations"] < ac["activations"]
    chunk = vram_breakdown(cfg, batch_size=8, seq_len=128, autocast=True,
                           checkpoint=True, chunked_logits=True)
    assert chunk["logits"] == 0 and chunk["activations"] < ckpt["activations"]
    assert chunk["total"] < base["total"]


def test_fits_and_measure_peak_cpu():
    cfg = _cfg()
    b = vram_breakdown(cfg, batch_size=8, seq_len=128)
    assert fits(b, 10 ** 12) and not fits(b, 1)
    # measure_peak is CUDA-only; on CPU it returns None (use the estimate).
    assert measure_peak(cfg, DiLoCoConfig(inner_steps=2), device="cpu",
                        batch_size=8, seq_len=128) is None


def test_measure_peak_round_construction_runs_on_cpu():
    """measure_peak's real round is CUDA-gated, so CI never exercises its shard
    shape. Run the identical construction on CPU to prove the seq_len-row shard
    trains a path without error (a wrong shape would only fail on a GPU)."""
    from opendipaco.backend import LocalBackend
    from opendipaco.schedule import AsyncScheduler
    from opendipaco.train.loop import DiPaCoEngine

    cfg = _cfg()
    seq_len = cfg.sequence_length
    engine = DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=2), LocalBackend(cfg.build_topology()),
                          device="cpu", seed=0, materialize="serial")
    worker = AsyncScheduler(engine, num_workers=1)
    path = cfg.build_topology().path_from_index(0)
    shard = torch.randint(0, cfg.backbone.vocab_size, (8, seq_len))   # seq_len rows
    contrib = worker._train_path(path, shard, 4, 0)
    assert contrib is not None and not contrib.empty
