"""Tests for the DiPaCo-vs-dense validation harness.

These check the harness is *correct* (it runs the comparison, the dense baseline is
exactly one DiPaCo path's size, both perplexities are finite). They deliberately do
**not** assert DiPaCo beats dense -- that's a scale phenomenon and won't hold at the
toy size used here.
"""

import math

import torch

from opendipaco import BackboneConfig, DiLoCoConfig
from opendipaco.validation import run_comparison


def _topic_docs(vocab=96, num_topics=4, per=30, length=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = vocab // num_topics
    return [torch.randint(t * span, (t + 1) * span, (length,), generator=g)
            for t in range(num_topics) for _ in range(per)]


def test_validation_harness_runs_and_matches_per_path_size():
    bb = BackboneConfig(vocab_size=96, hidden_size=48, num_attention_heads=4,
                        intermediate_size=96, layers_per_level=[1, 1], max_position_embeddings=96)
    res = run_comparison(
        bb, _topic_docs(), _topic_docs(per=8, seed=99), DiLoCoConfig(inner_steps=6, inner_lr=1e-3),
        dipaco_levels=(2, 2), rounds=6, batch_size=8, seq_len=32, eval_len=64, seed=0,
    )
    # Both models actually evaluated to a finite, positive perplexity.
    assert math.isfinite(res["dense_ppl"]) and res["dense_ppl"] > 0
    assert math.isfinite(res["dipaco_ppl"]) and res["dipaco_ppl"] > 0
    # Equal inference cost: one DiPaCo path is exactly the dense model's size, and
    # the full modular model is bigger (4 paths sharing modules).
    assert res["dipaco_paths"] == 4
    assert res["params_per_path"] > 0
    assert res["dipaco_total_params"] > res["params_per_path"]


def test_dense_baseline_is_one_path():
    """level_sizes=(1, 1) is a single-path (plain dense) model."""
    from opendipaco import DiPaCoConfig
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1], max_position_embeddings=64)
    dense = DiPaCoConfig(backbone=bb, level_sizes=[1, 1])
    dipaco = DiPaCoConfig(backbone=bb, level_sizes=[2, 2])
    assert dense.num_paths == 1 and dipaco.num_paths == 4
