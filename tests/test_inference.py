import torch
import torch.nn.functional as F

from opendipaco import BackboneConfig, DiPaCoConfig, build_module_bank
from opendipaco.inference import (
    compose_path,
    config_path,
    routed_perplexity,
    routed_window_perplexity,
)
from opendipaco.routing import BagOfTokensFeaturizer, KMeansRouter


class ConstRouter:
    """Always routes to a fixed path index (for deterministic tests)."""

    def __init__(self, idx):
        self.idx = idx

    def predict(self, features):
        return torch.full((features.size(0),), self.idx, dtype=torch.long)


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    # Independent experts so untrained paths are distinguishable for routing tests
    # (the default identical init would make every path score identically).
    return DiPaCoConfig(
        backbone=bb, level_sizes=[2, 2], sequence_length=32, identical_expert_init=False
    )


def _single_path_token_ppl(config, bank, seqs, path_idx):
    model = compose_path(config, bank, config_path(config, path_idx)).eval()
    with torch.no_grad():
        logits, _ = model(seqs)
    pred = logits[:, :-1, :]
    tgt = seqs[:, 1:]
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="sum")
    return float(torch.exp(loss / tgt.numel()))


def test_window_eval_runs_and_is_finite():
    torch.manual_seed(0)
    cfg = _cfg()
    bank = build_module_bank(cfg)
    seqs = torch.randint(0, 48, (6, 32))
    feat = BagOfTokensFeaturizer(48, feature_dim=32)
    router = KMeansRouter(cfg.num_paths, seed=0).fit(feat([s[:16] for s in seqs]))
    ppl = routed_window_perplexity(cfg, bank, seqs, router, feat, window=8, prefix_len=0)
    assert ppl > 0 and torch.isfinite(torch.tensor(ppl))


def test_window_equals_single_path_when_router_constant_and_window_covers_seq():
    """One window + a constant path == evaluating that single path over the seq."""
    torch.manual_seed(1)
    cfg = _cfg()
    bank = build_module_bank(cfg)
    seqs = torch.randint(0, 48, (5, 32))
    feat = BagOfTokensFeaturizer(48, feature_dim=16)
    router = ConstRouter(2)
    windowed = routed_window_perplexity(cfg, bank, seqs, router, feat, window=32, prefix_len=0)
    reference = _single_path_token_ppl(cfg, bank, seqs, path_idx=2)
    assert abs(windowed - reference) < 1e-3


def test_routed_perplexity_is_token_level():
    """Constant routing + prefix_len=0 -> equals single-path token-level ppl."""
    torch.manual_seed(3)
    cfg = _cfg()
    bank = build_module_bank(cfg)
    seqs = torch.randint(0, 48, (5, 32))
    feat = BagOfTokensFeaturizer(48, feature_dim=16)
    got = routed_perplexity(cfg, bank, seqs, ConstRouter(1), feat, prefix_len=0)
    reference = _single_path_token_ppl(cfg, bank, seqs, path_idx=1)
    assert abs(got - reference) < 1e-3


def test_routed_perplexity_excludes_routing_prefix():
    """The first prefix_len tokens are not scored -> a larger prefix scores fewer
    tokens, and excluding none matches the all-token reference."""
    torch.manual_seed(5)
    cfg = _cfg()
    bank = build_module_bank(cfg)
    seqs = torch.randint(0, 48, (5, 32))
    feat = BagOfTokensFeaturizer(48, feature_dim=16)
    full = routed_perplexity(cfg, bank, seqs, ConstRouter(0), feat, prefix_len=0)
    excluded = routed_perplexity(cfg, bank, seqs, ConstRouter(0), feat, prefix_len=16)
    assert full > 0 and excluded > 0
    assert abs(full - excluded) > 1e-6   # different token sets -> different perplexity
    # Excluding the whole sequence leaves nothing to score.
    import math as _math
    assert _math.isnan(routed_perplexity(cfg, bank, seqs, ConstRouter(0), feat, prefix_len=32))


def test_window_reroutes_on_topic_shift():
    """A topic shift across >2 chunks makes a later window route to a new path,
    so windowed perplexity differs from routing once from the (topic-A) prefix."""
    torch.manual_seed(4)
    cfg = _cfg()
    bank = build_module_bank(cfg)
    g = torch.Generator().manual_seed(0)
    topic_a = lambda n: torch.randint(0, 12, (n,), generator=g)   # noqa: E731
    topic_b = lambda n: torch.randint(36, 48, (n,), generator=g)  # noqa: E731
    # 4 chunks of 8 tokens: A, A, B, B -> the last window re-routes from a B chunk.
    seqs = torch.stack(
        [torch.cat([topic_a(8), topic_a(8), topic_b(8), topic_b(8)]) for _ in range(4)]
    )
    feat = BagOfTokensFeaturizer(48, feature_dim=32)
    router = KMeansRouter(cfg.num_paths, seed=0).fit(
        feat([topic_a(16) for _ in range(8)] + [topic_b(16) for _ in range(8)])
    )
    once = routed_perplexity(cfg, bank, seqs, router, feat, prefix_len=8)  # routes from topic A
    windowed = routed_window_perplexity(cfg, bank, seqs, router, feat, window=8, prefix_len=8)
    assert once > 0 and windowed > 0
    assert windowed != once  # re-routing picks a different path for the topic-B tail
