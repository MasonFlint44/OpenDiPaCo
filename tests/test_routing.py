import torch

from opendipaco import BackboneConfig, DiPaCoConfig, build_module_bank
from opendipaco.routing import (
    BagOfTokensFeaturizer,
    EmbeddingFeaturizer,
    HFEncoderFeaturizer,
    KMeansRouter,
    ModelFeaturizer,
)


def _bank_and_cfg(identical_expert_init=True):
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    cfg = DiPaCoConfig(
        backbone=bb, level_sizes=[2, 2], sequence_length=32,
        identical_expert_init=identical_expert_init,
    )
    torch.manual_seed(0)
    return build_module_bank(cfg), cfg


def test_model_featurizer_shapes_and_determinism():
    bank, cfg = _bank_and_cfg()
    feat = ModelFeaturizer(bank, cfg)
    assert feat.feature_dim == 32
    prefixes = [torch.randint(0, 48, (k,)) for k in (5, 12, 1)]
    a, b = feat(prefixes), feat(prefixes)
    assert a.shape == (3, 32)
    assert torch.allclose(a, b)
    assert torch.allclose(a.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_model_featurizer_masking_invariance():
    bank, cfg = _bank_and_cfg()
    feat = ModelFeaturizer(bank, cfg, normalize=False)
    short = torch.randint(0, 48, (4,))
    long = torch.randint(0, 48, (11,))
    assert torch.allclose(feat([short, long])[0], feat([short])[0], atol=1e-4)


def test_model_featurizer_path_neutral_at_init():
    """Identical-expert init -> any reference path yields the same features."""
    bank, cfg = _bank_and_cfg(identical_expert_init=True)
    prefixes = [torch.randint(0, 48, (8,)) for _ in range(4)]
    f00 = ModelFeaturizer(bank, cfg, reference_path=(0, 0))(prefixes)
    f11 = ModelFeaturizer(bank, cfg, reference_path=(1, 1))(prefixes)
    assert torch.allclose(f00, f11, atol=1e-5)
    # With independent experts the reference path matters.
    bank2, cfg2 = _bank_and_cfg(identical_expert_init=False)
    g00 = ModelFeaturizer(bank2, cfg2, reference_path=(0, 0))(prefixes)
    g11 = ModelFeaturizer(bank2, cfg2, reference_path=(1, 1))(prefixes)
    assert not torch.allclose(g00, g11)


def test_model_featurizer_distinguishes_inputs():
    bank, cfg = _bank_and_cfg()
    feat = ModelFeaturizer(bank, cfg)
    a = feat([torch.zeros(8, dtype=torch.long)])
    b = feat([torch.full((8,), 30, dtype=torch.long)])
    assert not torch.allclose(a, b)


def _topic_docs(vocab=64, topics=4, per_topic=30, length=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = vocab // topics
    docs, labels = [], []
    for t in range(topics):
        for _ in range(per_topic):
            docs.append(torch.randint(t * span, (t + 1) * span, (length,), generator=g))
            labels.append(t)
    return docs, torch.tensor(labels)


def test_embedding_featurizer_shapes_and_determinism():
    torch.manual_seed(0)
    emb = torch.nn.Embedding(64, 32)
    feat = EmbeddingFeaturizer(emb)
    prefixes = [torch.randint(0, 64, (k,)) for k in (5, 12, 1)]
    a = feat(prefixes)
    b = feat(prefixes)
    assert a.shape == (3, 32)
    assert torch.allclose(a, b)  # frozen + deterministic
    # normalized by default -> unit rows
    assert torch.allclose(a.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_embedding_featurizer_masking_invariance():
    """A prefix's pooled feature must not depend on batch right-padding."""
    emb = torch.nn.Embedding(64, 16)
    short = torch.randint(0, 64, (4,))
    long = torch.randint(0, 64, (11,))
    feat = EmbeddingFeaturizer(emb, normalize=False)
    batched = feat([short, long])           # short gets padded to length 11
    alone = feat([short])                   # no padding
    assert torch.allclose(batched[0], alone[0], atol=1e-5)


def test_embedding_featurizer_recovers_topics():
    docs, labels = _topic_docs(seed=1)
    emb = torch.nn.Embedding(64, 48)
    feats = EmbeddingFeaturizer(emb)(docs)
    router = KMeansRouter(num_paths=4, seed=0).fit(feats)
    clusters = router.predict(feats)
    # Each topic should map predominantly to a single cluster (high purity).
    for t in range(4):
        topic_clusters = clusters[labels == t]
        majority = topic_clusters.bincount().max().item()
        assert majority / len(topic_clusters) > 0.6


def test_hf_encoder_featurizer_without_network():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, intermediate_size=64, max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    model = LlamaModel(cfg)
    feat = HFEncoderFeaturizer(model)
    assert feat.feature_dim == 32
    prefixes = [torch.randint(0, 64, (k,)) for k in (6, 10, 3)]
    out = feat(prefixes)
    assert out.shape == (3, 32)
    assert not out.requires_grad
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-4)


def test_hf_encoder_masking_invariance():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, intermediate_size=64, max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    model = LlamaModel(cfg)
    feat = HFEncoderFeaturizer(model, normalize=False)
    short = torch.randint(0, 64, (4,))
    long = torch.randint(0, 64, (10,))
    batched = feat([short, long])
    alone = feat([short])
    # Causal attention + masked mean -> the short row is unaffected by padding.
    assert torch.allclose(batched[0], alone[0], atol=1e-4)


def test_bag_of_tokens_still_works():
    docs, _ = _topic_docs(seed=2)
    feats = BagOfTokensFeaturizer(64, feature_dim=32)(docs)
    assert feats.shape == (len(docs), 32)
