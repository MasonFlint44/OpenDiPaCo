import torch

from opendipaco import BackboneConfig, DiPaCoConfig, build_module_bank
from opendipaco.data import ShardedCorpus, chunk_documents, pack_sequences


def _cfg(identical_expert_init=True):
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[2, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(
        backbone=bb, level_sizes=[3, 2], sequence_length=16,
        identical_expert_init=identical_expert_init,
    )


# --- A: identical expert initialisation ------------------------------------
def test_experts_identical_by_default():
    torch.manual_seed(0)
    bank = build_module_bank(_cfg(identical_expert_init=True))
    # All experts of level 0 start identical (the paper's theta-bar).
    p0 = next(bank["L0E0"].layers[0].self_attn.q_proj.parameters())
    for e in (1, 2):
        pe = next(bank[f"L0E{e}"].layers[0].self_attn.q_proj.parameters())
        assert torch.equal(p0, pe)
    # Different levels are still independent modules (different shapes/positions).
    assert "L1E0" in bank


def test_experts_independent_when_disabled():
    torch.manual_seed(0)
    bank = build_module_bank(_cfg(identical_expert_init=False))
    p0 = next(bank["L0E0"].layers[0].self_attn.q_proj.parameters())
    p1 = next(bank["L0E1"].layers[0].self_attn.q_proj.parameters())
    assert not torch.equal(p0, p1)


def test_private_copies_identical_by_default():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1], max_position_embeddings=64)
    cfg = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16, embedding="private")
    torch.manual_seed(0)
    bank = build_module_bank(cfg)
    e0 = bank["embed.p0"].embed_tokens.weight
    for p in range(1, cfg.num_paths):
        assert torch.equal(e0, bank[f"embed.p{p}"].embed_tokens.weight)


# --- B: document-as-sequence packing ---------------------------------------
def test_chunk_documents_no_cross_document_windows():
    # Two documents with disjoint token ranges; a window must stay within one doc.
    a = torch.zeros(20, dtype=torch.long)        # doc A: all 0s, 20 tokens
    b = torch.ones(20, dtype=torch.long)         # doc B: all 1s, 20 tokens
    chunks = chunk_documents([a, b], seq_len=8)
    # 20 // 8 = 2 windows each -> 4 total, none mixing 0s and 1s.
    assert chunks.shape == (4, 8)
    for row in chunks:
        assert row.unique().numel() == 1  # pure single-document window

    packed = pack_sequences([a, b], seq_len=8)   # packing CAN mix the two docs
    assert any(row.unique().numel() > 1 for row in packed)


def test_document_pack_mode_in_corpus():
    docs = [torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)]
    assign = torch.tensor([0, 0])  # both to path 0
    corpus = ShardedCorpus.from_assignments(docs, assign, num_paths=2, seq_len=8, pack_mode="document")
    for row in corpus.shard(0):
        assert row.unique().numel() == 1  # document-coherent sequences


def test_invalid_pack_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="pack_mode"):
        ShardedCorpus.from_assignments([torch.zeros(8, dtype=torch.long)], torch.tensor([0]),
                                       num_paths=1, seq_len=8, pack_mode="nope")
