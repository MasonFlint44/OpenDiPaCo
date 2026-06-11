import pytest
import torch

from opendipaco import BackboneConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus, split_documents, tokenize_documents, train_tokenizer
from opendipaco.routing import BagOfTokensFeaturizer, KMeansRouter


class FakeTokenizer:
    """A minimal whitespace tokenizer (hash words into a fixed vocab).

    Stands in for a HuggingFace tokenizer so the data-pipeline core can be
    tested without the network or the ``[data]`` extra.
    """

    def __init__(self, vocab_size=64, eos_token_id=1):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def encode(self, text, add_special_tokens=False):
        return [2 + (hash(w) % (self.vocab_size - 2)) for w in text.split()]


def test_tokenize_documents_shapes_and_dtype():
    tok = FakeTokenizer()
    docs = tokenize_documents(["hello world foo", "a b c d"], tok)
    assert len(docs) == 2
    for d in docs:
        assert d.dtype == torch.long
        assert d.ndim == 1
    # eos appended by default.
    assert docs[0][-1].item() == tok.eos_token_id
    assert docs[0].numel() == 4  # 3 words + eos


def test_tokenize_documents_no_eos_and_truncation():
    tok = FakeTokenizer()
    docs = tokenize_documents(
        ["w " * 50], tok, add_eos=False, max_doc_tokens=10
    )
    assert docs[0].numel() == 10
    assert docs[0][-1].item() != tok.eos_token_id  # no eos when add_eos=False


def test_tokenize_documents_drops_short():
    tok = FakeTokenizer()
    docs = tokenize_documents(["", "a b c"], tok, add_eos=False, min_doc_tokens=2)
    assert len(docs) == 1  # empty dropped, the 3-token doc kept


def test_tokenized_docs_feed_sharded_corpus():
    tok = FakeTokenizer(vocab_size=64)
    texts = [f"topic{t} " * 30 for t in range(8)]
    docs = tokenize_documents(texts, tok)
    bb = BackboneConfig(
        vocab_size=64, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    cfg = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)
    feat = BagOfTokensFeaturizer(64, feature_dim=16)
    router = KMeansRouter(cfg.num_paths, seed=0).fit(feat([d[:16] for d in docs]))
    corpus = ShardedCorpus.from_documents(docs, router, feat, cfg.num_paths, 16)
    assert isinstance(corpus, ShardedCorpus)
    assert sum(corpus.doc_counts.values()) == len(docs)


def _sample_texts(n=40):
    words = ["the", "quick", "brown", "fox", "hello", "world", "foo", "bar",
             "lorem", "ipsum", "dolor", "sit", "amet", "alpha", "beta", "gamma"]
    import random
    g = random.Random(0)
    return [" ".join(g.choice(words) for _ in range(40)) for _ in range(n)]


def test_train_tokenizer_unigram_round_trips_and_feeds_pipeline():
    texts = _sample_texts()
    tok = train_tokenizer(texts, vocab_size=120, model="unigram")
    assert tok.vocab_size <= 120 and tok.eos_token_id is not None
    docs = tokenize_documents(texts, tok)
    assert docs and all(d.dtype == torch.long and d.ndim == 1 for d in docs)
    assert all(int(d.max()) < tok.vocab_size for d in docs)         # ids within the vocab
    assert docs[0][-1].item() == tok.eos_token_id                   # eos appended
    # The trained vocab is what BackboneConfig.vocab_size should track.
    bb = BackboneConfig(vocab_size=tok.vocab_size, hidden_size=16, num_attention_heads=2,
                        intermediate_size=32, layers_per_level=[1, 1], max_position_embeddings=64)
    assert bb.vocab_size == tok.vocab_size


def test_train_tokenizer_bpe_works():
    tok = train_tokenizer(_sample_texts(), vocab_size=120, model="bpe")
    ids = tok.encode("hello world", add_special_tokens=False)
    assert ids and all(0 <= i < tok.vocab_size for i in ids)


def test_train_tokenizer_rejects_unknown_model():
    with pytest.raises(ValueError):
        train_tokenizer(_sample_texts(), model="wordpiece")


def test_split_documents_is_deterministic_and_partitions():
    docs = [torch.tensor([i]) for i in range(100)]
    tr, va, te = split_documents(docs, val_fraction=0.2, test_fraction=0.1, seed=0)
    assert (len(tr), len(va), len(te)) == (70, 20, 10)
    # A partition (disjoint, covers everything) and reproducible across calls.
    ids = lambda part: {int(d) for d in part}
    assert ids(tr) | ids(va) | ids(te) == set(range(100))
    assert ids(tr).isdisjoint(ids(va)) and ids(va).isdisjoint(ids(te))
    tr2, va2, te2 = split_documents(docs, val_fraction=0.2, test_fraction=0.1, seed=0)
    assert ids(te) == ids(te2) and ids(va) == ids(va2)
    # A different seed gives a different held-out test split.
    _, _, te3 = split_documents(docs, val_fraction=0.2, test_fraction=0.1, seed=1)
    assert ids(te) != ids(te3)
