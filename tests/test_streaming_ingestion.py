"""Tests for sharded, resumable streaming ingestion.

These exercise the logic without the network by injecting a synthetic ``source``
(a list of strings) and a fake tokenizer. They cover: round-robin sharding
partitions the stream disjointly, the tokenizing stream filters/limits/resumes, the
shard cache round-trips ``(docs, next_index)``, and -- the headline -- a two-phase
ingest resumes from a partial cache and reaches exactly the same docs as one full
ingest (no duplication, no loss).
"""

import torch

from opendipaco.data import (
    ShardCache,
    ingest_c4_shard,
    pack_sequences,
    shard_stream,
    stream_documents,
)


class _FakeTok:
    """Deterministic word-length tokenizer: each whitespace word -> its length."""
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [len(w) for w in text.split()]


def _source(n, start=0):
    # Doc i has (i % 5) + 1 words, each word distinct -> deterministic token tensors.
    return [" ".join(f"w{i}_{j}" for j in range(i % 5 + 1)) for i in range(start, start + n)]


# -- sharding ----------------------------------------------------------------


def test_shard_stream_partitions_disjointly():
    src = _source(30)
    shards = [list(shard_stream(src, num_shards=3, shard_id=s)) for s in range(3)]
    idx = [[gi for gi, _ in sh] for sh in shards]
    assert idx[0][:3] == [0, 3, 6] and idx[1][:3] == [1, 4, 7]  # round-robin
    allidx = sorted(i for s in idx for i in s)
    assert allidx == list(range(30))                             # cover everything
    assert len(set(allidx)) == 30                                # disjoint


def test_shard_stream_rejects_bad_shard():
    for bad in [(-1, 2), (2, 2), (0, 0)]:
        try:
            list(shard_stream(_source(3), shard_id=bad[0], num_shards=bad[1]))
            assert False, "expected ValueError"
        except ValueError:
            pass


# -- tokenizing stream -------------------------------------------------------


def test_stream_documents_tokenizes_filters_limits():
    tok = _FakeTok()
    docs = list(stream_documents(_source(20), tok, limit=5))
    assert len(docs) == 5
    for gi, t in docs:
        assert torch.is_tensor(t) and t.dtype == torch.long
        assert t[-1].item() == tok.eos_token_id  # eos appended


def test_stream_documents_min_tokens_filter():
    tok = _FakeTok()
    # min_doc_tokens=4 keeps only docs with >= 3 words (3 words + eos = 4 tokens).
    kept = [t for _, t in stream_documents(_source(40), tok, add_eos=True, min_doc_tokens=4)]
    assert kept and all(len(t) >= 4 for t in kept)


def test_stream_documents_start_index_resumes():
    tok = _FakeTok()
    full = [gi for gi, _ in stream_documents(_source(20), tok)]
    tail = [gi for gi, _ in stream_documents(_source(20), tok, start_index=7)]
    assert min(tail) >= 7 and tail == [gi for gi in full if gi >= 7]


# -- shard cache -------------------------------------------------------------


def test_shard_cache_roundtrips(tmp_path):
    cache = ShardCache(tmp_path, shard_id=1, num_shards=4)
    assert cache.load() == ([], 0)
    docs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    cache.save(docs, next_index=12)
    got, nxt = cache.load()
    assert nxt == 12 and len(got) == 2 and torch.equal(got[0], docs[0])
    assert "shard_00001_of_00004.pt" == cache.path.name


# -- resumable ingest --------------------------------------------------------


def test_ingest_resumes_and_matches_full(tmp_path):
    tok = _FakeTok()
    src = _source(60)
    full = ingest_c4_shard(tmp_path / "full", shard_id=0, num_shards=1,
                           target_docs=15, tokenizer=tok, source=src)
    assert len(full) == 15

    # Two-phase into a *separate* dir: ingest 6, then resume to 15 (small flush so the
    # resume crosses a flush boundary). Must equal the single-shot full ingest exactly.
    part = tmp_path / "resumed"
    first = ingest_c4_shard(part, shard_id=0, num_shards=1, target_docs=6,
                            tokenizer=tok, source=src, flush_every=2)
    assert len(first) == 6
    second = ingest_c4_shard(part, shard_id=0, num_shards=1, target_docs=15,
                             tokenizer=tok, source=src, flush_every=2)
    assert len(second) == 15
    assert all(torch.equal(a, b) for a, b in zip(second, full))  # no dup / no loss


def test_ingest_already_complete_is_noop(tmp_path):
    tok = _FakeTok()
    src = _source(30)
    ingest_c4_shard(tmp_path, shard_id=0, num_shards=1, target_docs=10, tokenizer=tok, source=src)
    # A second call with a smaller/equal target returns from cache without streaming.
    again = ingest_c4_shard(tmp_path, shard_id=0, num_shards=1, target_docs=10,
                            tokenizer=tok, source=None)  # source=None would stream C4 if it tried
    assert len(again) == 10


def test_ingest_shards_are_disjoint_and_cover(tmp_path):
    tok = _FakeTok()
    src = _source(40)
    s0 = ingest_c4_shard(tmp_path, shard_id=0, num_shards=2, target_docs=10, tokenizer=tok, source=src)
    s1 = ingest_c4_shard(tmp_path, shard_id=1, num_shards=2, target_docs=10, tokenizer=tok, source=src)
    whole = [t for _, t in stream_documents(src, tok)]
    # Shard 0 = even global indices, shard 1 = odd; together they reconstruct the corpus.
    assert all(torch.equal(a, b) for a, b in zip(s0, whole[0::2]))
    assert all(torch.equal(a, b) for a, b in zip(s1, whole[1::2]))


def test_streamed_docs_feed_the_corpus_pipeline(tmp_path):
    tok = _FakeTok()
    docs = ingest_c4_shard(tmp_path, shard_id=0, num_shards=1, target_docs=12,
                           tokenizer=tok, source=_source(40, start=0).__add__(
                               [" ".join(f"x{j}" for j in range(8)) for _ in range(40)]))
    seqs = pack_sequences(docs, seq_len=8)  # the downstream consumer accepts them
    assert seqs.ndim == 2 and seqs.shape[1] == 8
