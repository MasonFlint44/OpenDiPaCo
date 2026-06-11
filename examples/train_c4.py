"""End-to-end DiPaCo on real text (C4), with checkpoint + resume.

Wires the real-corpus pipeline to the engine and demonstrates crash-safe
training:

    C4 (streamed, tokenized, cached) -> featurize -> k-means router -> shards
        -> train a few rounds -> save_checkpoint
        -> (new engine) load_checkpoint -> train a few more rounds

If the ``[data]`` extra or the C4 download is unavailable, it falls back to a
small synthetic topical corpus through the *same* code path, so the example
always runs offline.

    pip install -e ".[data]"      # for the real C4 path
    python examples/train_c4.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from opendipaco import (
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
    load_checkpoint,
    save_checkpoint,
)
from opendipaco.data import ShardedCorpus, load_c4_documents, pack_sequences
from opendipaco.inference import routed_perplexity
from opendipaco.routing import ModelFeaturizer, KMeansRouter


def synthetic_text_docs(vocab, num_topics=4, docs_per_topic=40, length=80, seed=0):
    """Offline fallback: topical token streams (no tokenizer/network needed)."""
    g = torch.Generator().manual_seed(seed)
    span = vocab // num_topics
    return [
        torch.randint(t * span, (t + 1) * span, (length,), generator=g)
        for t in range(num_topics)
        for _ in range(docs_per_topic)
    ]


def get_documents(fallback_vocab, cache_dir):
    """Real C4 if available, else a synthetic fallback (same downstream path).

    Returns ``(docs, vocab_size)``. For real C4 the vocab is the tokenizer's, so
    the backbone embedding is sized to match the token ids; the synthetic
    fallback uses a small ``fallback_vocab``.
    """
    try:
        from opendipaco.data import load_tokenizer

        # A pretrained ~32k vocab is a convenient proxy. For a closer match to the
        # paper (which trains its own 32k SentencePiece on the data), instead do:
        #   sample = [r["text"] for r in itertools.islice(stream, 50_000)]
        #   tok = train_tokenizer(sample, vocab_size=32000)   # from opendipaco.data
        tok = load_tokenizer("t5-base")
        docs = load_c4_documents(
            num_documents=2000,
            tokenizer=tok,
            max_doc_tokens=128,
            cache_path=Path(cache_dir) / "c4_cache.pt",
        )
        print(f"loaded {len(docs)} C4 documents (vocab {tok.vocab_size})")
        return docs, tok.vocab_size
    except Exception as e:  # ImportError (no extra) or any download/network error
        print(f"C4 unavailable ({type(e).__name__}); using synthetic topical corpus")
        return synthetic_text_docs(fallback_vocab), fallback_vocab


def main():
    torch.manual_seed(0)
    seq_len = 32

    with tempfile.TemporaryDirectory() as workdir:
        all_docs, vocab = get_documents(128, workdir)
        # Hold out a slice for evaluation.
        n_eval = max(1, len(all_docs) // 10)
        docs, eval_docs = all_docs[:-n_eval], all_docs[-n_eval:]
        bb = BackboneConfig(
            vocab_size=vocab, hidden_size=64, num_attention_heads=4,
            intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64,
        )
        config = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=seq_len)
        topo = config.build_topology()

        engine = DiPaCoEngine(
            config, DiLoCoConfig(inner_steps=8, inner_lr=1e-3), LocalBackend(topo), seed=0
        )
        feat = ModelFeaturizer(engine.global_modules(), config)
        kmeans = KMeansRouter(config.num_paths, seed=0).fit(feat([d[:32] for d in docs]))
        corpus = ShardedCorpus.from_documents(docs, kmeans, feat, config.num_paths, seq_len)
        print("shard sizes:", {p: corpus.num_sequences(p) for p in range(config.num_paths)})

        # Plan the full horizon up front so the cosine LR schedule spans the whole
        # run even though we stop and resume in the middle.
        engine.total_rounds = 12

        # Phase 1: train 6 rounds, then checkpoint (corpus saved too, so resume
        # needs no re-routing).
        engine.fit(corpus, num_rounds=6, batch_size=8, log_every=2)
        ckpt = Path(workdir) / "ckpt_round6"
        save_checkpoint(engine, ckpt, corpus=corpus)
        print(f"checkpointed at round {engine._global_round} -> {ckpt}")

        # Phase 2: a brand-new engine resumes from the checkpoint and finishes.
        resumed = DiPaCoEngine(
            config, DiLoCoConfig(inner_steps=8, inner_lr=1e-3), LocalBackend(topo), seed=7
        )
        out = load_checkpoint(resumed, ckpt)
        corpus = out["corpus"]
        print(f"resumed at round {resumed._global_round} (horizon {resumed.total_rounds})")
        resumed.fit(corpus, num_rounds=6, batch_size=8, log_every=2)

        # Eval at length 64 so there are tokens to score after the 32-token
        # routing prefix the perplexity excludes.
        eval_seqs = pack_sequences(eval_docs, 64)
        ppl = routed_perplexity(config, resumed.global_modules(), eval_seqs, kmeans, feat)
        print(f"routed eval perplexity after resume: {ppl:.2f}")


if __name__ == "__main__":
    main()
