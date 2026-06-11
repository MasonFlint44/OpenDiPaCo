"""Single-process DiPaCo simulation on synthetic data.

Runs all paths in one process (``LocalBackend``) -- the reference setup for
validating the method end to end. Demonstrates the full flow: featurize ->
k-means router -> shard -> DiPaCo training -> discriminative router -> routed
evaluation.

    python examples/train_synthetic.py
"""

from __future__ import annotations

import random

import torch

from opendipaco import (
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    fit_discriminative_router,
)
from opendipaco.backend import LocalBackend
from opendipaco.data import ShardedCorpus, pack_sequences
from opendipaco.inference import routed_perplexity, routed_window_perplexity
from opendipaco.routing import EmbeddingFeaturizer, KMeansRouter


def make_topic_docs(vocab_size, num_topics=4, docs_per_topic=50, length=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = vocab_size // num_topics
    docs = []
    for t in range(num_topics):
        lo = t * span
        for _ in range(docs_per_topic):
            docs.append(torch.randint(lo, lo + span, (length,), generator=g))
    return docs


def main():
    torch.manual_seed(0)
    vocab = 128
    bb = BackboneConfig(
        vocab_size=vocab, hidden_size=64, num_attention_heads=4,
        intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64,
    )
    config = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=32)
    topo = config.build_topology()

    docs = make_topic_docs(vocab)
    random.Random(123).shuffle(docs)  # mix topics before splitting (docs are topic-ordered)
    # Reserve a held-out subset for the discriminative router (the paper's "second,
    # much smaller part"); paths are trained only on the rest.
    holdout = len(docs) // 5
    router_docs, train_docs = docs[:holdout], docs[holdout:]

    # Learned featurizer: mean-pool a token-embedding table over the prefix.
    # In a real run, pass a *pretrained* embedding (or use HFEncoderFeaturizer
    # with a frozen model that shares your tokenizer).
    feat = EmbeddingFeaturizer(torch.nn.Embedding(vocab, 64))

    # 1) Generative router builds the shards (with a per-path validation split).
    kmeans = KMeansRouter(config.num_paths, seed=0).fit(feat([d[:32] for d in train_docs]))
    corpus = ShardedCorpus.from_documents(
        train_docs, kmeans, feat, config.num_paths, config.sequence_length, val_fraction=0.1
    )
    print("shard sizes:", {p: corpus.num_sequences(p) for p in range(config.num_paths)})

    # 2) Train DiPaCo (each path's best shard-validation checkpoint is kept).
    engine = DiPaCoEngine(config, DiLoCoConfig(inner_steps=8, inner_lr=1e-3), LocalBackend(topo), seed=0)
    engine.fit(corpus, num_rounds=20, batch_size=8, log_every=5)
    print("best shard-val loss:", {p: round(engine.best_val_loss[(p // 2, p % 2)], 3)
                                    for p in range(config.num_paths)})

    # 3) Discriminative router for test time: predict each held-out document's
    #    lowest-loss path (the paper's argmax-likelihood label / amortized E-step).
    disc = fit_discriminative_router(
        router_docs, config, engine.global_modules(), feat, config.sequence_length,
        val_fraction=0.2,
    )

    # 4) Evaluate with each path's early-stopped checkpoint (compose_best):
    #    route once per sequence vs. re-route every W tokens. The first 32 tokens
    #    (the routing prefix) are excluded from perplexity, per the paper; eval at
    #    length 64 so there are scored tokens after the prefix.
    eval_docs = make_topic_docs(vocab, docs_per_topic=10, seed=99)
    eval_seqs = pack_sequences(eval_docs, 64)
    bank = engine.global_modules()
    ppl_once = routed_perplexity(config, bank, eval_seqs, disc, feat, compose_fn=engine.compose_best)
    ppl_win = routed_window_perplexity(config, bank, eval_seqs, disc, feat, window=16,
                                       compose_fn=engine.compose_best)
    print(f"routed perplexity  (once per sequence): {ppl_once:.2f}")
    print(f"routed perplexity  (re-route every 16): {ppl_win:.2f}")


if __name__ == "__main__":
    main()
