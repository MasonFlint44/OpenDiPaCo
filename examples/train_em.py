"""DiPaCo training as EM, with overlapping shards.

Demonstrates the paper's coordinate-descent view of routing:

* **init**   -- generative (k-means) router builds the initial *overlapping*
  (top-2) shards;
* **M-step** -- train the paths on the current shards;
* **E-step** -- re-assign each document to its lowest-loss path(s) and rebuild
  the shards (``reshard_by_loss``);
* repeat, then fit a discriminative router on the final assignment for cheap
  test-time routing.

    python examples/train_em.py
"""

from __future__ import annotations

import random

import torch

from opendipaco import (
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
    fit_discriminative_router,
    reshard_by_loss,
)
from opendipaco.data import ShardedCorpus, pack_sequences
from opendipaco.inference import routed_perplexity
from opendipaco.routing import KMeansRouter, ModelFeaturizer


def make_topic_docs(vocab, num_topics=4, docs_per_topic=50, length=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = vocab // num_topics
    return [
        torch.randint(t * span, (t + 1) * span, (length,), generator=g)
        for t in range(num_topics)
        for _ in range(docs_per_topic)
    ]


def main():
    torch.manual_seed(0)
    vocab, seq_len = 128, 32
    bb = BackboneConfig(
        vocab_size=vocab, hidden_size=64, num_attention_heads=4,
        intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64,
    )
    config = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=seq_len)
    topo = config.build_topology()

    docs = make_topic_docs(vocab)
    random.Random(123).shuffle(docs)  # mix topics before splitting (docs are topic-ordered)
    # Reserve a held-out subset for the discriminative router (paper's "second part").
    holdout = len(docs) // 5
    router_docs, train_docs = docs[:holdout], docs[holdout:]

    engine = DiPaCoEngine(config, DiLoCoConfig(inner_steps=8, inner_lr=1e-3), LocalBackend(topo), seed=0)

    # Route on features from the DiPaCo model itself (the paper's z). The featurizer
    # aliases the engine's bank, so features co-evolve with training: at init they
    # are the (identical) pretrained-style features used for the first k-means
    # sharding, and after training the discriminative router sees trained features.
    feat = ModelFeaturizer(engine.global_modules(), config)

    # init: k-means router -> overlapping (top-2) shards.
    kmeans = KMeansRouter(config.num_paths, seed=0).fit(feat([d[:32] for d in train_docs]))
    corpus = ShardedCorpus.from_documents(train_docs, kmeans, feat, config.num_paths, seq_len, top_k=2)
    print("initial (top-2) shard sizes:", {p: corpus.num_sequences(p) for p in range(config.num_paths)})

    em_rounds, rounds_per_em = 3, 10
    engine.total_rounds = em_rounds * rounds_per_em  # one cosine over the whole EM run

    # Alternate M-step (train) and E-step (reshard by loss).
    for em_round in range(em_rounds):
        engine.fit(corpus, num_rounds=rounds_per_em, batch_size=8, log_every=0)
        corpus = reshard_by_loss(train_docs, config, engine.global_modules(), seq_len, top_k=2)
        sizes = {p: corpus.num_sequences(p) for p in range(config.num_paths)}
        print(f"[EM round {em_round}] re-sharded (top-2) sizes: {sizes}")

    # Discriminative router: predict each held-out doc's lowest-loss path (amortized E-step).
    disc = fit_discriminative_router(router_docs, config, engine.global_modules(), feat, seq_len)

    # Eval at length 64 so there are tokens to score after the excluded 32-token prefix.
    eval_seqs = pack_sequences(make_topic_docs(vocab, docs_per_topic=10, seed=99), 64)
    ppl = routed_perplexity(config, engine.global_modules(), eval_seqs, disc, feat)
    print(f"routed eval perplexity (discriminative router on EM labels): {ppl:.2f}")


if __name__ == "__main__":
    main()
