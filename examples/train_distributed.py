"""Distributed DiPaCo: one process (rank) per path.

Run with torchrun, using one process per path. For a 2x2 topology (4 paths)::

    torchrun --nproc_per_node=4 examples/train_distributed.py

Each rank trains a single path and synchronizes shared modules with the other
ranks via ``torch.distributed`` process subgroups. The data here is synthetic so
the example is self-contained; swap in a real tokenized corpus for actual runs.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig, DiPaCoEngine
from opendipaco.backend import TorchDistBackend
from opendipaco.data import ShardedCorpus
from opendipaco.routing import BagOfTokensFeaturizer, KMeansRouter


def build_synthetic_corpus(config, vocab_size, seq_len):
    # Four token "topics" so coarse routing has real structure.
    docs = []
    g = torch.Generator().manual_seed(0)
    for topic in range(4):
        lo = topic * (vocab_size // 4)
        hi = lo + (vocab_size // 4)
        for _ in range(40):
            docs.append(torch.randint(lo, hi, (60,), generator=g))
    feat = BagOfTokensFeaturizer(vocab_size, feature_dim=64)
    prefixes = [d[:32] for d in docs]
    router = KMeansRouter(config.num_paths, seed=0).fit(feat(prefixes))
    return ShardedCorpus.from_documents(docs, router, feat, config.num_paths, seq_len)


def main():
    backend_name = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend_name)
    rank = dist.get_rank()

    vocab = 64
    bb = BackboneConfig(
        vocab_size=vocab, hidden_size=64, num_attention_heads=4,
        intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64,
    )
    config = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=32)
    topo = config.build_topology()

    # Every rank builds the same corpus deterministically.
    corpus = build_synthetic_corpus(config, vocab, config.sequence_length)

    sync = TorchDistBackend(topo)
    device = sync.device
    engine = DiPaCoEngine(config, DiLoCoConfig(inner_steps=5, inner_lr=1e-3), sync, device=device, seed=0)
    engine.total_rounds = 8  # manual round loop -> tell the LR schedule the horizon

    for r in range(8):
        m = engine.run_round(corpus, batch_size=8, round_idx=r)
        if rank == 0:
            print(f"[round {r}] inner_loss={m.inner_loss:.4f} delta_norm={m.delta_norm:.4f}", flush=True)

    dist.barrier()
    if rank == 0:
        print("done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
