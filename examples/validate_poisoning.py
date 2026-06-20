"""Data-poisoning screen validation (W8 part 1; plan §1.1 trust wall).

Phase 3c's audit agrees on a *digest of the pseudo-gradient*, so N honest checkers
handed the same poisoned shard reproduce the same harmful gradient -> their digests
agree -> the poisoned update is applied. The W8 trusted-probe screen catches what
the digest can't: a poisoned update still **raises loss on a clean held-out probe**.

This harness measures that property directly on a real path (no networking, so it's
deterministic and fast): it trains the *same* path+base on a CLEAN shard and on a
POISONED shard (uniform-random tokens -- the model is dragged toward garbage), and
reports the probe-loss delta + the screen verdict for each. The clean update should
pass (delta <= the margin); the poisoned one should be flagged.

    python examples/validate_poisoning.py
    INNER=150 PROBE_DOCS=24 HIDDEN=96 python examples/validate_poisoning.py

HONEST CAVEAT: this shows the screen catches *crude* poisoning (random/garbage
data that hurts clean-data loss). A *targeted* backdoor tuned to leave clean-probe
loss unchanged can evade it -- the screen raises the bar, it doesn't close the
threat. And like all of Phase 3, the end-to-end convergence-under-attack verdict
rides the §0f WAN run; this validates the screening signal, not training.
"""

from __future__ import annotations

import os

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data.sharding import pack_sequences
from opendipaco.schedule.probe import TrustedProbe, is_harmful
from opendipaco.schedule.scheduler import AsyncScheduler
from opendipaco.train.loop import DiPaCoEngine
from opendipaco.backend.local import LocalBackend


def _i(name, default):
    return int(os.environ.get(name, default))


VOCAB = _i("VOCAB", 64)
HIDDEN = _i("HIDDEN", 64)
INNER = _i("INNER", 100)   # enough that random-token training visibly forgets structure
PROBE_DOCS = _i("PROBE_DOCS", 16)
DOCS = _i("DOCS", 32)
DOC_LEN = _i("DOC_LEN", 48)
SEQ = _i("SEQ", 16)
SEED = _i("SEED", 0)


def _topic_docs(n, gen):
    """``n`` structured 'clean' docs: each drawn from one narrow token band, so the
    model can learn next-token structure (and a poisoned update visibly hurts it)."""
    span = max(1, VOCAB // 4)
    return [torch.randint((t % 4) * span, min((t % 4 + 1) * span, VOCAB), (DOC_LEN,),
                          generator=gen)
            for t in range(n)]


def _poison_shard(gen):
    # Uniform-random tokens: training drags the model toward predicting noise, which
    # raises loss on the structured clean probe. (A structure-preserving relabel --
    # e.g. shifting every token -- would NOT poison: the model just learns the
    # shifted structure, so the screen wouldn't flag it.)
    return torch.randint(0, VOCAB, (DOCS, SEQ), generator=gen)


def _clean_shard(gen):
    return pack_sequences(_topic_docs(DOCS, gen), SEQ)


def _engine():
    bb = BackboneConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_attention_heads=4,
                        intermediate_size=HIDDEN * 2, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    cfg = DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=SEQ)
    return DiPaCoEngine(cfg, DiLoCoConfig(inner_steps=INNER, inner_lr=1e-3),
                        LocalBackend(cfg.build_topology()), seed=SEED, materialize="serial")


def _probe_delta(shard):
    """Train a fresh-from-base path on ``shard`` and return (before, after) probe
    loss + the screen verdict. The bank is the shared base; _train_path trains a
    copy, so each call starts from the same weights."""
    engine = _engine()
    worker = AsyncScheduler(engine, num_workers=1)
    path = engine.topology.path_from_index(0)
    g = torch.Generator().manual_seed(SEED + 1)
    probe = TrustedProbe(pack_sequences(_topic_docs(PROBE_DOCS, g), SEQ))
    engine._opt_state.pop(path, None)          # cold, like an audit checker
    c = worker._train_path(path, shard, batch_size=8, generation=0, probe=probe)
    return c.probe_before, c.probe_after


def main() -> None:
    g = torch.Generator().manual_seed(SEED)
    clean_b, clean_a = _probe_delta(_clean_shard(g))
    pois_b, pois_a = _probe_delta(_poison_shard(g))

    print(f"inner={INNER} probe_docs={PROBE_DOCS} hidden={HIDDEN}")
    print(f"  clean shard:    probe {clean_b:.4f} -> {clean_a:.4f} "
          f"(delta {clean_a - clean_b:+.4f})  harmful={is_harmful(clean_b, clean_a)}")
    print(f"  poisoned shard: probe {pois_b:.4f} -> {pois_a:.4f} "
          f"(delta {pois_a - pois_b:+.4f})  harmful={is_harmful(pois_b, pois_a)}")
    ok = (not is_harmful(clean_b, clean_a)) and is_harmful(pois_b, pois_a)
    if ok:
        print("  screen verdict: PASS (clean accepted, poisoned flagged)")
    else:
        print("  screen verdict: INCONCLUSIVE (tune INNER/PROBE_DOCS; small toy models are noisy)")


if __name__ == "__main__":
    main()
