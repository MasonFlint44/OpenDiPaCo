"""The per-path VRAM budget — measure-first for W3 (docs/w3-vram-design.md).

A worker holds one path, but its per-round training peak stacks the fetched
global + the local params + AdamW moments (2x) + grads + activations, and for a
real vocab the private embedding/head dominates. This shows that breakdown for a
config, what each W3 lever saves, and whether it fits a budget -- so we attack
the real peak instead of guessing.

    python examples/vram_budget.py
    VOCAB=128000 HIDDEN=4096 LAYERS_PER_LEVEL=8 LEVELS=4 SEQ_LEN=4096 BATCH=4 \
        BUDGET_GB=12 python examples/vram_budget.py

On CUDA it also reports the *measured* peak of a real round (the truth); the
calculator's parameter/optimizer terms are exact, the activation term coarse.
"""

from __future__ import annotations

import os

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.train.memory import measure_peak, vram_breakdown


def _i(name, d):
    return int(os.environ.get(name, d))


VOCAB = _i("VOCAB", 32000)
HIDDEN = _i("HIDDEN", 2048)
HEADS = _i("HEADS", 16)
INTERMEDIATE = _i("INTERMEDIATE", 5632)
LAYERS_PER_LEVEL = _i("LAYERS_PER_LEVEL", 4)
LEVELS = _i("LEVELS", 2)
EXPERTS = _i("EXPERTS", 4)        # experts per level (path picks one)
SEQ_LEN = _i("SEQ_LEN", 2048)
BATCH = _i("BATCH", 4)
BUDGET_GB = float(os.environ.get("BUDGET_GB", "12"))
DEVICE = os.environ.get("DEVICE", "cpu")

_MIB = 1024 ** 2
_GIB = 1024 ** 3


def _config():
    bb = BackboneConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_attention_heads=HEADS,
                        intermediate_size=INTERMEDIATE,
                        layers_per_level=[LAYERS_PER_LEVEL] * LEVELS,
                        max_position_embeddings=max(SEQ_LEN, 2048))
    return DiPaCoConfig(backbone=bb, level_sizes=[EXPERTS] * LEVELS, sequence_length=SEQ_LEN)


def _row(name, b):
    fit = "OK " if b["total"] <= BUDGET_GB * _GIB else "OVER"
    print(f"  {name:<28} params={b['params']/_GIB:5.2f} adam={b['adam']/_GIB:5.2f} "
          f"act={b['activations']/_GIB:5.2f}  total={b['total']/_GIB:6.2f} GB  [{fit}]")


def main() -> None:
    cfg = _config()
    base = vram_breakdown(cfg, batch_size=BATCH, seq_len=SEQ_LEN)
    print(f"path: {base['n_params']/1e6:.0f}M params "
          f"({base['n_private']/1e6:.0f}M private)   "
          f"vocab={VOCAB} hidden={HIDDEN} depth={LAYERS_PER_LEVEL*LEVELS}   "
          f"batch={BATCH} seq_len={SEQ_LEN}   budget={BUDGET_GB:.0f} GB\n")

    print("per-round VRAM peak (estimate; GB):")
    _row("baseline (fp32)", base)
    _row("+ inner_autocast", vram_breakdown(cfg, batch_size=BATCH, seq_len=SEQ_LEN,
                                            autocast=True))
    _row("+ activation checkpoint", vram_breakdown(cfg, batch_size=BATCH, seq_len=SEQ_LEN,
                                                   autocast=True, checkpoint=True))
    _row("+ chunked logits", vram_breakdown(cfg, batch_size=BATCH, seq_len=SEQ_LEN,
                                            autocast=True, checkpoint=True,
                                            chunked_logits=True))
    # 8-bit Adam (lossy): moments at ~1 byte each instead of fp32's 8 bytes total.
    stacked = dict(vram_breakdown(cfg, batch_size=BATCH, seq_len=SEQ_LEN, autocast=True,
                                  checkpoint=True, chunked_logits=True))
    stacked["adam"] = base["n_params"] * 2          # 2 moments x ~1 byte
    stacked["total"] = (stacked["params"] + stacked["global"] + stacked["adam"]
                        + stacked["grads"] + stacked["activations"])
    _row("+ 8-bit Adam (lossy)", stacked)

    measured = measure_peak(cfg, DiLoCoConfig(inner_steps=2), device=DEVICE,
                            batch_size=BATCH, seq_len=SEQ_LEN)
    if measured is not None:
        print(f"\n  measured peak (real round, CUDA): {measured/_GIB:.2f} GB")
    else:
        print("\n  (run with DEVICE=cuda for the measured real-round peak)")

    print("\nTakeaway: exact levers (checkpoint, chunked logits, the private-copy de-dup)\n"
          "carry no convergence risk and stack first; 8-bit Adam is the one lossy lever\n"
          "(rides the 0f run). Parameter/optimizer terms are exact; activations estimated\n"
          "-- the measured peak is the truth. See docs/w3-vram-design.md.")


if __name__ == "__main__":
    main()
