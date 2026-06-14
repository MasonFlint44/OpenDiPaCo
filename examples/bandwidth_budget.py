"""The bandwidth/round tradeoff — the free `inner_steps` lever (W2 / D8).

The W2 compression levers (delta-down, sparsification, int4) cut the bytes *per
sync*. ``inner_steps`` cuts the *number of syncs*: more local AdamW steps between
syncs means fewer rounds to train a fixed token budget, so total sync traffic
scales as **1/inner_steps** -- with **no precision cost and no new dynamics
risk** (it's the tuned DiLoCo knob the paper already exercises). The cheapest
byte is the sync you don't do, so raise ``inner_steps`` *before* paying any
compression-precision cost.

This is an analytical estimate (no training) of those two axes for one path:

    python examples/bandwidth_budget.py
    PARAMS=150e6 TOKENS=10e9 BATCH=16 SEQ_LEN=2048 UPLINK_MBPS=20 \
        python examples/bandwidth_budget.py

HONEST CAVEAT: a first-order model. It assumes the async cache is fully defeated
(full re-ship per round -- the W2 motivation), counts only the shared path
params (private modules ship ~once, amortized away), and ignores index/scale
overhead and protocol framing. Read the *ratios and scaling*, not the absolute
GB. Convergence at high ``inner_steps`` is a dynamics question for the WAN run.
"""

from __future__ import annotations

import os


def _f(name, default):
    return float(os.environ.get(name, default))


PARAMS = _f("PARAMS", 150e6)        # shared params on the path
TOKENS = _f("TOKENS", 10e9)         # training token budget
BATCH = _f("BATCH", 16)
SEQ_LEN = _f("SEQ_LEN", 2048)
UPLINK_MBPS = _f("UPLINK_MBPS", 20)  # consumer uplink (asymmetric; up is the wall)


def per_param_bytes(compress: str, down: str, density: float) -> tuple[float, float]:
    """(down, up) bytes per shared param per round. Weights ship bf16 unless a
    delta makes a lower-bit down viable (W2a); int4 packs at 0.5 B (W2c);
    sparsification sends ~density*(int32 index + value) (W2b)."""
    val = {"none": 4.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}[compress]
    # Down: full re-ship (bf16 floor, since raw weights can't go below bf16), or
    # an int8/int4 delta against the worker's keyframe.
    if down == "delta" and compress in ("int8", "int4"):
        down_b = val                        # the delta is quantized like a gradient
    else:
        down_b = max(val, 2.0) if compress != "none" else 4.0   # weights >= bf16
    # Up: dense pseudo-gradient at `val`, or sparse top-k (int32 index + value).
    up_b = density * (4.0 + val) if density < 1.0 else val
    return down_b, up_b


def main() -> None:
    tokens_per_round_per_inner = BATCH * SEQ_LEN     # tokens one inner step trains
    gib = 1024 ** 3

    print(f"path: {PARAMS/1e6:.0f}M shared params   budget: {TOKENS/1e9:.0f}B tokens"
          f"   batch={BATCH:.0f} seq_len={SEQ_LEN:.0f}   uplink={UPLINK_MBPS:.0f} Mbps\n")

    # 1. The W2 compression levers: bytes/param/round and the per-round sync size.
    print("compression levers (bytes per shared param per round):")
    configs = [
        ("none", "full", 1.0), ("bf16", "full", 1.0),
        ("int8", "delta", 1.0), ("int4", "delta", 1.0),
        ("int8", "delta", 0.1), ("int4", "delta", 0.05),
    ]
    print(f"  {'compress':>8} {'down':>6} {'density':>8}   {'down B/p':>9} {'up B/p':>8}"
          f"   {'GB/round':>9}")
    for compress, down, density in configs:
        d, u = per_param_bytes(compress, down, density)
        gb_round = PARAMS * (d + u) / gib
        print(f"  {compress:>8} {down:>6} {density:>8.2f}   {d:>9.2f} {u:>8.2f}"
              f"   {gb_round:>9.3f}")

    # 2. The free lever: total sync traffic ~ 1/inner_steps, at the int8+delta point.
    d, u = per_param_bytes("int8", "delta", 1.0)
    sync_bytes_round = PARAMS * (d + u)
    print(f"\ninner_steps lever (at compress=int8, down=delta -> "
          f"{(d+u):.1f} B/param/round):")
    print(f"  {'inner_steps':>11} {'rounds':>12} {'total sync':>12} {'up-time @uplink':>16}")
    for inner in (1, 10, 50, 100, 500):
        rounds = TOKENS / (inner * tokens_per_round_per_inner)
        total_gb = sync_bytes_round * rounds / gib
        up_gb = (PARAMS * u) * rounds / gib                  # the asymmetric wall
        up_hours = (up_gb * 8 * gib) / (UPLINK_MBPS * 1e6) / 3600
        print(f"  {inner:>11d} {rounds:>12,.0f} {total_gb:>10.1f} GB {up_hours:>13.1f} h")

    print("\nTakeaway: total sync traffic scales as 1/inner_steps for free (no precision\n"
          "cost), so raise inner_steps first; the compression levers then cut the\n"
          "remaining per-sync bytes. Both compound. (First-order estimate -- see the\n"
          "module docstring; convergence at high inner_steps rides the WAN run.)")


if __name__ == "__main__":
    main()
