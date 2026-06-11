"""Does DiPaCo's edge hold when data scales *with* paths? (isolates capacity)

The fixed-data sweep (`scale_sweep_gpu.py`) showed DiPaCo's margin collapsing past
~16 paths -- but that conflated "more capacity" with "less data per expert". This
sweep holds **data-per-path constant** (total docs = DOCS_PER_PATH * K^2), so each
expert always sees the same amount of data; only the path *count* (capacity at fixed
inference cost) grows. If the margin now holds or grows, the fixed-data collapse was
data/routing starvation, and the added capacity genuinely helps -- the paper's thesis.

Both models train on the same corpus at each point (equal inference cost; DiPaCo uses
K x the training compute -- that's the paper's deal). The max corpus (DOCS_PER_PATH *
max_K^2) is downloaded once and sliced per point.

    python examples/data_scaled_sweep_gpu.py
    DOCS_PER_PATH=2500 LEVELS_SWEEP=2,3,4,5,6 ROUNDS=30 python examples/data_scaled_sweep_gpu.py
"""

from __future__ import annotations

import itertools
import os
import time
from pathlib import Path

import torch

from opendipaco import BackboneConfig, DiLoCoConfig
from opendipaco.data import load_c4_documents, split_documents, train_tokenizer
from opendipaco.validation import run_comparison


def _int(name, d):
    return int(os.environ.get(name, d))


HIDDEN = _int("HIDDEN", 384)
HEADS = _int("HEADS", 6)
INTERMEDIATE = _int("INTERMEDIATE", 1024)
LAYERS = _int("LAYERS", 2)
VOCAB = _int("VOCAB", 8192)
SEQ_LEN = _int("SEQ_LEN", 128)
EVAL_LEN = _int("EVAL_LEN", 256)
ROUNDS = _int("ROUNDS", 30)
INNER_STEPS = _int("INNER_STEPS", 20)
BATCH = _int("BATCH", 24)
DOCS_PER_PATH = _int("DOCS_PER_PATH", 2500)
SEED = _int("SEED", 0)
LEVELS_SWEEP = [int(x) for x in os.environ.get("LEVELS_SWEEP", "2,3,4,5,6").split(",")]
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CACHE = os.environ.get("CACHE", "/tmp/opendipaco_c4_datascaled")

MAX_DOCS = DOCS_PER_PATH * max(LEVELS_SWEEP) ** 2


def get_corpus():
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    cache_file = Path(CACHE) / f"c4_{MAX_DOCS}_{VOCAB}.pt"
    try:
        from datasets import load_dataset

        tok = None
        if not cache_file.exists():
            stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
            sample = [r["text"] for r in itertools.islice(stream, 8000)]
            tok = train_tokenizer(sample, vocab_size=VOCAB, model="unigram")
        docs = load_c4_documents(num_documents=MAX_DOCS, tokenizer=tok, max_doc_tokens=512,
                                 cache_path=cache_file)
        print(f"loaded {len(docs)} C4 docs (vocab {VOCAB})", flush=True)
        return docs, VOCAB
    except Exception as e:
        print(f"C4 unavailable ({type(e).__name__}); synthetic corpus", flush=True)
        g = torch.Generator().manual_seed(0)
        span = VOCAB // 8
        return [torch.randint(t * span, (t + 1) * span, (160,), generator=g)
                for t in range(8) for _ in range(MAX_DOCS // 8)], VOCAB


def main():
    torch.manual_seed(0)
    all_docs, vocab = get_corpus()
    backbone = BackboneConfig(
        vocab_size=vocab, hidden_size=HIDDEN, num_attention_heads=HEADS,
        intermediate_size=INTERMEDIATE, layers_per_level=[LAYERS, LAYERS],
        max_position_embeddings=EVAL_LEN)
    diloco = DiLoCoConfig(inner_steps=INNER_STEPS, inner_lr=4e-4)
    print(f"device={DEVICE} | hidden={HIDDEN} | rounds={ROUNDS} | "
          f"~{DOCS_PER_PATH} docs/path held constant\n"
          f"sweeping K in {LEVELS_SWEEP} (paths=K^2, total docs=path*{DOCS_PER_PATH})\n", flush=True)

    rows = []
    for k in LEVELS_SWEEP:
        paths = k * k
        n_docs = min(DOCS_PER_PATH * paths, len(all_docs))
        docs = all_docs[:n_docs]
        train_docs, _, test_docs = split_documents(docs, test_fraction=0.1, seed=0)
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        t0 = time.time()
        res = run_comparison(backbone, train_docs, test_docs, diloco,
                             dipaco_levels=(k, k), rounds=ROUNDS, batch_size=BATCH,
                             seq_len=SEQ_LEN, eval_len=EVAL_LEN, device=DEVICE, seed=SEED)
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
        margin = res["dense_ppl"] - res["dipaco_ppl"]
        rows.append((paths, n_docs, res["dense_ppl"], res["dipaco_ppl"], margin, dt, mem))
        print(f"  K={k:>2} ({paths:>2} paths, {n_docs:>5} docs): dense {res['dense_ppl']:7.2f}  "
              f"DiPaCo {res['dipaco_ppl']:7.2f}  margin {margin:+7.2f}  ({dt:.0f}s, {mem:.1f}GB)",
              flush=True)

    print(f"\n=== data-scaled sweep ({DOCS_PER_PATH} docs/path constant, real C4) ===")
    print(f"{'paths':>6} {'docs':>7} {'dense':>9} {'DiPaCo':>9} {'margin':>9}")
    for paths, n_docs, dpp, kpp, margin, _dt, _m in rows:
        print(f"{paths:>6} {n_docs:>7} {dpp:>9.2f} {kpp:>9.2f} {margin:>+9.2f}")
    if len(rows) > 1:
        trend = rows[-1][4] - rows[0][4]
        verdict = ("HOLDS/GROWS with paths when data scales too -> capacity helps (thesis)"
                   if trend > -20 else "still degrades -> not just data starvation")
        print(f"\nmargin {rows[0][0]}->{rows[-1][0]} paths (data scaled): {trend:+.2f} ppl -> {verdict}")


if __name__ == "__main__":
    main()
