"""Does DiPaCo's edge over dense *grow with the number of paths*? (the paper's thesis)

Fixes the per-path backbone and sweeps the path count (K x K for K in LEVELS_SWEEP),
running the dense-vs-DiPaCo comparison at each point on the *same* real-C4 corpus.
Dense is always one path of the same backbone, so its perplexity should stay ~flat;
DiPaCo's should improve as paths (= shared-module capacity at fixed inference cost)
increase. A widening margin is DiPaCo's core claim, on real data, on one GPU.

    python examples/scale_sweep_gpu.py
    LEVELS_SWEEP=2,3,4,5,6 HIDDEN=384 ROUNDS=30 NUM_DOCS=10000 python examples/scale_sweep_gpu.py

The corpus + tokenizer are built once and reused across the sweep. Single 16 GB card,
so this is still far from the paper's regime -- read the *trend*, not absolute numbers.
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
NUM_DOCS = _int("NUM_DOCS", 10000)
SEED = _int("SEED", 0)
LEVELS_SWEEP = [int(x) for x in os.environ.get("LEVELS_SWEEP", "2,3,4,5,6").split(",")]
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CACHE = os.environ.get("CACHE", "/tmp/opendipaco_c4_sweep")


def get_corpus():
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    cache_file = Path(CACHE) / f"c4_{NUM_DOCS}_{VOCAB}.pt"
    try:
        from datasets import load_dataset

        if not cache_file.exists():
            stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
            sample = [r["text"] for r in itertools.islice(stream, 8000)]
            tok = train_tokenizer(sample, vocab_size=VOCAB, model="unigram")
        else:
            tok = None  # vocab known from cache; tokenizer not needed to reload docs
        docs = load_c4_documents(num_documents=NUM_DOCS, tokenizer=tok, max_doc_tokens=512,
                                 cache_path=cache_file) if tok else \
            load_c4_documents(num_documents=NUM_DOCS, cache_path=cache_file)
        print(f"loaded {len(docs)} C4 docs (vocab {VOCAB})", flush=True)
        return docs, VOCAB
    except Exception as e:
        print(f"C4 unavailable ({type(e).__name__}); synthetic corpus", flush=True)
        g = torch.Generator().manual_seed(0)
        span = VOCAB // 8
        return [torch.randint(t * span, (t + 1) * span, (160,), generator=g)
                for t in range(8) for _ in range(NUM_DOCS // 8)], VOCAB


def main():
    torch.manual_seed(0)
    docs, vocab = get_corpus()
    train_docs, _, test_docs = split_documents(docs, test_fraction=0.1, seed=0)
    backbone = BackboneConfig(
        vocab_size=vocab, hidden_size=HIDDEN, num_attention_heads=HEADS,
        intermediate_size=INTERMEDIATE, layers_per_level=[LAYERS, LAYERS],
        max_position_embeddings=EVAL_LEN)
    diloco = DiLoCoConfig(inner_steps=INNER_STEPS, inner_lr=4e-4)
    print(f"device={DEVICE} | backbone hidden={HIDDEN} layers/level={LAYERS} | "
          f"rounds={ROUNDS} | train={len(train_docs)} test={len(test_docs)}\n"
          f"sweeping K in {LEVELS_SWEEP} (paths = K^2)\n", flush=True)

    rows = []
    for k in LEVELS_SWEEP:
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
        rows.append((k * k, res["dense_ppl"], res["dipaco_ppl"], margin,
                     res["dipaco_total_params"], dt, mem))
        print(f"  K={k:>2} ({k*k:>2} paths): dense {res['dense_ppl']:7.2f}  "
              f"DiPaCo {res['dipaco_ppl']:7.2f}  margin {margin:+7.2f}  "
              f"({dt:.0f}s, {mem:.1f}GB, {res['dipaco_total_params']/1e6:.0f}M total)", flush=True)

    print("\n=== scale sweep: DiPaCo margin vs path count (real C4, "
          f"{HIDDEN}-hidden, {ROUNDS} rounds) ===")
    print(f"{'paths':>6} {'dense':>9} {'DiPaCo':>9} {'margin':>9} {'total_params':>13}")
    for paths, dpp, kpp, margin, tot, _dt, _m in rows:
        print(f"{paths:>6} {dpp:>9.2f} {kpp:>9.2f} {margin:>+9.2f} {tot/1e6:>11.0f}M")
    if len(rows) > 1:
        trend = rows[-1][3] - rows[0][3]
        print(f"\nmargin change from {rows[0][0]} -> {rows[-1][0]} paths: {trend:+.2f} ppl "
              f"({'GROWS with paths -> DiPaCo thesis' if trend > 0 else 'does not grow at this scale'})")


if __name__ == "__main__":
    main()
