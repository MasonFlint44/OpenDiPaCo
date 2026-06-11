"""Scaled DiPaCo-vs-dense validation on a GPU (the project's headline question).

Same harness as ``validate_c4.py`` but sized for a GPU and beyond toy scale: a real
C4 subset, a tokenizer trained on it, a multi-path DiPaCo model vs. a matched dense
baseline, compared at **equal inference cost** (one path executes). Auto-selects CUDA
when available; all scale knobs are env-overridable so you can push to fill the card:

    python examples/validate_c4_gpu.py
    HIDDEN=512 LEVELS=4 LAYERS=3 ROUNDS=40 NUM_DOCS=20000 python examples/validate_c4_gpu.py

HONEST CAVEAT: a single consumer GPU is still far from the paper's regime (256 paths,
150M/path, billions of tokens). DiPaCo's "matches/beats dense" result is a *scale*
phenomenon, so even here dense may win or it may be close -- read both numbers
honestly. This is the project's first real GPU comparison, not a paper reproduction.
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


def _int(name, default):
    return int(os.environ.get(name, default))


# Scale knobs (env-overridable). Defaults: a "medium" run, minutes on a 16GB card.
HIDDEN = _int("HIDDEN", 384)
HEADS = _int("HEADS", 6)
INTERMEDIATE = _int("INTERMEDIATE", 1024)
LAYERS = _int("LAYERS", 2)            # layers per level
LEVELS = _int("LEVELS", 4)           # experts per level -> LEVELS**2 paths
VOCAB = _int("VOCAB", 8192)
SEQ_LEN = _int("SEQ_LEN", 128)
EVAL_LEN = _int("EVAL_LEN", 256)
ROUNDS = _int("ROUNDS", 20)
INNER_STEPS = _int("INNER_STEPS", 20)
BATCH = _int("BATCH", 16)
NUM_DOCS = _int("NUM_DOCS", 6000)
SEED = _int("SEED", 0)
ROUTER_SEED = _int("ROUTER_SEED", 0)
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def get_corpus(cache_dir):
    """Real C4 + a tokenizer trained on it; synthetic topical fallback if offline."""
    try:
        from datasets import load_dataset

        stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
        sample = [r["text"] for r in itertools.islice(stream, 8000)]
        tok = train_tokenizer(sample, vocab_size=VOCAB, model="unigram")
        docs = load_c4_documents(num_documents=NUM_DOCS, tokenizer=tok, max_doc_tokens=512,
                                 cache_path=Path(cache_dir) / "c4_gpu.pt")
        print(f"loaded {len(docs)} C4 docs; tokenizer vocab={tok.vocab_size}", flush=True)
        return docs, tok.vocab_size
    except Exception as e:
        print(f"C4 unavailable ({type(e).__name__}); using synthetic topical corpus", flush=True)
        g = torch.Generator().manual_seed(0)
        span = VOCAB // 8
        docs = [torch.randint(t * span, (t + 1) * span, (160,), generator=g)
                for t in range(8) for _ in range(NUM_DOCS // 8)]
        return docs, VOCAB


def main():
    torch.manual_seed(0)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"device={DEVICE} | {LEVELS}x{LEVELS}={LEVELS**2} paths | hidden={HIDDEN} "
          f"layers/level={LAYERS} | rounds={ROUNDS} inner={INNER_STEPS} | docs={NUM_DOCS}",
          flush=True)

    import tempfile
    with tempfile.TemporaryDirectory() as workdir:
        docs, vocab = get_corpus(workdir)
        train_docs, _, test_docs = split_documents(docs, test_fraction=0.1, seed=0)

        backbone = BackboneConfig(
            vocab_size=vocab, hidden_size=HIDDEN, num_attention_heads=HEADS,
            intermediate_size=INTERMEDIATE, layers_per_level=[LAYERS, LAYERS],
            max_position_embeddings=EVAL_LEN,
        )
        diloco = DiLoCoConfig(inner_steps=INNER_STEPS, inner_lr=4e-4)
        print(f"train={len(train_docs)} test={len(test_docs)}; training dense + "
              f"{LEVELS}x{LEVELS} DiPaCo...", flush=True)

        t0 = time.time()
        res = run_comparison(
            backbone, train_docs, test_docs, diloco,
            dipaco_levels=(LEVELS, LEVELS), rounds=ROUNDS, batch_size=BATCH,
            seq_len=SEQ_LEN, eval_len=EVAL_LEN, device=DEVICE,
            seed=SEED, router_seed=ROUTER_SEED,
        )
        elapsed = time.time() - t0

        print("\n=== held-out perplexity (lower is better; one path executes) ===")
        print(f"  dense  ({res['params_per_path']:,} params/path): {res['dense_ppl']:.2f}")
        print(f"  DiPaCo ({res['dipaco_paths']} paths, {res['dipaco_total_params']:,} total, "
              f"{res['params_per_path']:,}/path executed): {res['dipaco_ppl']:.2f}")
        winner = "DiPaCo" if res["dipaco_ppl"] < res["dense_ppl"] else "dense"
        margin = abs(res["dipaco_ppl"] - res["dense_ppl"])
        print(f"  -> {winner} wins by {margin:.2f} ppl")
        print(f"\nwall time: {elapsed:.0f}s", flush=True)
        if DEVICE == "cuda":
            print(f"peak CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
