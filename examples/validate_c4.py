"""Validate the method: DiPaCo vs. a matched dense baseline on real C4.

The project's headline question: does a routed single DiPaCo path match/beat a dense
model of that same size? This trains both on a real C4 subset (paper-style: train a
~32k tokenizer on the data, hold out a test split) and compares held-out perplexity
at **equal inference cost** (one path executes). A dense baseline is just DiPaCo with
one expert per level, so the whole thing reuses the engine.

    python examples/validate_c4.py

HONEST CAVEAT: DiPaCo's advantage is a *scale* phenomenon. At the tiny size this demo
runs on CPU, **dense will likely win** — that is expected, not a bug. The harness is
correct and ready to run at the paper's scale on a GPU (bump the backbone, K, rounds,
and C4 size); only there does the comparison test DiPaCo's actual claim.
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import torch

from opendipaco import BackboneConfig, DiLoCoConfig
from opendipaco.data import load_c4_documents, split_documents, tokenize_documents, train_tokenizer
from opendipaco.validation import run_comparison

# Toy by default so it runs on CPU. For a real test, scale all of these way up
# (hidden_size, K via DIPACO_LEVELS, rounds, NUM_DOCS) and run on a GPU.
NUM_DOCS, SEQ_LEN, EVAL_LEN, ROUNDS, K = 1500, 32, 64, 12, 2


def topic_docs(vocab, num_topics=4, per=40, length=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = vocab // num_topics
    return [torch.randint(t * span, (t + 1) * span, (length,), generator=g)
            for t in range(num_topics) for _ in range(per)]


def get_corpus(cache_dir):
    """Real C4 with a freshly-trained 32k-style tokenizer, else a synthetic fallback."""
    try:
        from datasets import load_dataset

        stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
        sample = [r["text"] for r in itertools.islice(stream, 4000)]
        # Paper recipe: train our own tokenizer on the data (small vocab for the demo).
        tok = train_tokenizer(sample, vocab_size=2048, model="unigram")
        docs = load_c4_documents(num_documents=NUM_DOCS, tokenizer=tok, max_doc_tokens=128,
                                 cache_path=Path(cache_dir) / "c4.pt")
        print(f"loaded {len(docs)} C4 docs; trained tokenizer vocab={tok.vocab_size}", flush=True)
        return docs, tok.vocab_size
    except Exception as e:  # ImportError (no extra) or any download error
        print(f"C4 unavailable ({type(e).__name__}); using a synthetic topical corpus", flush=True)
        vocab = 128
        return topic_docs(vocab, per=NUM_DOCS // 4), vocab


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as workdir:
        docs, vocab = get_corpus(workdir)
        train_docs, _, test_docs = split_documents(docs, test_fraction=0.1, seed=0)

        backbone = BackboneConfig(
            vocab_size=vocab, hidden_size=64, num_attention_heads=4,
            intermediate_size=128, layers_per_level=[1, 1], max_position_embeddings=64,
        )
        print(f"train={len(train_docs)} test={len(test_docs)}; training dense + {K}x{K} DiPaCo "
              f"for {ROUNDS} rounds...", flush=True)
        res = run_comparison(
            backbone, train_docs, test_docs, DiLoCoConfig(inner_steps=8, inner_lr=1e-3),
            dipaco_levels=(K, K), rounds=ROUNDS, batch_size=8, seq_len=SEQ_LEN, eval_len=EVAL_LEN,
        )

        print("\n=== held-out perplexity (lower is better; one path executes) ===")
        print(f"  dense  ({res['params_per_path']:,} params): {res['dense_ppl']:.2f}")
        print(f"  DiPaCo ({res['dipaco_paths']} paths, {res['dipaco_total_params']:,} total params,"
              f" {res['params_per_path']:,}/path executed): {res['dipaco_ppl']:.2f}")
        winner = "DiPaCo" if res["dipaco_ppl"] < res["dense_ppl"] else "dense"
        print(f"  -> {winner} wins at this scale "
              f"(expected: dense, until scaled up — see the module docstring).")


if __name__ == "__main__":
    main()
