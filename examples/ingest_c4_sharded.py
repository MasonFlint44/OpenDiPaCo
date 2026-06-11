"""Sharded, resumable C4 ingestion — each host caches its own 1/N slice.

Real ingestion at the paper's scale can't pull the whole corpus into one list on one
box. ``ingest_c4_shard`` instead streams + tokenizes only *this host's* shard
(``shard_id`` of ``num_shards``) and caches ``(docs, next_index)`` atomically, so an
interrupted ingest **resumes** from where it left off rather than restarting.

    python examples/ingest_c4_sharded.py

This demo (1) ingests shard 0 of 2 with a tiny target to *simulate an interruption*,
then resumes the same shard to a larger target — showing it continues, not restarts;
and (2) ingests shard 1 to show the two shards are disjoint. Falls back to a synthetic
corpus when C4/``datasets`` isn't available, so it runs offline.
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

from opendipaco.data import ShardCache, ingest_c4_shard, train_tokenizer

NUM_SHARDS, TARGET_PER_SHARD = 2, 60


def get_source_and_tokenizer():
    """Real C4 text stream + a tokenizer trained on a sample, else a synthetic source."""
    try:
        from datasets import load_dataset

        stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
        sample = [r["text"] for r in itertools.islice(stream, 3000)]
        tok = train_tokenizer(sample, vocab_size=2048, model="unigram")
        # Re-stream rows (dicts with "text") for ingestion; ingest reads text_key="text".
        source = load_dataset("allenai/c4", "en", split="train", streaming=True)
        print(f"using real C4 (tokenizer vocab={tok.vocab_size})", flush=True)
        return source, tok
    except Exception as e:  # ImportError or any download error
        print(f"C4 unavailable ({type(e).__name__}); using a synthetic corpus", flush=True)
        texts = [" ".join(f"w{i}_{j}" for j in range(i % 7 + 1)) for i in range(400)]
        tok = train_tokenizer(texts, vocab_size=512, model="unigram")
        return texts, tok


def main():
    source, tok = get_source_and_tokenizer()
    with tempfile.TemporaryDirectory() as cache_dir:
        print(f"\n[shard 0] ingest a partial slice (simulating an interruption)...", flush=True)
        partial = ingest_c4_shard(cache_dir, shard_id=0, num_shards=NUM_SHARDS,
                                  target_docs=12, tokenizer=tok, source=source,
                                  flush_every=4, max_doc_tokens=128)
        _, resume_at = ShardCache(cache_dir, shard_id=0, num_shards=NUM_SHARDS).load()
        print(f"[shard 0] cached {len(partial)} docs; will resume at stream index {resume_at}",
              flush=True)

        print(f"[shard 0] resume to the full target...", flush=True)
        full = ingest_c4_shard(cache_dir, shard_id=0, num_shards=NUM_SHARDS,
                               target_docs=TARGET_PER_SHARD, tokenizer=tok, source=source,
                               flush_every=4, max_doc_tokens=128,
                               progress=lambda c, n: print(f"  ...{c} docs (next index {n})",
                                                           flush=True))
        print(f"[shard 0] complete: {len(full)} docs", flush=True)

        print(f"\n[shard 1] ingest the other disjoint slice...", flush=True)
        other = ingest_c4_shard(cache_dir, shard_id=1, num_shards=NUM_SHARDS,
                                target_docs=TARGET_PER_SHARD, tokenizer=tok, source=source,
                                flush_every=4, max_doc_tokens=128)
        print(f"[shard 1] complete: {len(other)} docs", flush=True)

        files = sorted(p.name for p in Path(cache_dir).glob("shard_*.pt"))
        print(f"\non-disk shard caches: {files}", flush=True)
        print(f"total ingested across 2 shards: {len(full) + len(other)} docs "
              f"(each host holds only its own ~{TARGET_PER_SHARD})", flush=True)


if __name__ == "__main__":
    main()
