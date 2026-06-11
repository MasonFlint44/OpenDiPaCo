"""Sharded, resumable streaming ingestion for corpora too big to hold in memory.

``load_c4_documents`` pulls the *whole* corpus into one list and caches it to one
file -- fine for a demo, hopeless at the paper's scale (billions of tokens across a
cluster). This module adds the two things a real ingest needs:

- **Sharding** -- ``shard_stream`` / ``stream_documents`` round-robin a stream by
  global position, so each of ``num_shards`` hosts ingests a disjoint ``1/N`` slice
  (pass that host's ``shard_id``). Documents are tokenized one at a time and yielded
  lazily, so memory is bounded by what you keep, not the corpus size.
- **Resumability** -- ingestion tracks the *next global stream index*; ``ShardCache``
  persists ``(docs, next_index)`` as a single atomic file, and ``ingest_c4_shard``
  resumes a partially-ingested shard from exactly where it left off (re-deriving only
  the un-flushed tail, never duplicating or losing a document).

Everything is dependency-injected on a raw ``source`` iterable (strings, or HF rows),
so the logic is testable without the network; ``stream_c4_documents`` /
``ingest_c4_shard`` wire it to the real C4 stream (lazy ``datasets`` import).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .c4 import _DATASETS_HINT
from .text import tokenize_one


def _extract_text(item, text_key: str) -> str:
    """A source item is either a raw string or a mapping (e.g. an HF row)."""
    return item[text_key] if isinstance(item, dict) else item


def shard_stream(source, *, num_shards: int, shard_id: int, start_index: int = 0):
    """Yield ``(global_index, item)`` for the items this shard owns.

    Round-robin by global position (``global_index % num_shards == shard_id``), so
    ``num_shards`` shards partition the stream disjointly. Resumes at ``start_index``,
    using the source's native ``.skip()`` (HF ``IterableDataset``) when available so a
    resume doesn't re-read the skipped prefix.
    """
    if num_shards < 1 or not (0 <= shard_id < num_shards):
        raise ValueError(f"bad shard {shard_id}/{num_shards}")
    owns = lambda i: num_shards == 1 or i % num_shards == shard_id  # noqa: E731
    if start_index and hasattr(source, "skip"):
        for offset, item in enumerate(source.skip(start_index)):
            gi = start_index + offset
            if owns(gi):
                yield gi, item
    else:
        for i, item in enumerate(source):
            if i >= start_index and owns(i):
                yield i, item


def stream_documents(source, tokenizer, *, num_shards: int = 1, shard_id: int = 0,
                     start_index: int = 0, limit: int | None = None, text_key: str = "text",
                     add_eos: bool = True, max_doc_tokens: int | None = None,
                     min_doc_tokens: int = 1):
    """Lazily yield ``(global_index, LongTensor)`` tokenized docs for one shard.

    Tokenizes per document with :func:`~opendipaco.data.text.tokenize_one` (identical
    tokens to the batch path), skipping filtered/empty docs. ``limit`` caps the number
    of *yielded* (kept) documents. ``global_index`` is the position in the underlying
    stream -- ``+1`` of the last one is the resume point.
    """
    count = 0
    for gi, item in shard_stream(source, num_shards=num_shards, shard_id=shard_id,
                                 start_index=start_index):
        doc = tokenize_one(_extract_text(item, text_key), tokenizer, add_eos=add_eos,
                           max_doc_tokens=max_doc_tokens, min_doc_tokens=min_doc_tokens)
        if doc is None:
            continue
        yield gi, doc
        count += 1
        if limit is not None and count >= limit:
            return


def _c4_stream(split: str):
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(_DATASETS_HINT) from e
    return load_dataset("allenai/c4", "en", split=split, streaming=True)


def stream_c4_documents(*, tokenizer, split: str = "train", num_shards: int = 1,
                        shard_id: int = 0, start_index: int = 0, limit: int | None = None,
                        add_eos: bool = True, max_doc_tokens: int | None = None,
                        min_doc_tokens: int = 1):
    """:func:`stream_documents` wired to the real C4 stream (one shard, resumable)."""
    yield from stream_documents(
        _c4_stream(split), tokenizer, num_shards=num_shards, shard_id=shard_id,
        start_index=start_index, limit=limit, add_eos=add_eos,
        max_doc_tokens=max_doc_tokens, min_doc_tokens=min_doc_tokens)


class ShardCache:
    """Crash-consistent on-disk cache for one data shard.

    Stores ``(docs, next_index)`` in a single atomically-renamed file
    (``<dir>/shard_<id>_of_<n>.pt``), so the persisted docs and the resume point are
    always consistent -- a crash between flushes just re-derives the un-flushed tail.
    """

    def __init__(self, dirpath, *, shard_id: int, num_shards: int):
        if num_shards < 1 or not (0 <= shard_id < num_shards):
            raise ValueError(f"bad shard {shard_id}/{num_shards}")
        self.dir = Path(dirpath)
        self.path = self.dir / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"

    def load(self) -> tuple[list[torch.Tensor], int]:
        """Return ``(docs, next_index)`` -- ``([], 0)`` if nothing is cached yet."""
        if not self.path.exists():
            return [], 0
        blob = torch.load(self.path, weights_only=True)
        return list(blob["docs"]), int(blob["next_index"])

    def save(self, docs: list[torch.Tensor], next_index: int) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = str(self.path) + ".tmp"
        torch.save({"docs": docs, "next_index": int(next_index), "count": len(docs)}, tmp)
        os.replace(tmp, self.path)  # atomic: docs + resume point land together


def ingest_c4_shard(cache_dir, *, shard_id: int, num_shards: int, target_docs: int,
                    tokenizer, split: str = "train", flush_every: int = 500,
                    source=None, text_key: str = "text", add_eos: bool = True,
                    max_doc_tokens: int | None = None, min_doc_tokens: int = 1,
                    progress=None) -> list[torch.Tensor]:
    """Resumably ingest ``target_docs`` documents for one shard into ``cache_dir``.

    Loads any partial shard cache and continues from its saved ``next_index``,
    streaming + tokenizing this shard's slice until it holds ``target_docs`` docs (or
    the stream ends), flushing every ``flush_every`` kept docs. Returns the docs.

    ``source`` (an iterable of strings / HF rows) overrides the C4 stream -- used by
    tests and to ingest a non-C4 corpus. ``progress(count, next_index)`` is called on
    each flush for live status.
    """
    cache = ShardCache(cache_dir, shard_id=shard_id, num_shards=num_shards)
    docs, next_index = cache.load()
    if len(docs) >= target_docs:
        return docs[:target_docs]

    tok_kw = dict(add_eos=add_eos, max_doc_tokens=max_doc_tokens, min_doc_tokens=min_doc_tokens)
    if source is None:
        stream = stream_c4_documents(tokenizer=tokenizer, split=split, num_shards=num_shards,
                                     shard_id=shard_id, start_index=next_index, **tok_kw)
    else:
        stream = stream_documents(source, tokenizer, num_shards=num_shards, shard_id=shard_id,
                                  start_index=next_index, text_key=text_key, **tok_kw)

    since_flush = 0
    for gi, doc in stream:
        docs.append(doc)
        next_index = gi + 1  # resume strictly after the last accepted doc
        since_flush += 1
        if since_flush >= flush_every:
            cache.save(docs, next_index)
            since_flush = 0
            if progress is not None:
                progress(len(docs), next_index)
        if len(docs) >= target_docs:
            break
    cache.save(docs, next_index)
    if progress is not None:
        progress(len(docs), next_index)
    return docs[:target_docs]
