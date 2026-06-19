"""Shard specs: ship the *recipe* for a shard, not its bytes.

(Internet-scale plan §0d / finding 1.3.) The coordinator/scheduler previously
held the whole tokenized corpus in RAM and shipped each path's packed shard
tensor over the wire. A shard *spec* instead describes how to materialize shard
``i`` deterministically from a public source:

* the **document source** — synthetic parameters, or a C4 stream slice
  (split, document budget, tokenizer name, truncation filters);
* the **routing state** — k-means centroids plus the deterministic
  bag-of-tokens featurizer's parameters (a few KB), or round-robin;
* the **packing parameters** (sequence length, pack mode, prefix length).

A worker that receives a spec regenerates or streams the documents, routes them
locally with the shipped router, keeps only its path's documents, and packs
them — memory bounded by batch + kept shard, optional on-disk cache, zero
corpus bytes on the wire. The server keeps a :class:`SpecCorpus`: the spec plus
per-path token counts (the alpha-weighting basis), holding no sequences.

**Determinism contract:** the same spec must yield the same shard everywhere.
The synthetic generator and the C4 row-budget/tokenize/filter semantics here are
the single source of truth — the launch roles build documents through these same
functions.

Known limits (deliberate): spec corpora carry no per-path validation split, so
per-path early stopping is off in spec mode; and a cold worker re-streams the
C4 prefix (pass ``cache_dir=`` to keep shards on disk; the resumable bulk-ingest
path remains :func:`~opendipaco.data.streaming.ingest_c4_shard`).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from ..routing.base import BagOfTokensFeaturizer
from ..routing.kmeans import KMeansRouter
from .sharding import chunk_documents, pack_sequences
from .text import tokenize_one

SPEC_VERSION = 1
_ASSIGN_BATCH = 256  # docs routed per featurizer batch (bounds memory)


# -- document sources ----------------------------------------------------------


def synthetic_documents(*, vocab_size: int, num_documents: int, doc_len: int,
                        topics: int, seed: int) -> list[torch.Tensor]:
    """The deterministic synthetic corpus (topic-banded random tokens).

    Single source of truth: the launcher and spec materialization both call
    this, so a worker regenerates byte-identical documents from the spec.
    """
    g = torch.Generator().manual_seed(seed)
    span = max(1, vocab_size // topics)
    per = max(1, num_documents // topics)
    return [torch.randint(t * span, min((t + 1) * span, vocab_size), (doc_len,), generator=g)
            for t in range(topics) for _ in range(per)]


def synthetic_source(*, vocab_size: int, num_documents: int, doc_len: int,
                     topics: int, seed: int) -> dict:
    return {"kind": "synthetic", "vocab_size": vocab_size, "num_documents": num_documents,
            "doc_len": doc_len, "topics": topics, "seed": seed}


def c4_source(*, num_documents: int, tokenizer: str, split: str = "train",
              max_doc_tokens: int | None = None, min_doc_tokens: int = 1,
              add_eos: bool = True) -> dict:
    """First ``num_documents`` *rows* of the C4 stream, tokenized + filtered.

    Row-budget semantics match ``load_c4_documents`` (the budget counts stream
    rows, not kept documents), so spec materialization sees exactly the same
    document list the server routed.
    """
    return {"kind": "c4", "num_documents": num_documents, "tokenizer": tokenizer,
            "split": split, "max_doc_tokens": max_doc_tokens,
            "min_doc_tokens": min_doc_tokens, "add_eos": add_eos}


def iter_spec_documents(spec: dict, *, source=None, tokenizer=None):
    """Yield the spec's documents, in their canonical order.

    ``source`` (an iterable of strings / HF rows) overrides the C4 stream for
    tests or non-C4 corpora; ``tokenizer`` overrides the spec-named one.
    """
    src = spec["source"]
    if src["kind"] == "synthetic":
        yield from synthetic_documents(
            vocab_size=src["vocab_size"], num_documents=src["num_documents"],
            doc_len=src["doc_len"], topics=src["topics"], seed=src["seed"])
        return
    if src["kind"] != "c4":
        raise ValueError(f"unknown source kind {src['kind']!r}")
    if source is None:
        from .streaming import _c4_stream
        source = _c4_stream(src["split"])
    if tokenizer is None:
        from .text import load_tokenizer
        tokenizer = load_tokenizer(src["tokenizer"])
    for i, item in enumerate(source):
        if i >= src["num_documents"]:  # row budget, like load_c4_documents
            return
        text = item["text"] if isinstance(item, dict) else item
        doc = tokenize_one(text, tokenizer, add_eos=src["add_eos"],
                           max_doc_tokens=src["max_doc_tokens"],
                           min_doc_tokens=src["min_doc_tokens"])
        if doc is not None:
            yield doc


# -- routing -------------------------------------------------------------------


def round_robin_routing() -> dict:
    return {"kind": "round_robin"}


def kmeans_routing(centroids: torch.Tensor, *, vocab_size: int, feature_dim: int,
                   seed: int = 0) -> dict:
    """A fitted k-means router + the bag-of-tokens featurizer that feeds it.

    The featurizer is reconstructed from ``(vocab_size, feature_dim, seed)`` —
    its projection is a frozen seeded matrix — so only the centroids (a few KB)
    actually travel.
    """
    return {"kind": "kmeans", "centroids": centroids.detach().cpu(),
            "featurizer": {"kind": "bag_of_tokens", "vocab_size": vocab_size,
                           "feature_dim": feature_dim, "seed": seed}}


def fit_routing_from_source(source: dict, *, num_paths: int, vocab_size: int,
                            seq_len: int, sample: int, feature_dim: int,
                            router_seed: int = 0, doc_source=None,
                            tokenizer=None) -> dict:
    """Fit the k-means router on a bounded, **deterministic** sample of the public
    source (the first ``sample`` documents) and return a :func:`kmeans_routing`
    dict -- or :func:`round_robin_routing` if the sample is too small to fit
    (W7b, ``docs/w7-data-decentralization-design.md``).

    The point: the operator builds the shard spec **without ever holding the whole
    corpus** -- it streams at most ``sample`` document prefixes to fit, then
    :meth:`SpecCorpus.build` streams again for the token counts. The fit matches
    the in-hand one (:func:`~opendipaco.launch.roles.build_spec_corpus`): a
    ``BagOfTokensFeaturizer`` (seed 0) over ``doc[:seq_len]`` prefixes, k-means at
    ``router_seed``.

    **Determinism / reproducibility:** the centroids are a pure function of the
    *exact prefix sequence* the source yields (k-means++ seeds off ``randperm(n)``
    where ``n`` is the number of docs actually streamed, so the count itself is an
    input). A synthetic source regenerates bit-identically; a C4 source is only
    reproducible if the stream replays the same rows (pin the dataset revision).
    Under that assumption any peer reproduces byte-identical centroids -- the basis
    for slice c's ``verify_routing``. Because it fits on a *sample* (not every
    document) the assignments differ from the full-corpus fit; this path is opt-in
    (``data.router_sample``) and not byte-identical to the unsampled run.

    If the sample yields fewer than ``num_paths`` documents (a tiny corpus, an
    over-small ``sample``, or aggressive C4 filtering), k-means cannot seed one
    centroid per cluster, so this degrades to round-robin -- mirroring the in-hand
    builder's ``len(docs) < num_paths`` fallback rather than crashing at startup.
    """
    if sample < 1:
        raise ValueError(f"router sample must be >= 1, got {sample}")
    feat = BagOfTokensFeaturizer(vocab_size, feature_dim=feature_dim)
    prefixes: list[torch.Tensor] = []
    for doc in iter_spec_documents({"source": source}, source=doc_source, tokenizer=tokenizer):
        prefixes.append(doc[:seq_len])
        if len(prefixes) >= sample:
            break
    if len(prefixes) < num_paths:
        return round_robin_routing()
    router = KMeansRouter(num_paths, seed=router_seed).fit(feat(prefixes))
    return kmeans_routing(router.centroids, vocab_size=vocab_size,
                          feature_dim=feature_dim)


def _build_predictor(spec: dict):
    """Return ``predict(prefixes) -> LongTensor`` for a spec, or None (round-robin)."""
    routing = spec["routing"]
    if routing["kind"] == "round_robin":
        return None
    if routing["kind"] != "kmeans":
        raise ValueError(f"unknown routing kind {routing['kind']!r}")
    f = routing["featurizer"]
    if f["kind"] != "bag_of_tokens":
        raise ValueError(f"unknown featurizer kind {f['kind']!r}")
    featurizer = BagOfTokensFeaturizer(f["vocab_size"], feature_dim=f["feature_dim"],
                                       seed=f["seed"])
    router = KMeansRouter(spec["num_paths"])
    router.centroids = routing["centroids"]
    return lambda prefixes: router.predict(featurizer(prefixes))


def iter_assignments(spec: dict, docs, *, source=None, tokenizer=None):
    """Yield ``(doc, path_index)`` for every document, batched through the router.

    ``docs`` may be ``None`` (stream from the spec) or an in-hand iterable (the
    server uses the documents it already loaded). Routing matches
    ``ShardedCorpus.from_documents``: route on the first ``prefix_len`` tokens.
    """
    if docs is None:
        docs = iter_spec_documents(spec, source=source, tokenizer=tokenizer)
    predict = _build_predictor(spec)
    if predict is None:
        for i, d in enumerate(docs):
            yield d, i % spec["num_paths"]
        return
    prefix_len = spec["prefix_len"]
    batch: list[torch.Tensor] = []
    for d in docs:
        batch.append(d)
        if len(batch) >= _ASSIGN_BATCH:
            for doc, p in zip(batch, predict([b[:prefix_len] for b in batch]).tolist()):
                yield doc, p
            batch = []
    if batch:
        for doc, p in zip(batch, predict([b[:prefix_len] for b in batch]).tolist()):
            yield doc, p


# -- the spec + materialization --------------------------------------------------


def make_shard_spec(*, source: dict, routing: dict, num_paths: int, seq_len: int,
                    pack_mode: str = "pack", prefix_len: int = 32) -> dict:
    if pack_mode not in ("pack", "document"):
        raise ValueError(f"pack_mode must be 'pack' or 'document', got {pack_mode!r}")
    return {"version": SPEC_VERSION, "source": source, "routing": routing,
            "num_paths": num_paths, "seq_len": seq_len, "pack_mode": pack_mode,
            "prefix_len": prefix_len}


def spec_fingerprint(spec: dict) -> str:
    """A stable hash of a spec (tensors hashed by their bytes) for cache naming."""
    def encode(obj):
        if torch.is_tensor(obj):
            t = obj.detach().cpu().contiguous()
            return {"$tensor": hashlib.sha256(
                t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [encode(v) for v in obj]
        return obj
    blob = json.dumps(encode(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def materialize_shard(spec: dict, path_index: int, *, source=None, tokenizer=None,
                      cache_dir=None) -> torch.Tensor:
    """Build path ``path_index``'s packed ``[N, seq_len]`` shard from a spec.

    Streams the spec's documents, routes them locally, keeps this path's, and
    packs them — identical to what ``ShardedCorpus.from_documents`` would have
    produced for that path. With ``cache_dir`` the result is kept on disk (keyed
    by spec fingerprint + path), so reconnecting workers skip the re-stream.
    """
    if not (0 <= path_index < spec["num_paths"]):
        raise ValueError(f"bad path_index {path_index} for {spec['num_paths']} paths")
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"spec_{spec_fingerprint(spec)}_path{path_index:04d}.pt"
        if cache_file.exists():
            return torch.load(cache_file, weights_only=True)
    kept = [doc for doc, p in iter_assignments(spec, None, source=source, tokenizer=tokenizer)
            if p == path_index]
    to_sequences = pack_sequences if spec["pack_mode"] == "pack" else chunk_documents
    packed = to_sequences(kept, spec["seq_len"])
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(cache_file) + ".tmp"
        torch.save(packed, tmp)
        os.replace(tmp, cache_file)
    return packed


# -- the server-side corpus -------------------------------------------------------


class SpecCorpus:
    """What the server holds in spec mode: the spec + per-path token counts.

    Drop-in for the ``ShardedCorpus`` surface the schedulers actually use
    (``shard_weight``, ``num_paths``, ``has_validation``) — but it owns no
    sequence tensors, and ``shard()`` refuses: in spec mode the server ships
    recipes, never bytes.
    """

    def __init__(self, spec: dict, token_counts: dict[int, int]):
        self.spec = spec
        self.token_counts = dict(token_counts)
        self.num_paths = spec["num_paths"]
        self.seq_len = spec["seq_len"]

    @classmethod
    def from_documents(cls, spec: dict, documents) -> "SpecCorpus":
        """Build from documents already in hand (one routing pass, counts only)."""
        counts = {p: 0 for p in range(spec["num_paths"])}
        for doc, p in iter_assignments(spec, documents):
            counts[p] += int(doc.numel())
        return cls(spec, counts)

    @classmethod
    def build(cls, spec: dict, *, source=None, tokenizer=None) -> "SpecCorpus":
        """Build by streaming the spec's source (the server never holds the docs)."""
        counts = {p: 0 for p in range(spec["num_paths"])}
        for doc, p in iter_assignments(spec, None, source=source, tokenizer=tokenizer):
            counts[p] += int(doc.numel())
        return cls(spec, counts)

    def shard(self, path_index: int) -> torch.Tensor:
        raise RuntimeError(
            "SpecCorpus holds no shard bytes; servers in spec mode ship the spec "
            "and workers materialize shards locally (materialize_shard)."
        )

    @property
    def has_validation(self) -> bool:
        return False  # no per-path val split in spec mode (documented limit)

    def val_shard(self, path_index: int):
        return None

    def shard_weight(self, path_index: int) -> float:
        total = sum(self.token_counts.values())
        return self.token_counts[path_index] / max(total, 1)
