"""Turn raw text into the per-document token tensors the rest of the stack uses.

The whole pipeline downstream of here -- ``ShardedCorpus.from_documents``,
routing, packing -- consumes a ``list[torch.LongTensor]``, one 1-D tensor of
token ids per document. This module is the (network-free, unit-testable) bridge
from strings to that representation; ``c4.py`` builds on it for the real corpus.
"""

from __future__ import annotations

from typing import Iterable

import torch

_TOKENIZER_HINT = (
    "A HuggingFace tokenizer is required. Install the data extra:\n"
    '    pip install -e ".[data]"'
)


def load_tokenizer(name: str = "t5-base"):
    """Load a pretrained HuggingFace tokenizer (lazy import).

    The default ``t5-base`` has a ~32k SentencePiece vocabulary, a convenient proxy
    for the paper's vocab. For a closer match, :func:`train_tokenizer` trains a fresh
    ~32k tokenizer *on your data* (what DiPaCo actually does). Either way, set
    ``BackboneConfig(vocab_size=tokenizer.vocab_size)`` so the two can't drift.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(_TOKENIZER_HINT) from e
    return AutoTokenizer.from_pretrained(name)


def train_tokenizer(
    texts: Iterable[str],
    *,
    vocab_size: int = 32000,
    model: str = "unigram",
    unk_token: str = "<unk>",
    eos_token: str = "</s>",
    pad_token: str = "<pad>",
):
    """Train a fresh subword tokenizer on a text sample (paper-faithful).

    DiPaCo trains its own ~32k SentencePiece vocabulary on the corpus rather than
    borrowing another model's; this does the same, returning a
    ``PreTrainedTokenizerFast`` that is a drop-in for :func:`tokenize_documents`
    (it has ``.encode(text, add_special_tokens=False)`` and ``.eos_token_id``).

    * ``model="unigram"`` — SentencePiece-style Unigram (the paper's family).
    * ``model="bpe"`` — byte-level BPE (GPT-style), no ``<unk>``.

    The realized vocabulary may be smaller than ``vocab_size`` on a small sample;
    read it back from ``tokenizer.vocab_size`` and use it for ``BackboneConfig``.
    """
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(_TOKENIZER_HINT) from e

    specials = [unk_token, eos_token, pad_token]
    if model == "unigram":
        tok = Tokenizer(models.Unigram())
        tok.pre_tokenizer = pre_tokenizers.Metaspace()  # SentencePiece-style word marking
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size, special_tokens=specials, unk_token=unk_token
        )
    elif model == "bpe":
        tok = Tokenizer(models.BPE(unk_token=unk_token))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials)
    else:
        raise ValueError(f"model must be 'unigram' or 'bpe', got {model!r}")

    tok.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token=unk_token, eos_token=eos_token, pad_token=pad_token,
    )


def split_documents(documents, *, val_fraction: float = 0.0, test_fraction: float = 0.0,
                    seed: int = 0):
    """Deterministically shuffle and split a document list into (train, val, test).

    A held-out split for honest evaluation (e.g. a dense-baseline comparison). The
    shuffle is seeded so the split is reproducible across runs and processes.
    """
    import random

    docs = list(documents)
    random.Random(seed).shuffle(docs)
    n = len(docs)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test = docs[:n_test]
    val = docs[n_test:n_test + n_val]
    train = docs[n_test + n_val:]
    return train, val, test


def tokenize_one(
    text: str,
    tokenizer,
    *,
    add_eos: bool = True,
    max_doc_tokens: int | None = None,
    min_doc_tokens: int = 1,
) -> torch.Tensor | None:
    """Tokenize a single string to a 1-D ``LongTensor``, or ``None`` if filtered.

    Same truncation/eos/min-length rules as :func:`tokenize_documents` for one
    document, so the streaming ingestion path produces identical tokens to the
    batch path. Returns ``None`` when the result is shorter than ``min_doc_tokens``.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None) if add_eos else None
    ids = tokenizer.encode(text, add_special_tokens=False)
    if max_doc_tokens is not None and len(ids) > max_doc_tokens:
        # Leave room for the eos so the cap is a true upper bound.
        ids = ids[: max_doc_tokens - 1] if eos_id is not None else ids[:max_doc_tokens]
    if eos_id is not None:
        ids = ids + [eos_id]
    if len(ids) < max(min_doc_tokens, 1):
        return None
    return torch.tensor(ids, dtype=torch.long)


def tokenize_documents(
    texts: Iterable[str],
    tokenizer,
    *,
    add_eos: bool = True,
    max_doc_tokens: int | None = None,
    min_doc_tokens: int = 1,
) -> list[torch.Tensor]:
    """Tokenize an iterable of strings into per-document ``LongTensor``s.

    * ``add_eos`` appends the tokenizer's eos id (when it has one) so packed
      windows carry a document boundary.
    * ``max_doc_tokens`` truncates each document to at most that many tokens.
    * ``min_doc_tokens`` drops documents shorter than this after tokenizing
      (default 1 -> drop only empties).

    Returns a list of 1-D ``torch.long`` tensors; documents that tokenize to
    nothing are skipped, so the list may be shorter than ``texts``.
    """
    docs: list[torch.Tensor] = []
    for text in texts:
        t = tokenize_one(text, tokenizer, add_eos=add_eos,
                         max_doc_tokens=max_doc_tokens, min_doc_tokens=min_doc_tokens)
        if t is not None:
            docs.append(t)
    return docs
