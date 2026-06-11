"""Real-corpus (C4) loader for DiPaCo training.

Streams documents from `allenai/c4` (the corpus the paper trains on), tokenizes
them with a HuggingFace tokenizer, and returns the ``list[torch.LongTensor]``
that ``ShardedCorpus.from_documents`` and the routers consume. A small on-disk
cache lets repeated runs skip the download + tokenization.

Everything that actually touches the network or HuggingFace is imported lazily,
so importing :mod:`opendipaco.data` never requires the ``[data]`` extra.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .text import load_tokenizer, tokenize_documents

_DATASETS_HINT = (
    "The 'datasets' package is required to stream C4. Install the data extra:\n"
    '    pip install -e ".[data]"'
)


def save_documents(documents: list[torch.Tensor], path: str | os.PathLike) -> None:
    """Persist a tokenized document list with an atomic temp+rename write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(documents, tmp)
    os.replace(tmp, path)


def load_documents(path: str | os.PathLike) -> list[torch.Tensor]:
    """Load a tokenized document list written by :func:`save_documents`."""
    return torch.load(path, weights_only=False)


def load_c4_documents(
    *,
    split: str = "train",
    num_documents: int = 10_000,
    tokenizer=None,
    tokenizer_name: str = "t5-base",
    streaming: bool = True,
    max_doc_tokens: int | None = None,
    min_doc_tokens: int = 1,
    add_eos: bool = True,
    cache_path: str | os.PathLike | None = None,
) -> list[torch.Tensor]:
    """Stream ``num_documents`` C4 documents and tokenize them.

    Parameters mirror the rest of the data API. ``tokenizer`` may be a
    pre-loaded HuggingFace tokenizer; otherwise one is loaded from
    ``tokenizer_name`` (default ``t5-base``, a ~32k vocab matching the paper).

    If ``cache_path`` is set and the file exists it is loaded directly (no
    download, no tokenization); otherwise the result is computed and written
    there for next time. Returns ``list[torch.LongTensor]``.
    """
    if cache_path is not None and Path(cache_path).exists():
        return load_documents(cache_path)

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(_DATASETS_HINT) from e

    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_name)

    ds = load_dataset("allenai/c4", "en", split=split, streaming=streaming)

    def texts():
        for i, row in enumerate(ds):
            if i >= num_documents:
                break
            yield row["text"]

    documents = tokenize_documents(
        texts(),
        tokenizer,
        add_eos=add_eos,
        max_doc_tokens=max_doc_tokens,
        min_doc_tokens=min_doc_tokens,
    )

    if cache_path is not None:
        save_documents(documents, cache_path)
    return documents
