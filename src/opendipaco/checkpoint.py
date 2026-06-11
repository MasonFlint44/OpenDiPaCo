"""Checkpoint / resume for :class:`DiPaCoEngine`.

A checkpoint is a **directory**, not a single file, because in the distributed
setting each rank owns only a slice of the module bank (see ``SyncBackend``).
Each rank writes its own ``rank{r}.pt`` (its ``engine.state_dict()``), plus a
shared ``meta.json`` and an optional ``corpus.pt`` so a resumed run can recover
its exact shards without re-routing. Single-process runs are just ``rank0.pt``.

Writes are atomic: everything lands in a sibling ``*.tmp`` directory that is
renamed into place only once complete.

    from opendipaco.checkpoint import save_checkpoint, load_checkpoint

    save_checkpoint(engine, "ckpts/round100", corpus=corpus)
    ...
    extra = load_checkpoint(engine, "ckpts/round100")   # restores into `engine`
"""

from __future__ import annotations

import json
import os
from pathlib import Path as FsPath

import torch

__all__ = ["save_checkpoint", "load_checkpoint", "latest_checkpoint"]


def _rank_of(engine) -> int:
    return int(getattr(engine.backend, "rank", 0))


def save_checkpoint(engine, dirpath: str | os.PathLike, *, corpus=None, extra=None) -> str:
    """Write ``engine``'s state for this rank into the checkpoint directory.

    Rank 0 also writes ``meta.json`` and (if given) ``corpus.pt`` / ``extra.pt``.
    The whole directory is built under ``<dirpath>.tmp`` and renamed into place,
    so a crashed write never leaves a half-written checkpoint.

    Returns the final directory path.
    """
    dirpath = FsPath(dirpath)
    rank = _rank_of(engine)
    state = engine.state_dict()

    # In single-process runs we can build the whole dir atomically. In
    # multi-rank runs every rank writes into the same final directory (their
    # filenames don't collide); a per-rank tmp+rename keeps each file atomic.
    dirpath.mkdir(parents=True, exist_ok=True)
    rank_file = dirpath / f"rank{rank}.pt"
    tmp = rank_file.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, rank_file)

    if rank == 0:
        meta = {
            "format": state["format"],
            "fingerprint": state["fingerprint"],
            "world_size": state["world_size"],
            "materialize": state["materialize"],
            "global_round": state["global_round"],
            "total_rounds": state["total_rounds"],
        }
        meta_tmp = dirpath / "meta.json.tmp"
        meta_tmp.write_text(json.dumps(meta, indent=2))
        os.replace(meta_tmp, dirpath / "meta.json")
        if corpus is not None:
            _atomic_save(corpus, dirpath / "corpus.pt")
        if extra is not None:
            _atomic_save(extra, dirpath / "extra.pt")

    engine.backend.barrier()
    return str(dirpath)


def load_checkpoint(engine, dirpath: str | os.PathLike, *, strict: bool = True) -> dict:
    """Restore this rank's slice from ``dirpath`` into ``engine`` (in place).

    Returns a dict that may contain ``"corpus"`` and/or ``"extra"`` if they were
    saved, so the caller can resume from the exact shards:

        out = load_checkpoint(engine, "ckpts/round100")
        corpus = out.get("corpus")
    """
    dirpath = FsPath(dirpath)
    rank = _rank_of(engine)
    rank_file = dirpath / f"rank{rank}.pt"
    if not rank_file.exists():
        raise FileNotFoundError(f"no checkpoint shard for rank {rank} at {rank_file}")

    state = torch.load(rank_file, map_location="cpu", weights_only=False)
    engine.load_state_dict(state, strict=strict)

    out: dict = {}
    corpus_file = dirpath / "corpus.pt"
    if corpus_file.exists():
        out["corpus"] = torch.load(corpus_file, map_location="cpu", weights_only=False)
    extra_file = dirpath / "extra.pt"
    if extra_file.exists():
        out["extra"] = torch.load(extra_file, map_location="cpu", weights_only=False)
    return out


def latest_checkpoint(root: str | os.PathLike) -> str | None:
    """Return the highest-round checkpoint directory under ``root`` (or ``None``).

    Considers immediate subdirectories that contain a ``meta.json``, ranking by
    its ``global_round``.
    """
    root = FsPath(root)
    if not root.is_dir():
        return None
    best: tuple[int, str] | None = None
    for child in root.iterdir():
        meta = child / "meta.json"
        if child.is_dir() and meta.exists():
            try:
                rnd = json.loads(meta.read_text()).get("global_round", -1)
            except (ValueError, OSError):
                continue
            if best is None or rnd > best[0]:
                best = (rnd, str(child))
    return best[1] if best else None


def _atomic_save(obj, path: FsPath) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
