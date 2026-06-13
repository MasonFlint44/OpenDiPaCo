"""Leaderless work assignment + version-vector staleness (Phase 4a).

The central :class:`~opendipaco.schedule.sharded.Scheduler` hands each free path
to whichever worker asks (a queue + a global ``_T`` clock + per-path
``_completed``). Decentralized, there is no queue and no global clock — every
worker computes its **own** assignment, and staleness is read off the
per-module version vectors that already exist (design ``docs/phase4-design.md``,
D1/D2):

* :func:`rank_workers` scores each live worker for a ``(path, generation)`` slot
  by ``sha256(salt ‖ path ‖ generation ‖ peer_id)`` — the same
  highest-random-weight idea as owner placement
  (:func:`~opendipaco.schedule.ownership.rank_owners`), but re-rolled **per
  generation** so paths spread across the pool over time and a dead worker
  stalls only one generation of one path.
* :func:`responsible_rank` turns "how long has this generation been open" into
  the rank that currently owns the slot: rank 0 holds it for ``lease_ttl``
  seconds, then rank 1 takes over, then rank 2, … — deterministic succession,
  no election (the same shape as owner promotion). A commit landing (the
  primary owner advancing the generation) is what frees the slot; everyone
  observes it by reading the owner's generation counter.
* :func:`is_assignee` composes the two: a worker trains ``(path, generation)``
  exactly when it holds the currently-responsible rank.

The generation counter and the ``(epoch, counter)`` module versions live on the
**owners** (Phase 2); this module is pure functions over values a worker reads
from them, so it is fully testable in isolation and identical on every node.
"""

from __future__ import annotations

import hashlib


def _path_str(path) -> str:
    """Canonical string for a path tuple, NUL-free and order-stable."""
    return "-".join(str(int(i)) for i in path)


def _wscore(salt: str, path, generation: int, peer_id: str) -> bytes:
    # NUL separators so distinct (path, gen, id) can't alias each other.
    token = f"{salt}\x00{_path_str(path)}\x00{int(generation)}\x00{peer_id}"
    return hashlib.sha256(token.encode("utf-8")).digest()


def rank_workers(path, generation: int, workers, *, salt: str = "") -> list[str]:
    """Live workers ranked for the ``(path, generation)`` slot, best first.

    ``workers`` is an iterable of peer-id strings (or ``{"peer_id": ...}``
    records, like a directory snapshot). Deterministic HRW order; ties (never,
    in practice — sha256) break on the peer id.
    """
    ids = [w["peer_id"] if isinstance(w, dict) else w for w in workers]
    return sorted(ids, key=lambda pid: (_wscore(salt, path, generation, pid), pid),
                  reverse=True)


def responsible_rank(elapsed: float, lease_ttl: float, n_workers: int) -> int:
    """Which rank currently owns a slot, given how long it has been open.

    Rank 0 holds the slot for ``lease_ttl`` seconds; if the generation hasn't
    advanced by then, rank 1 takes over for the next window, and so on, capped
    at the last live worker (after which the slot just waits for *someone* to
    finish — no one is dropped). ``elapsed`` is ``now − generation_opened_at``,
    both read locally; no coordination needed.
    """
    if n_workers <= 0:
        return 0
    if lease_ttl <= 0:
        return 0
    step = int(max(0.0, elapsed) // lease_ttl)
    return min(step, n_workers - 1)


def assignee(path, generation: int, workers, *, salt: str = "",
             elapsed: float = 0.0, lease_ttl: float = 0.0) -> str | None:
    """The peer id currently responsible for ``(path, generation)`` (or None if
    no live workers). With ``elapsed``/``lease_ttl`` defaulted, this is rank 0."""
    ranked = rank_workers(path, generation, workers, salt=salt)
    if not ranked:
        return None
    return ranked[responsible_rank(elapsed, lease_ttl, len(ranked))]


def is_assignee(peer_id: str, path, generation: int, workers, *, salt: str = "",
                elapsed: float = 0.0, lease_ttl: float = 0.0) -> bool:
    """Does ``peer_id`` currently own the ``(path, generation)`` slot?"""
    return peer_id is not None and assignee(
        path, generation, workers, salt=salt, elapsed=elapsed, lease_ttl=lease_ttl
    ) == peer_id


# -- version-vector staleness --------------------------------------------------


def _pair(v) -> tuple[int, int]:
    return (int(v[0]), int(v[1])) if v is not None else (0, 0)


def version_lag(fetched: dict, current: dict) -> int | None:
    """Staleness of a contribution from its fetched-vs-current base versions.

    Replaces the global ``_T − _issued[path]`` serialization-order staleness
    with a *local* measure the owner can compute: how far the ``(epoch,
    counter)`` versions the worker trained against lag the versions now. Returns
    the **max counter lag** across the path's keys when every key is in the
    **same epoch** in both maps; returns ``None`` — meaning "treat as
    over-bound, drop it" — if any key crossed an epoch boundary (a failover /
    remap happened mid-task, so the base is from a superseded ownership view)
    or is missing from ``current`` (remapped away). Cross-epoch drop matches
    Phase 2's "pushes in flight against the old primary fail" semantics.
    """
    if not fetched:
        return 0
    lag = 0
    for key, fv in fetched.items():
        if key not in current:
            return None  # key remapped away; base no longer authoritative
        fe, fc = _pair(fv)
        ce, cc = _pair(current[key])
        if fe != ce:
            return None  # crossed an epoch boundary -> conservatively stale
        d = cc - fc
        if d < 0:
            return None  # fetched a *newer* base than current? inconsistent -> drop
        lag = max(lag, d)
    return lag


# -- which owner coordinates a path's generation -------------------------------


def coordinator_key(path_keys) -> str:
    """The key whose primary owner coordinates a path's generation counter.

    Prefer the path's **private** module key (every path has exactly one
    embedding+head pair that is uniquely its own under the private policy);
    fall back to the lowest-sorted shared key for a fully-shared path. Pure
    function of the path's key set, so every node agrees without coordination.
    """
    from ..topology import is_private_key

    keys = sorted(path_keys)
    private = [k for k in keys if is_private_key(k)]
    return private[0] if private else keys[0]


def path_primary(path_keys, epoch_record) -> dict | None:
    """The owner record that coordinates this path under the given epoch (the
    primary — HRW rank 0 — of the path's :func:`coordinator_key`)."""
    from .ownership import owners_for

    owners = owners_for(coordinator_key(path_keys), epoch_record)
    return owners[0] if owners else None
