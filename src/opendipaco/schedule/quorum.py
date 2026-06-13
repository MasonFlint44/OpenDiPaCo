"""Quorum reads + cross-owner digest agreement (Phase 4c: Byzantine owners).

The decentralized swarm has no trusted owner, so a key's authoritative value is
whatever a **majority of its ``k`` replicas agree on**, never any single
owner's word (design ``docs/phase4-design.md`` D4). Two uses of the same idea:

* **Reads** — a worker (or a co-owner cold-syncing) fetches each replica's cheap
  ``(version, digest)``, takes :func:`confirm_version` (the highest version a
  quorum agree on), and downloads the *weights* only from a replica whose digest
  matches. A lone Byzantine owner serving poisoned bytes is outvoted: its digest
  is in the minority, so its bytes are never accepted.
* **Writes** — owners cross-check each other: an owner whose digest at a
  *confirmed* version contradicts the majority (:func:`divergent_peers`) is
  computing or serving something wrong, and its owner-behaviour reputation is
  debited toward eviction.

These are pure functions over ``(version, digest)`` reports — the network
fetching is the caller's (the owner replication loop, or the worker fetch) — so
the agreement rule is identical and testable everywhere.
"""

from __future__ import annotations


def _pair(v) -> tuple:
    return tuple(v) if isinstance(v, (tuple, list)) else (0, int(v))


def confirm_version(reports, quorum: int):
    """The ``(version, digest)`` at the **highest version** that at least
    ``quorum`` of ``reports`` agree on, or ``None`` if none reaches quorum.

    ``reports`` is an iterable of ``(version, digest)`` from distinct replicas.
    Versions are ``(epoch, counter)`` pairs, compared as tuples. A reader trusts
    only weights matching the returned digest; if the top version isn't agreed
    (replicas mid-sync), this returns the highest *older* version that is —
    liveness over freshness, the same trade as the replication loss window.
    """
    tally: dict = {}
    for v, d in reports:
        key = (_pair(v), d)
        tally[key] = tally.get(key, 0) + 1
    winners = [(v, d) for (v, d), n in tally.items() if n >= max(1, quorum)]
    if not winners:
        return None
    return max(winners, key=lambda vd: vd[0])  # highest agreed version


def divergent_peers(reports_by_peer: dict, confirmed) -> set:
    """Peers whose report **directly contradicts** the confirmed value — they
    report the confirmed version but a different digest.

    Conservative by design: a peer reporting an *older* version is merely behind
    (it will pull forward), and a peer reporting a *newer* version no one else
    has is unconfirmable, not yet wrong — neither is flagged. Only a same-version
    digest mismatch is a provable divergence, so an honest replica that is just
    lagging is never falsely blamed.
    """
    if confirmed is None:
        return set()
    cv, cd = confirmed
    cv = _pair(cv)
    bad = set()
    for pid, rep in reports_by_peer.items():
        if rep is None:
            continue
        v, d = _pair(rep[0]), rep[1]
        if v == cv and d != cd:
            bad.add(pid)
    return bad


def read_quorum_versions(addrs, keys, quorum: int, rpc) -> dict:
    """Confirmed ``{key: (version, digest)}`` across the replica ``addrs``.

    ``rpc(addr, msg)`` performs one request (the caller supplies transport);
    each replica answers a ``digest`` request with ``{key: [version, digest]}``.
    Unreachable replicas are skipped. A key with no quorum is omitted (the
    reader falls back to an older confirmed version, or waits).
    """
    per_key: dict = {k: [] for k in keys}
    for addr in addrs:
        try:
            reply = rpc(addr, {"type": "digest", "keys": list(keys)})
        except (OSError, ConnectionError):
            continue
        for k, vd in ((reply or {}).get("digests") or {}).items():
            if k in per_key and vd:
                per_key[k].append((vd[0], vd[1]))
    out = {}
    for k, reports in per_key.items():
        c = confirm_version(reports, quorum)
        if c is not None:
            out[k] = c
    return out
