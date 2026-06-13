"""Dynamic module ownership: rendezvous placement + signed epoch records (Phase 2a).

Static ``assign_shards`` fixes the key→server map at launch; this module is the
dynamic replacement (design: ``docs/phase2-design.md``). Membership changes are
batched into **epochs**: the scheduler builds an :func:`make_epoch_record` —
the eligible owner set, the replication factor ``k``, and a placement salt —
signs it with its :class:`~opendipaco.schedule.identity.PeerIdentity` (the
Phase 1 self-certifying record machinery, ``kind="epoch"``), and everyone who
holds the same record derives the same placement:

* :func:`rank_owners` scores each owner for a key by
  ``sha256(salt ‖ key ‖ peer_id)`` (highest-random-weight hashing) — an owner
  leaving moves only its own keys, one joining steals ~1/n of each key.
* :func:`owners_for` takes the top ``k``; **rank 0 is the primary** (writes),
  ranks 1..k−1 are backups, and the rank order doubles as the succession
  order on failover (no election).

The ``salt`` is a *run-level constant* carried in the record (default ``""``).
It deliberately does **not** change across epochs — re-salting would reshuffle
every key and destroy the minimal-disruption property; change it only to
re-randomize placement for a fresh run.

Eligibility (:func:`owner_eligible`) is a predicate over verified tracker peer
records: ``public`` reachability, an ``"owner"`` role offer, and an address.
Enrollment/liveness are the tracker's admission job; Phase 3 adds reputation
to this same predicate.
"""

from __future__ import annotations

import hashlib
import time

from .identity import PeerIdentity, sign_record, verify_record

OWNER_ROLE = "owner"
DEFAULT_REPLICATION = 3


def owner_eligible(record: dict) -> bool:
    """May this (already-verified) peer record host modules?"""
    if not isinstance(record, dict) or record.get("kind") != "peer":
        return False
    return (
        record.get("reachability") == "public"
        and OWNER_ROLE in (record.get("roles") or [])
        and bool(record.get("addr"))
        and isinstance(record.get("peer_id"), str)
    )


def _score(salt: str, key: str, peer_id: str) -> bytes:
    # NUL separators so ("ab","c") and ("a","bc") can't collide.
    return hashlib.sha256(f"{salt}\x00{key}\x00{peer_id}".encode("utf-8")).digest()


def rank_owners(key: str, owners, *, salt: str = "") -> list[dict]:
    """All owners ranked for ``key``, best first (deterministic HRW order).

    ``owners`` are ``{"peer_id": ..., "addr": [host, port]}`` entries (an epoch
    record's ``owners`` list). Ties are impossible in practice (sha256), but
    the peer id breaks them deterministically anyway.
    """
    return sorted(owners, key=lambda o: (_score(salt, key, o["peer_id"]), o["peer_id"]),
                  reverse=True)


def owners_for(key: str, epoch_record: dict, k: int | None = None) -> list[dict]:
    """The replica owners for ``key`` under an epoch: rank 0 = primary."""
    if k is None:
        k = int(epoch_record.get("k", DEFAULT_REPLICATION))
    ranked = rank_owners(key, epoch_record["owners"], salt=epoch_record.get("salt", ""))
    return ranked[: max(1, k)]


# -- epoch records -------------------------------------------------------------


def make_epoch_record(identity: PeerIdentity, *, epoch: int, owner_records,
                      k: int = DEFAULT_REPLICATION, salt: str = "",
                      bootstrap: bool = False) -> dict:
    """Build + sign the authoritative owner-set record for one epoch.

    ``owner_records`` are verified tracker peer records; each must pass
    :func:`owner_eligible` (raise early rather than publish a half-usable
    epoch). Only ``peer_id`` and ``addr`` are carried, sorted by peer id so
    the signed bytes don't depend on directory iteration order.

    ``bootstrap=True`` marks the *first* epoch of a fresh (untrained) run:
    owners applying it serve their seeded ``(0, 0)`` banks immediately instead
    of syncing -- without it, a brand-new cluster would deadlock with every
    owner waiting to pull from every other. Never set it on later epochs.
    """
    owners = []
    for r in owner_records:
        if not owner_eligible(r):
            raise ValueError(f"record not owner-eligible: {r.get('peer_id')!r}")
        owners.append({"peer_id": r["peer_id"], "addr": list(r["addr"])})
    owners.sort(key=lambda o: o["peer_id"])
    return sign_record(identity, {
        "kind": "epoch",
        "epoch": int(epoch),
        "k": int(k),
        "salt": salt,
        "bootstrap": bool(bootstrap),
        "owners": owners,
        "issued_at": time.time(),
    })


def verify_epoch_record(record, *, signer_pub: str | None = None) -> bool:
    """Signature-valid, well-formed, and (when ``signer_pub`` is given) signed
    by that key — consumers pin the scheduler's public key here so a cached or
    relayed copy can't be substituted by another identity."""
    if not verify_record(record) or record.get("kind") != "epoch":
        return False
    if signer_pub is not None and record.get("pub", "").lower() != signer_pub.lower():
        return False
    if not isinstance(record.get("epoch"), int) or record["epoch"] < 0:
        return False
    if not isinstance(record.get("k"), int) or record["k"] < 1:
        return False
    if not isinstance(record.get("issued_at"), (int, float)):
        return False
    owners = record.get("owners")
    if not isinstance(owners, list):
        return False
    seen = set()
    for o in owners:
        if not (isinstance(o, dict) and isinstance(o.get("peer_id"), str) and o.get("addr")):
            return False
        if o["peer_id"] in seen:  # duplicate ids would double an owner's HRW odds
            return False
        seen.add(o["peer_id"])
    return True


def epoch_newer(a: dict, b: dict | None) -> bool:
    """Is epoch record ``a`` strictly newer than ``b``? (``b=None`` -> yes.)

    Ordered by ``(epoch, issued_at)`` so a re-issued record for the same epoch
    number (e.g. after a scheduler restart) still supersedes the old copy.
    """
    if b is None:
        return True
    return (a["epoch"], a["issued_at"]) > (b["epoch"], b["issued_at"])


class EpochManager:
    """Hysteresis + rate limiting for owner-set changes (design D5).

    Feed it directory snapshots via :meth:`observe`; it answers with the owner
    records for a *due* new epoch, or ``None``. The rules:

    * an owner joins the desired set as soon as a valid eligible record is
      seen, but **leaves only after being unseen for** ``owner_grace`` seconds
      (a flapping owner -- gone and back within the grace -- causes no bump);
    * an owner re-registering with a *different address* counts as a change;
    * bumps are rate-limited to one per ``min_epoch_interval`` seconds, so a
      burst of churn batches into a single epoch;
    * an unchanged set never bumps.

    The manager only decides *when* and *who*; signing/publishing the record
    (and choosing the bootstrap flag) stays with the scheduler.
    """

    def __init__(self, *, owner_grace: float = 240.0, min_epoch_interval: float = 60.0,
                 is_eligible=None):
        self.owner_grace = owner_grace
        self.min_epoch_interval = min_epoch_interval
        # Optional reputation gate layered on owner_eligible: ``is_eligible(peer_id)``
        # excludes demoted peers from the owner set (Phase 3b). The reputation
        # floor sits above the threshold, so a fresh honest peer still qualifies.
        self.is_eligible = is_eligible
        self._seen: dict[str, tuple[float, dict]] = {}  # peer_id -> (last seen, record)
        self._current: set | None = None                # (peer_id, addr) signature
        self._last_bump: float | None = None

    def observe(self, records, *, now: float | None = None):
        """One directory snapshot in; the next epoch's owner records out (or None)."""
        now = time.monotonic() if now is None else now
        for r in records:
            if owner_eligible(r) and (self.is_eligible is None or self.is_eligible(r["peer_id"])):
                self._seen[r["peer_id"]] = (now, r)
        expired = [p for p, (t, _) in self._seen.items() if now - t >= self.owner_grace]
        for p in expired:
            del self._seen[p]
        if not self._seen:
            return None  # never publish an ownerless epoch; wait for the swarm
        live = {p: rec for p, (_, rec) in self._seen.items()}
        signature = {(p, tuple(rec["addr"])) for p, rec in live.items()}
        if signature == self._current:
            return None
        if self._last_bump is not None and now - self._last_bump < self.min_epoch_interval:
            return None  # change is pending; re-observed (and batched) next poll
        self._current = signature
        self._last_bump = now
        return [live[p] for p in sorted(live)]
