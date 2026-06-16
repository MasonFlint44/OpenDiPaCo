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
import json
import time

from .identity import PeerIdentity, sign_record, verify_record

OWNER_ROLE = "owner"
DEFAULT_REPLICATION = 3


def owner_addr(record: dict):
    """The address other peers dial to reach this owner: its direct ``addr``
    (a ``public`` peer) or its first relay ``/p2p-circuit`` addr (a ``nat`` peer
    reachable through relays, W1c). ``None`` if neither is present."""
    if record.get("addr"):
        return record["addr"]
    circuits = record.get("circuit_addrs") or []
    return circuits[0] if circuits else None


def owner_addrs(record: dict) -> list:
    """All addresses a dialer may try to reach this owner, in preference order:
    its direct addr (public) or *all* its relay circuit addrs (nat) — so a
    dialer can fail over across the owner's k relays (W1c)."""
    if record.get("addr"):
        return [record["addr"]]
    return list(record.get("circuit_addrs") or [])


def owner_addr_sig(record: dict) -> tuple:
    """A **hashable** signature of an owner's dialable addresses (its direct addr
    or *all* its relay circuits), for change detection. A raw NAT record has
    ``addr=None`` and only ``circuit_addrs``, so keying on ``addr`` directly
    would crash or miss relay-set changes; this keys on :func:`owner_addrs` and
    normalizes each entry (a ``[host, port]`` list -> tuple) so the result is set-
    safe and an owner's epoch bumps when any of its addresses change."""
    return tuple(tuple(a) if isinstance(a, list) else a for a in owner_addrs(record))


def owner_eligible(record: dict) -> bool:
    """May this (already-verified) peer record host modules?

    A ``public`` peer qualifies with a direct addr; a ``nat`` peer qualifies if
    it advertises at least one relay circuit addr (so a NAT'd consumer machine
    can serve as an owner, reached through a relay — the W1 goal)."""
    if not isinstance(record, dict) or record.get("kind") != "peer":
        return False
    if OWNER_ROLE not in (record.get("roles") or []) or not isinstance(
            record.get("peer_id"), str):
        return False
    if record.get("reachability") == "public" and record.get("addr"):
        return True
    return record.get("reachability") == "nat" and bool(record.get("circuit_addrs"))


def _issued_at(record: dict) -> float:
    """A record's ``issued_at`` as a float, or 0.0 if missing/malformed (so a
    tombstone freshness comparison never crashes on a bad record)."""
    ts = record.get("issued_at") if isinstance(record, dict) else None
    return float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else 0.0


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
        owners.append({"peer_id": r["peer_id"], "addr": owner_addr(r),
                       "addrs": owner_addrs(r)})
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


def _members_sig(owners) -> str:
    """A stable hash of an owner set (peer_id + addr), so identical membership
    maps to the same epoch on every node (Phase 4 D6)."""
    payload = [[o["peer_id"], o["addr"]] for o in owners]  # addr is a list (public) or str (nat)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def derive_epoch(owner_records, *, k: int = DEFAULT_REPLICATION, salt: str = "",
                 prev: dict | None = None, is_eligible=None) -> dict:
    """Derive the owner-set epoch **deterministically from the directory** — no
    central signer (Phase 4 D6). Every owner holding the same (gossiped,
    self-certifying) directory computes the same record: the eligible owners
    sorted by peer id, a membership hash, and an epoch number that bumps only
    when membership changes (so unchanged churn re-derives the *same* record and
    the version gate stays stable).

    ``is_eligible(peer_id)`` layers the reputation gate on
    :func:`owner_eligible`, so an owner debited for serving divergent weights
    (4c) is excluded from the next derived epoch — this is the eviction step.
    Returns an **unsigned** record (``deterministic: True``); authority comes
    from every node recomputing it, not from a signature, so a relayed copy is
    only ever a hint to be re-derived and matched.

    **Never** flagged ``bootstrap``: every node that starts derives its first
    epoch with ``prev=None``, so a ``prev is None -> bootstrap`` rule would make
    a peer *joining a running cluster* boot-serve its seeded ``(0, 0)`` bank as
    authoritative for keys the cluster has already trained — serving stale
    state. Gained keys therefore always **sync**; a genuine cold start (every
    owner identical at ``(0, 0)``) self-activates through the same equal-version
    reconciliation a full-cluster restart uses (Phase 2 2d), with no central
    bootstrap signal to forge or get wrong.
    """
    owners = []
    for r in owner_records:
        if owner_eligible(r) and (is_eligible is None or is_eligible(r["peer_id"])):
            owners.append({"peer_id": r["peer_id"], "addr": owner_addr(r),
                           "addrs": owner_addrs(r)})
    owners.sort(key=lambda o: o["peer_id"])
    sig = _members_sig(owners)
    if prev is not None and prev.get("members_sig") == sig and int(prev.get("k", k)) == int(k):
        return prev  # membership unchanged -> the very same epoch
    epoch_num = 0 if prev is None else int(prev["epoch"]) + 1
    return {
        "kind": "epoch", "epoch": epoch_num, "k": int(k), "salt": salt,
        "bootstrap": False, "owners": owners, "members_sig": sig,
        "deterministic": True, "issued_at": time.time(),
    }


def _epoch_well_formed(record) -> bool:
    if record.get("kind") != "epoch":
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


def verify_epoch_record(record, *, signer_pub: str | None = None,
                        allow_deterministic: bool = False) -> bool:
    """Signature-valid, well-formed, and (when ``signer_pub`` is given) signed
    by that key — consumers pin the scheduler's public key here so a cached or
    relayed copy can't be substituted by another identity.

    With ``allow_deterministic`` (decentralized mode, Phase 4 D6) a
    signer-less ``deterministic`` epoch is accepted on **structure** alone: its
    authority is that the consumer re-derives it from its own directory, so
    there is no signature to check. A *signed* record still takes the signed
    path even under this flag."""
    if not isinstance(record, dict):
        return False
    if allow_deterministic and record.get("deterministic") and "sig" not in record:
        return _epoch_well_formed(record)
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
    * an owner that has **explicitly, signed-deregistered** (a tracker
      tombstone, passed as ``tombstoned=``) leaves **immediately**, skipping the
      grace -- a graceful departure (W4b) shouldn't wait out the silence timer;
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

    def observe(self, records, *, tombstoned=None, now: float | None = None):
        """One directory snapshot in; the next epoch's owner records out (or None).

        ``tombstoned`` maps peer id -> the ``issued_at`` of that peer's signed
        deregister. A tombstoned peer is dropped from the desired set **now**,
        bypassing ``owner_grace`` (a clean leave fails over fast) -- but **only
        when the deregister supersedes the registration we hold**. A replayed
        *stale* tombstone (issued before the peer's current record -- e.g. an
        owner that gracefully left and later rejoined) must not fast-evict the
        live peer, so we compare ``issued_at`` rather than trusting the bare id.
        """
        now = time.monotonic() if now is None else now
        for r in records:
            if owner_eligible(r) and (self.is_eligible is None or self.is_eligible(r["peer_id"])):
                self._seen[r["peer_id"]] = (now, r)
        # Done after ingesting ``records`` so a fresh tombstone wins over a stale
        # record that still advertises the peer (relayed before the deregister),
        # while a stale tombstone loses to the fresher registration we just saw.
        for p, ts in dict(tombstoned or {}).items():
            seen = self._seen.get(p)
            if seen is not None and (ts is None or ts > _issued_at(seen[1])):
                del self._seen[p]
        expired = [p for p, (t, _) in self._seen.items() if now - t >= self.owner_grace]
        for p in expired:
            del self._seen[p]
        if not self._seen:
            return None  # never publish an ownerless epoch; wait for the swarm
        live = {p: rec for p, (_, rec) in self._seen.items()}
        signature = {(p, owner_addr_sig(rec)) for p, rec in live.items()}
        if signature == self._current:
            return None
        if self._last_bump is not None and now - self._last_bump < self.min_epoch_interval:
            return None  # change is pending; re-observed (and batched) next poll
        self._current = signature
        self._last_bump = now
        return [live[p] for p in sorted(live)]
