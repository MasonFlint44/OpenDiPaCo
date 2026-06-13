"""Per-peer reputation (Phase 3b, finding 1.1).

Authentication proves *who* a peer is; reputation tracks *how it has behaved*.
The scheduler maintains a score in ``[0, 1]`` per authenticated ``peer_id`` from
the signals it already sees — accepted vs. rejected commits today, redundant-
execution agreement in 3c — and uses it to gate owner eligibility, lease
priority, and rate-limit generosity.

The Sybil story (identities are free to mint): a *fresh* peer starts at
``floor`` (not the maximum), so minting a thousand identities buys a thousand
floor-scored contributors, not a thousand trusted ones. Influence above the
floor is **earned**; a peer that misbehaves is driven below it and demoted out
of the owner set. The floor sits *above* the owner-eligibility threshold so a
brand-new honest peer can still bootstrap a fresh cluster — reputation
*excludes proven-bad* peers, it does not *require a track record* to start.

Scores **decay toward the floor** with a half-life, applied lazily on access:
a peer can't bank reputation and then defect cost-free, and a demoted peer that
goes quiet recovers to neutral over time rather than being branded forever.

This is scheduler-owned and HMAC/anonymous-agnostic: a contribution with no
identity (``peer_id is None``, i.e. an HMAC trusted-cluster deployment) is not
tracked — reputation is the open-enrollment defense, off by construction when
everyone already shares a secret.
"""

from __future__ import annotations

import threading
import time


class Reputation:
    def __init__(self, *, floor: float = 0.5, credit: float = 0.02,
                 debit: float = 0.2, decay_halflife: float = 3600.0,
                 lo: float = 0.0, hi: float = 1.0):
        if not lo <= floor <= hi:
            raise ValueError("floor must be within [lo, hi]")
        self.floor = floor
        self.credit_step = credit
        self.debit_step = debit
        self.decay_halflife = decay_halflife
        self.lo, self.hi = lo, hi
        self._scores: dict[str, list] = {}  # peer_id -> [score, last_ts]
        self._lock = threading.Lock()

    def _decayed_locked(self, peer_id: str, now: float) -> float:
        entry = self._scores.get(peer_id)
        if entry is None:
            return self.floor
        score, ts = entry
        if self.decay_halflife > 0 and now > ts:
            # Geometric decay toward the floor (half the distance per half-life).
            score = self.floor + (score - self.floor) * 0.5 ** ((now - ts) / self.decay_halflife)
        return score

    def _set_locked(self, peer_id: str, score: float, now: float) -> None:
        self._scores[peer_id] = [min(self.hi, max(self.lo, score)), now]

    def get(self, peer_id: str | None) -> float:
        if peer_id is None:
            return self.floor
        now = time.monotonic()
        with self._lock:
            return self._decayed_locked(peer_id, now)

    def _adjust(self, peer_id: str | None, delta: float) -> float:
        if peer_id is None:
            return self.floor  # anonymous/HMAC: untracked
        now = time.monotonic()
        with self._lock:
            self._set_locked(peer_id, self._decayed_locked(peer_id, now) + delta, now)
            return self._scores[peer_id][0]

    def credit(self, peer_id: str | None) -> float:
        """Reward a validated/accepted contribution."""
        return self._adjust(peer_id, self.credit_step)

    def debit(self, peer_id: str | None) -> float:
        """Penalize a rejected/disagreeing contribution (debit > credit, so
        misbehavior costs more than a single good act repays)."""
        return self._adjust(peer_id, -self.debit_step)

    def eligible(self, peer_id: str | None, threshold: float) -> bool:
        """Is this peer at or above ``threshold`` (owner-eligibility gate)?

        Anonymous peers (no identity) are eligible: an HMAC deployment is a
        trusted cluster where reputation does not apply.
        """
        if peer_id is None:
            return True
        return self.get(peer_id) >= threshold

    def snapshot(self) -> dict:
        """Decayed scores as a plain ``{peer_id: score}`` dict for checkpointing."""
        now = time.monotonic()
        with self._lock:
            return {pid: self._decayed_locked(pid, now) for pid in self._scores}

    def restore(self, scores: dict) -> None:
        now = time.monotonic()
        with self._lock:
            for pid, score in (scores or {}).items():
                self._set_locked(str(pid), float(score), now)
