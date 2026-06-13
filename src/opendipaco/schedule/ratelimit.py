"""Per-peer token-bucket rate limiting (Phase 3b, finding 1.14).

Beyond ``max_connections`` and the message-size cap, an authenticated worker can
loop ``request`` to force the scheduler to emit large task payloads — bandwidth
amplification. A per-``peer_id`` token bucket bounds the rate of *expensive*
replies (a task carries weights/shard; an ``idle`` reply is cheap and costs no
token), so a flooding peer is throttled to backoff-paced idles without being
disconnected.

Reputation scales the bucket: a trusted peer gets a larger, faster-refilling
bucket and is effectively never limited in normal operation, while a fresh or
demoted peer gets a tighter one. Anonymous peers (``peer_id is None``, HMAC
trusted-cluster mode) are never limited — abuse protection is the
open-enrollment defense.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, *, capacity: float = 8.0, refill_per_sec: float = 2.0):
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self._buckets: dict[str, list] = {}  # peer_id -> [tokens, last_ts]
        self._lock = threading.Lock()

    @staticmethod
    def _scale(reputation: float) -> float:
        # Map a [0,1] reputation to a bucket multiplier in [0.5, 1.5]: a floor
        # peer (0.5) gets 1.0x, a proven peer (1.0) 1.5x, a demoted one (0) 0.5x.
        return 0.5 + reputation

    def allow(self, peer_id: str | None, *, reputation: float = 0.5,
              cost: float = 1.0) -> bool:
        """Consume ``cost`` tokens; return whether the (expensive) reply may go.

        Anonymous peers always pass. A peer with an empty bucket is denied (the
        caller sends a cheap backoff idle instead), and the bucket refills over
        time so the throttle lifts on its own.
        """
        if peer_id is None:
            return True
        scale = self._scale(reputation)
        cap = self.capacity * scale
        now = time.monotonic()
        with self._lock:
            entry = self._buckets.get(peer_id)
            tokens, ts = (cap, now) if entry is None else entry
            tokens = min(cap, tokens + (now - ts) * self.refill * scale)
            if tokens < cost:
                self._buckets[peer_id] = [tokens, now]
                return False
            self._buckets[peer_id] = [tokens - cost, now]
            return True
