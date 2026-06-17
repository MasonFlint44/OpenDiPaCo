"""Bandwidth throttle for a volunteer worker (W6b; design ``docs/w6-client-design.md`` D4).

A single :class:`TokenBucket` shared across all of a worker's sockets enforces a
hard ``--max-mbps`` ceiling on bytes **sent + received** (one pool, so the cap is
a true total). :class:`ThrottledSocket` is a thin proxy that consumes tokens
around the socket I/O the wire codec uses (``sendall`` / ``recv``); when the
bucket is empty those calls block, so the worker naturally back-pressures (tasks
just take longer) and never exceeds the cap. The bucket also tallies cumulative
bytes for the health line. ``None`` bucket = no throttle, byte-identical to before.
"""

from __future__ import annotations

import math
import threading
import time


# Uplink compression ordered by how much it shrinks the pseudo-gradient, so
# tailoring is never *lighter* than the run's configured base.
_COMPRESS_RANK = {"none": 0, "bf16": 1, "int8": 2, "int4": 3}


def tailor_encoding(max_mbps: float | None, *, base_compress: str = "none",
                    base_density: float = 1.0, hi: float = 20.0, lo: float = 5.0):
    """Per-worker uplink ``(compress, up_density)`` for a worker advertising a
    ``max_mbps`` budget (W6c, design D4b). Monotone in the budget and **never
    lighter than the run's base**: a worker with an ample (or no) budget gets the
    base; a tighter one gets int8, then int4 + sparser top-k. Only the *uplink*
    (pseudo-gradient) is tailored -- it's owner-decodable regardless -- so the
    shard (lossless int32 cast) and the redundant-execution digest (taken on the
    raw delta, pre-compression) are unaffected. ``down`` is left at the global:
    delta-down needs the owner's keyframe ring, so it can't be turned on per
    worker. ``max_mbps`` ``None`` / non-finite (NaN/inf) returns the base
    unchanged -- mirrors ``rate_from_mbps`` so a bad advertised budget can't pick
    a tier (or crash) instead of being ignored."""
    if max_mbps is None or not math.isfinite(max_mbps) or max_mbps >= hi:
        comp, dens = base_compress, base_density
    elif max_mbps >= lo:
        comp, dens = "int8", base_density
    else:  # tight budget (a degenerate <=0 lands here too: heaviest tier)
        comp, dens = "int4", min(base_density, 0.5)
    if _COMPRESS_RANK.get(base_compress, 0) > _COMPRESS_RANK.get(comp, 0):
        comp = base_compress                       # don't undercut the operator's base
    return comp, dens                              # dens already clamped <= base_density


def rate_from_mbps(max_mbps: float | None) -> float | None:
    """Megabits/sec -> bytes/sec, or ``None`` for "no cap". A non-positive or
    non-finite value (NaN/inf, e.g. ``--max-mbps nan``) is treated as no cap
    rather than silently producing a NaN rate that disables the throttle."""
    if max_mbps is None or not math.isfinite(max_mbps) or max_mbps <= 0:
        return None
    return max_mbps * 1e6 / 8.0


class TokenBucket:
    """Thread-safe byte token bucket: refills at ``rate`` bytes/sec, bursts up to
    ``capacity`` (default 1 s of rate). :meth:`take` blocks until ``n`` tokens are
    available — and never deadlocks for ``n`` larger than the capacity (it just
    waits proportionally longer)."""

    def __init__(self, rate_bytes_per_sec: float, *, capacity: float | None = None):
        self.rate = float(rate_bytes_per_sec)
        self.capacity = float(capacity if capacity is not None else rate_bytes_per_sec)
        self._tokens = self.capacity
        self._ts = time.monotonic()
        self._lock = threading.Lock()
        # Cumulative byte tallies for the health surface only. Updated with `+=`
        # from multiple worker threads without a lock: a rare lost update just
        # makes the displayed total drift slightly -- it never affects throttling
        # (take() is locked) and never crashes.
        self.sent_bytes = 0
        self.recv_bytes = 0

    def take(self, n: int) -> None:
        if n <= 0 or self.rate <= 0:
            return
        with self._lock:
            now = time.monotonic()
            # Refill, clamping the *positive* side to capacity (the burst limit);
            # a negative balance is debt and is never clamped, so it keeps paying
            # down over time. Consume n -- allowed to go negative, which is how a
            # request LARGER than the capacity is served (it would otherwise loop
            # forever waiting for tokens the cap never lets accrue).
            self._tokens = min(self.capacity, self._tokens + (now - self._ts) * self.rate)
            self._ts = now
            self._tokens -= n
            deficit = -self._tokens if self._tokens < 0 else 0.0
        # Sleep the deficit *outside* the lock so concurrent takers share the one
        # refill stream (the aggregate rate holds) and stop stays snappy.
        if deficit > 0:
            time.sleep(deficit / self.rate)


class ThrottledSocket:
    """Socket proxy that meters ``sendall``/``send``/``recv`` against a shared
    :class:`TokenBucket`. Everything else delegates to the wrapped socket.

    Send is charged **before** the write (so a big push waits for its tokens, not
    blasting the wire); recv is charged **after** the read (we can't know the size
    in advance, and TCP flow control turns the post-read delay into back-pressure
    on the sender)."""

    def __init__(self, sock, bucket: TokenBucket):
        self._sock = sock
        self._bucket = bucket

    def __getattr__(self, name):
        return getattr(self._sock, name)

    def sendall(self, data, *args):
        self._bucket.take(len(data))
        self._bucket.sent_bytes += len(data)
        return self._sock.sendall(data, *args)

    def send(self, data, *args) -> int:
        self._bucket.take(len(data))
        n = self._sock.send(data, *args)
        self._bucket.sent_bytes += n
        return n

    def recv(self, bufsize, *args):
        data = self._sock.recv(bufsize, *args)
        if data:
            self._bucket.recv_bytes += len(data)
            self._bucket.take(len(data))
        return data

    def close(self):
        return self._sock.close()


def throttled(sock, bucket: TokenBucket | None):
    """Wrap ``sock`` in a :class:`ThrottledSocket` when ``bucket`` is set, else
    return it unchanged (the no-cap path is byte-identical)."""
    return ThrottledSocket(sock, bucket) if bucket is not None else sock
