"""Tests for the W6b bandwidth throttle (docs/w6-client-design.md D4a)."""

import time

from opendipaco.schedule.throttle import (
    ThrottledSocket,
    TokenBucket,
    rate_from_mbps,
    tailor_encoding,
    throttled,
)


def test_tailor_encoding_ladder():
    # Ample / no budget -> base, unchanged.
    assert tailor_encoding(None, base_compress="none", base_density=1.0) == ("none", 1.0)
    assert tailor_encoding(50, base_compress="none", base_density=1.0) == ("none", 1.0)
    # Moderate budget -> int8, dense.
    assert tailor_encoding(8, base_compress="none", base_density=1.0) == ("int8", 1.0)
    # Tight budget -> int4 + sparser top-k.
    assert tailor_encoding(2, base_compress="none", base_density=1.0) == ("int4", 0.5)


def test_tailor_encoding_never_lighter_than_base():
    # An int4 run stays int4 even at an ample budget (don't undercut the operator).
    assert tailor_encoding(100, base_compress="int4", base_density=1.0)[0] == "int4"
    # A run already sparser than the tier keeps the tighter density.
    assert tailor_encoding(2, base_compress="none", base_density=0.25)[1] == 0.25
    # int8 base + tight budget escalates to int4.
    assert tailor_encoding(2, base_compress="int8", base_density=1.0)[0] == "int4"


def test_rate_from_mbps():
    assert rate_from_mbps(8) == 1e6          # 8 Mbit/s = 1 MB/s
    assert rate_from_mbps(None) is None
    assert rate_from_mbps(0) is None
    assert rate_from_mbps(-5) is None
    # Non-finite (e.g. `--max-mbps nan`/inf) must not yield a NaN rate that
    # silently disables the throttle -- treated as no cap.
    assert rate_from_mbps(float("nan")) is None
    assert rate_from_mbps(float("inf")) is None


def test_token_bucket_enforces_rate():
    # No burst (capacity 0): taking `rate/2` bytes must wait ~0.5s at `rate`/s.
    rate = 20000.0
    tb = TokenBucket(rate, capacity=0)
    t0 = time.monotonic()
    tb.take(int(rate * 0.4))
    dt = time.monotonic() - t0
    assert 0.3 < dt < 1.0                     # ~0.4s, generous for CI jitter


def test_token_bucket_burst_is_immediate():
    tb = TokenBucket(10000.0)                 # capacity defaults to 1s of rate
    t0 = time.monotonic()
    tb.take(10000)                            # the full burst -> no wait
    assert time.monotonic() - t0 < 0.1


def test_take_larger_than_capacity_does_not_deadlock():
    tb = TokenBucket(50000.0, capacity=1000)  # ask for 10x capacity
    t0 = time.monotonic()
    tb.take(10000)                            # ~0.2s, must complete (not hang)
    assert 0.05 < time.monotonic() - t0 < 1.0


def test_take_is_a_noop_when_rate_is_zero():
    tb = TokenBucket(0.0)
    t0 = time.monotonic()
    tb.take(1 << 20)
    assert time.monotonic() - t0 < 0.05       # no rate -> no throttle


class _FakeSock:
    def __init__(self):
        self.sent = bytearray()
        self._recv_queue = [b"hello", b"world", b""]
    def sendall(self, data):
        self.sent += data
    def recv(self, n):
        return self._recv_queue.pop(0)
    def close(self):
        self.closed = True


def test_throttled_socket_meters_and_delegates():
    tb = TokenBucket(1e9)                      # huge rate -> no real delay
    sock = _FakeSock()
    ts = ThrottledSocket(sock, tb)
    ts.sendall(b"abcde")
    assert sock.sent == b"abcde" and tb.sent_bytes == 5
    assert ts.recv(8) == b"hello" and tb.recv_bytes == 5
    assert ts.recv(8) == b"world" and tb.recv_bytes == 10
    assert ts.recv(8) == b"" and tb.recv_bytes == 10   # EOF not counted
    ts.close()
    assert sock.closed is True


def test_throttled_passthrough_when_no_bucket():
    sock = _FakeSock()
    assert throttled(sock, None) is sock               # no cap -> byte-identical path
    assert isinstance(throttled(sock, TokenBucket(1e9)), ThrottledSocket)
