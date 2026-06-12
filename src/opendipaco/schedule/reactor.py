"""A small selector-based TCP reactor + transport metrics.

Both the single-node ``CoordinatorServer`` (``distributed.py``) and the sharded
``Scheduler`` / ``ParameterServer`` (``sharded.py``) are reactor servers: a fixed
pool of ``io_threads`` selector loops serve many non-blocking connections (so the
thread footprint is bounded regardless of fleet size), with an accept thread that
round-robins connections to the I/O threads via a self-pipe wakeup, a
``max_connections`` cap, the pickle-free framing, and the HMAC auth handshake done
inline. Subclasses only implement :meth:`_ReactorServer._handle`.

NOTE: the wire format unpickles nothing (see ``wire.py``); auth proves key
possession but does not encrypt. For confidentiality pass a server ``ssl.SSLContext``
as ``tls=`` (see ``tls.py``): accepted sockets are TLS-wrapped and the handshake is
driven non-blocking in the I/O thread, before any framing/auth. Without it the
transport is plaintext (use only on a trusted network or an SSH tunnel).
"""

from __future__ import annotations

import collections
import hashlib
import hmac
import os
import selectors
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field

import torch

from .wire import DEFAULT_MAX_MSG_BYTES, _HEADER, acceptable_keys, decode, encode


def _verify_mac(mac: str, nonce: bytes, keys: list) -> bool:
    """Constant-time check that ``mac`` proves possession of one of ``keys`` (bytes)."""
    ok = False
    for k in keys:  # check all (no early-out) so timing doesn't reveal which matched
        ok |= hmac.compare_digest(mac, hmac.new(k, nonce, hashlib.sha256).hexdigest())
    return ok


def _nbytes(obj) -> int:
    """Total bytes of the tensors inside a message payload (for byte accounting)."""
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    return 0


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)}B" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


# -- metrics -----------------------------------------------------------------


@dataclass
class TransportMetrics:
    """Coordinator/scheduler-side counters for operating / measuring a run.

    ``bytes_opt`` should stay **0** (optimizer state is never shipped) and
    ``bytes_shard`` / ``tasks_with_shard`` show shards travel once per warm worker.
    (Some byte categories are single-coordinator-flavored; for the sharded servers
    the structural sharding -- not the byte breakdown -- is the point.)

    Beyond the end-of-run :meth:`report`, the live counters can be **streamed** while
    a run is in flight: :meth:`prometheus` renders them in Prometheus exposition
    format and ``schedule.observability`` serves/logs them (see
    :meth:`_ReactorServer.start_metrics_server`). Per-worker liveness is tracked via
    :meth:`record_worker` / :meth:`active_workers`.
    """

    # Summary keys that are point-in-time gauges; everything else is a cumulative
    # counter (these are class attrs, not dataclass fields -- no annotation).
    _GAUGES = frozenset({"max_staleness", "mean_staleness", "throughput_tasks_per_s",
                         "active_workers"})
    _LIVENESS_WINDOW = 60.0  # default "recently seen" horizon, seconds

    generations: int = 0
    tasks_sent: int = 0
    submits: int = 0
    heartbeats: int = 0
    nacks: int = 0
    idle: int = 0
    reclaims: int = 0
    dropped: int = 0
    tasks_with_shard: int = 0
    tasks_with_private_down: int = 0
    bytes_down: int = 0
    bytes_up: int = 0
    bytes_control: int = 0
    bytes_shared_weights: int = 0
    bytes_private_down: int = 0
    bytes_shard: int = 0
    bytes_shared_grad: int = 0
    bytes_private_up: int = 0
    bytes_opt: int = 0
    accepted_updates: int = 0
    stale_rejected: int = 0
    invalid_rejected: int = 0   # non-finite gradient/weights/loss -> contribution dropped
    norm_clipped: int = 0       # pseudo-gradients scaled down to max_update_norm
    staleness_sum: int = 0
    max_staleness: int = 0
    _wall: float = 0.0
    _worker_seen: dict = field(default_factory=dict)  # worker_id -> last-seen monotonic ts
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_worker(self, worker_id) -> None:
        """Mark a worker as alive now (any message carrying its id refreshes this)."""
        with self._lock:
            self._worker_seen[worker_id] = time.monotonic()

    def active_workers(self, window: float | None = None) -> int:
        """Count workers seen within ``window`` seconds (default ``_LIVENESS_WINDOW``)."""
        window = self._LIVENESS_WINDOW if window is None else window
        now = time.monotonic()
        with self._lock:
            return sum(1 for t in self._worker_seen.values() if now - t <= window)

    def record_update(self, staleness: int) -> None:
        with self._lock:
            self.accepted_updates += 1
            self.staleness_sum += staleness
            self.max_staleness = max(self.max_staleness, staleness)

    def record_stale_reject(self) -> None:
        with self._lock:
            self.stale_rejected += 1

    def record_invalid_reject(self) -> None:
        with self._lock:
            self.invalid_rejected += 1

    def record_norm_clip(self) -> None:
        with self._lock:
            self.norm_clipped += 1

    def record_out(self, msg: dict, nbytes: int) -> None:
        with self._lock:
            if msg.get("type") == "task":
                self.tasks_sent += 1
                self.bytes_down += nbytes
                self.bytes_shared_weights += _nbytes(msg.get("shared_weights"))
                self.bytes_opt += _nbytes(msg.get("opt_state"))
                if msg.get("private_weights"):
                    self.tasks_with_private_down += 1
                    self.bytes_private_down += _nbytes(msg["private_weights"])
                if msg.get("shard") is not None:
                    self.tasks_with_shard += 1
                    self.bytes_shard += _nbytes(msg["shard"])
            else:
                if msg.get("type") == "idle":
                    self.idle += 1
                self.bytes_control += nbytes

    def record_in(self, msg: dict, nbytes: int) -> None:
        with self._lock:
            kind = msg.get("type")
            if kind == "submit":
                self.submits += 1
                self.bytes_up += nbytes
                self.bytes_shared_grad += _nbytes(msg.get("shared_grad"))
                self.bytes_private_up += _nbytes(msg.get("private_weights"))
                self.bytes_opt += _nbytes(msg.get("opt_state"))
            elif kind == "heartbeat":
                self.heartbeats += 1
                self.bytes_control += nbytes
            else:
                if kind == "nack":
                    self.nacks += 1
                self.bytes_control += nbytes

    def summary(self) -> dict:
        now = time.monotonic()
        with self._lock:
            d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
            d["throughput_tasks_per_s"] = self.tasks_sent / self._wall if self._wall else 0.0
            d["mean_staleness"] = (
                self.staleness_sum / self.accepted_updates if self.accepted_updates else 0.0
            )
            d["active_workers"] = sum(
                1 for t in self._worker_seen.values() if now - t <= self._LIVENESS_WINDOW
            )
            return d

    def prometheus(self, namespace: str = "opendipaco_transport") -> str:
        """Render the live counters in Prometheus text exposition format.

        Counters get a ``_total`` suffix and ``# TYPE counter``; the derived gauges
        (staleness/throughput/active_workers) are emitted as gauges. Suitable for a
        scrape endpoint -- see ``schedule.observability.MetricsExporter``.
        """
        lines = []
        for k, v in self.summary().items():
            gauge = k in self._GAUGES
            name = f"{namespace}_{k}" if gauge else f"{namespace}_{k}_total"
            lines.append(f"# TYPE {name} {'gauge' if gauge else 'counter'}")
            lines.append(f"{name} {v}")
        return "\n".join(lines) + "\n"

    def report(self) -> str:
        s = self.summary()
        return "\n".join([
            f"accepted={s['accepted_updates']} stale_rejected={s['stale_rejected']} "
            f"mean_staleness={s['mean_staleness']:.1f} max_staleness={s['max_staleness']} "
            f"tasks={s['tasks_sent']} throughput={s['throughput_tasks_per_s']:.1f}/s",
            f"reclaims={s['reclaims']} nacks={s['nacks']} heartbeats={s['heartbeats']}",
            f"bytes  down={_fmt_bytes(s['bytes_down'])}  up={_fmt_bytes(s['bytes_up'])}  "
            f"control={_fmt_bytes(s['bytes_control'])}",
            f"  optimizer-on-wire={_fmt_bytes(s['bytes_opt'])}",
        ])


# -- reactor -----------------------------------------------------------------


@dataclass(eq=False)  # identity-hashable so a _Conn can live in sets/registries
class _Conn:
    sock: socket.socket
    inbuf: bytearray = field(default_factory=bytearray)
    outbuf: bytearray = field(default_factory=bytearray)
    authed: bool = False
    nonce: bytes | None = None
    want_write: bool = False
    closed: bool = False
    tls: bool = False          # socket is TLS-wrapped (ssl recv/send semantics)
    handshaking: bool = False  # TLS handshake not yet complete


class _IOThread:
    """One selector loop serving many non-blocking connections."""

    def __init__(self, server: "_ReactorServer"):
        self.server = server
        self.selector = selectors.DefaultSelector()
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self.selector.register(self._wake_r, selectors.EVENT_READ, data="wake")
        self._reg: collections.deque = collections.deque()
        self._reg_lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def add(self, conn: _Conn) -> None:
        with self._reg_lock:
            self._reg.append(conn)
        self.wake()

    def wake(self) -> None:
        try:
            self._wake_w.send(b"x")
        except OSError:
            pass

    def _loop(self) -> None:
        srv = self.server
        while not srv._io_stop and not srv._dead:
            try:
                events = self.selector.select(timeout=0.2)
            except OSError:
                break
            for key, mask in events:
                if key.data == "wake":
                    try:
                        self._wake_r.recv(1 << 16)
                    except OSError:
                        pass
                    self._drain_reg()
                    continue
                conn = key.data
                if conn.handshaking:
                    self._drive_handshake(conn)  # TLS handshake before any framing
                    continue
                if mask & selectors.EVENT_READ:
                    self._on_read(conn)
                if not conn.closed and (mask & selectors.EVENT_WRITE):
                    self._flush(conn)
        self._teardown()

    def _drain_reg(self) -> None:
        with self._reg_lock:
            news = list(self._reg)
            self._reg.clear()
        for conn in news:
            try:
                self.selector.register(conn.sock, selectors.EVENT_READ, data=conn)
            except (KeyError, ValueError, OSError):
                self._close(conn)
                continue
            if conn.handshaking:
                self._drive_handshake(conn)  # establish() runs once TLS completes
            else:
                self._establish(conn)

    def _establish(self, conn: _Conn) -> None:
        """Post-(handshake) setup: challenge the peer for auth, or admit it."""
        if self.server.auth_keys is not None:
            conn.nonce = os.urandom(32)
            self._queue(conn, {"type": "challenge", "nonce": conn.nonce.hex()})
        else:
            conn.authed = True
        if conn.outbuf:
            self._set_write(conn, True)

    def _drive_handshake(self, conn: _Conn) -> None:
        """Advance a non-blocking TLS handshake; toggle interest on WANT_READ/WRITE."""
        try:
            conn.sock.do_handshake()
        except ssl.SSLWantReadError:
            self._set_write(conn, False)
            return
        except ssl.SSLWantWriteError:
            self._set_write(conn, True)
            return
        except (ssl.SSLError, OSError):
            self._close(conn)
            return
        conn.handshaking = False
        self._set_write(conn, False)
        self._establish(conn)

    def _on_read(self, conn: _Conn) -> None:
        try:
            data = conn.sock.recv(1 << 20)
        except (BlockingIOError, ssl.SSLWantReadError):
            return
        except ssl.SSLWantWriteError:  # TLS wants to write before it can read
            self._set_write(conn, True)
            return
        except OSError:
            self._close(conn)
            return
        if not data:
            self._close(conn)
            return
        conn.inbuf += data
        self._consume(conn)
        # A single recv decrypts only what TLS has buffered so far, yet the
        # (level-triggered) kernel fd won't re-signal data already sitting in the
        # SSL object -- so drain whatever TLS has pending now.
        while not conn.closed and conn.tls and conn.sock.pending():
            try:
                data = conn.sock.recv(1 << 20)
            except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                break
            except OSError:
                self._close(conn)
                return
            if not data:
                self._close(conn)
                return
            conn.inbuf += data
            self._consume(conn)

    def _consume(self, conn: _Conn) -> None:
        """Frame + dispatch every complete message currently buffered in ``inbuf``."""
        while not conn.closed:
            if len(conn.inbuf) < _HEADER.size:
                break
            (n,) = _HEADER.unpack_from(conn.inbuf, 0)
            if n > self.server.max_msg_bytes:
                self._close(conn)
                return
            if len(conn.inbuf) < _HEADER.size + n:
                break
            body = bytes(conn.inbuf[_HEADER.size:_HEADER.size + n])
            del conn.inbuf[:_HEADER.size + n]
            try:
                msg = decode(body)
            except Exception:
                self._close(conn)
                return
            self._dispatch(conn, msg, n + _HEADER.size)

    def _dispatch(self, conn: _Conn, msg: dict, nin: int) -> None:
        srv = self.server
        if not conn.authed:
            ok = (
                srv.auth_keys is not None and conn.nonce is not None
                and msg.get("type") == "auth"
                and _verify_mac(msg.get("mac", ""), conn.nonce, srv.auth_keys)
            )
            if ok:
                conn.authed = True
                self._queue(conn, {"type": "welcome"})
            else:
                self._close(conn)
            return
        wid = msg.get("worker_id")
        if wid is not None:
            srv.metrics.record_worker(wid)  # any worker-tagged message = liveness
        srv.metrics.record_in(msg, nin)
        reply = srv._handle(msg, nin)  # subclass dispatch
        if reply is not None:
            nout = self._queue(conn, reply)
            srv.metrics.record_out(reply, nout)

    def _queue(self, conn: _Conn, msg: dict) -> int:
        data = encode(msg)
        framed = _HEADER.pack(len(data)) + data
        conn.outbuf += framed
        self._flush(conn)
        return len(framed)

    def _flush(self, conn: _Conn) -> None:
        if conn.closed:
            return
        if conn.outbuf:
            try:
                sent = conn.sock.send(conn.outbuf)
                del conn.outbuf[:sent]
            except (BlockingIOError, ssl.SSLWantWriteError):
                pass
            except ssl.SSLWantReadError:
                pass  # TLS wants to read first; read interest is always registered
            except OSError:
                self._close(conn)
                return
        self._set_write(conn, bool(conn.outbuf))

    def _set_write(self, conn: _Conn, on: bool) -> None:
        if conn.closed or conn.want_write == on:
            return
        events = selectors.EVENT_READ | (selectors.EVENT_WRITE if on else 0)
        try:
            self.selector.modify(conn.sock, events, data=conn)
            conn.want_write = on
        except (KeyError, OSError):
            pass

    def _close(self, conn: _Conn) -> None:
        if conn.closed:
            return
        conn.closed = True
        try:
            self.selector.unregister(conn.sock)
        except (KeyError, OSError):
            pass
        try:
            conn.sock.close()
        except OSError:
            pass
        self.server._conn_closed(conn)

    def _teardown(self) -> None:
        for key in list(self.selector.get_map().values()):
            if key.data != "wake" and not key.data.closed:
                self._close(key.data)
        try:
            self.selector.close()
        except OSError:
            pass


class _ReactorServer:
    """Base TCP server: bounded selector I/O threads + accept thread + auth.

    Subclasses implement :meth:`_handle` to turn a decoded message into an
    optional reply dict. Everything else -- bind, accept, framing, auth, metrics,
    shutdown -- is generic.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 0,
        auth_key=None,
        accept_keys=None,
        max_msg_bytes: int = DEFAULT_MAX_MSG_BYTES,
        io_threads: int = 4,
        max_connections: int = 1024,
        tls: "ssl.SSLContext | None" = None,
    ):
        # ``auth_key`` is this node's single secret (also its client identity);
        # ``accept_keys`` are extra secrets the server also accepts (rotation /
        # per-worker identity). ``auth_keys`` is the de-duplicated accept-list.
        self.auth_key = auth_key
        self.auth_keys = acceptable_keys(auth_key, accept_keys)
        self.max_msg_bytes = max_msg_bytes
        self.tls = tls  # server-side SSLContext; None -> plaintext (auth still applies)
        self.io_threads = max(1, io_threads)
        self.max_connections = max_connections
        self.metrics = TransportMetrics()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        deadline = time.monotonic() + 3.0  # tolerate a transient EADDRINUSE on restart
        while True:
            try:
                self._sock.bind((host, port))
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        self._sock.listen(128)
        self.port = self._sock.getsockname()[1]

        self._stop = False
        self._io_stop = False
        self._dead = False
        self._io: list[_IOThread] = []
        self._accept_thread: threading.Thread | None = None
        self._conn_lock = threading.Lock()
        self._registry: set = set()
        self._open = 0
        self._rr = 0
        self._metrics_exporter = None
        self._metrics_logger = None

    # -- subclass hook -------------------------------------------------------
    def _handle(self, msg: dict, nbytes: int):
        raise NotImplementedError

    # -- observability streaming --------------------------------------------
    def start_metrics_server(self, *, host: str = "0.0.0.0", port: int = 0,
                             namespace: str = "opendipaco_transport"):
        """Expose this server's live metrics over HTTP (``/metrics`` Prometheus,
        ``/`` report, ``/healthz``). Stopped automatically on :meth:`shutdown`."""
        from .observability import MetricsExporter
        self._metrics_exporter = MetricsExporter(
            self.metrics, host=host, port=port, namespace=namespace).start()
        return self._metrics_exporter

    def start_metrics_logging(self, *, interval: float = 10.0, sink=None):
        """Periodically emit a structured (JSON) snapshot of the live metrics while
        the run is in flight. Stopped automatically on :meth:`shutdown`."""
        from .observability import MetricsLogger
        self._metrics_logger = MetricsLogger(self.metrics, interval=interval, sink=sink).start()
        return self._metrics_logger

    # -- plumbing ------------------------------------------------------------
    def start(self) -> None:
        self._io = [_IOThread(self) for _ in range(self.io_threads)]
        for io in self._io:
            io.start()
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._stop and not self._dead:
            try:
                sock, _ = self._sock.accept()
            except OSError:
                return
            with self._conn_lock:
                if self._open >= self.max_connections:
                    sock.close()  # backpressure: refuse beyond the connection cap
                    continue
                self._open += 1
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
            if self.tls is not None:
                try:
                    # do_handshake_on_connect=False: drive the handshake non-blocking
                    # in the I/O thread so a slow peer can't stall the accept loop.
                    sock = self.tls.wrap_socket(sock, server_side=True,
                                                do_handshake_on_connect=False)
                except (ssl.SSLError, OSError):
                    sock.close()
                    with self._conn_lock:
                        self._open -= 1
                    continue
                conn = _Conn(sock=sock, tls=True, handshaking=True)
            else:
                conn = _Conn(sock=sock)
            with self._conn_lock:
                self._registry.add(conn)
            self._io[self._rr % len(self._io)].add(conn)
            self._rr += 1

    def _conn_closed(self, conn: _Conn) -> None:
        with self._conn_lock:
            if conn in self._registry:
                self._registry.discard(conn)
                self._open -= 1

    def shutdown(self) -> None:
        self._stop = True
        for obs in (self._metrics_exporter, self._metrics_logger):
            if obs is not None:
                obs.stop()
        try:
            self._sock.close()
        except OSError:
            pass
        time.sleep(0.2)  # grace: let in-flight workers receive a final reply
        self._io_stop = True
        for io in self._io:
            io.wake()
        time.sleep(0.1)

    def simulate_crash(self) -> None:
        """Drop all connections abruptly (testing coordinator restart/reconnect)."""
        self._dead = True
        self._io_stop = True
        with self._conn_lock:
            conns = list(self._registry)
        for conn in conns:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.sock.close()
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass
        for io in self._io:
            io.wake()
