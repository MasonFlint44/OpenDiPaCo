"""libp2p transport seam + trio↔threads bridge (W1a; docs/w1-nat-design.md).

The rest of the stack is synchronous (threads + the custom reactor); py-libp2p is
trio-async. This module is the **only** place those two worlds meet: a libp2p
host runs in a dedicated trio thread, and :class:`Libp2pTransport` exposes a
*synchronous* facade —

    t = Libp2pTransport(identity, handler=server._handle_bytes)
    t.start()
    reply = t.rpc(dial_info(peer_addr), {"type": "fetch", ...})   # blocks, returns the reply

— so the owner / scheduler / worker code calls it exactly like the raw-TCP path.
Our existing wire codec (:mod:`.wire`) frames every message over a Noise-secured
libp2p **stream** (so a relay, later, only ever sees ciphertext — design D7); the
inbound handler dispatches our synchronous ``_handle`` off the trio loop via
``trio.to_thread`` so a lock or a tensor copy never stalls libp2p.

Identity reconciles per D4: the libp2p host key is derived from the *same*
:class:`~opendipaco.schedule.identity.PeerIdentity` Ed25519 seed, so a peer's
app-layer id (``sha256(pubkey)``) and its libp2p id (``12D3KooW…``) descend from
one keypair. ``importorskip``-friendly: importing this module needs the optional
``[nat]`` extra (``libp2p``); nothing in the default install imports it.
"""

from __future__ import annotations

import threading

import multiaddr
import trio
from libp2p import new_host
from libp2p.crypto.ed25519 import Ed25519PrivateKey
from libp2p.crypto.keys import KeyPair
from libp2p.network.exceptions import SwarmException
from libp2p.network.stream.exceptions import StreamEOF, StreamError
from libp2p.peer.id import ID
from libp2p.peer.peerinfo import PeerInfo, info_from_p2p_addr
from libp2p.relay.circuit_v2.config import RelayConfig
from libp2p.relay.circuit_v2.protocol import PROTOCOL_ID as CIRCUIT_HOP_PROTOCOL
from libp2p.relay.circuit_v2.protocol import CircuitV2Protocol
from libp2p.relay.circuit_v2.transport import CircuitV2Transport
from libp2p.tools.async_service import background_trio_service

from .identity import PeerIdentity, peer_id_of
from .wire import DEFAULT_MAX_MSG_BYTES, _HEADER, decode, encode

RPC_PROTOCOL = "/opendipaco/rpc/1.0.0"
_CIRCUIT_SEP = "/p2p-circuit"


def dial_info(addr: str) -> PeerInfo:
    """A :class:`PeerInfo` from a full ``/…/p2p/<id>`` multiaddr (what
    :attr:`Libp2pTransport.addrs` advertises and a directory record carries)."""
    return info_from_p2p_addr(multiaddr.Multiaddr(addr))


def dial_circuit(circuit_addr: str) -> tuple[PeerInfo, PeerInfo]:
    """Split a ``<relay-ma>/p2p-circuit/p2p/<dest>`` addr into ``(dest_info,
    relay_info)`` — what a NAT'd peer advertises (W1b/D5) and a dialer reaches it
    through. The dest carries no transport addr; it's reached via the relay."""
    idx = circuit_addr.find(_CIRCUIT_SEP)
    relay_info = info_from_p2p_addr(multiaddr.Multiaddr(circuit_addr[:idx]))
    dest_b58 = circuit_addr.rsplit("/p2p/", 1)[1]
    return PeerInfo(ID.from_base58(dest_b58), []), relay_info


def _derive_keypair(identity: PeerIdentity) -> KeyPair:
    """libp2p host keypair from our PeerIdentity's Ed25519 seed (D4)."""
    priv = Ed25519PrivateKey.from_bytes(identity.private_bytes_raw())
    return KeyPair(priv, priv.get_public_key())


def serve_over_libp2p(server, *, identity: PeerIdentity | None = None,
                      listen_addrs=("/ip4/127.0.0.1/tcp/0",)) -> "Libp2pTransport":
    """Serve a server's ``_handle`` over libp2p (W1a owner-serving bridge).

    Additive and non-invasive: the existing TCP reactor is untouched (the byte-
    for-byte anchor): this starts a *parallel* libp2p host that bridges inbound
    streams to the same ``_handle(msg, nbytes, peer_id)``. libp2p's Noise
    handshake authenticates the peer cryptographically, so this path doesn't need
    the reactor's HMAC challenge. Returns the started transport; ``.addrs`` are
    the dialable ``/…/p2p/<id>`` multiaddrs to advertise (a public peer's direct
    addrs today; a NAT'd peer's circuit-relay addrs in W1b).
    """
    from .reactor import _nbytes

    ident = identity or getattr(server, "identity", None)
    if ident is None:
        raise ValueError("serve_over_libp2p needs an identity (server.identity or identity=)")

    def handler(msg, peer_id):
        # peer_id is the Noise-authenticated remote mapped to our app id (W1c), so
        # reputation / rate-limit / audit / owner-eligibility + enrollment gates
        # apply on the libp2p path exactly as on TCP.
        return server._handle(msg, _nbytes(msg), peer_id=peer_id)

    t = Libp2pTransport(ident, handler=handler, listen_addrs=listen_addrs).start()
    # Let the owner reuse this transport for outbound owner↔owner RPCs (_peer_rpc
    # dials co-owners over libp2p / through relays when their addr is a multiaddr).
    if hasattr(server, "libp2p"):
        server.libp2p = t
    return t


class Libp2pTransport:
    """A synchronous RPC transport backed by a libp2p host (W1a).

    ``handler`` (optional) is a **synchronous** ``msg -> reply`` callable served
    for inbound streams; omit it for a dial-only (worker) peer.
    """

    def __init__(self, identity: PeerIdentity, *, handler=None, relay: bool = False,
                 listen_addrs=("/ip4/127.0.0.1/tcp/0",),
                 max_msg_bytes: int = DEFAULT_MAX_MSG_BYTES, start_timeout: float = 30.0):
        self.identity = identity
        self._handler = handler
        self._relay = relay           # run Circuit Relay v2 in HOP mode (a relay peer, D6)
        self._listen = [multiaddr.Multiaddr(a) for a in listen_addrs]
        self._kp = _derive_keypair(identity)
        self.max_msg_bytes = max_msg_bytes
        self._start_timeout = start_timeout
        self._host = None
        self._token = None            # trio token: lets foreign threads call in
        self._stop = None             # trio.Event set on close
        self._nursery = None          # for reservation refreshers (D6)
        self._ctransport = None       # CircuitV2Transport: reserve / dial-through-relay
        self._ready = threading.Event()
        self._rpc_lock = threading.Lock()  # serialize outbound rpcs: concurrent
        self._thread = None                # same-peer dials race py-libp2p's swarm
        self._err = None
        self._libp2p_id = None
        self._addrs: list[str] = []
        self._circuit_addrs: list[str] = []  # /…/p2p-circuit/p2p/<self> per reserved relay

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> "Libp2pTransport":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(self._start_timeout):
            raise TimeoutError("libp2p host did not start in time")
        if self._err is not None:
            raise self._err
        return self

    def _run(self) -> None:
        try:
            trio.run(self._main)
        except Exception as e:  # noqa: BLE001 -- surface to start()/the caller
            self._err = e
            self._ready.set()

    async def _main(self) -> None:
        host = new_host(key_pair=self._kp)   # default security is Noise (D1/D7)
        self._host = host
        # Circuit Relay v2 runs on every peer: HOP (allow_hop) on a relay so it
        # forwards; STOP/CLIENT elsewhere so a peer can be reached through, or
        # dial through, a relay. The relayed stream is Noise-secured end-to-end,
        # so a relay only ever sees ciphertext (D7).
        cproto = CircuitV2Protocol(host, allow_hop=self._relay)
        async with host.run(self._listen), background_trio_service(cproto), \
                trio.open_nursery() as nursery:
            self._nursery = nursery
            self._ctransport = CircuitV2Transport(host, cproto, RelayConfig())
            if self._handler is not None:
                host.set_stream_handler(RPC_PROTOCOL, self._on_stream)
            self._token = trio.lowlevel.current_trio_token()
            self._libp2p_id = host.get_id()
            self._addrs = [str(a) for a in host.get_addrs()]
            self._stop = trio.Event()
            self._ready.set()
            await self._stop.wait()
            nursery.cancel_scope.cancel()   # stop reservation refreshers, unwind cleanly

    def close(self) -> None:
        if self._token is not None and self._stop is not None:
            try:
                trio.from_thread.run_sync(self._stop.set, trio_token=self._token)
            except (RuntimeError, trio.RunFinishedError):
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- identity / addressing -------------------------------------------------

    @property
    def libp2p_id(self) -> str:
        return str(self._libp2p_id)

    @property
    def addrs(self) -> list[str]:
        """Full dialable ``/…/p2p/<id>`` multiaddrs (for the directory record)."""
        return list(self._addrs)

    @property
    def circuit_addrs(self) -> list[str]:
        """``/…/p2p-circuit/p2p/<self>`` addrs for each reserved relay — what a
        NAT'd peer advertises so others can dial it through a relay (D5/D6)."""
        return list(self._circuit_addrs)

    # -- relay reservation (D6) ------------------------------------------------

    def reserve_on(self, relay_addr: str) -> str | None:
        """Reserve a forwarding slot on a relay; returns this peer's circuit addr
        through that relay (to advertise), or None if the reservation failed. A
        NAT'd owner reserves on k>=2 relays (D6) for failover + eclipse resistance."""
        if self._token is None:
            raise RuntimeError("transport not started")
        return trio.from_thread.run(self._reserve_async, relay_addr, trio_token=self._token)

    async def _reserve_async(self, relay_addr: str):
        relay_info = dial_info(relay_addr)
        await self._host.connect(relay_info)
        hop = await self._host.new_stream(relay_info.peer_id, [CIRCUIT_HOP_PROTOCOL])
        if not await self._ctransport.reserve(hop, relay_info.peer_id, self._nursery):
            return None
        circuit = f"{relay_addr}{_CIRCUIT_SEP}/p2p/{self.libp2p_id}"
        if circuit not in self._circuit_addrs:
            self._circuit_addrs.append(circuit)
        return circuit

    # -- RPC (sync facade over the trio loop) ----------------------------------

    def rpc(self, target, msg, *, timeout: float | None = None):
        """Open a stream to ``target``, send ``msg``, return the reply. ``target``
        is a :class:`PeerInfo`, a direct ``/…/p2p/<id>`` addr, or a
        ``/…/p2p-circuit/p2p/<id>`` circuit addr (dialed through the relay).
        Blocks the calling thread until the trio round-trip completes."""
        if self._token is None:
            raise RuntimeError("transport not started")
        with self._rpc_lock:
            try:
                return trio.from_thread.run(self._rpc_async, target, msg, timeout,
                                            trio_token=self._token)
            except (SwarmException, StreamError, StreamEOF, trio.TooSlowError,
                    trio.BrokenResourceError, trio.ClosedResourceError) as e:
                # Surface libp2p/trio transport faults as ConnectionError so callers
                # (the worker loop) handle them with their existing OSError paths.
                raise ConnectionError(f"libp2p rpc to {target} failed: {e}") from e

    async def _rpc_async(self, target, msg, timeout):
        with trio.fail_after(timeout) if timeout else _nullcm():
            stream = await self._open_stream(target)
            try:
                await self._send(stream, msg)
                return await self._recv(stream)
            finally:
                await stream.close()

    async def _open_stream(self, target):
        """Connect + open an RPC stream, retrying transient failures — concurrent
        dials to the same peer (across workers, and a worker's heartbeat racing
        its fetch/push) can momentarily fail stream setup in py-libp2p's swarm;
        a short backoff lets a half-open connection settle or re-dial."""
        last = None
        for attempt in range(6):
            try:
                peer_id = await self._connect(target)
                return await self._host.new_stream(peer_id, [RPC_PROTOCOL])
            except (SwarmException, StreamError, trio.BrokenResourceError) as e:
                last = e
                await trio.sleep(0.05 * (attempt + 1))
        raise ConnectionError(f"could not open stream to {target}: {last}")

    async def _connect(self, target):
        """Resolve+connect to ``target`` (direct or relayed); return its peer id."""
        if isinstance(target, str):
            if _CIRCUIT_SEP in target:                      # reach a NAT'd peer via its relay
                dest, relay = dial_circuit(target)
                await self._ctransport.dial_peer_info(dest, relay_info=relay)
                return dest.peer_id
            target = dial_info(target)
        await self._host.connect(target)
        return target.peer_id

    @staticmethod
    def _remote_peer_id(stream) -> str | None:
        """Our app-layer (sha256) peer id for the stream's remote, derived from
        the identity libp2p **already authenticated** via Noise (W1c). For
        Ed25519 the libp2p id embeds the pubkey, and our id is sha256(pubkey), so
        no directory lookup is needed -- the binding is the key itself. None if
        the remote used a non-extractable key (e.g. RSA)."""
        try:
            pub = stream.muxed_conn.peer_id.extract_public_key()
            return peer_id_of(pub.to_bytes().hex()) if pub is not None else None
        except Exception:  # noqa: BLE001 -- never let id extraction break serving
            return None

    async def _on_stream(self, stream) -> None:
        try:
            msg = await self._recv(stream)
            if msg is None:
                return
            peer_id = self._remote_peer_id(stream)
            reply = await trio.to_thread.run_sync(self._handler, msg, peer_id)
            if reply is not None:
                await self._send(stream, reply)
        except Exception:  # noqa: BLE001 -- serving untrusted peers: one bad request
            pass            # (vanished peer, malformed frame, handler error) must
            #                  never escape and kill the host; trio.Cancelled (a
            #                  BaseException) still propagates so shutdown works
        finally:
            await stream.close()

    # -- framing: our wire codec over a libp2p stream --------------------------

    # libp2p's Noise channel rejects any single write over 65535 bytes, so frames
    # (a task's shard, a path's weights, pseudo-gradients) are written in chunks;
    # the length-prefixed reader reassembles them transparently.
    _WRITE_CHUNK = 32768

    async def _send(self, stream, obj) -> None:
        payload = encode(obj)
        framed = _HEADER.pack(len(payload)) + payload
        for i in range(0, len(framed), self._WRITE_CHUNK):
            await stream.write(framed[i:i + self._WRITE_CHUNK])

    async def _recv(self, stream):
        header = await self._read_exactly(stream, _HEADER.size)
        if header is None:
            return None
        (n,) = _HEADER.unpack(header)
        if n > self.max_msg_bytes:
            # A Byzantine/buggy peer must not crash us: a bad frame is a transport
            # fault (ConnectionError) the worker's retry/next-replica paths handle.
            raise ConnectionError(f"incoming frame of {n} bytes exceeds cap {self.max_msg_bytes}")
        body = await self._read_exactly(stream, n)
        if body is None:
            return None
        try:
            return decode(body)
        except Exception as e:  # noqa: BLE001 -- a malformed frame is a transport fault
            raise ConnectionError(f"malformed frame: {e}") from e

    @staticmethod
    async def _read_exactly(stream, n: int) -> bytes | None:
        chunks: list[bytes] = []
        got = 0
        while got < n:
            try:
                b = await stream.read(n - got)
            except StreamEOF:
                return None  # libp2p signals close by raising, not by returning b""
            if not b:
                return None  # stream closed mid-frame
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)


class _Libp2pLink:
    """A ``_WorkerLink``-compatible seam backed by a libp2p transport (W1b step 3).

    Drop-in for the worker's TCP ``_WorkerLink``: the worker loop speaks only to
    this interface, so it runs unchanged over libp2p. ``sched_addr`` and the PS
    addrs in routing are multiaddrs (direct or ``/p2p-circuit`` for a relayed
    owner). libp2p multiplexes streams over reused connections, so there is no
    socket cache to manage and no "prefer connected" preference to express."""

    def __init__(self, transport: "Libp2pTransport", sched_addr: str, *,
                 rpc_timeout: float = 120.0, heartbeat_timeout: float = 15.0):
        self._t = transport
        self._sched = sched_addr
        # Bounded waits: a peer that is alive but unresponsive (half-open NAT
        # connection, network stall) must NOT hang the worker forever holding the
        # rpc lock. A timeout surfaces as ConnectionError -> the worker's
        # retry / next-replica paths absorb it. rpc covers large weight transfers,
        # so it is generous; the heartbeat is short.
        self._rpc_timeout = rpc_timeout
        self._hb_timeout = heartbeat_timeout

    def sch_rpc(self, msg):
        return self._t.rpc(self._sched, msg, timeout=self._rpc_timeout)

    def sch_send(self, msg) -> None:
        try:
            self._t.rpc(self._sched, msg, timeout=self._hb_timeout)  # reply is None
        except Exception:  # noqa: BLE001 -- a heartbeat is best-effort
            pass

    def connected(self, addr) -> bool:
        return False  # libp2p reuses connections itself; no preference to express

    def ps_rpc(self, addr, msg):
        return self._t.rpc(addr, msg, timeout=self._rpc_timeout)

    def close(self) -> None:
        self._t.close()


class _nullcm:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
