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
from libp2p.peer.peerinfo import PeerInfo, info_from_p2p_addr

from .identity import PeerIdentity
from .wire import DEFAULT_MAX_MSG_BYTES, _HEADER, decode, encode

RPC_PROTOCOL = "/opendipaco/rpc/1.0.0"


def dial_info(addr: str) -> PeerInfo:
    """A :class:`PeerInfo` from a full ``/…/p2p/<id>`` multiaddr (what
    :attr:`Libp2pTransport.addrs` advertises and a directory record carries)."""
    return info_from_p2p_addr(multiaddr.Multiaddr(addr))


def _derive_keypair(identity: PeerIdentity) -> KeyPair:
    """libp2p host keypair from our PeerIdentity's Ed25519 seed (D4)."""
    priv = Ed25519PrivateKey.from_bytes(identity.private_bytes_raw())
    return KeyPair(priv, priv.get_public_key())


class Libp2pTransport:
    """A synchronous RPC transport backed by a libp2p host (W1a).

    ``handler`` (optional) is a **synchronous** ``msg -> reply`` callable served
    for inbound streams; omit it for a dial-only (worker) peer.
    """

    def __init__(self, identity: PeerIdentity, *, handler=None,
                 listen_addrs=("/ip4/127.0.0.1/tcp/0",),
                 max_msg_bytes: int = DEFAULT_MAX_MSG_BYTES, start_timeout: float = 30.0):
        self.identity = identity
        self._handler = handler
        self._listen = [multiaddr.Multiaddr(a) for a in listen_addrs]
        self._kp = _derive_keypair(identity)
        self.max_msg_bytes = max_msg_bytes
        self._start_timeout = start_timeout
        self._host = None
        self._token = None            # trio token: lets foreign threads call in
        self._stop = None             # trio.Event set on close
        self._ready = threading.Event()
        self._thread = None
        self._err = None
        self._libp2p_id = None
        self._addrs: list[str] = []

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
        async with host.run(self._listen):
            if self._handler is not None:
                host.set_stream_handler(RPC_PROTOCOL, self._on_stream)
            self._token = trio.lowlevel.current_trio_token()
            self._libp2p_id = host.get_id()
            self._addrs = [str(a) for a in host.get_addrs()]
            self._stop = trio.Event()
            self._ready.set()
            await self._stop.wait()

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

    # -- RPC (sync facade over the trio loop) ----------------------------------

    def rpc(self, target: PeerInfo, msg, *, timeout: float | None = None):
        """Open a stream to ``target``, send ``msg``, return the reply. Blocks
        the calling (foreign) thread until the trio loop completes the round-trip."""
        if self._token is None:
            raise RuntimeError("transport not started")
        return trio.from_thread.run(self._rpc_async, target, msg, timeout,
                                    trio_token=self._token)

    async def _rpc_async(self, target: PeerInfo, msg, timeout):
        with trio.fail_after(timeout) if timeout else _nullcm():
            await self._host.connect(target)
            stream = await self._host.new_stream(target.peer_id, [RPC_PROTOCOL])
            try:
                await self._send(stream, msg)
                return await self._recv(stream)
            finally:
                await stream.close()

    async def _on_stream(self, stream) -> None:
        try:
            msg = await self._recv(stream)
            if msg is None:
                return
            reply = await trio.to_thread.run_sync(self._handler, msg)
            if reply is not None:
                await self._send(stream, reply)
        except (trio.BrokenResourceError, trio.ClosedResourceError, ValueError):
            pass  # peer vanished / oversized frame -> drop this stream, keep serving
        finally:
            await stream.close()

    # -- framing: our wire codec over a libp2p stream --------------------------

    async def _send(self, stream, obj) -> None:
        data = encode(obj)
        await stream.write(_HEADER.pack(len(data)) + data)

    async def _recv(self, stream):
        header = await self._read_exactly(stream, _HEADER.size)
        if header is None:
            return None
        (n,) = _HEADER.unpack(header)
        if n > self.max_msg_bytes:
            raise ValueError(f"incoming message of {n} bytes exceeds cap {self.max_msg_bytes}")
        body = await self._read_exactly(stream, n)
        return None if body is None else decode(body)

    @staticmethod
    async def _read_exactly(stream, n: int) -> bytes | None:
        chunks: list[bytes] = []
        got = 0
        while got < n:
            b = await stream.read(n - got)
            if not b:
                return None  # stream closed mid-frame
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)


class _nullcm:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
