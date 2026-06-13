"""Tests for the libp2p transport seam + trio↔threads bridge (W1a).

These need the optional ``[nat]`` extra (libp2p); the baseline suite skips them.
They prove the integration boundary: our wire codec + a synchronous handler run
over a Noise-secured libp2p stream, driven from ordinary threads, and the libp2p
identity is derived from our PeerIdentity (D4).
"""

import pytest

pytest.importorskip("libp2p")

import torch  # noqa: E402

from opendipaco.schedule import PeerIdentity  # noqa: E402
from opendipaco.schedule.p2p import Libp2pTransport, _derive_keypair, dial_info  # noqa: E402


def test_libp2p_key_derives_from_peer_identity():
    """Same Ed25519 seed -> same libp2p peer id (deterministic, reconciled)."""
    ident = PeerIdentity.generate()
    a = _derive_keypair(ident)
    b = _derive_keypair(ident)
    from libp2p.peer.id import ID
    assert str(ID.from_pubkey(a.public_key)) == str(ID.from_pubkey(b.public_key))
    # A different identity yields a different libp2p id.
    other = _derive_keypair(PeerIdentity.generate())
    assert str(ID.from_pubkey(a.public_key)) != str(ID.from_pubkey(other.public_key))


def test_transport_round_trips_a_wire_frame():
    """A client transport RPCs a server transport; the sync handler runs off the
    trio loop and the reply comes back -- our frames over a libp2p Noise stream."""
    seen = {}

    def handler(msg):
        seen["msg"] = msg
        return {"type": "ack", "echo": msg.get("n")}

    server = Libp2pTransport(PeerIdentity.generate(), handler=handler).start()
    client = Libp2pTransport(PeerIdentity.generate()).start()  # dial-only
    try:
        reply = client.rpc(dial_info(server.addrs[0]), {"type": "ping", "n": 7}, timeout=20)
        assert reply == {"type": "ack", "echo": 7}
        assert seen["msg"]["type"] == "ping"
    finally:
        client.close()
        server.close()


def test_transport_carries_tensors():
    """The wire codec handles tensors over the stream (weights/grads ride this)."""
    def handler(msg):
        t = msg["w"]
        return {"type": "ack", "w": t * 2}

    server = Libp2pTransport(PeerIdentity.generate(), handler=handler).start()
    client = Libp2pTransport(PeerIdentity.generate()).start()
    try:
        w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        reply = client.rpc(dial_info(server.addrs[0]), {"type": "push", "w": w}, timeout=20)
        assert torch.equal(reply["w"], w * 2)
    finally:
        client.close()
        server.close()


def test_addrs_are_dialable_p2p_multiaddrs():
    server = Libp2pTransport(PeerIdentity.generate(), handler=lambda m: {"ok": True}).start()
    try:
        assert server.addrs and all("/p2p/" in a for a in server.addrs)
        info = dial_info(server.addrs[0])
        assert str(info.peer_id) == server.libp2p_id
    finally:
        server.close()
