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


# -- an owner (ParameterServer) served over libp2p -----------------------------


def _cfg():
    from opendipaco import BackboneConfig, DiPaCoConfig
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def test_parameter_server_fetch_and_push_over_libp2p():
    """The W1 payoff seam: an owner serves its real fetch/push RPC surface over a
    libp2p Noise stream, with the same effect as a direct call (TCP parity)."""
    from opendipaco import DiLoCoConfig
    from opendipaco.schedule import ParameterServer, make_grant
    from opendipaco.schedule.p2p import serve_over_libp2p

    cfg = _cfg()
    keys = sorted(cfg.build_topology().module_keys())
    ps = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=4), host="127.0.0.1", port=0,
                         identity=PeerIdentity.generate())
    owner = serve_over_libp2p(ps)
    client = Libp2pTransport(PeerIdentity.generate()).start()
    try:
        target = dial_info(owner.addrs[0])
        # fetch over libp2p == direct fetch (versions + weights match).
        shared = next(k for k in keys if not _is_private(k))
        reply = client.rpc(target, {"type": "fetch", "keys": [shared], "have": {}}, timeout=20)
        assert tuple(reply["versions"][shared]) == (0, 0)
        assert shared in reply["weights"]

        # push a grant over libp2p -> the owner applies it (version bumps).
        path = cfg.build_topology().path_from_index(0)
        grad = [torch.ones_like(p) for p in ps.bank[shared].parameters()]
        grant = make_grant(path, [shared], 1.0, "tok-libp2p")
        ack = client.rpc(target, {"grant": grant, "updates": {shared: {"grad": grad}},
                                  "type": "push"}, timeout=20)
        assert ack["applied"]
        assert ps._versions[shared] == (0, 1)          # the push landed, via libp2p
    finally:
        client.close()
        owner.close()
        ps.shutdown()


def _is_private(key):
    from opendipaco.topology import is_private_key
    return is_private_key(key)
