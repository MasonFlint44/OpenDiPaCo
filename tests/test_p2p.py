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

    def handler(msg, peer_id):
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
    def handler(msg, peer_id):
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


def test_authenticated_peer_id_is_threaded_to_the_handler():
    """W1c: the Noise-authenticated remote is mapped to OUR app peer id and passed
    to the handler, so reputation / rate-limit / enrollment gates apply over
    libp2p exactly as on TCP. The id matches the dialer's PeerIdentity."""
    seen = {}

    def handler(msg, peer_id):
        seen["pid"] = peer_id
        return {"ok": True}

    server = Libp2pTransport(PeerIdentity.generate(), handler=handler).start()
    client_id = PeerIdentity.generate()
    client = Libp2pTransport(client_id).start()
    try:
        client.rpc(dial_info(server.addrs[0]), {"x": 1}, timeout=20)
        assert seen["pid"] == client_id.peer_id   # authenticated, not None
    finally:
        client.close()
        server.close()


def test_unauthenticatable_peer_is_refused_over_libp2p():
    """A non-Ed25519 peer (whose libp2p id can't yield our app peer id) is denied
    service, so it can't slip past the reputation/rate-limit/Sybil gates as
    'trusted anonymous' (W1c). An Ed25519 peer is served."""
    from libp2p.crypto.rsa import create_new_key_pair as rsa_kp

    from opendipaco import DiLoCoConfig
    from opendipaco.schedule import ParameterServer
    from opendipaco.schedule.p2p import serve_over_libp2p

    cfg = _cfg()
    keys = sorted(cfg.build_topology().module_keys())
    ps = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=4), host="127.0.0.1", port=0,
                         identity=PeerIdentity.generate())
    owner = serve_over_libp2p(ps)                  # require_identity=True (default)
    rsa_client = Libp2pTransport(PeerIdentity.generate())
    rsa_client._kp = rsa_kp(2048)                  # non-Ed25519 host key
    rsa_client.start()
    ed_client = Libp2pTransport(PeerIdentity.generate()).start()
    try:
        shared = next(k for k in keys if not _is_private(k))
        req = {"type": "fetch", "keys": [shared], "have": {}}
        assert rsa_client.rpc(dial_info(owner.addrs[0]), req, timeout=20) is None   # refused
        assert ed_client.rpc(dial_info(owner.addrs[0]), req, timeout=20)["versions"]  # served
    finally:
        rsa_client.close()
        ed_client.close()
        owner.close()
        ps.shutdown()


def test_addrs_are_dialable_p2p_multiaddrs():
    server = Libp2pTransport(PeerIdentity.generate(), handler=lambda m, pid: {"ok": True}).start()
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


# -- W1b: Circuit Relay v2 (reach a NAT'd peer through a relay) -----------------


def test_relayed_rpc_round_trip():
    """A listener reachable ONLY through a relay: it reserves on the relay,
    advertises a circuit addr, and a dialer reaches it through the relay. The
    relay runs no RPC handler -- it purely forwards (Noise e2e: it sees only
    ciphertext, D7)."""
    relay = Libp2pTransport(PeerIdentity.generate(), relay=True).start()
    seen = {}

    def handler(msg, peer_id):
        seen["n"] = msg.get("n")
        return {"type": "ack", "echo": msg.get("n")}

    listener = Libp2pTransport(PeerIdentity.generate(), handler=handler).start()
    dialer = Libp2pTransport(PeerIdentity.generate()).start()
    try:
        circuit = listener.reserve_on(relay.addrs[0])
        assert circuit and "/p2p-circuit/" in circuit
        assert circuit in listener.circuit_addrs
        # Reach the listener through the relay -- using the circuit addr, never a
        # direct addr of the listener.
        reply = dialer.rpc(circuit, {"type": "ping", "n": 11}, timeout=30)
        assert reply == {"type": "ack", "echo": 11}
        assert seen["n"] == 11
    finally:
        dialer.close()
        listener.close()
        relay.close()


def test_server_survives_a_raising_handler():
    """A handler error on one request (a malformed/hostile call) must not kill
    the host -- it drops that stream and keeps serving (Byzantine hardening)."""
    calls = {"n": 0}

    def handler(msg, peer_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")     # first request blows up the handler
        return {"ok": True}

    server = Libp2pTransport(PeerIdentity.generate(), handler=handler).start()
    client = Libp2pTransport(PeerIdentity.generate()).start()
    try:
        info = dial_info(server.addrs[0])
        assert client.rpc(info, {"x": 1}, timeout=20) is None   # no reply, stream dropped
        assert client.rpc(info, {"x": 2}, timeout=20) == {"ok": True}  # host survived
    finally:
        client.close()
        server.close()


def test_server_times_out_a_stuck_request():
    """A request that ties up the handler past serve_timeout (a stall / handler
    that blocks; same path a slow-loris read hits) is dropped, and the server
    keeps serving — bounded inbound work, no unbounded resource hold."""
    import time as _time

    def handler(msg, peer_id):
        if msg.get("slow"):
            _time.sleep(5)          # blocks longer than the 1s serve_timeout
        return {"ok": True}

    server = Libp2pTransport(PeerIdentity.generate(), handler=handler,
                             serve_timeout=1.0).start()
    client = Libp2pTransport(PeerIdentity.generate()).start()
    try:
        info = dial_info(server.addrs[0])
        assert client.rpc(info, {"slow": True}, timeout=10) is None     # timed out, dropped
        assert client.rpc(info, {"slow": False}, timeout=10) == {"ok": True}  # survived
    finally:
        client.close()
        server.close()


def test_oversized_reply_is_a_connection_error_not_a_crash():
    """A frame over the receiver's cap (a buggy/Byzantine peer) surfaces as
    ConnectionError -- which the worker's retry/next-replica paths handle -- not
    an uncaught crash."""
    def handler(msg, peer_id):
        return {"big": torch.zeros(5000)}              # ~20 KB, over the 1 KB cap

    server = Libp2pTransport(PeerIdentity.generate(), handler=handler).start()
    client = Libp2pTransport(PeerIdentity.generate(), max_msg_bytes=1024).start()
    try:
        with pytest.raises(ConnectionError):
            client.rpc(dial_info(server.addrs[0]), {"x": 1}, timeout=20)
    finally:
        client.close()
        server.close()


def test_sharded_cluster_trains_over_libp2p():
    """The W1b orchestration payoff: a full sharded cluster -- scheduler + 2
    parameter servers + 2 workers -- runs end-to-end over libp2p (addresses are
    multiaddrs, RPCs are Noise streams), training to its generation budget."""
    import threading

    from opendipaco import DiLoCoConfig
    from opendipaco.data import ShardedCorpus
    from opendipaco.schedule import (
        ParameterServer, Scheduler, assign_shards, run_sharded_worker,
    )
    from opendipaco.schedule.p2p import serve_over_libp2p

    cfg = _cfg()
    diloco = DiLoCoConfig(inner_steps=4, inner_lr=1e-3)
    g = torch.Generator().manual_seed(0)
    span = 48 // 4
    docs = [torch.randint(t * span, (t + 1) * span, (32,), generator=g)
            for t in range(4) for _ in range(8)]
    assign = torch.tensor([i % cfg.num_paths for i in range(len(docs))])
    corpus = ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, 16)

    keys = cfg.build_topology().module_keys()
    shards = [[k for k, s in assign_shards(keys, 2).items() if s == i] for i in range(2)]
    pss = [ParameterServer(cfg, sk, diloco, host="127.0.0.1", port=0,
                           identity=PeerIdentity.generate()) for sk in shards]
    ps_t = [serve_over_libp2p(ps) for ps in pss]          # owners over libp2p
    ps_addrs = [t.addrs[0] for t in ps_t]                 # multiaddr per shard

    sched = Scheduler(cfg, corpus, ps_addrs, diloco, batch_size=8, host="127.0.0.1",
                      port=0, identity=PeerIdentity.generate())
    sched_t = serve_over_libp2p(sched)
    workers = [threading.Thread(
        target=run_sharded_worker, args=(cfg, diloco, sched_t.addrs[0]),
        kwargs=dict(transport="libp2p", identity=PeerIdentity.generate(),
                    heartbeat_interval=2.0), daemon=True) for _ in range(2)]
    for w in workers:
        w.start()
    try:
        completed = sched.fit(num_generations=2, total_generations=2)
        assert sum(completed.values()) >= 2 * cfg.num_paths   # the budget was met
        assert sched.metrics.accepted_updates > 0             # updates landed over libp2p
    finally:
        sched_t.close()
        for t in ps_t:
            t.close()
        sched.shutdown()
        for ps in pss:
            ps.shutdown()
        for w in workers:
            w.join(timeout=10)


def test_rpc_fails_over_across_candidate_addrs():
    """W1c multi-relay failover at the transport: rpc tried over a list of
    candidate addrs uses the first that works (a dead relay -> the next)."""
    server = Libp2pTransport(PeerIdentity.generate(),
                             handler=lambda m, pid: {"ok": True}).start()
    dead_t = Libp2pTransport(PeerIdentity.generate()).start()
    dead = dead_t.addrs[0]
    dead_t.close()                       # well-formed addr, now unreachable (a dead relay)
    client = Libp2pTransport(PeerIdentity.generate()).start()
    try:
        # All-dead -> ConnectionError; [dead, live] -> fails over to live.
        with pytest.raises(ConnectionError):
            client.rpc([dead], {"x": 1}, timeout=8)
        assert client.rpc([dead, server.addrs[0]], {"x": 1}, timeout=20) == {"ok": True}
    finally:
        client.close()
        server.close()


def test_owner_to_owner_rpc_over_libp2p():
    """Owners dial each other over libp2p via _peer_rpc (W1c) — replication,
    gossip, and digest-audit ride this. Here owner A fetches owner B's digest
    over a libp2p stream (a multiaddr addr routes to the libp2p path)."""
    from opendipaco import DiLoCoConfig
    from opendipaco.schedule import ParameterServer
    from opendipaco.schedule.p2p import serve_over_libp2p

    cfg = _cfg()
    keys = sorted(cfg.build_topology().module_keys())
    a = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=4), host="127.0.0.1", port=0,
                        identity=PeerIdentity.generate())
    b = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=4), host="127.0.0.1", port=0,
                        identity=PeerIdentity.generate())
    ta, tb = serve_over_libp2p(a), serve_over_libp2p(b)
    try:
        assert a.libp2p is ta                       # serve_over_libp2p wired the transport
        reply = a._peer_rpc(tb.addrs[0], {"type": "digest", "keys": None})
        assert reply["type"] == "digest" and reply["digests"]
    finally:
        ta.close()
        tb.close()
        a.shutdown()
        b.shutdown()


def test_nat_owner_served_through_a_relay():
    """The W1 payoff: a ParameterServer (owner) with no usable direct route is
    reached through a relay -- fetch over the circuit returns its versioned
    weights, exactly like the direct libp2p path."""
    from opendipaco import DiLoCoConfig
    from opendipaco.schedule import ParameterServer
    from opendipaco.schedule.p2p import serve_over_libp2p

    cfg = _cfg()
    keys = sorted(cfg.build_topology().module_keys())
    ps = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=4), host="127.0.0.1", port=0,
                         identity=PeerIdentity.generate())
    relay = Libp2pTransport(PeerIdentity.generate(), relay=True).start()
    owner = serve_over_libp2p(ps)                 # the NAT'd owner
    client = Libp2pTransport(PeerIdentity.generate()).start()
    try:
        circuit = owner.reserve_on(relay.addrs[0])
        assert circuit in owner.circuit_addrs
        shared = next(k for k in keys if not _is_private(k))
        reply = client.rpc(circuit, {"type": "fetch", "keys": [shared], "have": {}}, timeout=30)
        assert tuple(reply["versions"][shared]) == (0, 0)
        assert shared in reply["weights"]         # weights fetched through the relay
    finally:
        client.close()
        owner.close()
        relay.close()
        ps.shutdown()
