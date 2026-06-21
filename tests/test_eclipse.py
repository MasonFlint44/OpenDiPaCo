"""W8 (part 2) slice a: multi-seed bootstrap union (docs/w8-eclipse-sybil-design.md).

A newcomer bootstrapping from one tracker can be eclipsed by a malicious/partitioned
seed that withholds honest peers. fetch_directory_multi unions self-certifying
records across several seeds, so one honest seed restores what others omit; verified
tombstones suppress a replayed within-TTL record.
"""

import time

from opendipaco.schedule import (
    PeerIdentity,
    Tracker,
    deregister_peer,
    fetch_directory,
    fetch_directory_multi,
    register_peer,
)


def _tracker():
    t = Tracker(host="127.0.0.1", port=0, ttl=30.0, open_enrollment=True)
    t.start()
    return t


def _addr(t):
    return ("127.0.0.1", t.port)


def test_union_recovers_peers_a_seed_omits():
    """The eclipse defense: a seed that serves only its own peers (omitting the
    honest ones) can't isolate a newcomer once a second honest seed is in the mix."""
    good, evil = _tracker(), _tracker()
    a, b, e = PeerIdentity.generate(), PeerIdentity.generate(), PeerIdentity.generate()
    try:
        ga, ea = _addr(good), _addr(evil)
        register_peer(ga, a, reachability="public", peer_addr=("10.0.0.1", 1), roles=["owner"])
        register_peer(ga, b, roles=["worker"])
        register_peer(ea, e, reachability="public", peer_addr=("10.0.0.2", 2), roles=["owner"])
        # Single (malicious/partitioned) seed: the newcomer sees only the evil set.
        assert {r["peer_id"] for r in fetch_directory(ea)} == {e.peer_id}
        # Multi-seed union restores the honest peers the evil seed withheld.
        recs, answered = fetch_directory_multi([ea, ga])
        assert answered == 2
        assert {r["peer_id"] for r in recs} == {a.peer_id, b.peer_id, e.peer_id}
    finally:
        good.shutdown()
        evil.shutdown()


def test_union_keeps_the_freshest_record_per_peer():
    """Same peer on two seeds with different issued_at -> the freshest wins (a seed
    can't pin a victim to a stale addr if another serves the newer one)."""
    t1, t2 = _tracker(), _tracker()
    a = PeerIdentity.generate()
    try:
        a1, a2 = _addr(t1), _addr(t2)
        register_peer(a1, a, reachability="public", peer_addr=("10.0.0.1", 100))
        time.sleep(0.02)
        register_peer(a2, a, reachability="public", peer_addr=("10.0.0.1", 200))   # newer
        recs, _ = fetch_directory_multi([a1, a2])
        assert [r["addr"] for r in recs] == [["10.0.0.1", 200]]
    finally:
        t1.shutdown()
        t2.shutdown()


def test_tombstone_suppresses_a_replayed_within_ttl_record():
    """A departed peer's deregister tombstone (from an honest seed) suppresses a
    malicious seed replaying its still-within-TTL registration."""
    good, evil = _tracker(), _tracker()
    a = PeerIdentity.generate()
    try:
        ga, ea = _addr(good), _addr(evil)
        register_peer(ga, a, roles=["worker"])
        register_peer(ea, a, roles=["worker"])     # the evil seed holds the live record
        time.sleep(0.02)
        deregister_peer(ga, a)                      # graceful leave -> tombstone on good
        # Negative control: the evil seed alone DOES still serve the live record,
        # so the suppression below is load-bearing (not vacuously absent).
        assert a.peer_id in {r["peer_id"] for r in fetch_directory(ea)}
        recs, answered = fetch_directory_multi([ga, ea])
        assert answered == 2
        assert a.peer_id not in {r["peer_id"] for r in recs}
    finally:
        good.shutdown()
        evil.shutdown()


def test_all_seeds_unreachable_returns_empty():
    recs, answered = fetch_directory_multi([("127.0.0.1", 1), ("127.0.0.1", 2)], timeout=1.0)
    assert recs == [] and answered == 0


def test_single_seed_matches_fetch_directory():
    t = _tracker()
    a, b = PeerIdentity.generate(), PeerIdentity.generate()
    try:
        addr = _addr(t)
        register_peer(addr, a, roles=["worker"])
        register_peer(addr, b, roles=["owner"], reachability="public", peer_addr=("10.0.0.1", 9))
        multi, answered = fetch_directory_multi([addr])
        assert answered == 1
        assert ({r["peer_id"] for r in multi}
                == {r["peer_id"] for r in fetch_directory(addr)} == {a.peer_id, b.peer_id})
    finally:
        t.shutdown()


def test_seed_quorum_drops_records_only_one_seed_serves():
    """seed_quorum M keeps a peer only if >= M seeds serve it: a Sybil a single
    malicious seed injects (served by 1) is dropped at M=2, while a peer all seeds
    know survives. M=1 is the pure union (both kept)."""
    t1, t2, t3 = _tracker(), _tracker(), _tracker()
    a, b = PeerIdentity.generate(), PeerIdentity.generate()
    try:
        addrs = [_addr(t1), _addr(t2), _addr(t3)]
        for ad in addrs:
            register_peer(ad, a, roles=["worker"])        # a known to all 3 seeds
        register_peer(addrs[0], b, roles=["worker"])       # b injected by 1 seed only
        union, _ = fetch_directory_multi(addrs, seed_quorum=1)
        assert {r["peer_id"] for r in union} == {a.peer_id, b.peer_id}
        quorum, _ = fetch_directory_multi(addrs, seed_quorum=2)
        assert {r["peer_id"] for r in quorum} == {a.peer_id}   # b dropped (served by 1)
    finally:
        t1.shutdown()
        t2.shutdown()
        t3.shutdown()


def test_seed_quorum_tradeoff_drops_an_honest_peer_few_seeds_know():
    """The documented downside of M>1: an HONEST peer only one seed knows is also
    dropped, so seed_quorum can re-introduce eclipse (it's opt-in for that reason)."""
    t1, t2 = _tracker(), _tracker()
    honest = PeerIdentity.generate()
    try:
        addrs = [_addr(t1), _addr(t2)]
        register_peer(addrs[0], honest, roles=["owner"], reachability="public",
                      peer_addr=("10.0.0.1", 1))         # only seed t1 knows it
        assert {r["peer_id"] for r in fetch_directory_multi(addrs, seed_quorum=1)[0]} == {honest.peer_id}
        assert fetch_directory_multi(addrs, seed_quorum=2)[0] == []   # dropped under M=2
    finally:
        t1.shutdown()
        t2.shutdown()


def test_seed_quorum_config_validation():
    import pytest

    from opendipaco.launch import LaunchConfig
    seeds = [["h2", 6], ["h3", 7]]                          # + primary = 3 seeds total
    ok = LaunchConfig.from_dict({"tracker": {"port": 5, "seeds": seeds, "seed_quorum": 3}})
    assert ok.tracker.seed_quorum == 3
    with pytest.raises(ValueError, match="seed_quorum"):   # > #distinct seeds -> drops everything
        LaunchConfig.from_dict({"tracker": {"port": 5, "seeds": seeds, "seed_quorum": 4}})
    with pytest.raises(ValueError, match="seed_quorum"):   # < 1
        LaunchConfig.from_dict({"tracker": {"port": 5, "seed_quorum": 0}})
    # A seed equal to the primary doesn't add reach: the quorum bound is the
    # DISTINCT seed count the worker dedups to (else a max quorum would be
    # unreachable at runtime -> silent self-eclipse).
    with pytest.raises(ValueError, match="seed_quorum"):
        LaunchConfig.from_dict({"tracker": {"connect_host": "h", "port": 5,
                                            "seeds": [["h", 5], ["h2", 6]],
                                            "seed_quorum": 3}})   # distinct = {h:5, h2:6} = 2


def test_tracker_seeds_config_parses_and_validates():
    import pytest

    from opendipaco.launch import LaunchConfig
    ok = LaunchConfig.from_dict({"tracker": {"port": 5, "seeds": [["h2", 6], ["h3", 7]]}})
    assert ok.tracker.seeds == [["h2", 6], ["h3", 7]]
    # A typo'd entry fails at load, not deep in the worker.
    with pytest.raises(ValueError, match="tracker.seeds"):
        LaunchConfig.from_dict({"tracker": {"seeds": [["h2"]]}})          # missing port
    with pytest.raises(ValueError, match="tracker.seeds"):
        LaunchConfig.from_dict({"tracker": {"seeds": [["h2", "bad"]]}})   # non-int port


def test_dedup_seeds_canonicalizes_primary_first_and_int_ports():
    """The one helper every seed path goes through: primary first, distinct extras,
    every entry (host, int(port)) -- so a str vs int port for one tracker collapses."""
    from opendipaco.schedule.tracker import dedup_seeds
    # primary first; a dup of the primary and a str-port alias of an extra collapse.
    out = dedup_seeds(("p", "5"), [["p", 5], ["h2", "6"], ["h2", 6], ["h3", 7]])
    assert out == [("p", 5), ("h2", 6), ("h3", 7)]


def test_fetch_directory_multi_counts_an_aliased_seed_once():
    """One physical tracker named twice (int vs str port) must count ONCE toward the
    quorum -- else a single tracker could self-satisfy seed_quorum=2, defeating the
    injection-resistance the quorum exists to provide."""
    t = _tracker()
    a = PeerIdentity.generate()
    try:
        host, port = _addr(t)
        register_peer((host, port), a, roles=["worker"])
        # Same tracker, two spellings of the port. With per-(host,port) dedup that
        # coerces the port, this is ONE distinct seed -> served once -> dropped at M=2.
        aliased = [(host, port), (host, str(port))]
        assert {r["peer_id"] for r in fetch_directory_multi(aliased, seed_quorum=1)[0]} == {a.peer_id}
        assert fetch_directory_multi(aliased, seed_quorum=2)[0] == []   # not 2 distinct seeds
    finally:
        t.shutdown()


def test_multi_home_register_tolerates_a_garbage_or_refusing_seed(monkeypatch):
    """A seed returning a garbage frame raises a non-OSError (struct.error/ValueError
    from the codec); it must NOT abort registration with the healthy seeds (one bad
    seed can't kill the heartbeat thread and drop the peer everywhere). A tracker that
    *refuses* returns a status dict (no raise) and is logged, not silently lost."""
    from opendipaco.schedule import tracker as tk
    from opendipaco.schedule.sharded import _multi_home_register
    called = []

    def fake_register(addr, identity, **kw):
        called.append(addr)
        if addr == ("bad", 1):
            raise ValueError("garbage frame")              # non-OSError, as from the wire codec
        if addr == ("full", 2):
            return {"type": "refused", "reason": "directory full"}
        return {"type": "ok"}

    monkeypatch.setattr(tk, "register_peer", fake_register)
    _multi_home_register([("bad", 1), ("full", 2), ("good", 3)], PeerIdentity.generate(),
                         reachability="public", peer_addr=None, roles=("worker",),
                         capabilities=None, auth_key=None, tls=None)
    # All three attempted in order -- the garbage seed didn't short-circuit the loop.
    assert called == [("bad", 1), ("full", 2), ("good", 3)]
