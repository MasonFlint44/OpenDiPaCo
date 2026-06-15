"""Tests for W4b graceful departure (docs/w4-churn-design.md §D3 parts 1+3).

Three mechanisms: (1) a signed deregister tombstone makes the scheduler's
EpochManager fail an owner over *immediately*, skipping owner_grace; (2) a forged
tombstone for a live peer is rejected (only the peer can deregister itself);
(3) a worker returning its lease (nack) frees the path for immediate re-lease.
All beside the unchanged TTL+grace path.
"""

import time

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.schedule import (
    EpochManager,
    ParameterServer,
    PeerIdentity,
    Scheduler,
    Tracker,
    make_peer_record,
    sign_record,
)

BATCH = 8


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _corpus(cfg):
    from opendipaco.data import ShardedCorpus
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(0, 40, (48,), generator=g) for _ in range(16)]
    assign = torch.tensor([i % cfg.num_paths for i in range(16)])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _owner_record(identity, port=9000):
    return make_peer_record(identity, reachability="public",
                            addr=("127.0.0.1", port), roles=("owner",))


def _await(pred, timeout, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


# -- (1) EpochManager: a tombstone skips owner_grace ----------------------------


def test_tombstone_drops_owner_immediately_skipping_grace():
    """A tombstoned owner leaves the desired set *now*, even with a long grace
    that would otherwise retain a merely-silent owner."""
    ids = [PeerIdentity.generate() for _ in range(2)]
    a, b = (_owner_record(i, 9000 + n) for n, i in enumerate(ids))
    a_id, b_id = ids[0].peer_id, ids[1].peer_id
    mgr = EpochManager(owner_grace=1000.0, min_epoch_interval=0.0)

    due = mgr.observe([a, b], now=0.0)
    assert {r["peer_id"] for r in due} == {a_id, b_id}
    # b is silent but within the (huge) grace -> normally retained, no bump.
    assert mgr.observe([a], now=1.0) is None
    # Now b is tombstoned (even though a stale record still lists it): dropped now.
    due = mgr.observe([a, b], tombstoned={b_id}, now=2.0)
    assert {r["peer_id"] for r in due} == {a_id}      # b gone despite grace=1000


def test_tombstone_removal_persists_through_rate_limit():
    """A tombstone removes the owner from the desired set even when the bump is
    rate-limited: when the next bump is due, the tombstoned owner stays out."""
    ids = [PeerIdentity.generate() for _ in range(2)]
    a, b = (_owner_record(i, 9000 + n) for n, i in enumerate(ids))
    a_id = ids[0].peer_id
    mgr = EpochManager(owner_grace=1000.0, min_epoch_interval=100.0)

    mgr.observe([a, b], now=0.0)                       # first bump: {a, b}
    # b tombstoned, but within min_epoch_interval -> deferred (None) ...
    assert mgr.observe([a, b], tombstoned={ids[1].peer_id}, now=1.0) is None
    # ... and when the rate limit clears, the due set excludes b.
    due = mgr.observe([a], now=200.0)
    assert due is not None and {r["peer_id"] for r in due} == {a_id}


# -- (2) tracker: tombstone surfaced; forged tombstone rejected ------------------


def test_tracker_surfaces_tombstone_and_rejects_forgery():
    from opendipaco.schedule.tracker import deregister_peer, fetch_directory_and_tombstones

    ida, idb = PeerIdentity.generate(), PeerIdentity.generate()
    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True, ttl=30.0)
    tracker.start()
    taddr = ("127.0.0.1", tracker.port)
    try:
        from opendipaco.schedule.tracker import tracker_rpc
        for i, port in ((ida, 9001), (idb, 9002)):
            tracker_rpc(taddr, {"type": "register", "record": _owner_record(i, port)})
        assert {r["peer_id"] for r in tracker.records()} == {ida.peer_id, idb.peer_id}

        # b deregisters itself: tombstoned, dropped from records, surfaced atomically.
        deregister_peer(taddr, idb)
        assert tracker.tombstones() == [idb.peer_id]
        recs, tombs = fetch_directory_and_tombstones(taddr, roles=["owner"],
                                                     reachability="public")
        assert {r["peer_id"] for r in recs} == {ida.peer_id} and tombs == [idb.peer_id]

        # Forged tombstone for the *live* peer a: claim a's peer_id but sign with
        # b's key. verify_record rejects (peer_id != hash(b.pub)); a stays.
        forged = sign_record(idb, {"kind": "deregister", "issued_at": time.time()})
        forged["peer_id"] = ida.peer_id                # lie about whose departure
        assert tracker._deregister(forged)["type"] == "refused"
        assert ida.peer_id in {r["peer_id"] for r in tracker.records()}
    finally:
        tracker.shutdown()


def test_expel_tombstone_is_not_surfaced_for_fast_evict():
    """A tracker-initiated expel tombstones the peer (drops it from the directory)
    but carries no peer-signed deregister, so it is NOT surfaced as a fast-evict
    tombstone -- a compromised tracker can't skip owner_grace. Only a peer's own
    signed deregister fast-evicts; an expel falls back to TTL+grace."""
    from opendipaco.schedule.tracker import fetch_directory_and_tombstones, tracker_rpc

    ida, idb = PeerIdentity.generate(), PeerIdentity.generate()
    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True, ttl=30.0)
    tracker.start()
    taddr = ("127.0.0.1", tracker.port)
    try:
        for i, port in ((ida, 9001), (idb, 9002)):
            tracker_rpc(taddr, {"type": "register", "record": _owner_record(i, port)})
        tracker.expel(idb)                                 # tracker-forced, unsigned
        assert idb.peer_id in tracker.tombstones()         # tombstoned (record None)
        recs, tombs = fetch_directory_and_tombstones(taddr, roles=["owner"],
                                                     reachability="public")
        assert idb.peer_id not in {r["peer_id"] for r in recs}   # gone from directory
        assert tombs == []                                 # but NOT a fast-evict signal
    finally:
        tracker.shutdown()


# -- (3) worker lease return (nack) frees the path immediately -------------------


def test_scheduler_nack_frees_lease_and_fences_stale_token():
    """A nack with the live lease frees the path for immediate re-lease; a nack
    with a stale token (zombie / already re-leased) is refused."""
    cfg = _cfg()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], DiLoCoConfig(inner_steps=2),
                      batch_size=BATCH, host="127.0.0.1", port=0)
    sched.start()
    try:
        sched._completed = {p: 0 for p in sched.paths}   # fit() does this; we poke directly
        sched._serving = True
        sched._target = sched._T + 10 * len(sched.paths)
        task = sched._next_task({"worker_id": "w1", "warm_paths": [], "cached_shards": []})
        assert task["type"] == "task"
        path, lease = task["path"], task["lease"]
        assert path in sched._inflight

        # Stale token: refused, lease untouched.
        assert sched._nack({"path": path, "lease": "wrong"})["freed"] is False
        assert path in sched._inflight

        # Live token: freed now, path re-leasable immediately.
        assert sched._nack({"path": path, "lease": lease})["freed"] is True
        assert path not in sched._inflight
        again = sched._next_task({"worker_id": "w2", "warm_paths": [], "cached_shards": []})
        assert again["type"] == "task" and again["path"] == path   # re-leased at once
        # A second nack with the original (now stale) lease can't free the re-lease.
        assert sched._nack({"path": path, "lease": lease})["freed"] is False
        assert path in sched._inflight
    finally:
        sched.shutdown()


def test_worker_stop_event_exits_cleanly_without_leasing():
    """The graceful-leave plumbing through run_sharded_worker: a worker whose
    stop_event is already set leaves at the loop top (after connecting) without
    taking a task, and the call returns promptly instead of blocking."""
    import threading

    from opendipaco.schedule import run_sharded_worker

    cfg = _cfg()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], DiLoCoConfig(inner_steps=2),
                      batch_size=BATCH, host="127.0.0.1", port=0)
    sched.start()
    stop = threading.Event()
    stop.set()
    try:
        t = threading.Thread(
            target=run_sharded_worker,
            args=(cfg, DiLoCoConfig(inner_steps=2), ("127.0.0.1", sched.port)),
            kwargs=dict(stop_event=stop), daemon=True)
        t.start()
        t.join(timeout=10)
        assert not t.is_alive()                 # exited promptly, didn't block on tasks
    finally:
        sched.shutdown()


# -- owner graceful shutdown deregisters -----------------------------------------


def test_owner_graceful_shutdown_deregisters():
    """ParameterServer.shutdown(graceful=True) sends a signed deregister so the
    tracker tombstones it (skipping grace downstream); graceful=False does not."""
    cfg = _cfg()
    ident = PeerIdentity.generate()
    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True, ttl=30.0)
    tracker.start()
    ps = ParameterServer(cfg, [], DiLoCoConfig(), host="127.0.0.1", port=0,
                         identity=ident, replicate_interval=60.0)
    try:
        ps.start()
        ps.start_tracker_heartbeat(("127.0.0.1", tracker.port), "127.0.0.1", interval=0.2)
        assert _await(lambda: ident.peer_id in {r["peer_id"] for r in tracker.records()}, 3)
        ps.shutdown(graceful=True)
        # Tombstoned: out of records, surfaced as a tombstone.
        assert _await(lambda: ident.peer_id in tracker.tombstones(), 3)
        assert ident.peer_id not in {r["peer_id"] for r in tracker.records()}
    finally:
        tracker.shutdown()
