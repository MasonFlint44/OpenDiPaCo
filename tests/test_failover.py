"""Tests for failover orchestration (internet-scale plan, Phase 2c).

EpochManager hysteresis, runtime epoch polling by owners, zombie/lame-duck
handling on remap, and the marquee scenario: a primary owner dies mid-run,
tracker liveness drives an epoch bump, the backup is promoted, and training
completes with bounded loss.
"""

import threading
import time

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import (
    EpochManager,
    ParameterServer,
    PeerIdentity,
    Scheduler,
    Tracker,
    make_epoch_record,
    make_grant,
    make_peer_record,
    owners_for,
    run_sharded_worker,
)
from opendipaco.topology import is_private_key

BATCH = 8


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _diloco():
    return DiLoCoConfig(inner_steps=4, inner_lr=1e-3)


def _corpus(cfg):
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(8)]
    assign = torch.tensor([i % cfg.num_paths for i in range(32)])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _owner_record(identity, port):
    return make_peer_record(identity, reachability="public",
                            addr=("127.0.0.1", port), roles=("owner",))


def test_epoch_manager_hysteresis():
    """An owner leaves only after owner_grace of silence; flaps cause no bump;
    bumps are rate-limited; an unchanged set never bumps."""
    ids = [PeerIdentity.generate() for _ in range(3)]
    recs = {i.peer_id: _owner_record(i, 9000 + n) for n, i in enumerate(ids)}
    a, b, c = (recs[i.peer_id] for i in ids)
    mgr = EpochManager(owner_grace=10.0, min_epoch_interval=5.0)

    assert mgr.observe([], now=0.0) is None                  # empty swarm: no epoch
    due = mgr.observe([a, b], now=1.0)                       # first sighting bumps
    assert {r["peer_id"] for r in due} == {a["peer_id"], b["peer_id"]}
    assert mgr.observe([a, b], now=2.0) is None              # unchanged set
    assert mgr.observe([a], now=4.0) is None                 # b silent, within grace
    assert mgr.observe([a, b], now=5.0) is None              # b flapped back: no bump
    # c joins -> change is due, but rate-limited until min_epoch_interval passes.
    assert mgr.observe([a, b, c], now=5.5) is None           # 4.5s since the bump
    due = mgr.observe([a, b, c], now=6.5)                    # batched, on schedule
    assert {r["peer_id"] for r in due} == set(recs)
    # b goes silent for good: dropped only after grace, in one batched bump.
    assert mgr.observe([a, c], now=15.0) is None             # 8.5s silent: not yet
    assert mgr.observe([a, c], now=16.0) is None             # 9.5s silent: not yet
    due = mgr.observe([a, c], now=17.0)                      # >10s silent: dropped
    assert {r["peer_id"] for r in due} == {a["peer_id"], c["peer_id"]}
    # A re-registration on a new address counts as a change.
    moved = _owner_record(ids[0], 9999)
    assert mgr.observe([moved, c], now=18.0) is None         # rate-limited first
    due = mgr.observe([moved, c], now=23.0)
    assert any(r["addr"] == ["127.0.0.1", 9999] for r in due)


def test_owner_polls_scheduler_for_epochs():
    """A running owner learns ownership changes by polling the scheduler's epoch
    RPC from its replication loop: the bootstrap first epoch boot-serves, a
    later epoch is adopted (and a stale one ignored) without restarts."""
    cfg = _cfg()
    sched_id = PeerIdentity.generate()
    own_id = PeerIdentity.generate()
    sched = Scheduler(cfg, _corpus(cfg), [], _diloco(), batch_size=BATCH,
                      host="127.0.0.1", port=0, identity=sched_id,
                      admitted_peers=[own_id])
    sched.start()
    ps = ParameterServer(cfg, [], _diloco(), host="127.0.0.1", port=0,
                         identity=own_id, scheduler_pub=sched_id.public_key_hex,
                         scheduler_addr=("127.0.0.1", sched.port),
                         replicate_interval=60.0)
    try:
        ps._poll_epoch()                       # nothing published yet: a no-op
        assert ps._epoch is None
        r0 = sched.publish_epoch([_owner_record(own_id, ps.port)], k=1)
        assert r0["bootstrap"] is True         # fresh run: auto-flagged
        ps._poll_epoch()
        assert ps._epoch_num == 0 and ps.owned_keys
        assert ps._active == ps.owned_keys     # bootstrap epoch boot-serves

        other = PeerIdentity.generate()
        r1 = sched.publish_epoch(
            [_owner_record(own_id, ps.port), _owner_record(other, 9999)], k=1)
        assert r1["bootstrap"] is False        # only the first epoch bootstraps
        ps._poll_epoch()
        assert ps._epoch_num == 1
        assert ps.owned_keys < set(cfg.build_topology().module_keys())  # split now
        ps._poll_epoch()                       # idempotent on the same record
        assert ps._epoch_num == 1
    finally:
        ps.shutdown()
        sched.shutdown()


def test_zombie_fenced_and_lame_duck_lifecycle():
    """An owner remapped away from a key refuses writes for it (zombie fencing)
    but keeps serving reads for one epoch (lame duck) -- during which the new
    owner cold-syncs *from* it -- and drops it on the epoch after."""
    cfg = _cfg()
    sched_id = PeerIdentity.generate()
    ida, idb = PeerIdentity.generate(), PeerIdentity.generate()
    pss = [ParameterServer(cfg, [], _diloco(), host="127.0.0.1", port=0, identity=i,
                           replicate_interval=60.0,
                           admitted_peers=[p for p in (ida, idb) if p is not i])
           for i in (ida, idb)]
    a, b = pss
    recs = [_owner_record(ida, a.port), _owner_record(idb, b.port)]
    epoch0 = make_epoch_record(sched_id, epoch=0, owner_records=recs, k=1, bootstrap=True)
    for ps in pss:
        ps.apply_epoch(epoch0)
        ps.start()
    try:
        # k=1: each key has a single owner. Train one of A's keys.
        key = next(k for k in sorted(a.owned_keys) if not is_private_key(k))
        path = cfg.build_topology().path_from_index(0)
        grad = [torch.ones_like(p) for p in a.bank[key].parameters()]
        assert a._push({"grant": make_grant(path, [key], 1.0, "t0"),
                        "updates": {key: {"grad": grad}}})["applied"] is True
        trained = {n: p.detach().clone() for n, p in a.bank[key].named_parameters()}

        # Epoch 1 hands everything to B. A fences writes but still serves reads.
        epoch1 = make_epoch_record(sched_id, epoch=1, owner_records=recs[1:], k=1)
        for ps in pss:
            ps.apply_epoch(epoch1)
        r = a._push({"grant": make_grant(path, [key], 1.0, "t1"),
                     "updates": {key: {"grad": grad}}})
        assert r["applied"] is False and r["reason"] == "not_primary" and r["epoch"] == 1
        assert key in a._fetch({"type": "fetch", "keys": [key], "have": {}})["weights"]

        # B gained the key syncing; its pull sources include last epoch's owner
        # (the lame duck A) -- that is where the trained bytes actually live.
        assert key in b.owned_keys and key not in b._active
        b.admit_peer(ida), a.admit_peer(idb)
        assert b._replicate_once()[key] == "active"
        got = dict(b.bank[key].named_parameters())
        assert all(torch.equal(trained[n], got[n]) for n in trained)
        assert b._versions[key] == a._versions[key]

        # The epoch after that, A finally drops the lame duck.
        epoch2 = make_epoch_record(sched_id, epoch=2, owner_records=recs[1:], k=1)
        a.apply_epoch(epoch2)
        assert key not in a.bank
        assert key in a._fetch({"type": "fetch", "keys": [key], "have": {}})["missing"]
    finally:
        for ps in pss:
            ps.shutdown()


def test_kill_primary_mid_run_completes_with_promotion():
    """The marquee 2c scenario over real TCP: tracker liveness + watch_tracker
    drive epochs; a primary owner crashes mid-run; its record expires, the
    epoch bumps it out, backups are promoted, workers fail over; training
    still reaches the update target."""
    cfg, dl = _cfg(), _diloco()
    sched_id = PeerIdentity.generate()
    ids = [PeerIdentity.generate() for _ in range(3)]

    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True, ttl=1.0)
    tracker.start()
    taddr = ("127.0.0.1", tracker.port)
    sched = Scheduler(cfg, _corpus(cfg), [], dl, batch_size=BATCH,
                      host="127.0.0.1", port=0, identity=sched_id, auth_key="t",
                      admitted_peers=ids, heartbeat_timeout=2.0)
    sched.start()
    pss = [ParameterServer(cfg, [], dl, host="127.0.0.1", port=0, identity=i,
                           auth_key="t", scheduler_pub=sched_id.public_key_hex,
                           scheduler_addr=("127.0.0.1", sched.port),
                           replicate_interval=0.2,
                           admitted_peers=[p for p in ids if p is not i])
           for i in ids]
    workers = []
    try:
        for ps in pss:
            ps.start()
            ps.start_tracker_heartbeat(taddr, "127.0.0.1", interval=0.3)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(tracker.records()) < 3:
            time.sleep(0.05)  # all three owners registered before the first epoch
        assert len(tracker.records()) == 3
        sched.watch_tracker(taddr, k=2, owner_grace=2.0, min_epoch_interval=0.5,
                            poll_interval=0.2)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(ps._epoch is None for ps in pss):
            time.sleep(0.05)  # owners adopt the bootstrap epoch via their polls
        assert all(ps._epoch_num == 0 and ps._active == ps.owned_keys for ps in pss)

        workers = [threading.Thread(
            target=run_sharded_worker, args=(cfg, dl, ("127.0.0.1", sched.port)),
            kwargs=dict(seed=0, auth_key="t", heartbeat_interval=0.5,
                        reconnect=True, reconnect_timeout=10.0),
            daemon=True) for _ in range(2)]
        for w in workers:
            w.start()

        victim = pss[0]
        def kill_after_progress():
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and sched._T < sched._target // 3:
                time.sleep(0.05)
            victim.simulate_crash()  # drops every connection; heartbeat stops
        killer = threading.Thread(target=kill_after_progress, daemon=True)
        killer.start()

        completed = sched.fit(num_generations=4, total_generations=4)
        killer.join(timeout=30)
        assert sum(completed.values()) >= sched._target  # training rode out the death

        final = sched._epoch_record
        assert final["epoch"] >= 1                        # the death bumped the epoch
        assert victim.peer_id not in {o["peer_id"] for o in final["owners"]}
        # Every key is served by live, promoted owners under the final epoch.
        survivors = pss[1:]
        for k in cfg.build_topology().module_keys():
            owner_ids = {o["peer_id"] for o in owners_for(k, final)}
            assert owner_ids <= {ps.peer_id for ps in survivors}
            assert any(k in ps._active for ps in survivors if ps.peer_id in owner_ids)
    finally:
        sched.shutdown()
        for ps in pss:
            ps.shutdown()
        tracker.shutdown()
        for w in workers:
            w.join(timeout=15)
