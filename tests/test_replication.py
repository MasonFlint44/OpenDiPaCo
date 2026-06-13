"""Tests for replicated dynamic ownership (internet-scale plan, Phase 2b).

Pull-based replication between owner peers: (epoch, counter) version pairs,
the syncing→active lifecycle, owner-session gating of exact-state pulls,
primary-only writes, and a full rendezvous-mode training run where workers
fetch from any replica and push to primaries.
"""

import threading
import time

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    Scheduler,
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
    # Private embeddings so replication is exercised on private modules too.
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16,
                        embedding="private")


def _diloco():
    return DiLoCoConfig(inner_steps=4, inner_lr=1e-3)


def _corpus(cfg):
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(8)]
    assign = torch.tensor([i % cfg.num_paths for i in range(32)])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _two_owner_cluster(cfg, sched_id, *, k=2, replicate_interval=60.0, start=True):
    """Two owner PSs with identities + a signed epoch over both (every key gets
    both as replicas when k=2; HRW decides which is primary per key)."""
    ids = [PeerIdentity.generate() for _ in range(2)]
    pss = [ParameterServer(cfg, [], _diloco(), host="127.0.0.1", port=0,
                           identity=i, replicate_interval=replicate_interval,
                           admitted_peers=[p for p in ids if p is not i])
           for i in ids]
    recs = [make_peer_record(i, reachability="public",
                             addr=("127.0.0.1", ps.port), roles=("owner",))
            for i, ps in zip(ids, pss)]
    epoch = make_epoch_record(sched_id, epoch=0, owner_records=recs, k=k)
    for ps in pss:
        ps.apply_epoch(epoch, bootstrap=True)
        if start:
            ps.start()
    return ids, pss, recs, epoch


def _primary_backup(pss, key, epoch):
    prim_id = owners_for(key, epoch)[0]["peer_id"]
    prim = next(ps for ps in pss if ps.peer_id == prim_id)
    back = next(ps for ps in pss if ps.peer_id != prim_id)
    return prim, back


def _push_ones(ps, cfg, key, token, weight=1.0):
    path = cfg.build_topology().path_from_index(0)
    grad = [torch.ones_like(p) for p in ps.bank[key].parameters()]
    return ps._push({"grant": make_grant(path, [key], weight, token),
                     "updates": {key: {"grad": grad}}})


def _shutdown(pss):
    for ps in pss:
        ps.shutdown()


def test_version_pairs_order_across_epochs():
    """Versions are (epoch, counter): the counter resets on an epoch bump and the
    pair stays totally ordered, so a later epoch's first write supersedes any
    count from an earlier one (the failover re-issue ambiguity, design D5)."""
    cfg = _cfg()
    sched_id = PeerIdentity.generate()
    ids, pss, recs, epoch = _two_owner_cluster(cfg, sched_id, start=False)
    try:
        key = next(k for k in pss[0].owned_keys if not is_private_key(k))
        prim, _ = _primary_backup(pss, key, epoch)
        assert prim._versions[key] == (0, 0)
        assert _push_ones(prim, cfg, key, "t1")["applied"] is True
        assert _push_ones(prim, cfg, key, "t2")["applied"] is True
        assert prim._versions[key] == (0, 2)

        bumped = make_epoch_record(sched_id, epoch=3, owner_records=recs, k=2)
        for ps in pss:
            ps.apply_epoch(bumped)
        assert _push_ones(prim, cfg, key, "t3")["applied"] is True
        assert prim._versions[key] == (3, 1)       # counter reset under the new epoch
        assert (3, 1) > (0, 2)                     # ...but still strictly newer
        # A stale (older) epoch record is ignored.
        prim.apply_epoch(epoch)
        assert prim._epoch_num == 3
    finally:
        _shutdown(pss)


def test_non_primary_push_refused():
    """Backups copy state; they never apply writes. A push routed to a backup is
    refused with the current epoch so the worker can learn it had stale routing."""
    cfg = _cfg()
    ids, pss, recs, epoch = _two_owner_cluster(cfg, PeerIdentity.generate(), start=False)
    try:
        key = next(k for k in pss[0].owned_keys if not is_private_key(k))
        prim, back = _primary_backup(pss, key, epoch)
        before = {n: p.detach().clone() for n, p in back.bank[key].named_parameters()}
        r = _push_ones(back, cfg, key, "tok")
        assert r["applied"] is False and r["reason"] == "not_primary" and r["epoch"] == 0
        assert all(torch.equal(before[n], p) for n, p in back.bank[key].named_parameters())
        assert back._versions[key] == (0, 0)
    finally:
        _shutdown(pss)


def test_include_state_gated_on_owner_session():
    """Exact-state pulls (weights + momentum, uncompressed) are served only to a
    session identity-authenticated as one of the key's owners; everyone else
    gets 'missing' -- never a silently degraded or unauthorized copy."""
    cfg = _cfg()
    ids, pss, recs, epoch = _two_owner_cluster(cfg, PeerIdentity.generate(), start=False)
    try:
        key = next(k for k in pss[0].owned_keys if not is_private_key(k))
        prim, back = _primary_backup(pss, key, epoch)
        msg = {"type": "fetch", "keys": [key], "have": {}, "include_state": True}

        assert key in prim._fetch(dict(msg), peer_id=None)["missing"]          # unauthenticated
        stranger = PeerIdentity.generate()
        assert key in prim._fetch(dict(msg), peer_id=stranger.peer_id)["missing"]  # non-owner
        reply = prim._fetch(dict(msg), peer_id=back.peer_id)                   # fellow owner
        assert key in reply["weights"] and key in reply["state"]
        assert tuple(reply["versions"][key]) == prim._versions[key]
        # A normal worker fetch (no include_state) never carries momentum.
        worker_reply = prim._fetch({"type": "fetch", "keys": [key], "have": {}})
        assert "state" not in worker_reply and key in worker_reply["weights"]
    finally:
        _shutdown(pss)


def test_pull_replication_bit_equal_including_momentum():
    """After a replication pass the backup holds the primary's exact bytes --
    weights AND outer momentum -- for shared and private keys alike. A second
    pass is a no-op (idempotent, version-gated)."""
    cfg = _cfg()
    ids, pss, recs, epoch = _two_owner_cluster(cfg, PeerIdentity.generate())
    try:
        shared = [k for k in pss[0].owned_keys if not is_private_key(k)]
        private = [k for k in pss[0].owned_keys if is_private_key(k)]
        # Train every shared key on its primary (twice, so momentum is non-trivial)
        # and store fresh private state on each private key's primary.
        for i, k in enumerate(shared):
            prim, _ = _primary_backup(pss, k, epoch)
            _push_ones(prim, cfg, k, f"s{i}a")
            _push_ones(prim, cfg, k, f"s{i}b", weight=0.5)
        path = cfg.build_topology().path_from_index(0)
        for i, k in enumerate(private):
            prim, _ = _primary_backup(pss, k, epoch)
            sd = {n: t + 1.0 for n, t in prim.bank[k].state_dict().items()}
            r = prim._push({"grant": make_grant(path, [k], 1.0, f"p{i}"), "private": {k: sd}})
            assert r["applied"] is True

        for ps in pss:
            ps._replicate_once()
        for k in shared + private:
            prim, back = _primary_backup(pss, k, epoch)
            assert back._versions[k] == prim._versions[k]
            a = dict(prim.bank[k].named_parameters())
            b = dict(back.bank[k].named_parameters())
            assert all(torch.equal(a[n], b[n]) for n in a), f"weights differ for {k}"
            if not is_private_key(k):
                sa = prim._outer_opts[k].state_dict()["state"]
                sb = back._outer_opts[k].state_dict()["state"]
                assert set(sa) == set(sb) and sa, f"momentum missing for {k}"
                for idx in sa:
                    assert torch.equal(sa[idx]["momentum_buffer"],
                                       sb[idx]["momentum_buffer"]), f"momentum differs {k}"

        snap = {k: {n: p.detach().clone()
                    for n, p in pss[1].bank[k].named_parameters()} for k in shared}
        pss[1]._replicate_once()  # idempotent: nothing newer, nothing changes
        for k in shared:
            assert all(torch.equal(snap[k][n], p)
                       for n, p in pss[1].bank[k].named_parameters())
    finally:
        _shutdown(pss)


def test_joiner_syncs_before_serving():
    """A peer joining a live cluster (bootstrap=False) serves nothing until its
    pull catches each owned key up; after one pass against live replicas its
    keys are active and byte-identical to the source."""
    cfg = _cfg()
    sched_id = PeerIdentity.generate()
    ids, pss, recs, epoch = _two_owner_cluster(cfg, sched_id)
    joiner_id = PeerIdentity.generate()
    joiner = None
    try:
        # Some training first, so the joiner has real state to catch up to.
        shared = [k for k in pss[0].owned_keys if not is_private_key(k)]
        for i, k in enumerate(shared):
            prim, _ = _primary_backup(pss, k, epoch)
            _push_ones(prim, cfg, k, f"j{i}")
        for ps in pss:
            ps._replicate_once()

        rec3 = recs + [make_peer_record(joiner_id, reachability="public",
                                        addr=("127.0.0.1", 1), roles=("owner",))]
        # (addr above is a placeholder; the joiner serves on its own port and
        # only *pulls* in this test, so nothing dials it.)
        epoch1 = make_epoch_record(sched_id, epoch=1, owner_records=rec3, k=2)
        joiner = ParameterServer(cfg, [], _diloco(), host="127.0.0.1", port=0,
                                 identity=joiner_id, epoch_record=epoch1,
                                 bootstrap=False, replicate_interval=60.0)
        assert joiner.owned_keys and not joiner._active  # owns keys, serves none
        some = sorted(joiner.owned_keys)[0]
        assert some in joiner._fetch({"type": "fetch", "keys": [some], "have": {}})["missing"]

        for ps in pss:
            ps.apply_epoch(epoch1)
            ps.admit_peer(joiner_id)  # enrollment: the swarm admits the new owner
        results = joiner._replicate_once()
        assert set(results.values()) == {"active"}
        assert joiner._active == joiner.owned_keys
        for k in joiner.owned_keys:
            src = next(ps for ps in pss if k in ps.bank and ps.peer_id in
                       {o["peer_id"] for o in owners_for(k, epoch1)})
            a = dict(src.bank[k].named_parameters())
            b = dict(joiner.bank[k].named_parameters())
            assert all(torch.equal(a[n], b[n]) for n in a)
    finally:
        if joiner is not None:
            joiner.shutdown()
        _shutdown(pss)


def test_rendezvous_end_to_end_with_replication():
    """The full Phase 2b shape over real TCP: a rendezvous-mode scheduler
    (epoch-derived routing, Ed25519 grants), two replicated owners syncing in
    the background, and a worker that fetches from replicas and pushes to
    primaries. Training reaches the target and the replicas converge to
    identical versions and bytes."""
    cfg, dl = _cfg(), _diloco()
    sched_id = PeerIdentity.generate()
    ids = [PeerIdentity.generate() for _ in range(2)]
    pss = [ParameterServer(cfg, [], dl, host="127.0.0.1", port=0,
                           identity=i, replicate_interval=0.2, auth_key="t",
                           scheduler_pub=sched_id.public_key_hex,
                           admitted_peers=[p for p in ids if p is not i])
           for i in ids]
    recs = [make_peer_record(i, reachability="public",
                             addr=("127.0.0.1", ps.port), roles=("owner",))
            for i, ps in zip(ids, pss)]
    sched = Scheduler(cfg, _corpus(cfg), [], dl, batch_size=BATCH,
                      host="127.0.0.1", port=0, identity=sched_id, auth_key="t")
    epoch = sched.publish_epoch(recs, k=2)
    for ps in pss:
        ps.apply_epoch(epoch, bootstrap=True)
        ps.start()
    sched.start()
    w = threading.Thread(target=run_sharded_worker, args=(cfg, dl, ("127.0.0.1", sched.port)),
                         kwargs=dict(seed=0, auth_key="t", heartbeat_interval=1.0), daemon=True)
    w.start()
    try:
        completed = sched.fit(num_generations=2, total_generations=2)
        assert sum(completed.values()) >= sched._target
        w.join(timeout=10)

        shared = [k for k in pss[0].owned_keys if not is_private_key(k)]
        assert any(pss[0]._versions[k] > (0, 0) or pss[1]._versions[k] > (0, 0)
                   for k in shared)  # training moved the bank
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:  # background pull loops converge
            if all(pss[0]._versions[k] == pss[1]._versions[k] for k in shared):
                break
            time.sleep(0.1)
        for k in shared:
            assert pss[0]._versions[k] == pss[1]._versions[k], f"replicas diverged on {k}"
            a = dict(pss[0].bank[k].named_parameters())
            b = dict(pss[1].bank[k].named_parameters())
            assert all(torch.equal(a[n], b[n]) for n in a)
    finally:
        sched.shutdown()
        _shutdown(pss)
        w.join(timeout=10)
