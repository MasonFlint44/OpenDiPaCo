"""Tests for per-key persistence + recovery (internet-scale plan, Phase 2d).

Per-key checkpoint files that survive ownership remaps, warm restarts that
delta-sync instead of cold-syncing, exhaustive reconciliation that adopts the
max replica version (the promotion-safety property), the scheduler's signed
recovery manifest with its readiness gate, and the rendezvous-mode `run_local`
smoke (the `opendipaco run` path with `ownership.mode: rendezvous`).
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.launch.config import LaunchConfig
from opendipaco.launch.roles import run_local
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    Scheduler,
    make_epoch_record,
    make_grant,
    make_peer_record,
    owners_for,
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


def _cluster(cfg, sched_id, ids, *, k=None):
    pss = [ParameterServer(cfg, [], _diloco(), host="127.0.0.1", port=0, identity=i,
                           replicate_interval=60.0, auth_key="t",
                           admitted_peers=[p for p in ids if p is not i])
           for i in ids]
    recs = [make_peer_record(i, reachability="public",
                             addr=("127.0.0.1", ps.port), roles=("owner",))
            for i, ps in zip(ids, pss)]
    epoch = make_epoch_record(sched_id, epoch=0, owner_records=recs,
                              k=k or len(ids), bootstrap=True)
    for ps in pss:
        ps.apply_epoch(epoch)
        ps.start()
    return pss, recs, epoch


def _push_ones(ps, cfg, key, token, weight=1.0):
    path = cfg.build_topology().path_from_index(0)
    grad = [torch.ones_like(p) for p in ps.bank[key].parameters()]
    return ps._push({"grant": make_grant(path, [key], weight, token),
                     "updates": {key: {"grad": grad}}})


def _params(ps, key):
    return {n: p.detach().clone() for n, p in ps.bank[key].named_parameters()}


def _same(a, b):
    return all(torch.equal(a[n], b[n]) for n in a)


def test_warm_restart_delta_syncs_not_cold(tmp_path):
    """A restarted owner loads its per-key files (trained keys come back at
    their saved version, NOT (0,0)), keeps them out of service until one
    reconciliation pass, and ends byte-identical to the live replica."""
    cfg = _cfg()
    sched_id = PeerIdentity.generate()
    ida, idb = PeerIdentity.generate(), PeerIdentity.generate()
    pss, recs, epoch = _cluster(cfg, sched_id, [ida, idb], k=2)
    a, b = pss
    ckpt = str(tmp_path / "ck")
    restarted = None
    try:
        trained = [k for k in sorted(a.owned_keys) if not is_private_key(k)][:2]
        for i, k in enumerate(trained):
            prim = a if owners_for(k, epoch)[0]["peer_id"] == a.peer_id else b
            _push_ones(prim, cfg, k, f"w{i}")
        for ps in pss:
            ps._replicate_once()  # replicas in sync before the checkpoint
        saved_versions = b.save_modules(ckpt)
        saved_bytes = {k: _params(b, k) for k in trained}
        assert all(saved_versions[k] == (0, 1) for k in trained)

        b.shutdown()
        restarted = ParameterServer(cfg, [], _diloco(), host="127.0.0.1", port=0,
                                    identity=idb, epoch_record=epoch, resume_dir=ckpt,
                                    replicate_interval=60.0, auth_key="t",
                                    admitted_peers=[ida])
        # Warm start: trained keys resumed at their saved version with their
        # saved bytes (not the (0,0) cold init), but NOT served yet.
        for k in trained:
            assert restarted._versions[k] == (0, 1)
            assert _same(saved_bytes[k], _params(restarted, k))
            assert k not in restarted._active
        # Untouched (0,0) keys boot-serve straight away (universal bytes).
        assert any(v == (0, 0) for v in restarted._versions.values())
        assert all(k in restarted._active
                   for k, v in restarted._versions.items() if v == (0, 0))

        results = restarted._replicate_once()  # reconcile against the live owner
        assert restarted._active == restarted.owned_keys
        assert all(r == "active" for r in results.values())
        for k in trained:
            assert _same(_params(a, k), _params(restarted, k))
    finally:
        a.shutdown()
        if restarted is not None:
            restarted.shutdown()


def test_restarting_primary_adopts_max_across_replicas(tmp_path):
    """The promotion-safety property: a primary restarting from a stale disk
    consults EVERY replica and adopts the newest state before serving --
    activating after the first answer could resurrect old bytes under a
    version some backup has already passed."""
    cfg = _cfg()
    sched_id = PeerIdentity.generate()
    ids = [PeerIdentity.generate() for _ in range(3)]
    pss, recs, epoch = _cluster(cfg, sched_id, ids, k=3)
    ckpt = str(tmp_path / "ck")
    restarted = None
    try:
        key = next(k for k in sorted(pss[0]._all_keys) if not is_private_key(k))
        prim_id = owners_for(key, epoch)[0]["peer_id"]
        prim = next(ps for ps in pss if ps.peer_id == prim_id)
        backups = [ps for ps in pss if ps is not prim]

        _push_ones(prim, cfg, key, "t1")
        _push_ones(prim, cfg, key, "t2")
        prim.save_modules(ckpt)                      # disk freezes at (0, 2)
        for b in backups:
            b._replicate_once()                      # both backups at (0, 2)
        for i in range(3):
            _push_ones(prim, cfg, key, f"t3-{i}")    # primary runs on to (0, 5)
        backups[0]._replicate_once()                 # one backup follows to (0, 5)
        ahead, behind = backups                      # ...the other stays at (0, 2)
        assert ahead._versions[key] == (0, 5) and behind._versions[key] == (0, 2)

        prim_identity = prim.identity
        prim.shutdown()                              # crash: memory (0,5) is gone
        restarted = ParameterServer(cfg, [], _diloco(), host="127.0.0.1", port=0,
                                    identity=prim_identity, epoch_record=epoch,
                                    resume_dir=ckpt, replicate_interval=60.0,
                                    auth_key="t",
                                    admitted_peers=[i for i in ids
                                                    if i is not prim_identity])
        assert restarted._versions[key] == (0, 2)    # warm but stale
        assert key not in restarted._active
        restarted._replicate_once()
        # It consulted both backups and adopted the max, byte-for-byte.
        assert restarted._versions[key] == (0, 5)
        assert _same(_params(ahead, key), _params(restarted, key))
        assert key in restarted._active
    finally:
        for ps in pss[1:] + pss[:1]:
            try:
                ps.shutdown()
            except Exception:
                pass
        if restarted is not None:
            restarted.shutdown()


def test_full_cluster_restart_gated_by_manifest(tmp_path):
    """Cluster recovery (design D7): a checkpoint writes per-key files + the
    scheduler's signed manifest; after a full restart the scheduler refuses to
    serve until live owners hold >= the manifest version for every key, which
    the owners reach by reconciling among themselves."""
    cfg = _cfg()
    sched_id = PeerIdentity.generate()
    ids = [PeerIdentity.generate() for _ in range(2)]
    pss, recs, epoch = _cluster(cfg, sched_id, ids, k=2)
    ckpt = str(tmp_path / "ck")
    sched = Scheduler(cfg, _corpus(cfg), [], _diloco(), batch_size=BATCH,
                      host="127.0.0.1", port=0, identity=sched_id, auth_key="t")
    with sched._lock:
        sched._epoch_record = epoch
        sched._T = 7  # pretend training happened (drives scheduler.pt too)
        sched._completed = {p: 1 for p in sched.paths}
    restarted = []
    sched2 = None
    try:
        shared = [k for k in sorted(pss[0]._all_keys) if not is_private_key(k)]
        for i, k in enumerate(shared):
            prim_id = owners_for(k, epoch)[0]["peer_id"]
            prim = next(ps for ps in pss if ps.peer_id == prim_id)
            _push_ones(prim, cfg, k, f"m{i}")
        for ps in pss:
            ps._replicate_once()
        sched._checkpoint_cluster(ckpt)              # per-key files + manifest
        assert sched._manifest is not None
        assert all(tuple(v) == (0, 1) for k, v in sched._manifest["keys"].items()
                   if k in shared)
        for ps in pss:
            ps.shutdown()
        sched.shutdown()

        # --- full restart ---
        sched2 = Scheduler(cfg, _corpus(cfg), [], _diloco(), batch_size=BATCH,
                           host="127.0.0.1", port=0, identity=sched_id, auth_key="t")
        sched2._load_state(ckpt)
        assert sched2._T == 7 and sched2._manifest is not None
        assert not sched2._recovery_ready()          # no epoch, no owners yet

        restarted = [ParameterServer(cfg, [], _diloco(), host="127.0.0.1", port=0,
                                     identity=i, epoch_record=None, resume_dir=None,
                                     replicate_interval=60.0, auth_key="t",
                                     admitted_peers=[p for p in ids if p is not i])
                     for i in ids]
        recs2 = [make_peer_record(i, reachability="public",
                                  addr=("127.0.0.1", ps.port), roles=("owner",))
                 for i, ps in zip(ids, restarted)]
        epoch2 = sched2.publish_epoch(recs2, k=2)
        assert epoch2["bootstrap"] is False          # resumed run: never re-bootstrap
        for ps in restarted:
            ps._resume_dir = ckpt                    # normally the ctor's resume_dir
            ps.apply_epoch(epoch2)
            ps.start()
        # All keys resumed > (0,0) are syncing -> the gate must hold us back.
        assert not sched2._recovery_ready()
        for ps in restarted:
            ps._replicate_once()                     # owners reconcile mutually
        for ps in restarted:
            assert ps._active == ps.owned_keys
        assert sched2._recovery_ready()              # ...and the gate opens
        for k in shared:
            holders = [ps for ps in restarted if k in ps.bank]
            assert holders and all(ps._versions[k] == (0, 1) for ps in holders)
    finally:
        for ps in restarted:
            ps.shutdown()
        if sched2 is not None:
            sched2.shutdown()


def test_run_local_rendezvous_smoke():
    """`opendipaco run` with `ownership.mode: rendezvous`: tracker + scheduler
    (epoch authority) + dynamic owners + workers, all wired in-process; the
    run trains to its budget."""
    cfg = LaunchConfig.from_dict({
        "mode": "sharded",
        "model": {"vocab_size": 48, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "max_position_embeddings": 64,
                  "layers_per_level": [1, 1], "level_sizes": [2, 2],
                  "sequence_length": 16},
        "diloco": {"inner_steps": 2, "inner_lr": 1e-3, "inner_lr_schedule": "constant"},
        "data": {"source": "synthetic", "num_documents": 32, "synthetic_doc_len": 24,
                 "routing": "round_robin"},
        "sharded": {"num_shards": 2},
        "tracker": {"ttl": 2.0},
        "ownership": {"mode": "rendezvous", "k": 2, "replicate_interval": 0.2,
                      "owner_grace": 4.0, "min_epoch_interval": 0.3,
                      "epoch_poll_interval": 0.2, "heartbeat_interval": 0.5},
        "transport": {"host": "127.0.0.1", "port": 0, "heartbeat_timeout": 5.0},
        "run": {"generations": 2, "batch_size": 4, "local_workers": 2},
    })
    scheduler, completed = run_local(cfg)
    assert sum(completed.values()) >= scheduler._target
    assert scheduler._epoch_record is not None       # epochs actually drove routing
    assert len(scheduler._epoch_record["owners"]) >= 1
