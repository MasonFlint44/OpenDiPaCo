"""Tests for W4c primary drain on graceful departure (docs/w4-churn-design.md D3
part 2). A leaving primary pushes its latest state to rank-1 (the successor), so
a promoted backup holds the *last* accepted push -- collapsing the failover loss
window to ~0. The push-direction inverse of the include_state replication pull:
exact bytes + outer momentum, version-gated, owner-session gated.
"""

import time

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    make_epoch_record,
    make_grant,
    make_peer_record,
    owners_for,
)
from opendipaco.topology import is_private_key


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _diloco():
    return DiLoCoConfig(inner_steps=4, inner_lr=1e-3)


def _two_owners_k2(cfg):
    """Two owners, k=2: every key is owned by both (rank 0 = primary, rank 1 =
    backup), bootstrap epoch so both boot active."""
    sched_id = PeerIdentity.generate()
    ids = [PeerIdentity.generate() for _ in range(2)]
    pss = [ParameterServer(cfg, [], _diloco(), host="127.0.0.1", port=0, identity=i,
                           replicate_interval=60.0,
                           admitted_peers=[p for p in ids if p is not i])
           for i in ids]
    recs = [make_peer_record(i, reachability="public", addr=("127.0.0.1", ps.port),
                             roles=("owner",)) for i, ps in zip(ids, pss)]
    epoch = make_epoch_record(sched_id, epoch=0, owner_records=recs, k=2, bootstrap=True)
    for ps in pss:
        ps.apply_epoch(epoch)
        ps.start()
    return pss, epoch


def _shared_key(cfg, epoch, pss):
    """A shared key plus its (primary, backup) owners under the epoch."""
    topo = cfg.build_topology()
    path = topo.path_from_index(0)
    key = next(k for k in topo.path_module_keys(path) if not is_private_key(k))
    owners = owners_for(key, epoch)
    primary = next(ps for ps in pss if ps.peer_id == owners[0]["peer_id"])
    backup = next(ps for ps in pss if ps.peer_id == owners[1]["peer_id"])
    return path, key, primary, backup


def test_drain_pushes_latest_state_to_rank1_exact_bytes():
    """The primary advances a key (push), leaving rank-1 a version behind; a
    drain makes rank-1 hold the primary's *exact* bytes + version -- the loss
    window collapses to 0 instead of ~replicate_interval."""
    cfg = _cfg()
    pss, epoch = _two_owners_k2(cfg)
    try:
        path, key, primary, backup = _shared_key(cfg, epoch, pss)
        grad = [torch.ones_like(p) for p in primary.bank[key].parameters()]
        assert primary._push({"grant": make_grant(path, [key], 1.0, "t0"),
                              "updates": {key: {"grad": grad}}})["applied"] is True
        # rank-1 is now stale (it never pulled): different version, different bytes.
        assert backup._versions[key] != primary._versions[key]

        drained = primary._drain_to_backups()
        assert drained.get(key) == "drained"
        # rank-1 now holds the primary's exact version *and* bytes (+ momentum).
        assert backup._versions[key] == primary._versions[key]
        pa = dict(primary.bank[key].named_parameters())
        pb = dict(backup.bank[key].named_parameters())
        assert all(torch.equal(pa[n], pb[n]) for n in pa)        # exact, not bf16
    finally:
        for ps in pss:
            ps.shutdown()


def test_drain_recv_is_version_gated():
    """A drain carrying a version <= what the receiver holds is ignored
    (idempotent / last-writer-wins); a strictly newer one is adopted."""
    cfg = _cfg()
    pss, epoch = _two_owners_k2(cfg)
    try:
        path, key, primary, backup = _shared_key(cfg, epoch, pss)
        # Bring rank-1 current first (a real drain), then re-send the same version.
        grad = [torch.ones_like(p) for p in primary.bank[key].parameters()]
        primary._push({"grant": make_grant(path, [key], 1.0, "t0"),
                       "updates": {key: {"grad": grad}}})
        primary._drain_to_backups()
        v = backup._versions[key]
        before = {n: p.clone() for n, p in backup.bank[key].named_parameters()}

        # Re-drain the *same* version: receiver must ignore it (not newer).
        from opendipaco.schedule.sharded import _opt_to_wire, _optimizer_state_to_cpu, _state_to_cpu
        payload = {"version": v, "weights": _state_to_cpu(primary.bank[key].state_dict()),
                   "state": _opt_to_wire(_optimizer_state_to_cpu(
                       primary._outer_opts[key].state_dict()))}
        r = backup._drain_recv({"states": {key: payload}}, primary.peer_id)
        assert r["adopted"] == [] and backup._versions[key] == v
        after = dict(backup.bank[key].named_parameters())
        assert all(torch.equal(before[n], after[n]) for n in before)
    finally:
        for ps in pss:
            ps.shutdown()


def test_drain_recv_rejects_non_owner_session():
    """Drain state is applied only from a session authenticated as a current
    owner of the key (same gate as the include_state pull); a stranger's push --
    even with a newer version -- is refused."""
    cfg = _cfg()
    pss, epoch = _two_owners_k2(cfg)
    try:
        path, key, primary, backup = _shared_key(cfg, epoch, pss)
        from opendipaco.schedule.sharded import _state_to_cpu
        newer = (primary._versions[key][0], primary._versions[key][1] + 5)
        payload = {"version": newer, "weights": _state_to_cpu(primary.bank[key].state_dict())}
        r = backup._drain_recv({"states": {key: payload}}, "not-an-owner-peer-id")
        assert r["adopted"] == []                                # refused
        assert backup._versions[key] != newer
    finally:
        for ps in pss:
            ps.shutdown()


def test_drain_refused_in_decentralized_mode():
    """Decentralized mode never applies a pushed drain (a single source's version
    isn't quorum-confirmable -> poison hazard); the pull+quorum path applies."""
    cfg = _cfg()
    pss, epoch = _two_owners_k2(cfg)
    try:
        path, key, primary, backup = _shared_key(cfg, epoch, pss)
        from opendipaco.schedule.sharded import _state_to_cpu
        backup.schedule_mode = "decentralized"
        newer = (primary._versions[key][0], primary._versions[key][1] + 5)
        payload = {"version": newer, "weights": _state_to_cpu(primary.bank[key].state_dict())}
        assert backup._drain_recv({"states": {key: payload}}, primary.peer_id)["adopted"] == []
        # And a decentralized *sender* drains nothing.
        primary.schedule_mode = "decentralized"
        assert primary._drain_to_backups() == {}
    finally:
        for ps in pss:
            ps.shutdown()


def test_drain_failure_degrades_to_pull_window_no_wedge():
    """If rank-1 is unreachable, the drain is best-effort: it raises nothing and
    drains nothing, falling back to the normal replicate_interval pull window."""
    cfg = _cfg()
    pss, epoch = _two_owners_k2(cfg)
    try:
        path, key, primary, backup = _shared_key(cfg, epoch, pss)
        grad = [torch.ones_like(p) for p in primary.bank[key].parameters()]
        primary._push({"grant": make_grant(path, [key], 1.0, "t0"),
                       "updates": {key: {"grad": grad}}})
        backup.shutdown()                                        # successor is gone
        time.sleep(0.1)
        assert primary._drain_to_backups() == {}                # no wedge, no raise
    finally:
        for ps in pss:
            ps.shutdown()


def test_draining_primary_refuses_writes():
    """Once draining (graceful leave committed), the primary refuses pushes so
    none race the drain and get silently lost; the worker re-routes to the new
    primary after the bump (the existing not_primary retry path)."""
    cfg = _cfg()
    pss, epoch = _two_owners_k2(cfg)
    try:
        path, key, primary, _backup = _shared_key(cfg, epoch, pss)
        with primary._lock:
            primary._draining = True
        grad = [torch.ones_like(p) for p in primary.bank[key].parameters()]
        r = primary._push({"grant": make_grant(path, [key], 1.0, "t0"),
                           "updates": {key: {"grad": grad}}})
        assert r["applied"] is False and r["reason"] == "not_primary"
    finally:
        for ps in pss:
            ps.shutdown()


def test_graceful_shutdown_drains_before_leaving():
    """shutdown(graceful=True) runs the drain end to end: a stale rank-1 holds the
    departing primary's last push afterward."""
    cfg = _cfg()
    pss, epoch = _two_owners_k2(cfg)
    try:
        path, key, primary, backup = _shared_key(cfg, epoch, pss)
        grad = [torch.ones_like(p) for p in primary.bank[key].parameters()]
        primary._push({"grant": make_grant(path, [key], 1.0, "t0"),
                       "updates": {key: {"grad": grad}}})
        assert backup._versions[key] != primary._versions[key]
        want = primary._versions[key]
        primary.shutdown(graceful=True)
        assert backup._versions[key] == want                    # drained on the way out
    finally:
        for ps in pss:
            ps.shutdown()
