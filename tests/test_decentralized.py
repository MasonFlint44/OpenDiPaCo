"""Tests for decentralized scheduling — owner-as-coordinator (Phase 4b).

With no scheduler, each path's *primary owner of its coordinator key* runs the
commit: it version-fences the path's generation, checks the HRW assignee, gates
on reputation/rate-limit/loss/staleness, and **mints an Ed25519 grant signed
with its own identity** that the path's co-owners verify against the epoch
record. These exercise the owner-side commit + grant machinery directly (the
same style as test_aggregate / test_reputation), not the full worker loop.
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    RateLimiter,
    Reputation,
    coordinator_key,
    grant_signed_by,
    make_epoch_record,
    make_grant,
    make_peer_record,
    owners_for,
    path_primary,
)
from opendipaco.topology import is_private_key


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    # Private embedding/head -> every path has a unique private coordinator key.
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16,
                        embedding="private", head="private")


def _diloco():
    return DiLoCoConfig(inner_steps=4, inner_lr=1e-3)


def _epoch(idents, *, scheduler_ident):
    recs = [make_peer_record(idn, reachability="public",
                             addr=("127.0.0.1", 9000 + i), roles=("owner",))
            for i, idn in enumerate(idents)]
    return make_epoch_record(scheduler_ident, epoch=0, owner_records=recs, k=3)


def _owner(ident, epoch, **kw):
    return ParameterServer(_cfg(), [], _diloco(), host="127.0.0.1", port=0,
                           identity=ident, epoch_record=epoch,
                           schedule_mode="decentralized", **kw)


def _setup(n_owners=3, **owner_kw):
    """A decentralized cluster's epoch + a path and the owner that coordinates
    it (primary of the path's private coordinator key)."""
    sched = PeerIdentity.generate()
    idents = [PeerIdentity.generate() for _ in range(n_owners)]
    epoch = _epoch(idents, scheduler_ident=sched)
    topo = _cfg().build_topology()
    path = topo.path_from_index(0)
    keys = topo.path_module_keys(path)
    prim_id = path_primary(keys, epoch)["peer_id"]
    coord_ident = next(i for i in idents if i.peer_id == prim_id)
    coord = _owner(coord_ident, epoch, **owner_kw)
    return epoch, path, keys, coord, idents


def _base(coord, keys):
    """The (epoch, counter) base versions a worker would report for the keys the
    coordinator holds (everything starts at (0, 0))."""
    return {k: coord._versions[k] for k in keys if k in coord._versions}


# -- the coordinator + version-fence -------------------------------------------


def test_coordinator_mints_grant_and_advances_generation():
    epoch, path, keys, coord, _ = _setup()
    try:
        assert coord._coordinates_locked(path)            # it is the path's coordinator
        ck = coordinator_key(keys)
        assert is_private_key(ck) and ck in coord.owned_keys
        out = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                             "base_versions": _base(coord, keys)}, peer_id="w")
        assert out["accepted"] and out["generation"] == 1
        assert grant_signed_by(out["grant"], coord.peer_id)  # signed by the coordinator
        assert out["push_weight"] > 0
    finally:
        coord.shutdown()


def test_version_fence_drops_a_stale_generation():
    epoch, path, keys, coord, _ = _setup()
    try:
        base = _base(coord, keys)
        first = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                               "base_versions": base}, peer_id="w")
        assert first["accepted"]                           # gen 0 -> advances to 1
        again = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                               "base_versions": base}, peer_id="w2")
        assert not again["accepted"] and again["reason"] == "stale_generation"
        # The current generation (1) is accepted.
        nxt = coord._commit({"path": list(path), "generation": 1, "loss": 1.0,
                             "base_versions": base}, peer_id="w2")
        assert nxt["accepted"] and nxt["generation"] == 2
    finally:
        coord.shutdown()


def test_non_coordinator_owner_refuses_commit():
    epoch, path, keys, coord, idents = _setup()
    other = next(i for i in idents if i.peer_id != coord.peer_id)
    o = _owner(other, epoch)
    try:
        out = o._commit({"path": list(path), "generation": 0, "loss": 1.0,
                         "base_versions": {}}, peer_id="w")
        assert not out["accepted"] and out["reason"] == "not_coordinator"
    finally:
        o.shutdown()


def test_bad_loss_is_refused_and_debits():
    rep = Reputation(floor=0.5, credit=0.1, debit=0.3)
    epoch, path, keys, coord, _ = _setup(reputation=rep)
    try:
        out = coord._commit({"path": list(path), "generation": 0,
                             "loss": float("nan"), "base_versions": _base(coord, keys)},
                            peer_id="bad")
        assert not out["accepted"] and out["reason"] == "bad_loss"
        assert rep.get("bad") < 0.5                         # debited
    finally:
        coord.shutdown()


def test_accepted_commit_credits_reputation():
    rep = Reputation(floor=0.5, credit=0.1, debit=0.3)
    epoch, path, keys, coord, _ = _setup(reputation=rep)
    try:
        coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                       "base_versions": _base(coord, keys)}, peer_id="good")
        assert rep.get("good") > 0.5                        # credited
    finally:
        coord.shutdown()


def test_throttled_committer_gets_backoff_not_a_grant():
    rl = RateLimiter(capacity=1, refill_per_sec=0.0)
    epoch, path, keys, coord, _ = _setup(rate_limiter=rl)
    try:
        base = _base(coord, keys)
        a = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                           "base_versions": base}, peer_id="p")
        assert a["accepted"]                                # spends the one token
        b = coord._commit({"path": list(path), "generation": 1, "loss": 1.0,
                           "base_versions": base}, peer_id="p")
        assert not b["accepted"] and b["reason"] == "throttled"
    finally:
        coord.shutdown()


def test_stale_base_version_is_rejected():
    epoch, path, keys, coord, _ = _setup(staleness_bound=2)
    try:
        ck = coordinator_key(keys)
        # Advance the coordinator key's version far past the worker's base.
        with coord._lock:
            coord._versions[ck] = (0, 10)
        out = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                             "base_versions": {ck: (0, 0)}}, peer_id="w")
        assert not out["accepted"] and out["reason"] == "stale"
    finally:
        coord.shutdown()


# -- HRW-assignee gate ---------------------------------------------------------


def test_only_the_hrw_assignee_may_commit():
    from opendipaco.schedule import rank_workers
    workers = [f"w{i}" for i in range(6)]
    epoch, path, keys, coord, _ = _setup(worker_set=lambda: workers)
    try:
        ranked = rank_workers(path, 0, workers)            # generation 0
        base = _base(coord, keys)
        # A non-assignee is refused...
        bad = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                             "base_versions": base}, peer_id=ranked[-1])
        assert not bad["accepted"] and bad["reason"] == "not_assignee"
        # ...the rank-0 assignee is accepted.
        ok = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                            "base_versions": base}, peer_id=ranked[0])
        assert ok["accepted"]
    finally:
        coord.shutdown()


# -- owner-minted grant on the push path ---------------------------------------


def test_co_owner_accepts_a_grant_signed_by_the_path_primary():
    """A push to a *shared* key's owner is accepted iff the grant was signed by
    the path's primary owner (resolved from the epoch record) — the
    decentralized replacement for the scheduler_pub check."""
    epoch, path, keys, coord, idents = _setup()
    shared = next(k for k in keys if not is_private_key(k))
    sowner_id = owners_for(shared, epoch)[0]["peer_id"]
    sowner = next((_owner(i, epoch) for i in idents if i.peer_id == sowner_id))
    try:
        commit = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                                "base_versions": _base(coord, keys)}, peer_id="w")
        grant = commit["grant"]
        grad = [torch.ones_like(p) for p in sowner.bank[shared].parameters()]
        ack = sowner._push({"grant": grant, "updates": {shared: {"grad": grad}}})
        assert ack["applied"]                              # primary's grant honored
    finally:
        coord.shutdown()
        sowner.shutdown()


def test_forged_and_replayed_grants_are_refused():
    epoch, path, keys, coord, idents = _setup()
    shared = next(k for k in keys if not is_private_key(k))
    sowner_id = owners_for(shared, epoch)[0]["peer_id"]
    sowner = next((_owner(i, epoch) for i in idents if i.peer_id == sowner_id))
    try:
        keyset = list(keys)
        grad = [torch.ones_like(p) for p in sowner.bank[shared].parameters()]
        # Forged: a grant for this path signed by some *other* identity.
        forger = PeerIdentity.generate()
        forged = make_grant(path, keyset, 1.0, "tok-f", identity=forger)
        assert not sowner._push({"grant": forged,
                                 "updates": {shared: {"grad": grad}}})["applied"]
        # Genuine grant from the coordinator: applied once, refused on replay.
        real = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                              "base_versions": _base(coord, keys)}, peer_id="w")["grant"]
        assert sowner._push({"grant": real, "updates": {shared: {"grad": grad}}})["applied"]
        assert not sowner._push({"grant": real,
                                 "updates": {shared: {"grad": grad}}})["applied"]  # replay
    finally:
        coord.shutdown()
        sowner.shutdown()


# -- central mode is untouched -------------------------------------------------


def test_central_mode_owner_rejects_commit_rpc():
    # A central-mode parameter server has no coordinator role; the commit RPC is
    # inert (the scheduler still owns commits). Guards the bit-identical default.
    ps = ParameterServer(_cfg(), sorted(_cfg().build_topology().module_keys()),
                         _diloco(), host="127.0.0.1", port=0)
    try:
        assert ps.schedule_mode == "central"
        out = ps._commit({"path": [0, 0], "generation": 0, "loss": 1.0}, peer_id="w")
        assert not out["accepted"] and out["reason"] == "not_decentralized"
    finally:
        ps.shutdown()
