"""Tests for deterministic epochs + directory gossip (Phase 4d).

With no scheduler the owner-set epoch is a deterministic function of the
self-certifying directory (derive_epoch) rather than a scheduler signature, and
owners learn membership by gossiping the directory among themselves. These cover
the derivation rules (determinism, stable numbering, reputation eviction), the
signer-less verification path, directory import/serve, and the owner deriving +
applying its own epoch — closing the 4c eviction loop.
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    Reputation,
    derive_epoch,
    make_peer_record,
    verify_epoch_record,
)


def _recs(idents, *, roles=("owner",), base=9000):
    return [make_peer_record(i, reachability="public", addr=("127.0.0.1", base + j),
                             roles=roles) for j, i in enumerate(idents)]


# -- deterministic derivation --------------------------------------------------


def test_derive_is_deterministic_across_nodes():
    idents = [PeerIdentity.generate() for _ in range(4)]
    recs = _recs(idents)
    a = derive_epoch(recs, k=3, salt="run")
    b = derive_epoch(list(reversed(recs)), k=3, salt="run")   # different order
    assert a["owners"] == b["owners"] and a["members_sig"] == b["members_sig"]
    assert a["epoch"] == 0 and a["bootstrap"] and a.get("deterministic")
    assert "sig" not in a and "pub" not in a                  # signer-less


def test_epoch_number_is_stable_until_membership_changes():
    idents = [PeerIdentity.generate() for _ in range(3)]
    e0 = derive_epoch(_recs(idents), k=3)
    # Re-deriving the same membership returns the SAME epoch (no churn -> no bump).
    e0b = derive_epoch(_recs(idents), k=3, prev=e0)
    assert e0b["epoch"] == e0["epoch"] == 0
    # A new owner joining bumps the epoch.
    idents2 = idents + [PeerIdentity.generate()]
    e1 = derive_epoch(_recs(idents2), k=3, prev=e0)
    assert e1["epoch"] == 1 and len(e1["owners"]) == 4


def test_derive_evicts_a_reputation_gated_peer():
    idents = [PeerIdentity.generate() for _ in range(3)]
    rep = Reputation(floor=0.5, debit=0.4)
    bad = idents[1].peer_id
    rep.debit(bad)                                            # -> 0.1, below the gate
    e = derive_epoch(_recs(idents), k=3,
                     is_eligible=lambda pid: rep.eligible(pid, 0.25))
    owners = {o["peer_id"] for o in e["owners"]}
    assert bad not in owners                                  # evicted
    assert idents[0].peer_id in owners and idents[2].peer_id in owners


def test_non_eligible_records_are_filtered():
    good = PeerIdentity.generate()
    nat = make_peer_record(PeerIdentity.generate(), reachability="nat")   # no addr/owner
    non_owner = make_peer_record(PeerIdentity.generate(), reachability="public",
                                 addr=("127.0.0.1", 1), roles=("relay",))
    e = derive_epoch([*_recs([good]), nat, non_owner], k=3)
    assert [o["peer_id"] for o in e["owners"]] == [good.peer_id]


# -- signer-less verification --------------------------------------------------


def test_verify_accepts_deterministic_only_when_allowed():
    e = derive_epoch(_recs([PeerIdentity.generate()]), k=2)
    assert not verify_epoch_record(e)                        # no signature -> rejected by default
    assert verify_epoch_record(e, allow_deterministic=True)  # accepted in decentralized mode
    bad = dict(e, owners="nope")
    assert not verify_epoch_record(bad, allow_deterministic=True)  # still structure-checked


# -- owner derive + apply + directory gossip -----------------------------------


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _dec_owner(ident, **kw):
    return ParameterServer(_cfg(), [], DiLoCoConfig(inner_steps=4), host="127.0.0.1",
                           port=0, identity=ident, schedule_mode="decentralized", **kw)


def test_owner_imports_directory_and_derives_its_epoch():
    idents = [PeerIdentity.generate() for _ in range(3)]
    recs = _recs(idents)
    o = _dec_owner(idents[0], k=3)
    o._self_record = recs[0]                                  # what it gossips onward
    try:
        assert o._epoch is None and not o.owned_keys          # starts bare
        n = o.import_directory(recs)
        assert n >= 2                                          # co-owners imported (self served separately)
        rec = o.derive_and_apply_epoch()
        assert rec is not None and o._epoch is not None
        assert len(o._epoch["owners"]) == 3
        # It now owns (and serves) the keys HRW places on it, at seeded (0, 0).
        assert o.owned_keys and o.owned_keys <= o._all_keys
        assert o.owned_keys <= o._active                      # bootstrap epoch -> active
    finally:
        o.shutdown()


def test_directory_rpc_round_trips_between_owners():
    idents = [PeerIdentity.generate() for _ in range(2)]
    recs = _recs(idents)
    a, b = _dec_owner(idents[0]), _dec_owner(idents[1])
    a._self_record, b._self_record = recs[0], recs[1]
    try:
        # b learns about a by importing a's served directory.
        served = a._directory_rpc({})["records"]
        assert any(r["peer_id"] == idents[0].peer_id for r in served)   # a includes itself
        b.import_directory(served)
        assert idents[0].peer_id in b._directory
    finally:
        a.shutdown()
        b.shutdown()


def test_owner_eviction_loop_excludes_a_divergent_owner():
    """End-to-end of 4c+4d: a co-owner debited for digest divergence is dropped
    from the epoch the owner re-derives (the eviction the 4c audit feeds)."""
    idents = [PeerIdentity.generate() for _ in range(3)]
    recs = _recs(idents)
    rep = Reputation(floor=0.5, debit=0.4)
    o = _dec_owner(idents[0], k=3, reputation=rep, min_owner_reputation=0.25)
    o._self_record = recs[0]
    try:
        o.import_directory(recs)
        e0 = o.derive_and_apply_epoch()
        assert len(e0["owners"]) == 3
        # The 4c audit debits a divergent co-owner...
        o._apply_digest_audit({"k": {idents[2].peer_id: ((0, 1), "evil"),
                                     idents[0].peer_id: ((0, 1), "good"),
                                     idents[1].peer_id: ((0, 1), "good")}})
        assert rep.get(idents[2].peer_id) < 0.25
        # ...so the next derived epoch evicts it.
        e1 = o.derive_and_apply_epoch()
        owners = {ow["peer_id"] for ow in e1["owners"]}
        assert idents[2].peer_id not in owners and e1["epoch"] > e0["epoch"]
    finally:
        o.shutdown()


def test_central_owner_ignores_decentralized_rpcs():
    ps = ParameterServer(_cfg(), sorted(_cfg().build_topology().module_keys()),
                         DiLoCoConfig(inner_steps=4), host="127.0.0.1", port=0)
    try:
        assert ps.derive_and_apply_epoch() is None            # no-op in central mode
        torch.manual_seed(0)
    finally:
        ps.shutdown()
