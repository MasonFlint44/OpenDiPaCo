"""Tests for dynamic module ownership (internet-scale plan, Phase 2a).

HRW placement (determinism + the minimal-disruption property that justifies
rendezvous hashing), signed epoch records, Ed25519-signed commit grants beside
HMAC, the scheduler's epoch RPC, and the tracker's pinned epoch cache.
"""

import threading

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    Scheduler,
    Tracker,
    assign_shards,
    get_epoch,
    make_epoch_record,
    make_grant,
    make_peer_record,
    owner_eligible,
    owners_for,
    put_epoch,
    rank_owners,
    run_sharded_worker,
    verify_epoch_record,
    verify_grant,
)
from opendipaco.schedule.ownership import epoch_newer
from opendipaco.schedule.wire import decode, encode
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


def _owners(n):
    return [{"peer_id": f"peer-{i:02d}", "addr": ["host", 9000 + i]} for i in range(n)]


def _owner_record(identity, port=9100):
    return make_peer_record(identity, reachability="public",
                            addr=("127.0.0.1", port), roles=("owner",))


# -- HRW placement -------------------------------------------------------------


def test_hrw_deterministic_and_order_independent():
    """The ranking is a pure function of (salt, key, peer ids) — stable across
    calls and independent of the input list's order."""
    owners = _owners(6)
    keys = [f"L{i}E{j}" for i in range(4) for j in range(4)]
    for key in keys:
        a = rank_owners(key, owners)
        b = rank_owners(key, list(reversed(owners)))
        assert [o["peer_id"] for o in a] == [o["peer_id"] for o in b]
    # Different keys land on different primaries (the load actually spreads).
    primaries = {rank_owners(k, owners)[0]["peer_id"] for k in keys}
    assert len(primaries) > 1
    # The salt re-randomizes placement.
    salted = {k: [o["peer_id"] for o in rank_owners(k, owners, salt="other")] for k in keys}
    assert any(salted[k] != [o["peer_id"] for o in rank_owners(k, owners)] for k in keys)


def test_hrw_minimal_disruption_on_owner_loss():
    """Removing one owner only changes the sets that contained it: each such key
    keeps its surviving replicas in the same relative order and gains exactly
    one new member; every other key's set is untouched."""
    owners = _owners(8)
    keys = [f"B{i}.p{j}" for i in range(10) for j in range(4)]
    epoch = {"owners": owners, "k": 3, "salt": ""}
    before = {k: [o["peer_id"] for o in owners_for(k, epoch)] for k in keys}

    lost = owners[3]["peer_id"]
    survivors = [o for o in owners if o["peer_id"] != lost]
    epoch2 = {"owners": survivors, "k": 3, "salt": ""}
    touched = 0
    for k in keys:
        after = [o["peer_id"] for o in owners_for(k, epoch2)]
        if lost not in before[k]:
            assert after == before[k]  # unaffected keys keep their exact set
        else:
            touched += 1
            kept = [p for p in before[k] if p != lost]
            assert [p for p in after if p in kept] == kept  # succession order kept
            assert len(set(after) - set(before[k])) == 1     # exactly one newcomer
    assert touched > 0  # the property was actually exercised


def test_hrw_join_steals_only_a_fraction():
    """An owner joining changes some sets but never evicts more than one member
    per key (it can only displace the lowest-ranked replica)."""
    owners = _owners(6)
    keys = [f"L{i}E{j}" for i in range(6) for j in range(6)]
    epoch = {"owners": owners, "k": 3, "salt": ""}
    before = {k: [o["peer_id"] for o in owners_for(k, epoch)] for k in keys}
    epoch2 = {"owners": owners + [{"peer_id": "peer-new", "addr": ["host", 9999]}],
              "k": 3, "salt": ""}
    changed = 0
    for k in keys:
        after = [o["peer_id"] for o in owners_for(k, epoch2)]
        evicted = set(before[k]) - set(after)
        assert len(evicted) <= 1
        if after != before[k]:
            changed += 1
            assert "peer-new" in after
    assert 0 < changed < len(keys)  # it joined some sets, not all


# -- eligibility + epoch records -------------------------------------------------


def test_owner_eligibility_predicate():
    ident = PeerIdentity.generate()
    assert owner_eligible(_owner_record(ident))
    # NAT peers, non-owners, and non-peer records are all ineligible.
    assert not owner_eligible(make_peer_record(ident, reachability="nat", roles=("owner",)))
    assert not owner_eligible(make_peer_record(
        ident, reachability="public", addr=("127.0.0.1", 9100), roles=("relay",)))
    assert not owner_eligible({"kind": "epoch", "peer_id": "x", "addr": ["h", 1]})


def test_epoch_record_roundtrip_tamper_and_signer_pin():
    sched_id, other = PeerIdentity.generate(), PeerIdentity.generate()
    owner_ids = [PeerIdentity.generate() for _ in range(3)]
    record = make_epoch_record(sched_id, epoch=0,
                               owner_records=[_owner_record(i) for i in owner_ids])
    assert verify_epoch_record(record)
    assert verify_epoch_record(record, signer_pub=sched_id.public_key_hex)
    # Pinning to a different signer fails even though the signature is valid.
    assert not verify_epoch_record(record, signer_pub=other.public_key_hex)
    # Any tampering with the signed body invalidates it.
    tampered = dict(record, owners=record["owners"][:-1])
    assert not verify_epoch_record(tampered)
    tampered = dict(record, k=1)
    assert not verify_epoch_record(tampered)
    # The record survives the wire codec byte-identically for verification.
    assert verify_epoch_record(decode(encode(record)), signer_pub=sched_id.public_key_hex)


def test_epoch_record_rejects_ineligible_and_duplicates():
    sched_id = PeerIdentity.generate()
    nat = make_peer_record(PeerIdentity.generate(), reachability="nat", roles=("owner",))
    try:
        make_epoch_record(sched_id, epoch=0, owner_records=[nat])
        raise AssertionError("ineligible record must be refused at build time")
    except ValueError:
        pass
    # A forged record with a duplicated peer id fails verification (a dup would
    # double that owner's HRW odds).
    o = _owner_record(PeerIdentity.generate())
    rec = make_epoch_record(sched_id, epoch=0, owner_records=[o])
    dup = {k: v for k, v in rec.items() if k not in ("pub", "sig", "peer_id")}
    dup["owners"] = rec["owners"] * 2
    from opendipaco.schedule import sign_record
    assert not verify_epoch_record(sign_record(sched_id, dup))


def test_epoch_ordering():
    a = {"epoch": 1, "issued_at": 100.0}
    assert epoch_newer(a, None)
    assert epoch_newer({"epoch": 2, "issued_at": 50.0}, a)       # higher epoch wins
    assert epoch_newer({"epoch": 1, "issued_at": 101.0}, a)      # re-issue supersedes
    assert not epoch_newer({"epoch": 1, "issued_at": 100.0}, a)  # equal is not newer
    assert not epoch_newer({"epoch": 0, "issued_at": 999.0}, a)


# -- Ed25519 grants ---------------------------------------------------------------


def test_signed_grant_accept_forge_and_downgrade():
    cfg = _cfg()
    sched_id, worker_id_ = PeerIdentity.generate(), PeerIdentity.generate()
    path = cfg.build_topology().path_from_index(0)
    keys = cfg.build_topology().path_module_keys(path)
    pub = sched_id.public_key_hex

    good = make_grant(path, keys, 0.5, "tok-1", identity=sched_id)
    assert verify_grant(good, None, scheduler_pub=pub)
    # ...including after a wire round trip (canonical JSON is re-derivable).
    assert verify_grant(decode(encode(good)), None, scheduler_pub=pub)

    # A grant signed by any other identity (e.g. the worker's own) is refused.
    forged = make_grant(path, keys, 100.0, "tok-2", identity=worker_id_)
    assert not verify_grant(forged, None, scheduler_pub=pub)
    # Tampering with the signed weight invalidates it.
    assert not verify_grant(dict(good, weight=100.0), None, scheduler_pub=pub)
    # No HMAC/unsigned downgrade when a signature is required.
    assert not verify_grant(make_grant(path, keys, 0.5, "tok-3", grant_key="k"),
                            None, scheduler_pub=pub)
    assert not verify_grant(make_grant(path, keys, 0.5, "tok-4"), None, scheduler_pub=pub)
    # The HMAC path is unchanged by the new mode.
    mac = make_grant(path, keys, 0.5, "tok-5", grant_key="shared")
    assert verify_grant(mac, "shared") and not verify_grant(mac, "wrong")


def test_ps_push_requires_scheduler_signature():
    """A parameter server with ``scheduler_pub=`` applies only scheduler-signed
    grants: worker-signed, HMAC, and replayed grants are refused."""
    cfg = _cfg()
    sched_id, worker_id_ = PeerIdentity.generate(), PeerIdentity.generate()
    keys = sorted(cfg.build_topology().module_keys())
    ps = ParameterServer(cfg, keys, _diloco(), host="127.0.0.1", port=0,
                         scheduler_pub=sched_id.public_key_hex)
    try:
        k0 = next(k for k in keys if not is_private_key(k))
        grad = {"grad": [torch.ones_like(p) for p in ps.bank[k0].parameters()]}
        path = cfg.build_topology().path_from_index(0)

        v0 = ps._versions[k0]
        forged = make_grant(path, keys, 1.0, "tok-f", identity=worker_id_)
        assert ps._push({"grant": forged, "updates": {k0: grad}})["applied"] is False
        hmac_grant = make_grant(path, keys, 1.0, "tok-h", grant_key="anything")
        assert ps._push({"grant": hmac_grant, "updates": {k0: grad}})["applied"] is False
        assert ps._versions[k0] == v0  # nothing was applied

        good = make_grant(path, keys, 1.0, "tok-g", identity=sched_id)
        assert ps._push({"grant": good, "updates": {k0: grad}})["applied"] is True
        assert ps._versions[k0] == (0, v0[1] + 1)  # (epoch, counter) pairs (2b)
        # Single-use: replaying the signed grant is refused like an HMAC one.
        assert ps._push({"grant": good, "updates": {k0: grad}})["applied"] is False
        assert ps._versions[k0] == (0, v0[1] + 1)
    finally:
        ps.shutdown()


def test_sharded_end_to_end_with_signed_grants():
    """A full scheduler + 2 PS + worker run where grants are Ed25519-signed and
    no ``grant_key`` exists anywhere: training reaches the update target."""
    cfg, dl = _cfg(), _diloco()
    sched_id = PeerIdentity.generate()
    ks = assign_shards(cfg.build_topology().module_keys(), 2)
    shards = [[k for k, s in ks.items() if s == i] for i in range(2)]
    pss = [ParameterServer(cfg, sk, dl, host="127.0.0.1", port=0,
                           scheduler_pub=sched_id.public_key_hex) for sk in shards]
    for ps in pss:
        ps.start()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", ps.port) for ps in pss], dl,
                      batch_size=BATCH, host="127.0.0.1", port=0, identity=sched_id)
    sched.start()
    w = threading.Thread(target=run_sharded_worker, args=(cfg, dl, ("127.0.0.1", sched.port)),
                         kwargs=dict(seed=0, heartbeat_interval=1.0), daemon=True)
    w.start()
    try:
        completed = sched.fit(num_generations=2, total_generations=2)
        assert sum(completed.values()) >= sched._target
        assert any(v > (0, 0) for ps in pss for v in ps._versions.values())  # pushes landed
    finally:
        sched.shutdown()
        for ps in pss:
            ps.shutdown()
        w.join(timeout=10)


# -- scheduler epoch RPC + tracker cache ------------------------------------------


def test_scheduler_publishes_and_serves_epochs():
    cfg = _cfg()
    sched_id = PeerIdentity.generate()
    owner_recs = [_owner_record(PeerIdentity.generate(), port=9200 + i) for i in range(3)]
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], _diloco(),
                      batch_size=BATCH, host="127.0.0.1", port=0, identity=sched_id)
    try:
        assert sched._handle({"type": "epoch"}, 0) == {"type": "epoch", "record": None}
        r0 = sched.publish_epoch(owner_recs)
        assert verify_epoch_record(r0, signer_pub=sched_id.public_key_hex)
        assert r0["epoch"] == 0 and len(r0["owners"]) == 3
        served = sched._handle({"type": "epoch"}, 0)["record"]
        assert served == r0
        # Re-publication bumps the epoch number monotonically.
        r1 = sched.publish_epoch(owner_recs[:2])
        assert r1["epoch"] == 1 and len(r1["owners"]) == 2
        # The mapping is derivable from the served record alone.
        assert len(owners_for("L0E0", r1)) == 2  # k=3 capped by 2 live owners
    finally:
        sched.shutdown()

    # Without an identity there is nothing to sign with.
    plain = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], _diloco(),
                      batch_size=BATCH, host="127.0.0.1", port=0)
    try:
        plain.publish_epoch(owner_recs)
        raise AssertionError("publish_epoch without identity must raise")
    except RuntimeError:
        pass
    finally:
        plain.shutdown()


def test_tracker_epoch_cache_pins_signer_and_orders_epochs():
    sched_id, rogue = PeerIdentity.generate(), PeerIdentity.generate()
    owner_recs = [_owner_record(PeerIdentity.generate())]
    r0 = make_epoch_record(sched_id, epoch=0, owner_records=owner_recs)
    r1 = make_epoch_record(sched_id, epoch=1, owner_records=owner_recs)
    rogue_rec = make_epoch_record(rogue, epoch=2, owner_records=owner_recs)

    t = Tracker(host="127.0.0.1", port=0, open_enrollment=True)
    t.start()
    try:
        addr = ("127.0.0.1", t.port)
        assert put_epoch(addr, r1)["type"] == "epoch_cached"
        # Stale (lower epoch) and unsigned/garbage puts are refused.
        assert put_epoch(addr, r0)["type"] == "refused"
        assert put_epoch(addr, {"kind": "epoch", "epoch": 3})["type"] == "refused"
        # The first valid put pinned the signer: another identity can't displace it.
        assert put_epoch(addr, rogue_rec)["reason"] == "wrong signer"
        got = get_epoch(addr, signer_pub=sched_id.public_key_hex)
        assert got is not None and got["epoch"] == 1
        # A client pinning a different signer treats the cache as empty.
        assert get_epoch(addr, signer_pub=rogue.public_key_hex) is None
    finally:
        t.shutdown()

    # An explicitly configured signer refuses even the *first* put from others.
    t2 = Tracker(host="127.0.0.1", port=0, open_enrollment=True,
                 epoch_signer=sched_id.public_key_hex)
    t2.start()
    try:
        addr2 = ("127.0.0.1", t2.port)
        assert put_epoch(addr2, rogue_rec)["reason"] == "wrong signer"
        assert put_epoch(addr2, r0)["type"] == "epoch_cached"
    finally:
        t2.shutdown()
