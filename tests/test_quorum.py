"""Tests for quorum reads + cross-owner digest agreement (Phase 4c).

The Byzantine-owner defense: a content digest that tolerates fp recompute noise
but not real divergence (state_digest), the pure agreement rules (confirm a
version by majority, flag a same-version digest contradiction), the reader-side
quorum fetch, and the owner-side audit — including the end-to-end property that a
single Byzantine owner among k=3 is outvoted on read and flagged for eviction.
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    Reputation,
    confirm_version,
    divergent_peers,
    make_epoch_record,
    make_peer_record,
    read_quorum_versions,
)
from opendipaco.schedule.compress import state_digest


# -- state digest --------------------------------------------------------------


def test_state_digest_is_stable_and_sensitive():
    sd = {"w": torch.randn(8, 8), "b": torch.randn(8), "n": torch.tensor([3])}
    d = state_digest(sd)
    assert d == state_digest(dict(reversed(list(sd.items()))))   # key-order independent
    # A tiny fp perturbation (below the int8 quantization step) keeps the digest...
    sd2 = {"w": sd["w"].clone(), "b": sd["b"].clone(), "n": sd["n"].clone()}
    sd2["w"][0, 0] += 1e-7
    assert state_digest(sd2) == d
    # ...a real change flips it.
    sd3 = {"w": sd["w"] * 2.0, "b": sd["b"].clone(), "n": sd["n"].clone()}
    assert state_digest(sd3) != d
    sd4 = {"w": sd["w"].clone(), "b": sd["b"].clone(), "n": torch.tensor([4])}
    assert state_digest(sd4) != d                                # int buffers hash exactly


# -- pure agreement rules ------------------------------------------------------


def test_confirm_version_takes_the_highest_agreed():
    reports = [((0, 5), "a"), ((0, 5), "a"), ((0, 5), "b")]   # 2 agree on "a"
    assert confirm_version(reports, 2) == ((0, 5), "a")
    # The top version isn't agreed (split); fall back to the highest that is.
    split = [((0, 6), "x"), ((0, 6), "y"), ((0, 5), "z"), ((0, 5), "z")]
    assert confirm_version(split, 2) == ((0, 5), "z")
    # Nothing reaches quorum.
    assert confirm_version([((0, 1), "a"), ((0, 2), "b")], 2) is None
    # Newer epoch outranks a higher counter in an older epoch.
    assert confirm_version([((1, 0), "p"), ((1, 0), "p"), ((0, 9), "q"), ((0, 9), "q")],
                           2) == ((1, 0), "p")


def test_divergent_peers_flags_only_same_version_contradiction():
    confirmed = ((0, 5), "good")
    reports = {
        "honest": ((0, 5), "good"),
        "byz": ((0, 5), "evil"),       # same version, different digest -> flagged
        "behind": ((0, 4), "old"),     # merely lagging -> not flagged
        "ahead": ((0, 6), "future"),   # unconfirmable, not yet wrong -> not flagged
        "down": None,                  # unreachable -> not flagged
    }
    assert divergent_peers(reports, confirmed) == {"byz"}
    assert divergent_peers(reports, None) == set()


def test_read_quorum_versions_skips_unreachable_and_confirms():
    digests = {
        ("h", 1): {"A": [[0, 3], "da"], "B": [[0, 2], "db"]},
        ("h", 2): {"A": [[0, 3], "da"], "B": [[0, 2], "db"]},
        ("h", 3): {"A": [[0, 3], "EVIL"]},  # one Byzantine replica for A
    }

    def rpc(addr, msg):
        if addr == ("h", 9):  # an unreachable replica
            raise ConnectionError
        return {"digests": {k: v for k, v in digests[addr].items() if k in msg["keys"]}}

    out = read_quorum_versions([("h", 1), ("h", 2), ("h", 3), ("h", 9)],
                               ["A", "B"], quorum=2, rpc=rpc)
    assert out["A"] == ((0, 3), "da")   # the honest majority, not "EVIL"
    assert out["B"] == ((0, 2), "db")


# -- owner-side digest RPC + audit ---------------------------------------------


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _owners(n=3, **kw):
    sched = PeerIdentity.generate()
    idents = [PeerIdentity.generate() for _ in range(n)]
    recs = [make_peer_record(i, reachability="public", addr=("127.0.0.1", 9000 + j),
                             roles=("owner",)) for j, i in enumerate(idents)]
    epoch = make_epoch_record(sched, epoch=0, owner_records=recs, k=n)
    return [ParameterServer(_cfg(), [], DiLoCoConfig(inner_steps=4), host="127.0.0.1",
                            port=0, identity=i, epoch_record=epoch,
                            schedule_mode="decentralized", **kw) for i in idents], epoch


def test_digest_rpc_reports_active_keys():
    owners, _ = _owners()
    o = owners[0]
    try:
        rep = o._digests({"keys": None})
        assert rep["type"] == "digest" and rep["digests"]
        for k, (v, d) in rep["digests"].items():
            assert k in o._active and isinstance(d, str) and len(v) == 2
    finally:
        for w in owners:
            w.shutdown()


def test_byzantine_owner_is_outvoted_and_flagged():
    """k=3, all own every key. Corrupt one owner's copy of a shared key: the
    honest two agree, so quorum reads return their digest and the Byzantine
    owner is flagged divergent (toward eviction)."""
    owners, epoch = _owners(reputation=Reputation(floor=0.5, debit=0.3))
    try:
        from opendipaco.topology import is_private_key
        key = next(k for k in owners[0]._active if not is_private_key(k))
        # Byzantine: owner[2] mutates its copy of the key (same version, new bytes).
        byz = owners[2]
        with byz._lock:
            for p in byz.bank[key].parameters():
                p.data.add_(1.0)
        # Assemble each owner's (version, digest) report for the key.
        reports = {}
        for o in owners:
            d = o._digests({"keys": [key]})["digests"][key]
            reports[o.peer_id] = (tuple(d[0]), d[1])
        confirmed = confirm_version(list(reports.values()), 2)
        assert confirmed is not None
        assert reports[owners[0].peer_id][1] == confirmed[1]   # honest digest confirmed
        assert reports[byz.peer_id][1] != confirmed[1]         # Byzantine differs
        # The owner-side audit debits exactly the Byzantine owner.
        flagged = owners[0]._apply_digest_audit({key: reports})
        assert flagged[key] == {byz.peer_id}
        assert owners[0].reputation.get(byz.peer_id) < 0.5
        assert owners[0].reputation.get(owners[1].peer_id) == 0.5  # honest untouched
    finally:
        for w in owners:
            w.shutdown()


def test_audit_does_not_blame_a_lagging_owner():
    owners, _ = _owners(reputation=Reputation(floor=0.5, debit=0.3))
    try:
        from opendipaco.topology import is_private_key
        key = next(k for k in owners[0]._active if not is_private_key(k))
        gd = owners[0]._digests({"keys": [key]})["digests"][key][1]
        # The honest pair is at (0, 1); the third owner is a version behind at
        # (0, 0) -- merely lagging, with no contradiction at the confirmed version.
        reports = {
            owners[0].peer_id: ((0, 1), gd),
            owners[1].peer_id: ((0, 1), gd),
            owners[2].peer_id: ((0, 0), "older"),              # behind, not divergent
        }
        flagged = owners[0]._apply_digest_audit({key: reports})
        assert not flagged                                     # lagging != divergent
        assert owners[0].reputation.get(owners[2].peer_id) == 0.5
    finally:
        for w in owners:
            w.shutdown()
