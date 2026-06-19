"""Tests for redundant execution (internet-scale plan, Phase 3c).

The premise the whole slice rests on — two workers training the *same path from
the same pinned base* produce the same digest — plus the scheduler's audit
arbitration (agreement credits, the odd one out is debited, a 2-way split is
inconclusive, timeouts don't punish) and the check-offer logic (distinct
workers, oversupply only).
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import ParameterServer, Reputation, Scheduler, make_grant
from opendipaco.schedule.compress import pseudograd_digest
from opendipaco.schedule.distributed import _build_worker_engine
from opendipaco.schedule.scheduler import AsyncScheduler
from opendipaco.topology import is_private_key

BATCH = 8


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _docs():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(8)]


def _corpus(cfg):
    assign = torch.tensor([i % cfg.num_paths for i in range(32)])
    return ShardedCorpus.from_assignments(_docs(), assign, cfg.num_paths, cfg.sequence_length)


# -- digest reproducibility (the version-pinning premise) ----------------------


def test_same_base_same_digest_perturbed_base_differs():
    """Two cold workers training the same (path, gen, shard, seed) from an
    identical base produce identical pseudo-gradients -> identical digest. A
    different base (one outer step elsewhere) lands on a different digest. This
    is exactly why a checker must pin the primary's base."""
    cfg, dl = _cfg(), DiLoCoConfig(inner_steps=4, inner_lr=1e-3)
    corpus = _corpus(cfg)
    path = cfg.build_topology().path_from_index(0)
    shard = corpus.shard(0)

    def digest_from(seeded_perturb):
        engine = _build_worker_engine(cfg, dl, "cpu", 0)
        if seeded_perturb:  # nudge the base weights (a different version)
            with torch.no_grad():
                for m in engine.bank.values():
                    for p in m.parameters():
                        p.add_(0.05)
        worker = AsyncScheduler(engine, num_workers=1)
        worker.seed = 0
        return pseudograd_digest(worker._train_path(path, shard, BATCH, 0).shared_delta)

    a, b = digest_from(False), digest_from(False)
    assert a == b                    # identical base -> identical digest
    assert digest_from(True) != a    # perturbed base -> different digest


# -- owner version history + pinned fetch --------------------------------------


def test_owner_retains_and_serves_pinned_base():
    """An owner with version_history keeps recent states so a checker can fetch
    the *exact base* a primary trained against, even after the module advanced;
    an aged-out pin returns 'missing' so the audit aborts rather than compares
    against the wrong base."""
    cfg = _cfg()
    keys = sorted(cfg.build_topology().module_keys())
    ps = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=4), host="127.0.0.1",
                         port=0, version_history=3)
    try:
        k = next(x for x in keys if not is_private_key(x))
        v0 = ps._versions[k]
        base0 = {n: p.detach().clone() for n, p in ps.bank[k].named_parameters()}
        path = cfg.build_topology().path_from_index(0)
        for tok in ("a", "b"):       # advance the module twice
            g = [torch.ones_like(p) for p in ps.bank[k].parameters()]
            ps._push({"grant": make_grant(path, [k], 1.0, tok), "updates": {k: {"grad": g}}})
        assert ps._versions[k] == (0, 2)

        pinned = ps._fetch({"type": "fetch", "keys": [k], "pin": {k: list(v0)}})
        got = pinned["weights"][k]
        assert tuple(pinned["versions"][k]) == v0
        assert all(torch.equal(base0[n], got[n]) for n in base0)  # exact old base
        # A version never retained -> missing -> the auditor abstains.
        assert k in ps._fetch({"type": "fetch", "keys": [k], "pin": {k: [9, 9]}}).get("missing", [])
    finally:
        ps.shutdown()


# -- scheduler audit arbitration -----------------------------------------------


def _serving(**kw):
    cfg = _cfg()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], DiLoCoConfig(inner_steps=4),
                      batch_size=BATCH, host="127.0.0.1", port=0,
                      reputation=Reputation(floor=0.5, credit=0.1, debit=0.3,
                                            decay_halflife=0.0), **kw)
    with sched._lock:
        sched._serving = True
        sched._target = 10 ** 9
        sched._completed = {p: 0 for p in sched.paths}
    return sched


def _audit_primary(sched, peer, digest, base=None):
    """Lease an audited primary task and commit it with a digest. Returns the
    audit key (path, gen)."""
    task = sched._next_task({"worker_id": peer}, peer_id=peer)
    assert task.get("audit"), "rate=1.0 should audit every primary"
    sched._commit({"path": task["path"], "lease": task["lease"],
                   "base": base or {"L0E0": [0, 1]}, "digest": digest}, peer_id=peer)
    return (tuple(task["path"]), task["gen_id"])


def _check(sched, key, peer, digest, probe=None):
    """Reserve a checker slot (as _next_task's oversupply branch does), then
    commit its result echoing the issued lease -- only an assigned checker's
    vote counts. ``probe`` is an optional ``(before, after)`` clean-probe loss
    pair the checker reports for the W8 data-poisoning screen."""
    path, gen = key
    with sched._lock:
        _, _, _, lease, *_ = sched._reserve_check_locked(key, peer)
    msg = {"check_only": True, "path": list(path), "gen_id": gen,
           "digest": digest, "lease": lease, "worker_id": peer}
    if probe is not None:
        msg["probe_before"], msg["probe_after"] = probe
    sched._commit_check(msg, peer_id=peer)


def test_unassigned_check_commit_is_ignored():
    """A peer that wasn't reserved as a checker can't manufacture the verdict by
    spamming check_only commits (Codex P1): unassigned/forged-lease results
    don't count toward the audit."""
    sched = _serving(redundancy=3, redundancy_rate=1.0)
    try:
        key = _audit_primary(sched, "P", "X")
        # An unassigned peer with a guessed lease -- ignored, audit stays open.
        for i in range(5):
            sched._commit_check({"check_only": True, "path": list(key[0]),
                                 "gen_id": key[1], "digest": "FORGED",
                                 "lease": f"guess{i}", "worker_id": "ATTACKER"},
                                peer_id="ATTACKER")
        assert key in sched._audits and not sched._audits[key]["checks"]
        # A reserved checker echoing the wrong lease also doesn't count.
        with sched._lock:
            _, _, _, real_lease, *_ = sched._reserve_check_locked(key, "C1")
        sched._commit_check({"check_only": True, "path": list(key[0]), "gen_id": key[1],
                             "digest": "X", "lease": "wrong", "worker_id": "C1"},
                            peer_id="C1")
        assert not sched._audits[key]["checks"]
        # The same checker can't double-vote with its real lease either.
        for _ in range(3):
            sched._commit_check({"check_only": True, "path": list(key[0]), "gen_id": key[1],
                                 "digest": "X", "lease": real_lease, "worker_id": "C1"},
                                peer_id="C1")
        assert len(sched._audits[key]["checks"]) == 1
    finally:
        sched.shutdown()


def test_audit_all_agree_credits_everyone():
    sched = _serving(redundancy=3, redundancy_rate=1.0)
    try:
        key = _audit_primary(sched, "P", "X")
        _check(sched, key, "C1", "X")
        _check(sched, key, "C2", "X")              # target reached -> resolves
        assert key not in sched._audits
        assert all(sched.reputation.get(p) > 0.5 for p in ("P", "C1", "C2"))
    finally:
        sched.shutdown()


def test_audit_debits_the_odd_one_out():
    sched = _serving(redundancy=3, redundancy_rate=1.0)
    try:
        # The primary is the minority (fabricated update); two checkers agree.
        key = _audit_primary(sched, "BADPRIMARY", "FABRICATED")
        _check(sched, key, "C1", "X")
        _check(sched, key, "C2", "X")
        assert sched.reputation.get("BADPRIMARY") < 0.5   # debited
        assert sched.reputation.get("C1") > 0.5 and sched.reputation.get("C2") > 0.5
    finally:
        sched.shutdown()

    sched = _serving(redundancy=3, redundancy_rate=1.0)
    try:
        # A lying checker is the minority; primary + the other checker agree.
        key = _audit_primary(sched, "P", "X")
        _check(sched, key, "HONEST", "X")
        _check(sched, key, "LIAR", "Y")
        assert sched.reputation.get("LIAR") < 0.5
        assert sched.reputation.get("P") > 0.5 and sched.reputation.get("HONEST") > 0.5
    finally:
        sched.shutdown()


# -- W8 trusted-probe data-poisoning screen ------------------------------------


def test_probe_screen_flags_poisoned_contribution():
    """The checkers agree on the digest (they reproduce the same poisoned data),
    but a quorum reports the update RAISED clean-probe loss -> the audit flags it
    and (opt-in) debits the primary."""
    sched = _serving(redundancy=3, redundancy_rate=1.0, probe_quorum=2,
                     probe=torch.randint(0, 48, (4, 16)), probe_debit=True)
    try:
        key = _audit_primary(sched, "P", "X")
        _check(sched, key, "C1", "X", probe=(1.0, 2.0))   # +1.0 loss -> harmful
        _check(sched, key, "C2", "X", probe=(1.0, 2.0))   # quorum of 2 harmful
        assert sched.metrics.poison_flagged == 1
        assert sched.reputation.get("P") < 0.5            # debited (probe_debit on)
    finally:
        sched.shutdown()


def test_probe_screen_passes_honest_contribution():
    """An update that doesn't raise clean-probe loss is not flagged, even though
    the digests agree -- the screen is orthogonal to the digest tally."""
    sched = _serving(redundancy=3, redundancy_rate=1.0, probe_quorum=2,
                     probe=torch.randint(0, 48, (4, 16)), probe_debit=True)
    try:
        key = _audit_primary(sched, "P", "X")
        _check(sched, key, "C1", "X", probe=(1.0, 1.0))   # unchanged
        _check(sched, key, "C2", "X", probe=(1.0, 0.9))   # improved
        assert sched.metrics.poison_flagged == 0
        assert sched.reputation.get("P") > 0.5            # not blamed
    finally:
        sched.shutdown()


def test_probe_screen_needs_a_quorum():
    """One harmful report (below quorum) doesn't flag -- a single lying/faulty
    checker can't manufacture a poisoning verdict."""
    sched = _serving(redundancy=3, redundancy_rate=1.0, probe_quorum=2,
                     probe=torch.randint(0, 48, (4, 16)), probe_debit=True)
    try:
        key = _audit_primary(sched, "P", "X")
        _check(sched, key, "C1", "X", probe=(1.0, 9.0))   # one harmful
        _check(sched, key, "C2", "X", probe=(1.0, 1.0))   # one clean -> 1 < quorum 2
        assert sched.metrics.poison_flagged == 0
        assert sched.reputation.get("P") > 0.5
    finally:
        sched.shutdown()


def test_probe_screen_off_by_default_is_inert():
    """probe_quorum 0 (default) -> the screen never fires, even on harmful reports
    (byte-identical to the pre-W8 audit)."""
    sched = _serving(redundancy=3, redundancy_rate=1.0)   # no probe, quorum 0
    try:
        key = _audit_primary(sched, "P", "X")
        _check(sched, key, "C1", "X", probe=(1.0, 5.0))
        _check(sched, key, "C2", "X", probe=(1.0, 5.0))
        assert sched.metrics.poison_flagged == 0
        assert sched.reputation.get("P") > 0.5            # only the digest tally ran
    finally:
        sched.shutdown()


def test_probe_config_is_validated():
    """probe_quorum can't exceed the available checkers (else the screen silently
    never fires), and debiting needs a >=2 corroboration floor (else one Byzantine
    checker punishes an honest primary)."""
    import pytest
    cfg = _cfg()
    common = dict(batch_size=BATCH, host="127.0.0.1", port=0)
    with pytest.raises(ValueError, match="exceeds the available checkers"):
        Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], DiLoCoConfig(inner_steps=4),
                  redundancy=3, probe_quorum=3, **common)        # only 2 checkers
    with pytest.raises(ValueError, match="probe_debit requires probe_quorum >= 2"):
        Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], DiLoCoConfig(inner_steps=4),
                  redundancy=2, probe_quorum=1, probe_debit=True, **common)


def test_probe_screen_skips_uncommitted_primary():
    """A contribution the primary never committed (audit timed out before its
    commit) is neither flagged nor debited, even with harmful checker probes --
    the update never landed."""
    sched = _serving(redundancy=3, redundancy_rate=0.0, probe_quorum=2,
                     probe=torch.randint(0, 48, (4, 16)), probe_debit=True)
    try:
        key = (list(sched.paths)[0], 0)
        with sched._lock:
            sched._audits[key] = {"target": 2, "base": {"L0E0": [0, 1]},
                                  "primary_digest": None, "primary_peer": None,
                                  "checks": [], "members": {"P"}, "checked": set(),
                                  "check_leases": {},
                                  "probes": [("C1", 1.0, 9.0), ("C2", 1.0, 9.0)],
                                  "deadline": 0.0}
            sched._resolve_audit_locked(key)
        assert sched.metrics.poison_flagged == 0
    finally:
        sched.shutdown()


def test_check_task_carries_the_probe():
    """When a probe is configured, the check task ships it so the checker can
    screen (only checks carry it -- the primary's own probe would be untrusted)."""
    probe = torch.randint(0, 48, (4, 16))
    sched = _serving(redundancy=3, redundancy_rate=0.0, probe=probe)
    try:
        paths = list(sched.paths)
        for i, _ in enumerate(paths):                     # exhaust primary work
            sched._next_task({"worker_id": f"w{i}"}, peer_id=f"w{i}")
        import time as _t
        key = (paths[0], 0)
        with sched._lock:
            sched._audits[key] = {"target": 2, "base": {"L0E0": [0, 1]},
                                  "primary_digest": "X", "primary_peer": "P",
                                  "checks": [], "members": {"P"}, "probes": [],
                                  "deadline": _t.monotonic() + 100}
        chk = sched._next_task({"worker_id": "C1"}, peer_id="C1")
        assert chk.get("check_only")
        assert "probe" in chk and torch.equal(chk["probe"], probe) and chk["probe_batch"] == 8
        # A normal (non-check) task never carries the probe.
        sched2 = _serving(redundancy=3, redundancy_rate=0.0, probe=probe)
        try:
            t = sched2._next_task({"worker_id": "w"}, peer_id="w")
            assert not t.get("check_only") and "probe" not in t
        finally:
            sched2.shutdown()
    finally:
        sched.shutdown()


def test_two_way_split_is_inconclusive():
    sched = _serving(redundancy=2, redundancy_rate=1.0)   # target = 1 checker
    try:
        key = _audit_primary(sched, "P", "X")
        _check(sched, key, "C1", "Y")              # X vs Y, no majority
        assert key not in sched._audits             # resolved (target met)...
        # P keeps only its commit-accept credit (0.5 + 0.1); the inconclusive
        # audit neither credits (would be 0.7) nor debits it. The checker, which
        # has no commit credit, stays exactly at the floor.
        assert sched.reputation.get("P") == 0.6
        assert sched.reputation.get("C1") == 0.5
    finally:
        sched.shutdown()


def test_timeout_resolves_without_punishing():
    sched = _serving(redundancy=3, redundancy_rate=1.0, audit_timeout=0.0)
    try:
        key = _audit_primary(sched, "P", "X")       # deadline = now (audit_timeout 0)
        assert key in sched._audits
        with sched._lock:
            sched._reclaim_inflight_locked()        # sweep resolves the timed-out audit
        assert key not in sched._audits
        # Only the commit-accept credit; the lone-digest audit is inconclusive.
        assert sched.reputation.get("P") == 0.6
    finally:
        sched.shutdown()


# -- check offers --------------------------------------------------------------


def test_check_offered_only_to_distinct_workers_on_oversupply():
    sched = _serving(redundancy=3, redundancy_rate=0.0)   # no auto-audit; we inject one
    try:
        # Lease every path so there is no primary work left.
        paths = list(sched.paths)
        for i, _ in enumerate(paths):
            t = sched._next_task({"worker_id": f"w{i}"}, peer_id=f"w{i}")
            assert t["type"] == "task" and not t.get("check_only")
        # Inject an open audit (a primary already committed elsewhere).
        import time as _t
        key = (paths[0], 0)
        with sched._lock:
            sched._audits[key] = {"target": 2, "base": {"L0E0": [0, 1]},
                                  "primary_digest": "X", "primary_peer": "P",
                                  "checks": [], "members": {"P"},
                                  "deadline": _t.monotonic() + 100}
        # A surplus, distinct worker gets a check task (oversupply absorbed, §1.9).
        chk = sched._next_task({"worker_id": "C1"}, peer_id="C1")
        assert chk.get("check_only") and tuple(chk["path"]) == paths[0]
        assert chk["base"] == {"L0E0": [0, 1]}
        # The primary can't be asked to check its own work.
        assert sched._next_task({"worker_id": "P"}, peer_id="P")["type"] == "idle"
    finally:
        sched.shutdown()


# -- private-module proposal gating (D5/3a) ------------------------------------


def _priv_cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16,
                        embedding="private")


def _priv_ps(**kw):
    cfg = _priv_cfg()
    # grant_key set so proposals must carry a *real* scheduler grant (forged /
    # replayed ones are refused) -- the gate that binds proposals to assignments.
    kw.setdefault("grant_key", "sched-secret")
    return ParameterServer(cfg, sorted(cfg.build_topology().module_keys()),
                           DiLoCoConfig(inner_steps=4), host="127.0.0.1", port=0, **kw)


def _private_state(ps, key, fill):
    return {n: torch.full_like(p, fill) for n, p in ps.bank[key].state_dict().items()}


def _grant(path, k, token):
    return make_grant(path, [k], 0.0, token, grant_key="sched-secret")


def test_proposal_applies_only_on_quorum_agreement():
    """Under the proposal policy a private push is inert until ``private_quorum``
    distinct scheduler-issued grants agree on the exact state; then it applies."""
    ps = _priv_ps(private_policy="proposal", private_quorum=2)
    try:
        k = next(x for x in ps.owned_keys if is_private_key(x))
        path = ps._topology.paths_through_module(k)[0]
        v0 = ps._versions[k]
        good = _private_state(ps, k, 1.0)

        # The primary's granted push -> held as one vote, nothing applied yet.
        ps._push({"grant": _grant(path, k, "primary"), "private": {k: good}}, peer_id="A")
        assert ps._versions[k] == v0
        # A checker's proposal with a *distinct* grant agreeing -> quorum -> applied.
        ps._private_proposal({"private": {k: good}, "grant": _grant(path, k, "check1")},
                             peer_id="B")
        assert ps._versions[k] > v0
        assert all(torch.equal(p.detach(), torch.full_like(p, 1.0))
                   for p in ps.bank[k].parameters())
    finally:
        ps.shutdown()


def test_lone_or_forged_proposal_never_applies():
    """One assigned actor holds one grant; replaying it or forging a second
    can't reach quorum -- a malicious owner-path worker can at most stall its
    private module, never poison it."""
    ps = _priv_ps(private_policy="proposal", private_quorum=2)
    try:
        k = next(x for x in ps.owned_keys if is_private_key(x))
        path = ps._topology.paths_through_module(k)[0]
        v0 = ps._versions[k]
        bad = _private_state(ps, k, 9.0)
        ps._push({"grant": _grant(path, k, "primary"), "private": {k: bad}}, peer_id="ATK")
        for _ in range(5):  # replay the same grant -> rejected, no second vote
            ps._private_proposal({"private": {k: bad}, "grant": _grant(path, k, "primary")},
                                 peer_id="ATK")
        assert ps._versions[k] == v0
        # A forged grant (wrong key) is refused outright -- can't conjure a vote.
        forged = make_grant(path, [k], 0.0, "forged", grant_key="wrong")
        ps._private_proposal({"private": {k: bad}, "grant": forged}, peer_id="ATK2")
        assert ps._versions[k] == v0          # never applied -> not poisoned
    finally:
        ps.shutdown()


def test_disagreeing_proposals_do_not_corroborate():
    """Two distinct grants proposing *different* states form a quorum on neither."""
    ps = _priv_ps(private_policy="proposal", private_quorum=2)
    try:
        k = next(x for x in ps.owned_keys if is_private_key(x))
        path = ps._topology.paths_through_module(k)[0]
        v0 = ps._versions[k]
        ps._push({"grant": _grant(path, k, "primary"),
                  "private": {k: _private_state(ps, k, 1.0)}}, peer_id="A")
        ps._private_proposal({"private": {k: _private_state(ps, k, 2.0)},
                              "grant": _grant(path, k, "check1")}, peer_id="B")
        assert ps._versions[k] == v0          # no two agree -> nothing applied
    finally:
        ps.shutdown()


def test_overwrite_policy_applies_verbatim():
    """The default policy is unchanged: a single private push applies at once."""
    ps = _priv_ps(private_policy="overwrite")
    try:
        k = next(x for x in ps.owned_keys if is_private_key(x))
        path = ps._topology.paths_through_module(k)[0]
        v0 = ps._versions[k]
        ps._push({"grant": _grant(path, k, "g"),
                  "private": {k: _private_state(ps, k, 1.0)}}, peer_id="A")
        assert ps._versions[k] > v0           # verbatim, no corroboration needed
    finally:
        ps.shutdown()


def test_uncommitted_audit_is_reaped():
    """An audit whose primary never commits (worker died) must still expire and
    be reaped -- otherwise it leaks and its (path, gen) slot blocks re-auditing."""
    sched = _serving(redundancy=3, redundancy_rate=1.0, audit_timeout=0.0)
    try:
        task = sched._next_task({"worker_id": "P"}, peer_id="P")
        key = (tuple(task["path"]), task["gen_id"])
        assert key in sched._audits and sched._audits[key]["base"] is None
        with sched._lock:
            sched._reclaim_inflight_locked()     # the reaper sweeps timed-out audits
        assert key not in sched._audits          # popped, not leaked
    finally:
        sched.shutdown()


def test_private_proposal_bucket_is_bounded():
    """Proposals that never reach quorum (persistent disagreement) are FIFO-capped
    per key, not accumulated unboundedly across generations."""
    ps = _priv_ps(private_policy="proposal", private_quorum=99)  # never reaches quorum
    try:
        k = next(x for x in ps.owned_keys if is_private_key(x))
        path = ps._topology.paths_through_module(k)[0]
        for i in range(ps._PRIVATE_PROPOSAL_MAX + 10):   # each a distinct state/grant
            ps._private_proposal({"private": {k: _private_state(ps, k, float(i + 1))},
                                  "grant": _grant(path, k, f"t{i}")}, peer_id=f"p{i}")
        assert len(ps._private_proposals[k]) == ps._PRIVATE_PROPOSAL_MAX  # capped
        assert ps._versions[k] == (0, 0)                 # quorum unreachable -> nothing applied
    finally:
        ps.shutdown()


def test_no_audits_when_rate_zero():
    sched = _serving(redundancy=3, redundancy_rate=0.0)
    try:
        for i in range(len(sched.paths)):
            t = sched._next_task({"worker_id": f"w{i}"}, peer_id=f"w{i}")
            assert not t.get("audit")
        assert not sched._audits                     # never sampled -> byte-identical path
    finally:
        sched.shutdown()
