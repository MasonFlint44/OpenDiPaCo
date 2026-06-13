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


def _check(sched, key, peer, digest):
    path, gen = key
    sched._commit_check({"check_only": True, "path": list(path), "gen_id": gen,
                         "digest": digest}, peer_id=peer)


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


def test_no_audits_when_rate_zero():
    sched = _serving(redundancy=3, redundancy_rate=0.0)
    try:
        for i in range(len(sched.paths)):
            t = sched._next_task({"worker_id": f"w{i}"}, peer_id=f"w{i}")
            assert not t.get("audit")
        assert not sched._audits                     # never sampled -> byte-identical path
    finally:
        sched.shutdown()
