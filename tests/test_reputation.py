"""Tests for reputation + rate limiting (internet-scale plan, Phase 3b).

The reputation substrate (floor start, earned climb, debited demotion, decay
toward floor, anonymous pass-through, checkpoint round-trip), the token-bucket
rate limiter, and their wiring into the scheduler: accepted commits credit the
committing peer, bad-loss commits debit it, a throttled peer gets a backoff
idle instead of a task, reputation survives a restart, and a demoted peer is
excluded from the owner set.
"""

import time

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import (
    PeerIdentity,
    RateLimiter,
    Reputation,
    Scheduler,
    make_peer_record,
)
from opendipaco.schedule.ownership import EpochManager

BATCH = 8


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _corpus(cfg):
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(8)]
    assign = torch.tensor([i % cfg.num_paths for i in range(32)])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _serving_scheduler(**kw):
    cfg = _cfg()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", 1)], DiLoCoConfig(inner_steps=4),
                      batch_size=BATCH, host="127.0.0.1", port=0, **kw)
    with sched._lock:
        sched._serving = True
        sched._target = 10 ** 9
        sched._completed = {p: 0 for p in sched.paths}
    return sched


# -- reputation unit -----------------------------------------------------------


def test_fresh_peer_starts_at_floor_and_earns_above_it():
    r = Reputation(floor=0.5, credit=0.1, debit=0.3)
    assert r.get("p") == 0.5                     # never-seen -> floor, not max
    assert r.eligible("p", 0.25)                 # floor clears the owner gate...
    assert not r.eligible("p", 0.6)              # ...but isn't "trusted" yet
    for _ in range(10):
        r.credit("p")
    assert r.get("p") > 0.99                     # earned, clamped at hi


def test_debit_demotes_below_owner_threshold():
    r = Reputation(floor=0.5, credit=0.02, debit=0.3)
    r.debit("bad")
    assert r.get("bad") < 0.25                   # one bad act drops it past the gate
    assert not r.eligible("bad", 0.25)
    # And a single good act doesn't repay it (debit > credit by design).
    r.credit("bad")
    assert not r.eligible("bad", 0.25)


def test_decay_pulls_toward_floor():
    r = Reputation(floor=0.5, decay_halflife=100.0)
    r._scores["p"] = [1.0, time.monotonic() - 100.0]   # one half-life stale
    assert abs(r.get("p") - 0.75) < 1e-3               # halved the distance to floor
    r._scores["q"] = [0.0, time.monotonic() - 100.0]
    assert abs(r.get("q") - 0.25) < 1e-3               # recovers upward too


def test_anonymous_peer_is_untracked_and_eligible():
    r = Reputation(floor=0.5)
    assert r.get(None) == 0.5
    assert r.eligible(None, 0.99)                # HMAC/trusted-cluster: ungated
    assert r.credit(None) == 0.5 and r.debit(None) == 0.5  # no-ops


def test_snapshot_restore_round_trip():
    r = Reputation(floor=0.5, credit=0.1, decay_halflife=0.0)  # no decay -> exact
    for _ in range(3):
        r.credit("p")
    snap = r.snapshot()
    r2 = Reputation(floor=0.5, decay_halflife=0.0)
    r2.restore(snap)
    assert abs(r2.get("p") - r.get("p")) < 1e-6


# -- rate limiter unit ---------------------------------------------------------


def test_token_bucket_throttles_then_refills():
    rl = RateLimiter(capacity=2, refill_per_sec=0.0)
    assert rl.allow("p", reputation=0.5)         # cap = 2 * (0.5+0.5) = 2
    assert rl.allow("p", reputation=0.5)
    assert not rl.allow("p", reputation=0.5)     # exhausted
    rl2 = RateLimiter(capacity=1, refill_per_sec=1000.0)
    assert rl2.allow("p", reputation=0.5)
    time.sleep(0.05)
    assert rl2.allow("p", reputation=0.5)        # refilled


def test_higher_reputation_gets_a_bigger_bucket():
    trusted = RateLimiter(capacity=2, refill_per_sec=0.0)
    n = 0
    while trusted.allow("hi", reputation=1.0):   # cap = 2 * 1.5 = 3
        n += 1
    assert n == 3
    fresh = RateLimiter(capacity=2, refill_per_sec=0.0)
    m = 0
    while fresh.allow("lo", reputation=0.0):     # cap = 2 * 0.5 = 1
        m += 1
    assert m == 1


def test_anonymous_never_limited():
    rl = RateLimiter(capacity=1, refill_per_sec=0.0)
    assert all(rl.allow(None) for _ in range(100))


# -- scheduler integration -----------------------------------------------------


def test_accepted_commit_credits_bad_loss_debits():
    sched = _serving_scheduler(reputation=Reputation(floor=0.5, credit=0.1, debit=0.3))
    try:
        good = sched._next_task({"worker_id": "w"}, peer_id="good")
        acc = sched._commit({"path": good["path"], "lease": good["lease"]}, peer_id="good")
        assert acc["accepted"] and sched.reputation.get("good") > 0.5  # credited

        bad = sched._next_task({"worker_id": "w"}, peer_id="bad")
        rej = sched._commit({"path": bad["path"], "lease": bad["lease"],
                             "loss": float("nan")}, peer_id="bad")
        assert not rej["accepted"] and sched.reputation.get("bad") < 0.5  # debited
    finally:
        sched.shutdown()


def test_throttled_peer_gets_idle_not_task():
    sched = _serving_scheduler(rate_limiter=RateLimiter(capacity=1, refill_per_sec=0.0))
    try:
        first = sched._next_task({"worker_id": "w"}, peer_id="p")
        assert first["type"] == "task"           # spends the one token
        second = sched._next_task({"worker_id": "w"}, peer_id="p")
        assert second["type"] == "idle"          # throttled, not disconnected
        # A different peer has its own bucket and is unaffected.
        assert sched._next_task({"worker_id": "w2"}, peer_id="q")["type"] == "task"
    finally:
        sched.shutdown()


def test_reputation_survives_checkpoint(tmp_path):
    ckpt = str(tmp_path / "ck")
    sched = _serving_scheduler(reputation=Reputation(floor=0.5, credit=0.1, decay_halflife=0.0))
    try:
        for _ in range(3):
            sched.reputation.credit("p")
        earned = sched.reputation.get("p")
        sched._checkpoint_cluster(ckpt)          # PS unreachable, but scheduler.pt is written
    finally:
        sched.shutdown()
    sched2 = _serving_scheduler(reputation=Reputation(floor=0.5, decay_halflife=0.0))
    try:
        sched2._load_state(ckpt)
        assert abs(sched2.reputation.get("p") - earned) < 1e-6
    finally:
        sched2.shutdown()


def test_demoted_peer_excluded_from_owner_set():
    """The EpochManager reputation predicate keeps a demoted peer out of the
    owner set even though its record is owner-eligible; a floor peer stays in."""
    rep = Reputation(floor=0.5, debit=0.4)
    rep.debit("demoted")                         # -> 0.1, below the 0.25 gate
    mgr = EpochManager(owner_grace=10.0, min_epoch_interval=0.0,
                       is_eligible=lambda pid: rep.eligible(pid, 0.25))
    ids = {name: PeerIdentity.generate() for name in ("good", "demoted")}
    # Map the generated peer_ids back to our labels for the predicate.
    rep._scores[ids["demoted"].peer_id] = rep._scores.pop("demoted")
    recs = [make_peer_record(ident, reachability="public",
                             addr=("127.0.0.1", 9000 + i), roles=("owner",))
            for i, ident in enumerate(ids.values())]
    due = mgr.observe(recs, now=1.0)
    owners = {r["peer_id"] for r in due}
    assert ids["good"].peer_id in owners
    assert ids["demoted"].peer_id not in owners  # excluded despite a valid record
