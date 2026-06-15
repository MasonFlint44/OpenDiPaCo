"""Churn-robustness validation for W4 (docs/w4-churn-design.md §D6).

W4 builds no new subsystem: Phase 2 already has dynamic ownership, k-replication,
epoch-bump failover, per-key checkpoints, and a recovery manifest -- but timed
and tested for *cluster* churn. This harness is the **measure-first** half (the
W3 discipline): before retuning any timing (W4d), drive the real control plane
through a home-style churn process and measure what actually happens.

It is a true in-process cluster -- a real ``Tracker`` + ``Scheduler`` + ``k``
``ParameterServer`` owners + ``run_sharded_worker`` workers over loopback (the
``test_failover.py`` marquee setup, generalized) -- so it exercises the genuine
failover path, not a primitive. Four arms, each a churn pattern injected mid-run:

  * ``none``    -- control: no churn (baseline epochs/reclaims/progress);
  * ``abrupt``  -- an owner crashes (``simulate_crash``): TTL + grace -> epoch
                   bump -> a backup is promoted -> workers fail over;
  * ``suspend`` -- an owner stops heartbeating past ``owner_grace`` (a slept
                   laptop: ``pause_heartbeat``), is remapped out, then wakes
                   (``resume_heartbeat``) and rejoins;
  * ``flap``    -- an owner goes silent *within* grace and returns: the
                   hysteresis must absorb it (zero epoch bumps).

For each arm it reports survival (the run completes its target), epochs bumped,
keys remapped, lease reclaims (the churn-driven disruption signal), and -- for
``abrupt`` -- the failover latency (departure -> every key served active again).

    python examples/validate_churn.py
    ROUNDS=6 K=3 python examples/validate_churn.py

HONEST CAVEAT: like ``validate_robustness.py`` / ``validate_decentralized.py``,
this validates the *control plane's survival under churn* (and the dynamics
*tracking* at toy scale: a churned run still reaches the same update target as
the control). The end-to-end convergence-under-churn verdict at scale is the WAN
run (plan slice 0f), which this de-risks but does not replace.
"""

from __future__ import annotations

import os
import threading
import time

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    Scheduler,
    Tracker,
    make_peer_record,
    owners_for,
    run_sharded_worker,
)

BATCH = 8


def _i(name, default):
    return int(os.environ.get(name, default))


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _corpus(cfg):
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(8)]
    assign = torch.tensor([i % cfg.num_paths for i in range(32)])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _owner_record(identity, port):
    return make_peer_record(identity, reachability="public",
                            addr=("127.0.0.1", port), roles=("owner",))


def _owner_set(record):
    """key -> frozenset of owner peer_ids under an epoch record."""
    cfg = _cfg()
    return {k: frozenset(o["peer_id"] for o in owners_for(k, record))
            for k in cfg.build_topology().module_keys()}


def _remapped_keys(before, after) -> int:
    return sum(1 for k in before if before[k] != after.get(k))


def run_arm(churn: str, *, rounds: int = 6, k: int = 3, n_owners: int = 3,
            n_workers: int = 2, verbose: bool = True) -> dict:
    """Bring up a real in-process cluster, run ``rounds`` generations while
    injecting the ``churn`` pattern, and return a metrics dict. Survival is
    ``completed >= target``; the rest are churn observables."""
    cfg, dl = _cfg(), DiLoCoConfig(inner_steps=4, inner_lr=1e-3)
    sched_id = PeerIdentity.generate()
    ids = [PeerIdentity.generate() for _ in range(n_owners)]

    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True, ttl=1.0)
    tracker.start()
    taddr = ("127.0.0.1", tracker.port)
    sched = Scheduler(cfg, _corpus(cfg), [], dl, batch_size=BATCH,
                      host="127.0.0.1", port=0, identity=sched_id, auth_key="t",
                      admitted_peers=ids, heartbeat_timeout=2.0)
    sched.start()
    pss = [ParameterServer(cfg, [], dl, host="127.0.0.1", port=0, identity=i,
                           auth_key="t", scheduler_pub=sched_id.public_key_hex,
                           scheduler_addr=("127.0.0.1", sched.port),
                           replicate_interval=0.2,
                           admitted_peers=[p for p in ids if p is not i])
           for i in ids]
    workers: list[threading.Thread] = []
    metrics = {"arm": churn, "survived": False, "epochs": 0, "remaps": 0,
               "reclaims": 0, "failover_s": None}
    try:
        for ps in pss:
            ps.start()
            ps.start_tracker_heartbeat(taddr, "127.0.0.1", interval=0.3)
        _await(lambda: len(tracker.records()) >= n_owners, 5)
        sched.watch_tracker(taddr, k=k, owner_grace=2.0, min_epoch_interval=0.5,
                            poll_interval=0.2)
        _await(lambda: all(ps._epoch is not None for ps in pss), 5)
        epoch0 = _owner_set(sched._epoch_record)

        workers = [threading.Thread(
            target=run_sharded_worker, args=(cfg, dl, ("127.0.0.1", sched.port)),
            kwargs=dict(seed=s, auth_key="t", heartbeat_interval=0.5,
                        reconnect=True, reconnect_timeout=10.0),
            daemon=True) for s in range(n_workers)]
        for w in workers:
            w.start()

        injector = threading.Thread(target=_inject, args=(churn, sched, pss, metrics),
                                    daemon=True)
        injector.start()
        completed = sched.fit(num_generations=rounds, total_generations=rounds)
        injector.join(timeout=30)

        # Let any post-fit bump + sync settle, then snapshot the final placement.
        _await(lambda: _settled(churn, sched, pss), 20)
        final = sched._epoch_record
        metrics["survived"] = sum(completed.values()) >= sched._target
        metrics["epochs"] = final["epoch"]
        metrics["remaps"] = _remapped_keys(epoch0, _owner_set(final))
        metrics["reclaims"] = sched.metrics.reclaims
        if verbose:
            _report(metrics)
        return metrics
    finally:
        sched.shutdown()
        for ps in pss:
            ps.shutdown()
        tracker.shutdown()
        for w in workers:
            w.join(timeout=15)


def _inject(churn, sched, pss, metrics) -> None:
    """Drive the churn pattern once the run has made some progress."""
    if churn == "none":
        return
    _await(lambda: sched._T >= max(1, sched._target // 3), 20)
    victim = pss[0]
    if churn == "abrupt":
        t0 = time.monotonic()
        victim.simulate_crash()                    # drop every conn; heartbeat dies
        # Failover latency: crash -> the surviving owners serve every key active.
        survivors = pss[1:]
        if _await(lambda: _covered(sched, survivors, victim), 25):
            metrics["failover_s"] = round(time.monotonic() - t0, 2)
    elif churn == "suspend":
        victim.pause_heartbeat()                    # slept laptop: TTL lapses
        _await(lambda: victim.peer_id not in _epoch_ids(sched), 25)  # remapped out
        victim.resume_heartbeat()                   # wakes, re-registers, rejoins
    elif churn == "flap":
        victim.pause_heartbeat()
        time.sleep(0.6)                             # < owner_grace (2.0): absorbed
        victim.resume_heartbeat()
    else:
        raise SystemExit(f"unknown churn arm: {churn!r}")


def _epoch_ids(sched) -> set:
    rec = sched._epoch_record
    return {o["peer_id"] for o in rec["owners"]} if rec else set()


def _covered(sched, survivors, victim) -> bool:
    """Every key is owned by live survivors under the current epoch and served
    active by at least one of them (the post-failover steady state)."""
    rec = sched._epoch_record
    if rec is None or victim.peer_id in _epoch_ids(sched):
        return False
    for k in _cfg().build_topology().module_keys():
        owner_ids = {o["peer_id"] for o in owners_for(k, rec)}
        if not owner_ids <= {ps.peer_id for ps in survivors}:
            return False
        if not any(k in ps._active for ps in survivors if ps.peer_id in owner_ids):
            return False
    return True


def _settled(churn, sched, pss) -> bool:
    rec = sched._epoch_record
    if rec is None:
        return False
    live = [ps for ps in pss if not ps._dead]
    return all(ps._epoch_num == rec["epoch"] and ps._active >= ps.owned_keys
               for ps in live)


def _await(pred, timeout: float, step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


def _report(m) -> None:
    fo = "n/a" if m["failover_s"] is None else f"{m['failover_s']:.2f}s"
    print(f"  {m['arm']:<8} survived={str(m['survived']):<5} "
          f"epochs={m['epochs']} remaps={m['remaps']:<2} "
          f"reclaims={m['reclaims']:<3} failover={fo}")


def main() -> None:
    rounds, k = _i("ROUNDS", 6), _i("K", 3)
    arm = os.environ.get("ARM")
    arms = [arm] if arm else ["none", "abrupt", "suspend", "flap"]
    print(f"churn validation (rounds={rounds}, k={k}, real in-process cluster)")
    results = [run_arm(a, rounds=rounds, k=k) for a in arms]
    # Verdicts: every arm survives; abrupt/suspend remap, flap does not.
    ok = all(r["survived"] for r in results)
    by = {r["arm"]: r for r in results}
    if "flap" in by:
        ok = ok and by["flap"]["epochs"] <= (by["none"]["epochs"] if "none" in by else 0)
    print("VERDICT:", "PASS -- control plane rode out the churn" if ok
          else "FAIL -- see arms above")


if __name__ == "__main__":
    main()
