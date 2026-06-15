"""Tests for W4a churn harness + the suspend/resume hook (docs/w4-churn-design.md).

W4a is measure-first: a real in-process cluster driven through home-style churn,
reporting survival + failover metrics. These pin the new ``pause_heartbeat`` /
``resume_heartbeat`` suspend hook and the harness's arm contracts (survival,
abrupt failover, flap absorption) so a regression in the failover path is caught
in CI, not only on the WAN run.
"""

import time

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.schedule import ParameterServer, PeerIdentity, Tracker


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _await(pred, timeout, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


def test_pause_resume_heartbeat_lapses_and_restores_tracker_record():
    """pause_heartbeat() stops TTL refresh while the server stays up (a slept
    laptop): the tracker record expires. resume_heartbeat() re-registers it.
    Default is unpaused, so the normal liveness path is unchanged."""
    cfg = _cfg()
    ident = PeerIdentity.generate()
    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True, ttl=0.5)
    tracker.start()
    ps = ParameterServer(cfg, [], DiLoCoConfig(), host="127.0.0.1", port=0,
                         identity=ident, replicate_interval=60.0)
    try:
        ps.start()
        ps.start_tracker_heartbeat(("127.0.0.1", tracker.port), "127.0.0.1",
                                   interval=0.15)
        assert not ps._hb_paused.is_set()                       # unpaused default
        assert _await(lambda: ident.peer_id in {r["peer_id"] for r in tracker.records()}, 3)

        ps.pause_heartbeat()                                    # sleep: stop refreshing
        # The record lapses once its TTL passes with no re-registration.
        assert _await(lambda: ident.peer_id not in {r["peer_id"] for r in tracker.records()}, 3)

        ps.resume_heartbeat()                                   # wake: re-register
        assert _await(lambda: ident.peer_id in {r["peer_id"] for r in tracker.records()}, 3)
    finally:
        ps.shutdown()
        tracker.shutdown()


def test_churn_arm_abrupt_survives_and_fails_over():
    """The marquee churn arm through the harness: an owner crashes mid-run, the
    epoch bumps it out, a backup is promoted, the run still completes, and the
    failover latency is measured (departure -> every key served by survivors)."""
    from validate_churn import run_arm

    m = run_arm("abrupt", rounds=4, verbose=False)
    assert m["survived"]                       # training rode out the crash
    assert m["epochs"] >= 1                    # the death bumped the epoch
    assert m["remaps"] >= 1                    # the dead owner's keys moved
    assert m["failover_s"] is not None         # backups served every key again


def test_churn_arm_flap_is_absorbed():
    """An owner silent within owner_grace and back causes no epoch bump (the
    hysteresis the harness exists to size), and the run completes."""
    from validate_churn import run_arm

    m = run_arm("flap", rounds=4, verbose=False)
    assert m["survived"]
    assert m["epochs"] == 0 and m["remaps"] == 0


def test_churn_arm_none_is_quiet():
    """Control: no churn -> no epoch bumps, no remaps, no lease reclaims."""
    from validate_churn import run_arm

    m = run_arm("none", rounds=4, verbose=False)
    assert m["survived"] and m["epochs"] == 0 and m["remaps"] == 0 and m["reclaims"] == 0
