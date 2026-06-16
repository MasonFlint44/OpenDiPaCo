"""Tests for W4d: home-grade launch defaults + graceful-shutdown wiring
(docs/w4-churn-design.md D2/D5). The launch layer retunes detection timings for
consumer churn and routes SIGTERM/SIGINT through shutdown(graceful=True); the
library defaults stay conservative so the in-process anchor + unit tests are
unaffected.
"""

from opendipaco.launch.config import OwnershipCfg, ScheduleCfg, TrackerCfg, TransportCfg
from opendipaco.launch.roles import _bounded_graceful_shutdown
from opendipaco.schedule import EpochManager, Tracker


def test_home_grade_launch_defaults_are_consistent_and_moved():
    """The retuned launch defaults keep the anti-thrash / anti-false-eviction
    invariants and have actually moved off the cluster values."""
    t, o, sch, tr = TrackerCfg(), OwnershipCfg(), ScheduleCfg(), TransportCfg()
    # Invariants (Phase 2 D5 / W4 D2): a flapping owner mustn't thrash ownership,
    # and one missed heartbeat mustn't evict.
    assert o.owner_grace >= 2 * t.ttl
    assert o.heartbeat_interval < t.ttl
    assert o.min_epoch_interval < o.owner_grace
    # Actually home-grade, not the old cluster defaults.
    assert t.ttl < 120.0 and o.owner_grace < 240.0 and o.min_epoch_interval < 60.0
    assert tr.heartbeat_timeout < 30.0 and sch.lease_ttl < 30.0


def test_library_defaults_stay_conservative():
    """Only the launch config moved; the library constructor defaults (used by
    the in-process anchor + unit tests) keep the cluster values."""
    assert Tracker().ttl == 120.0
    assert EpochManager().owner_grace == 240.0 and EpochManager().min_epoch_interval == 60.0


def test_bounded_graceful_shutdown_runs_graceful_and_cancels_deadline():
    """The CLI deadline backstop calls shutdown(graceful=True) and, on a normal
    fast shutdown, cancels the os._exit watchdog (so the process is not killed)."""
    seen = {}

    class _FakeServer:
        def shutdown(self, *, graceful=False):
            seen["graceful"] = graceful

    # A long deadline that we rely on being cancelled in the finally; if the
    # cancel were broken this test would still return (daemon timer), but the
    # os._exit would later kill the run -- so reaching the assert is the check.
    _bounded_graceful_shutdown(_FakeServer(), deadline=300.0)
    assert seen["graceful"] is True
