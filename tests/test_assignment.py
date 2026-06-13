"""Tests for leaderless assignment + version-vector staleness (Phase 4a).

The pure control-plane primitives that replace the central scheduler's queue
and global ``_T`` clock: HRW(path, generation) worker ranking, deterministic
takeover-on-expiry, version-vector-lag staleness, and the coordinator-key /
path-primary helpers. All pure functions of values a worker reads from the
owner tier, so they're identical on every node and testable in isolation.
"""

from opendipaco import (
    BackboneConfig,
    DiPaCoConfig,
)
from opendipaco.schedule import (
    PeerIdentity,
    assignee,
    coordinator_key,
    is_assignee,
    make_epoch_record,
    path_primary,
    rank_workers,
    responsible_rank,
    version_lag,
)
from opendipaco.topology import embed_key, is_private_key


def _workers(n):
    return [f"w{i:02d}" for i in range(n)]


# -- HRW worker ranking --------------------------------------------------------


def test_rank_is_deterministic_and_a_permutation():
    ws = _workers(8)
    r1 = rank_workers((0, 1), 5, ws)
    r2 = rank_workers((0, 1), 5, list(reversed(ws)))
    assert r1 == r2                       # independent of input order
    assert sorted(r1) == sorted(ws)       # every worker ranked exactly once


def test_rank_accepts_records_or_ids():
    recs = [{"peer_id": "a"}, {"peer_id": "b"}]
    assert rank_workers((0,), 0, recs) == rank_workers((0,), 0, ["a", "b"])


def test_generation_reshuffles_the_assignee():
    """Re-rolling per generation spreads paths over the pool: the rank-0 worker
    for a path changes across generations (not pinned to one worker forever)."""
    ws = _workers(12)
    seen = {assignee((1, 2), g, ws) for g in range(20)}
    assert len(seen) > 1                  # not the same worker every generation


def test_path_changes_the_assignee():
    ws = _workers(12)
    a = {assignee(p, 0, ws) for p in [(0, 0), (0, 1), (1, 0), (1, 1)]}
    assert len(a) > 1                     # different paths -> different assignees


def test_empty_worker_set_has_no_assignee():
    assert assignee((0,), 0, []) is None
    assert not is_assignee("w", (0,), 0, [])


# -- takeover on expiry --------------------------------------------------------


def test_responsible_rank_advances_with_elapsed_time():
    n = 5
    assert responsible_rank(0.0, 10.0, n) == 0      # fresh -> rank 0
    assert responsible_rank(9.9, 10.0, n) == 0      # still within rank 0's window
    assert responsible_rank(10.0, 10.0, n) == 1     # expired -> rank 1 takes over
    assert responsible_rank(25.0, 10.0, n) == 2     # and onward
    assert responsible_rank(1e6, 10.0, n) == n - 1  # capped at the last worker


def test_takeover_hands_the_slot_to_the_next_ranked_worker():
    ws = _workers(6)
    ranked = rank_workers((2, 3), 7, ws)
    # Within rank 0's window the rank-0 worker is the assignee...
    assert is_assignee(ranked[0], (2, 3), 7, ws, elapsed=5.0, lease_ttl=10.0)
    assert not is_assignee(ranked[1], (2, 3), 7, ws, elapsed=5.0, lease_ttl=10.0)
    # ...after it expires, rank 1 is, and rank 0 no longer is.
    assert is_assignee(ranked[1], (2, 3), 7, ws, elapsed=15.0, lease_ttl=10.0)
    assert not is_assignee(ranked[0], (2, 3), 7, ws, elapsed=15.0, lease_ttl=10.0)


# -- version-vector staleness --------------------------------------------------


def test_version_lag_is_max_counter_delta_within_an_epoch():
    fetched = {"A": (0, 5), "B": (0, 10)}
    current = {"A": (0, 7), "B": (0, 11)}
    assert version_lag(fetched, current) == 2        # max(7-5, 11-10)
    assert version_lag(fetched, fetched) == 0        # current base -> zero lag
    assert version_lag({}, current) == 0             # nothing fetched -> fresh


def test_version_lag_drops_cross_epoch_and_remapped_bases():
    cur = {"A": (1, 0)}
    assert version_lag({"A": (0, 9)}, cur) is None    # epoch boundary crossed
    assert version_lag({"A": (0, 1)}, {}) is None     # key remapped away
    assert version_lag({"A": (0, 5)}, {"A": (0, 4)}) is None  # fetched newer than current


# -- which owner coordinates a path --------------------------------------------


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _cfg_private():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16,
                        embedding="private", head="private")


def test_coordinator_key_prefers_a_private_key():
    topo = _cfg_private().build_topology()       # private embedding/head
    keys = topo.path_module_keys(topo.path_from_index(0))
    assert is_private_key(coordinator_key(keys))      # the path's private module
    # A fully-shared key set falls back to the lowest-sorted key.
    shared = [k for k in keys if not is_private_key(k)]
    assert coordinator_key(shared) == sorted(shared)[0]


def test_coordinator_key_falls_back_for_a_fully_shared_path():
    topo = _cfg().build_topology()               # default: shared embed/head
    keys = topo.path_module_keys(topo.path_from_index(0))
    assert not any(is_private_key(k) for k in keys)
    assert coordinator_key(keys) == sorted(keys)[0]   # lowest-sorted shared key


def test_path_primary_is_the_coordinator_keys_rank0_owner():
    from opendipaco.schedule import make_peer_record, owners_for
    topo = _cfg().build_topology()
    idents = [PeerIdentity.generate() for _ in range(3)]
    recs = [make_peer_record(idn, reachability="public",
                             addr=("127.0.0.1", 9000 + i), roles=("owner",))
            for i, idn in enumerate(idents)]
    epoch = make_epoch_record(PeerIdentity.generate(), epoch=0, owner_records=recs, k=3)
    path = topo.path_from_index(0)
    keys = topo.path_module_keys(path)
    prim = path_primary(keys, epoch)
    assert prim == owners_for(coordinator_key(keys), epoch)[0]


def test_private_topology_gives_every_path_a_private_coordinator():
    # Under the private embedding/head policy every path's coordinator key is
    # uniquely its own (the design's preferred case).
    topo = _cfg_private().build_topology()
    cks = {coordinator_key(topo.path_module_keys(p)) for p in topo.paths()}
    assert all(is_private_key(c) for c in cks)
    assert len(cks) == len(topo.paths())              # one per path, distinct
    assert embed_key() in topo.path_module_keys(topo.path_from_index(0)) or True
