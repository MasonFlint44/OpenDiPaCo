"""Tests for the decentralized worker runtime (Phase 4 worker loop, slice a).

The worker has no scheduler. Per iteration it derives the epoch locally from the
directory, self-assigns a ``(path, generation)`` it is the HRW rank for,
quorum-fetches that path's bases from the keys' ``k`` replicas, trains, commits
to the path's coordinator (which version-fences the slot and mints the grant),
and pushes the pseudo-gradient to all ``k`` owners. These drive the loop's
mechanics through a fake in-process link (``addr -> owner._handle``) -- the same
transport seam TCP/libp2p use -- so no sockets are needed.
Design: docs/decentralized-worker-loop-design.md.
"""

import time

import pytest
import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import (
    AsyncScheduler,
    ParameterServer,
    PeerIdentity,
    make_epoch_record,
    make_peer_record,
    owners_for,
    path_primary,
    rank_workers,
)
from opendipaco.schedule.assignment import responsible_rank
from opendipaco.schedule.compress import state_digest
from opendipaco.schedule.ownership import owner_addr
from opendipaco.schedule.sharded import (
    _build_worker_engine,
    _decentralized_routing,
    _fetch_quorum_bases,
    _pick_assigned_path,
    _push_all_owners,
    _serve_decentralized,
)
from opendipaco.topology import is_private_key

LEASE_TTL = 30.0


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    # Private embedding/head -> every path has a unique private coordinator key.
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16,
                        embedding="private", head="private")


def _diloco():
    return DiLoCoConfig(inner_steps=2, inner_lr=1e-3)


def _corpus(cfg):
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(0, 40, (48,), generator=g) for _ in range(16)]
    assign = torch.tensor([i % cfg.num_paths for i in range(16)])
    return ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, cfg.sequence_length)


def _cluster(n_owners=3, k=3, **owner_kw):
    """A decentralized cluster: a signed epoch, the owner peer records, and the
    (unstarted) owners keyed by peer id. ``_handle`` runs fully in memory."""
    sched = PeerIdentity.generate()
    idents = [PeerIdentity.generate() for _ in range(n_owners)]
    recs = [make_peer_record(idn, reachability="public", addr=("127.0.0.1", 9000 + i),
                             roles=("owner",)) for i, idn in enumerate(idents)]
    epoch = make_epoch_record(sched, epoch=0, owner_records=recs, k=k)
    owners = {idn.peer_id: ParameterServer(
        _cfg(), [], _diloco(), host="127.0.0.1", port=0, identity=idn,
        epoch_record=epoch, schedule_mode="decentralized", k=k, read_quorum=2,
        **owner_kw) for idn in idents}
    return epoch, recs, owners


class _FakeLink:
    """Routes ``ps_rpc(addr, msg)`` to the in-process owner at ``addr`` via its
    ``_handle`` -- the transport seam, with no sockets. ``drop`` addrs simulate an
    unreachable owner; ``caller`` is the peer id owners see (for reputation /
    assignee gating)."""

    def __init__(self, epoch, owners, *, caller=None, drop=()):
        self._by_addr = {tuple(owner_addr(o)): owners[o["peer_id"]]
                         for o in epoch["owners"]}
        self.caller = caller
        self.drop = {self.addr_key(a) for a in drop}
        self.calls: list = []

    def addr_key(self, a):
        return tuple(a) if isinstance(a, (list, tuple)) else a

    def connected(self, addr):
        return False

    def ps_rpc(self, addr, msg):
        addr = self.addr_key(addr)
        self.calls.append((addr, msg.get("type")))
        if addr in self.drop:
            raise OSError(f"unreachable {addr}")
        ps = self._by_addr.get(addr)
        if ps is None:
            raise OSError(f"no owner at {addr}")
        return ps._handle(msg, 0, peer_id=self.caller)

    def close(self):
        pass


def _shutdown(owners):
    for ps in owners.values():
        ps.shutdown()


def _worker_rec(ident):
    return make_peer_record(ident, reachability="nat", roles=("worker",))


# -- self-assignment (D3) ------------------------------------------------------


def test_assignee_claims_its_slot_and_a_non_member_skips():
    epoch, recs, owners = _cluster()
    topo = _cfg().build_topology()
    try:
        wids = [PeerIdentity.generate().peer_id for _ in range(4)]
        link = _FakeLink(epoch, owners)
        # The worker that is rank-0 of the first-scanned path claims it (HRW need
        # not give every worker a slot, so pick a known assignee deterministically).
        first = topo.path_from_index(0)
        r0 = rank_workers(first, 0, wids)[0]
        got = _pick_assigned_path(link, topo, epoch, wids, r0, salt="", lease_ttl=LEASE_TTL)
        assert got is not None
        path, g, routing = got
        assert path == first and rank_workers(path, g, wids)[0] == r0
        assert routing and all(routing[k] for k in routing)
        # A peer that isn't in the worker directory is assignee of nothing.
        assert _pick_assigned_path(link, topo, epoch, wids, "stranger",
                                   salt="", lease_ttl=LEASE_TTL) is None
    finally:
        _shutdown(owners)


def test_lease_ttl_zero_disables_takeover():
    """A coordinator reporting lease_ttl=0 means 'never hand the slot to a
    successor' (responsible_rank -> rank 0 forever). The worker must honor the
    reported 0, not silently fall back to its own default (the falsy-zero trap)."""
    epoch, recs, owners = _cluster(lease_ttl=0.0)
    topo = _cfg().build_topology()
    try:
        wids = [PeerIdentity.generate().peer_id for _ in range(4)]
        first = topo.path_from_index(0)
        r0 = rank_workers(first, 0, wids)[0]
        coord_id = path_primary(topo.path_module_keys(first), epoch)["peer_id"]
        with owners[coord_id]._lock:                     # generation open "forever"
            owners[coord_id]._gen[first] = [0, time.monotonic() - 1e6]
        link = _FakeLink(epoch, owners)
        # Despite the huge age, takeover is disabled -> rank 0 still owns it. The
        # 999.0 fallback must NOT be used (it would hand the slot to a successor).
        got = _pick_assigned_path(link, topo, epoch, wids, r0, salt="", lease_ttl=999.0)
        assert got is not None and got[0] == first
    finally:
        _shutdown(owners)


def test_decentralized_owner_rejects_lossy_compression():
    """Quorum reads confirm weights by cross-replica byte-digest agreement, which
    lossy downlink compression breaks -> reject it at construction (loud) rather
    than livelock every worker on a digest that can never match."""
    sched = PeerIdentity.generate()
    idn = PeerIdentity.generate()
    rec = make_peer_record(idn, reachability="public", addr=("127.0.0.1", 9000),
                           roles=("owner",))
    ep = make_epoch_record(sched, epoch=0, owner_records=[rec], k=1)
    with pytest.raises(ValueError, match="compress='none'"):
        ParameterServer(_cfg(), [], _diloco(), host="127.0.0.1", port=0, identity=idn,
                        epoch_record=ep, schedule_mode="decentralized", k=1,
                        compress="int8")


def test_takeover_on_expiry_hands_a_stalled_slot_to_the_successor():
    epoch, recs, owners = _cluster()
    topo = _cfg().build_topology()
    try:
        wids = [PeerIdentity.generate().peer_id for _ in range(4)]
        first = topo.path_from_index(0)            # scanned first by _pick
        ranked = rank_workers(first, 0, wids)
        # Age the first path's generation well past lease_ttl so a successor owns
        # it; leave the others fresh (rank 0).
        big = 5 * LEASE_TTL
        tk = ranked[responsible_rank(big, LEASE_TTL, len(wids))]
        assert tk != ranked[0]                      # takeover actually moved the holder
        coord_id = path_primary(topo.path_module_keys(first), epoch)["peer_id"]
        with owners[coord_id]._lock:
            owners[coord_id]._gen[first] = [0, time.monotonic() - big]
        link = _FakeLink(epoch, owners)
        # The successor picks up the stalled first path...
        tk_pick = _pick_assigned_path(link, topo, epoch, wids, tk, salt="",
                                      lease_ttl=LEASE_TTL)
        assert tk_pick is not None and tk_pick[0] == first
        # ...and the former rank-0 no longer claims it (gets a later path or none).
        r0_pick = _pick_assigned_path(link, topo, epoch, wids, ranked[0], salt="",
                                      lease_ttl=LEASE_TTL)
        assert r0_pick is None or r0_pick[0] != first
    finally:
        _shutdown(owners)


# -- quorum-fetch (D4) ---------------------------------------------------------


def test_quorum_fetch_loads_the_majority_base_not_a_byzantine_replica():
    epoch, recs, owners = _cluster()
    topo = _cfg().build_topology()
    engine = _build_worker_engine(_cfg(), _diloco(), "cpu", 0)
    try:
        path = topo.path_from_index(0)
        shared = next(k for k in topo.path_module_keys(path) if not is_private_key(k))
        replicas = owners_for(shared, epoch)
        honest = owners[replicas[1]["peer_id"]]
        byz = owners[replicas[0]["peer_id"]]        # the rank-0 (first-tried) owner lies
        honest_digest = honest._digests({"keys": [shared]})["digests"][shared][1]
        with byz._lock:                             # poison the Byzantine replica's bytes
            for p in byz.bank[shared].parameters():
                p.data.mul_(100.0)
        bad_digest = byz._digests({"keys": [shared]})["digests"][shared][1]
        assert bad_digest != honest_digest

        link = _FakeLink(epoch, owners)
        routing = {shared: _decentralized_routing(topo, path, epoch, link)[shared]}
        fetched = _fetch_quorum_bases(engine, link, routing, read_quorum=2, cold=False)
        assert fetched[shared] == (0, 0)
        loaded = state_digest({n: p.detach().cpu()
                               for n, p in engine.bank[shared].state_dict().items()})
        assert loaded == honest_digest              # the majority bytes, not the liar's
    finally:
        _shutdown(owners)


def test_quorum_fetch_raises_when_no_quorum_is_reachable():
    epoch, recs, owners = _cluster()
    topo = _cfg().build_topology()
    engine = _build_worker_engine(_cfg(), _diloco(), "cpu", 0)
    try:
        path = topo.path_from_index(0)
        shared = next(k for k in topo.path_module_keys(path) if not is_private_key(k))
        replicas = owners_for(shared, epoch)
        # Drop two of three replicas -> only one digest -> no read quorum of 2.
        drop = [owner_addr(r) for r in replicas[1:]]
        link = _FakeLink(epoch, owners, drop=drop)
        routing = {shared: _decentralized_routing(topo, path, epoch, link)[shared]}
        try:
            _fetch_quorum_bases(engine, link, routing, read_quorum=2, cold=False)
            assert False, "expected OSError (no quorum)"
        except OSError:
            pass
    finally:
        _shutdown(owners)


# -- push to all k owners (D6) -------------------------------------------------


def test_push_lands_at_the_primary_and_co_owners_refuse():
    epoch, recs, owners = _cluster()
    topo = _cfg().build_topology()
    try:
        path = topo.path_from_index(0)
        keys = topo.path_module_keys(path)
        coord_id = path_primary(keys, epoch)["peer_id"]
        coord = owners[coord_id]
        link = _FakeLink(epoch, owners, caller="w")
        commit = coord._commit({"path": list(path), "generation": 0, "loss": 1.0,
                                "base_versions": {k: coord._versions[k]
                                                  for k in keys if k in coord._versions}},
                               peer_id="w")
        grant = commit["grant"]
        routing = _decentralized_routing(topo, path, epoch, link)
        shared_payload = {k: [torch.ones_like(p) for p in owners[
            owners_for(k, epoch)[0]["peer_id"]].bank[k].parameters()]
            for k in routing if not is_private_key(k)}
        # The push lands (the true primary of each key applied) -> nothing failed.
        failed = _push_all_owners(routing, grant, shared_payload, {}, link)
        assert failed == set()
        # Each shared key's primary advanced past (0, 0); a backup did not.
        for k in shared_payload:
            ranked = owners_for(k, epoch)
            assert owners[ranked[0]["peer_id"]]._versions[k] > (0, 0)
            assert owners[ranked[-1]["peer_id"]]._versions[k] == (0, 0)
        # The grant is single-use per server: a replay lands nowhere.
        assert _push_all_owners(routing, grant, shared_payload, {}, link) == set(shared_payload)
    finally:
        _shutdown(owners)


# -- one full iteration end to end ---------------------------------------------


def _run_iters(epoch, recs, owners, n_iters, *, max_tasks=None):
    cfg = _cfg()
    topo = cfg.build_topology()
    engine = _build_worker_engine(cfg, _diloco(), "cpu", 0)
    worker = AsyncScheduler(engine, num_workers=1)
    worker.seed = 0
    wident = PeerIdentity.generate()
    link = _FakeLink(epoch, owners, caller=wident.peer_id)
    directory = list(recs) + [_worker_rec(wident)]
    state = {"done": 0}
    clean = _serve_decentralized(
        link, engine, worker, wident.peer_id, _corpus(cfg), lambda: directory,
        k=3, salt="", read_quorum=2, lease_ttl=LEASE_TTL, batch_size=8,
        total_rounds=n_iters, max_tasks=max_tasks, poll_interval=0.0, state=state,
        warm=set(), max_iters=n_iters)
    return state, clean, topo


def test_one_iteration_commits_and_advances_the_generation():
    epoch, recs, owners = _cluster()
    try:
        state, _, topo = _run_iters(epoch, recs, owners, 1)
        assert state["done"] == 1
        # The single worker is assignee of every path, so it serves the first one.
        first = topo.path_from_index(0)
        coord = owners[path_primary(topo.path_module_keys(first), epoch)["peer_id"]]
        assert coord._gen[first][0] == 1                      # generation advanced 0 -> 1
        # The path's shared modules moved off their seeded (0, 0).
        for k in topo.path_module_keys(first):
            if not is_private_key(k):
                assert owners[owners_for(k, epoch)[0]["peer_id"]]._versions[k] > (0, 0)
    finally:
        _shutdown(owners)


def test_generation_advances_monotonically_across_iterations():
    epoch, recs, owners = _cluster()
    try:
        state, clean, topo = _run_iters(epoch, recs, owners, 3)
        assert state["done"] == 3                             # no stale double-counts
        first = topo.path_from_index(0)
        coord = owners[path_primary(topo.path_module_keys(first), epoch)["peer_id"]]
        assert coord._gen[first][0] == 3                      # fenced, one advance per commit
    finally:
        _shutdown(owners)


def test_max_tasks_stops_the_loop_cleanly():
    epoch, recs, owners = _cluster()
    try:
        state, clean, _ = _run_iters(epoch, recs, owners, 10, max_tasks=2)
        assert clean is True and state["done"] == 2
    finally:
        _shutdown(owners)
