"""Tests for robust aggregation (internet-scale plan, Phase 3a, finding 1.1).

The pure aggregator (trimmed-mean/median tolerance, c=1 identity, summed-weight
magnitude) and its wiring into the owner: quorum buffering, timeout flush, the
bit-identical ``off`` path, and the end-to-end property that a sign-flip
minority moves the robust owner's bank exactly like the honest-only update
while the unprotected sum is poisoned.
"""

import pytest
import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.schedule import ParameterServer, make_grant, robust_delta
from opendipaco.schedule.aggregate import check_aggregate
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


# -- the pure aggregator -------------------------------------------------------


def test_single_contribution_is_identity():
    """c=1: direction is the grad itself, magnitude is its weight -> exactly
    weight*grad, the seam that keeps `off` and degree-1 quorums bit-identical."""
    g = [torch.randn(3, 4), torch.randn(5)]
    for agg in ("mean", "trimmed_mean", "median"):
        delta, wsum = robust_delta([(0.7, g)], aggregate=agg)
        assert wsum == 0.7
        assert all(torch.equal(d, t) for d, t in zip(delta, g))  # not just close


def test_weight_sum_is_the_magnitude():
    g = [torch.ones(4)]
    _, wsum = robust_delta([(0.5, g), (0.25, [t.clone() for t in g]),
                            (1.0, [t.clone() for t in g])], aggregate="mean")
    assert wsum == pytest.approx(1.75)


def test_trimmed_mean_and_median_drop_a_minority_outlier():
    honest = [torch.ones(6)]
    adv = [torch.full((6,), -100.0)]
    contribs = [(1.0, honest), (1.0, [t.clone() for t in honest]), (1.0, adv)]
    for agg in ("trimmed_mean", "median"):
        delta, _ = robust_delta(contribs, aggregate=agg)
        assert torch.allclose(delta[0], torch.ones(6)), agg  # adversary trimmed out
    # An outlier in either direction is rejected, and on any subset of coords.
    mixed = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 1.0]),
             torch.tensor([1e6, 1.0, -1e6])]
    delta, _ = robust_delta([(1.0, [mixed[0]]), (1.0, [mixed[1]]), (1.0, [mixed[2]])],
                            aggregate="trimmed_mean")
    assert torch.allclose(delta[0], torch.ones(3))


def test_mean_is_used_below_three():
    """Fewer than three contributions can't be trimmed; trimmed_mean falls back
    to the plain mean rather than dropping everything."""
    a, b = [torch.tensor([2.0, 4.0])], [torch.tensor([4.0, 8.0])]
    delta, _ = robust_delta([(1.0, a), (1.0, b)], aggregate="trimmed_mean")
    assert torch.allclose(delta[0], torch.tensor([3.0, 6.0]))


def test_check_aggregate_rejects_unknown():
    with pytest.raises(ValueError):
        check_aggregate("krum")


# -- owner-side quorum buffering ----------------------------------------------


def _ps(robustness="off", **kw):
    return ParameterServer(_cfg(), sorted(_cfg().build_topology().module_keys()),
                           _diloco(), host="127.0.0.1", port=0, robustness=robustness, **kw)


def _shared_key(ps, degree):
    topo = ps._topology
    return next(k for k in sorted(ps.owned_keys)
                if not is_private_key(k) and topo.sharing_count(k) == degree)


def _grad_ones(ps, key, scale=1.0):
    return [torch.ones_like(p) * scale for p in ps.bank[key].parameters()]


def _push(ps, key, token, *, weight=1.0, scale=1.0):
    path = ps._topology.path_from_index(0)
    return ps._push({"grant": make_grant(path, [key], weight, token),
                     "updates": {key: {"grad": _grad_ones(ps, key, scale)}}})


def test_off_applies_each_push_immediately():
    ps = _ps("off")
    try:
        k = _shared_key(ps, 4)            # embed/head: shared by every path
        assert ps._versions[k] == (0, 0)
        _push(ps, k, "t1")
        assert ps._versions[k] == (0, 1)  # applied on arrival
        _push(ps, k, "t2")
        assert ps._versions[k] == (0, 2)
        assert not ps._buffers                # nothing buffered in off mode
    finally:
        ps.shutdown()


def test_on_buffers_until_quorum_then_applies_once():
    ps = _ps("on", quorum_target=3)
    try:
        k = _shared_key(ps, 4)            # sharing_count 4 -> quorum min(4,3)=3
        assert ps._quorum_c(k) == 3
        before = {n: p.detach().clone() for n, p in ps.bank[k].named_parameters()}
        _push(ps, k, "t1")
        _push(ps, k, "t2")
        assert ps._versions[k] == (0, 0)  # buffered, not yet applied
        assert all(torch.equal(before[n], p) for n, p in ps.bank[k].named_parameters())
        assert len(ps._buffers[k]) == 2
        _push(ps, k, "t3")               # quorum reached -> one aggregated step
        assert ps._versions[k] == (0, 1)
        assert k not in ps._buffers       # buffer cleared
        assert any(not torch.equal(before[n], p)
                   for n, p in ps.bank[k].named_parameters())
    finally:
        ps.shutdown()


def test_low_degree_module_aggregates_at_its_degree():
    ps = _ps("on", quorum_target=3)
    try:
        k = _shared_key(ps, 2)           # a level expert: shared by 2 paths
        assert ps._quorum_c(k) == 2       # capped by degree, not quorum_target
        _push(ps, k, "t1")
        assert ps._versions[k] == (0, 0)
        _push(ps, k, "t2")
        assert ps._versions[k] == (0, 1)  # flushes at 2
    finally:
        ps.shutdown()


def test_timeout_flushes_a_partial_buffer():
    ps = _ps("on", quorum_target=3, quorum_timeout=0.0)  # 0 == flush on next sweep
    try:
        k = _shared_key(ps, 4)
        _push(ps, k, "t1")               # 1 of 3; never reaches quorum
        assert ps._versions[k] == (0, 0)
        ps._sweep_buffers()              # the replication loop calls this periodically
        assert ps._versions[k] == (0, 1)  # partial buffer applied (liveness valve)
        assert k not in ps._buffers
    finally:
        ps.shutdown()


def test_sign_flip_minority_tracks_honest_update_exactly():
    """The end-to-end guarantee: with c=3, two honest pushes and one sign-flipped
    adversary, the robust owner's bank lands *exactly* where three honest pushes
    would (trimmed mean of {1,1,-k} == trimmed mean of {1,1,1} == 1), while the
    unprotected sum is dragged far away."""
    robust, honest_ref, naive = _ps("on", quorum_target=3), _ps("on", quorum_target=3), _ps("off")
    try:
        k = _shared_key(robust, 4)
        for tok in ("a", "b"):  # two honest pushes to each owner
            _push(robust, k, tok)
            _push(honest_ref, k, tok)
            _push(naive, k, tok)
        _push(robust, k, "c", scale=-500.0)  # robust + naive: a sign-flip adversary
        _push(naive, k, "c", scale=-500.0)
        _push(honest_ref, k, "c")            # reference: a third honest push

        r = dict(robust.bank[k].named_parameters())
        h = dict(honest_ref.bank[k].named_parameters())
        n = dict(naive.bank[k].named_parameters())
        assert all(torch.allclose(r[name], h[name]) for name in r)  # adversary erased
        assert any(not torch.allclose(r[name], n[name]) for name in r)  # sum was poisoned
    finally:
        for ps in (robust, honest_ref, naive):
            ps.shutdown()
