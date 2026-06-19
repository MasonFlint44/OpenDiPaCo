"""W7a: the worker's bounded shard cache (docs/w7-data-decentralization-design.md).

A long-lived worker that fails over across many paths must not keep every shard
it ever leased resident. ``_ShardCache`` is an LRU keyed by path; eviction only
changes *when* a shard is rebuilt (re-materialized from the spec / reloaded),
never its bytes.
"""

import pytest
import torch

from opendipaco.data.spec import make_shard_spec, materialize_shard, synthetic_source
from opendipaco.schedule.distributed import _ShardCache


def test_one_path_worker_never_evicts():
    c = _ShardCache(4)
    c[(0,)] = torch.zeros(3)
    for _ in range(10):
        assert c[(0,)] is not None          # repeated leases of the same path
    assert list(c) == [(0,)] and len(c) == 1


def test_evicts_least_recently_used_at_capacity():
    c = _ShardCache(2)
    c[(0,)] = torch.tensor([0.0])
    c[(1,)] = torch.tensor([1.0])
    assert set(c) == {(0,), (1,)}
    c[(2,)] = torch.tensor([2.0])           # over cap -> drop the LRU ((0,))
    assert set(c) == {(1,), (2,)}
    assert (0,) not in c and len(c) == 2


def test_get_marks_most_recently_used():
    c = _ShardCache(2)
    c[(0,)] = torch.tensor([0.0])
    c[(1,)] = torch.tensor([1.0])
    _ = c[(0,)]                             # touch (0,) -> now MRU, (1,) is LRU
    c[(2,)] = torch.tensor([2.0])          # evicts (1,), keeps the touched (0,)
    assert set(c) == {(0,), (2,)}


def test_advertisement_snapshot_is_safe():
    # list(cache) (the `cached_shards` advertisement) must reflect current keys
    # and never raise even though __getitem__ reorders the mapping.
    c = _ShardCache(3)
    for p in range(3):
        c[(p,)] = torch.tensor([float(p)])
    assert sorted(list(c)) == [(0,), (1,), (2,)]


def test_maxsize_must_be_positive():
    with pytest.raises(ValueError, match="maxsize"):
        _ShardCache(0)


def test_none_maxsize_uses_library_default():
    # None (a stripped-manifest / unset config knob) -> the library default, not a
    # crash: a worker that inherits no operator value still gets a bounded cache.
    c = _ShardCache(None)
    for p in range(6):
        c[(p,)] = torch.tensor([float(p)])
    assert len(c) == 4                       # default cap held


def test_eviction_rematerializes_byte_identically():
    # The eviction contract: a dropped shard rebuilt from the same spec is
    # bit-identical (materialization is deterministic), so training is unchanged.
    spec = make_shard_spec(
        source=synthetic_source(vocab_size=40, num_documents=16, doc_len=20,
                                topics=2, seed=0),
        routing={"kind": "round_robin"}, num_paths=2, seq_len=16)
    first = materialize_shard(spec, 0)
    c = _ShardCache(1)
    c[(0,)] = first
    c[(1,)] = materialize_shard(spec, 1)   # over cap -> evicts path 0
    assert (0,) not in c
    rebuilt = materialize_shard(spec, 0)   # the next lease re-materializes
    assert torch.equal(first, rebuilt)
