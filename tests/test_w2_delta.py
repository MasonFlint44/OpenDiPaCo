"""Tests for W2a delta-down (docs/w2-bandwidth-design.md).

The async weights cache is structurally defeated (every contribution bumps a
version), so the owner re-ships full weights. Delta-down instead ships an int8
``current - keyframe`` when the worker holds a recent keyframe (in the owner's
version ring), falling back to a full ship (a new keyframe) when it ages out.
These check the codec, the owner's full/delta decision, byte-identity when off,
and an end-to-end run.
"""

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.schedule import ParameterServer, make_grant
from opendipaco.schedule.compress import apply_state_delta, encode_state_delta
from opendipaco.topology import is_private_key


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _maxdiff(a: dict, b: dict) -> float:
    return max((a[n].float() - b[n].float()).abs().max().item() for n in a)


# -- codec ---------------------------------------------------------------------


def test_state_delta_roundtrip_within_int8_bound():
    """apply_state_delta(base, encode(cur, base)) reconstructs cur to within one
    int8 step; non-floating tensors are carried verbatim (exact)."""
    base = {"w": torch.randn(8, 8), "n": torch.tensor([3, 1, 4], dtype=torch.long)}
    cur = {"w": base["w"] + 0.01 * torch.randn(8, 8),
           "n": torch.tensor([3, 1, 4], dtype=torch.long)}
    recon = apply_state_delta(base, encode_state_delta(cur, base))
    step = (cur["w"] - base["w"]).abs().max().item() / 127.0
    assert (recon["w"] - cur["w"]).abs().max().item() <= step          # within a step
    assert torch.equal(recon["n"], cur["n"])                           # int verbatim, exact
    # The delta recovers nearly all of the change vs. doing nothing (the keyframe).
    assert _maxdiff(recon, cur) < _maxdiff(base, cur) / 50


# -- owner full/delta decision -------------------------------------------------


def _push(ps, key, path, tok, scale=0.01):
    grad = {"grad": [scale * torch.ones_like(p) for p in ps.bank[key].parameters()]}
    ps._push({"grant": make_grant(path, [key], 1.0, tok), "updates": {key: grad}})


def test_owner_ships_delta_then_falls_back_to_full_when_keyframe_ages_out():
    cfg = _cfg()
    keys = sorted(cfg.build_topology().module_keys())
    shared = next(k for k in keys if not is_private_key(k))
    path = cfg.build_topology().path_from_index(0)
    # down="delta" defaults the version ring to depth 8.
    ps = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=2), host="127.0.0.1",
                         port=0, down="delta")
    try:
        # Keyframe at (0,0): the full ship a fresh worker gets.
        full = ps._fetch({"keys": [shared], "have": {}})
        assert "__delta__" not in full["weights"][shared]
        keyframe = full["weights"][shared]
        assert tuple(full["versions"][shared]) == (0, 0)

        _push(ps, shared, path, "t1")                  # (0,0) -> (0,1), ring keeps (0,0)
        reply = ps._fetch({"keys": [shared], "have": {shared: [0, 0]}})
        payload = reply["weights"][shared]
        assert "__delta__" in payload and tuple(payload["base"]) == (0, 0)   # a delta!
        recon = apply_state_delta(keyframe, payload["tensors"])
        cur = {n: v for n, v in ps.bank[shared].state_dict().items()}
        assert _maxdiff(recon, cur) < _maxdiff(keyframe, cur) / 20          # reconstructs

        # Push past the ring depth (8): (0,0) ages out -> full fallback (new keyframe).
        for i in range(9):
            _push(ps, shared, path, f"a{i}")
        reply = ps._fetch({"keys": [shared], "have": {shared: [0, 0]}})
        assert "__delta__" not in reply["weights"][shared]                  # full fallback
    finally:
        ps.shutdown()


def test_down_full_never_ships_a_delta():
    """down="full" (default) is byte-identical to the pre-W2 path: a stale fetch
    always gets full weights, never a delta payload."""
    cfg = _cfg()
    keys = sorted(cfg.build_topology().module_keys())
    shared = next(k for k in keys if not is_private_key(k))
    path = cfg.build_topology().path_from_index(0)
    ps = ParameterServer(cfg, keys, DiLoCoConfig(inner_steps=2), host="127.0.0.1",
                         port=0, version_history=8)   # ring present, but down="full"
    try:
        _push(ps, shared, path, "t1")
        reply = ps._fetch({"keys": [shared], "have": {shared: [0, 0]}})
        assert "__delta__" not in reply["weights"][shared]                  # always full
    finally:
        ps.shutdown()


# -- end to end ----------------------------------------------------------------


def test_run_local_sharded_trains_with_delta_down():
    """A full sharded cluster with transport.down="delta" trains to budget: if
    reconstruction were wrong the workers would train from garbage and the run
    would not complete with finite loss."""
    from opendipaco.launch import run_local
    from opendipaco.launch.config import LaunchConfig

    cfg = LaunchConfig.from_dict({
        "mode": "sharded",
        "model": {"vocab_size": 64, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "max_position_embeddings": 64,
                  "layers_per_level": [1, 1], "level_sizes": [2, 2], "sequence_length": 16},
        "diloco": {"inner_steps": 4, "inner_lr": 1e-3},
        "data": {"source": "synthetic", "num_documents": 64},
        "transport": {"compress": "int8", "down": "delta"},
        "sharded": {"num_shards": 2, "parameter_servers": [["127.0.0.1", 0], ["127.0.0.1", 0]]},
        "run": {"generations": 2, "batch_size": 8, "local_workers": 2},
    })
    server, completed = run_local(cfg)
    assert sum(completed.values()) >= 2 * cfg.model.level_sizes[0] * cfg.model.level_sizes[1]
    assert server.metrics.accepted_updates > 0
