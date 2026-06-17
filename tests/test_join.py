"""Tests for W6 slice a — the consumer client (`opendipaco join`).

Covers the run manifest (build strips secrets / verify / pin / fingerprint /
rebuild), device resolution, the manifest RPC served by a real scheduler, and an
end-to-end `run_join` that fetches the manifest and trains against an in-process
sharded run with no config file. Design: docs/w6-client-design.md.
"""

import threading

import pytest

from opendipaco.launch import LaunchConfig, build_corpus, build_documents, dipaco_config, diloco_config
from opendipaco.launch.client import fetch_manifest, resolve_device, run_join
from opendipaco.launch.manifest import (
    build_manifest,
    manifest_fingerprint,
    manifest_to_config,
    verify_manifest,
)
from opendipaco.schedule import ParameterServer, PeerIdentity, Scheduler, assign_shards


def _cfg(**over):
    d = {
        "mode": "sharded",
        "model": {"vocab_size": 64, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "max_position_embeddings": 64,
                  "layers_per_level": [1, 1], "level_sizes": [2, 2], "sequence_length": 16},
        "diloco": {"inner_steps": 2, "inner_lr": 0.01},
        "data": {"source": "synthetic", "num_documents": 32},
        "transport": {"host": "127.0.0.1", "port": 0, "auth_key": "shh", "grant_key": "g"},
        "run": {"generations": 2, "batch_size": 8},
        "sharded": {"num_shards": 2},
    }
    d.update(over)
    return LaunchConfig.from_dict(d)


# -- manifest: build / verify / pin / fingerprint / rebuild --------------------


def test_build_manifest_strips_secrets_keeps_public():
    cfg = _cfg(transport={"host": "127.0.0.1", "port": 0, "auth_key": "shh",
                          "grant_key": "g", "scheduler_pub": "abcd"})
    m = build_manifest(cfg)
    t = m["config"]["transport"]
    assert t["auth_key"] is None and t["grant_key"] is None        # secrets stripped
    assert t["scheduler_pub"] == "abcd"                            # public key kept (grants)
    assert m["config"]["model"]["hidden_size"] == 32               # model carried


def test_verify_manifest_signed_unsigned_and_pinned():
    cfg = _cfg()
    idn = PeerIdentity.generate()
    signed = build_manifest(cfg, identity=idn)
    unsigned = build_manifest(cfg)
    assert verify_manifest(unsigned)                               # TOFU accepts unsigned
    assert verify_manifest(signed)                                 # and a valid signature
    assert verify_manifest(signed, server_pub=idn.public_key_hex)  # pin matches
    assert not verify_manifest(signed, server_pub=PeerIdentity.generate().public_key_hex)
    assert not verify_manifest(unsigned, server_pub=idn.public_key_hex)  # pin needs a sig
    tampered = dict(signed, config=dict(signed["config"], mode="coordinator"))
    assert not verify_manifest(tampered)                           # broken signature refused
    assert not verify_manifest({"kind": "nope"})


def test_fingerprint_is_content_stable_across_resign():
    cfg = _cfg()
    a = build_manifest(cfg, identity=PeerIdentity.generate())
    b = build_manifest(cfg, identity=PeerIdentity.generate())   # different signer, same config
    assert manifest_fingerprint(a) == manifest_fingerprint(b)
    assert manifest_fingerprint(build_manifest(_cfg(run={"generations": 9}))) != manifest_fingerprint(a)


def test_manifest_to_config_applies_overrides_and_validates():
    m = build_manifest(_cfg())
    cfg = manifest_to_config(m, overrides={
        "transport": {"connect_host": "node-7", "port": 4321, "auth_key": "mine"},
        "run": {"device": "cpu", "max_tasks": 3},
        "diloco": {"inner_autocast": True}})
    assert cfg.connect_addr() == ("node-7", 4321)
    assert cfg.transport.auth_key == "mine" and cfg.run.device == "cpu"
    assert cfg.diloco.inner_autocast is True
    # None overrides are ignored (don't clobber a manifest value with a missing flag).
    cfg2 = manifest_to_config(m, overrides={"run": {"device": None}})
    assert cfg2.run.device == _cfg().run.device


# -- device resolution ---------------------------------------------------------


def test_resolve_device_cpu_path():
    cfg = dipaco_config(_cfg().model)
    dev, over, notes = resolve_device(cfg, requested="cpu", batch_size=8, seq_len=16)
    assert dev == "cpu" and over == {} and any("cpu" in n for n in notes)


def test_resolve_device_autodetect_falls_to_cpu_without_gpu():
    import torch
    if torch.cuda.is_available():
        pytest.skip("GPU present; this checks the no-GPU fallback")
    cfg = dipaco_config(_cfg().model)
    dev, _, _ = resolve_device(cfg, requested=None, batch_size=8, seq_len=16)
    assert dev == "cpu"


# -- the manifest RPC, served by a real scheduler ------------------------------


def _start_sharded(cfg):
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    corpus = build_corpus(cfg, model, build_documents(cfg))
    keys = model.build_topology().module_keys()
    shards = [[k for k, s in assign_shards(keys, 2).items() if s == i] for i in range(2)]
    pss = [ParameterServer(model, sk, diloco, host="127.0.0.1", port=0, auth_key="shh")
           for sk in shards]
    for ps in pss:
        ps.start()
    sched = Scheduler(model, corpus, [("127.0.0.1", ps.port) for ps in pss], diloco,
                      batch_size=8, host="127.0.0.1", port=0, auth_key="shh")
    sched.start()
    return sched, pss


def test_manifest_rpc_roundtrips_through_the_scheduler():
    cfg = _cfg()
    sched, pss = _start_sharded(cfg)
    try:
        sched.serve_manifest(build_manifest(cfg, identity=PeerIdentity.generate()))
        fetched = fetch_manifest(("127.0.0.1", sched.port), auth_key="shh")
        assert verify_manifest(fetched)
        rebuilt = manifest_to_config(fetched)
        assert rebuilt.model.hidden_size == cfg.model.hidden_size
        assert rebuilt.transport.auth_key is None                  # secret didn't cross
    finally:
        sched.shutdown()
        for ps in pss:
            ps.shutdown()


def test_fetch_manifest_raises_when_none_published():
    cfg = _cfg()
    sched, pss = _start_sharded(cfg)
    try:
        with pytest.raises(SystemExit, match="no run manifest"):
            fetch_manifest(("127.0.0.1", sched.port), auth_key="shh")
    finally:
        sched.shutdown()
        for ps in pss:
            ps.shutdown()


# -- end to end: join a run with no config file --------------------------------


def test_run_join_trains_against_an_in_process_run():
    """A flags-only `join` fetches the manifest, autodetects cpu, builds its
    config, and trains real tasks against the scheduler -- no config file."""
    cfg = _cfg()
    sched, pss = _start_sharded(cfg)
    sched.serve_manifest(build_manifest(cfg))                      # unsigned -> TOFU
    fit = threading.Thread(target=lambda: sched.fit(num_generations=2, total_generations=2),
                           daemon=True)
    fit.start()
    join = threading.Thread(target=run_join, kwargs=dict(
        scheduler=f"127.0.0.1:{sched.port}", auth_key="shh", max_tasks=16,
        device="cpu", quiet=True), daemon=True)
    join.start()
    try:
        fit.join(timeout=60)
        assert not fit.is_alive()                                  # the run reached its budget
        assert sum(sched._completed.values()) > 0                  # the join worker did the work
    finally:
        sched.shutdown()
        for ps in pss:
            ps.shutdown()
        join.join(timeout=10)
