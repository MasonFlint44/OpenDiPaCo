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
                          "grant_key": "g", "scheduler_pub": "abcd"},
               data={"source": "synthetic", "num_documents": 32,
                     "shard_cache_dir": "/home/operator/shards", "cache_path": "/op/c4.pt"})
    m = build_manifest(cfg)
    t = m["config"]["transport"]
    assert t["auth_key"] is None and t["grant_key"] is None        # secrets stripped
    assert t["scheduler_pub"] == "abcd"                            # public key kept (grants)
    assert m["config"]["model"]["hidden_size"] == 32               # model carried
    # Operator-local filesystem paths stripped (meaningless off-box + a TOFU
    # write-sink); the joiner supplies its own via --data-dir.
    assert m["config"]["data"]["shard_cache_dir"] is None
    assert m["config"]["data"]["cache_path"] is None


def test_manifest_strips_max_mbps():
    # max_mbps is the volunteer's own --max-mbps, not a run property; carrying the
    # operator's value would silently cap joiners.
    m = build_manifest(_cfg(transport={"host": "127.0.0.1", "port": 0, "auth_key": "s",
                                        "max_mbps": 5.0}))
    assert m["config"]["transport"]["max_mbps"] is None


def test_manifest_strips_worker_resource_knobs(tmp_path):
    # worker_max_batch / worker_max_shards describe the JOINER's hardware budget,
    # not the run (W7a): a joiner that omits its flags must fall back to its own
    # default, never inherit the operator's value. And a stripped manifest must
    # still rebuild into a valid config (worker_max_shards -> None -> default).
    m = build_manifest(_cfg(run={"generations": 2, "batch_size": 8,
                                 "worker_max_batch": 4, "worker_max_shards": 32}))
    assert m["config"]["run"]["worker_max_batch"] is None
    assert m["config"]["run"]["worker_max_shards"] is None
    cfg = manifest_to_config(m)                       # round-trips without crashing
    assert cfg.run.worker_max_shards is None          # default applies at the worker
    # An explicit joiner override still wins over the (stripped) base.
    cfg2 = manifest_to_config(m, overrides={"run": {"worker_max_shards": 2}})
    assert cfg2.run.worker_max_shards == 2


def test_config_rejects_bad_max_mbps():
    base = {"mode": "sharded",
            "sharded": {"num_shards": 2, "parameter_servers": [["127.0.0.1", 1], ["127.0.0.1", 2]]}}
    for bad in (0, -5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_mbps"):
            LaunchConfig.from_dict({**base, "transport": {"max_mbps": bad}})
    ok = LaunchConfig.from_dict({**base, "transport": {"max_mbps": 5.0}})
    assert ok.transport.max_mbps == 5.0


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


def test_forced_unavailable_accelerator_errors_early():
    import torch
    cfg = dipaco_config(_cfg().model)
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="cuda"):
            resolve_device(cfg, requested="cuda", batch_size=8, seq_len=16)
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        with pytest.raises(SystemExit, match="mps"):
            resolve_device(cfg, requested="mps", batch_size=8, seq_len=16)
    # An unrecognized device (typo) is rejected up front, not handed to torch.
    with pytest.raises(SystemExit, match="not recognized"):
        resolve_device(cfg, requested="gpu", batch_size=8, seq_len=16)


def test_join_rejects_a_bad_identity_path():
    with pytest.raises(SystemExit, match="could not load --identity"):
        run_join(scheduler="127.0.0.1:1", identity_key="/no/such/key.pem", quiet=True)


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


def test_manifest_fetch_authenticates_with_identity():
    """Per-peer-auth deployment (admitted_peers, no shared auth_key): the manifest
    fetch must present the worker's Ed25519 identity, not the (absent) HMAC key --
    otherwise an identity-auth volunteer can never even fetch the manifest."""
    cfg = _cfg()
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    corpus = build_corpus(cfg, model, build_documents(cfg))
    worker_id, sched_id = PeerIdentity.generate(), PeerIdentity.generate()
    sched = Scheduler(model, corpus, [("127.0.0.1", 1)], diloco, batch_size=8,
                      host="127.0.0.1", port=0, identity=sched_id,
                      admitted_peers=[worker_id])           # identity-gated, no auth_key
    sched.start()
    try:
        sched.serve_manifest(build_manifest(cfg, identity=sched_id))
        # The admitted identity fetches successfully...
        m = fetch_manifest(("127.0.0.1", sched.port), auth_key=worker_id)
        assert verify_manifest(m, server_pub=sched_id.public_key_hex)
        # ...while the HMAC/None credential (what join sent before the fix) is refused.
        with pytest.raises((OSError, SystemExit)):
            fetch_manifest(("127.0.0.1", sched.port), auth_key=None)
    finally:
        sched.shutdown()


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


def test_bandwidth_bucket_meters_the_worker_traffic():
    """The W6b throttle is wired into the worker's real sockets: training through
    run_worker_role with a shared bucket tallies bytes sent AND received."""
    from opendipaco.launch.roles import run_worker_role
    from opendipaco.schedule.throttle import TokenBucket
    cfg = _cfg()
    sched, pss = _start_sharded(cfg)
    fit = threading.Thread(target=lambda: sched.fit(num_generations=2, total_generations=2),
                           daemon=True)
    fit.start()
    bucket = TokenBucket(1e9)                                   # huge rate: meter, don't slow
    wcfg = manifest_to_config(build_manifest(cfg), overrides={
        "transport": {"connect_host": "127.0.0.1", "port": sched.port, "auth_key": "shh"},
        "run": {"device": "cpu"}})
    worker = threading.Thread(target=run_worker_role,
                              kwargs=dict(cfg=wcfg, max_tasks=16, bucket=bucket), daemon=True)
    worker.start()
    try:
        fit.join(timeout=60)
        assert not fit.is_alive()
        assert bucket.sent_bytes > 0 and bucket.recv_bytes > 0  # both directions metered
    finally:
        sched.shutdown()
        for ps in pss:
            ps.shutdown()
        worker.join(timeout=10)


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
