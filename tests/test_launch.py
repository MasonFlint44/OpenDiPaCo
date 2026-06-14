"""Tests for the config-driven launch CLI.

Cover config parsing/validation, the section->core-dataclass builders, corpus
building, TLS/auth wiring, the ``init-config`` / ``gen-cert`` CLI commands, and --
the payoff -- an all-in-one ``run_local`` that trains end-to-end in both coordinator
and sharded modes through the same config path the CLI uses.
"""

import json

import pytest
import torch

from opendipaco.data import ShardedCorpus
from opendipaco.launch import (
    LaunchConfig,
    build_corpus,
    build_documents,
    dipaco_config,
    diloco_config,
    load_config,
    run_local,
)
from opendipaco.launch.cli import main
from opendipaco.launch.config import backbone_config


def _tiny_dict(mode="coordinator"):
    d = {
        "mode": mode,
        "model": {"vocab_size": 64, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "max_position_embeddings": 64,
                  "layers_per_level": [1, 1], "level_sizes": [2, 2], "sequence_length": 16},
        "diloco": {"inner_steps": 4, "inner_lr": 0.001},
        "data": {"source": "synthetic", "num_documents": 64},
        "transport": {"host": "127.0.0.1", "port": 0},
        "run": {"generations": 2, "batch_size": 8, "local_workers": 2},
    }
    if mode == "sharded":
        d["sharded"] = {"num_shards": 2}
    return d


# -- config parsing / validation ---------------------------------------------


def test_from_dict_defaults_and_sections():
    cfg = LaunchConfig.from_dict({})
    assert cfg.mode == "coordinator"
    assert cfg.model.level_sizes == [4, 4] and cfg.run.generations == 10


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        LaunchConfig.from_dict({"model": {"hidden_size": 8, "bogus": 1}})
    with pytest.raises(ValueError):
        LaunchConfig.from_dict({"nonsense": {}})


def test_from_dict_rejects_bad_mode():
    with pytest.raises(ValueError):
        LaunchConfig.from_dict({"mode": "potato"})


def test_transport_kind_parses_and_validates():
    """W1d: transport.kind selects the substrate (tcp default, libp2p opt-in)."""
    assert LaunchConfig.from_dict({}).transport.kind == "tcp"
    lp = LaunchConfig.from_dict({"transport": {"kind": "libp2p",
                                               "relays": ["/ip4/1.2.3.4/tcp/9/p2p/Qm"],
                                               "connect_libp2p": "/ip4/1.2.3.4/tcp/9/p2p/Qm"}})
    assert lp.transport.kind == "libp2p"
    assert lp.transport.relays == ["/ip4/1.2.3.4/tcp/9/p2p/Qm"]
    assert lp.transport.dcutr is True
    with pytest.raises(ValueError):
        LaunchConfig.from_dict({"transport": {"kind": "carrier-pigeon"}})


def test_transport_down_parses_and_validates():
    """W2a: transport.down selects the weights downlink policy (full default)."""
    assert LaunchConfig.from_dict({}).transport.down == "full"
    assert LaunchConfig.from_dict({"transport": {"down": "delta"}}).transport.down == "delta"
    with pytest.raises(ValueError):
        LaunchConfig.from_dict({"transport": {"down": "gzip"}})


def test_libp2p_routes_gate():
    """W1d: libp2p routing is wired for static sharded mode; rendezvous keeps
    routing over TCP (epoch records carry TCP addrs) until multiaddr discovery
    lands, and tcp mode never routes over libp2p."""
    from opendipaco.launch.roles import _libp2p_routes

    assert _libp2p_routes(LaunchConfig.from_dict({})) is False         # tcp default
    static = LaunchConfig.from_dict({"transport": {"kind": "libp2p"}})
    assert _libp2p_routes(static) is True
    rdv = LaunchConfig.from_dict({"transport": {"kind": "libp2p"},
                                  "ownership": {"mode": "rendezvous"}})
    assert _libp2p_routes(rdv) is False                                # routes over TCP


def test_scheduler_keeps_multiaddr_ps_addrs(tmp_path):
    """W1d Fix: a libp2p scheduler's PS addresses are multiaddr strings; routing
    must pass them through, not tuple() them into per-character shards."""
    from opendipaco.schedule import Scheduler

    cfg = LaunchConfig.from_dict(_tiny_dict("sharded"))
    model = dipaco_config(cfg.model)
    docs = [torch.randint(0, cfg.model.vocab_size, (8,)) for _ in range(8)]
    assign = torch.tensor([i % model.num_paths for i in range(len(docs))])
    corpus = ShardedCorpus.from_assignments(docs, assign, model.num_paths, 8)
    ma = ["/ip4/127.0.0.1/tcp/4001/p2p/Qm1", "/ip4/127.0.0.1/tcp/4002/p2p/Qm2"]
    sched = Scheduler(model, corpus, ma, diloco_config(cfg.diloco), batch_size=4,
                      host="127.0.0.1", port=0)
    assert sched.ps_addrs == ma                       # strings intact, not tupled
    assert all(v[0] in ma for v in sched._routing.values())


def test_load_config_json_and_yaml(tmp_path):
    d = _tiny_dict()
    jp = tmp_path / "c.json"
    jp.write_text(json.dumps(d))
    cfg = load_config(jp)
    assert cfg.model.vocab_size == 64 and cfg.transport.port == 0

    yaml = pytest.importorskip("yaml")
    yp = tmp_path / "c.yaml"
    yp.write_text(yaml.safe_dump(d))
    assert load_config(yp).model.sequence_length == 16


def test_connect_addr_resolves_bind_wildcard():
    cfg = LaunchConfig.from_dict({"transport": {"host": "0.0.0.0", "port": 1234}})
    assert cfg.connect_addr() == ("127.0.0.1", 1234)
    cfg2 = LaunchConfig.from_dict({"transport": {"host": "0.0.0.0", "connect_host": "node-7"}})
    assert cfg2.connect_addr()[0] == "node-7"


# -- builders ----------------------------------------------------------------


def test_builders_produce_core_configs():
    cfg = LaunchConfig.from_dict(_tiny_dict())
    bb = backbone_config(cfg.model)
    assert bb.vocab_size == 64 and bb.layers_per_level == [1, 1]
    dp = dipaco_config(cfg.model)
    assert dp.num_paths == 4 and dp.sequence_length == 16
    dl = diloco_config(cfg.diloco)
    assert dl.inner_steps == 4 and dl.inner_lr == 0.001


def test_build_corpus_round_robin_and_kmeans():
    cfg = LaunchConfig.from_dict(_tiny_dict())
    model = dipaco_config(cfg.model)
    docs = build_documents(cfg)
    assert docs and all(torch.is_tensor(d) for d in docs)
    rr = build_corpus(LaunchConfig.from_dict({**_tiny_dict(), "data":
                      {"source": "synthetic", "num_documents": 64, "routing": "round_robin"}}),
                      model, docs)
    km = build_corpus(cfg, model, docs)
    assert isinstance(rr, ShardedCorpus) and isinstance(km, ShardedCorpus)


def test_tls_context_builders_from_config(tmp_path):
    from opendipaco.launch.roles import build_tls_client, build_tls_server
    from opendipaco.schedule import generate_selfsigned_cert
    cert, key = generate_selfsigned_cert(str(tmp_path))
    off = LaunchConfig.from_dict({})
    assert build_tls_server(off) is None and build_tls_client(off) is None
    on = LaunchConfig.from_dict({"tls": {"enabled": True, "certfile": cert, "keyfile": key,
                                         "cafile": cert}})
    assert build_tls_server(on) is not None and build_tls_client(on) is not None


# -- CLI utility commands ----------------------------------------------------


def test_init_config_roundtrips(tmp_path):
    out = tmp_path / "gen.json"
    assert main(["init-config", "--out", str(out), "--mode", "sharded", "--format", "json"]) == 0
    cfg = load_config(out)
    assert cfg.mode == "sharded" and len(cfg.sharded.parameter_servers) == 2


def test_gen_cert_cli(tmp_path):
    assert main(["gen-cert", "--out", str(tmp_path / "certs")]) == 0
    assert (tmp_path / "certs" / "cert.pem").exists()


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])


# -- end-to-end: the whole cluster, in-process -------------------------------


def _target(cfg):
    return dipaco_config(cfg.model).num_paths * cfg.run.generations


def test_run_local_coordinator_trains():
    cfg = LaunchConfig.from_dict(_tiny_dict("coordinator"))
    server, completed = run_local(cfg)
    assert sum(completed.values()) >= _target(cfg)
    assert server.metrics.accepted_updates > 0


def test_run_local_sharded_trains():
    cfg = LaunchConfig.from_dict(_tiny_dict("sharded"))
    scheduler, completed = run_local(cfg)
    assert sum(completed.values()) >= _target(cfg)
    assert not hasattr(scheduler, "bank")  # the scheduler holds no weights


def test_robustness_config_parses_and_defaults_off():
    cfg = LaunchConfig.from_dict(_tiny_dict())
    assert cfg.robustness.mode == "off"          # default: no behavior change
    assert cfg.robustness.private_policy == "overwrite"
    cfg2 = LaunchConfig.from_dict({**_tiny_dict(), "robustness": {
        "mode": "on", "aggregate": "median", "redundancy_rate": 0.2,
        "private_policy": "proposal"}})
    assert cfg2.robustness.mode == "on" and cfg2.robustness.aggregate == "median"
    assert cfg2.robustness.redundancy_rate == 0.2


def test_proposal_policy_rejects_guaranteed_stall_config():
    import pytest
    # redundancy < 2: no checker can corroborate -> private modules would freeze.
    with pytest.raises(ValueError):
        LaunchConfig.from_dict({**_tiny_dict(), "robustness": {
            "private_policy": "proposal", "redundancy": 1}})
    # quorum above the replica count: a proposal could never reach quorum.
    with pytest.raises(ValueError):
        LaunchConfig.from_dict({**_tiny_dict(), "robustness": {
            "private_policy": "proposal", "redundancy": 2, "private_quorum": 3}})
    # A sane proposal config is accepted.
    cfg = LaunchConfig.from_dict({**_tiny_dict(), "robustness": {
        "private_policy": "proposal", "redundancy": 3, "private_quorum": 2}})
    assert cfg.robustness.private_policy == "proposal"


def test_schedule_config_parses_and_defaults_central():
    cfg = LaunchConfig.from_dict(_tiny_dict())
    assert cfg.schedule.mode == "central"        # default: today's single scheduler
    assert cfg.schedule.lease_ttl == 30.0 and cfg.schedule.read_quorum == 2
    cfg2 = LaunchConfig.from_dict({**_tiny_dict(), "ownership": {"mode": "rendezvous"},
                                   "schedule": {"mode": "decentralized", "lease_ttl": 12.0}})
    assert cfg2.schedule.mode == "decentralized" and cfg2.schedule.lease_ttl == 12.0


def test_decentralized_requires_rendezvous_ownership():
    # Built on the replicated owner tier -> needs rendezvous ownership; the
    # default static ownership has no owners to mint grants, so reject at load.
    with pytest.raises(ValueError):
        LaunchConfig.from_dict({**_tiny_dict(), "schedule": {"mode": "decentralized"}})
    with pytest.raises(ValueError):
        LaunchConfig.from_dict({**_tiny_dict(), "schedule": {"mode": "potato"}})
    # The matching pair is accepted.
    ok = LaunchConfig.from_dict({**_tiny_dict(), "ownership": {"mode": "rendezvous"},
                                 "schedule": {"mode": "decentralized"}})
    assert ok.schedule.mode == "decentralized"


def test_run_local_sharded_with_robustness_on():
    """`opendipaco run` (sharded) with robustness on: owner-side robust
    aggregation + reputation gates engaged, run still reaches its budget."""
    d = _tiny_dict("sharded")
    d["robustness"] = {"mode": "on", "quorum_target": 2, "quorum_timeout": 0.5}
    cfg = LaunchConfig.from_dict(d)
    scheduler, completed = run_local(cfg)
    assert sum(completed.values()) >= _target(cfg)
    assert scheduler.reputation is not None       # the gate substrate is live


def test_decentralized_owner_kw_built_only_in_decentralized_mode():
    from opendipaco.launch.roles import _decentralized_owner_kw
    assert _decentralized_owner_kw(LaunchConfig.from_dict(_tiny_dict())) == {}  # central
    cfg = LaunchConfig.from_dict({**_tiny_dict("sharded"),
                                  "ownership": {"mode": "rendezvous", "k": 3},
                                  "schedule": {"mode": "decentralized", "read_quorum": 2}})
    kw = _decentralized_owner_kw(cfg)
    assert kw["schedule_mode"] == "decentralized" and kw["k"] == 3 and kw["read_quorum"] == 2
    assert kw["reputation"] is not None and kw["rate_limiter"] is not None


def test_run_local_rejects_decentralized_with_a_pointer():
    from opendipaco.launch import run_local
    cfg = LaunchConfig.from_dict({**_tiny_dict("sharded"),
                                  "ownership": {"mode": "rendezvous"},
                                  "schedule": {"mode": "decentralized"}})
    with pytest.raises(ValueError, match="decentralized"):
        run_local(cfg)


def test_advertise_host_defaults_to_bind_host():
    """A rendezvous owner's tracker record must advertise a dialable address:
    explicit ownership.advertise_host, else transport.connect_host, else the
    bind host -- only a wildcard bind falls back to loopback (Codex P2)."""
    from opendipaco.launch.roles import _advertise_host

    assert _advertise_host(LaunchConfig.from_dict(
        {"transport": {"host": "203.0.113.7"}})) == "203.0.113.7"
    assert _advertise_host(LaunchConfig.from_dict(
        {"transport": {"host": "0.0.0.0"}})) == "127.0.0.1"
    assert _advertise_host(LaunchConfig.from_dict(
        {"transport": {"host": "0.0.0.0", "connect_host": "owner.example"}})) == "owner.example"
    assert _advertise_host(LaunchConfig.from_dict(
        {"transport": {"host": "203.0.113.7", "connect_host": "owner.example"},
         "ownership": {"advertise_host": "advertise.example"}})) == "advertise.example"
