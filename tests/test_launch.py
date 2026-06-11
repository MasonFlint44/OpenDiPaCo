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
