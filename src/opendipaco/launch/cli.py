"""``opendipaco`` command-line launcher.

Subcommands (each reads a cluster config; see ``init-config``):

    opendipaco run         --config c.yaml          # all-in-one local cluster
    opendipaco coordinator --config c.yaml          # single-node async coordinator
    opendipaco scheduler   --config c.yaml          # sharded scheduler (no weights)
    opendipaco ps          --config c.yaml --shard-id N
    opendipaco worker      --config c.yaml [--max-tasks N]
    opendipaco ingest      --config c.yaml --shard-id N   # sharded resumable ingest
    opendipaco tracker     --config c.yaml           # rendezvous directory (Phase 1)
    opendipaco relay       --config c.yaml           # libp2p Circuit Relay v2 (NAT traversal)
    opendipaco init-config --out c.yaml [--mode sharded]  # write a starter config
    opendipaco gen-cert    --out certs/              # self-signed cert for TLS
    opendipaco gen-identity --out peer.pem           # Ed25519 peer identity

``run`` is the quickest way to see a cluster work end-to-end on one box; the per-role
commands are what you launch across hosts (same config file everywhere).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .config import DataCfg, DiLoCoCfg, LaunchConfig, ModelCfg, RunCfg, ShardedCfg, load_config
from .roles import (
    run_coordinator,
    run_ingest,
    run_local,
    run_parameter_server,
    run_relay,
    run_scheduler,
    run_tracker,
    run_worker_role,
)


def _report(server, completed) -> None:
    print("per-path updates:", completed)
    print(server.metrics.report())


def cmd_run(args) -> int:
    server, completed = run_local(load_config(args.config))
    _report(server, completed)
    return 0


def cmd_coordinator(args) -> int:
    server, completed = run_coordinator(load_config(args.config))
    _report(server, completed)
    return 0


def cmd_scheduler(args) -> int:
    server, completed = run_scheduler(load_config(args.config))
    _report(server, completed)
    return 0


def cmd_ps(args) -> int:
    cfg = load_config(args.config)
    print(f"parameter server shard {args.shard_id}/{cfg.sharded.num_shards} "
          f"on {cfg.transport.host}; Ctrl-C to stop", flush=True)
    run_parameter_server(cfg, args.shard_id)
    return 0


def cmd_worker(args) -> int:
    run_worker_role(load_config(args.config), max_tasks=args.max_tasks)
    return 0


def cmd_ingest(args) -> int:
    docs = run_ingest(load_config(args.config), args.shard_id)
    print(f"ingested {len(docs)} documents for shard {args.shard_id}")
    return 0


def cmd_init_config(args) -> int:
    # A small, fast-to-run starter: `opendipaco run` on it finishes in seconds.
    # Scale model / data / generations up (and set run.device=cuda) for real training.
    cfg = LaunchConfig(
        mode=args.mode,
        model=ModelCfg(vocab_size=4096, hidden_size=128, num_attention_heads=4,
                       intermediate_size=256, max_position_embeddings=128,
                       layers_per_level=[1, 1], level_sizes=[2, 2], sequence_length=64),
        diloco=DiLoCoCfg(inner_steps=10),
        data=DataCfg(source="synthetic", num_documents=512),
        run=RunCfg(generations=3, batch_size=16, local_workers=2),
    )
    if args.mode == "sharded":
        cfg.sharded = ShardedCfg(num_shards=2,
                                 parameter_servers=[["127.0.0.1", 29501], ["127.0.0.1", 29502]])
    data = dataclasses.asdict(cfg)
    out = Path(args.out)
    fmt = args.format or {".yaml": "yaml", ".yml": "yaml", ".json": "json"}.get(out.suffix, "yaml")
    if fmt == "json":
        import json
        out.write_text(json.dumps(data, indent=2) + "\n")
    else:
        try:
            import yaml
        except ImportError:  # pragma: no cover
            print("PyYAML not installed; use --format json", file=sys.stderr)
            return 1
        header = ("# opendipaco starter config — small and fast to run; scale model/data/\n"
                  "# generations up (and set run.device=cuda) for real training.\n")
        out.write_text(header + yaml.safe_dump(data, sort_keys=False))
    print(f"wrote {args.mode} config to {out}")
    return 0


def cmd_gen_cert(args) -> int:
    from ..schedule import generate_selfsigned_cert
    certfile, keyfile = generate_selfsigned_cert(args.out, hosts=args.hosts.split(","))
    print(f"cert: {certfile}\nkey:  {keyfile}")
    return 0


def cmd_relay(args) -> int:
    cfg = load_config(args.config)
    print(f"libp2p relay on {cfg.transport.libp2p_listen}; Ctrl-C to stop", flush=True)
    run_relay(cfg)
    return 0


def cmd_tracker(args) -> int:
    cfg = load_config(args.config)
    print(f"tracker on {cfg.tracker.host}:{cfg.tracker.port} "
          f"(ttl={cfg.tracker.ttl}s, open_enrollment={cfg.tracker.open_enrollment}); "
          f"Ctrl-C to stop", flush=True)
    run_tracker(cfg)
    return 0


def cmd_gen_identity(args) -> int:
    from ..schedule import PeerIdentity
    ident = PeerIdentity.generate()
    path = ident.save(args.out)
    print(f"key:     {path}")
    print(f"peer id: {ident.peer_id}")
    print(f"pubkey:  {ident.public_key_hex}")
    print("set transport.identity_key to the key path on this peer; add the pubkey")
    print("to transport.admitted_peers (servers) / tracker.enroll_peers (tracker).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opendipaco", description="DiPaCo cluster launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(name, help):
        p = sub.add_parser(name, help=help)
        p.add_argument("--config", required=True, help="cluster config (.yaml/.toml/.json)")
        return p

    with_config("run", "stand up the whole cluster locally").set_defaults(func=cmd_run)
    with_config("coordinator", "run a single-node coordinator").set_defaults(func=cmd_coordinator)
    with_config("scheduler", "run the sharded scheduler").set_defaults(func=cmd_scheduler)

    p_ps = with_config("ps", "run one parameter-server shard")
    p_ps.add_argument("--shard-id", type=int, required=True)
    p_ps.set_defaults(func=cmd_ps)

    p_w = with_config("worker", "run a worker")
    p_w.add_argument("--max-tasks", type=int, default=None)
    p_w.set_defaults(func=cmd_worker)

    p_in = with_config("ingest", "resumably ingest a data shard")
    p_in.add_argument("--shard-id", type=int, required=True)
    p_in.set_defaults(func=cmd_ingest)

    with_config("tracker", "run the rendezvous directory").set_defaults(func=cmd_tracker)
    with_config("relay", "run a libp2p Circuit Relay v2 relay (NAT traversal)").set_defaults(
        func=cmd_relay)

    p_id = sub.add_parser("gen-identity", help="generate an Ed25519 peer identity")
    p_id.add_argument("--out", required=True, help="path for the private-key PEM")
    p_id.set_defaults(func=cmd_gen_identity)

    p_init = sub.add_parser("init-config", help="write a starter config file")
    p_init.add_argument("--out", required=True)
    p_init.add_argument("--mode", choices=["coordinator", "sharded"], default="coordinator")
    p_init.add_argument("--format", choices=["yaml", "json"], default=None)
    p_init.set_defaults(func=cmd_init_config)

    p_cert = sub.add_parser("gen-cert", help="generate a self-signed TLS cert (dev)")
    p_cert.add_argument("--out", required=True, help="directory to write cert.pem/key.pem")
    p_cert.add_argument("--hosts", default="localhost,127.0.0.1")
    p_cert.set_defaults(func=cmd_gen_cert)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
