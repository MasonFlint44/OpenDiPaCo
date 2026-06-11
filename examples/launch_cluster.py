"""Stand up a whole DiPaCo cluster from one config file via the launcher.

This is the library-level mirror of the ``opendipaco`` CLI: it writes a small config
and runs the all-in-one local cluster (coordinator + workers, or scheduler +
parameter servers + workers) in one process. On real hosts you'd instead run, with
the *same* config file on each machine:

    opendipaco scheduler --config cluster.yaml          # one host
    opendipaco ps        --config cluster.yaml --shard-id 0   # each PS host
    opendipaco ps        --config cluster.yaml --shard-id 1
    opendipaco worker    --config cluster.yaml          # each worker host

    python examples/launch_cluster.py [coordinator|sharded]
"""

from __future__ import annotations

import sys

from opendipaco.launch import LaunchConfig, run_local

BASE = {
    "model": {"vocab_size": 128, "hidden_size": 64, "num_attention_heads": 4,
              "intermediate_size": 128, "max_position_embeddings": 64,
              "layers_per_level": [1, 1], "level_sizes": [2, 2], "sequence_length": 32},
    "diloco": {"inner_steps": 8, "inner_lr": 1e-3},
    "data": {"source": "synthetic", "num_documents": 160, "routing": "kmeans"},
    "transport": {"host": "127.0.0.1", "port": 0},
    "run": {"generations": 6, "batch_size": 8, "local_workers": 3},
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "coordinator"
    cfg_dict = {**BASE, "mode": mode}
    if mode == "sharded":
        cfg_dict["sharded"] = {"num_shards": 2}
    cfg = LaunchConfig.from_dict(cfg_dict)

    print(f"launching a local '{mode}' cluster "
          f"({cfg.run.local_workers} workers"
          f"{', 2 parameter servers' if mode == 'sharded' else ''})...", flush=True)
    server, completed = run_local(cfg)

    print("\nper-path updates (uneven = async):", completed)
    print(server.metrics.report())
    if mode == "sharded":
        print("(scheduler holds no weights:", not hasattr(server, "bank"), "-> weights are sharded)")


if __name__ == "__main__":
    main()
