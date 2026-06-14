"""NAT-traversal validation for W1 (docs/w1-nat-design.md): a full sharded
cluster where the **owners have no usable direct address** and are reached only
*through a relay*, training end-to-end over libp2p.

This is the runnable, end-to-end half of W1 — the unit tests prove each seam
(relayed round-trip, NAT'd owner fetch, enrollment, DCUtR fallback); this stands
the whole thing up at once and trains to a budget, the way ``opendipaco run``
would over libp2p:

  * a Circuit Relay v2 **relay** (a public peer running ``allow_hop``);
  * ``SHARDS`` **NAT'd owners** (ParameterServers served over libp2p) that each
    reserve a forwarding slot on the relay and advertise only their
    ``/p2p-circuit`` address — no peer ever dials them directly;
  * a **scheduler** and ``WORKERS`` **workers** over libp2p that reach the owners
    only through the relay (Noise e2e: the relay forwards ciphertext, D7), with
    DCUtR enabled (a best-effort relayed->direct upgrade, D9).

It reports that updates landed (the data plane flowed through the relay) and that
the relay forwarded traffic. Env-overridable:

    python examples/validate_nat.py
    SHARDS=2 WORKERS=2 GENERATIONS=2 python examples/validate_nat.py

HONEST CAVEAT: this runs in one process over loopback, so it validates the
*libp2p control + data plane through a relay* (reservation, circuit dialing,
enrollment, e2e Noise, the worker/owner/scheduler loop over libp2p) — not real
NAT/CGNAT traversal or DCUtR hole-punch *success*, which need multiple machines
behind real NATs and ride the 0f WAN bring-up (docs/viability-roadmap.md).
"""

from __future__ import annotations

import os
import threading

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    Scheduler,
    assign_shards,
    run_sharded_worker,
)
from opendipaco.schedule.p2p import Libp2pTransport, serve_over_libp2p


def _i(name, default):
    return int(os.environ.get(name, default))


SHARDS = _i("SHARDS", 2)
WORKERS = _i("WORKERS", 2)
GENERATIONS = _i("GENERATIONS", 2)
SEED = _i("SEED", 0)


def _cfg() -> DiPaCoConfig:
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1],
                        max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def main() -> None:
    cfg = _cfg()
    diloco = DiLoCoConfig(inner_steps=4, inner_lr=1e-3)
    g = torch.Generator().manual_seed(SEED)
    span = 48 // 4
    docs = [torch.randint(t * span, (t + 1) * span, (32,), generator=g)
            for t in range(4) for _ in range(8)]
    assign = torch.tensor([i % cfg.num_paths for i in range(len(docs))])
    corpus = ShardedCorpus.from_assignments(docs, assign, cfg.num_paths, 16)

    # 1. The relay: a public peer that forwards (allow_hop). It runs no RPC
    #    handler -- relayed streams are Noise-secured end-to-end, so it only ever
    #    sees ciphertext.
    relay = Libp2pTransport(PeerIdentity.generate(), relay=True).start()
    print(f"relay up: {relay.addrs[0]}")

    # 2. NAT'd owners: served over libp2p, each reserves on the relay and is
    #    addressed ONLY by its circuit addr (no direct route is ever advertised).
    keys = cfg.build_topology().module_keys()
    shards = [[k for k, s in assign_shards(keys, SHARDS).items() if s == i]
              for i in range(SHARDS)]
    pss = [ParameterServer(cfg, sk, diloco, host="127.0.0.1", port=0,
                           identity=PeerIdentity.generate()) for sk in shards]
    ps_t = [serve_over_libp2p(ps) for ps in pss]
    circuit_addrs = []
    for t in ps_t:
        circuit = t.reserve_on(relay.addrs[0])
        assert circuit and "/p2p-circuit/" in circuit, "owner failed to reserve on the relay"
        circuit_addrs.append(circuit)
    print(f"{SHARDS} NAT'd owners reachable only via relay:")
    for c in circuit_addrs:
        print(f"  {c}")

    # 3. The scheduler over libp2p, routing workers to the owners' circuit addrs.
    sched = Scheduler(cfg, corpus, circuit_addrs, diloco, batch_size=8,
                      host="127.0.0.1", port=0, identity=PeerIdentity.generate())
    sched_t = serve_over_libp2p(sched)

    # 4. Workers over libp2p: they reach the scheduler and the owners only through
    #    the relay's circuit addrs (DCUtR may upgrade to direct where a NAT allows).
    workers = [threading.Thread(
        target=run_sharded_worker, args=(cfg, diloco, sched_t.addrs[0]),
        kwargs=dict(transport="libp2p", identity=PeerIdentity.generate(),
                    heartbeat_interval=2.0), daemon=True) for _ in range(WORKERS)]
    for w in workers:
        w.start()

    try:
        completed = sched.fit(num_generations=GENERATIONS, total_generations=GENERATIONS)
        accepted = sched.metrics.accepted_updates
        budget = GENERATIONS * cfg.num_paths
        print("\n--- result ---")
        print(f"path updates completed: {sum(completed.values())} (budget {budget})")
        print(f"updates accepted over the relayed data plane: {accepted}")
        ok = sum(completed.values()) >= budget and accepted > 0
        print("VERDICT:", "PASS — trained through the relay" if ok else "FAIL")
    finally:
        sched_t.close()
        for t in ps_t:
            t.close()
        sched.shutdown()
        for ps in pss:
            ps.shutdown()
        relay.close()
        for w in workers:
            w.join(timeout=10)


if __name__ == "__main__":
    main()
