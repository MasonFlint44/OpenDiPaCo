"""Single-box dynamics sweep — the no-internet half of plan §0f.

The §0f WAN run answers two separable questions. The **systems** half (real
latency, NAT, bandwidth, churn) genuinely needs distributed nodes. The
**dynamics** half — *do the changes we made to the synchronous DiPaCo algorithm
still converge?* — does not: convergence is a property of the algorithm, not the
network, so it runs on one box. This harness is that half.

The synchronous engine is the deterministic anchor (sum sharing paths'
pseudo-gradients, one outer step per round, √P rescale). We compare its
converged held-out perplexity against the **real in-process async sharded path**
(per-contribution outer steps, inverse-staleness damping, no √P — the actual
production dynamics, driven over localhost with worker oversupply so commits
genuinely race and go stale), with two further deltas layered on:

* ``int8`` wire compression (Phase 0c) — does quantized communication still train?
* robust aggregation (Phase 3, ``robustness: on``) — owner-side quorum buffering
  applies one aggregated outer step across sharing paths instead of one-at-a-time,
  which changes the outer step even with no adversaries.
* ``delta-down`` (W2a, ``down: delta``) — the worker trains from a keyframe + int8
  delta reconstruction of the weights instead of the exact (bf16) weights; does
  the bounded reconstruction error still train?
* ``sparse-up`` (W2b, ``up_density < 1``) — the worker sends only the top fraction
  of each pseudo-gradient (error-feeding the rest); does sparsified communication
  still train?
* ``int4`` (W2c, ``compress: int4``) — int4 per-group pseudo-gradients/deltas; does
  4-bit communication still train? (plus a ``W2 stacked`` arm with all three on.)
* ``8-bit Adam`` (W3d, ``optim_8bit``) — blockwise int8 optimizer moments; does the
  quantized inner optimizer still train?
* ``dedup-private`` (W3d, ``dedup_private``) — aliasing the worker's private modules
  changes warm-round private warming; does it still converge?
* ``het-batch`` (W5, ``het_batch``) — half the workers train a smaller batch, so
  paths see mixed batch sizes round to round (the per-path batch heterogeneity
  throughput-measured task sizing introduces); does it still converge? (One box
  has no real *speed* heterogeneity, so the batch mix is injected directly; the
  straggler/wall-time benefit of sizing rides the WAN run.)
* ``decentralized`` (Phase 4, ``schedule: decentralized``) — the **scheduler-less**
  write path with a **single** self-assigning worker: it quorum-reads bases,
  commits to each path's owner-coordinator (which mints the grant), and **pushes
  to all k owners**, each of which applies the granted step independently. Does
  that control + write path converge comparably to the central anchor? (Single
  writer only — see ``DEC_WORKERS``: multi-writer agreement on a shared module
  needs order-free aggregation and is the §0f *systems* half.)

All configs train the **same** corpus/sharding/seed and are evaluated by the same
router-free metric (best-path perplexity on a held-out split), so the numbers are
directly comparable. Env-overridable:

    python examples/validate_dynamics.py
    ROUNDS=40 HIDDEN=128 LEVELS=3 NUM_DOCS=400 WORKERS=6 python examples/validate_dynamics.py

HONEST CAVEAT — what this does and does NOT cover:
  • COVERS, end-to-end, on one box: async staleness dynamics, int8 compression,
    robust-aggregation dynamics, and Phase 4's **single-writer decentralized
    push-to-all-k write path** (self-assign + quorum reads + owner-minted grants +
    independent per-owner application), each vs. the synchronous anchor.
  • Does NOT cover: real-WAN *systems* behavior (latency / NAT / bandwidth / real
    churn) — that still needs the multi-node run — and, specifically for the
    decentralized arm, **multi-writer convergence on a shared module** (concurrent
    pushes interleave per-owner and the outer step is order-dependent, so it needs
    order-free generation-keyed aggregation), epoch-transition version skew, and
    partial-push repair. Those are the §0f *systems* half, still owed.
  • This is a SMALL-SCALE check: read the gap-to-anchor *trend*, not absolute
    perplexity. A green run is evidence the deltas converge, not a scale proof.
"""

from __future__ import annotations

import os
import threading
import time

import torch

from opendipaco import BackboneConfig, DiLoCoConfig, DiPaCoConfig, DiPaCoEngine
from opendipaco.backend import LocalBackend
from opendipaco.data import ShardedCorpus, pack_sequences
from opendipaco.inference import compose_path, config_path, perplexity
from opendipaco.schedule import (
    ParameterServer,
    PeerIdentity,
    Scheduler,
    Tracker,
    assign_shards,
    derive_epoch,
    make_peer_record,
    owners_for,
    path_primary,
    run_decentralized_worker,
    run_sharded_worker,
)


def _i(name, d):
    return int(os.environ.get(name, d))


HIDDEN = _i("HIDDEN", 96)
HEADS = _i("HEADS", 4)
INTERMEDIATE = _i("INTERMEDIATE", 192)
LAYERS = _i("LAYERS", 2)
VOCAB = _i("VOCAB", 256)
SEQ_LEN = _i("SEQ_LEN", 32)
LEVELS = _i("LEVELS", 2)                 # K -> K x K paths
ROUNDS = _i("ROUNDS", 40)                # sync rounds; async gets ~the same per-path updates
INNER_STEPS = _i("INNER_STEPS", 10)
INNER_LR = float(os.environ.get("INNER_LR", "3e-4"))   # stable regime for the toy task
BATCH = _i("BATCH", 8)
NUM_DOCS = _i("NUM_DOCS", 240)
NUM_SHARDS = _i("NUM_SHARDS", 2)
DEC_OWNERS = _i("DEC_OWNERS", 3)         # decentralized owner nodes (k-way replicated)
DEC_K = _i("DEC_K", 3)                   # replication factor for the decentralized arm
# Decentralized workers default to 1 -- and this is a real boundary, not just a
# throughput convenience. A single self-assigning worker produces a *globally
# ordered* push stream (it pushes one contribution to all k owners before the
# next), so every owner applies the same ordered sequence and the k independent
# outer steps stay byte-identical -- the convergence this arm verifies. With
# SEVERAL workers pushing to the *same shared module* concurrently, their pushes
# interleave differently at each owner, and the outer optimizer (SGD+Nesterov) is
# order-dependent, so the owners would diverge unless the writes are made
# order-free (the robust-aggregation buffer generalized to a generation-keyed,
# arrival-order-independent aggregate). That multi-writer agreement -- together
# with epoch-skew version stamping, partial-push repair, and real churn -- is the
# §0f *systems* half (WAN run), still owed. So this arm validates the single-writer
# decentralized write-path dynamics; it does NOT claim multi-writer convergence.
DEC_WORKERS = _i("DEC_WORKERS", 1)
DEC_TIMEOUT = float(os.environ.get("DEC_TIMEOUT", "300"))  # per-path target safety bound
WORKERS = _i("WORKERS", 0)               # 0 -> default to num_paths (full concurrency)
SEED = _i("SEED", 0)
DEVICE = os.environ.get("DEVICE", "cpu")
# A pass threshold: the async variants should land within this factor of the
# anchor's perplexity (dynamics differ, so we want "comparable", not "equal").
TOL = float(os.environ.get("TOL", "1.5"))


def _topic_docs(n, length, *, seed):
    """Token 'topics' so routing has real structure (a path can specialize)."""
    g = torch.Generator().manual_seed(seed)
    span = VOCAB // 4
    per = max(1, n // 4)
    return [torch.randint(t * span, (t + 1) * span, (length,), generator=g)
            for t in range(4) for _ in range(per)]


def _config():
    bb = BackboneConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_attention_heads=HEADS,
                        intermediate_size=INTERMEDIATE, layers_per_level=[1] * LAYERS,
                        max_position_embeddings=max(64, SEQ_LEN))
    return DiPaCoConfig(backbone=bb, level_sizes=[LEVELS, LEVELS], sequence_length=SEQ_LEN)


def _corpus(config):
    docs = _topic_docs(NUM_DOCS, SEQ_LEN * 2, seed=SEED)
    assign = torch.tensor([i % config.num_paths for i in range(len(docs))])
    return ShardedCorpus.from_assignments(docs, assign, config.num_paths, SEQ_LEN)


def _val_seqs():
    return pack_sequences(_topic_docs(64, SEQ_LEN * 2, seed=SEED + 9999), SEQ_LEN)


def best_path_ppl(config, modules, val_seqs) -> float:
    """Held-out perplexity under the single best-fitting path (router-free, so
    it's identical-by-construction across configs and just measures how well the
    shared+private modules learned)."""
    with torch.no_grad():
        return min(perplexity(compose_path(config, modules, config_path(config, i)), val_seqs)
                   for i in range(config.num_paths))


def train_sync(config, diloco, corpus) -> dict:
    """The deterministic anchor: synchronous DiPaCo rounds."""
    eng = DiPaCoEngine(config, diloco, LocalBackend(config.build_topology()),
                       seed=SEED, device=DEVICE)
    eng.total_rounds = ROUNDS
    eng.fit(corpus, num_rounds=ROUNDS, batch_size=BATCH, log_every=0)
    return eng.global_modules()


def train_async(config, diloco, corpus, *, compress="none", robustness="off",
                down="full", up_density=1.0, optim_8bit=False, dedup_private=False,
                het_batch=False) -> dict:
    """The real in-process async sharded path (localhost TCP, worker oversupply
    so commits race + go stale). Returns the merged authoritative bank from the
    parameter servers -- no node held it whole.

    ``het_batch`` (W5) gives half the workers a smaller advertised batch cap, so
    paths train with **mixed batch sizes round to round** -- the per-path batch
    heterogeneity that throughput-measured sizing introduces. (One box can't
    create real *speed* heterogeneity, so we inject the batch heterogeneity it
    produces directly; the wall-time/straggler benefit rides the WAN run.)"""
    import dataclasses
    # W3d worker-side levers live on the worker's diloco (8-bit inner optimizer;
    # private-copy de-dup in _train_path).
    diloco = dataclasses.replace(diloco, optim_8bit=optim_8bit, dedup_private=dedup_private)
    keys = config.build_topology().module_keys()
    shards = [[k for k, s in assign_shards(keys, NUM_SHARDS).items() if s == i]
              for i in range(NUM_SHARDS)]
    ps_kw = {"down": down}
    if robustness == "on":
        # Small quorum/flush windows so partial buffers flush inside a short run.
        ps_kw |= dict(robustness="on", quorum_target=2, quorum_timeout=0.5,
                      replicate_interval=0.2)
    pss = [ParameterServer(config, sk, diloco, host="127.0.0.1", port=0, **ps_kw)
           for sk in shards]
    for ps in pss:
        ps.start()
    sched = Scheduler(config, corpus, [("127.0.0.1", ps.port) for ps in pss], diloco,
                      batch_size=BATCH, host="127.0.0.1", port=0, compress=compress,
                      down=down, up_density=up_density)
    sched.start()
    n_workers = WORKERS or config.num_paths
    # W5: half the cohort advertises a smaller batch cap -> per-path batch varies.
    def _mb(i):
        return max(1, BATCH // 2) if (het_batch and i % 2 == 1) else None
    workers = [threading.Thread(
        target=run_sharded_worker, args=(config, diloco, ("127.0.0.1", sched.port)),
        kwargs=dict(seed=SEED, heartbeat_interval=2.0, max_batch_size=_mb(i)), daemon=True)
        for i in range(n_workers)]
    for w in workers:
        w.start()
    try:
        sched.fit(num_generations=ROUNDS, total_generations=ROUNDS)
    finally:
        sched.shutdown()
        merged = {k: m for ps in pss for k, m in ps.bank.items()}
        for ps in pss:
            ps.shutdown()
        for w in workers:
            w.join(timeout=10)
    return merged


def _path_generation(owner_by_id, topo, epoch, path) -> int:
    """The path's current generation, read from its owner-coordinator."""
    prim = path_primary(topo.path_module_keys(path), epoch)
    ps = owner_by_id.get(prim["peer_id"]) if prim else None
    if ps is None:
        return 0
    with ps._lock:
        return ps._gen.get(path, [0])[0]


def train_decentralized(config, diloco, corpus, *, n_owners=DEC_OWNERS, k=DEC_K) -> dict:
    """The real in-process **decentralized** path (Phase 4): no scheduler. A
    ``DEC_WORKERS`` worker self-assigns from a gossiped directory, quorum-reads each
    path's bases, commits to the path's owner-coordinator (which version-fences the
    generation and mints the grant), and pushes to all ``k`` owners -- **each owner
    applies the granted step independently** (``_may_write_locked``), so the k
    replicas converge to the same bytes and a quorum read can confirm them. Owners
    cold-start from a bootstrap epoch and run to a per-path generation target.
    Returns the merged authoritative bank (each key from its primary).

    Defaults to **one** worker (see ``DEC_WORKERS``): with one writer the push
    stream is globally ordered, so the k order-dependent outer steps stay
    byte-identical. This exercises the push-to-all-``k`` WRITE path + quorum reads +
    owner-minted grants + owner-coordinated generations end to end on one box -- the
    *single-writer* dynamics half of §0f. Multi-writer agreement on a shared module
    (order-free aggregation), epoch-skew versioning, and partial-push repair are the
    WAN *systems* half; the read-side quorum primitive alone is in
    ``validate_decentralized.py``."""
    if k < 2 or n_owners < 2:
        raise ValueError(f"decentralized arm needs DEC_K>=2 and DEC_OWNERS>=2 "
                         f"(quorum reads + replication need co-owners); got "
                         f"k={k}, owners={n_owners}")
    auth = os.urandom(8).hex()
    ids = [PeerIdentity.generate() for _ in range(n_owners)]
    pubs = [i.public_key_hex for i in ids]
    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True, ttl=30.0, auth_key=auth)
    tracker.start()
    taddr = ("127.0.0.1", tracker.port)
    weights = {p: corpus.shard_weight(p) for p in range(config.num_paths)}
    owners = []
    for i in range(n_owners):
        ps = ParameterServer(
            config, [], diloco, host="127.0.0.1", port=0, identity=ids[i],
            schedule_mode="decentralized", k=k, read_quorum=(k // 2) + 1, salt="",
            replicate_interval=0.2, corpus_weights=weights, auth_key=auth,
            admitted_peers=[p for j, p in enumerate(pubs) if j != i])
        ps.start()
        owners.append(ps)
    owner_by_id = {ps.peer_id: ps for ps in owners}
    recs = [make_peer_record(ids[i], reachability="public",
                             addr=("127.0.0.1", owners[i].port), roles=("owner",))
            for i in range(n_owners)]
    epoch0 = derive_epoch(recs, k=k, salt="", prev=None)
    for ps in owners:
        ps.apply_epoch(epoch0, bootstrap=True)
        ps.start_tracker_heartbeat(taddr, "127.0.0.1", interval=1.0, auth_key=auth)
    topo = config.build_topology()

    stop = threading.Event()

    def worker_target():
        run_decentralized_worker(
            config, diloco, taddr, corpus, identity=PeerIdentity.generate(), seed=SEED,
            device=DEVICE, auth_key=auth, k=k, salt="", read_quorum=(k // 2) + 1,
            lease_ttl=30.0, batch_size=BATCH, total_rounds=ROUNDS,
            heartbeat_interval=1.0, stop_event=stop)

    workers = [threading.Thread(target=worker_target, daemon=True) for _ in range(DEC_WORKERS)]
    for w in workers:
        w.start()
    deadline = time.monotonic() + DEC_TIMEOUT
    try:
        while time.monotonic() < deadline:
            if all(_path_generation(owner_by_id, topo, epoch0, p) >= ROUNDS
                   for p in topo.paths()):
                break
            time.sleep(0.05)
        else:
            # Timed out before every path hit the target: the bank is undertrained,
            # so the ppl below reflects a *systems* stall (a stuck worker / no
            # quorum), NOT a dynamics regression. Flag it loudly so the WEAK verdict
            # isn't misread -- raise DEC_TIMEOUT or check the swarm.
            gens = [_path_generation(owner_by_id, topo, epoch0, p) for p in topo.paths()]
            print(f"  WARNING: decentralized arm TIMED OUT after {DEC_TIMEOUT:.0f}s at "
                  f"generations {gens} (target {ROUNDS}); the ppl below is undertrained "
                  f"(a systems stall, not a dynamics verdict).", flush=True)
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=10)
        # Merge each key from its primary (the authoritative copy), after workers
        # are quiesced and before teardown.
        merged = {key: owner_by_id[owners_for(key, epoch0)[0]["peer_id"]].bank[key]
                  for key in topo.module_keys()
                  if owners_for(key, epoch0)[0]["peer_id"] in owner_by_id}
        for ps in owners:
            ps.shutdown(graceful=True)
        tracker.shutdown()
    return merged


def main() -> None:
    torch.manual_seed(SEED)
    config = _config()
    diloco = DiLoCoConfig(inner_steps=INNER_STEPS, inner_lr=INNER_LR)
    corpus = _corpus(config)
    val = _val_seqs()
    paths = config.num_paths

    print(f"model: hidden={HIDDEN} layers={LAYERS} vocab={VOCAB}  paths={paths} "
          f"({LEVELS}x{LEVELS})  shards={NUM_SHARDS}  workers={WORKERS or paths}")
    print(f"budget: ~{ROUNDS} per-path updates x {INNER_STEPS} inner steps  "
          f"docs={NUM_DOCS} seq_len={SEQ_LEN}  seed={SEED}\n")

    # Uniform-random perplexity is the no-learning reference; the topic spans
    # give a learnable optimum well below it. "Learned" = clearly under random.
    random_ppl = float(VOCAB)
    learned_ceiling = random_ppl * 0.8
    print(f"  (reference: uniform-random ppl = {random_ppl:.0f}; learning means well below it)\n")

    anchor = best_path_ppl(config, train_sync(config, diloco, corpus), val)
    print(f"  sync (anchor)          best-path val ppl = {anchor:8.3f}")

    # Guard against a vacuous verdict: if the anchor didn't learn or diverged,
    # "comparable to the anchor" means nothing. This synthetic task is sensitive
    # to over-training (push ROUNDS/INNER_LR too high and the toy LM diverges) --
    # that's the data, not the deltas. A credible large run uses real C4.
    if not (anchor < learned_ceiling):
        print(f"\nINCONCLUSIVE: the anchor didn't train stably at this scale "
              f"(ppl {anchor:.1f} vs random {random_ppl:.0f}). Lower INNER_LR / ROUNDS, "
              f"raise NUM_DOCS, or use a real corpus -- the synthetic toy task is fragile, "
              f"not a verdict on the deltas.")
        return

    variants = [
        ("async", dict(compress="none", robustness="off")),
        ("async + int8", dict(compress="int8", robustness="off")),
        ("async + robust agg", dict(compress="none", robustness="on")),
        ("async + delta-down", dict(compress="int8", down="delta")),   # W2a
        ("async + sparse-up", dict(compress="int8", up_density=0.25)),  # W2b
        ("async + int4", dict(compress="int4")),                        # W2c
        ("async + W2 stacked", dict(compress="int4", down="delta", up_density=0.25)),
        ("async + 8-bit Adam", dict(optim_8bit=True)),                   # W3d
        ("async + dedup-private", dict(dedup_private=True)),             # W3d
        ("async + het-batch (W5)", dict(het_batch=True)),                # W5: per-path batch mix
    ]
    worst, all_learned = 1.0, True

    def _measure(name, modules):
        nonlocal worst, all_learned
        ppl = best_path_ppl(config, modules, val)
        ratio = ppl / anchor
        worst = max(worst, ratio)
        learned = ppl < learned_ceiling
        all_learned &= learned
        flag = "OFF-ANCHOR" if ratio > TOL else ("DIDN'T LEARN" if not learned else "ok")
        print(f"  {name:<24} best-path val ppl = {ppl:8.3f}   "
              f"{ratio:4.2f}x anchor   [{flag}]")

    for name, kw in variants:
        _measure(name, train_async(config, diloco, corpus, **kw))

    # The Phase 4 decentralized write path (no scheduler; push-to-all-k + quorum
    # reads + owner-minted grants). Structurally different from the async sweep, so
    # it runs through its own driver -- the §0f on-box verdict for decentralized.
    _measure(f"decentralized (P4, k={DEC_K})", train_decentralized(config, diloco, corpus))

    print()
    if all_learned and worst <= TOL:
        print(f"PASS: the anchor learned, and every async variant both learned and stayed "
              f"within {TOL:.2f}x of it (worst {worst:.2f}x). The async / int8 / robust-agg "
              f"deltas track the algorithm on one box.")
    else:
        print(f"WEAK: a variant didn't learn or landed >{TOL:.2f}x off the anchor "
              f"(worst {worst:.2f}x). Re-run with more ROUNDS/NUM_DOCS; if it persists, the "
              f"dynamics need a look BEFORE the WAN run.")
    print("\nCaveat: small-scale, one box. Validates async/int8/robust/decentralized "
          "*dynamics*, not WAN systems behavior (latency/NAT/bandwidth/real churn). "
          "See the module docstring.")


if __name__ == "__main__":
    main()
