# Road to viable volunteer training — what remains after Phases 0–4

[internet-scale-plan.md](internet-scale-plan.md) laid out the gaps against
"train over the public internet on volunteer consumer hardware" and the
peer-to-peer transition (Phases 0–4). **Those phases have landed** (Phase 4 at
the control-plane level): the two walls that doc named — bandwidth/SPOF and
trust — are addressed *in principle*, and the central scheduler is gone.

This doc exists because **"architecturally complete" is not "viable."** The
phases built the distributed-systems *skeleton*; running a *large* model on
*consumer GPUs* over *consumer links* still needs work the phases deliberately
deferred or only half-solved. What follows is that remaining work, prioritized
and labelled honestly so we don't mistake a research question for an
engineering task.

Severity (relative to the goal):
- **B0** — blocks running on the target *at all*
- **B1** — blocks "large model" / "consumer GPU" specifically
- **B2** — ecosystem / viability hardening

Effort kind:
- **[eng]** — we know how; it needs building
- **[eng-hard]** — known-but-nontrivial systems work
- **[research]** — genuinely open; the answer may be an algorithm change, not code

> **The one honest meta-point:** the hardest unknown is not on this list as a
> feature — it is whether the *training dynamics* hold. Async + extreme
> heterogeneity + high staleness at swarm scale is unvalidated, and DiPaCo/DiLoCo
> were demonstrated in far gentler settings. If §0 below comes back negative,
> most of the rest is moot. Validate the dynamics before pouring effort into
> transport.

---

## 0. The gating milestone — sequence everything behind this

0f was always two separable questions; splitting them is what lets most of its
value be captured without distributed hardware.

### 0f-dynamics · *do our changes to the algorithm still converge?* · B0 · [eng] · ◑ harnessed

Convergence is a property of the **algorithm, not the network**, so it runs on
one box. `examples/validate_dynamics.py` does exactly this: it trains the
synchronous engine (the deterministic anchor) and the **real in-process async
sharded path** (per-contribution outer steps, inverse-staleness damping, worker
oversupply so commits genuinely race) on the *same* corpus, layering int8
compression (Phase 0c) and robust aggregation (Phase 3), and reports each
variant's held-out perplexity vs. the anchor. At toy scale the async / int8 /
robust-agg deltas **converge comparably to the anchor** (within ~1× — they
track, sometimes beat it), and the harness self-reports `INCONCLUSIVE` rather
than a vacuous pass when the anchor itself doesn't train at the chosen scale.
This **de-risks the dynamics deltas** that the DiPaCo paper (synchronous) does
not cover. The harness now also includes a **decentralized arm** (Phase 4's
scheduler-less **push-to-all-`k`** write path: self-assign + quorum reads +
owner-minted grants + independent per-owner application), driven by the
now-landed worker loop below — it converges to ~0.98× the anchor for a **single
writer** on one box. *Still owed (the systems half, below):* the same comparison
at real scale, and specifically **multi-writer** convergence on a shared module
(concurrent pushes interleave per-owner and the SGD+Nesterov step is
order-dependent → needs order-free, generation-keyed aggregation), epoch-skew
version stamping, and partial-push repair under churn.

### 0f-systems · *does it work over real consumer links?* · B0 · [research + eng]

The half that genuinely needs **distributed hardware**: a real WAN run (e.g.
three homes + a VPS) measuring convergence and behavior under real latency, NAT,
asymmetric bandwidth, and real churn — and exercising the **decentralized worker
loop** (self-assign → quorum-read bases → commit → **push to all `k` owners**).
The loop and a single-process `run_local` for it are **landed** (it trains on-box;
`run_local` no longer refuses decentralized mode); only the at-scale convergence
verdict and real-WAN behavior still need this hardware.
Closing this also lands the three Phase 4 residuals in
[phase4-design.md](phase4-design.md): convergent/Byzantine-robust **epoch
numbering** across joins, **owner-side α shard-size weighting** (uniform today),
and wiring **`worker_set`** (HRW-assignee enforcement) into the launch path.

*Why it gates everything:* 0f-dynamics is cheap insurance (now taken) that the
algorithm changes don't diverge; 0f-systems is the expensive proof that it
survives the actual network. The project is currently **proceeding on the
assumption that the synchronous DiPaCo result extrapolates** (0f-systems
deferred for lack of multi-node hardware), with 0f-dynamics standing in as the
on-box evidence. If a real run ever contradicts it, transport work above pauses
until the dynamics are understood.

---

## Tier B0 — blockers to run on the target at all

### W1 · NAT traversal / a relay tier · B0 · [eng-hard] · **landed (W1a–W1d)**

**Was the single biggest *unbuilt practical* blocker; now built.** Workers stay
dial-out-only (NAT-friendly), and the **owner** tier — which assumed public
reachability — can now be a NAT'd consumer machine reached *through relays*.
Built on **py-libp2p** as a transport+NAT substrate behind a `transport.kind:
libp2p` seam (TCP stays the default, bit-identical anchor): Circuit Relay v2 (a
`relay` role + `opendipaco relay` CLI), end-to-end **Noise** through the circuit
(so a relay forwards only ciphertext — this *subsumes* the deferred per-frame
signed envelopes), k≥2 relay reservations per NAT'd peer, **DCUtR** best-effort
relayed→direct hole-punch (D9), and NAT'd owners as a first-class tier
(`owner_eligible` accepts a relay-reachable peer; HRW placement / replication /
gossip / quorum-audit all dial circuit addrs). Identity reconciles with no app
churn — the libp2p host key derives from the peer's Ed25519 `PeerIdentity` and
the app id stays `sha256(pubkey)`. Authentication (Noise) **and** enrollment
(`admitted_peers` allowlist) both apply on the libp2p path. Design +
per-slice status in [w1-nat-design.md](w1-nat-design.md); end-to-end harness
`examples/validate_nat.py` (a relay + NAT'd owners + scheduler + workers training
through the relay); libp2p tests behind the `[nat]` extra with their own CI job.

*Remaining (0f-WAN-coupled, not a W1 gap):* **automatic discovery** of libp2p
multiaddrs through the tracker (today multiaddrs are wired explicitly or
in-process — the data/control plane over libp2p is proven, only the rendezvous
*of* addresses is still manual), plus real NAT/CGNAT + DCUtR-success measurement
and throughput at scale, which need the multi-node §0f run.

### W2 · Bandwidth: delta encoding + sparsification + lower-bit quant · B0 · [eng + research] · **landed (W2a–W2c)**

Design + per-slice status in [w2-bandwidth-design.md](w2-bandwidth-design.md).
**W2a (delta-down)**: `transport.down: delta` ships int8 `current − keyframe`
weights (keyframe + non-chained deltas bound the quant error; owner version ring
+ full fallback). **W2b (structured sparsification)**: `transport.up_density < 1`
sends only the top fraction of each pseudo-gradient (per-row for 2-D weights),
error-feeding the dropped mass. **W2c (sub-int8)**: `transport.compress: int4`
adds int4 per-group quantization for the up pseudo-gradient and the down delta
(~8× up). All off by default, byte-identical at full/dense/none, error-fed, and
Byzantine-hardened at the decode boundary; `validate_dynamics.py` has converging
`delta-down` / `sparse-up` / `int4` / `W2 stacked` arms. The free **`inner_steps`
lever** (more local work per sync ⇒ total traffic ∝ 1/inner_steps, no precision
cost) is documented + quantified in `examples/bandwidth_budget.py` — raise it
before paying any compression cost. The WAN §0f run stays the final convergence
verdict for the lossy levers (the on-box arms de-risk but don't replace it).

Phase 0c got ~2× down / 4× up, but the "ship only changed weights" cache is
**structurally defeated in async mode** (every accepted contribution bumps the
shared modules' versions, so full bf16 weights re-ship nearly every round). For
a 150M-param path that is already ~300 MB down + ~150 MB up per round (~10
min/round on a 20 Mbps uplink); for a *large* model over asymmetric consumer
uplinks it is fatal. Needs **delta encoding against the worker's held version**,
**structured sparsification**, and **sub-int8 quantization** — the error-feedback
machinery (`compress.py`) is the foundation to build on. The convergence impact
of aggressive compression must ride the §0f run (it is a dynamics change).
Leaning harder on DiLoCo's `inner_steps` (more local work between syncs) is the
cheapest complementary lever and costs no new code.

---

## Tier B1 — needed for "large model" + "consumer GPU" specifically

### W3 · Fit one path in consumer VRAM · B1 · [eng] · **landed (W3a–W3d)**

A worker holds **one path**, not the whole model, but a large path can still
exceed consumer VRAM (the per-round peak ≈ `4P + activations`, with the private
embed/head dominant). Design + per-slice status in
[w3-vram-design.md](w3-vram-design.md). **Measure-first** with a VRAM profiler
(`examples/vram_budget.py`), then **exact levers default-on** — activation
checkpointing, chunked cross-entropy (avoids the `[tokens, vocab]` logits), and a
fix so tied embed/head actually halve the dominant chunk — plus **§0f-gated lossy
levers** (blockwise 8-bit AdamW moments; private-copy de-dup/warming), off by
default with converging `validate_dynamics` arms. Per the profiler the exact
levers alone fit a ~540M path well under 12 GB (19 → 10.8 GB). Deferred (not
needed for 12 GB): the PCIe-bound CPU offloads (optimizer offload superseded by
8-bit Adam; embedding gather covered by tying). `inner_autocast` composes with
all of it.

### W4 · Churn robustness at consumer reality · B1 · [eng / tuning] ✅

Failover, replication, and recovery were built in Phase 2 — but for *cluster*
churn. W4 makes them survive *home* churn (sleep/reboot/drop). **Landed**
(`docs/w4-churn-design.md`): a measure-first churn stress harness
(`examples/validate_churn.py`: a real in-process cluster driven through abrupt /
graceful / suspend / flap / join arms, reporting survival + failover latency);
**graceful departure** — a signed deregister tombstones the leaver so failover
skips `owner_grace` (~10× faster than abrupt at the harness's timings), the
departing primary **drains** its latest state (weights + outer momentum) to the
rank-1 successor so the loss window collapses to ~0, and a leaving worker nacks
its in-flight lease for immediate re-lease; and **home-grade launch timings**
(`tracker.ttl` 120→30 s, `owner_grace` 240→60 s, lease reclaim 30→20 s, …, kept
`owner_grace ≥ 2·ttl`), with the library/anchor defaults left conservative and
`SIGTERM`/`SIGINT` routed through `shutdown(graceful=True)` under a hard deadline.
The convergence-under-churn verdict at WAN scale rides the §0f run (the harness
is the on-box half).

### W5 · Throughput-measured task sizing & pacing · B1 · [eng] ✅

Heterogeneity basics existed (bf16 autocast, `max_batch_size` caps), but tasks
were not sized from *measured* speed, so one slow-but-alive volunteer held a
path's lease far longer than a fast one and straggled every module that path
feeds. **Landed** (`docs/w5-task-sizing-design.md`): the scheduler measures each
worker's **effective rate** (lease→commit ÷ task work, EMA — captures slow links
as well as slow compute, no new trust) and sizes each task toward a target
`task_seconds` — **batch first** (gentle, bandwidth-neutral) down to a floor of
1, then fewer `inner_steps`; **shrink-only**, so a fast worker gets the exact
configured task and the off path is byte-identical. A worker too slow even for
the minimum task is **parked** (gets `idle`, holds no path; one request per
cooldown re-measures so a recovered worker rejoins). Audited tasks **pin their
size** so a checker reproduces the primary exactly (also fixing a latent
heterogeneous-`max_batch` false-divergence bug). Off by default
(`run.task_seconds`); the convergence-at-scale + wall-time straggler verdict
rides the §0f run, with an on-box `validate_dynamics.py` `het-batch` arm (per-path
batch heterogeneity converges ~0.9× the anchor).

---

## Tier B2 — ecosystem & trust for an open swarm

### W6 · A consumer client · B2 · [eng / UX]

The `opendipaco` CLI drives every role, but there is no volunteer-grade client:
one-command "join this run," GPU autodetection, honor a bandwidth cap, pause/
resume on sleep, and surface contribution/health. Without this, "consumer
hardware" means "people who can write a YAML config and open a port."

### W7 · Finish data decentralization · B2 · [eng] · ✅ **mostly landed**

`data.ship: spec` decentralized shard *materialization* (Phase 0d); W7
(`docs/w7-data-decentralization-design.md`) closes the rest of the central data
dependencies:

- **Bounded worker memory** ✅ — the worker shard cache is a bounded LRU
  (`run.worker_max_shards` / `join --max-shards`), so a worker that fails over
  across many paths no longer holds every shard it ever leased; an evicted shard
  re-materializes from the spec on its next lease (byte-identical training).
- **No central data authority for the router** ✅ — `data.router_sample` fits the
  k-means router on a bounded streamed sample (`build_server_corpus`), so the
  server never holds the whole corpus in RAM; token counts stream too.
- **Peer-verifiable routing** ✅ — `verify_routing` / `join --verify-routing`
  re-fits the shipped router from the public source and refuses to train on a
  mismatch (opt-in, belt-and-suspenders).
- **Owed:** EM re-sharding is still central (and not wired into the async/sharded
  loop) — decentralizing it changes *global* assignments and needs consensus, so
  it's §0f/research-shaped, tracked in `docs/remaining-gaps.md` alongside the WAN
  run. Routing verification is also a no-op in `schedule.mode: decentralized`
  (no shard-spec materialization seam) and tolerant across very different
  numerical stacks — both noted in the design doc.

### W8 · Trust beyond the current threat model · B2 · [research] · ◑ **poisoning + eclipse landed; incentives open**

Three open trust problems; **data poisoning** is now defended
(`docs/w8-data-poisoning-design.md`): Phase 3's robust aggregation + redundant
execution agree on a *digest*, which a worker training on **poisoned data** can
pass (every honest checker reproduces the same harmful gradient). The W8
trusted-probe screen catches what the digest can't — the audit checker measures
the reproduced update's loss on a small clean **probe**, and a quorum reporting a
loss rise flags the contribution (`robustness.probe_docs`/`probe_quorum`, off by
default; `examples/validate_poisoning.py`). It's a heuristic that raises the bar
on crude poisoning (a targeted backdoor tuned to preserve clean-probe loss can
evade), post-hoc like the digest audit, and its efficacy at scale rides the §0f
run — all recorded in `remaining-gaps.md`.

**Eclipse / Sybil-at-the-tracker** is now defended for the *eclipse* half
(`docs/w8-eclipse-sybil-design.md`): a newcomer bootstraps from **multiple seeds**
and takes the **union** of the self-certifying directory records, so a malicious or
partitioned seed that withholds honest peers can't isolate it (one honest seed
restores them); `seed_quorum` (M-of-N) optionally filters single-seed Sybil
injection; `examples/validate_eclipse.py`. **The design review corrected a false
premise here:** "control is reputation-gated" is *not* true — a fresh identity
starts above the owner-eligibility threshold and the worker's HRW isn't
reputation-filtered, so **fresh Sybils can become owners**. Closing that needs
identities to be non-free (stake), which is the **incentives** problem below — so
it's explicitly deferred there, not pretended solved.

**Still open (research-shaped): incentives** — the reputation system tracks
behavior but rewards nothing; sustained volunteer participation (BOINC/Petals-style)
needs a reason to contribute, and a **stake** is also the only effective
Sybil-of-control defense (enrollment breaks open volunteering; proof-of-work is a
poor fit for a compute network). Eclipse/Sybil part-2 deliberately stopped at the
buildable eclipse defense and handed fresh-Sybil-control to this.

---

## Sequencing

```
   0f-dynamics ✅ on-box (validate_dynamics.py: async/int8/robust + decentralized
                          push-to-all-k deltas converge at toy scale)
        │
        ▼
        ┌──────────────────── 0f-systems: WAN verdict (deferred / assumed) ────────────┐
        │  (decentralized worker loop landed on-box; real multi-node run owed)          │
        │  GATES EVERYTHING BELOW                                                        │
        └──────────────────────────────────────┬──────────────────────────────────────┘
                                                ▼
                        ┌───────────────────────┴───────────────────────┐
                  W1 NAT / relay ✅                                W2 bandwidth ✅   (parallel; both B0)
                        └───────────────────────┬───────────────────────┘
                                                ▼
                  W3 VRAM fit ✅  ·  W4 churn ✅  ·  W5 task sizing ✅       (B1: "large + consumer")
                                                ▼
          W6 client ✅  ·  W7 data plane ✅  ·  W8 trust ◑ (poisoning+eclipse ✅ / incentives ⬜)   (B2)
```

**Bottom line:** Phases 0–4 removed what had to be *trusted* or *central*. The
remaining work is what makes it *survivable on the actual hardware and links* —
and it splits cleanly into two known-but-real systems problems (NAT, bandwidth),
a set of large-model/consumer-GPU engineering tasks, and a few genuinely open
questions (convergence at scale, data poisoning, incentives). Gate all of it on
the §0f dynamics verdict.

## What this doc is *not*

[remaining-gaps.md](remaining-gaps.md) tracks a *different* goal — faithfully
reproducing DiPaCo on a trusted cluster — and its severities are relative to
that, not this. Items here are about the volunteer-internet target.
