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
not cover. *Still owed:* the same comparison at real scale, and the one delta
this harness can't reach — Phase 4's decentralized **push-to-all-`k`** write
path, which needs the worker loop below.

### 0f-systems · *does it work over real consumer links?* · B0 · [research + eng]

The half that genuinely needs **distributed hardware**: a real WAN run (e.g.
three homes + a VPS) measuring convergence and behavior under real latency, NAT,
asymmetric bandwidth, and real churn — and exercising the **decentralized worker
loop** (self-assign → quorum-read bases → commit → **push to all `k` owners**,
plus a single-process `run_local` for it; today it refuses decentralized mode).
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

### W2 · Bandwidth: delta encoding + sparsification + lower-bit quant · B0 · [eng + research] · **W2a–W2c landed**

Design + per-slice status in [w2-bandwidth-design.md](w2-bandwidth-design.md).
**W2a (delta-down)**: `transport.down: delta` ships int8 `current − keyframe`
weights (keyframe + non-chained deltas bound the quant error; owner version ring
+ full fallback). **W2b (structured sparsification)**: `transport.up_density < 1`
sends only the top fraction of each pseudo-gradient (per-row for 2-D weights),
error-feeding the dropped mass. **W2c (sub-int8)**: `transport.compress: int4`
adds int4 per-group quantization for the up pseudo-gradient and the down delta
(~8× up). All off by default, byte-identical at full/dense/none, error-fed, and
Byzantine-hardened at the decode boundary; `validate_dynamics.py` has converging
`delta-down` / `sparse-up` / `int4` / `W2 stacked` arms. Remaining: the free
`inner_steps` docs lever. The WAN §0f run stays the final convergence verdict.

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

### W3 · Fit one path in consumer VRAM · B1 · [eng]

DiPaCo's premise already helps enormously — a worker holds **one path**, not the
whole model — but a single large path can still exceed 8–24 GB, and the
embedding/head (vocab × hidden, per-path-private under the paper semantics) is
often the dominant chunk. Needs activation checkpointing, CPU/disk **offload**
of the path's non-resident modules, and/or quantized (int8/int4) training.
`inner_autocast` (bf16 inner loop) is a start, not a finish.

### W4 · Churn robustness at consumer reality · B1 · [eng / tuning]

Failover, replication, and recovery are built (Phase 2) — but for *cluster*
churn. Home machines sleep, reboot, and drop their links constantly, at rates a
cluster never sees. Needs the replication factor `k`, `owner_grace`,
`replicate_interval`, and lease/takeover timings tuned for (and stress-tested
against) high churn, and likely graceful suspend/resume so a closing laptop
hands off cleanly instead of timing out.

### W5 · Throughput-measured task sizing & pacing · B1 · [eng]

Heterogeneity basics exist (bf16 autocast, `max_batch_size` caps), but tasks are
not sized from *measured* tokens/sec, so one slow volunteer can straggle a whole
module. The capability profile workers already advertise (§1.10) is the place to
put a measured throughput estimate; per-worker pacing (and possibly per-worker
`inner_steps`, which changes the inner-LR schedule and so needs care) follows.

---

## Tier B2 — ecosystem & trust for an open swarm

### W6 · A consumer client · B2 · [eng / UX]

The `opendipaco` CLI drives every role, but there is no volunteer-grade client:
one-command "join this run," GPU autodetection, honor a bandwidth cap, pause/
resume on sleep, and surface contribution/health. Without this, "consumer
hardware" means "people who can write a YAML config and open a port."

### W7 · Finish data decentralization · B2 · [eng]

`data.ship: spec` decentralized shard *materialization* (Phase 0d), but the
server still fits the k-means router and computes token counts centrally at
startup, EM re-sharding is still central, and the worker-side `shard_cache` is
unbounded in RAM (the disk cache bounds re-streaming, not memory). For a swarm
with no central data authority these need to move peer-side and be bounded.

### W8 · Trust beyond the current threat model · B2 · [research]

Phase 3's robust aggregation defends **weight-space** attacks, but a worker
training honestly on **poisoned data** produces plausible gradients that pass
every finite/norm/agreement check — undefended today. Also open: **eclipse /
Sybil-at-the-tracker** (a malicious or partitioned bootstrap seed isolating a
newcomer), and **incentives** — the reputation system tracks behavior but
rewards nothing, and sustained volunteer participation (BOINC/Petals-style)
needs a reason to contribute. These are research-shaped, not just unbuilt.

---

## Sequencing

```
   0f-dynamics ✅ on-box (validate_dynamics.py: deltas converge at toy scale)
        │
        ▼
        ┌──────────────────── 0f-systems: WAN verdict (deferred / assumed) ────────────┐
        │  (decentralized worker loop + real multi-node run)  GATES EVERYTHING BELOW    │
        └──────────────────────────────────────┬──────────────────────────────────────┘
                                                ▼
                        ┌───────────────────────┴───────────────────────┐
                  W1 NAT / relay ✅                                W2 bandwidth      (parallel; both B0)
                        └───────────────────────┬───────────────────────┘
                                                ▼
                       W3 VRAM fit   ·   W4 churn   ·   W5 task sizing       (B1: "large + consumer")
                                                ▼
                       W6 client   ·   W7 data plane   ·   W8 trust/incentives   (B2: ecosystem)
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
