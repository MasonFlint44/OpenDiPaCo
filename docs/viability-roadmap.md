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

### 0f · WAN convergence validation + the decentralized worker loop · B0 · [research + eng]

Two things, one milestone. **(a)** A real WAN run (e.g. three homes + a VPS) of
the async path that measures convergence vs. the synchronous anchor — the
project's standing §1.4 debt, now also the acceptance test for Phase 3's
robustness dynamics *and* Phase 4's decentralized write path. **(b)** The
**decentralized worker loop** that 0f must exercise: self-assign → quorum-read
bases → commit to the coordinator → **push to all `k` owners**, plus a
single-process `run_local` for it (today it refuses decentralized mode). Closing
this also lands the three Phase 4 residuals recorded in
[phase4-design.md](phase4-design.md): convergent/Byzantine-robust **epoch
numbering** across joins, **owner-side α shard-size weighting** (uniform today),
and wiring **`worker_set`** (HRW-assignee enforcement) into the launch path.

*Why it gates everything:* it is cheap relative to its information value. It
either says "the dynamics hold — now go solve NAT and bandwidth" or "they don't
— stop building transport." Build nothing below in earnest until 0f answers.

---

## Tier B0 — blockers to run on the target at all

### W1 · NAT traversal / a relay tier · B0 · [eng-hard]

**The single biggest *unbuilt practical* blocker.** Workers are dial-out-only
(NAT-friendly — keep that), but the **owner** tier assumes peers are publicly
reachable, and almost no consumer machine behind home NAT/CGNAT is. Today you
would need volunteers with public IPs to host the weight shards, which
contradicts "consumer hardware." Needs UDP hole-punching (STUN-style rendezvous,
which the tracker is well placed to broker) and/or a **relay** role for the
peers that can't be punched through. The `relay` role offer is already
*reserved* in the protocol (Phase 1/Phase 2 D10) but unbuilt; per-frame signed
envelopes were deferred waiting for exactly this relayed data-plane path. Known
techniques, real systems effort.

### W2 · Bandwidth: delta encoding + sparsification + lower-bit quant · B0 · [eng + research]

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
        ┌─────────────────────────── 0f: dynamics verdict ───────────────────────────┐
        │  (decentralized worker loop + WAN convergence run)  GATES EVERYTHING BELOW   │
        └──────────────────────────────────────┬──────────────────────────────────────┘
                                                ▼
                        ┌───────────────────────┴───────────────────────┐
                  W1 NAT / relay                                   W2 bandwidth      (parallel; both B0)
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
