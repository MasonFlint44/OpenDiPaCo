# W8 (part 2) · Eclipse / Sybil-at-the-tracker

Status: **design (revised after design review)**. Slices a/b/c below.

Second of W8's three trust problems (data poisoning is part 1; incentives is part
3). Efficacy at scale rides the §0f WAN run; the mechanism lands and is tested
on-box.

> **Design-review correction.** An earlier draft claimed "control is already
> reputation-gated, so fresh Sybils can't become owners." **That is false** and is
> the reason this doc was rescoped — see "What is *not* closed" below.

## What's already closed

The tracker directory is **self-certifying**: every record is an Ed25519
`make_peer_record` signed by the peer, re-verified on fetch (`fetch_directory`
drops anything failing `verify_record`); deregister tombstones are likewise
peer-signed and re-verified. So a malicious tracker/relay **cannot forge** a peer
record, fabricate liveness for an identity it doesn't hold, or forge a
deregister. `max_peers` bounds the directory size (global cap).

## What is *not* closed (the real threat surface)

1. **Eclipse of a newcomer.** A cold-start peer bootstraps from a **single**
   tracker (`tracker_connect_addr()` → one `(host, port)`; the decentralized
   worker's `directory_fn` fetches from that one seed, sharded.py). A malicious
   *or merely partitioned* tracker serves a **filtered** directory — omitting
   honest peers, leaving attacker-controlled ones — and the newcomer HRW-self-
   assigns / `derive_epoch`s over a poisoned view. A *warm* peer cross-checks via
   gossip (it unions the seed with current epoch owners, sharded.py `_gossip_once`);
   a **newcomer has no owners**, so its view collapses to the one tracker.

2. **Fresh Sybils can become owners (the hard one — NOT closed).** The review
   found the reputation gate does **not** prevent this:
   - `derive_epoch`'s `is_eligible` predicate is **optional, default `None`**
     (`ownership.py`), and the **decentralized worker loop** calls it with **no**
     predicate (`_serve_decentralized`, sharded.py) — its HRW self-assignment runs
     over *every* `owner_eligible` directory peer regardless of reputation.
   - Even where `is_eligible` *is* wired (the owner-side `ParameterServer` /
     `EpochManager`), a **fresh identity starts at the reputation floor (0.5),
     which is above `min_owner_reputation` (0.25)** — reputation *excludes
     proven-bad* peers, it does not *require earned* standing. So a brand-new
     identity is owner-eligible by design.

   ⟹ A Sybil-bloated directory **does** skew owner selection, and fresh Sybils
   **can** become owners. Closing this needs identities to be non-free, which is
   the **incentives/stake** problem (part 3) — enrollment breaks open
   volunteering, and proof-of-work is a poor fit for a compute network (the
   attacker already has GPUs; it just burns honest compute). **Deferred to part 3,
   stated plainly here so we don't pretend otherwise.**

## Scope of this part

Build the honestly-buildable hardening; do **not** claim Sybil-of-control is solved.

### Defense 1 — multi-seed bootstrap + union, keyed on pinned seed pubkeys
A newcomer bootstraps from **several seeds it pins by Ed25519 pubkey** (not just
host:port) and takes the **union** of verified records (freshest per `peer_id`).
- *Eclipse (omission):* defeated as long as ≥1 pinned seed is honest+reachable —
  one honest seed restores any peer the others withhold. Dead/malicious seeds are
  skipped (best-effort → also better availability).
- *Pinned pubkeys are the trust root.* `TrackerCfg.seeds` carries `(host, port,
  pubkey)`; a seed's reply is attributed to its pinned identity (the directory
  records are self-certifying regardless, but pinning lets us *attribute* a
  withholding/garbage seed and refuse an unpinned one on a paranoid run). The seed
  *list's* provenance is the irreducible trust input — it must come from a trusted
  channel, **not** the W6 manifest (which has the same provenance hole). Documented
  as such; no out-of-band magic claimed.
- *Union tradeoff (acknowledged):* the union is monotone — it also aggregates each
  seed's admitted records, so a malicious pinned seed can **inject** its
  signature-valid Sybils into the view. Union is therefore an **omission**
  defense, not a Sybil defense (Sybil-of-control is part 3). An optional
  `seed_quorum` (accept a record only if ≥M of N seeds serve it) trades
  omission-resistance for injection-resistance; off by default (M=1 = pure union).
- *Tombstones unioned too:* the multi-seed path unions verified deregister
  tombstones and a tombstone suppresses a same-or-older registration for that
  `peer_id` across all seeds — else a malicious seed could replay a departed
  peer's still-within-TTL record to resurrect it.

### Defense 2 — worker-side reputation filter (a real consistency fix)
Wire `is_eligible` (reputation ≥ `min_owner_reputation`) into the **worker-side**
`derive_epoch` (`_serve_decentralized`), matching the owner-side. This doesn't
stop *fresh* Sybils (floor > threshold — that's part 3), but it makes
**proven-bad** owners excluded *everywhere* (today a debited owner is dropped
owner-side but still HRW-selected in the worker's own view). Closes the
worker/owner inconsistency the review surfaced.

### Defense 3 — admission dampening + observability (honestly bounded)
- Per-source registration rate-limit on the open tracker (reuse the Phase-3
  `RateLimiter`): bounds bloat *rate* from one source. **Honest limit:** "source"
  is per-connection/IP, so a multi-IP adversary bypasses it, and `max_peers` then
  becomes an honest-peer-**exclusion DoS** lever (a flood fills it, refusing
  honest newcomers). It bounds rate, not count; real resistance is part 3.
- A seed serving a view that is a subset of the union beyond a margin is
  **surfaced as a metric/log** (an eclipsing/partitioned seed is observable). This
  is **observability, not a relied-on defense** — it's blind to a *targeted*
  one-peer omission (below any margin), and the actual protection is the union +
  ≥1 honest seed.

## Slices

### Slice a — multi-seed union (pinned), tombstones, worker-side rep filter
- `tracker.fetch_directory_multi(seeds, ...)`: per seed fetch+verify, union by
  `peer_id` keeping freshest `issued_at`, union+apply verified tombstones, skip
  unreachable/erroring seeds; pinned-pubkey attribution. Mirror the owner-side
  `_directory` merge.
- `TrackerCfg.seeds: list[[host, port, pubkey]]`; `tracker_connect_seeds()`
  returns the full pinned list (the primary + extras). One seed (default) =
  byte-identical to today.
- Wire the decentralized worker / scheduler bootstrap through the union; wire
  `is_eligible` into the worker-side `derive_epoch`.
- Tests: filtering seed + honest seed → union recovers omitted peers; tombstone
  suppresses a replayed within-TTL record; all-seeds-down → empty, no crash;
  single-seed unchanged; a proven-bad (debited) owner is excluded from the
  worker's epoch.

### Slice b — `seed_quorum` (M-of-N) + admission rate-limit + divergence metric
- Optional `seed_quorum` injection-resistance knob (M=1 default = union).
- Per-source registration rate-limit on the open tracker; refuse beyond it.
- Divergence metric/log for a withholding seed (observability).

### Slice c — validation arm + docs
- Harness: a malicious seed can't isolate a newcomer given ≥1 honest pinned seed;
  show the union recovers omitted owners and a debited owner is excluded. Roadmap
  + `remaining-gaps.md` updates (incl. the fresh-Sybil-owner gap → part 3).

## Honest limitations (state them)
- **Fresh-Sybil-as-owner is NOT closed** — identities are free; the only effective
  fix (stake) is the incentives layer (part 3). This part bounds eclipse + waste,
  not Sybil-of-control.
- **≥1 honest, reachable, pinned seed required** for eclipse-resistance; with all
  seeds hostile the newcomer is eclipsed (irreducible without trusted seed
  provenance).
- **Union aggregates Sybils** (monotone) — `seed_quorum` trades that against
  omission-resistance; neither is free.
- **Rate-limit bounds rate, not count**; `max_peers` is a DoS-exclusion lever
  under multi-source flooding.
- **Divergence detection is observability**, blind to targeted one-peer omission.
- **Efficacy at scale rides §0f.**
