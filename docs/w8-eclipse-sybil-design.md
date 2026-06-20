# W8 (part 2) · Eclipse / Sybil-at-the-tracker

Status: **design** (slices a/b/c below land incrementally).

Second of W8's three trust problems (data poisoning is part 1; incentives is part
3). Like the rest of Phase 3 / W8, the convergence/efficacy-at-scale verdict rides
the §0f WAN run; the mechanism lands and is tested on-box.

## What's already closed (so we don't re-solve it)

The tracker directory is **self-certifying**: every record is an Ed25519
`make_peer_record` signed by the peer, re-verified on fetch (`fetch_directory`
drops anything failing `verify_record`). So a malicious tracker (or relay)
**cannot forge** a peer record, fabricate liveness for an identity it doesn't
hold, or forge a deregister tombstone. And **control is already
reputation-gated** (Phase 3): owner-eligibility, quorum membership, and lease
priority all require earned reputation, which redundant execution feeds — a flood
of fresh identities sits at the floor and can become neither owner nor quorum
member, so it cannot poison weights. `max_peers` bounds the directory size.

## The gap

A tracker can't forge records, but it can **omit** them and **admit** them:

1. **Eclipse of a newcomer.** A cold-start peer bootstraps from a **single**
   tracker (`tracker_connect_addr()` → one `(host, port)`; `fetch_directory(addr)`
   → that one seed). A malicious *or merely partitioned* tracker serves a
   **filtered** directory — omitting honest peers, leaving only attacker-controlled
   ones — so the newcomer's entire view is the attacker's. It then HRW-self-assigns
   and `derive_epoch`s over a poisoned directory, talking only to Sybils. The
   Phase-4 design names this exactly: *"at worst it partitions a newcomer."* A
   *warm* peer that already knows ≥1 honest owner cross-checks via gossip; a
   newcomer has only the seed(s).

2. **Sybil bloat / skew.** Identities are free, so on an `open_enrollment` tracker
   an attacker registers many — up to `max_peers` — dominating the directory.
   They can't gain *control* (reputation-gated, above), but they skew HRW
   placement and waste assignments/leases on Sybil workers.

## The defenses

### Defense 1 (the spine) — multi-seed bootstrap + union
A newcomer bootstraps from **several independent seeds** and takes the **union**
of verified records (freshest per `peer_id`). Omission-eclipse needs *every* seed
to withhold a peer; one honest seed restores it. Dead/malicious seeds are skipped
(best-effort), so this also improves availability. A seed serving a view that's a
large subset of the union (withholding peers others serve) is **flagged** — an
eclipsing/partitioned seed becomes detectable, not silent. Honest limit: with
**zero** honest seeds the newcomer is still eclipsed (irreducible — trust must
enter somewhere; pin seed pubkeys out-of-band for the strongest form).

### Defense 2 — Sybil dampening at admission
Identities being free, the directory can't be made Sybil-*proof* without
enrollment / proof-of-work / stake (research, owed). What's buildable: a
**per-source registration rate-limit** on the open-enrollment tracker (reusing
the Phase-3 `RateLimiter`) bounds how fast one source fills the directory, and
`max_peers` caps the total. This bounds *bloat rate*, not the steady-state — the
honest claim is "control is gated by reputation; the directory bounds waste".

### Defense 3 — gossip cross-validation (warm peers, mostly already there)
A peer that knows owners already pulls the directory via gossip (Phase 4 D7); a
persistent divergence between a seed and gossip is the same withholding signal as
Defense 1's, surfaced for a warm peer. Small addition on top of the union.

## Slices

### Slice a — multi-seed bootstrap + union
- `tracker.fetch_directory_multi(addrs, ...)`: fetch from each seed, verify, union
  by `peer_id` keeping the freshest `issued_at`; skip unreachable/erroring seeds;
  return the merged directory (+ which seeds answered).
- `TrackerCfg.seeds: list[[host,port]]` (extra bootstrap trackers beyond the
  primary); `tracker_connect_addrs()` returns the full list. Wire the
  rendezvous/decentralized fetch path through the union. One seed (default) is
  byte-identical to today.
- Tests: a filtering seed + an honest seed → union recovers the omitted peers;
  all-seeds-down → empty (no crash); single-seed unchanged.

### Slice b — divergence flag + admission rate-limit
- Flag a seed whose served view is a proper subset of the union beyond a margin
  (metric/warning) — an eclipsing/partitioned seed is surfaced.
- Per-source registration rate-limit on the open-enrollment tracker; refuse
  "rate limited" beyond it. Off / generous by default (byte-identical to today
  for honest cadences).

### Slice c — validation arm + docs
- A harness demonstrating eclipse-resistance (a malicious seed can't isolate a
  newcomer given ≥1 honest seed) and the Sybil-dampening bound. Roadmap +
  `remaining-gaps.md` updates.

## Honest limitations (state them)
- **≥1 honest seed required.** Multi-seed defeats omission-eclipse only if at
  least one seed is honest+reachable; with all seeds hostile the newcomer is
  eclipsed (irreducible without out-of-band trust — pin seed pubkeys).
- **Not Sybil-proof.** Free identities mean the directory can be bloated; we
  bound the *rate* and rely on reputation to gate *control*. Enrollment / PoW /
  stake for true Sybil resistance is owed (part 3 / research).
- **Efficacy at scale rides §0f**, like the rest of the trust layer.
