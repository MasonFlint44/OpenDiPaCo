# W8 (part 1) · Data-poisoning defense

Status: **design** (slices a/b/c below land incrementally).

W8 in the roadmap is three open problems: **data poisoning**, **eclipse/Sybil at
the tracker**, and **incentives**. This doc covers the first — the others are
their own designs. Like Phase 3 robustness, the point here is a *dynamics*
property, so the convergence/efficacy verdict rides the §0f run; the mechanism
and its unit behavior land and are tested on-box, off by default.

## What's already closed (so we don't re-solve it)

A worker submitting a bad **shared-weight update** is Phase 3's job (robust
aggregation + version-pinned redundant execution + reputation). Specifically, in
the **sharded path the input data is server-authoritative**: the scheduler ships
either the packed `shard` bytes or the `shard_spec`, and the worker materializes
deterministically from it (`_materialize_from_spec`) — the worker never supplies
its own input data. So:

- **Input substitution** (worker trains on data it wasn't assigned) — closed in
  the sharded path: data comes from the scheduler/spec, not the worker.
- **Routing integrity** (the shipped router matches the public source) — W7c
  `verify_routing`.
- **Computation integrity** (the worker computed the right pseudo-gradient for
  its base+data) — Phase 3c version-pinned redundant execution: a checker
  reproduces the primary's update from the pinned base and the audit compares
  pseudo-gradient **digests**.

## The gap

Phase 3c's audit agrees on a **digest of the pseudo-gradient**. That catches a
worker whose *computation* diverges, but **not adversarial data *content***: if
the corpus/shard the worker trained on is itself poisoned (label-flips, a
backdoor trigger, gradient-shaped-but-plausible content), every honest checker
handed the *same* poisoned shard reproduces the *same* harmful pseudo-gradient →
the digests **agree** → the audit *credits* everyone and the poisoned update is
applied. Finite/norm/agreement checks all pass. This is the undefended case the
roadmap names: "a worker training honestly on poisoned data produces plausible
gradients that pass every check."

Two flavors:
1. **Poisoned corpus content** (sharded path): the data is server/spec-supplied,
   so no single worker is to blame — but the update still harms the model. The
   defense must *reject the contribution* (and surface it), not debit a worker.
2. **Worker-chosen data** (decentralized path, `schedule.mode: decentralized`):
   `run_decentralized_worker` builds its corpus *locally* (`build_corpus`), so a
   worker genuinely picks its own data — and there is no redundant-execution
   audit there. (W7 already rejects `decentralized + ship: spec`; bringing
   spec-pinned, verifiable data into the decentralized path is an owed W7
   follow-up that would also help here.)

## The mechanism — trusted-probe screening

The canonical data-poisoning defense that does **not** need re-execution
agreement or worker oversupply to *agree on a digest*: evaluate a contribution's
**effect on a small trusted, clean probe set**. A poisoned update that passed
every weight-space check still *raises loss on clean held-out data* (or fails to
lower it the way an honest update does). The Phase 3c **audit checker already
composes and trains the full path** on the pinned base — so it can, at marginal
cost, also measure the probe-loss effect of the update it just reproduced and
report it.

- **Trusted probe.** A small, clean, operator-curated held-out batch (like a val
  split), shipped with the run config / manifest. Trust model: operator-curated,
  pinned/TOFU like the manifest (W6). Decentralized probe-trust (quorum-curated)
  is harder — owed, noted below.
- **Signal.** The checker reports `probe_delta = probe_loss(trained_local) −
  probe_loss(base)` for the path it reproduced (both on the trusted probe). An
  honest update drives this ≤ ~0 (or within a margin); a poisoned update drives
  it clearly positive.
- **Verdict.** Separate from the digest-agreement reputation tally (which stays
  Phase 3c's). Probe screening is a **contribution-quality gate**: when a quorum
  of checkers report `probe_delta` above a margin, the audited contribution is
  **rejected** (not aggregated) and recorded. Reputation impact is threat-model
  aware — debit the worker only where it *chose* its data; for server-supplied
  data the contribution is rejected and surfaced (a corpus-quality alarm), not
  blamed on the worker.

## Slices

### Slice a — the probe-screening primitive
- `probe.py` (new, or in `aggregate.py`): `probe_loss(path_model, probe_batch)`
  and `screen_delta(before, after, *, abs_margin, rel_margin) -> bool` (harmful).
  A `TrustedProbe` holder (token tensors + how it's carried/serialized). Pure,
  on-box unit-testable: a label-flipped/poisoned shard yields a clearly positive
  `probe_delta`; a clean shard ≈ 0 or negative.
- Off by default everywhere; no behavior change when no probe is configured.

### Slice b — wire into the audit checker + resolution
- The checker (`check_only` path) computes `probe_delta` on its reproduced update
  and reports it in the check commit; the owner's `_resolve_audit_locked` gates:
  a quorum of harmful `probe_delta`s rejects the contribution + records a metric
  (+ reputation per the threat-model rule). `robustness.probe_*` config (margin,
  quorum, source), off by default, byte-identical when off.
- Probe data plumbing: carried in the run config / manifest (operator-curated),
  shipped to checkers like the shard recipe.

### Slice c — validation arm + docs
- `examples/validate_robustness.py` poisoned-shard arm: a label-flipped /
  backdoored shard's contribution is flagged by the probe screen while a clean
  one passes. Roadmap + `remaining-gaps.md` updates.

## Honest limitations (state them, don't paper over)
- **The probe must be trusted.** A poisoned probe inverts the defense; trust
  rides the manifest pinning (W6), and decentralized quorum-curation is owed.
- **Heuristic, not a proof.** Clean-probe loss catches crude poisoning
  (label-flip, gradient-ascent-disguised, broad backdoors); a *targeted* backdoor
  tuned to leave clean-probe loss unchanged can evade it. This raises the bar,
  it doesn't close the threat.
- **Coverage rides audit sampling/oversupply**, exactly like Phase 3c — an
  un-audited contribution isn't probed. Not a per-contribution guarantee.
- **Efficacy/convergence rides the §0f run**, like all of robustness; the unit
  tests validate the screen primitive, not end-to-end training under attack.
