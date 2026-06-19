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

### Slice b — wire the mechanism into the audit checker + resolution — **landed**
- `Contribution` carries `probe_before`/`probe_after`; `_train_path(probe=)`
  measures the clean-probe loss on the freshly-composed model **before** the
  inner steps (== base) and on the trained model **after** (sidesteps the
  `dedup_private` base-recompose hazard). No-op when no probe.
- The checker (`check_only` path) builds a `TrustedProbe` from the task's probe
  tensor (cast `long`) and reports `(before, after)`; only checks carry the probe
  (the primary's own probe would be self-reported, untrusted).
- The owner's `_resolve_audit_locked` runs the screen **independently of the
  digest tally**: a quorum (`probe_quorum`, 0 = off) of checkers reporting
  `is_harmful` records a `poison_flagged` metric, and — opt-in via `probe_debit`
  — debits the primary. **Post-hoc like the digest audit (no rollback):** in the
  sharded path the data is server-supplied, so a harmful verdict is a
  *corpus-poisoning alarm*, not grounds to blame the faithful worker — hence
  `probe_debit` defaults off (it's only correct in the worker-chose-data model).
- `Scheduler(probe=, probe_quorum=, probe_abs_margin=, probe_rel_margin=,
  probe_debit=)`; `probe_quorum=0` (default) is byte-identical to the pre-W8
  audit. Tests in `tests/test_redundancy.py`.

### Slice b/c boundary
Operator-facing plumbing — the `robustness.probe_*` launch config, loading the
probe from a held-out source, and carrying it through the manifest — moves to
slice c with the validation arm (it's the operator surface, not the mechanism).
- **Wiring note (from slice-a review):** the checker reproduces via
  `worker._train_path(...)`, which builds the `PathModel` *internally* and returns
  only a `Contribution` — the trained model never escapes, and no *base* model is
  composed at audit time. So slice b can't just call `probe_loss(model)` at the
  call site; the clean shape is to thread an optional probe into `_train_path` so
  it measures **base loss before the inner steps** and **trained loss after** on
  the same composed model, and stashes the `(before, after)` pair on the
  `Contribution`. (Watch the `dedup_private` path, which mutates private modules
  in place — take the base measurement before training there.)
- **Caveats to honor in slice b:** measure the probe in the *same numeric regime*
  as training (under `inner_autocast`, probe under autocast too, else the margin
  is calibrated on a different scale); validate the probe's `seq_len` against
  `max_position_embeddings` (an over-long probe gets OOD RoPE positions, silent
  garbage loss); ensure the probe tensor is `long` after manifest/wire
  round-trip (the embedding lookup needs it).

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
