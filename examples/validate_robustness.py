"""Adversarial-robustness validation for Phase 3 aggregation (plan §1.4 / §1.1).

Phase 3 is the first phase whose *point* is a dynamics property, so green unit
tests aren't the verdict — this harness measures the property directly: with a
fraction of Byzantine contributors, how far does the applied update drift from
the honest-only update under ``robustness: off`` (plain weighted mean, today)
versus ``on`` (coordinate-wise trimmed mean / median)?

It runs at the aggregator level (no networking) so it is deterministic and
fast. Each round, honest contributors produce a shared target direction plus
small noise; adversaries produce an attack (sign-flip, large-norm, or
Gaussian); both aggregators combine them, and we report the mean L2 distance of
the result from the honest target. Env-overridable:

    python examples/validate_robustness.py
    ADV_FRAC=0.4 N_CONTRIB=5 ATTACK=largenorm AGG=median python examples/validate_robustness.py

Note on the aggregates: ``trimmed_mean`` (the default) drops exactly *one*
extreme per coordinate per side, so its breakdown point is one adversary
regardless of contributor count — at the default quorum cap of 3 it coincides
with the median. ``median`` tolerates a true (up to ~half) minority and is the
stronger knob when more than one adversary can reach a module; this harness
makes the difference visible (try ``N_CONTRIB=5 ADV_FRAC=0.4`` with each).

HONEST CAVEAT: this validates the *aggregation primitive's* breakdown
behavior, not end-to-end training convergence. The full convergence verdict —
that an `on` run trains comparably with no adversaries and degrades gracefully
with them — is a WAN training run (plan slice 0f), which this harness's
findings de-risk but do not replace.
"""

from __future__ import annotations

import os

import torch

from opendipaco.schedule.aggregate import robust_delta


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


N_CONTRIB = _i("N_CONTRIB", 3)        # contributors per round (= default quorum cap)
ADV_FRAC = _f("ADV_FRAC", 0.34)       # fraction Byzantine (-> 1 of 3, a minority)
ROUNDS = _i("ROUNDS", 200)
DIM = _i("DIM", 4096)
ATTACK = os.environ.get("ATTACK", "signflip")   # signflip | largenorm | gaussian
AGG = os.environ.get("AGG", "trimmed_mean")     # trimmed_mean | median
NOISE = _f("NOISE", 0.1)              # honest per-contributor noise
SEED = _i("SEED", 0)


def _attack(target: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    if ATTACK == "signflip":
        return -target * 5.0                       # push hard the wrong way
    if ATTACK == "largenorm":
        return torch.randn(DIM, generator=gen) * 50.0
    return torch.randn(DIM, generator=gen) * 3.0   # gaussian junk


def _mean_delta(contribs):
    """The ``robustness: off`` baseline: plain weighted mean (today's behavior)."""
    return robust_delta(contribs, aggregate="mean")[0]


def main() -> None:
    gen = torch.Generator().manual_seed(SEED)
    n_adv = round(N_CONTRIB * ADV_FRAC)
    off_err = on_err = 0.0
    for _ in range(ROUNDS):
        target = torch.randn(DIM, generator=gen)   # the honest direction this round
        contribs = []
        for i in range(N_CONTRIB):
            if i < n_adv:
                g = _attack(target, gen)
            else:
                g = target + torch.randn(DIM, generator=gen) * NOISE
            contribs.append((1.0, [g]))
        # Compare each aggregate's *direction* against the honest target,
        # normalizing out the summed-weight magnitude.
        off = _mean_delta(contribs)[0]               # delta is a per-param list
        on = robust_delta(contribs, aggregate=AGG)[0][0]
        off_err += float((off - target).norm() / target.norm())
        on_err += float((on - target).norm() / target.norm())
    off_err /= ROUNDS
    on_err /= ROUNDS

    print(f"contributors={N_CONTRIB}  adversaries={n_adv} ({ADV_FRAC:.0%})  "
          f"attack={ATTACK}  aggregate={AGG}  rounds={ROUNDS}")
    print(f"  off (plain mean)     mean ‖agg-target‖/‖target‖ = {off_err:.3f}")
    print(f"  on  ({AGG:<12})  mean ‖agg-target‖/‖target‖ = {on_err:.3f}")
    if n_adv == 0:
        print("  (no adversaries: both should be ~the honest noise floor)")
    elif n_adv < N_CONTRIB / 2:
        verdict = "PASS" if on_err < off_err * 0.5 else "WEAK"
        print(f"  minority adversaries -> robust aggregate should resist: {verdict} "
              f"({off_err / max(on_err, 1e-9):.1f}x closer)")
    else:
        print("  adversaries are a MAJORITY -> out of the threat model (D1); "
              "no aggregate can save this. Expect both to be poor.")


if __name__ == "__main__":
    main()
