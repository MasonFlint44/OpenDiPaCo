"""W8a: the trusted-probe screening primitive (docs/w8-data-poisoning-design.md).

probe_loss(model, sequences) is the model's clean-probe loss; is_harmful(before,
after) is the owner-side verdict that an update raised probe loss beyond a margin.
A poisoned update that passed every weight-space check still raises clean-probe
loss -- that's the signal these catch.
"""

import math

import torch
from torch import nn

from opendipaco.schedule.probe import TrustedProbe, is_harmful, probe_loss

VOCAB = 8


class _StubLM(nn.Module):
    """A toy LM whose logits put (almost) all mass on a fixed token, so its loss is
    low exactly when the probe's targets are that token -- lets us control probe
    loss without training a real path."""

    def __init__(self, favored_token: int):
        super().__init__()
        self.favored = favored_token
        self._p = nn.Parameter(torch.zeros(1))  # gives the module a device/params

    def forward(self, x):
        logits = torch.full((*x.shape, VOCAB), -5.0)
        logits[..., self.favored] = 5.0
        return logits + self._p, None


def _probe(token: int, n: int = 4, length: int = 6) -> torch.Tensor:
    return torch.full((n, length), token, dtype=torch.long)


def test_probe_loss_low_when_model_predicts_the_targets():
    # Probe of all-token-3 sequences: a model favoring 3 scores them well.
    good = probe_loss(_StubLM(favored_token=3), _probe(3))
    bad = probe_loss(_StubLM(favored_token=5), _probe(3))   # predicts the wrong token
    assert math.isfinite(good) and math.isfinite(bad)
    assert good < bad                                       # wrong-token model loses more


def test_probe_loss_empty_is_nan():
    assert math.isnan(probe_loss(_StubLM(0), torch.empty(0, 6, dtype=torch.long)))
    assert math.isnan(probe_loss(_StubLM(0), None))
    # A length-1 sequence has no (prediction, target) pair -> nothing scorable.
    assert math.isnan(probe_loss(_StubLM(0), torch.zeros(4, 1, dtype=torch.long)))


def test_probe_loss_restores_train_mode():
    m = _StubLM(0)
    m.train()
    probe_loss(m, _probe(0))
    assert m.training is True
    m.eval()
    probe_loss(m, _probe(0))
    assert m.training is False


def test_is_harmful_flags_a_loss_increase_beyond_margin():
    assert is_harmful(1.0, 2.0) is True                     # big rise -> harmful
    assert is_harmful(1.0, 1.0) is False                    # no change
    assert is_harmful(1.0, 0.5) is False                    # improvement
    # Within the margin (abs 0.05 + rel 0.02*1.0 = 0.07) -> not harmful.
    assert is_harmful(1.0, 1.05) is False
    assert is_harmful(1.0, 1.10) is True


def test_is_harmful_margin_scales_with_loss():
    # rel_margin makes the tolerated rise scale with the base loss.
    assert is_harmful(100.0, 101.0) is False                # 1% rise on a big loss -> noise
    assert is_harmful(100.0, 105.0) is True                 # 5% rise -> harmful


def test_is_harmful_nan_is_not_harmful():
    # Can't screen (empty/failed probe) -> don't block (documented coverage caveat).
    assert is_harmful(float("nan"), 2.0) is False
    assert is_harmful(1.0, float("nan")) is False
    assert is_harmful(float("inf"), float("inf")) is False


def test_trusted_probe_holder():
    probe = TrustedProbe(_probe(3), batch_size=2)
    assert probe.empty is False
    assert math.isfinite(probe.loss(_StubLM(3)))
    assert TrustedProbe(None).empty is True
    assert TrustedProbe(torch.empty(0, 6, dtype=torch.long)).empty is True


def test_poisoned_update_is_caught_by_the_screen():
    # End-to-end primitive flow: base model is decent on the clean probe; the
    # "poisoned" update degrades it -> the before/after screen flags it, while an
    # honest (improving) update passes.
    probe = TrustedProbe(_probe(3))
    base = probe.loss(_StubLM(favored_token=3))             # decent
    poisoned = probe.loss(_StubLM(favored_token=6))         # degraded by the update
    honest = probe.loss(_StubLM(favored_token=3))           # unchanged/good
    assert is_harmful(base, poisoned) is True
    assert is_harmful(base, honest) is False
