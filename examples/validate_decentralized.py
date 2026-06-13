"""Byzantine-owner validation for Phase 4 quorum reads (plan §1.4 / Phase 4 D4).

Phase 4 removes the trusted scheduler, so a key's value is whatever a majority
of its ``k`` replica owners agree on — never a single owner's word. This harness
measures that property directly, at the digest level (no networking, so it is
deterministic and fast), the same way ``validate_robustness.py`` measures the
aggregation primitive.

Each round: ``K`` owners hold a key. Honest owners hold the *same* bytes (they
replicated/recomputed the same aggregate), so their ``state_digest`` agrees; a
fraction are Byzantine and serve corrupted bytes. We then check, over the
owners' ``(version, digest)`` reports:

  * a naive reader (trusts one owner) can be poisoned — how often it reads bad
    bytes is just the Byzantine fraction;
  * a quorum reader (``confirm_version``) returns the honest majority's digest
    whenever Byzantine owners are a minority, and never the poison;
  * the cross-owner audit (``divergent_peers``) flags exactly the Byzantine
    owners (toward eviction), with no honest owner falsely blamed.

Env-overridable:

    python examples/validate_decentralized.py
    K=5 BYZ=2 ROUNDS=500 python examples/validate_decentralized.py

HONEST CAVEAT: this validates the *read-side quorum + divergence-detection
primitive*, not end-to-end decentralized training. The full verdict — that the
push-to-all-``k`` write path (each owner independently aggregating, so a
Byzantine *primary* can't poison via replication) converges comparably to the
central path and degrades gracefully under a Byzantine owner — is the WAN run
(plan slice 0f), which this harness de-risks but does not replace.
"""

from __future__ import annotations

import os

import torch

from opendipaco.schedule.compress import state_digest
from opendipaco.schedule.quorum import confirm_version, divergent_peers


def _i(name, default):
    return int(os.environ.get(name, default))


K = _i("K", 3)               # replica owners per key
BYZ = _i("BYZ", 1)           # Byzantine owners (a minority when BYZ < K/2)
ROUNDS = _i("ROUNDS", 300)
DIM = _i("DIM", 1024)
QUORUM = _i("QUORUM", (K // 2) + 1)   # majority
SEED = _i("SEED", 0)


def main() -> None:
    gen = torch.Generator().manual_seed(SEED)
    naive_poisoned = 0      # a single-owner reader gets bad bytes
    quorum_poisoned = 0     # the quorum reader gets bad bytes
    quorum_blind = 0        # the quorum reader can't confirm any version
    flagged_exactly = 0     # the audit flagged exactly the Byzantine set
    for _ in range(ROUNDS):
        honest = {"w": torch.randn(DIM, DIM // 16, generator=gen)}
        honest_d = state_digest(honest)
        version = (0, 1)
        peers = [f"o{i}" for i in range(K)]
        byz = set(peers[:BYZ])
        reports = {}
        for p in peers:
            if p in byz:
                bad = {"w": honest["w"] + torch.randn(DIM, DIM // 16, generator=gen)}
                reports[p] = (version, state_digest(bad))
            else:
                reports[p] = (version, honest_d)

        # Naive reader: trust a uniformly-random single owner.
        pick = peers[int(torch.randint(K, (1,), generator=gen))]
        if reports[pick][1] != honest_d:
            naive_poisoned += 1

        # Quorum reader.
        confirmed = confirm_version(list(reports.values()), QUORUM)
        if confirmed is None:
            quorum_blind += 1
        elif confirmed[1] != honest_d:
            quorum_poisoned += 1

        # Cross-owner audit: who is flagged divergent?
        if divergent_peers(reports, confirmed) == byz:
            flagged_exactly += 1

    print(f"owners K={K}  byzantine={BYZ}  quorum={QUORUM}  rounds={ROUNDS}")
    print(f"  naive reader  (1 owner)  poisoned reads = {naive_poisoned / ROUNDS:.0%}")
    print(f"  quorum reader            poisoned reads = {quorum_poisoned / ROUNDS:.0%}"
          f"   (unconfirmable rounds = {quorum_blind / ROUNDS:.0%})")
    print(f"  audit flagged exactly the byzantine set  = {flagged_exactly / ROUNDS:.0%}")
    if BYZ < K / 2:
        ok = quorum_poisoned == 0 and flagged_exactly == ROUNDS
        print(f"  minority byzantine -> quorum must resist + flag: "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        print("  byzantine majority is OUT OF SCOPE (no permissionless protocol "
              "survives >50%); expect the quorum reader to be defeated.")


if __name__ == "__main__":
    main()
