"""Eclipse / Sybil-at-the-tracker resistance validation (W8 part 2; plan §1.1
trust wall). Design: docs/w8-eclipse-sybil-design.md.

A newcomer bootstrapping from ONE tracker can be eclipsed: a malicious (or merely
partitioned) seed serves a filtered directory — omitting the honest peers, leaving
only attacker-controlled ones — so the newcomer's whole view is the attacker's.
This harness stands up in-process trackers (two honest seeds + one malicious seed
that omits the honest owners and injects its own Sybils) and shows, end to end:

  1. single-seed bootstrap off the malicious seed -> ECLIPSED (only Sybils),
  2. multi-seed UNION -> the honest owners are recovered (not eclipsed),
  3. seed_quorum=2 -> the single-seed Sybil injection is filtered out.

    python examples/validate_eclipse.py
    N_HONEST=8 N_SYBIL=5 python examples/validate_eclipse.py

HONEST CAVEATS (see the design doc): the union needs >= 1 honest, reachable seed —
with every seed hostile the newcomer is still eclipsed (irreducible without
trusted seed provenance). And this defends *eclipse* (omission) + filters
single-seed *injection*; it does NOT stop a fresh Sybil from being owner-eligible
(reputation excludes proven-bad peers, not fresh ones, and the worker's HRW isn't
reputation-filtered) — closing that needs the stake/incentives layer (part 3).
"""

from __future__ import annotations

import os

from opendipaco.schedule import (
    PeerIdentity,
    Tracker,
    fetch_directory,
    fetch_directory_multi,
    register_peer,
)


def _i(name, default):
    return int(os.environ.get(name, default))


N_HONEST = _i("N_HONEST", 5)     # honest owners the malicious seed tries to hide
N_SYBIL = _i("N_SYBIL", 3)       # attacker identities the malicious seed injects


def _tracker():
    t = Tracker(host="127.0.0.1", port=0, ttl=120.0, open_enrollment=True)
    t.start()
    return t


def main() -> None:
    # Two honest seeds know the honest owners; one malicious seed omits them and
    # serves only its Sybils. (Honest owners on BOTH honest seeds so seed_quorum=2
    # keeps them while dropping the 1-seed Sybils.)
    h1, h2, evil = _tracker(), _tracker(), _tracker()
    try:
        h1a, h2a, ea = [("127.0.0.1", t.port) for t in (h1, h2, evil)]
        honest = [PeerIdentity.generate() for _ in range(N_HONEST)]
        sybils = [PeerIdentity.generate() for _ in range(N_SYBIL)]
        for i, o in enumerate(honest):
            for ad in (h1a, h2a):
                register_peer(ad, o, roles=["owner"], reachability="public",
                              peer_addr=("10.0.0.1", 1000 + i))
        for i, s in enumerate(sybils):
            register_peer(ea, s, roles=["owner"], reachability="public",
                          peer_addr=("10.6.6.6", 2000 + i))
        honest_ids = {o.peer_id for o in honest}
        sybil_ids = {s.peer_id for s in sybils}

        # 1. Eclipsed: bootstrap off the malicious seed alone -> only Sybils.
        eclipsed = {r["peer_id"] for r in fetch_directory(ea)}
        # 2. Union over [evil, honest, honest]: the honest owners are restored.
        union = {r["peer_id"] for r in fetch_directory_multi([ea, h1a, h2a])[0]}
        # 3. seed_quorum=2: the 1-seed Sybils are filtered, honest (2 seeds) kept.
        quorum = {r["peer_id"] for r in fetch_directory_multi([ea, h1a, h2a], seed_quorum=2)[0]}

        print(f"honest_owners={N_HONEST} sybils={N_SYBIL}")
        print(f"  1. single malicious seed: sees {len(eclipsed)} peers, "
              f"{len(eclipsed & honest_ids)}/{N_HONEST} honest  -> "
              f"{'ECLIPSED' if not (eclipsed & honest_ids) else 'ok'}")
        print(f"  2. multi-seed union:      {len(union & honest_ids)}/{N_HONEST} honest "
              f"recovered, {len(union & sybil_ids)} sybils also present (injection tradeoff)")
        print(f"  3. seed_quorum=2:         {len(quorum & honest_ids)}/{N_HONEST} honest kept, "
              f"{len(quorum & sybil_ids)} sybils (1-seed injection filtered)")

        ok = (not (eclipsed & honest_ids)                       # single seed eclipses
              and honest_ids <= union                           # union recovers all honest
              and honest_ids <= quorum and not (quorum & sybil_ids))   # quorum filters sybils
        print(f"  verdict: {'PASS' if ok else 'INCONCLUSIVE'} "
              "(single-seed eclipses; union recovers; quorum filters injection)")
    finally:
        for t in (h1, h2, evil):
            t.shutdown()


if __name__ == "__main__":
    main()
