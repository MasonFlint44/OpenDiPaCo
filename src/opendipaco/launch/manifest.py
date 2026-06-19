"""Run manifest — what a flags-only ``opendipaco join`` fetches from the swarm
(W6, design ``docs/w6-client-design.md`` D2).

A volunteer with only ``--tracker``/``--scheduler`` + a credential does not have
the model architecture, and a worker can't build its ``PathModel`` without it.
So the operator's server publishes a **signed run manifest** — the full launch
config **minus secrets** (no auth keys, no TLS key material, no weights) — and
the joining worker fetches it, verifies it, and rebuilds a :class:`LaunchConfig`
locally, overlaying only the connection flags it was given.

Trust: the manifest is :func:`~opendipaco.schedule.identity.sign_record`-signed
by the run's identity when the operator runs with one. A joiner may **pin** the
key (``--server-pub``, verify-or-refuse) or accept it **TOFU**.

TOFU is genuinely weak on an unauthenticated/unencrypted channel: a MITM (or a
malicious tracker) can rewrite an unsigned manifest, or even strip the signature
off a signed one — the joiner can't tell a sig-stripped manifest from a
genuinely-unsigned run. So TOFU is *use-at-your-own-risk*; **pin with
``--server-pub`` or run over TLS on an untrusted network.** A wrong manifest
can't poison the *weights* (those stay grant/quorum-gated), but it can waste the
volunteer's compute and steer side-effects the worker config controls (which data
to materialize, which relays to dial) — which is why operator-local paths are
stripped (see ``_STRIP``) and pinning is the safe default to recommend.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import time

from ..schedule.identity import verify_record
from .config import LaunchConfig

MANIFEST_KIND = "run_manifest"

# Operator-only / secret / operator-local config the manifest must never carry.
# Public keys a worker *needs* (``transport.scheduler_pub`` to verify grants) are
# deliberately kept. Beyond credentials and TLS key material, **operator-local
# filesystem paths** are stripped too: they are meaningless on a volunteer's box,
# and under TOFU an attacker-served manifest could otherwise point the joiner's
# ``torch.save``/cache at an attacker-chosen path (a write sink). The joiner
# supplies its own cache via ``--data-dir``.
_STRIP = {
    # Credentials + volunteer-local resource knobs. These describe the *joiner's
    # own* hardware budget (its bandwidth, VRAM, RAM), not a property of the run,
    # so carrying the operator's value would silently constrain volunteers --
    # max_mbps -> the joiner's `--max-mbps`; worker_max_batch -> its VRAM batch
    # cap; worker_max_shards -> its resident-shard RAM cap (the joiner sets these
    # via flags/config, falling back to the library default, never the operator's).
    "transport": ("auth_key", "grant_key", "identity_key", "accept_keys",
                  "admitted_peers", "max_mbps"),
    "run": ("worker_max_batch", "worker_max_shards", "verify_routing"),
    "tls": ("certfile", "keyfile", "cafile"),
    "data": ("cache_path", "shard_cache_dir"),
}


def build_manifest(cfg: LaunchConfig, *, identity=None) -> dict:
    """The public run manifest for ``cfg``: the launch config minus secrets,
    signed when ``identity`` is given (else an unsigned, TOFU-only manifest)."""
    body = dataclasses.asdict(cfg)
    for section, keys in _STRIP.items():
        for k in keys:
            if k in body.get(section, {}):
                body[section][k] = [] if isinstance(body[section][k], list) else None
    record = {"kind": MANIFEST_KIND, "config": body, "issued_at": time.time()}
    if identity is not None:
        from ..schedule.identity import sign_record
        return sign_record(identity, record)
    return record


def manifest_fingerprint(manifest: dict) -> str:
    """A short, stable fingerprint of the manifest's *content* (config body),
    printed for out-of-band verification under TOFU. Independent of the
    signature, so a re-signed but identical config fingerprints the same."""
    body = json.dumps(manifest.get("config", {}), sort_keys=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()[:16]


def verify_manifest(manifest, *, server_pub: str | None = None) -> bool:
    """Structurally a manifest, and — when ``server_pub`` is pinned — signed by
    exactly that key. With no pin, an *unsigned* manifest is accepted (TOFU); a
    manifest that carries a signature must still have a valid one (so a tampered
    signed manifest is refused even unpinned)."""
    if not (isinstance(manifest, dict) and manifest.get("kind") == MANIFEST_KIND
            and isinstance(manifest.get("config"), dict)):
        return False
    signed = "sig" in manifest
    if server_pub is not None:
        return (signed and verify_record(manifest)
                and manifest.get("pub", "").lower() == server_pub.lower())
    return verify_record(manifest) if signed else True


def manifest_to_config(manifest: dict, *, overrides: dict | None = None) -> LaunchConfig:
    """Rebuild a :class:`LaunchConfig` from a (verified) manifest, overlaying
    ``overrides`` (the joiner's connection flags + hardware) section-by-section.

    ``overrides`` is a partial config dict, e.g.
    ``{"transport": {"connect_host": h, "port": p, "auth_key": s}, "run": {"device": "cuda"}}``.
    Reuses :meth:`LaunchConfig.from_dict` so the merged result is validated
    exactly like a file-loaded config."""
    body = copy.deepcopy(manifest["config"])
    for section, vals in (overrides or {}).items():
        body.setdefault(section, {}).update({k: v for k, v in vals.items() if v is not None})
    return LaunchConfig.from_dict(body)
