"""Per-peer cryptographic identity (internet-scale plan, Phase 1).

The HMAC auth proves possession of a *shared* secret — fine for a trusted
cluster, unworkable for a volunteer swarm where each peer needs its own
revocable identity and where directory records must be relayable without
trusting the relayer. This module adds:

* :class:`PeerIdentity` — an Ed25519 keypair. The **peer id** is
  ``sha256(raw public key)`` (hex), so identities are self-derived and
  collision-resistant; the private key lives in a PEM file (mode 0600).
* **Challenge auth** — :meth:`PeerIdentity.auth_response` signs a server's
  nonce (domain-separated), and :func:`verify_auth` checks it against the raw
  public key. The reactor accepts this alongside HMAC: HMAC remains the
  bootstrap/enrollment-token mechanism, identity is the per-peer one.
* **Signed records** — :func:`sign_record` / :func:`verify_record` make a
  JSON-scalar dict *self-certifying*: the signature covers a canonical
  encoding and embeds the public key, so a record fetched from an untrusted
  relay (or a stale tracker cache) verifies on its own. The tracker's
  directory is built from these.

Crypto comes from the ``cryptography`` package (already the optional
``[launch]`` extra used by TLS cert generation); it is imported lazily so the
core engine keeps zero hard dependencies on it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

AUTH_CONTEXT = b"opendipaco-peer-auth:"      # domain separation for challenge signatures
RECORD_CONTEXT = b"opendipaco-peer-record:"  # ... and for signed directory records

_CRYPTO_HINT = (
    "Peer identity needs the 'cryptography' package. Install the launch extra:\n"
    '    pip install -e ".[launch]"'
)


def _ed25519():
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_CRYPTO_HINT) from e
    return ed25519


def peer_id_of(public_key_hex: str) -> str:
    """The peer id derived from a raw Ed25519 public key (hex): sha256 of it."""
    return hashlib.sha256(bytes.fromhex(public_key_hex)).hexdigest()


class PeerIdentity:
    """An Ed25519 keypair: this peer's durable, revocable identity."""

    def __init__(self, private_key):
        self._key = private_key
        raw = self._key.public_key().public_bytes_raw()
        self.public_key_hex: str = raw.hex()
        self.peer_id: str = peer_id_of(self.public_key_hex)

    # -- create / persist -----------------------------------------------------
    @classmethod
    def generate(cls) -> "PeerIdentity":
        return cls(_ed25519().Ed25519PrivateKey.generate())

    @classmethod
    def load(cls, path) -> "PeerIdentity":
        from cryptography.hazmat.primitives import serialization

        data = Path(path).read_bytes()
        key = serialization.load_pem_private_key(data, password=None)
        if not isinstance(key, _ed25519().Ed25519PrivateKey):
            raise ValueError(f"{path} is not an Ed25519 private key")
        return cls(key)

    def save(self, path) -> str:
        """Write the private key as PEM with owner-only permissions."""
        from cryptography.hazmat.primitives import serialization

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pem = self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            # The mode arg above only applies when the file is *created*; an
            # overwritten pre-existing file would keep its old (possibly
            # world-readable) permissions, so clamp explicitly.
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
        except OSError:
            os.close(fd)
            raise
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
        return str(path)

    # -- challenge auth (the reactor handshake) --------------------------------
    def auth_response(self, nonce: bytes) -> dict:
        """Answer a server's auth challenge: prove this identity signed the nonce."""
        sig = self._key.sign(AUTH_CONTEXT + nonce)
        return {"pub": self.public_key_hex, "sig": sig.hex()}

    # -- record signing ---------------------------------------------------------
    def sign(self, data: bytes) -> bytes:
        return self._key.sign(data)


def _verify(public_key_hex: str, signature: bytes, data: bytes) -> bool:
    try:
        key = _ed25519().Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(signature, data)
        return True
    except Exception:
        return False


def verify_auth(pub: str, sig_hex: str, nonce: bytes) -> bool:
    """Server side of the challenge: did ``pub`` sign this nonce?"""
    try:
        return _verify(pub, bytes.fromhex(sig_hex), AUTH_CONTEXT + nonce)
    except (ValueError, TypeError):
        return False


# -- self-certifying records ------------------------------------------------------


def _canonical(record: dict) -> bytes:
    """Canonical bytes of a record's signed fields (everything but pub/sig)."""
    body = {k: v for k, v in record.items() if k not in ("pub", "sig")}
    return RECORD_CONTEXT + json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sign_record(identity: PeerIdentity, record: dict) -> dict:
    """Return ``record`` + ``peer_id``/``pub``/``sig`` so it verifies on its own.

    The record must contain only JSON scalars/lists/dicts (it travels through
    the wire codec's JSON structure and is canonically re-encoded to verify).
    """
    out = dict(record)
    out["peer_id"] = identity.peer_id
    out["pub"] = identity.public_key_hex
    out["sig"] = identity.sign(_canonical(out)).hex()
    return out


def verify_record(record) -> bool:
    """A record is valid iff its signature matches its embedded public key and
    its ``peer_id`` is honestly derived from that key (so an attacker can't
    sign someone else's id with their own key)."""
    if not isinstance(record, dict):
        return False
    pub, sig = record.get("pub"), record.get("sig")
    if not isinstance(pub, str) or not isinstance(sig, str):
        return False
    try:
        if record.get("peer_id") != peer_id_of(pub):
            return False
        return _verify(pub, bytes.fromhex(sig), _canonical(record))
    except (ValueError, TypeError):
        return False
