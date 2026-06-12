"""Rendezvous tracker: membership + reachability for the peer swarm (Phase 1).

The tracker is the control-plane *directory*, deliberately light (the
BitTorrent shape): peers register a **self-certifying signed record** — id,
public key, reachability tier, optionally an address, role offers, and a
capability profile — and anyone can fetch the directory. Because records are
signed by the peer itself (``identity.sign_record``), they remain trustworthy
when relayed: a client's cached directory can be **imported** into a fresh
tracker, so losing the tracker degrades the swarm instead of halting it
(bootstrap a replacement from any peer's cache).

Reachability tiers, not NAT traversal: ``"nat"`` peers are dial-out-only
clients (today's workers — they register with no address); ``"public"`` peers
advertise an address and thereby volunteer to host the P2P plane (module
owners and relays, consumed in Phase 2).

Liveness is TTL-based: a registration expires after ``ttl`` seconds unless the
peer re-registers (its heartbeat). Records are ordered by their signed
``issued_at``, so a stale relayed copy can never displace a newer one, and an
explicit (signed) deregistration leaves a tombstone that blocks re-imports of
older copies until the TTL passes.

Enrollment is the admission decision: ``open_enrollment=True`` accepts any
validly-signed record (gate influence later, by reputation — plan Phase 3);
otherwise only enrolled public keys may register (``enroll_peers=`` /
``enroll()``; ``expel()`` revokes and drops the peer). Transport-level auth
(HMAC ``auth_key`` as an enrollment token, TLS, or ``admitted_peers``)
composes underneath as usual.
"""

from __future__ import annotations

import threading
import time

from .identity import PeerIdentity, sign_record, verify_record
from .reactor import DEFAULT_MAX_MSG_BYTES, _ReactorServer
from .sharded import _ps_connect, _rpc

REACHABILITY = ("public", "nat")


def make_peer_record(identity: PeerIdentity, *, reachability: str = "nat",
                     addr=None, roles=(), capabilities=None) -> dict:
    """Build + sign this peer's directory record (``issued_at`` = now).

    ``"public"`` reachability requires ``addr=(host, port)`` — that is the
    address other peers will dial; ``"nat"`` peers carry no address (they only
    dial out).
    """
    if reachability not in REACHABILITY:
        raise ValueError(f"reachability must be one of {REACHABILITY}, got {reachability!r}")
    if reachability == "public" and not addr:
        raise ValueError("a 'public' peer must advertise addr=(host, port)")
    if reachability == "nat" and addr:
        raise ValueError("a 'nat' peer is dial-out-only; it cannot advertise an addr")
    record = {
        "kind": "peer",
        "reachability": reachability,
        "addr": list(addr) if addr else None,
        "roles": sorted(roles),
        "capabilities": dict(capabilities or {}),
        "issued_at": time.time(),
    }
    return sign_record(identity, record)


class Tracker(_ReactorServer):
    """The directory server. See the module docstring for the protocol."""

    def __init__(self, *, host: str = "0.0.0.0", port: int = 0, ttl: float = 120.0,
                 open_enrollment: bool = False, enroll_peers=None, auth_key=None,
                 **reactor_kw):
        super().__init__(host=host, port=port, auth_key=auth_key, **reactor_kw)
        self.ttl = ttl
        self.open_enrollment = open_enrollment
        self._dir_lock = threading.Lock()
        self._enrolled: set[str] = {self._pub(p) for p in (enroll_peers or [])}
        # peer_id -> {"record": dict | None (tombstone), "issued_at": float,
        #             "expires": monotonic deadline}
        self._peers: dict[str, dict] = {}

    @staticmethod
    def _pub(p) -> str:
        return getattr(p, "public_key_hex", p).lower()

    # -- enrollment -------------------------------------------------------------
    def enroll(self, pub) -> None:
        with self._dir_lock:
            self._enrolled.add(self._pub(pub))

    def expel(self, pub) -> None:
        """Revoke enrollment and immediately drop the peer's record (tombstoned,
        so a relayed copy can't be re-imported until the TTL passes)."""
        pub = self._pub(pub)
        with self._dir_lock:
            self._enrolled.discard(pub)
            for pid, entry in list(self._peers.items()):
                rec = entry.get("record")
                if rec is not None and rec.get("pub", "").lower() == pub:
                    self._tombstone_locked(pid, time.time())

    def _enrolled_ok(self, pub: str) -> bool:
        return self.open_enrollment or pub.lower() in self._enrolled

    # -- handlers ----------------------------------------------------------------
    def _handle(self, msg: dict, nbytes: int):
        kind = msg.get("type")
        if kind == "register":
            return self._register(msg.get("record"))
        if kind == "deregister":
            return self._deregister(msg.get("record"))
        if kind == "directory":
            return self._directory(msg)
        if kind == "import":
            return self._import(msg.get("records") or [])
        return None

    def _register(self, record) -> dict:
        ok, reason = self._admit(record)
        if not ok:
            return {"type": "refused", "reason": reason}
        with self._dir_lock:
            self._purge_locked()
            entry = self._peers.get(record["peer_id"])
            if entry is not None and record["issued_at"] <= entry["issued_at"]:
                return {"type": "refused", "reason": "stale record"}
            self._peers[record["peer_id"]] = {
                "record": record, "issued_at": record["issued_at"],
                "expires": time.monotonic() + self.ttl,
            }
        return {"type": "registered", "ttl": self.ttl}

    def _deregister(self, record) -> dict:
        """A signed record with ``"kind": "deregister"`` removes the peer."""
        if not verify_record(record) or record.get("kind") != "deregister":
            return {"type": "refused", "reason": "bad signature"}
        with self._dir_lock:
            entry = self._peers.get(record["peer_id"])
            if entry is not None and record.get("issued_at", 0) <= entry["issued_at"]:
                return {"type": "refused", "reason": "stale record"}
            self._tombstone_locked(record["peer_id"], record.get("issued_at", time.time()))
        return {"type": "deregistered"}

    def _directory(self, msg: dict) -> dict:
        roles = set(msg.get("roles") or [])
        reachability = msg.get("reachability")
        with self._dir_lock:
            self._purge_locked()
            records = [e["record"] for e in self._peers.values() if e["record"] is not None]
        if roles:
            records = [r for r in records if roles & set(r.get("roles") or [])]
        if reachability:
            records = [r for r in records if r.get("reachability") == reachability]
        return {"type": "directory", "records": records}

    def _import(self, records) -> dict:
        """Bulk-adopt relayed records (gossip / bootstrap-from-a-peer's-cache)."""
        accepted = 0
        for rec in records:
            if self._register(rec).get("type") == "registered":
                accepted += 1
        return {"type": "imported", "accepted": accepted}

    # -- internals ------------------------------------------------------------------
    def _admit(self, record) -> tuple[bool, str]:
        if not verify_record(record):
            return False, "bad signature"
        if record.get("kind") != "peer":
            return False, "not a peer record"
        if not self._enrolled_ok(record["pub"]):
            return False, "not enrolled"
        if record.get("reachability") not in REACHABILITY:
            return False, "bad reachability"
        if record["reachability"] == "public" and not record.get("addr"):
            return False, "public peer without addr"
        if not isinstance(record.get("issued_at"), (int, float)):
            return False, "missing issued_at"
        return True, ""

    def _tombstone_locked(self, peer_id: str, issued_at: float) -> None:
        self._peers[peer_id] = {"record": None, "issued_at": issued_at,
                                "expires": time.monotonic() + self.ttl}

    def _purge_locked(self) -> None:
        now = time.monotonic()
        for pid in [p for p, e in self._peers.items() if now >= e["expires"]]:
            del self._peers[pid]

    # -- in-process accessors ----------------------------------------------------
    def records(self) -> list[dict]:
        with self._dir_lock:
            self._purge_locked()
            return [e["record"] for e in self._peers.values() if e["record"] is not None]


# -- client side --------------------------------------------------------------------


def tracker_rpc(addr, msg, *, auth_key=None, tls=None, timeout: float = 10.0,
                max_msg_bytes: int = DEFAULT_MAX_MSG_BYTES):
    """One connect-auth-request-reply round trip to a tracker."""
    sock = _ps_connect(tuple(addr), auth_key, max_msg_bytes, timeout,
                       tls=tls, server_hostname=addr[0])
    try:
        return _rpc(sock, msg, max_msg_bytes)
    finally:
        sock.close()


def register_peer(addr, identity: PeerIdentity, *, reachability: str = "nat",
                  peer_addr=None, roles=(), capabilities=None, auth_key=None,
                  tls=None, timeout: float = 10.0) -> dict:
    """Register (or heartbeat-refresh) this peer with a tracker.

    Re-call before the returned ``ttl`` elapses to stay listed.
    """
    record = make_peer_record(identity, reachability=reachability, addr=peer_addr,
                              roles=roles, capabilities=capabilities)
    return tracker_rpc(addr, {"type": "register", "record": record},
                       auth_key=auth_key, tls=tls, timeout=timeout)


def deregister_peer(addr, identity: PeerIdentity, *, auth_key=None, tls=None,
                    timeout: float = 10.0) -> dict:
    record = sign_record(identity, {"kind": "deregister", "issued_at": time.time()})
    return tracker_rpc(addr, {"type": "deregister", "record": record},
                       auth_key=auth_key, tls=tls, timeout=timeout)


def fetch_directory(addr, *, roles=None, reachability=None, auth_key=None, tls=None,
                    timeout: float = 10.0, verify: bool = True) -> list[dict]:
    """Fetch (and by default re-verify) the tracker's directory.

    Verification makes the records trustworthy even via an untrusted relay;
    keep it on unless you have just produced the records yourself.
    """
    reply = tracker_rpc(addr, {"type": "directory", "roles": list(roles or []),
                               "reachability": reachability},
                        auth_key=auth_key, tls=tls, timeout=timeout)
    records = (reply or {}).get("records") or []
    return [r for r in records if verify_record(r)] if verify else records


def import_records(addr, records, *, auth_key=None, tls=None, timeout: float = 10.0) -> dict:
    """Push relayed records into a tracker (bootstrap a replacement from a cache)."""
    return tracker_rpc(addr, {"type": "import", "records": list(records)},
                       auth_key=auth_key, tls=tls, timeout=timeout)
