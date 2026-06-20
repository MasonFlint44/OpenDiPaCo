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
from .ownership import epoch_newer, verify_epoch_record
from .reactor import DEFAULT_MAX_MSG_BYTES, _ReactorServer
from .sharded import _ps_connect, _rpc

REACHABILITY = ("public", "nat")


def make_peer_record(identity: PeerIdentity, *, reachability: str = "nat",
                     addr=None, roles=(), capabilities=None, circuit_addrs=None) -> dict:
    """Build + sign this peer's directory record (``issued_at`` = now).

    ``"public"`` reachability requires ``addr=(host, port)`` — that is the
    address other peers will dial; ``"nat"`` peers carry no direct address (they
    only dial out), but may advertise ``circuit_addrs`` — ``/p2p-circuit``
    multiaddrs through relays they've reserved on — so a NAT'd peer can be
    reached (and serve as an owner) through a relay (W1c).
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
    if circuit_addrs:  # omitted when absent, so non-relay records are unchanged
        record["circuit_addrs"] = list(circuit_addrs)
    return sign_record(identity, record)


# Tracker frames are JSON records, never tensors -- a much smaller message cap
# than the transport default (4 GiB, sized for weights) bounds what one peer can
# make the tracker buffer.
TRACKER_MAX_MSG_BYTES = 16 * 1024 * 1024


class Tracker(_ReactorServer):
    """The directory server. See the module docstring for the protocol.

    ``max_peers`` bounds the directory (identities are free to generate, so an
    open-enrollment tracker would otherwise grow without limit; registrations
    beyond the cap are refused "directory full"). An imported *stale* record is
    bounded by ``ttl``: it expires like any other unless its peer heartbeats.
    """

    def __init__(self, *, host: str = "0.0.0.0", port: int = 0, ttl: float = 120.0,
                 open_enrollment: bool = False, enroll_peers=None, auth_key=None,
                 max_peers: int = 65536, max_msg_bytes: int = TRACKER_MAX_MSG_BYTES,
                 epoch_signer=None, **reactor_kw):
        super().__init__(host=host, port=port, auth_key=auth_key,
                         max_msg_bytes=max_msg_bytes, **reactor_kw)
        self.ttl = ttl
        self.max_peers = max_peers
        self.open_enrollment = open_enrollment
        self._dir_lock = threading.Lock()
        self._enrolled: set[str] = {self._pub(p) for p in (enroll_peers or [])}
        # peer_id -> {"record": dict | None (tombstone), "issued_at": float,
        #             "expires": monotonic deadline}
        self._peers: dict[str, dict] = {}
        # Cached owner-set epoch record (Phase 2a): a convenience copy of the
        # scheduler's signed record, served to bootstrapping owners. The cache
        # is not the authority -- consumers re-verify against the scheduler's
        # pub. ``epoch_signer=`` pins who may put it; if unset, the first valid
        # put pins the signer (set it explicitly on open-enrollment trackers,
        # or any enrolled peer could race to pin its own "epochs").
        self._epoch: dict | None = None
        self._epoch_signer: str | None = self._pub(epoch_signer) if epoch_signer else None
        # Optional run manifest (W6): the operator's launch config minus secrets,
        # served to flags-only `opendipaco join` clients so they can build their
        # worker without a config file. Set via `serve_manifest`; None = unset.
        self._manifest: dict | None = None

    def serve_manifest(self, manifest: dict | None) -> None:
        """Publish the run manifest this tracker serves to joining workers (W6)."""
        self._manifest = manifest

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
    def _handle(self, msg: dict, nbytes: int, peer_id: str | None = None):
        kind = msg.get("type")
        if kind == "register":
            return self._register(msg.get("record"))
        if kind == "deregister":
            return self._deregister(msg.get("record"))
        if kind == "directory":
            return self._directory(msg)
        if kind == "import":
            return self._import(msg.get("records") or [])
        if kind == "epoch_put":
            return self._epoch_put(msg.get("record"))
        if kind == "epoch_get":
            with self._dir_lock:
                return {"type": "epoch", "record": self._epoch}
        if kind == "manifest":   # W6: serve the run manifest to a joining worker
            return {"type": "manifest", "manifest": self._manifest}
        return None

    def _epoch_put(self, record) -> dict:
        if not verify_epoch_record(record):
            return {"type": "refused", "reason": "bad epoch record"}
        pub = record["pub"].lower()
        with self._dir_lock:
            if self._epoch_signer is not None and pub != self._epoch_signer:
                return {"type": "refused", "reason": "wrong signer"}
            if not epoch_newer(record, self._epoch):
                return {"type": "refused", "reason": "stale epoch"}
            self._epoch_signer = pub  # first valid put pins the signer
            self._epoch = record
        return {"type": "epoch_cached"}

    def _register(self, record) -> dict:
        ok, reason = self._admit(record)
        if not ok:
            return {"type": "refused", "reason": reason}
        with self._dir_lock:
            self._purge_locked()
            entry = self._peers.get(record["peer_id"])
            if entry is not None and record["issued_at"] <= entry["issued_at"]:
                return {"type": "refused", "reason": "stale record"}
            if entry is None and len(self._peers) >= self.max_peers:
                return {"type": "refused", "reason": "directory full"}
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
            self._tombstone_locked(record["peer_id"], record.get("issued_at", time.time()),
                                   deregister=record)
        return {"type": "deregistered"}

    def _directory(self, msg: dict) -> dict:
        roles = set(msg.get("roles") or [])
        reachability = msg.get("reachability")
        with self._dir_lock:
            self._purge_locked()
            records = [e["record"] for e in self._peers.values() if e["record"] is not None]
            # Tombstones (explicit signed deregistrations still within TTL): the
            # scheduler uses these to fail a graceful leave over *now*, skipping
            # owner_grace (W4b). We surface the **signed deregister record**, not a
            # bare peer id, so the consumer re-verifies it (a compromised/relayed
            # tracker can't fabricate a fast-eviction -- the self-certifying
            # principle). Tracker-initiated expel tombstones carry no signed
            # record, so they are *not* surfaced and fall back to TTL+grace. Only
            # sent when asked; atomic with ``records`` under the directory lock.
            tombs = ([e["deregister"] for e in self._peers.values()
                      if e["record"] is None and e.get("deregister")]
                     if msg.get("include_tombstones") else None)
        if roles:
            records = [r for r in records if roles & set(r.get("roles") or [])]
        if reachability:
            records = [r for r in records if r.get("reachability") == reachability]
        reply = {"type": "directory", "records": records}
        if tombs is not None:
            reply["tombstones"] = tombs
        return reply

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

    def _tombstone_locked(self, peer_id: str, issued_at: float, deregister=None) -> None:
        # ``deregister`` is the peer's own signed deregister record (None for a
        # tracker-initiated expel). Retained so the directory can surface a
        # *verifiable* tombstone for fast-eviction (W4b); expel tombstones carry
        # none and so never trigger a grace-skipping eviction downstream.
        self._peers[peer_id] = {"record": None, "deregister": deregister,
                                "issued_at": issued_at,
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

    def tombstones(self) -> list[str]:
        """Peer ids with a live (within-TTL) explicit deregistration (W4b)."""
        with self._dir_lock:
            self._purge_locked()
            return [pid for pid, e in self._peers.items() if e["record"] is None]


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


def fetch_directory_and_tombstones(addr, *, roles=None, reachability=None, auth_key=None,
                                   tls=None, timeout: float = 10.0, verify: bool = True):
    """Like :func:`fetch_directory` but also returns the explicit-deregistration
    tombstones (one atomic round trip). The scheduler's epoch watcher uses them
    to fail a graceful leave over immediately, skipping ``owner_grace`` (W4b).

    Returns ``(records, {peer_id: deregister_issued_at})``. Each tombstone is the
    peer's own signed deregister, verified here (signature + honest peer_id +
    kind) so a fabricated/relayed one can't fast-evict; its ``issued_at`` is
    carried so the consumer can ignore a **stale** replay (a tombstone older
    than the peer's current registration -- e.g. an owner that left and rejoined)
    rather than evicting a live peer."""
    reply = tracker_rpc(addr, {"type": "directory", "roles": list(roles or []),
                               "reachability": reachability, "include_tombstones": True},
                        auth_key=auth_key, tls=tls, timeout=timeout) or {}
    records = reply.get("records") or []
    tombs = reply.get("tombstones") or []
    if verify:
        records = [r for r in records if verify_record(r)]
        tombs = [t for t in tombs if verify_record(t) and t.get("kind") == "deregister"]
    out = {}
    for t in tombs:
        ts = t.get("issued_at")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            out[t["peer_id"]] = float(ts)
    return records, out


def _record_issued_at(r):
    ts = r.get("issued_at")
    return float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else None


def fetch_directory_multi(seeds, *, roles=None, reachability=None, auth_key=None,
                          tls=None, timeout: float = 10.0):
    """Bootstrap the directory from **multiple seeds**, returning the **union** of
    verified records (freshest per ``peer_id``) with verified deregister tombstones
    suppressing same-or-older registrations (W8 eclipse defense; design
    ``docs/w8-eclipse-sybil-design.md``).

    Records are self-certifying, so the union is sound regardless of which seed
    served them: omission-eclipse needs *every* seed to withhold a peer, and one
    honest+reachable seed restores it. Unreachable/erroring/malicious seeds are
    skipped (best-effort -> also better availability). ``seeds`` is an iterable of
    ``(host, port)`` (a trailing pinned pubkey, if present, is ignored here --
    pinning is a transport-auth concern, not what makes the union sound). Returns
    ``(records, seeds_answered)``.
    """
    merged: dict[str, dict] = {}    # peer_id -> freshest verified record
    tombs: dict[str, float] = {}    # peer_id -> max verified deregister issued_at
    answered = 0
    for seed in seeds:
        addr = (seed[0], seed[1])
        try:
            records, t = fetch_directory_and_tombstones(
                addr, roles=roles, reachability=reachability,
                auth_key=auth_key, tls=tls, timeout=timeout)
        except Exception:
            # A seed is untrusted and best-effort: skip ANY failure (unreachable,
            # a malformed/garbage reply that escapes the codec as struct.error /
            # AttributeError / RuntimeError, etc.) rather than let one bad seed
            # crash the caller. Records that DO come through are verify_record'd.
            continue
        answered += 1
        for pid, ts in t.items():
            if pid not in tombs or ts > tombs[pid]:
                tombs[pid] = ts
        for r in records:           # already verify_record-checked by the fetch
            pid, ts = r.get("peer_id"), _record_issued_at(r)
            if not isinstance(pid, str) or ts is None:
                continue
            cur = merged.get(pid)
            if cur is None or ts > _record_issued_at(cur):
                merged[pid] = r
    # A tombstone suppresses a registration at or before its issued_at, so a
    # malicious seed can't resurrect a departed peer by replaying its still-within-
    # TTL record; a peer that left then rejoined (newer registration) survives.
    out = [r for pid, r in merged.items()
           if pid not in tombs or _record_issued_at(r) > tombs[pid]]
    return out, answered


def import_records(addr, records, *, auth_key=None, tls=None, timeout: float = 10.0) -> dict:
    """Push relayed records into a tracker (bootstrap a replacement from a cache)."""
    return tracker_rpc(addr, {"type": "import", "records": list(records)},
                       auth_key=auth_key, tls=tls, timeout=timeout)


def put_epoch(addr, record, *, auth_key=None, tls=None, timeout: float = 10.0) -> dict:
    """Cache the scheduler's signed epoch record on a tracker (Phase 2a)."""
    return tracker_rpc(addr, {"type": "epoch_put", "record": record},
                       auth_key=auth_key, tls=tls, timeout=timeout)


def get_epoch(addr, *, signer_pub=None, auth_key=None, tls=None,
              timeout: float = 10.0) -> dict | None:
    """Fetch the cached epoch record; re-verified locally before it is returned
    (pass ``signer_pub`` — the scheduler's public key — to also pin the signer,
    since the tracker cache is a convenience, not the authority)."""
    reply = tracker_rpc(addr, {"type": "epoch_get"},
                        auth_key=auth_key, tls=tls, timeout=timeout)
    record = (reply or {}).get("record")
    if record is None or not verify_epoch_record(record, signer_pub=signer_pub):
        return None
    return record
