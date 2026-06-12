"""Sharded coordinator: a light Scheduler + K ParameterServers (gap #11).

The single-node :class:`~opendipaco.schedule.CoordinatorServer` holds the whole
bank. For models too big for one node, this module splits the bank across **K
ParameterServers** (each owns a disjoint shard of module keys: weights +
per-module outer optimizers + versions) coordinated by one light **Scheduler**
(task queue + async clock + staleness; *no weights*). A worker leases a path from
the scheduler, **fetches** that path's modules from the ParameterServers that own
them, trains, **commits** to the scheduler (which accepts/rejects on staleness and
returns a **commit grant** carrying the damped weight), and **pushes** the
pseudo-gradients to the owning ParameterServers, presenting the grant. Model
memory *and* weight bandwidth are sharded; the scheduler stays light.

A push without a grant is refused, the applied weight and allowed keys come from
the grant (not the worker), and each grant is single-use per server (no replay).
With ``grant_key=`` set on both the scheduler and the parameter servers, grants
are HMAC-signed so a worker that only holds a *worker* auth key cannot forge one;
keep ``grant_key`` secret from workers. Leases carry a unique token the worker
echoes on commit/heartbeat, so a reclaimed-and-re-leased path can't be committed
by a zombie worker.

This reuses the reactor (`reactor.py`), wire/auth (`wire.py`), and the worker's
warm-cache + ``AsyncScheduler._train_path`` machinery. It is scale-only and
unvalidated at toy size (same async-dynamics caveat as the single coordinator).
"""

from __future__ import annotations

import collections
import hashlib
import hmac
import json
import math
import os
import ssl
import threading
import time
import uuid

import torch

from ..model import build_module_bank
from ..optim.diloco import apply_outer_grads, make_outer_optimizer
from ..topology import is_private_key
from ..train.loop import _optimizer_state_to_cpu, _state_to_cpu
from .compress import (
    check_mode,
    compress_shard,
    compress_state,
    maybe_dequantize,
    restore_shard,
)
from .distributed import (
    _build_worker_engine,
    _commit_residuals,
    _compress_contribution,
    _load_into,
    _load_private,
    _materialize_from_spec,
)
from .guard import all_finite, clip_norm_, loss_ok
from .identity import sign_record, verify_record
from .ownership import (
    EpochManager,
    epoch_newer,
    make_epoch_record,
    owners_for,
    verify_epoch_record,
)
from .reactor import DEFAULT_MAX_MSG_BYTES, _ReactorServer
from .scheduler import AsyncScheduler
from .wire import _key_bytes, client_handshake, recv_msg, send_msg


def assign_shards(keys, num_shards: int) -> dict:
    """Assign module keys to shards round-robin over ``sorted(keys)``.

    Sorted order is stable across processes (unlike per-process-salted ``hash``),
    so scheduler, parameter servers, and workers agree on the routing.
    """
    return {k: i % num_shards for i, k in enumerate(sorted(keys))}


# -- commit grants ------------------------------------------------------------
#
# The scheduler is where staleness is decided, but the parameter servers are
# where weights live. A *grant* carries the scheduler's verdict to the servers:
# the accepted path, the damped push weight, the keys the push may touch, and
# the (unique) lease token. The weight/keys always come from the grant rather
# than the worker. Two signing modes (Phase 2a, design D3):
#
# * ``grant_key`` — HMAC over the canonical payload. Fine for operator-run
#   parameter servers, but the key must be shared with every server, so any
#   server could forge grants. Keep it secret from workers.
# * ``identity`` — the scheduler signs with its Ed25519 ``PeerIdentity``
#   (``sign_record``, ``kind="grant"``); owners verify against the scheduler's
#   *public* key (``scheduler_pub=``). Nothing secret leaves the scheduler,
#   so this is the mode for volunteer-run owners. It wins when both are set.


def _grant_payload(grant: dict) -> bytes:
    """Canonical signed bytes of a grant (everything except the mac)."""
    return json.dumps(
        {"path": list(grant["path"]), "token": grant["token"],
         "weight": grant["weight"], "keys": list(grant["keys"])},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def make_grant(path, keys, weight: float, token: str, grant_key=None, *,
               identity=None) -> dict:
    grant = {"path": list(path), "token": token, "weight": float(weight),
             "keys": sorted(keys)}
    if identity is not None:  # Ed25519 mode: verifiable with a public key only
        return sign_record(identity, {"kind": "grant", **grant})
    if grant_key is not None:
        grant["mac"] = hmac.new(
            _key_bytes(grant_key), _grant_payload(grant), hashlib.sha256
        ).hexdigest()
    return grant


def verify_grant(grant, grant_key, *, scheduler_pub: str | None = None) -> bool:
    """Structurally valid and correctly signed for the configured mode.

    With ``scheduler_pub`` set, only an Ed25519 grant signed by exactly that
    key passes (an HMAC or unsigned grant is refused — a worker must not be
    able to downgrade the check). Otherwise ``grant_key`` selects the HMAC
    check, and with neither set the check is structural only.
    """
    if not isinstance(grant, dict) or not grant.get("token"):
        return False
    if scheduler_pub is not None:
        return (verify_record(grant) and grant.get("kind") == "grant"
                and grant.get("pub", "").lower() == scheduler_pub.lower())
    try:
        payload = _grant_payload(grant)
    except (KeyError, TypeError):
        return False
    if grant_key is None:
        return True
    expected = hmac.new(_key_bytes(grant_key), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(grant.get("mac", ""), expected)


def _opt_to_wire(sd: dict) -> dict:
    """Optimizer state_dicts key their ``state`` by parameter *index* (int) --
    the wire codec only takes str dict keys, so stringify for transport."""
    return {"state": {str(i): st for i, st in sd["state"].items()},
            "param_groups": sd["param_groups"]}


def _opt_from_wire(sd: dict) -> dict:
    return {"state": {int(i): st for i, st in sd["state"].items()},
            "param_groups": sd["param_groups"]}


def _version_pair(v) -> tuple:
    """Coerce a stored version to the (epoch, counter) pair form (Phase 2b);
    pre-pair checkpoints stored bare ints, which were all epoch-0."""
    return tuple(v) if isinstance(v, (tuple, list)) else (0, int(v))


# -- parameter server --------------------------------------------------------


class ParameterServer(_ReactorServer):
    """Owns module keys: their weights, versions, and outer optimizers.

    ``fetch`` returns the requested owned weights (versioned; private only when the
    worker is cold); ``push`` applies a weighted per-module outer step to owned
    shared modules and stores owned private modules. A push must present a commit
    grant from the scheduler: the weight and allowed keys are taken from the grant,
    each grant is single-use here, and signed per the configured mode
    (``grant_key=`` HMAC, or ``scheduler_pub=`` Ed25519 -- see the grants section).

    Two ownership modes (Phase 2b, design D2/D4/D6):

    * **static** -- ``owned_keys`` fixed at launch (today's trusted-cluster
      shape); the server is primary for everything it owns and versions carry
      epoch 0.
    * **dynamic** -- pass ``identity=`` and an ``epoch_record=``: the owned set
      is *derived* (``owners_for`` over the record), the server accepts pushes
      only for keys it is **primary** for, and a replication thread pulls every
      backup/syncing key from its authoritative replica each
      ``replicate_interval`` seconds (the failover loss window). Keys gained at
      runtime (:meth:`apply_epoch`) start ``syncing`` and are served only after
      a successful pull; keys lost stay servable (lame duck). Versions are
      ``(epoch, counter)`` pairs so a promoted backup can never re-issue an old
      version number with different bytes.
    """

    _SEEN_GRANTS_MAX = 4096  # replay window; older tokens age out FIFO

    def __init__(self, config, owned_keys, diloco, *, host="0.0.0.0", port=0,
                 auth_key=None, device="cpu", resume_dir=None, grant_key=None,
                 scheduler_pub=None, max_update_norm=None, compress="none",
                 identity=None, epoch_record=None, replicate_interval=10.0,
                 peer_auth=None, peer_tls=None, bootstrap=True, bank_seed=0,
                 scheduler_addr=None, **reactor_kw):
        super().__init__(host=host, port=port, auth_key=auth_key, **reactor_kw)
        self.config = config
        self.diloco = diloco
        self.device = torch.device(device)
        self.grant_key = grant_key
        # Ed25519 grant mode: the scheduler's *public* key. When set it is the
        # only accepted grant form (HMAC grants are refused -- no downgrade).
        self.scheduler_pub = scheduler_pub
        # Non-finite pushes are always refused; the optional norm cap clips
        # oversized pseudo-gradients per module (see ``guard.py``).
        self.max_update_norm = max_update_norm
        # Downlink (fetch) compression; pushes are decoded self-describingly.
        self.compress = check_mode(compress)
        self._seen_grants: collections.OrderedDict = collections.OrderedDict()

        self.identity = identity
        self.peer_id = getattr(identity, "peer_id", None)
        self._all_keys = set(config.build_topology().module_keys())
        self._epoch = None
        self._epoch_num = 0
        if epoch_record is not None:
            if self.peer_id is None:
                raise ValueError("epoch_record= needs identity=")
            if not verify_epoch_record(epoch_record):
                raise ValueError("invalid epoch record")
            self._epoch = epoch_record
            self._epoch_num = epoch_record["epoch"]
            owned_keys = [k for k in self._all_keys
                          if self.peer_id in self._owner_ids(k, epoch_record)]
        self.owned_keys = set(owned_keys)
        # What this peer presents when *dialing* other owners for replication;
        # its identity by default, so the source can gate state on the session.
        self._peer_auth = peer_auth if peer_auth is not None else (identity or auth_key)
        self._peer_tls = peer_tls
        self._peer_conns: dict = {}
        self._repl_interval = replicate_interval
        self._repl_stop = threading.Event()
        self._repl_thread = None
        self._beat_thread = None
        # With a scheduler address the replication loop also polls for newer
        # epoch records, so ownership changes reach owners without restarts.
        self._scheduler_addr = tuple(scheduler_addr) if scheduler_addr else None
        self._prev_epoch = None  # last epoch's record: its owners are fallback pull sources

        # Build the bank as a pure function of (config, bank_seed): every owner
        # passing the same seed gets bit-identical modules, which is what lets
        # version (0, 0) mean the same bytes on every replica (see _versions).
        self.bank_seed = bank_seed
        full = build_module_bank(config, seed=bank_seed)
        self.bank = {k: full[k].to(self.device) for k in self.owned_keys}
        self._lock = threading.Lock()
        # (epoch, counter) per owned key, *including* private modules: pull
        # replication needs an order on private stores too. Counters start at
        # (0, 0), which identifies the seeded freshly-built bank -- equal
        # versions imply equal bytes, everywhere, always.
        self._versions = {k: (0, 0) for k in self.owned_keys}
        self._outer_opts = {
            k: make_outer_optimizer({k: self.bank[k]}, diloco)
            for k in self.owned_keys if not is_private_key(k)
        }
        # Served keys. ``bootstrap=True`` (a cluster start): everything owned
        # serves immediately -- every owner built the identical (0, 0) bank.
        # ``bootstrap=False`` (joining a live cluster): serve nothing until the
        # replication pull catches each key up.
        self._active = set(self.owned_keys) if bootstrap else set()
        self._saved_versions: dict = {}  # (dir, key) -> last persisted version
        # Restart: load per-key checkpoint files (dynamic mode; keys gained
        # later via apply_epoch warm-start too), else the legacy shard blob.
        # Resumed keys must reconcile with their replicas before serving --
        # only the universal (0, 0) state may keep boot-serving.
        self._resume_dir = resume_dir
        if resume_dir is not None:
            with self._lock:
                loaded = {k for k in self.owned_keys if self._load_module_locked(k, resume_dir)}
            if loaded:
                self._active -= {k for k in loaded if self._versions[k] != (0, 0)}
            elif os.path.exists(os.path.join(resume_dir, self._shard_name())):
                self.load_shard(resume_dir)  # restart this shard from a checkpoint

    @staticmethod
    def _owner_ids(key, record) -> set:
        return {o["peer_id"] for o in owners_for(key, record)}

    def _handle(self, msg: dict, nbytes: int, peer_id: str | None = None):
        kind = msg.get("type")
        if kind == "fetch":
            return self._fetch(msg, peer_id)
        if kind == "push":
            return self._push(msg)
        if kind == "checkpoint":
            if self._epoch is not None:  # dynamic mode: per-key files + versions
                return {"type": "ack", "versions": self.save_modules(msg["dir"])}
            self.save_shard(msg["dir"])
            return {"type": "ack"}
        if kind == "status":
            with self._lock:
                return {"type": "status", "epoch": self._epoch_num,
                        "versions": {k: self._versions[k] for k in self._active
                                     if k in self._versions}}
        return None

    # -- lifecycle helpers (call under self._lock) -----------------------------

    def _state_allowed_locked(self, key, peer_id) -> bool:
        """May this session pull replication state (exact weights + momentum)?

        Static mode: yes (trusted cluster, replication unused anyway). Dynamic:
        only a session identity-authenticated as one of the key's owners in the
        current epoch.
        """
        if self._epoch is None:
            return True
        return peer_id is not None and peer_id in self._owner_ids(key, self._epoch)

    def _primary_locked(self, key) -> bool:
        """Is this server the (active) primary for ``key``? Static mode: always."""
        if self._epoch is None:
            return True
        if key not in self._active:
            return False  # a syncing primary's base state is stale; refuse writes
        owners = owners_for(key, self._epoch)
        return bool(owners) and owners[0]["peer_id"] == self.peer_id

    def _bump_version_locked(self, key) -> None:
        e, c = self._versions[key]
        self._versions[key] = (self._epoch_num, c + 1 if e == self._epoch_num else 1)

    def _fetch(self, msg: dict, peer_id: str | None = None) -> dict:
        have = msg.get("have", {})
        cold = msg.get("cold", False)
        want_state = bool(msg.get("include_state"))
        weights, versions, state, missing = {}, {}, {}, []
        with self._lock:
            for k in msg.get("keys", []):
                if k not in self.bank:
                    missing.append(k)
                    continue
                if want_state:
                    # Replication pull: exact bytes (replicas must be bit-equal,
                    # so no wire compression here) + outer momentum, all-or-
                    # nothing per key -- an unauthorized session gets "missing",
                    # never a silently degraded copy. Syncing keys ARE served
                    # here (the version is honest): after a full-cluster
                    # restart, replicas reconcile by exchanging their resumed
                    # state and adopting the max -- if syncing peers refused
                    # each other, recovery would deadlock.
                    if not self._state_allowed_locked(k, peer_id):
                        missing.append(k)
                        continue
                    versions[k] = self._versions[k]
                    if tuple(have.get(k) or ()) != self._versions[k]:
                        weights[k] = _state_to_cpu(self.bank[k].state_dict())
                        if k in self._outer_opts:
                            state[k] = _opt_to_wire(_optimizer_state_to_cpu(
                                self._outer_opts[k].state_dict()))
                    continue
                if k not in self._active:
                    missing.append(k)  # still syncing -> the worker tries a replica
                    continue
                if is_private_key(k):
                    if cold:  # ship the path's private modules only on a cold start
                        weights[k] = compress_state(
                            _state_to_cpu(self.bank[k].state_dict()), self.compress)
                else:
                    versions[k] = self._versions[k]
                    if tuple(have.get(k) or ()) != self._versions[k]:  # ship only stale
                        weights[k] = compress_state(
                            _state_to_cpu(self.bank[k].state_dict()), self.compress)
        out = {"type": "weights", "weights": weights, "versions": versions}
        if want_state:
            out["state"] = state
        if missing:
            out["missing"] = missing
        return out

    def apply_epoch(self, record, *, bootstrap: bool | None = None) -> None:
        """Adopt a newer owner-set epoch.

        Gained keys are built fresh (the seeded ``(0, 0)`` bank state) and
        enter ``syncing`` -- served and writable only after the replication
        pull catches them up. ``bootstrap`` marks a coordinated cluster *start*
        instead: every owner boots the identical ``(0, 0)`` bank, so owned keys
        serve immediately (there is nobody to sync from); ``None`` (the polled
        path) defers to the record's signed ``bootstrap`` flag. A crashed owner
        restarting *within* the same epoch must resume from its checkpoint (2d)
        or wait for the next bump -- re-applying a bootstrap epoch over a lost
        disk would serve ``(0, 0)`` for state its backups have since trained.

        Lost keys leave ``owned_keys`` but stay servable (lame duck) for one
        more epoch -- replica fallback can still read them while the new
        owners sync -- and are dropped on the epoch after that.
        """
        if self.peer_id is None:
            raise RuntimeError("apply_epoch needs identity=")
        if not verify_epoch_record(record):
            raise ValueError("invalid epoch record")
        if bootstrap is None:
            bootstrap = bool(record.get("bootstrap"))
        with self._lock:
            if self._epoch is not None and not epoch_newer(record, self._epoch):
                return
            owned = {k for k in self._all_keys if self.peer_id in self._owner_ids(k, record)}
            gained = owned - set(self.bank)
            if gained:
                # Seeded build: gained keys start at the same (0, 0) bytes as
                # everyone else's, so the version gate stays truthful. A key
                # with a local checkpoint file warm-starts from it instead and
                # delta-syncs forward (still via the version gate).
                full = build_module_bank(self.config, seed=self.bank_seed)
                for k in gained:
                    self.bank[k] = full[k].to(self.device)
                    self._versions[k] = (0, 0)
                    if not is_private_key(k) and k not in self._outer_opts:
                        self._outer_opts[k] = make_outer_optimizer({k: self.bank[k]}, self.diloco)
                    if self._resume_dir is not None:
                        self._load_module_locked(k, self._resume_dir)
            for k in owned:
                if not is_private_key(k) and k not in self._outer_opts:
                    self._outer_opts[k] = make_outer_optimizer({k: self.bank[k]}, self.diloco)
            # Trim lame ducks from two epochs ago: keys neither owned now nor
            # in the outgoing epoch (self.owned_keys, about to be replaced)
            # have had a full epoch for their new owners to sync, and holding
            # them forever would defeat the sharding.
            for k in set(self.bank) - owned - self.owned_keys:
                del self.bank[k]
                self._versions.pop(k, None)
                self._outer_opts.pop(k, None)
                self._active.discard(k)
            self.owned_keys = owned
            if bootstrap:
                # Boot-serve only the universal seeded state: a key resumed
                # from disk at v > (0, 0) must reconcile with its replicas
                # first (they may hold newer), even under a bootstrap epoch.
                self._active |= {k for k in owned if self._versions[k] == (0, 0)}
            self._prev_epoch = self._epoch
            self._epoch = record
            self._epoch_num = record["epoch"]

    # -- pull replication (design D4: backups poll, idempotent, version-gated) --

    def start(self) -> None:
        super().start()
        self._repl_thread = threading.Thread(target=self._replicate_loop, daemon=True)
        self._repl_thread.start()

    def _replicate_loop(self) -> None:
        while not self._repl_stop.wait(self._repl_interval):
            if self._stop or self._dead:
                return
            try:
                self._poll_epoch()
                self._replicate_once()
            except Exception:  # noqa: BLE001 -- a peer mid-restart must not kill the puller
                pass

    def _poll_epoch(self) -> None:
        """Ask the scheduler for the current epoch record; adopt it if newer.

        This is how ownership changes reach a running owner (design D1): the
        scheduler is the authority, owners poll. The record verifies against
        ``scheduler_pub`` when pinned (and again inside ``apply_epoch``).
        """
        if self._scheduler_addr is None or self.peer_id is None:
            return
        reply = self._peer_rpc(self._scheduler_addr, {"type": "epoch"})
        record = (reply or {}).get("record")
        if record and verify_epoch_record(record, signer_pub=self.scheduler_pub):
            self.apply_epoch(record)

    def _replicate_once(self) -> dict:
        """One delta-sync pass: every owned key pulls from its authoritative
        source -- backups from the primary (then fellow replicas), a syncing
        primary from its backups. Returns ``{key: "active" | "pending"}``.
        """
        results: dict = {}
        with self._lock:
            epoch = self._epoch
            if epoch is None or self.peer_id is None:
                return {}
            candidates: dict = {}  # key -> [addr, ...] to try, in order
            for k in sorted(self.owned_keys):
                owners = owners_for(k, epoch)
                ids = [o["peer_id"] for o in owners]
                if self.peer_id not in ids:
                    continue
                if ids[0] == self.peer_id:
                    if k in self._active:
                        continue  # active primary: authoritative, nothing to pull
                    srcs = owners[1:]  # syncing primary catches up from backups
                else:
                    srcs = [owners[0]] + [o for o in owners[1:] if o["peer_id"] != self.peer_id]
                if self._prev_epoch is not None:
                    # Last epoch's owners are this key's lame ducks: when every
                    # *current* replica is still syncing (a remap moved the key
                    # wholesale), they are the ones actually holding the data.
                    srcs += [o for o in owners_for(k, self._prev_epoch)
                             if o["peer_id"] != self.peer_id]
                addrs, seen = [], set()
                for o in srcs:
                    a = tuple(o["addr"])
                    if a not in seen:
                        seen.add(a)
                        addrs.append(a)
                if addrs:
                    candidates[k] = addrs
                    results[k] = "pending"
                elif k not in self._active:
                    # Sole owner of the key (no other replica anywhere): it is
                    # authoritative by definition -- e.g. a k=1 restart.
                    self._active.add(k)
                    results[k] = "active"
            was_active = set(self._active)
        served: set = set()
        pending = dict(candidates)
        while pending:
            k0 = next(iter(pending))
            addr = pending[k0][0]
            batch = [k for k, cands in pending.items() if cands[0] == addr]
            with self._lock:
                have = {k: self._versions.get(k) for k in batch}
            try:
                reply = self._peer_rpc(addr, {"type": "fetch", "keys": batch,
                                              "have": have, "include_state": True})
            except (OSError, ConnectionError):
                reply = None
            answers = (reply or {}).get("versions") or {}
            with self._lock:
                for k in batch:
                    pending[k] = pending[k][1:]  # this source has been consulted
                    if k not in self.bank:  # trimmed by a concurrent apply_epoch
                        pending.pop(k, None)
                        continue
                    v = answers.get(k)
                    if v is not None:
                        v = tuple(v)
                        if v > self._versions[k]:
                            sd = (reply.get("weights") or {}).get(k)
                            if sd is not None:
                                self.bank[k].load_state_dict(
                                    {n: t.to(self.device) for n, t in sd.items()})
                                osd = (reply.get("state") or {}).get(k)
                                if osd is not None and k in self._outer_opts:
                                    self._outer_opts[k].load_state_dict(_opt_from_wire(osd))
                                self._versions[k] = v
                        if self._versions[k] >= v:
                            served.add(k)
                        # An already-active key refreshes from its first
                        # responsive source; a *syncing* key keeps consulting
                        # every replica and adopts the max -- activating after
                        # one answer could miss a backup that ran ahead of the
                        # others (and of our own disk) before a crash.
                        if k in was_active:
                            pending.pop(k)
                            results[k] = "active"
                            continue
                    if not pending[k]:
                        pending.pop(k)
                        if k in served:  # every replica consulted; we hold the max
                            self._active.add(k)
                            results[k] = "active"
        return results

    def start_tracker_heartbeat(self, tracker_addr, advertise_host, *, roles=("owner",),
                                interval=30.0, capabilities=None, auth_key=None,
                                tls=None) -> None:
        """Register this owner with a tracker and keep the record fresh.

        Registers ``(advertise_host, self.port)`` as a ``public`` peer offering
        ``roles``, then re-registers every ``interval`` seconds (keep it under
        the tracker's TTL -- liveness *is* the heartbeat). When this owner
        dies, its record expires and the scheduler's epoch manager eventually
        re-maps its keys (design D5).
        """
        if self.identity is None:
            raise RuntimeError("start_tracker_heartbeat needs identity=")
        from .tracker import register_peer  # lazy: tracker imports this module

        addr = tuple(tracker_addr)

        def beat():
            while not (self._stop or self._dead):
                try:
                    register_peer(addr, self.identity, reachability="public",
                                  peer_addr=(advertise_host, self.port), roles=roles,
                                  capabilities=capabilities, auth_key=auth_key, tls=tls)
                except (OSError, ConnectionError):
                    pass  # tracker briefly away; the next beat retries
                if self._repl_stop.wait(interval):
                    return

        self._beat_thread = threading.Thread(target=beat, daemon=True)
        self._beat_thread.start()

    def _peer_rpc(self, addr, msg):
        sock = self._peer_conns.get(addr)
        if sock is None:
            sock = _ps_connect(addr, self._peer_auth, self.max_msg_bytes, 5.0,
                               tls=self._peer_tls, server_hostname=addr[0])
            self._peer_conns[addr] = sock
        try:
            return _rpc(sock, msg, self.max_msg_bytes)
        except OSError:
            self._peer_conns.pop(addr, None)
            try:
                sock.close()
            except OSError:
                pass
            raise

    def shutdown(self) -> None:
        self._repl_stop.set()
        for s in self._peer_conns.values():
            try:
                s.close()
            except OSError:
                pass
        self._peer_conns.clear()
        super().shutdown()

    def _push(self, msg: dict) -> dict:
        grant = msg.get("grant")
        if not verify_grant(grant, self.grant_key, scheduler_pub=self.scheduler_pub):
            return {"type": "ack", "applied": False}  # no/forged grant -> refuse
        # Decode (possibly quantized) gradients outside the lock; a malformed
        # encoding refuses the push rather than crashing the server.
        try:
            updates = {k: maybe_dequantize(u["grad"])
                       for k, u in (msg.get("updates") or {}).items()
                       if isinstance(u, dict)}
        except (TypeError, KeyError, ValueError):
            self.metrics.record_invalid_reject()
            return {"type": "ack", "applied": False}
        private = msg.get("private") or {}
        weight = float(grant["weight"])
        allowed = set(grant["keys"])
        skipped, n_applied = [], 0
        with self._lock:
            if grant["token"] in self._seen_grants:
                return {"type": "ack", "applied": False}  # replay -> refuse
            self._seen_grants[grant["token"]] = True  # consumed even if invalid below
            while len(self._seen_grants) > self._SEEN_GRANTS_MAX:
                self._seen_grants.popitem(last=False)
            # Validate before touching the shard: one applied NaN poisons it.
            if not (all_finite(updates) and all_finite(private)):
                self.metrics.record_invalid_reject()
                return {"type": "ack", "applied": False}
            for k, grad in updates.items():
                if is_private_key(k) or k not in allowed:
                    continue
                if k not in self.owned_keys:
                    if self._epoch is not None:  # zombie routing: make the miss visible
                        skipped.append(k)
                    continue
                if not self._primary_locked(k):  # backups copy state, never apply writes
                    skipped.append(k)
                    continue
                if self.max_update_norm is not None:
                    if clip_norm_(grad, self.max_update_norm) > self.max_update_norm:
                        self.metrics.record_norm_clip()
                apply_outer_grads(self.bank[k], [weight * g.to(self.device) for g in grad])
                self._outer_opts[k].step()
                self._outer_opts[k].zero_grad(set_to_none=True)
                self._bump_version_locked(k)
                n_applied += 1
            for k, sd in private.items():
                if k not in allowed:
                    continue
                if k not in self.owned_keys:
                    if self._epoch is not None:
                        skipped.append(k)
                    continue
                if not self._primary_locked(k):
                    skipped.append(k)
                    continue
                _load_into(self, k, sd)  # store latest private (authoritative-local)
                self._bump_version_locked(k)
                n_applied += 1
            epoch_num = self._epoch_num
        if skipped:
            # Stale routing (the worker's epoch is behind): refuse outright when
            # nothing landed so the loss is visible, list partial skips otherwise.
            return {"type": "ack", "applied": n_applied > 0, "skipped": sorted(skipped),
                    "reason": "not_primary", "epoch": epoch_num}
        return {"type": "ack", "applied": True}

    # -- per-key persistence (design D7: remap-proof checkpoints) ---------------

    @staticmethod
    def _module_file(key: str) -> str:
        # One file per module key (not per key *set* like the legacy shard blob),
        # so ownership remaps never invalidate what is already on disk.
        return f"module_{hashlib.sha256(key.encode()).hexdigest()[:16]}.pt"

    def save_modules(self, dirpath: str) -> dict:
        """Persist every held key to its own file; return ``{key: version}``.

        Files whose version is already on disk are skipped (an idle module
        costs nothing per checkpoint). The returned versions feed the
        scheduler's signed recovery manifest.
        """
        os.makedirs(dirpath, exist_ok=True)
        saved = {}
        with self._lock:
            blobs = {}
            for k in self.bank:
                v = self._versions.get(k, (0, 0))
                saved[k] = v
                if self._saved_versions.get((dirpath, k)) == v:
                    continue
                opt = self._outer_opts.get(k)
                blobs[k] = {"key": k, "version": list(v),
                            "weights": _state_to_cpu(self.bank[k].state_dict()),
                            "outer_opt": (_optimizer_state_to_cpu(opt.state_dict())
                                          if opt is not None else None)}
        for k, blob in blobs.items():
            tmp = os.path.join(dirpath, self._module_file(k) + ".tmp")
            torch.save(blob, tmp)
            os.replace(tmp, os.path.join(dirpath, self._module_file(k)))  # atomic
            self._saved_versions[(dirpath, k)] = tuple(blob["version"])
        return saved

    def _load_module_locked(self, key: str, dirpath: str) -> bool:
        """Warm-start one key from its checkpoint file (if present)."""
        path = os.path.join(dirpath, self._module_file(key))
        if not os.path.exists(path):
            return False
        blob = torch.load(path, map_location=self.device, weights_only=True)
        self.bank[key].load_state_dict(
            {n: v.to(self.device) for n, v in blob["weights"].items()})
        self._versions[key] = _version_pair(blob["version"])
        if blob.get("outer_opt") is not None and key in self._outer_opts:
            self._outer_opts[key].load_state_dict(blob["outer_opt"])
        self._saved_versions[(dirpath, key)] = self._versions[key]
        return True

    def _shard_name(self) -> str:
        # Stable across processes (unlike per-process-salted ``hash``), so a shard
        # saved by one process is found by the restarted one with the same keys.
        digest = hashlib.sha256(",".join(sorted(self.owned_keys)).encode()).hexdigest()[:16]
        return f"shard_{digest}.pt"

    def save_shard(self, dirpath: str) -> None:
        """Persist this shard's weights + versions + outer-optimizer momentum
        to ``dir/shard_<stable-hash>.pt``."""
        os.makedirs(dirpath, exist_ok=True)
        with self._lock:
            blob = {"weights": {k: _state_to_cpu(m.state_dict()) for k, m in self.bank.items()},
                    "versions": dict(self._versions),
                    "outer_opts": {k: _optimizer_state_to_cpu(o.state_dict())
                                   for k, o in self._outer_opts.items()}}
        tmp = os.path.join(dirpath, self._shard_name() + ".tmp")
        torch.save(blob, tmp)
        os.replace(tmp, os.path.join(dirpath, self._shard_name()))  # atomic

    def load_shard(self, dirpath: str) -> None:
        blob = torch.load(os.path.join(dirpath, self._shard_name()),
                          map_location=self.device, weights_only=True)
        with self._lock:
            for k, sd in blob["weights"].items():
                if k in self.bank:
                    self.bank[k].load_state_dict({n: v.to(self.device) for n, v in sd.items()})
            self._versions.update({k: _version_pair(v) for k, v in blob["versions"].items()
                                   if k in self._versions})
            for k, sd in blob.get("outer_opts", {}).items():
                if k in self._outer_opts:  # restore Nesterov momentum, not just weights
                    self._outer_opts[k].load_state_dict(sd)


# -- scheduler (no weights) --------------------------------------------------


class Scheduler(_ReactorServer):
    """Light async scheduler: task queue + clock + staleness; holds **no weights**.

    Owns the path→PS routing (so it can tell a worker where each module lives) and
    the corpus (training data + α shard-weights), but the model bank lives on the
    :class:`ParameterServer` shards.
    """

    def __init__(self, config, corpus, ps_addrs, diloco, batch_size, *,
                 host="0.0.0.0", port=0, auth_key=None, seed=0,
                 staleness_bound=None, staleness_weight="inverse",
                 heartbeat_timeout=30.0, ps_tls=None, grant_key=None,
                 identity=None, compress="none", idle_backoff=None, **reactor_kw):
        super().__init__(host=host, port=port, auth_key=auth_key, **reactor_kw)
        self.ps_tls = ps_tls  # client context for the scheduler's checkpoint RPCs to PSs
        self.grant_key = grant_key  # shared with the PSs (not workers) to sign grants
        # With an Ed25519 identity, grants are signed instead (servers verify
        # via ``scheduler_pub=``) and epoch records can be published (Phase 2a).
        self.identity = identity
        self._epoch_record = None
        self._epoch_floor = 0  # next epoch number must be >= this (restart safety)
        self._manifest = None  # signed recovery point (design D7); gates resume
        self._watch_stop = threading.Event()  # stops the watch_tracker thread
        self._watch_thread = None
        self.compress = check_mode(compress)  # stamped on tasks; workers follow it
        self.idle_backoff = idle_backoff      # server-paced idle polling (retry_in)
        self._worker_caps: dict = {}          # worker_id -> advertised capabilities
        self.config = config
        self.corpus = corpus
        self.diloco = diloco
        self.batch_size = batch_size
        self.seed = seed
        self.topology = config.build_topology()
        self.paths = list(self.topology.paths())
        self.staleness_weight = staleness_weight
        self.staleness_bound = (
            staleness_bound if staleness_bound is not None else 2 * len(self.paths)
        )
        self.heartbeat_timeout = heartbeat_timeout
        self.total_rounds = None

        # key -> (host, port) of the owning parameter server.
        self.ps_addrs = [tuple(a) for a in ps_addrs]
        # Routing values are *replica lists* in rank order (primary first); the
        # static map has one entry per key. With no ps_addrs the scheduler is in
        # rendezvous mode: routing derives from the published epoch instead.
        if self.ps_addrs:
            key_shard = assign_shards(self.topology.module_keys(), len(self.ps_addrs))
            self._routing = {k: [list(self.ps_addrs[s])] for k, s in key_shard.items()}
        else:
            self._routing = {}

        self._lock = threading.Lock()
        self._serving = False
        self._T = 0
        self._target = 0
        self._completed: dict = {}
        self._inflight: dict = {}
        self._issued: dict = {}
        self._lease: dict = {}  # path -> current lease token (fences commits)
        self._owner: dict = {}

    def _handle(self, msg: dict, nbytes: int, peer_id: str | None = None):
        kind = msg.get("type")
        if kind == "request":
            return self._next_task(msg)
        if kind == "commit":
            return self._commit(msg)
        if kind == "heartbeat":
            self._heartbeat(msg)
            return None
        if kind == "epoch":
            with self._lock:
                return {"type": "epoch", "record": self._epoch_record}
        return None

    def publish_epoch(self, owner_records, *, k=3, salt="", bootstrap=None) -> dict:
        """Build, sign, and serve the next owner-set epoch.

        ``owner_records`` are verified, owner-eligible tracker peer records
        (e.g. ``fetch_directory(..., roles=["owner"], reachability="public")``).
        The record is served via the ``epoch`` RPC (owners poll it), drives
        per-task routing, and can be cached on the tracker (``put_epoch``).

        ``bootstrap=None`` decides automatically: only the *first* epoch of a
        run that has done no training (no prior epoch, clock at 0) is flagged,
        so owners boot-serve their seeded banks exactly once; every later
        epoch makes gained keys sync (see ``ParameterServer.apply_epoch``).
        """
        if self.identity is None:
            raise RuntimeError("publish_epoch needs the scheduler's identity=")
        with self._lock:
            if bootstrap is None:
                bootstrap = self._epoch_record is None and self._T == 0
            # Numbering must survive a scheduler restart: owners refuse any
            # record that isn't strictly newer than the one they hold, so
            # restarting at 0 would wedge failover forever. The floor comes
            # from the resumed checkpoint and/or the tracker's cached record.
            num = max(self._epoch_floor,
                      0 if self._epoch_record is None else self._epoch_record["epoch"] + 1)
            record = make_epoch_record(self.identity, epoch=num,
                                       owner_records=owner_records, k=k, salt=salt,
                                       bootstrap=bootstrap)
            self._epoch_record = record
            self._epoch_floor = num + 1
        return record

    def watch_tracker(self, tracker_addr, *, k=3, salt="", owner_grace=240.0,
                      min_epoch_interval=60.0, poll_interval=5.0, tracker_auth=None,
                      tracker_tls=None, cache_on_tracker=True) -> "EpochManager":
        """Drive owner-set epochs from tracker liveness (design D5).

        A background thread polls the tracker's directory for ``owner``-role
        ``public`` peers; an :class:`~opendipaco.schedule.ownership.EpochManager`
        applies the hysteresis (an owner must be gone ``owner_grace`` seconds
        to be dropped; bumps are batched and rate-limited to one per
        ``min_epoch_interval``). Each due change is signed via
        :meth:`publish_epoch` and, with ``cache_on_tracker``, cached back onto
        the tracker for bootstrapping owners. Stops with :meth:`shutdown`.
        """
        if self.identity is None:
            raise RuntimeError("watch_tracker needs the scheduler's identity=")
        from .tracker import fetch_directory, get_epoch, put_epoch  # lazy: tracker imports this

        manager = EpochManager(owner_grace=owner_grace, min_epoch_interval=min_epoch_interval)
        addr = tuple(tracker_addr)
        # Restart continuity: re-adopt our own cached record from the tracker
        # (it is self-signed -- nobody else can plant one). This restores both
        # the owner set for routing and the epoch numbering floor, so the next
        # bump supersedes what live owners already hold instead of being
        # refused as stale.
        try:
            cached = get_epoch(addr, signer_pub=self.identity.public_key_hex,
                               auth_key=tracker_auth, tls=tracker_tls)
        except (OSError, ConnectionError):
            cached = None
        if cached is not None:
            with self._lock:
                if self._epoch_record is None or epoch_newer(cached, self._epoch_record):
                    self._epoch_record = cached
                    self._epoch_floor = max(self._epoch_floor, cached["epoch"] + 1)

        def watch():
            while True:
                try:
                    records = fetch_directory(addr, roles=["owner"], reachability="public",
                                              auth_key=tracker_auth, tls=tracker_tls)
                    due = manager.observe(records)
                    if due is not None:
                        record = self.publish_epoch(due, k=k, salt=salt)
                        if cache_on_tracker:
                            put_epoch(addr, record, auth_key=tracker_auth, tls=tracker_tls)
                except (OSError, ConnectionError):
                    pass  # tracker briefly away; next poll retries
                if self._watch_stop.wait(poll_interval) or self._stop or self._dead:
                    return

        self._watch_thread = threading.Thread(target=watch, daemon=True)
        self._watch_thread.start()
        return manager

    def _idle(self) -> dict:
        msg = {"type": "idle"}
        if self.idle_backoff is not None:
            msg["retry_in"] = self.idle_backoff
        return msg

    def _next_task(self, req: dict) -> dict:
        wid = req.get("worker_id")
        warm = {tuple(p) for p in req.get("warm_paths", [])}
        cached = {tuple(p) for p in req.get("cached_shards", [])}
        caps = req.get("capabilities") or {}
        with self._lock:
            if caps:
                self._worker_caps[wid] = caps
            if not self._serving or self._T >= self._target:
                return {"type": "stop"} if self._stop else self._idle()
            if not self._routing and self._epoch_record is None:
                return self._idle()  # rendezvous mode before the first epoch
            self._reclaim_inflight_locked()
            eligible = [p for p in self._completed if p not in self._inflight]
            if not eligible:
                return self._idle()
            path = min(eligible, key=lambda p: (self._completed[p], p not in warm, p))
            lease = uuid.uuid4().hex  # unique per lease; fences commit/heartbeat
            self._owner[path] = wid
            self._inflight[path] = time.monotonic() + self.heartbeat_timeout
            self._issued[path] = self._T
            self._lease[path] = lease
            generation = self._completed[path]
            keys = self.topology.path_module_keys(path)
            if self._epoch_record is not None:  # rendezvous: replicas in rank order
                routing = {k: [list(o["addr"]) for o in owners_for(k, self._epoch_record)]
                           for k in keys}
            else:
                routing = {k: self._routing[k] for k in keys}
        # Data plane: shard bytes, or just the recipe for a spec corpus.
        shard, shard_spec = None, None
        if path not in cached:
            if hasattr(self.corpus, "spec"):
                shard_spec = {"path_index": self.topology.path_index(path),
                              "spec": self.corpus.spec}
            else:
                shard = self.corpus.shard(self.topology.path_index(path))
        return {
            "type": "task",
            "gen_id": generation,
            "lease": lease,
            "path": path,
            "routing": routing,
            "compress": self.compress,  # uplink encoding the worker should use
            "shard": compress_shard(shard, self.compress),
            "shard_spec": shard_spec,
            "batch_size": (max(1, min(self.batch_size, int(caps["max_batch"])))
                           if caps.get("max_batch") else self.batch_size),
            "total_rounds": self.total_rounds,
            "seed": self.seed,
        }

    def _commit(self, msg: dict) -> dict:
        path = msg["path"]
        with self._lock:
            lease = self._lease.get(path)
            if path not in self._inflight or msg.get("lease") != lease:
                # stale / already freed / not the current lease holder
                return {"type": "commit_ack", "accepted": False}
            staleness = self._T - self._issued.get(path, self._T)
            self._inflight.pop(path, None)
            self._lease.pop(path, None)
            if staleness > self.staleness_bound:
                self.metrics.record_stale_reject()
                return {"type": "commit_ack", "accepted": False}
            # A non-finite inner loss means the worker's training diverged (or its
            # hardware is faulty) -- don't grant a push for it. The empty-shard
            # no-op convention (loss=NaN, nothing to push) stays accepted.
            if not loss_ok(msg.get("loss"), empty=bool(msg.get("empty"))):
                self.metrics.record_invalid_reject()
                return {"type": "commit_ack", "accepted": False}
            self._T += 1
            self._completed[path] = self._completed.get(path, 0) + 1
            damp = 1.0 / (1.0 + staleness) if self.staleness_weight == "inverse" else 1.0
            push_weight = self.corpus.shard_weight(self.topology.path_index(path)) * damp
            self.metrics.record_update(staleness)
            # The grant carries the verdict to the parameter servers: weight and
            # allowed keys come from here, the lease token makes it single-use.
            grant = make_grant(path, self.topology.path_module_keys(path),
                               push_weight, lease, self.grant_key,
                               identity=self.identity)
            return {"type": "commit_ack", "accepted": True,
                    "push_weight": push_weight, "grant": grant}

    def _heartbeat(self, msg: dict) -> None:
        path = msg["path"]
        with self._lock:
            if path in self._inflight and msg.get("lease") == self._lease.get(path):
                self._inflight[path] = time.monotonic() + self.heartbeat_timeout

    def _reclaim_inflight_locked(self) -> None:
        now = time.monotonic()
        for path, deadline in list(self._inflight.items()):
            if now >= deadline:
                del self._inflight[path]
                self._lease.pop(path, None)  # invalidate the token: zombies can't commit
                self._owner[path] = None
                self.metrics.reclaims += 1

    def fit(self, num_generations: int, *, total_generations=None, log_every=0,
            reclaim_interval=0.05, checkpoint_dir=None, checkpoint_every=0, resume=False):
        """Run until each path has had ~``num_generations`` updates.

        A **cluster checkpoint** (every ``checkpoint_every`` updates, if
        ``checkpoint_dir`` is set) saves the scheduler's clock and tells every
        parameter server to persist its shard. To restart, relaunch each
        ``ParameterServer(resume_dir=checkpoint_dir)`` and call ``fit(resume=True,
        checkpoint_dir=…)``; workers reconnect on their own.
        """
        if resume and checkpoint_dir:
            self._load_state(checkpoint_dir)
            # Recovery gate (design D7): with a manifest, don't serve until a
            # live owner holds >= the manifest version for every key (owners
            # reconcile among themselves via replication while we wait).
            while self._manifest is not None and not self._stop and not self._dead:
                if self._recovery_ready():
                    break
                time.sleep(0.5)
        self.total_rounds = total_generations if total_generations is not None else num_generations
        with self._lock:
            self._completed = {p: self._completed.get(p, 0) for p in self.paths}
            self._inflight, self._issued = {}, {}
            self._target = self._T + num_generations * len(self.paths)
            self._serving = True
        t0 = time.monotonic()
        last_ckpt = self._T
        while True:
            with self._lock:
                self._reclaim_inflight_locked()
                done = self._T >= self._target
            if done or self._stop or self._dead:
                break
            if checkpoint_dir and checkpoint_every and (self._T - last_ckpt) >= checkpoint_every:
                self._checkpoint_cluster(checkpoint_dir)
                last_ckpt = self._T
            time.sleep(reclaim_interval)
        with self._lock:
            self._serving = False
        if checkpoint_dir and checkpoint_every:
            self._checkpoint_cluster(checkpoint_dir)
        self.metrics._wall += time.monotonic() - t0
        return dict(self._completed)

    def _checkpoint_cluster(self, dirpath: str) -> None:
        """Save the scheduler clock and trigger every parameter server to persist.

        In rendezvous mode the owners' checkpoint acks carry the versions they
        persisted; their per-key maximum becomes the signed **recovery
        manifest** (design D7) -- the floor a restarted cluster must reach
        (some live owner holding >= the manifest version for every key) before
        the scheduler serves tasks again.
        """
        os.makedirs(dirpath, exist_ok=True)
        with self._lock:
            record = self._epoch_record
            addrs = (sorted({tuple(o["addr"]) for o in record["owners"]})
                     if record is not None else self.ps_addrs)
        held: dict = {}
        for addr in addrs:
            try:
                s = _ps_connect(addr, self.auth_key, DEFAULT_MAX_MSG_BYTES, 5.0,
                                tls=self.ps_tls, server_hostname=addr[0])
                reply = _rpc(s, {"type": "checkpoint", "dir": dirpath}, DEFAULT_MAX_MSG_BYTES)
                s.close()
            except OSError:
                continue  # a PS that's momentarily unreachable is checkpointed next time
            for k, v in ((reply or {}).get("versions") or {}).items():
                held[k] = max(held.get(k, (0, 0)), tuple(v))
        if record is not None and self.identity is not None and held:
            manifest = sign_record(self.identity, {
                "kind": "manifest", "epoch": record["epoch"],
                "keys": {k: list(v) for k, v in held.items()},
                "issued_at": time.time(),
            })
            tmp = os.path.join(dirpath, "manifest.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            os.replace(tmp, os.path.join(dirpath, "manifest.json"))
            with self._lock:
                self._manifest = manifest
        with self._lock:
            state = {"T": self._T, "completed": dict(self._completed),
                     "epoch": -1 if record is None else record["epoch"]}
        tmp = os.path.join(dirpath, "scheduler.pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(dirpath, "scheduler.pt"))

    def _load_state(self, dirpath: str) -> None:
        path = os.path.join(dirpath, "scheduler.pt")
        if not os.path.exists(path):
            return
        state = torch.load(path, weights_only=True)
        with self._lock:
            self._T = state["T"]
            self._completed = dict(state["completed"])
            self._epoch_floor = max(self._epoch_floor, int(state.get("epoch", -1)) + 1)
        mpath = os.path.join(dirpath, "manifest.json")
        if self.identity is not None and os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                manifest = json.load(f)
            # Only this scheduler's own signature counts: a planted manifest
            # could otherwise lower (or inflate) the recovery floor.
            if (verify_record(manifest) and manifest.get("kind") == "manifest"
                    and manifest.get("pub", "").lower() == self.identity.public_key_hex):
                with self._lock:
                    self._manifest = manifest

    def _recovery_ready(self) -> bool:
        """Does some live owner hold >= the manifest version for every key?

        Polls the current epoch owners' ``status`` (their *active* keys). Until
        this holds, serving tasks would train against data older than the
        recovery point.
        """
        with self._lock:
            record, manifest = self._epoch_record, self._manifest
        if manifest is None:
            return True
        if record is None:
            return False  # no owner set yet (watch_tracker hasn't published)
        held: dict = {}
        for addr in sorted({tuple(o["addr"]) for o in record["owners"]}):
            try:
                s = _ps_connect(addr, self.auth_key, DEFAULT_MAX_MSG_BYTES, 5.0,
                                tls=self.ps_tls, server_hostname=addr[0])
                reply = _rpc(s, {"type": "status"}, DEFAULT_MAX_MSG_BYTES)
                s.close()
            except (OSError, ConnectionError):
                continue
            for k, v in ((reply or {}).get("versions") or {}).items():
                held[k] = max(held.get(k, (0, 0)), tuple(v))
        return all(held.get(k, (-1, -1)) >= tuple(v) for k, v in manifest["keys"].items())

    def shutdown(self) -> None:
        self._watch_stop.set()
        with self._lock:
            self._serving = False
        super().shutdown()


# -- sharded worker ----------------------------------------------------------


def run_sharded_worker(config, diloco, scheduler_addr, *, device="cpu", seed=0,
                       auth_key=None, max_tasks=None, heartbeat_interval=3.0,
                       poll_interval=0.02, max_msg_bytes=DEFAULT_MAX_MSG_BYTES,
                       connect_timeout=10.0, reconnect=False, reconnect_timeout=30.0,
                       fault_hook=None, tls=None, tls_hostname=None,
                       data_dir=None, data_source=None, data_tokenizer=None,
                       max_batch_size=None):
    """Train path-tasks for a sharded scheduler + parameter servers.

    Per task: lease from the scheduler, fetch the path's modules from the owning
    parameter servers, train, commit (accept/reject + damped weight), and push the
    pseudo-gradients to the owning servers. Warm caches (private modules, Adam
    state, shard) persist across tasks. With ``reconnect`` a dropped scheduler/PS
    connection is retried (e.g. a coordinator restart); warm caches survive.

    With a spec corpus on the scheduler, tasks carry a shard recipe instead of
    bytes and the worker materializes its shard locally (``data/spec.py``);
    ``data_dir`` / ``data_source`` / ``data_tokenizer`` as in ``run_worker``.
    """
    engine = _build_worker_engine(config, diloco, device, seed)
    worker = AsyncScheduler(engine, num_workers=1)
    wid = uuid.uuid4().hex
    warm: set = set()
    shard_cache: dict = {}
    versions: dict = {}          # shared key -> held version
    ps_conns: dict = {}          # (host, port) -> connected socket
    residuals: dict = {}         # path -> {key: [tensors]}: compression error feedback
    data_ctx = {"dir": data_dir, "source": data_source, "tokenizer": data_tokenizer}
    caps = {"device": str(device)}
    if max_batch_size is not None:
        caps["max_batch"] = int(max_batch_size)
    state = {"done": 0}

    first = True
    backoff = 0.05
    while True:
        try:
            sch = _ps_connect(tuple(scheduler_addr), auth_key, max_msg_bytes,
                              connect_timeout if first else reconnect_timeout,
                              tls=tls, server_hostname=tls_hostname or scheduler_addr[0])
        except ConnectionError:
            return  # scheduler unreachable
        first = False
        clean = False
        try:
            clean = _serve_sharded(sch, engine, worker, wid, warm, shard_cache, versions,
                                   ps_conns, residuals, data_ctx, caps, state, auth_key,
                                   max_msg_bytes, connect_timeout, heartbeat_interval,
                                   poll_interval, max_tasks, fault_hook, tls=tls)
        except (OSError, ConnectionError):
            clean = False  # disconnected -> reconnect (if enabled)
        finally:
            try:
                sch.close()
            except OSError:
                pass
        if clean or not reconnect:
            for s in ps_conns.values():
                try:
                    s.close()
                except OSError:
                    pass
            return
        # Reconnect: drop stale PS sockets; they reconnect lazily next task.
        for s in ps_conns.values():
            try:
                s.close()
            except OSError:
                pass
        ps_conns.clear()
        time.sleep(backoff)
        backoff = min(backoff * 2, 1.0)


def _serve_sharded(sch, engine, worker, wid, warm, shard_cache, versions, ps_conns,
                   residuals, data_ctx, caps, state, auth_key, max_msg_bytes,
                   connect_timeout, heartbeat_interval, poll_interval, max_tasks,
                   fault_hook, *, tls=None) -> bool:
    """One scheduler connection: serve tasks. Returns True on a clean finish (stop /
    budget), raises ``OSError`` on a disconnect (so the caller can reconnect)."""
    send_lock = threading.Lock()

    def sch_send(m):
        with send_lock:
            send_msg(sch, m)

    def ps_sock(addr):
        if addr not in ps_conns:
            ps_conns[addr] = _ps_connect(addr, auth_key, max_msg_bytes, connect_timeout,
                                         tls=tls, server_hostname=addr[0])
        return ps_conns[addr]

    while True:
        sch_send({"type": "request", "worker_id": wid,
                  "warm_paths": list(warm), "cached_shards": list(shard_cache),
                  "capabilities": caps})
        task = recv_msg(sch, max_msg_bytes)
        if task is None:
            raise OSError("scheduler disconnected")  # not a clean stop -> reconnect
        if task["type"] == "stop":
            return True
        if task["type"] == "idle":
            time.sleep(task.get("retry_in") or poll_interval)  # server-paced when set
            continue

        path = task["path"]
        lease = task.get("lease")
        worker.seed = task["seed"]
        engine.total_rounds = task["total_rounds"]
        # Routing values are replica addr lists in rank order (primary first).
        routing = {k: [tuple(a) for a in addrs] for k, addrs in task["routing"].items()}
        cold = path not in warm

        def fetch_keys():
            """Fetch each key from its first responsive replica: prefer an
            already-connected owner, else rank order; a replica that is down or
            still syncing ("missing") falls back to the next one (design D8)."""
            pending = {
                k: [a for a in addrs if a in ps_conns] + [a for a in addrs if a not in ps_conns]
                for k, addrs in routing.items()
            }
            while pending:
                addr = next(iter(pending.values()))[0]
                batch = [k for k, cands in pending.items() if cands[0] == addr]
                try:
                    reply = _rpc(ps_sock(addr), {
                        "type": "fetch", "keys": batch, "cold": cold,
                        "have": {k: versions.get(k) for k in batch if not is_private_key(k)},
                    }, max_msg_bytes)
                    if reply is None:
                        raise OSError(f"replica {addr} closed")
                except (OSError, ConnectionError):
                    ps_conns.pop(addr, None)
                    for k in batch:
                        pending[k] = pending[k][1:]
                        if not pending[k]:
                            raise OSError(f"no replica could serve {k}")
                    continue
                missing = set(reply.get("missing") or [])
                for k, sd in reply["weights"].items():
                    _load_into(engine, k, sd)
                versions.update(reply.get("versions", {}))
                for k in batch:
                    if k in missing:
                        pending[k] = pending[k][1:]
                        if not pending[k]:
                            raise OSError(f"no replica could serve {k}")
                    else:
                        pending.pop(k)

        stop_beat = threading.Event()
        beat = threading.Thread(target=_sch_heartbeat,
                                args=(sch_send, stop_beat, heartbeat_interval, wid, lease, path),
                                daemon=True)
        beat.start()
        try:
            if fault_hook is not None:
                fault_hook(path, 1)
            fetch_keys()
            if cold:
                engine._opt_state.pop(path, None)  # reset Adam on a cold start
                residuals.pop(path, None)          # and any stale error-feedback carry
            if task.get("shard") is not None:
                shard_cache[path] = restore_shard(task["shard"])
            elif task.get("shard_spec") is not None and path not in shard_cache:
                shard_cache[path] = _materialize_from_spec(task["shard_spec"], data_ctx)
            shard = shard_cache[path]
            contrib = worker._train_path(path, shard, task["batch_size"], task["gen_id"])
        finally:
            stop_beat.set()
            beat.join(timeout=1)

        ack = _rpc_send(sch, send_lock, max_msg_bytes,
                        {"type": "commit", "path": path, "worker_id": wid, "lease": lease,
                         "loss": contrib.loss, "empty": contrib.empty})
        if ack is None:
            raise OSError("scheduler disconnected during commit")
        if ack.get("accepted"):
            grant = ack["grant"]  # carries the push weight + allowed keys to the PSs
            # Encode only after acceptance, so the error-feedback residual always
            # reflects an update that is actually pushed.
            shared_payload, private_payload, pending_res = _compress_contribution(
                contrib, task.get("compress") or "none", residuals, path
            )
            _commit_residuals(residuals, path, pending_res)
            by_primary: dict = {}  # writes go to rank 0 only (design D3)
            for k, addrs in routing.items():
                by_primary.setdefault(addrs[0], []).append(k)
            for addr, keys in by_primary.items():
                updates = {k: {"grad": shared_payload[k]}
                           for k in keys if not is_private_key(k) and k in shared_payload}
                private = {k: private_payload[k]
                           for k in keys if is_private_key(k) and k in private_payload}
                if not updates and not private:
                    continue
                _rpc(ps_sock(addr),
                     {"type": "push", "grant": grant, "updates": updates, "private": private},
                     max_msg_bytes)
            engine._opt_state[path] = contrib.opt_state          # warm-back
            _load_private(engine, contrib.private_state)
            warm.add(path)
            state["done"] += 1
            if max_tasks is not None and state["done"] >= max_tasks:
                return True
        # rejected -> discard the contribution; warm caches stay


def _ps_connect(addr, auth_key, max_msg_bytes, timeout, *, tls=None, server_hostname=None):
    import socket as _socket
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            s = _socket.create_connection(addr, timeout=timeout)
            s.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            if tls is not None:
                s = tls.wrap_socket(s, server_hostname=server_hostname or addr[0])
            if not client_handshake(s, auth_key):
                s.close()
                raise PermissionError(f"auth rejected by {addr}")
            return s
        except ssl.SSLError:  # handshake/cert failure is fatal -- don't retry-to-timeout
            s.close()
            raise
        except PermissionError:  # auth rejection is fatal too (it's an OSError subclass)
            raise
        except OSError as e:
            last = e
            time.sleep(0.05)
    raise ConnectionError(f"could not connect to {addr}: {last}")


def _rpc(sock, msg, max_msg_bytes):
    send_msg(sock, msg)
    return recv_msg(sock, max_msg_bytes)


def _rpc_send(sock, lock, max_msg_bytes, msg):
    with lock:
        send_msg(sock, msg)
        return recv_msg(sock, max_msg_bytes)


def _sch_heartbeat(sch_send, stop_beat, interval, wid, lease, path):
    while not stop_beat.wait(interval):
        try:
            sch_send({"type": "heartbeat", "lease": lease, "path": path, "worker_id": wid})
        except OSError:
            return
