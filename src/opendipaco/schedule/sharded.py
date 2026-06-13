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
import random
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
    pseudograd_digest,
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
from .aggregate import check_aggregate, robust_delta
from .guard import all_finite, clip_norm_, loss_ok
from .ratelimit import RateLimiter
from .reputation import Reputation
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
    _PRIVATE_PROPOSAL_MAX = 16  # distinct held proposals per key before FIFO eviction

    def __init__(self, config, owned_keys, diloco, *, host="0.0.0.0", port=0,
                 auth_key=None, device="cpu", resume_dir=None, grant_key=None,
                 scheduler_pub=None, max_update_norm=None, compress="none",
                 identity=None, epoch_record=None, replicate_interval=10.0,
                 peer_auth=None, peer_tls=None, bootstrap=True, bank_seed=0,
                 scheduler_addr=None, robustness="off", quorum_target=3,
                 quorum_timeout=30.0, aggregate="trimmed_mean", version_history=1,
                 private_policy="overwrite", private_quorum=2, **reactor_kw):
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

        # Robust aggregation (Phase 3a, finding 1.1). ``off`` (default) is
        # today's behavior, bit-identical: each shared push applies immediately.
        # ``on`` buffers a shared module's contributions and applies one robust
        # aggregate (see ``aggregate.py``) once a quorum of
        # ``min(sharing_degree, quorum_target)`` arrives or ``quorum_timeout``
        # elapses. Private modules are untouched here (policy 3a is slice 3d).
        if robustness not in ("off", "on"):
            raise ValueError(f"robustness must be 'off' or 'on', got {robustness!r}")
        self.robust = robustness
        self.quorum_target = max(1, int(quorum_target))
        self.quorum_timeout = float(quorum_timeout)
        self.aggregate = check_aggregate(aggregate)
        self._topology = config.build_topology()
        self._buffers: dict = {}     # key -> [(weight, [grad tensors]), ...]
        self._buffer_ts: dict = {}   # key -> monotonic ts of the buffer's first entry
        # Version history (Phase 3c): retain the last ``version_history`` states
        # per key so a redundant-execution checker can fetch the *exact base* a
        # primary trained against even after the module has advanced. 1 (default)
        # = off, no snapshots, zero cost on the normal path.
        self.version_history = max(1, int(version_history))
        self._history: dict = {}     # key -> OrderedDict[version -> cpu state_dict]
        # Private-module policy (Phase 3d, decision D5/3a). ``overwrite``
        # (default) applies a private push verbatim (today). ``proposal`` holds
        # private pushes as inert proposals and applies one only when
        # ``private_quorum`` distinct authenticated peers agree on the *exact*
        # state -- so a lone owner-path worker can at most stall its private
        # module, never poison it. Corroboration comes from redundant execution.
        if private_policy not in ("overwrite", "proposal"):
            raise ValueError(f"private_policy must be 'overwrite' or 'proposal', "
                             f"got {private_policy!r}")
        self.private_policy = private_policy
        self.private_quorum = max(2, int(private_quorum))
        # key -> {digest: {"state": sd, "peers": set[peer_id]}}
        self._private_proposals: dict = {}

        self.identity = identity
        self.peer_id = getattr(identity, "peer_id", None)
        self._all_keys = set(self._topology.module_keys())
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
            return self._push(msg, peer_id)
        if kind == "private_proposal":
            return self._private_proposal(msg, peer_id)
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

    def _pinned_state_locked(self, key, version):
        """The CPU state of ``key`` at exactly ``version`` (live or retained),
        or None if that version is no longer available."""
        if key not in self.bank:
            return None
        if version == self._versions[key]:
            return _state_to_cpu(self.bank[key].state_dict())
        hist = self._history.get(key)
        if hist is not None and version in hist:
            return _state_to_cpu(hist[version])
        return None

    # -- private proposals (Phase 3d, policy D5/3a) ----------------------------

    def _private_proposal(self, msg: dict, peer_id: str | None = None) -> dict:
        """A private-state *proposal* from a redundant-execution checker. It must
        carry a **scheduler-issued check grant** (single-use, scoped to the
        path's private keys), so only peers the scheduler actually assigned to
        recompute this work can vote -- two colluding/Sybil peers can't fabricate
        agreement by pushing matching garbage they never earned the right to
        propose. Inert until ``private_quorum`` distinct grants agree on the
        owner-computed digest."""
        if self.private_policy != "proposal":
            return {"type": "ack", "applied": False, "reason": "not_proposal_mode"}
        grant = msg.get("grant")
        if not verify_grant(grant, self.grant_key, scheduler_pub=self.scheduler_pub):
            return {"type": "ack", "applied": False}  # unassigned -> no vote
        try:
            states = {k: v for k, v in (msg.get("private") or {}).items()
                      if is_private_key(k)}
        except (TypeError, AttributeError):
            return {"type": "ack", "applied": False}
        if not all_finite(states):
            self.metrics.record_invalid_reject()
            return {"type": "ack", "applied": False}
        token, allowed = grant["token"], set(grant["keys"])
        applied = []
        with self._lock:
            if token in self._seen_grants:
                return {"type": "ack", "applied": False}  # replay -> no double vote
            self._seen_grants[token] = True
            while len(self._seen_grants) > self._SEEN_GRANTS_MAX:
                self._seen_grants.popitem(last=False)
            for k, sd in states.items():
                if k in self.owned_keys and k in allowed and self._primary_locked(k):
                    if self._record_private_proposal_locked(k, sd, token):
                        applied.append(k)
        return {"type": "ack", "applied": bool(applied)}

    def _record_private_proposal_locked(self, key, state, token) -> bool:
        """Tally a private proposal by its *owner-computed* digest (a worker
        can't lie about which state it proposed) and the distinct scheduler
        **grant token** that authorized it (so distinct votes mean distinct
        scheduler-assigned recomputations, not just distinct identities). Apply
        when ``private_quorum`` distinct grants agree. Returns whether it
        applied."""
        digest = pseudograd_digest({key: list(state.values())})
        bucket = self._private_proposals.setdefault(key, collections.OrderedDict())
        entry = bucket.get(digest)
        if entry is None:
            # Bound the per-key bucket: proposals that never reach quorum
            # (persistent disagreement, or checkers aborting on aged-out bases)
            # would otherwise accumulate full state-dicts across generations.
            # Evicting the oldest can only delay an apply, never apply wrongly.
            entry = bucket[digest] = {"state": state, "tokens": set()}
            while len(bucket) > self._PRIVATE_PROPOSAL_MAX:
                bucket.popitem(last=False)
        entry["tokens"].add(token)
        if len(entry["tokens"]) >= self.private_quorum:
            self._record_history_locked(key)
            _load_into(self, key, state)  # corroborated -> authoritative
            self._bump_version_locked(key)
            self._private_proposals.pop(key, None)  # clear all proposals for the key
            return True
        return False

    def _record_history_locked(self, key) -> None:
        """Snapshot the *current* (pre-mutation) state under its version, so a
        redundant-execution checker can later fetch this exact base (Phase 3c).
        No-op unless ``version_history > 1``."""
        if self.version_history <= 1:
            return
        hist = self._history.setdefault(key, collections.OrderedDict())
        hist[self._versions[key]] = _state_to_cpu(self.bank[key].state_dict())
        while len(hist) > self.version_history:
            hist.popitem(last=False)

    # -- robust aggregation (Phase 3a) -----------------------------------------

    def _apply_outer_locked(self, key, weight, grad) -> None:
        """One weighted outer step on a module (the unbuffered/``off`` path)."""
        self._record_history_locked(key)  # retain the pre-step base for auditors
        apply_outer_grads(self.bank[key], [weight * g.to(self.device) for g in grad])
        self._outer_opts[key].step()
        self._outer_opts[key].zero_grad(set_to_none=True)
        self._bump_version_locked(key)

    def _quorum_c(self, key) -> int:
        """Contributions to buffer before aggregating: the module's sharing
        degree, capped at ``quorum_target`` (a module shared by fewer paths than
        the target can never reach it, so it aggregates at its full degree)."""
        return max(1, min(self._topology.sharing_count(key), self.quorum_target))

    def _flush_buffer_locked(self, key) -> None:
        """Robustly aggregate the buffered contributions and apply one step."""
        contribs = self._buffers.pop(key, None)
        self._buffer_ts.pop(key, None)
        if not contribs:
            return
        delta, weight_sum = robust_delta(contribs, aggregate=self.aggregate)
        self._record_history_locked(key)
        apply_outer_grads(self.bank[key], [weight_sum * d.to(self.device) for d in delta])
        self._outer_opts[key].step()
        self._outer_opts[key].zero_grad(set_to_none=True)
        self._bump_version_locked(key)

    def _sweep_buffers(self) -> None:
        """Flush buffers whose oldest contribution has waited past the timeout
        (liveness valve: a stalled path must not freeze a module forever).
        Swept at the replication loop's cadence."""
        if self.robust == "off":
            return
        now = time.monotonic()
        with self._lock:
            due = [k for k, ts in self._buffer_ts.items()
                   if self._buffers.get(k) and now - ts >= self.quorum_timeout]
            for k in due:
                self._flush_buffer_locked(k)

    def _flush_all_buffers_locked(self) -> None:
        """Apply every pending robust buffer now. Called before persisting or
        shutting down: buffered contributions were already *accepted* (the
        scheduler advanced its clock), so dropping them on a checkpoint/resume
        or an early stop would lose committed training work."""
        for k in list(self._buffers):
            self._flush_buffer_locked(k)

    def _fetch(self, msg: dict, peer_id: str | None = None) -> dict:
        have = msg.get("have", {})
        cold = msg.get("cold", False)
        want_state = bool(msg.get("include_state"))
        pin = msg.get("pin") or {}  # {key: version}: redundant-execution checker
        weights, versions, state, missing = {}, {}, {}, []
        with self._lock:
            for k in msg.get("keys", []):
                if k not in self.bank:
                    missing.append(k)
                    continue
                if k in pin:
                    # Pinned fetch: serve the exact version the audited primary
                    # trained against (live, or a retained snapshot), so a
                    # checker reproduces the computation. Aged-out -> missing,
                    # and the auditor aborts (no false disagreement).
                    snap = self._pinned_state_locked(k, tuple(pin[k]))
                    if snap is None:
                        missing.append(k)
                    else:
                        weights[k] = compress_state(snap, self.compress)
                        versions[k] = tuple(pin[k])
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
                        versions[k] = self._versions[k]  # so an auditor can pin it
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
                self._sweep_buffers()  # flush quorum buffers past their timeout
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
        with self._lock:
            self._flush_all_buffers_locked()  # don't drop accepted-but-buffered work
        for s in self._peer_conns.values():
            try:
                s.close()
            except OSError:
                pass
        self._peer_conns.clear()
        super().shutdown()

    def _push(self, msg: dict, peer_id: str | None = None) -> dict:
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
                if self.robust == "off":
                    self._apply_outer_locked(k, weight, grad)
                else:
                    # Buffer; apply one robust aggregate at quorum (here) or on
                    # timeout (the replication loop sweeps). Accepted == buffered.
                    buf = self._buffers.setdefault(k, [])
                    buf.append((weight, [g.to(self.device) for g in grad]))
                    self._buffer_ts.setdefault(k, time.monotonic())
                    if len(buf) >= self._quorum_c(k):
                        self._flush_buffer_locked(k)
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
                if self.private_policy == "proposal":
                    # Hold as a proposal; apply only on quorum agreement (D5/3a).
                    # The primary's own (verified, single-use) commit grant is
                    # its vote, counted alongside the checkers' check grants.
                    self._record_private_proposal_locked(k, sd, grant["token"])
                    n_applied += 1
                    continue
                self._record_history_locked(k)  # retain pre-store base for auditors
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
            self._flush_all_buffers_locked()  # don't checkpoint accepted-but-buffered work away
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
            self._flush_all_buffers_locked()  # persist accepted-but-buffered work
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
                 identity=None, compress="none", idle_backoff=None,
                 reputation=None, rate_limiter=None, min_owner_reputation=0.25,
                 redundancy=3, redundancy_rate=0.0, audit_timeout=60.0,
                 private_policy="overwrite", **reactor_kw):
        super().__init__(host=host, port=port, auth_key=auth_key, **reactor_kw)
        self.ps_tls = ps_tls  # client context for the scheduler's checkpoint RPCs to PSs
        self.grant_key = grant_key  # shared with the PSs (not workers) to sign grants
        # Reputation (Phase 3b): scores authenticated peers from commit outcomes,
        # gates owner eligibility (>= min_owner_reputation; the floor sits above
        # it so fresh peers bootstrap), and scales the rate limiter. Both default
        # to constructed instances; pass configured ones, or None to disable.
        self.reputation = reputation if reputation is not None else Reputation()
        self.rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        self.min_owner_reputation = min_owner_reputation
        # Redundant execution (Phase 3c): a sampled fraction of tasks are
        # *audited* -- the primary reports a pinned base + digest, and surplus
        # workers re-run it from that exact base as checkers. Disagreement burns
        # reputation, agreement rewards it; this also absorbs the §1.9 oversupply
        # (surplus workers do checks instead of idling). rate 0.0 (default) =
        # off, no audits, byte-identical to Phase 2.
        self.redundancy = max(1, int(redundancy))
        self.redundancy_rate = float(redundancy_rate)
        self.audit_timeout = float(audit_timeout)
        self._audits: dict = {}  # (path, gen) -> audit record
        # Under the private proposal policy (D5/3a), private-bearing tasks are
        # *always* audited so checkers corroborate the private state before the
        # owner applies it (without an audit a proposal never reaches quorum).
        self.private_policy = private_policy
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
            return self._next_task(msg, peer_id)
        if kind == "commit":
            return self._commit(msg, peer_id)
        if kind == "routing":
            return self._fresh_routing(msg)
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

        manager = EpochManager(
            owner_grace=owner_grace, min_epoch_interval=min_epoch_interval,
            is_eligible=lambda pid: self.reputation.eligible(pid, self.min_owner_reputation))
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

    def _routing_locked(self, keys) -> dict:
        """Replica addr lists per key, rank order (primary first), per the
        current epoch (rendezvous) or the static shard map."""
        if self._epoch_record is not None:
            return {k: [list(o["addr"]) for o in owners_for(k, self._epoch_record)]
                    for k in keys}
        return {k: self._routing[k] for k in keys}

    def _fresh_routing(self, msg: dict) -> dict:
        """Re-resolve a path's routing on demand: a worker whose push was
        refused as ``not_primary`` (epoch changed mid-task) retries its grant
        once against the *current* primaries (grants are single-use per
        server, so re-presenting one to a new primary is sound)."""
        try:
            keys = self.topology.path_module_keys(tuple(msg.get("path") or ()))
            with self._lock:
                if not self._routing and self._epoch_record is None:
                    return {"type": "routing", "routing": None}
                epoch = None if self._epoch_record is None else self._epoch_record["epoch"]
                return {"type": "routing", "routing": self._routing_locked(keys),
                        "epoch": epoch}
        except (KeyError, IndexError, TypeError, ValueError):
            return {"type": "routing", "routing": None}  # malformed path

    def _next_task(self, req: dict, peer_id: str | None = None) -> dict:
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
            member = peer_id or wid
            eligible = [p for p in self._completed if p not in self._inflight]
            check_only, base, audit, check_private, check_grant = (
                False, None, False, False, None)
            rep = self.reputation.get(peer_id)
            if not eligible:
                # No primary work: absorb the surplus as a redundant check for an
                # open audit (§1.9), if one needs this distinct worker.
                cand = self._find_check_locked(member)
                if cand is None:
                    return self._idle()
                if not self.rate_limiter.allow(peer_id, reputation=rep):
                    return self._idle()
                (path, generation, base, lease, check_private,
                 check_grant) = self._reserve_check_locked(cand, member)
                check_only = True
            else:
                # Rate limit only the *expensive* path (issuing a task with a
                # weight/shard payload): a throttled peer gets a cheap backoff
                # idle, not a disconnect (§1.14). Reputation scales its bucket.
                if not self.rate_limiter.allow(peer_id, reputation=rep):
                    return self._idle()
                path = min(eligible, key=lambda p: (self._completed[p], p not in warm, p))
                lease = uuid.uuid4().hex  # unique per lease; fences commit/heartbeat
                self._owner[path] = wid
                self._inflight[path] = time.monotonic() + self.heartbeat_timeout
                self._issued[path] = self._T
                self._lease[path] = lease
                generation = self._completed[path]
                # Sample this task for an audit: the primary will report a pinned
                # base + digest, and checkers re-run it from that base. A
                # private-bearing path under the proposal policy is *always*
                # audited (checkers must corroborate the private state).
                has_private = any(is_private_key(k)
                                  for k in self.topology.path_module_keys(path))
                want_audit = (self.private_policy == "proposal" and has_private) or (
                    self.redundancy_rate > 0 and random.random() < self.redundancy_rate)
                if (self.redundancy > 1 and want_audit
                        and (path, generation) not in self._audits):
                    audit = True
                    self._audits[(path, generation)] = {
                        "target": self.redundancy - 1, "base": None,
                        "primary_digest": None, "primary_peer": None,
                        "checks": [], "members": {member},
                        # A creation deadline so an audit whose primary never
                        # commits (worker died) still expires and is reaped --
                        # the primary's commit extends it for the checkers.
                        "deadline": time.monotonic() + self.audit_timeout,
                        "private": has_private and self.private_policy == "proposal"}
            keys = self.topology.path_module_keys(path)
            routing = self._routing_locked(keys)
        # Data plane: shard bytes, or just the recipe for a spec corpus. A check
        # is cold, so it always gets the shard (ignores the worker's cache).
        shard, shard_spec = None, None
        if check_only or path not in cached:
            if hasattr(self.corpus, "spec"):
                shard_spec = {"path_index": self.topology.path_index(path),
                              "spec": self.corpus.spec}
            else:
                shard = self.corpus.shard(self.topology.path_index(path))
        task = {
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
        if check_only:
            task["check_only"], task["base"] = True, base
            if check_private:  # also submit private proposals to the owners
                task["private_proposal"] = True
                task["grant"] = check_grant  # authorizes the proposal at the owner
        elif audit:
            task["audit"] = True
        return task

    # -- redundant execution (Phase 3c) ----------------------------------------

    def _find_check_locked(self, member):
        """An open audit that still needs a checker and that ``member`` hasn't
        already taken a slot in (a peer must not check its own work)."""
        for key, a in self._audits.items():
            if a["base"] is None or a.get("resolved") or member in a["members"]:
                continue
            if len(a["members"]) - 1 < a["target"]:  # members includes the primary
                return key
        return None

    def _reserve_check_locked(self, key, member):
        a = self._audits[key]
        a["members"].add(member)
        path, generation = key
        lease = uuid.uuid4().hex
        # Record the issued lease so only this assigned checker's result counts
        # (a peer can't fabricate audit votes by guessing an open audit).
        a.setdefault("check_leases", {})[member] = lease
        grant = None
        if a.get("private"):
            # A scheduler-signed grant scoped to the path's private keys lets the
            # owner accept this checker's private proposal as a real vote.
            private_keys = [k for k in self.topology.path_module_keys(path)
                            if is_private_key(k)]
            grant = make_grant(path, private_keys, 0.0, lease, self.grant_key,
                               identity=self.identity)
        return path, generation, a["base"], lease, bool(a.get("private")), grant

    def _commit_check(self, msg: dict, peer_id: str | None = None) -> dict:
        """Record a checker's digest against its audit; resolve if complete.

        Only a result from a peer that was *reserved* as a checker for this
        audit, echoing its issued lease, and voting once, is counted -- so an
        unassigned peer can't manufacture the reputation verdict by spamming
        ``check_only`` commits."""
        key = (tuple(msg.get("path") or ()), msg.get("gen_id"))
        member = peer_id or msg.get("worker_id")
        with self._lock:
            a = self._audits.get(key)
            if (a is not None and not a.get("resolved")
                    and a.get("check_leases", {}).get(member) == msg.get("lease")
                    and member not in a.setdefault("checked", set())):
                a["checked"].add(member)  # one vote per assigned checker
                a["checks"].append((peer_id, msg.get("digest")))
                self._maybe_resolve_audit_locked(key, time.monotonic())
        return {"type": "commit_ack", "check": True}

    def _maybe_resolve_audit_locked(self, key, now) -> None:
        a = self._audits.get(key)
        if a is None or a.get("resolved"):
            return
        complete = a["base"] is not None and len(a["checks"]) >= a["target"]
        timed_out = a["deadline"] is not None and now >= a["deadline"]
        if complete or timed_out:
            self._resolve_audit_locked(key)

    def _resolve_audit_locked(self, key) -> None:
        """Tally the audited replicas' digests and adjust reputation: agreement
        with a majority credits, the odd one out is debited. Needs >= 2 present
        and >= 2 agreeing to assign blame (a 2-way split is inconclusive)."""
        a = self._audits.pop(key, None)
        if a is None:
            return
        present = []
        if a["primary_digest"] is not None:
            present.append((a["primary_peer"], a["primary_digest"]))
        present += [(p, d) for p, d in a["checks"] if d is not None]
        digests = [d for _, d in present]
        if len(digests) >= 2:
            top, n = collections.Counter(digests).most_common(1)[0]
            if n >= 2:  # a real agreeing majority -> blame is assignable
                for peer, d in present:
                    if peer is None:
                        continue  # anonymous/HMAC: untracked
                    if d == top:
                        self.reputation.credit(peer)
                    else:
                        self.reputation.debit(peer)
                        self.metrics.record_invalid_reject()

    def _resolve_audits_locked(self, now) -> None:
        for key in list(self._audits):
            self._maybe_resolve_audit_locked(key, now)

    def _commit(self, msg: dict, peer_id: str | None = None) -> dict:
        if msg.get("check_only"):
            return self._commit_check(msg, peer_id)
        path = msg["path"]
        # Reputation verdict, applied after the lock: ``True`` credit (accepted),
        # ``False`` debit (the worker reported a non-finite loss -- diverged or
        # faulty), ``None`` neutral (stale / lost-lease: timing, not behavior).
        rep = None
        try:
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
                # A non-finite inner loss means the worker's training diverged (or
                # its hardware is faulty) -- don't grant a push for it. The
                # empty-shard no-op convention (loss=NaN, nothing) stays accepted.
                if not loss_ok(msg.get("loss"), empty=bool(msg.get("empty"))):
                    self.metrics.record_invalid_reject()
                    rep = False
                    return {"type": "commit_ack", "accepted": False}
                gen = self._completed.get(path, 0)
                self._T += 1
                self._completed[path] = gen + 1
                damp = 1.0 / (1.0 + staleness) if self.staleness_weight == "inverse" else 1.0
                push_weight = self.corpus.shard_weight(self.topology.path_index(path)) * damp
                self.metrics.record_update(staleness)
                # The grant carries the verdict to the parameter servers: weight
                # and allowed keys come from here, the lease token makes it single-use.
                grant = make_grant(path, self.topology.path_module_keys(path),
                                   push_weight, lease, self.grant_key,
                                   identity=self.identity)
                # If this primary was sampled for an audit, record the base it
                # pinned + its digest and open the window for checkers (D2).
                a = self._audits.get((tuple(path), gen))
                if a is not None and a["base"] is None:
                    a["base"] = msg.get("base")
                    a["primary_digest"] = msg.get("digest")
                    a["primary_peer"] = peer_id
                    a["deadline"] = time.monotonic() + self.audit_timeout
                    if a["base"] is None:  # primary couldn't pin (empty/no base) -> drop
                        self._audits.pop((tuple(path), gen), None)
                rep = True
                return {"type": "commit_ack", "accepted": True,
                        "push_weight": push_weight, "grant": grant}
        finally:
            if rep is True:
                self.reputation.credit(peer_id)
            elif rep is False:
                self.reputation.debit(peer_id)

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
        self._resolve_audits_locked(now)  # close out timed-out / complete audits

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
                     "epoch": -1 if record is None else record["epoch"],
                     "reputation": self.reputation.snapshot()}
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
        self.reputation.restore(state.get("reputation") or {})
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


class _CheckAborted(Exception):
    """A redundant-execution check can't reproduce its pinned base (aged out)."""


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
        check_only = bool(task.get("check_only"))
        audit = bool(task.get("audit"))
        # A check pins the audited primary's exact base; an audited primary (and
        # any check) runs *cold* so the computation is reproducible by replicas
        # -- a warm inner-optimizer state can't cross the wire (a core invariant).
        pin = {k: tuple(v) for k, v in (task.get("base") or {}).items()} if check_only else None
        cold = (path not in warm) or audit or check_only

        def fetch_keys(pin=None):
            """Fetch each key from its first responsive replica: prefer an
            already-connected owner, else rank order; a replica that is down or
            still syncing ("missing") falls back to the next one (design D8).

            With ``pin={key: version}`` (a redundant-execution check) the owner
            must serve those exact versions; an aged-out pin can't be reproduced,
            so it raises ``_CheckAborted`` and the caller abstains."""
            pending = {
                k: [a for a in addrs if a in ps_conns] + [a for a in addrs if a not in ps_conns]
                for k, addrs in routing.items()
            }
            while pending:
                addr = next(iter(pending.values()))[0]
                batch = [k for k, cands in pending.items() if cands[0] == addr]
                req = {"type": "fetch", "keys": batch, "cold": cold,
                       "have": {} if pin else
                               {k: versions.get(k) for k in batch if not is_private_key(k)}}
                if pin:
                    req["pin"] = {k: list(pin[k]) for k in batch if k in pin}
                try:
                    reply = _rpc(ps_sock(addr), req, max_msg_bytes)
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
                        if pin:
                            raise _CheckAborted  # pinned base gone -> can't reproduce
                        pending[k] = pending[k][1:]
                        if not pending[k]:
                            raise OSError(f"no replica could serve {k}")
                    else:
                        pending.pop(k)

        def load_shard_locked():
            if task.get("shard") is not None:
                shard_cache[path] = restore_shard(task["shard"])
            elif task.get("shard_spec") is not None and path not in shard_cache:
                shard_cache[path] = _materialize_from_spec(task["shard_spec"], data_ctx)
            return shard_cache[path]

        if check_only:
            # A pure verification replica: reproduce the primary's update from its
            # pinned base and report only a digest -- never push, never warm.
            digest, contrib = None, None
            try:
                fetch_keys(pin)
                engine._opt_state.pop(path, None)  # cold
                contrib = worker._train_path(path, load_shard_locked(),
                                             task["batch_size"], task["gen_id"])
                if not contrib.empty:
                    digest = pseudograd_digest(contrib.shared_delta)
            except _CheckAborted:
                pass  # abstain: the base aged out
            # Private proposal policy: also submit this (cold, reproduced)
            # private state to the owners; the owner applies it only once enough
            # distinct peers agree on the exact bytes (D5/3a).
            if (task.get("private_proposal") and contrib is not None
                    and contrib.private_state):
                by_owner: dict = {}
                for k, sd in contrib.private_state.items():
                    if k in routing:
                        by_owner.setdefault(routing[k][0], {})[k] = sd
                for addr, states in by_owner.items():
                    try:
                        _rpc(ps_sock(addr),
                             {"type": "private_proposal", "private": states,
                              "grant": task.get("grant")}, max_msg_bytes)
                    except (OSError, ConnectionError):
                        ps_conns.pop(addr, None)  # owner away; corroboration just waits
            engine._opt_state.pop(path, None)  # leave no warm trace of the check
            ack = _rpc_send(sch, send_lock, max_msg_bytes,
                            {"type": "commit", "check_only": True, "path": path,
                             "worker_id": wid, "lease": lease, "digest": digest,
                             "gen_id": task["gen_id"]})
            if ack is None:
                raise OSError("scheduler disconnected during check commit")
            continue

        stop_beat = threading.Event()
        beat = threading.Thread(target=_sch_heartbeat,
                                args=(sch_send, stop_beat, heartbeat_interval, wid, lease, path),
                                daemon=True)
        beat.start()
        base, digest = None, None
        try:
            if fault_hook is not None:
                fault_hook(path, 1)
            fetch_keys()
            if cold:
                engine._opt_state.pop(path, None)  # reset Adam on a cold start
                residuals.pop(path, None)          # and any stale error-feedback carry
            shard = load_shard_locked()
            contrib = worker._train_path(path, shard, task["batch_size"], task["gen_id"])
            if audit and not contrib.empty:
                # Pin the exact base for checkers, and digest this contribution.
                base = {k: list(versions[k]) for k in routing if k in versions}
                digest = pseudograd_digest(contrib.shared_delta)
        finally:
            stop_beat.set()
            beat.join(timeout=1)

        commit = {"type": "commit", "path": path, "worker_id": wid, "lease": lease,
                  "loss": contrib.loss, "empty": contrib.empty}
        if audit:
            commit["digest"], commit["base"] = digest, base
        ack = _rpc_send(sch, send_lock, max_msg_bytes, commit)
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

            def drop_conn(addr):
                s = ps_conns.pop(addr, None)
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass

            failed = _push_group(routing, grant, shared_payload, private_payload,
                                 ps_sock, drop_conn, max_msg_bytes)
            if failed:
                # The epoch moved under this task: the old primary refused (or
                # died). The scheduler already accepted the commit, so re-route
                # the failed keys to the *current* primaries and re-present the
                # grant once (it is single-use per server, so a new primary
                # accepts it). A second failure drops the update -- the
                # documented bounded-loss window.
                fresh = _rpc_send(sch, send_lock, max_msg_bytes,
                                  {"type": "routing", "path": path})
                if fresh is None:
                    raise OSError("scheduler disconnected during push retry")
                fresh_routing = fresh.get("routing") or {}
                retry = {k: [tuple(a) for a in fresh_routing[k]]
                         for k in failed if k in fresh_routing}
                if retry:
                    _push_group(retry, grant, shared_payload, private_payload,
                                ps_sock, drop_conn, max_msg_bytes)
            engine._opt_state[path] = contrib.opt_state          # warm-back
            _load_private(engine, contrib.private_state)
            warm.add(path)
            state["done"] += 1
            if max_tasks is not None and state["done"] >= max_tasks:
                return True
        # rejected -> discard the contribution; warm caches stay


def _push_group(routing, grant, shared_payload, private_payload, ps_sock, drop_conn,
                max_msg_bytes) -> set:
    """Push each key's update to its primary (``routing[k][0]``).

    Returns the keys whose update did **not** land for a retryable reason:
    the server refused them as ``not_primary`` (the worker's routing is from
    an older epoch) or the primary was unreachable. Non-retryable refusals
    (bad grant, replay) return nothing -- a retry could never succeed.
    """
    by_primary: dict = {}  # writes go to rank 0 only (design D3)
    for k, addrs in routing.items():
        by_primary.setdefault(tuple(addrs[0]), []).append(k)
    failed: set = set()
    for addr, keys in by_primary.items():
        updates = {k: {"grad": shared_payload[k]}
                   for k in keys if not is_private_key(k) and k in shared_payload}
        private = {k: private_payload[k]
                   for k in keys if is_private_key(k) and k in private_payload}
        if not updates and not private:
            continue
        try:
            reply = _rpc(ps_sock(addr),
                         {"type": "push", "grant": grant, "updates": updates,
                          "private": private},
                         max_msg_bytes)
        except (OSError, ConnectionError):
            drop_conn(addr)
            failed |= set(updates) | set(private)
            continue
        if reply and reply.get("reason") == "not_primary":
            failed |= set(reply.get("skipped") or [])
    return failed


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
