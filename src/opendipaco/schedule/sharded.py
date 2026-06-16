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
    apply_state_delta,
    check_mode,
    compress_shard,
    compress_state,
    encode_state_delta,
    maybe_dequantize,
    pseudograd_digest,
    restore_shard,
    state_digest,
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
from .assignment import coordinator_key, is_assignee, path_primary, version_lag
from .quorum import confirm_version, divergent_peers, read_quorum_versions, valid_report
from .identity import peer_id_of, sign_record, verify_record
from .ownership import (
    EpochManager,
    derive_epoch,
    epoch_newer,
    make_epoch_record,
    owner_addr,
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


def grant_signed_by(grant, expected_peer_id: str | None) -> bool:
    """Decentralized grant check (Phase 4 D3): an Ed25519 grant signed by exactly
    ``expected_peer_id`` — the path's primary owner, who mints the grant in place
    of the (absent) scheduler. Co-owners derive ``expected_peer_id`` from the
    epoch record, so no shared secret and no central signer are involved; an
    HMAC/unsigned grant or a grant from any other identity is refused."""
    if not isinstance(grant, dict) or not grant.get("token") or expected_peer_id is None:
        return False
    if not (verify_record(grant) and grant.get("kind") == "grant"):
        return False
    try:
        return peer_id_of(grant.get("pub", "")) == expected_peer_id
    except (ValueError, TypeError):
        return False


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


def _addr_key(addr):
    """A hashable, transport-opaque handle for a peer address (W1b orchestration).

    TCP addresses cross the wire as JSON and arrive as ``[host, port]`` lists ->
    normalize to a ``(host, port)`` tuple (hashable, what connection caches and
    routing have always keyed on). A libp2p multiaddr is a string and is already
    hashable, so it passes through unchanged. Replaces the ``tuple(addr)``
    coercions that turned a multiaddr into a tuple of characters."""
    return tuple(addr) if isinstance(addr, (list, tuple)) else addr


def _declared_shape(payload):
    """The dense shape a pseudo-gradient payload *claims*, without decoding it
    (so the receiver can bound the allocation a sparse/int4 decode would make
    before densifying). ``None`` if the payload is malformed/unrecognized."""
    try:
        if torch.is_tensor(payload):
            return tuple(payload.shape)                      # fp32 / bf16
        if isinstance(payload, dict):
            if torch.is_tensor(payload.get("q")):
                return tuple(payload["q"].shape)             # int8 {"q","s"}
            if "q4" in payload:
                return tuple(int(s) for s in payload["shape"])   # int4 per-group
            if "sp" in payload:
                return tuple(int(s) for s in payload["sp"])      # sparse top-k
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _route_target(owner):
    """The dial target a scheduler advertises for one **epoch owner entry**. A
    multi-relay NAT owner returns its **full circuit-addr candidate list**
    (``owner["addrs"]``, populated by ``make_epoch_record``) so a worker can fail
    over across relays (``Libp2pTransport.rpc`` tries them in order); a single-
    address owner returns that one address verbatim -- so a public/TCP owner
    stays ``[host, port]`` exactly as before (byte-identical routing)."""
    addrs = owner.get("addrs") or [owner["addr"]]
    return addrs if len(addrs) > 1 else owner["addr"]


# W5a: EMA weight for the per-worker effective-rate estimate. ~0.3 adapts over a
# handful of tasks -- fast enough to follow a worker that throttles, slow enough
# to ignore a one-off cold fetch or GC pause (design open-Q 2).
_RATE_EMA_ALPHA = 0.3


def _safe_version(v):
    """A *wire* version coerced to an (epoch, counter) pair, or None if
    malformed. Decentralized sources may be Byzantine (Phase 4), so a bad
    version in a fetch reply must be ignored — not crash the replication pass
    (which would skip gossip/audit too). Well-formed pairs are unchanged, so the
    central/rendezvous path behaves exactly as before."""
    if (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in v)):
        return (int(v[0]), int(v[1]))
    return None


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
                 private_policy="overwrite", private_quorum=2,
                 schedule_mode="central", salt="", lease_ttl=30.0, worker_set=None,
                 corpus_weights=None, reputation=None, rate_limiter=None,
                 min_owner_reputation=0.25, staleness_bound=None,
                 staleness_weight="inverse", read_quorum=2, k=3,
                 directory_ttl=120.0, down="full", **reactor_kw):
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
        # Delta-down (W2a): ship "current - keyframe" int8 instead of full bf16
        # weights when the worker holds a recent keyframe (in the version ring).
        # "full" (default) is byte-identical to today. Delta needs the ring, so a
        # delta owner with no history gets a default depth (the ring depth is the
        # keyframe interval: a keyframe that ages out forces a full refresh).
        if down not in ("full", "delta"):
            raise ValueError(f"down must be 'full' or 'delta', got {down!r}")
        self.down = down
        if self.down == "delta" and self.version_history <= 1:
            self.version_history = 8
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
        # The libp2p transport that serves this owner (set by serve_over_libp2p);
        # owner↔owner RPCs (_peer_rpc) reuse it to dial co-owners over libp2p /
        # through relays when their addr is a multiaddr (W1c).
        self.libp2p = None
        self._all_keys = set(self._topology.module_keys())
        self._epoch = None
        self._epoch_num = 0
        if epoch_record is not None:
            if self.peer_id is None:
                raise ValueError("epoch_record= needs identity=")
            if not verify_epoch_record(
                    epoch_record, allow_deterministic=(schedule_mode == "decentralized")):
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
        # Set => the tracker heartbeat skips re-registration (the server stays up
        # but stops refreshing its TTL): a deterministic *suspend* injection for
        # the churn harness (examples/validate_churn.py). Clear by default, so
        # the normal path is unchanged; resume_heartbeat() clears it.
        self._hb_paused = threading.Event()
        # Set on a graceful shutdown (W4c): the owner stops accepting writes so a
        # push can't race the drain and land on the departing primary (it would
        # be lost on failover). A refused push retries to the new primary post-bump.
        self._draining = False
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

        # Decentralized coordination (Phase 4, design D2/D3/D5): with no
        # scheduler, each path's *primary owner of its coordinator key* becomes
        # that path's coordinator -- it tracks the path's generation counter
        # (the version-fence), mints the Ed25519 commit grant (signed with its
        # own identity, verified by co-owners against the epoch record), and
        # hosts the reputation / rate-limit gates for commits it serves. All off
        # in ``central`` mode, which leaves the scheduler the trust root.
        if schedule_mode not in ("central", "decentralized"):
            raise ValueError(f"schedule_mode must be 'central' or 'decentralized', "
                             f"got {schedule_mode!r}")
        # Decentralized reads confirm a key by **byte-digest agreement** across its
        # k replicas (quorum reads, Phase 4c): a worker trusts only weights whose
        # digest matches the confirmed one. Lossy downlink compression breaks that
        # -- the digest is computed over the raw fp32 state (``_digests``) but the
        # fetch would ship a bf16/int8 reconstruction, whose re-quantized digest
        # differs, so *every* replica is rejected and the worker stalls. Same
        # invariant as "never bf16 a replication payload"; reject it loudly at
        # construction rather than livelock silently. (``down="delta"`` is fine:
        # the quorum fetch sends no ``have``, so the owner ships a full payload.)
        if schedule_mode == "decentralized" and self.compress != "none":
            raise ValueError(
                "schedule_mode='decentralized' requires compress='none': quorum "
                "reads confirm weights by cross-replica byte-digest agreement, "
                "which lossy downlink compression breaks. Disable transport.compress "
                "for a decentralized run.")
        self.schedule_mode = schedule_mode
        self.salt = salt
        self.lease_ttl = float(lease_ttl)
        # () -> iterable of live worker peer-ids (the gossiped directory in 4d);
        # None -> skip the HRW-assignee check and rely on the version-fence alone.
        self.worker_set = worker_set
        # path_index -> alpha shard weight (from the data spec's token counts);
        # absent entries default to 1.0 (uniform).
        self.corpus_weights = dict(corpus_weights or {})
        self.reputation = reputation if reputation is not None else Reputation()
        self.rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        self.min_owner_reputation = min_owner_reputation
        self.staleness_weight = staleness_weight
        self.staleness_bound = (staleness_bound if staleness_bound is not None
                                else 2 * len(self._topology.paths()))
        # Replicas a read cross-checks before trusting bytes, and the agreement
        # threshold for confirming a version / flagging a divergent owner (4c).
        self.read_quorum = max(1, int(read_quorum))
        # Decentralized epochs are derived from a gossiped directory, not signed
        # by a scheduler (D6): k + a self-certifying peer directory (peer_id ->
        # record) the owner imports from co-owners and the seed tracker.
        self._k = max(1, int(k))
        self.directory_ttl = float(directory_ttl)
        self._directory: dict = {}   # peer_id -> verified peer record (TTL-pruned)
        self._self_record = None     # this owner's own peer record, gossiped onward
        self._seed_addr = None       # bootstrap tracker (gossip survives its loss)
        self._tracker_auth = None    # tracker creds, for a graceful deregister (W4b)
        self._tracker_tls = None
        # path tuple -> [generation, opened_at] (the per-path clock + fence).
        self._gen: dict = {}

    @staticmethod
    def _owner_ids(key, record) -> set:
        return {o["peer_id"] for o in owners_for(key, record)}

    def _handle(self, msg: dict, nbytes: int, peer_id: str | None = None):
        kind = msg.get("type")
        if kind == "fetch":
            return self._fetch(msg, peer_id)
        if kind == "drain":
            return self._drain_recv(msg, peer_id)
        if kind == "push":
            return self._push(msg, peer_id)
        if kind == "commit":  # decentralized: this owner coordinates the path
            return self._commit(msg, peer_id)
        if kind == "generation":  # decentralized: report a path's current (g, opened_at)
            return self._generation(msg)
        if kind == "digest":  # decentralized: cheap (version, content-digest) for quorum reads
            return self._digests(msg)
        if kind == "directory":  # decentralized: gossip the peer directory (tracker = seed)
            return self._directory_rpc(msg)
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
        if self._draining:
            return False  # leaving gracefully: refuse writes so none race the drain
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

    def _private_well_shaped_locked(self, private) -> bool:
        """Every private state-dict must match its module's keys and tensor
        shapes, so a malformed push is refused rather than crashing the strict
        ``load_state_dict`` in ``_load_into`` (dtype is free -- it casts)."""
        for k, sd in private.items():
            mod = self.bank.get(k)
            if mod is None:
                continue
            if not isinstance(sd, dict):
                return False
            ref = mod.state_dict()
            if set(sd) != set(ref) or any(
                    not torch.is_tensor(sd[n]) or sd[n].shape != ref[n].shape for n in ref):
                return False
        return True

    def _down_payload_locked(self, key, have_version):
        """The downlink weights for a stale shared key (W2a). In ``down="delta"``
        mode, if the worker's keyframe (``have_version``) is still in the version
        ring, ship an int8 delta ``current - keyframe`` against the *same bytes
        the worker holds* (``compress_state``-d history, so reconstruction error
        is one bounded int8 step); otherwise -- delta off, no keyframe, or the
        keyframe aged out of the ring -- ship the full weights (a fresh
        keyframe). ``down="full"`` and the no-``have`` case are byte-identical to
        the pre-W2 path."""
        cur = _state_to_cpu(self.bank[key].state_dict())
        if self.down == "delta" and have_version:
            held = self._pinned_state_locked(key, have_version)
            if held is not None:
                base = compress_state(held, self.compress)  # == the bytes the worker holds
                # The delta is always quantized (its small range is the point);
                # int4 when the owner runs int4, else int8 (W2a/W2c).
                dmode = "int4" if self.compress == "int4" else "int8"
                return {"__delta__": list(self._versions[key]), "base": list(have_version),
                        "tensors": encode_state_delta(cur, base, dmode)}
        return compress_state(cur, self.compress)

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
            if not self._private_well_shaped_locked(states):  # malformed -> no crash on apply
                self.metrics.record_invalid_reject()
                return {"type": "ack", "applied": False}
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
                        weights[k] = self._down_payload_locked(k, tuple(have.get(k) or ()))
        out = {"type": "weights", "weights": weights, "versions": versions}
        if want_state:
            out["state"] = state
        if missing:
            out["missing"] = missing
        return out

    def _drain_recv(self, msg: dict, peer_id: str | None = None) -> dict:
        """Receive a departing primary's pushed state for keys it owns (W4c
        graceful-leave drain — the push-direction inverse of the include_state
        pull). Applied **version-gated** (only strictly-newer bytes, idempotent /
        last-writer-wins) and only from a session authenticated as a current
        owner of the key (same gate as the pull). Exact bytes + outer momentum,
        never compressed. Returns the keys adopted.

        Refused in decentralized mode: a pushed version isn't quorum-confirmable,
        so a Byzantine primary could poison a backup that then reaches false
        read-quorum (the same hazard `_replicate_once` guards). There, the normal
        pull + quorum path applies; the drain is a central/rendezvous optimization.
        """
        if self.schedule_mode == "decentralized":
            return {"type": "drained", "adopted": []}
        adopted = []
        with self._lock:
            for k, payload in (msg.get("states") or {}).items():
                if k not in self.bank or not isinstance(payload, dict):
                    continue
                if not self._state_allowed_locked(k, peer_id):  # sender must own k
                    continue
                v = _safe_version(payload.get("version"))
                sd = payload.get("weights")
                if v is None or sd is None or v <= self._versions[k]:
                    continue  # not newer (or malformed) -> ignore
                self.bank[k].load_state_dict({n: t.to(self.device) for n, t in sd.items()})
                osd = payload.get("state")
                if osd is not None and k in self._outer_opts:
                    self._outer_opts[k].load_state_dict(_opt_from_wire(osd))
                self._versions[k] = v
                adopted.append(k)
        return {"type": "drained", "adopted": adopted}

    def _drain_to_backups(self) -> dict:
        """Graceful-leave drain (W4c): for each key this owner is the **active
        primary** of, push its exact current state (weights + outer momentum +
        version) to **rank-1**, the successor that promotion will make primary.
        Rank-1 then holds the *last* accepted push, collapsing the failover loss
        window from <= replicate_interval to ~0 for a clean leave.

        Best-effort: any send failure falls back to the normal pull window, so a
        drain is never worse than an abrupt leave. No-op in static/decentralized
        mode (no successor concept / pushed state isn't quorum-safe)."""
        if self.schedule_mode == "decentralized":
            return {}
        with self._lock:
            epoch = self._epoch
            if epoch is None or self.peer_id is None:
                return {}
            by_peer: dict = {}  # successor peer_id -> (dial target, {key: payload})
            for k in sorted(self.owned_keys):
                if k not in self._active or k not in self.bank:
                    continue
                owners = owners_for(k, epoch)
                if len(owners) < 2 or owners[0]["peer_id"] != self.peer_id:
                    continue  # only the primary drains, and only if a successor exists
                rank1 = owners[1]
                payload = {"version": self._versions[k],
                           "weights": _state_to_cpu(self.bank[k].state_dict())}
                if k in self._outer_opts:
                    payload["state"] = _opt_to_wire(
                        _optimizer_state_to_cpu(self._outer_opts[k].state_dict()))
                by_peer.setdefault(rank1["peer_id"],
                                   (self._owner_targets(rank1), {}))[1][k] = payload
        results: dict = {}
        for target, states in by_peer.values():       # sent outside the lock
            try:
                reply = self._peer_rpc(target, {"type": "drain", "states": states})
                for k in (reply or {}).get("adopted", []):
                    results[k] = "drained"
            except (OSError, ConnectionError):
                pass  # best-effort -> fall back to the replicate_interval pull window
        return results

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
        # Decentralized epochs are signer-less and derived by recomputation
        # (D6), so accept a well-formed deterministic record; central/rendezvous
        # epochs still require the scheduler's signature.
        if not verify_epoch_record(
                record, allow_deterministic=(self.schedule_mode == "decentralized")):
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
                if self.schedule_mode == "decentralized":
                    self._gossip_once()         # directory gossip + re-derive epoch (4d)
                    self._audit_digests_once()  # cross-check co-owners' digests (4c)
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
                    # _owner_targets gives a NAT co-owner's full relay candidate
                    # list (libp2p, tried in order for failover) or its single TCP
                    # addr; dedup by peer_id since a candidate list isn't hashable.
                    if o["peer_id"] in seen:
                        continue
                    seen.add(o["peer_id"])
                    addrs.append(self._owner_targets(o))
                if addrs:
                    candidates[k] = addrs
                    results[k] = "pending"
                elif k not in self._active:
                    # Sole owner of the key (no other replica anywhere): it is
                    # authoritative by definition -- e.g. a k=1 restart.
                    self._active.add(k)
                    results[k] = "active"
            was_active = set(self._active)
        # Decentralized safety (Codex P1): a Byzantine source can serve a poisoned
        # higher version; adopting it blindly would let it reach read-quorum (the
        # source + the deceived backup) and be *confirmed* before the audit runs.
        # So in decentralized mode never mutate local state to a version whose
        # (version, digest) isn't already agreed by a quorum of the key's
        # replicas -- an unconfirmed higher version is left for a later pass.
        confirmed: dict = {}
        if self.schedule_mode == "decentralized" and epoch is not None and candidates:
            addrs = sorted({_addr_key(o["addr"]) for k in candidates
                            for o in owners_for(k, epoch)})
            confirmed = read_quorum_versions(
                addrs, list(candidates), self.read_quorum,
                lambda a, m: self._peer_rpc(a, m))
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
                    v = _safe_version(answers.get(k))  # ignore a Byzantine source's malformed version
                    if v is not None:
                        if v > self._versions[k]:
                            sd = (reply.get("weights") or {}).get(k)
                            # Decentralized: adopt only quorum-confirmed bytes
                            # (the offered version's digest must match what a
                            # quorum of replicas agreed, so poison can't sneak in
                            # via a single source). Central/rendezvous: trust the
                            # authoritative source as before (Phase 2).
                            if sd is not None and self.schedule_mode == "decentralized":
                                c = confirmed.get(k)
                                if not (c is not None and tuple(c[0]) == v
                                        and state_digest(sd) == c[1]):
                                    sd = None  # unconfirmed -> leave for a later pass
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
        from .tracker import make_peer_record, register_peer  # lazy: tracker imports this

        addr = tuple(tracker_addr)
        # The tracker is the bootstrap seed for gossip (D7); a fresh self-record
        # is what this owner gossips onward so its membership propagates even
        # after the tracker is gone.
        self._seed_addr = addr
        # Remembered so a graceful shutdown can deregister (W4b): the scheduler's
        # epoch watcher then fails this owner over immediately, skipping grace.
        self._tracker_auth, self._tracker_tls = auth_key, tls

        def beat():
            while not (self._stop or self._dead):
                if not self._hb_paused.is_set():  # suspended: let the TTL lapse
                    try:
                        self._self_record = make_peer_record(
                            self.identity, reachability="public",
                            addr=(advertise_host, self.port), roles=roles,
                            capabilities=capabilities)
                        register_peer(addr, self.identity, reachability="public",
                                      peer_addr=(advertise_host, self.port), roles=roles,
                                      capabilities=capabilities, auth_key=auth_key, tls=tls)
                    except (OSError, ConnectionError):
                        pass  # tracker briefly away; the next beat retries
                if self._repl_stop.wait(interval):
                    return

        self._beat_thread = threading.Thread(target=beat, daemon=True)
        self._beat_thread.start()

    def pause_heartbeat(self) -> None:
        """Stop refreshing the tracker TTL while the server keeps running -- a
        deterministic *suspend* (sleep) injection for the churn harness. The
        record lapses after the tracker's TTL exactly as a slept laptop would."""
        self._hb_paused.set()

    def resume_heartbeat(self) -> None:
        """Resume tracker re-registration after :meth:`pause_heartbeat` (wake)."""
        self._hb_paused.clear()

    def _owner_targets(self, owner):
        """Dial target(s) for an owner epoch entry: a candidate list of its relay
        circuit addrs (libp2p — tried in order for multi-relay failover, W1c) or
        a single ``(host, port)`` (TCP)."""
        if self.libp2p is not None:
            return owner.get("addrs") or [owner["addr"]]
        return _addr_key(owner["addr"])

    def _peer_rpc(self, addr, msg):
        # libp2p owners (W1c): a co-owner's addr is a multiaddr (direct or a
        # /p2p-circuit through a relay), or a *list* of its k relay circuit addrs
        # tried in order for failover -> dial over the owner's libp2p transport,
        # which handles connection reuse + relay routing.
        if self.libp2p is not None and isinstance(addr, (str, list)):
            return self.libp2p.rpc(addr, msg, timeout=60.0)
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

    def shutdown(self, *, graceful: bool = False) -> None:
        """Stop the owner. ``graceful=True`` first sends a signed deregister to
        the tracker (W4b) so the scheduler's epoch watcher fails this owner over
        **immediately**, skipping ``owner_grace`` -- a clean leave instead of a
        timeout. Best-effort: if the tracker is unreachable, the TTL+grace path
        still applies, so a graceful shutdown is never worse than an abrupt one.
        The default (``graceful=False``) is byte-identical to before."""
        self._repl_stop.set()
        if graceful and self.peer_id is not None:
            # Stop the heartbeat *before* deregistering, else a re-registration
            # racing the deregister could land after it (with a newer issued_at)
            # and resurrect the record, undoing the tombstone. We set _repl_stop
            # above (which ends the beat loop) and join the beat thread; any
            # register already in flight carries an older issued_at, so the
            # tracker refuses it as stale behind our (newer) tombstone.
            self._hb_paused.set()
            if self._beat_thread is not None:
                self._beat_thread.join(timeout=2.0)
            # Stop accepting writes, then flush any buffered (already-accepted)
            # robust-aggregation work into the bank, *then* drain. Order matters:
            # a push that raced the drain would land on the departing primary and
            # be lost, and buffered contributions flushed *after* the drain would
            # never reach rank-1 -- both re-open the loss window. With writes
            # fenced and buffers flushed first, the state we drain is final; a
            # refused push retries to the new primary once the epoch bumps (W4c).
            with self._lock:
                self._draining = True
                self._flush_all_buffers_locked()  # accepted work -> bank, before draining
            # Drain *before* deregistering: while we're still the valid primary
            # (epoch not yet bumped), push our latest state to each key's rank-1
            # successor, so a promoted backup holds the last accepted push (W4c).
            self._drain_to_backups()
            if self.identity is not None and self._seed_addr is not None:
                try:
                    from .tracker import deregister_peer  # lazy: tracker imports this
                    # Short timeout: a closing node (laptop lid) must not block on
                    # a slow/absent tracker -- past the budget, fall back to grace.
                    deregister_peer(self._seed_addr, self.identity, timeout=3.0,
                                    auth_key=self._tracker_auth, tls=self._tracker_tls)
                except (OSError, ConnectionError):
                    pass  # tracker away -> fall back to TTL+grace expiry
        with self._lock:
            self._flush_all_buffers_locked()  # don't drop accepted-but-buffered work
        for s in self._peer_conns.values():
            try:
                s.close()
            except OSError:
                pass
        self._peer_conns.clear()
        super().shutdown()

    # -- decentralized coordination (Phase 4 D2/D3/D5) -------------------------

    def _coordinates_locked(self, path) -> bool:
        """Is this owner the coordinator for ``path`` -- the active primary of
        the path's :func:`coordinator_key`? Only the coordinator advances the
        path's generation and mints its grants."""
        if self.schedule_mode != "decentralized" or self._epoch is None:
            return False
        ck = coordinator_key(self._topology.path_module_keys(path))
        return self._primary_locked(ck)

    def _generation(self, msg: dict) -> dict:
        """Report a coordinated path's current generation, so a self-assigning
        worker knows which generation to commit (the version-fence value)."""
        try:
            path = tuple(msg["path"])
        except (KeyError, TypeError):
            return {"type": "generation", "ok": False}
        with self._lock:
            if not self._coordinates_locked(path):
                return {"type": "generation", "ok": False, "epoch": self._epoch_num}
            g, opened = self._gen.setdefault(path, [0, time.monotonic()])
            # ``age`` (seconds the current generation has been open) lets a
            # self-assigning worker compute takeover-on-expiry without sharing a
            # clock with the owner: it feeds ``responsible_rank(elapsed=age, ...)``,
            # the same value the coordinator uses at commit (``now - opened``).
            return {"type": "generation", "ok": True, "generation": g,
                    "age": max(0.0, time.monotonic() - opened),
                    "epoch": self._epoch_num, "lease_ttl": self.lease_ttl,
                    "staleness_bound": self.staleness_bound}

    def _commit(self, msg: dict, peer_id: str | None = None) -> dict:
        """A self-assigned worker commits ``(path, generation)`` to its
        coordinator (this owner), which version-fences the slot, verifies the
        worker is the HRW assignee, gates on reputation/rate-limit/loss/staleness
        exactly as the central scheduler did, then **mints an Ed25519 grant
        signed with its own identity** (the path's co-owners verify it against
        the epoch record). Replaces ``Scheduler._commit`` in decentralized mode;
        the central path is untouched."""
        if self.schedule_mode != "decentralized":
            return {"type": "commit_ack", "accepted": False, "reason": "not_decentralized"}
        try:
            path = tuple(msg["path"])
        except (KeyError, TypeError):
            return {"type": "commit_ack", "accepted": False}
        rep_outcome = None  # True credit (accepted) / False debit (bad loss) / None neutral
        try:
            with self._lock:
                if not self._coordinates_locked(path):
                    return {"type": "commit_ack", "accepted": False,
                            "reason": "not_coordinator", "epoch": self._epoch_num}
                entry = self._gen.setdefault(path, [0, time.monotonic()])
                g, opened = entry
                if msg.get("generation") != g:
                    # Version-fence: the slot already advanced (someone committed
                    # this generation) or the worker is stale -> dropped commit.
                    return {"type": "commit_ack", "accepted": False,
                            "reason": "stale_generation", "generation": g}
                # HRW-assignee check: load distribution + anti-grab. Skipped when
                # no worker directory is available, leaving the fence as the gate.
                member = peer_id or msg.get("worker_id")
                workers = list(self.worker_set() or []) if self.worker_set else []
                if workers and not is_assignee(
                        member, path, g, workers, salt=self.salt,
                        elapsed=time.monotonic() - opened, lease_ttl=self.lease_ttl):
                    return {"type": "commit_ack", "accepted": False, "reason": "not_assignee"}
                rep = self.reputation.get(peer_id)
                if not self.rate_limiter.allow(peer_id, reputation=rep):
                    return {"type": "commit_ack", "accepted": False, "reason": "throttled"}
                if not loss_ok(msg.get("loss"), empty=bool(msg.get("empty"))):
                    self.metrics.record_invalid_reject()
                    rep_outcome = False
                    return {"type": "commit_ack", "accepted": False, "reason": "bad_loss"}
                # Staleness = version-vector lag over the path keys this owner
                # holds (>= the coordinator key, whose counter tracks this path's
                # generations). Cross-epoch / remapped base -> drop (Phase 2
                # failover semantics).
                base = {k: v for k, v in (msg.get("base_versions") or {}).items()
                        if k in self._versions}
                lag = version_lag(base, {k: self._versions[k] for k in base})
                if lag is None or lag > self.staleness_bound:
                    self.metrics.record_stale_reject()
                    return {"type": "commit_ack", "accepted": False, "reason": "stale"}
                entry[0], entry[1] = g + 1, time.monotonic()  # fence closes, slot reopens
                damp = 1.0 / (1.0 + lag) if self.staleness_weight == "inverse" else 1.0
                push_weight = self.corpus_weights.get(self._topology.path_index(path), 1.0) * damp
                grant = make_grant(path, self._topology.path_module_keys(path),
                                   push_weight, uuid.uuid4().hex, identity=self.identity)
                self.metrics.record_update(lag)
                rep_outcome = True
                return {"type": "commit_ack", "accepted": True, "generation": g + 1,
                        "push_weight": push_weight, "grant": grant}
        finally:
            if rep_outcome is True:
                self.reputation.credit(peer_id)
            elif rep_outcome is False:
                self.reputation.debit(peer_id)

    def _digests(self, msg: dict) -> dict:
        """Cheap ``{key: [version, content-digest]}`` for this owner's active
        keys, so a reader cross-checks across replicas (quorum reads) without
        pulling weights. States are snapshotted under the lock and hashed
        outside it."""
        requested = msg.get("keys")
        with self._lock:
            sel = set(requested) & self._active if requested else set(self._active)
            snap = {k: (self._versions[k], _state_to_cpu(self.bank[k].state_dict()))
                    for k in sel if k in self.bank}
            epoch = self._epoch_num
        digests = {k: [list(v), state_digest(sd)] for k, (v, sd) in snap.items()}
        return {"type": "digest", "digests": digests, "epoch": epoch}

    def _apply_digest_audit(self, reports_by_key: dict) -> dict:
        """Confirm each key's value by quorum and debit the owner-behaviour
        reputation of any replica whose digest at the confirmed version
        contradicts the majority (design D4). ``reports_by_key`` is
        ``{key: {peer_id: (version, digest)}}``; returns the flagged peers per
        key. Pure given the reports, so the network gather (below) and tests
        share one rule."""
        flagged: dict = {}
        for k, reports in reports_by_key.items():
            confirmed = confirm_version(list(reports.values()), self.read_quorum)
            bad = divergent_peers(reports, confirmed)
            for pid in bad:
                self.reputation.debit(pid)  # owner-behaviour debit -> eviction (4d)
            if bad:
                flagged[k] = bad
        return flagged

    def _audit_digests_once(self) -> dict:
        """Gather co-owners' digests for the keys this owner holds and audit
        them (decentralized replication-loop step). A co-owner mid-restart that
        doesn't answer simply isn't in the tally — no false blame."""
        if self.schedule_mode != "decentralized" or self._epoch is None or self.peer_id is None:
            return {}
        with self._lock:
            epoch = self._epoch
            keys = sorted(self.owned_keys & self._active)
            snap = {k: (self._versions[k], _state_to_cpu(self.bank[k].state_dict()))
                    for k in keys if k in self.bank}
        reports: dict = {k: {self.peer_id: (tuple(v), state_digest(sd))}
                         for k, (v, sd) in snap.items()}
        by_peer: dict = {}  # peer_id -> (dial target(s), [keys it co-owns with us])
        for k in reports:
            for o in owners_for(k, epoch):
                if o["peer_id"] == self.peer_id:
                    continue
                # _owner_targets gives a co-owner's relay candidates (libp2p) so
                # the digest fetch fails over across its k relays (W1c).
                by_peer.setdefault(o["peer_id"], (self._owner_targets(o), []))[1].append(k)
        for pid, (addr, ks) in by_peer.items():
            try:
                reply = self._peer_rpc(addr, {"type": "digest", "keys": ks})
            except (OSError, ConnectionError):
                reply = None
            for k, vd in ((reply or {}).get("digests") or {}).items():
                r = valid_report(vd)  # drop a Byzantine co-owner's malformed report
                if k in reports and r is not None:
                    reports[k][pid] = r
        return self._apply_digest_audit(reports)

    # -- directory gossip + deterministic epochs (Phase 4 D6/D7) ---------------

    @staticmethod
    def _issued_at(record) -> float | None:
        """A record's ``issued_at`` if it is a real number, else None.
        ``verify_record`` only checks the signature, so a validly-signed but
        malformed record (non-numeric ``issued_at``) must be rejected before any
        TTL arithmetic (Codex P2) -- otherwise one bad record aborts gossip."""
        ts = record.get("issued_at")
        return float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else None

    def _prune_directory_locked(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        for p in [p for p, r in self._directory.items()
                  if (t := self._issued_at(r)) is None or now - t > self.directory_ttl]:
            del self._directory[p]

    def import_directory(self, records, *, now: float | None = None) -> int:
        """Merge gossiped peer records into the local directory (newest
        ``issued_at`` wins; expired/malformed records are dropped). Each record
        is self-certifying, so importing from any peer is safe -- this is how the
        swarm's membership survives the tracker (design D7)."""
        now = time.time() if now is None else now
        added = 0
        with self._lock:
            for r in records:
                if not (isinstance(r, dict) and verify_record(r) and r.get("kind") == "peer"):
                    continue
                pid = r.get("peer_id")
                ts = self._issued_at(r)
                if not isinstance(pid, str) or ts is None or now - ts > self.directory_ttl:
                    continue
                cur = self._directory.get(pid)
                if cur is None or ts > self._issued_at(cur):
                    self._directory[pid] = r
                    added += 1
            self._prune_directory_locked(now)
        return added

    def _directory_rpc(self, msg: dict) -> dict:
        """Serve this owner's directory view (its own record first) so a peer can
        gossip from it -- the tracker is only a bootstrap seed."""
        with self._lock:
            self._prune_directory_locked()
            recs = [r for r in self._directory.values() if r["peer_id"] != self.peer_id]
            if self._self_record is not None:
                recs.append(self._self_record)
        return {"type": "directory", "records": recs}

    def derive_and_apply_epoch(self):
        """Derive the epoch deterministically from the local directory (D6) and
        adopt it if newer. The reputation gate excludes owners debited for
        divergence (4c) -- this is the eviction step. Returns the live record."""
        if self.schedule_mode != "decentralized" or self.peer_id is None:
            return None
        with self._lock:
            self._prune_directory_locked()
            recs = [r for r in self._directory.values() if r["peer_id"] != self.peer_id]
            if self._self_record is not None:
                recs.append(self._self_record)
            prev = self._epoch
        record = derive_epoch(
            recs, k=self._k, salt=self.salt, prev=prev,
            is_eligible=lambda pid: self.reputation.eligible(pid, self.min_owner_reputation))
        if prev is None or epoch_newer(record, prev):
            self.apply_epoch(record)
            return record
        return prev

    def _gossip_once(self) -> None:
        """Pull directories from the seed tracker + current co-owners, import
        them, and re-derive the epoch (decentralized replication-loop step)."""
        if self.schedule_mode != "decentralized" or self.peer_id is None:
            return
        with self._lock:
            addrs = {self._seed_addr} if self._seed_addr else set()
            self_addr = None if self._self_record is None else tuple(self._self_record["addr"])
            if self._epoch is not None:
                for o in self._epoch["owners"]:
                    a = _addr_key(o["addr"])
                    if a != self_addr:
                        addrs.add(a)
        for addr in addrs:
            try:
                reply = self._peer_rpc(addr, {"type": "directory"})
            except (OSError, ConnectionError):
                reply = None
            recs = (reply or {}).get("records")
            if recs:
                self.import_directory(recs)
        self.derive_and_apply_epoch()

    def _expected_grant_signer(self, grant) -> str | None:
        """The peer-id that legitimately mints a push grant for this path under
        the current epoch -- the primary owner of the path's coordinator key
        (decentralized mode). The grant carries the path; the epoch resolves the
        primary, so co-owners agree without a shared secret."""
        try:
            path_keys = self._topology.path_module_keys(tuple(grant["path"]))
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        prim = path_primary(path_keys, self._epoch) if self._epoch is not None else None
        return prim["peer_id"] if prim else None

    def _push(self, msg: dict, peer_id: str | None = None) -> dict:
        grant = msg.get("grant")
        if self.schedule_mode == "decentralized":
            # No scheduler: the grant must be signed by the path's primary owner.
            ok = grant_signed_by(grant, self._expected_grant_signer(grant))
        else:
            ok = verify_grant(grant, self.grant_key, scheduler_pub=self.scheduler_pub)
        if not ok:
            return {"type": "ack", "applied": False}  # no/forged grant -> refuse
        # Decode (possibly quantized) gradients outside the lock; a malformed
        # encoding refuses the push rather than crashing the server. Snapshot the
        # target modules' parameter shapes under the lock first, then validate the
        # *declared* shape of each payload BEFORE maybe_dequantize densifies it: a
        # sparse/int4 payload declares its dense shape and the decode allocates
        # math.prod(shape), which max_msg_bytes (a bound on the encoded frame)
        # does NOT cover -- so a tiny push could claim a huge shape and OOM us.
        raw_updates = msg.get("updates") or {}
        private = msg.get("private") or {}
        if not (isinstance(raw_updates, dict) and isinstance(private, dict)):
            self.metrics.record_invalid_reject()  # a non-dict would crash .items()
            return {"type": "ack", "applied": False}
        with self._lock:
            expected = {k: [tuple(p.shape) for p in self.bank[k].parameters()]
                        for k in raw_updates if k in self.bank}
        try:
            updates = {}
            for k, u in raw_updates.items():
                shapes = expected.get(k)         # foreign key (not ours) -> not decoded
                if shapes is None or not isinstance(u, dict):
                    continue
                grad = u.get("grad")
                if (not isinstance(grad, list) or len(grad) != len(shapes)
                        or any(_declared_shape(p) != s for p, s in zip(grad, shapes))):
                    raise ValueError("grad payload shape mismatch")
                updates[k] = maybe_dequantize(grad)
        except (TypeError, KeyError, ValueError):
            self.metrics.record_invalid_reject()
            return {"type": "ack", "applied": False}
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
            # (Grad shapes/count were already checked against the target params
            # pre-decode; private state is validated here.)
            if not (all_finite(updates) and all_finite(private)
                    and self._private_well_shaped_locked(private)):
                self.metrics.record_invalid_reject()
                return {"type": "ack", "applied": False}
            # Foreign update keys (pushed to a PS that doesn't hold them -- stale
            # routing) were not decoded; still report them so the worker re-routes.
            if self._epoch is not None:
                skipped.extend(k for k in raw_updates if k not in self.bank)
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
                 private_policy="overwrite", down="full", up_density=1.0,
                 task_seconds=None, park_factor=3.0, min_task_rate=None, **reactor_kw):
        super().__init__(host=host, port=port, auth_key=auth_key, **reactor_kw)
        # W5b: target wall-time per task. None (default) -> no sizing, every task
        # is the configured (batch_size, inner_steps) -> byte-identical anchor.
        # When set, slow workers get smaller tasks so their lease lands in ~this.
        self.task_seconds = task_seconds
        # W5c: a worker whose minimum task (batch=1, inner=1) is estimated to take
        # longer than task_seconds * park_factor (or whose rate is below an
        # absolute min_task_rate) is *parked* -- given idle instead of a lease so
        # it can't hold a path hostage. Parking is re-evaluated each request and
        # lets one through per cooldown to re-measure, so a worker that speeds up
        # rejoins. Only active when task_seconds is set.
        self.park_factor = park_factor
        self.min_task_rate = min_task_rate
        self._parked: dict = {}   # worker_id -> monotonic time it was last let through
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
        # Downlink policy stamped on tasks (W2a): "delta" tells the worker to keep
        # keyframe baselines and send keyframe versions in `have`, so owners can
        # ship deltas. "full" (default) is byte-identical to today. Owners must
        # also run down="delta" to actually ship deltas (self-describing payloads
        # keep a mismatch safe: a full-mode owner just ships full).
        if down not in ("full", "delta"):
            raise ValueError(f"down must be 'full' or 'delta', got {down!r}")
        self.down = down
        # Up-path sparsification (W2b): the worker keeps each pseudo-gradient's
        # top `up_density` fraction (per-row for 2-D weights) and error-feeds the
        # dropped mass. 1.0 (default) = dense = byte-identical. Stamped on tasks.
        if not 0.0 < up_density <= 1.0:
            raise ValueError(f"up_density must be in (0, 1], got {up_density!r}")
        self.up_density = float(up_density)
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
        self.ps_addrs = [_addr_key(a) for a in ps_addrs]
        # Routing values are *replica lists* in rank order (primary first); the
        # static map has one entry per key. With no ps_addrs the scheduler is in
        # rendezvous mode: routing derives from the published epoch instead.
        if self.ps_addrs:
            key_shard = assign_shards(self.topology.module_keys(), len(self.ps_addrs))
            self._routing = {k: [self.ps_addrs[s]] for k, s in key_shard.items()}
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
        # W5a: per-worker effective rate (tokens / lease-second), measured from
        # lease->commit timing. dynamics-neutral; W5b sizes tasks from it.
        self._lease_at: dict = {}    # path -> monotonic lease time (current lease)
        self._lease_work: dict = {}  # path -> tokens issued in the leased task
        self._worker_rate: dict = {}  # worker_id -> EMA effective rate

    def _handle(self, msg: dict, nbytes: int, peer_id: str | None = None):
        kind = msg.get("type")
        if kind == "request":
            return self._next_task(msg, peer_id)
        if kind == "commit":
            return self._commit(msg, peer_id)
        if kind == "routing":
            return self._fresh_routing(msg)
        if kind == "nack":
            return self._nack(msg)
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
        from .tracker import (  # lazy: tracker imports this
            fetch_directory_and_tombstones,
            get_epoch,
            put_epoch,
        )

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
                    records, tombstoned = fetch_directory_and_tombstones(
                        addr, roles=["owner"], reachability="public",
                        auth_key=tracker_auth, tls=tracker_tls)
                    due = manager.observe(records, tombstoned=tombstoned)
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
            return {k: [_route_target(o) for o in owners_for(k, self._epoch_record)]
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
            # Per-worker batch ceiling, clamped to its advertised memory cap; W5b
            # sizes a task down from this toward task_seconds. ``inner`` defaults
            # to the configured count (the ceiling).
            batch_cap = (max(1, min(self.batch_size, int(caps["max_batch"])))
                         if caps.get("max_batch") else self.batch_size)
            inner = self.diloco.inner_steps
            # W5b: size (batch, inner) for this worker toward task_seconds (shrink
            # only; the full configured task when off / fast / no estimate). Done
            # before parking so the park cadence uses the task it would *actually*
            # get. The check branch overrides these with the audited primary's pin.
            batch, inner = self._size_task_locked(wid, batch_cap)
            # W5c: park a worker too slow even for the minimum task so it holds no
            # path (or check). Re-measure one task per `park_factor` of *this
            # worker's own* sized-task time -- a fixed wall cooldown would be shorter
            # than a parked worker's task time and so never actually idle it.
            # Using the sized-task time (not seq/rate) means a worker parked by the
            # absolute min_task_rate floor -- whose task isn't shrunk by
            # task_seconds -- is idled too, not just one whose task overshoots.
            if self._too_slow_locked(wid):
                now = time.monotonic()
                last = self._parked.get(wid)
                task_time = batch * inner * self.config.sequence_length / self._worker_rate[wid]
                if last is not None and now - last < self.park_factor * task_time:
                    return self._idle()
                self._parked[wid] = now   # let this one through to re-measure
            else:
                self._parked.pop(wid, None)
            if not eligible:
                # No primary work: absorb the surplus as a redundant check for an
                # open audit (§1.9), if one needs this distinct worker.
                cand = self._find_check_locked(member)
                if cand is None:
                    return self._idle()
                if not self.rate_limiter.allow(peer_id, reputation=rep):
                    return self._idle()
                (path, generation, base, lease, check_private, check_grant,
                 pin_batch, pin_inner) = self._reserve_check_locked(cand, member)
                check_only = True
                # Reproduce the audited primary's exact computation (design D8):
                # a check must use the *primary's* batch/inner, not this checker's
                # own (sized) values -- else the digest diverges (and a heterogeneous
                # max_batch would falsely flag every audit).
                batch = pin_batch or batch_cap
                inner = pin_inner or self.diloco.inner_steps
            else:
                # Rate limit only the *expensive* path (issuing a task with a
                # weight/shard payload): a throttled peer gets a cheap backoff
                # idle, not a disconnect (§1.14). Reputation scales its bucket.
                if not self.rate_limiter.allow(peer_id, reputation=rep):
                    return self._idle()
                path = min(eligible, key=lambda p: (self._completed[p], p not in warm, p))
                lease = uuid.uuid4().hex  # unique per lease; fences commit/heartbeat
                # batch, inner already sized above (before the parking check).
                self._owner[path] = wid
                self._inflight[path] = time.monotonic() + self.heartbeat_timeout
                self._issued[path] = self._T
                self._lease[path] = lease
                # W5a: remember when this lease opened and how much work it carries
                # (the *sized* work), so its commit yields an effective-rate sample.
                self._lease_at[path] = time.monotonic()
                self._lease_work[path] = batch * inner * self.config.sequence_length
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
                        # Pin the primary's task size so checkers reproduce its
                        # exact computation (design D8).
                        "batch": batch, "inner": inner,
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
            "density": self.up_density,  # uplink top-k sparsification (W2b); 1.0 = dense
            "down": self.down,          # downlink policy: keep keyframes for deltas (W2a)
            "shard": compress_shard(shard, self.compress),
            "shard_spec": shard_spec,
            "batch_size": batch,
            "total_rounds": self.total_rounds,
            "seed": self.seed,
        }
        # Per-task inner_steps travels only when sizing is active or for a check
        # (which pins the primary's count); absent otherwise -> the worker uses
        # its configured default -> byte-identical to the pre-W5 task (D6).
        if self.task_seconds or check_only:
            task["inner_steps"] = inner
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
        return (path, generation, a["base"], lease, bool(a.get("private")), grant,
                a.get("batch"), a.get("inner"))

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
                    # stale / already freed / not the current lease holder. Don't
                    # touch the rate timing -- it belongs to the *current* lease.
                    return {"type": "commit_ack", "accepted": False}
                staleness = self._T - self._issued.get(path, self._T)
                self._inflight.pop(path, None)
                self._lease.pop(path, None)
                # W5a: claim this lease's rate sample now (only *recorded* on an
                # accepted, non-empty commit below; rejects just drop it).
                rate_wid = self._owner.get(path)
                lease_t0 = self._lease_at.pop(path, None)
                lease_work = self._lease_work.pop(path, None)
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
                # W5a: a real, accepted unit of work -> an effective-rate sample.
                if lease_t0 is not None and not msg.get("empty"):
                    self._record_rate_locked(rate_wid, lease_work,
                                             time.monotonic() - lease_t0)
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

    def _nack(self, msg: dict) -> dict:
        """A worker voluntarily returning its lease (graceful leave, W4b): free
        the in-flight lease **now** so the path is re-leasable immediately
        instead of after the heartbeat timeout. Fenced by the lease token (a
        stale/zombie nack can't free a path that's since been re-leased), and
        reputation-neutral -- returning work cleanly is not a fault."""
        path = msg.get("path")
        with self._lock:
            if path in self._inflight and msg.get("lease") == self._lease.get(path):
                self._inflight.pop(path, None)
                self._lease.pop(path, None)
                self._lease_at.pop(path, None)    # returned, not committed: no rate sample
                self._lease_work.pop(path, None)
                return {"type": "nack_ack", "freed": True}
        return {"type": "nack_ack", "freed": False}

    def _record_rate_locked(self, wid, work, elapsed) -> None:
        """Fold one (work, lease-duration) sample into a worker's EMA effective
        rate (tokens / second). Caller holds the lock. Guards against a zero/
        negative duration (instant in-process commit) so the EMA can't blow up."""
        if wid is None or work is None or work <= 0 or elapsed <= 0:
            return
        rate = work / elapsed
        prev = self._worker_rate.get(wid)
        self._worker_rate[wid] = rate if prev is None else (
            (1 - _RATE_EMA_ALPHA) * prev + _RATE_EMA_ALPHA * rate)

    def worker_rate(self, wid):
        """A worker's measured effective rate (tokens per lease-second) as an EMA,
        or ``None`` if it hasn't completed a task yet (W5a). W5b sizes the next
        task from this so the lease lands in ~``task_seconds``."""
        with self._lock:
            return self._worker_rate.get(wid)

    def _size_task_locked(self, wid, batch_ceiling):
        """Size ``(batch_size, inner_steps)`` for ``wid`` so its lease lands in
        ~``task_seconds``, from its measured rate (W5b, design D2/D3). **Shrink
        only**: the configured ``(batch_ceiling, inner_steps)`` is the ceiling, so
        a fast worker (or one with no estimate yet, or sizing off) gets the full
        configured task -- the anchor task. A slow worker gets a smaller **batch**
        first (the gentle, bandwidth-neutral lever that keeps the inner-step
        count fixed); ``inner_steps`` shrinks only once batch is floored at 1 and
        the task still overshoots the target."""
        inner = self.diloco.inner_steps
        rate = self._worker_rate.get(wid)
        if not self.task_seconds or not rate:
            return batch_ceiling, inner
        seq = self.config.sequence_length
        target = rate * self.task_seconds                      # target tokens
        batch = max(1, min(batch_ceiling, round(target / (inner * seq))))
        if batch == 1 and inner * seq > target:                # batch floored, still over
            inner = max(1, min(inner, round(target / seq)))
        return batch, inner

    def _too_slow_locked(self, wid) -> bool:
        """Is ``wid`` too slow to lease without straggling -- its measured rate
        implies even the minimum task (batch=1, inner=1 -> seq tokens) overshoots
        ``task_seconds * park_factor`` (or it is below an absolute
        ``min_task_rate``)? Only when sizing is on and an estimate exists (a fresh
        worker bootstraps with a full task and isn't parked)."""
        if not self.task_seconds:
            return False
        rate = self._worker_rate.get(wid)
        if not rate:
            return False
        if self.min_task_rate is not None and rate < self.min_task_rate:
            return True
        return self.config.sequence_length / rate > self.task_seconds * self.park_factor

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
                self._lease_at.pop(path, None)    # reclaimed, not committed: no rate sample
                self._lease_work.pop(path, None)
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
            self._lease_at, self._lease_work = {}, {}  # no stale rate timing across fits (W5a)
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
            addrs = (sorted({_addr_key(o["addr"]) for o in record["owners"]})
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
        for addr in sorted({_addr_key(o["addr"]) for o in record["owners"]}):
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
                       max_batch_size=None, transport="tcp", identity=None,
                       stop_event=None):
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
    versions: dict = {}          # shared key -> held (trained-against / nominal) version
    keyframes: dict = {}         # shared key -> (version, exact state) baseline for deltas (W2a)
    residuals: dict = {}         # path -> {key: [tensors]}: compression error feedback
    data_ctx = {"dir": data_dir, "source": data_source, "tokenizer": data_tokenizer}
    caps = {"device": str(device)}
    if max_batch_size is not None:
        caps["max_batch"] = int(max_batch_size)
    state = {"done": 0}

    if transport == "libp2p":
        # libp2p path: scheduler_addr + routing addrs are multiaddrs; the worker
        # dials owners (direct or through a relay) over Noise streams. Lazy import
        # so the default install never imports libp2p (W1, optional [nat] extra).
        from .p2p import Libp2pTransport, _Libp2pLink

        if identity is None:
            raise ValueError("transport='libp2p' needs identity=")
        # Honor the worker's frame cap on its libp2p transport too, so a malicious
        # owner/scheduler can't push an oversized reply against the 4 GiB default.
        link = _Libp2pLink(
            Libp2pTransport(identity, max_msg_bytes=max_msg_bytes).start(), scheduler_addr)
        backoff, fails, last_done = 0.05, 0, state["done"]
        try:
            while fails < 8:   # give up only after sustained no-progress failures
                try:
                    if _serve_sharded(link, engine, worker, wid, warm, shard_cache,
                                      versions, keyframes, residuals, data_ctx, caps, state,
                                      heartbeat_interval, poll_interval, max_tasks,
                                      fault_hook, stop_event):
                        return  # clean finish (stop / budget / graceful leave)
                except (OSError, ConnectionError):
                    pass  # transient libp2p fault (e.g. a raced dial) -> retry
                if state["done"] > last_done:    # progress -> the fault was transient
                    fails, backoff, last_done = 0, 0.05, state["done"]
                else:                             # no progress -> scheduler likely gone
                    fails += 1
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 1.0)
        finally:
            link.close()
        return

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
        # One link per scheduler connection: it owns the PS connection cache, so
        # a reconnect naturally drops stale PS sockets (fresh link next loop).
        link = _WorkerLink(sch, auth_key=auth_key, max_msg_bytes=max_msg_bytes,
                           connect_timeout=connect_timeout, tls=tls)
        clean = False
        try:
            clean = _serve_sharded(link, engine, worker, wid, warm, shard_cache, versions,
                                   keyframes, residuals, data_ctx, caps, state,
                                   heartbeat_interval, poll_interval, max_tasks, fault_hook,
                                   stop_event)
        except (OSError, ConnectionError):
            clean = False  # disconnected -> reconnect (if enabled)
        finally:
            link.close()
            try:
                sch.close()
            except OSError:
                pass
        if clean or not reconnect:
            return
        if stop_event is not None and stop_event.is_set():
            return  # graceful leave during a disconnect: don't reconnect
        time.sleep(backoff)
        backoff = min(backoff * 2, 1.0)


def _serve_sharded(link, engine, worker, wid, warm, shard_cache, versions, keyframes,
                   residuals, data_ctx, caps, state, heartbeat_interval, poll_interval,
                   max_tasks, fault_hook, stop_event=None) -> bool:
    """One scheduler connection: serve tasks. Returns True on a clean finish (stop /
    budget), raises ``OSError`` on a disconnect (so the caller can reconnect). All
    peer comms go through ``link`` (the transport seam), so TCP and libp2p share
    this loop verbatim.

    ``stop_event`` (W4b graceful leave): when set, the worker stops between tasks
    and -- if it had *just* leased one -- nacks it, so the path is re-leasable
    immediately instead of after the lease timeout. A worker already mid-train
    (the expensive part, which can't be interrupted) finishes and commits that
    task, then leaves at the next loop top -- no work lost, no nack needed."""
    while True:
        if stop_event is not None and stop_event.is_set():
            return True  # graceful leave between tasks
        task = link.sch_rpc({"type": "request", "worker_id": wid,
                             "warm_paths": list(warm), "cached_shards": list(shard_cache),
                             "capabilities": caps})
        if task is None:
            raise OSError("scheduler disconnected")  # not a clean stop -> reconnect
        if task["type"] == "stop":
            return True
        if task["type"] == "idle":
            time.sleep(task.get("retry_in") or poll_interval)  # server-paced when set
            continue

        path = task["path"]
        lease = task.get("lease")
        # Graceful leave raced the lease: return it now (fenced by the token) so
        # the scheduler re-leases the path immediately, not after a timeout.
        if stop_event is not None and stop_event.is_set():
            if lease is not None and not task.get("check_only"):
                link.sch_rpc({"type": "nack", "path": path, "lease": lease, "worker_id": wid})
            return True
        worker.seed = task["seed"]
        engine.total_rounds = task["total_rounds"]
        # Routing values are replica addr lists in rank order (primary first).
        # link.addr_key picks the per-transport dial target: a hashable (host,
        # port) for TCP, the raw multiaddr / k-relay candidate *list* for libp2p
        # (a NAT owner's relays, which rpc fails over across).
        routing = {k: [link.addr_key(a) for a in addrs]
                   for k, addrs in task["routing"].items()}
        check_only = bool(task.get("check_only"))
        audit = bool(task.get("audit"))
        down_delta = task.get("down") == "delta"   # keep keyframes, send keyframe `have` (W2a)
        # Evict keyframes for keys outside this task's path (D3): bounds the
        # baseline cache to the current path's shared keys instead of every key
        # ever fetched. A dropped keyframe just costs one full re-fetch -- never
        # worse than full mode.
        for stale in [k for k in keyframes if k not in routing]:
            del keyframes[stale]
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
                k: [a for a in addrs if link.connected(a)]
                + [a for a in addrs if not link.connected(a)]
                for k, addrs in routing.items()
            }
            while pending:
                addr = next(iter(pending.values()))[0]
                batch = [k for k, cands in pending.items() if cands[0] == addr]
                # In delta mode `have` is ONLY the keyframe version (the bytes we
                # hold exactly) or None -- never the trained-against version,
                # which is a *lossy* reconstruction we can't delta-decode against.
                # No keyframe -> None -> the owner ships a full (a new keyframe).
                # An audited primary must train from the owner's EXACT current
                # bytes: its checkers pin that version and fetch it full, so a
                # lossy delta reconstruction would make honest replicas disagree.
                # Sending no `have` forces a full fetch (a fresh exact keyframe).
                def _have(k):
                    if not down_delta:
                        return versions.get(k)
                    if audit:
                        return None
                    return keyframes[k][0] if k in keyframes else None
                req = {"type": "fetch", "keys": batch, "cold": cold,
                       "have": {} if pin else
                               {k: _have(k) for k in batch if not is_private_key(k)}}
                if pin:
                    req["pin"] = {k: list(pin[k]) for k in batch if k in pin}
                try:
                    reply = link.ps_rpc(addr, req)
                    if reply is None:
                        raise OSError(f"replica {addr} closed")
                except (OSError, ConnectionError):
                    for k in batch:
                        pending[k] = pending[k][1:]
                        if not pending[k]:
                            raise OSError(f"no replica could serve {k}")
                    continue
                missing = set(reply.get("missing") or [])
                reply_versions = reply.get("versions", {})
                for k, sd in reply["weights"].items():
                    if isinstance(sd, dict) and "__delta__" in sd:
                        # Reconstruct current = keyframe + dequant(delta). The
                        # keyframe is exact and unchanged, so error is one bounded
                        # int8 step (non-chained); the keyframe is NOT advanced.
                        kf = keyframes.get(k)
                        base_v = tuple(sd.get("base") or ())
                        if kf is None or kf[0] != base_v:
                            raise OSError(f"delta for {k} vs keyframe {base_v} not held")
                        try:
                            recon = apply_state_delta(kf[1], sd["tensors"])
                        except (TypeError, ValueError, RuntimeError) as e:
                            # A malformed delta from a buggy/Byzantine owner must
                            # not crash the worker: drop the suspect keyframe (so
                            # the retry re-fetches a full) and treat it as a
                            # replica fault the reconnect/next-replica path absorbs.
                            keyframes.pop(k, None)
                            raise OSError(f"malformed delta for {k}: {e}") from e
                        _load_into(engine, k, recon)
                    else:
                        _load_into(engine, k, sd)
                        if down_delta and not is_private_key(k) and k in reply_versions:
                            keyframes[k] = (tuple(reply_versions[k]), sd)  # new keyframe (full)
                versions.update(reply_versions)
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
                                             task["batch_size"], task["gen_id"],
                                             task.get("inner_steps"))
                if not contrib.empty:
                    digest = pseudograd_digest(contrib.shared_delta)
            except _CheckAborted:
                pass  # abstain: the base aged out
            # Private proposal policy: also submit this (cold, reproduced)
            # private state to the owners; the owner applies it only once enough
            # distinct peers agree on the exact bytes (D5/3a).
            if (task.get("private_proposal") and contrib is not None
                    and contrib.private_state):
                # Group by a hashable key but dial the raw target (a libp2p
                # candidate *list* isn't hashable yet must stay a list to fail over).
                by_owner: dict = {}
                for k, sd in contrib.private_state.items():
                    if k in routing:
                        target = routing[k][0]
                        by_owner.setdefault(_addr_key(target), (target, {}))[1][k] = sd
                for addr, states in by_owner.values():
                    try:
                        link.ps_rpc(addr, {"type": "private_proposal", "private": states,
                                           "grant": task.get("grant")})
                    except (OSError, ConnectionError):
                        pass  # owner away; corroboration just waits (link drops the conn)
            engine._opt_state.pop(path, None)  # leave no warm trace of the check
            ack = link.sch_rpc({"type": "commit", "check_only": True, "path": path,
                                "worker_id": wid, "lease": lease, "digest": digest,
                                "gen_id": task["gen_id"]})
            if ack is None:
                raise OSError("scheduler disconnected during check commit")
            continue

        stop_beat = threading.Event()
        beat = threading.Thread(target=_sch_heartbeat,
                                args=(link.sch_send, stop_beat, heartbeat_interval, wid,
                                      lease, path),
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
            contrib = worker._train_path(path, shard, task["batch_size"], task["gen_id"],
                                         task.get("inner_steps"))
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
        ack = link.sch_rpc(commit)
        if ack is None:
            raise OSError("scheduler disconnected during commit")
        if ack.get("accepted"):
            grant = ack["grant"]  # carries the push weight + allowed keys to the PSs
            # Encode only after acceptance, so the error-feedback residual always
            # reflects an update that is actually pushed.
            shared_payload, private_payload, pending_res = _compress_contribution(
                contrib, task.get("compress") or "none", residuals, path,
                density=task.get("density") or 1.0,
            )
            _commit_residuals(residuals, path, pending_res)

            failed = _push_group(routing, grant, shared_payload, private_payload, link)
            if failed:
                # The epoch moved under this task: the old primary refused (or
                # died). The scheduler already accepted the commit, so re-route
                # the failed keys to the *current* primaries and re-present the
                # grant once (it is single-use per server, so a new primary
                # accepts it). A second failure drops the update -- the
                # documented bounded-loss window.
                fresh = link.sch_rpc({"type": "routing", "path": path})
                if fresh is None:
                    raise OSError("scheduler disconnected during push retry")
                fresh_routing = fresh.get("routing") or {}
                retry = {k: [link.addr_key(a) for a in fresh_routing[k]]
                         for k in failed if k in fresh_routing}
                if retry:
                    _push_group(retry, grant, shared_payload, private_payload, link)
            engine._opt_state[path] = contrib.opt_state          # warm-back
            _load_private(engine, contrib.private_state)
            warm.add(path)
            state["done"] += 1
            if max_tasks is not None and state["done"] >= max_tasks:
                return True
        # rejected -> discard the contribution; warm caches stay


def _push_group(routing, grant, shared_payload, private_payload, link) -> set:
    """Push each key's update to its primary (``routing[k][0]``) via the link.

    Returns the keys whose update did **not** land for a retryable reason:
    the server refused them as ``not_primary`` (the worker's routing is from
    an older epoch) or the primary was unreachable. Non-retryable refusals
    (bad grant, replay) return nothing -- a retry could never succeed.
    """
    by_primary: dict = {}  # group_key -> (dial_target, [keys]); writes to rank 0 only (D3)
    for k, addrs in routing.items():
        # Hashable group key, but dial the raw target -- a libp2p relay candidate
        # list must stay a list so rpc fails over across the owner's k relays.
        by_primary.setdefault(_addr_key(addrs[0]), (addrs[0], []))[1].append(k)
    failed: set = set()
    for addr, keys in by_primary.values():
        updates = {k: {"grad": shared_payload[k]}
                   for k in keys if not is_private_key(k) and k in shared_payload}
        private = {k: private_payload[k]
                   for k in keys if is_private_key(k) and k in private_payload}
        if not updates and not private:
            continue
        try:
            reply = link.ps_rpc(addr, {"type": "push", "grant": grant,
                                       "updates": updates, "private": private})
        except (OSError, ConnectionError):
            failed |= set(updates) | set(private)   # link drops the stale conn itself
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


class _WorkerLink:
    """The worker's comm seam to the scheduler + parameter servers (W1b step 2).

    A persistent request/reply channel to the scheduler (``sch_rpc``) with a
    fire-and-forget path for heartbeats (``sch_send``), plus a connection-cached
    request/reply to each parameter server (``ps_rpc``). This TCP implementation
    preserves today's behavior exactly (same sockets, same ``send_lock``-serialized
    writes, same ``_ps_connect`` cache, same drop-on-error); the libp2p
    implementation (W1b step 3) swaps the byte pipe without touching the worker
    loop, which now speaks only to this seam."""

    def __init__(self, sch_sock, *, auth_key, max_msg_bytes, connect_timeout, tls=None):
        self._sch = sch_sock
        self._lock = threading.Lock()
        self._auth = auth_key
        self._max = max_msg_bytes
        self._timeout = connect_timeout
        self._tls = tls
        self._ps: dict = {}   # addr_key -> connected socket

    def sch_send(self, msg) -> None:
        with self._lock:
            send_msg(self._sch, msg)

    def sch_rpc(self, msg):
        with self._lock:                      # serialize writes; heartbeat waits its turn
            send_msg(self._sch, msg)
            return recv_msg(self._sch, self._max)

    def addr_key(self, a):
        """Normalize a routing entry to this transport's dial target. TCP keys its
        socket cache by address, so an entry must be a hashable ``(host, port)``."""
        return _addr_key(a)

    def connected(self, addr) -> bool:
        return addr in self._ps

    def ps_rpc(self, addr, msg):
        sock = self._ps.get(addr)
        if sock is None:
            sock = _ps_connect(addr, self._auth, self._max, self._timeout,
                               tls=self._tls, server_hostname=addr[0])
            self._ps[addr] = sock
        try:
            return _rpc(sock, msg, self._max)
        except (OSError, ConnectionError):
            self._drop(addr)               # stale socket -> reconnect lazily next call
            raise

    def _drop(self, addr) -> None:
        s = self._ps.pop(addr, None)
        if s is not None:
            try:
                s.close()
            except OSError:
                pass

    def close(self) -> None:
        for s in self._ps.values():
            try:
                s.close()
            except OSError:
                pass
        self._ps.clear()


def _sch_heartbeat(sch_send, stop_beat, interval, wid, lease, path):
    while not stop_beat.wait(interval):
        try:
            sch_send({"type": "heartbeat", "lease": lease, "path": path, "worker_id": wid})
        except OSError:
            return


# -- decentralized worker loop (Phase 4 worker runtime) -----------------------
#
# The central worker leases from a scheduler; the decentralized worker has none.
# It derives the epoch locally from the gossiped/seed directory (the signer-less
# ``derive_epoch``, D2/D6), self-assigns a ``(path, generation)`` it is the HRW
# rank for (D3), quorum-fetches that path's bases from the keys' k replicas (D4),
# trains, commits to the path's coordinator -- the primary owner of its
# coordinator key, which version-fences the slot and mints the Ed25519 grant
# (D5) -- and pushes the pseudo-gradient to all k owners of each key (D6:
# whichever is the true primary applies; co-owners replicate the result). Every
# peer comm goes through ``link`` (the transport seam), so a fake in-process link
# drives this loop in tests exactly as TCP/libp2p does in a run.
# Design: docs/decentralized-worker-loop-design.md.


def _verified_peers(records) -> list:
    """Self-certifying peer records from a (possibly hostile) directory snapshot.
    A malicious tracker/owner can serve arbitrary frames, so every record is
    signature-verified before it can influence epoch derivation or HRW choice."""
    return [r for r in records if isinstance(r, dict)
            and verify_record(r) and r.get("kind") == "peer"]


def _worker_directory_ids(records) -> list[str]:
    """Peer-ids of the worker-role records in a verified directory -- the HRW
    candidate set for self-assignment (the set the owner's worker_set also sees)."""
    return [r["peer_id"] for r in records
            if "worker" in (r.get("roles") or []) and isinstance(r.get("peer_id"), str)]


def _decentralized_routing(topology, path, epoch, link) -> dict:
    """``{key: [dial target per replica, primary first]}`` for a path under an
    epoch -- the read/push fan-out targets in HRW rank order. A NAT owner with no
    dialable address (no relay yet) is skipped for that key."""
    routing = {}
    for key in topology.path_module_keys(path):
        targets = [link.addr_key(owner_addr(o)) for o in owners_for(key, epoch)
                   if owner_addr(o) is not None]
        if targets:
            routing[key] = targets
    return routing


def _pick_assigned_path(link, topology, epoch, workers, peer_id, *, salt, lease_ttl):
    """The first ``(path, generation, routing)`` this worker is the current HRW
    assignee for (rank 0, or a successor once the generation has stayed open past
    ``lease_ttl`` -- takeover-on-expiry), or ``None`` if assignee of nothing.

    Reads each path's ``(generation, age)`` from its coordinator (the primary
    owner of the path's coordinator key, ``generation`` RPC); a coordinator that
    doesn't answer (down, or not this epoch's primary) skips that path this pass.
    """
    for path in topology.paths():
        prim = path_primary(topology.path_module_keys(path), epoch)
        if prim is None or owner_addr(prim) is None:
            continue
        try:
            rep = link.ps_rpc(link.addr_key(owner_addr(prim)),
                              {"type": "generation", "path": list(path)})
        except (OSError, ConnectionError):
            continue
        if not (rep and rep.get("ok")):
            continue
        g = int(rep["generation"])
        # Use the coordinator's reported lease_ttl when present -- an explicit None
        # check, not ``or``, so a legitimate 0.0 ("no successor takeover", which
        # responsible_rank honors as rank-0-forever) isn't silently replaced by the
        # worker's default and made to disagree with the coordinator's commit gate.
        rep_ttl = rep.get("lease_ttl")
        if is_assignee(peer_id, path, g, workers, salt=salt,
                       elapsed=float(rep.get("age") or 0.0),
                       lease_ttl=float(rep_ttl if rep_ttl is not None else lease_ttl)):
            return path, g, _decentralized_routing(topology, path, epoch, link)
    return None


def _fetch_quorum_bases(engine, link, routing, read_quorum, *, cold) -> dict:
    """Quorum-read each shared key's base across its replicas, load the confirmed
    bytes into the engine bank, and return ``{key: (epoch, counter)}`` the worker
    trained against (reported at commit for version-lag staleness, D4).

    Shared keys: gather ``(version, digest)`` from every replica, ``confirm_version``
    (the highest version a ``read_quorum`` majority agree on), then download the
    weights from a replica whose digest matches -- a lone Byzantine owner serving
    poisoned bytes is in the minority, so its bytes are never loaded. Private keys
    (path-local, no quorum) are fetched from the primary, and only on a **cold**
    start (a warm worker keeps its own private state, like the central loop).
    Raises ``OSError`` if a shared key can't be confirmed or no matching replica
    serves it (the caller skips the task: replicas mid-sync, liveness over a
    forced read)."""
    shared = {k: a for k, a in routing.items() if not is_private_key(k)}
    confirmed = read_quorum_versions(
        sorted({a for addrs in shared.values() for a in addrs}, key=repr),
        list(shared), read_quorum, lambda addr, msg: link.ps_rpc(addr, msg))
    fetched: dict = {}
    for key, addrs in shared.items():
        c = confirmed.get(key)
        if c is None:
            raise OSError(f"no read quorum for {key}")
        cv, cd = c
        for addr in addrs:
            try:
                reply = link.ps_rpc(addr, {"type": "fetch", "keys": [key], "cold": cold})
            except (OSError, ConnectionError):
                continue
            sd = (reply or {}).get("weights", {}).get(key)
            if sd is None or state_digest(sd) != cd:  # absent, or not the confirmed bytes
                continue
            _load_into(engine, key, sd)
            fetched[key] = tuple(cv)
            break
        else:
            raise OSError(f"no replica served the confirmed base for {key}")
    if cold:  # private base from the primary (cold fetch ships it)
        for key, addrs in routing.items():
            if not is_private_key(key):
                continue
            try:
                reply = link.ps_rpc(addrs[0], {"type": "fetch", "keys": [key], "cold": True})
            except (OSError, ConnectionError):
                continue
            sd = (reply or {}).get("weights", {}).get(key)
            if sd is not None:
                _load_into(engine, key, sd)
                v = (reply.get("versions") or {}).get(key)
                if v is not None:
                    fetched[key] = tuple(v)
    return fetched


def _push_all_owners(routing, grant, shared_payload, private_payload, link) -> set:
    """Push the pseudo-gradient to **all k owners** of each key (D6). Whichever
    owner is the true primary applies it; co-owners refuse it as ``not_primary``
    (harmless -- they receive the result by replication), so one minted grant --
    single-use *per server* -- authorizes the push at every replica and a stale
    view of which owner is primary still lands. Returns the keys that applied at
    **no** owner (retryable: the epoch moved, so re-derive and retry once)."""
    by_owner: dict = {}  # dial target -> [keys this owner replicates]
    for key, addrs in routing.items():
        for addr in addrs:
            by_owner.setdefault(addr, []).append(key)
    pushed = (set(shared_payload) | set(private_payload)) & set(routing)
    applied: set = set()
    for addr, keys in by_owner.items():
        updates = {k: {"grad": shared_payload[k]} for k in keys
                   if not is_private_key(k) and k in shared_payload}
        private = {k: private_payload[k] for k in keys
                   if is_private_key(k) and k in private_payload}
        if not updates and not private:
            continue
        try:
            reply = link.ps_rpc(addr, {"type": "push", "grant": grant,
                                       "updates": updates, "private": private})
        except (OSError, ConnectionError):
            continue
        if reply and reply.get("applied"):
            applied |= (set(updates) | set(private)) - set(reply.get("skipped") or [])
    return pushed - applied


def _serve_decentralized(link, engine, worker, peer_id, corpus, directory_fn, *,
                         k, salt, read_quorum, lease_ttl, batch_size, total_rounds,
                         max_tasks, poll_interval, state, warm,
                         stop_event=None, fault_hook=None, max_iters=None):
    """The decentralized worker loop. Returns ``True`` on a clean finish (task
    budget / graceful leave). ``directory_fn()`` returns the current directory
    snapshot (tracker seed + owner gossip); the epoch is derived locally from it.
    ``max_iters`` bounds the loop for tests (one assigned task per iteration)."""
    topology = engine.topology
    engine.total_rounds = total_rounds
    epoch_prev = None
    backoff = poll_interval
    iters = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            return True
        if max_tasks is not None and state["done"] >= max_tasks:
            return True
        if max_iters is not None and iters >= max_iters:
            return False
        iters += 1
        records = _verified_peers(directory_fn())
        epoch = derive_epoch(records, k=k, salt=salt, prev=epoch_prev)
        epoch_prev = epoch
        workers = _worker_directory_ids(records)
        picked = (None if not epoch["owners"] or peer_id not in workers
                  else _pick_assigned_path(link, topology, epoch, workers, peer_id,
                                           salt=salt, lease_ttl=lease_ttl))
        if picked is None:
            time.sleep(backoff)
            backoff = min(backoff * 2, 1.0)
            continue
        backoff = poll_interval
        path, g, routing = picked
        cold = path not in warm
        try:
            fetched = _fetch_quorum_bases(engine, link, routing, read_quorum, cold=cold)
        except (OSError, ConnectionError):
            continue  # replicas mid-sync / unreachable -> re-scan
        if cold:
            engine._opt_state.pop(path, None)  # reset Adam on a cold start
        if fault_hook is not None:
            fault_hook(path, 1)
        contrib = worker._train_path(path, corpus.shard(topology.path_index(path)),
                                     batch_size, g)
        prim = path_primary(topology.path_module_keys(path), epoch)
        commit = {"type": "commit", "path": list(path), "generation": g,
                  "worker_id": peer_id, "loss": contrib.loss, "empty": contrib.empty,
                  "base_versions": {kk: list(v) for kk, v in fetched.items()}}
        try:
            ack = link.ps_rpc(link.addr_key(owner_addr(prim)), commit)
        except (OSError, ConnectionError):
            continue
        if not (ack and ack.get("accepted")):
            continue  # version-fenced / stale / throttled -> re-scan
        grant = ack["grant"]
        # Uncompressed pseudo-gradient: decentralized mode disallows lossy
        # compression (the quorum byte-agreement invariant; rejected at owner
        # construction), so the contribution's deltas ship as-is -- no
        # error-feedback residual to carry.
        shared_payload, private_payload = contrib.shared_delta, contrib.private_state
        failed = _push_all_owners(routing, grant, shared_payload, private_payload, link)
        if failed:
            # The epoch moved under this task: re-derive and retry the failed keys
            # once against the current owners (the grant is single-use per server,
            # so a fresh primary accepts it). A second miss drops the update -- the
            # documented bounded-loss window, same as the central retry.
            epoch2 = derive_epoch(_verified_peers(directory_fn()), k=k, salt=salt,
                                  prev=epoch_prev)
            epoch_prev = epoch2
            retry = {kk: rr for kk, rr in
                     _decentralized_routing(topology, path, epoch2, link).items()
                     if kk in failed}
            if retry:
                _push_all_owners(retry, grant, shared_payload, private_payload, link)
        engine._opt_state[path] = contrib.opt_state  # warm-back (reused if we re-pick it)
        _load_private(engine, contrib.private_state)
        warm.add(path)
        state["done"] += 1


def run_decentralized_worker(config, diloco, tracker_addr, corpus, *, identity,
                             device="cpu", seed=0, auth_key=None, k=3, salt="",
                             read_quorum=2, lease_ttl=30.0, batch_size=8,
                             total_rounds=0, max_tasks=None, reachability="nat",
                             heartbeat_interval=3.0, poll_interval=0.05,
                             max_msg_bytes=DEFAULT_MAX_MSG_BYTES, connect_timeout=10.0,
                             tls=None, stop_event=None, fault_hook=None):
    """Self-assigning worker for a decentralized swarm (``schedule.mode:
    decentralized``): no scheduler, no central grant signer. It registers with
    the rendezvous tracker (role ``worker``), then loops :func:`_serve_decentralized`
    -- derive the epoch locally, self-assign a path, quorum-fetch, train, commit
    to the path's coordinator, push to all k owners. Needs an ``identity`` (it is
    HRW-scored by ``peer_id`` and derives epochs)."""
    from .tracker import fetch_directory, register_peer

    engine = _build_worker_engine(config, diloco, device, seed)
    worker = AsyncScheduler(engine, num_workers=1)
    worker.seed = seed  # run-level constant: (path, generation) -> identical compute
    state = {"done": 0}
    warm: set = set()

    def _register():
        register_peer(tracker_addr, identity, reachability=reachability,
                      roles=("worker",), auth_key=auth_key, tls=tls)

    stop_beat = threading.Event()

    def _beat():
        # Register once up front, then refresh the TTL. Tolerate a tracker that is
        # briefly unreachable at launch (coordinated bring-up / tracker failover):
        # until it answers, the worker just isn't in the directory yet, so it
        # self-assigns nothing -- it must not crash the worker, which would forfeit
        # the steady-state resilience the loop otherwise has.
        first = True
        while first or not stop_beat.wait(heartbeat_interval):
            first = False
            try:
                _register()
            except (OSError, ConnectionError):
                pass
    beat = threading.Thread(target=_beat, daemon=True)
    beat.start()

    def directory_fn():
        try:
            return fetch_directory(tracker_addr, auth_key=auth_key, tls=tls)
        except (OSError, ConnectionError):
            return []  # tracker blip: a previous epoch persists until it answers

    link = _WorkerLink(None, auth_key=auth_key, max_msg_bytes=max_msg_bytes,
                       connect_timeout=connect_timeout, tls=tls)
    try:
        _serve_decentralized(
            link, engine, worker, identity.peer_id, corpus, directory_fn,
            k=k, salt=salt, read_quorum=read_quorum, lease_ttl=lease_ttl,
            batch_size=batch_size, total_rounds=total_rounds, max_tasks=max_tasks,
            poll_interval=poll_interval, state=state, warm=warm,
            stop_event=stop_event, fault_hook=fault_hook)
    finally:
        stop_beat.set()
        beat.join(timeout=1)
        link.close()
    return state["done"]
