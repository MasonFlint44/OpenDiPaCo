"""Multi-node transport for the async scheduler -- plain TCP, no extra deps.

The in-process :class:`AsyncScheduler` runs workers as threads sharing the bank.
This module lets those workers live in **other processes / on other machines**,
with **stateful workers** so only the irreducible DiLoCo traffic crosses the wire:

* :class:`CoordinatorServer` wraps an ``AsyncScheduler``. It holds the
  authoritative bank + outer optimizer, the per-generation task queue
  (``_Generation``), a per-shared-module **version**, and a path -> worker
  **owner** map for data locality. It **streams** the reduce -- folding each
  submitted pseudo-gradient into a running accumulator and dropping it, instead
  of buffering every path's delta.
* :func:`run_worker` is the client loop. Workers keep their path's **private
  modules, inner-optimizer (Adam) state, and data shard warm across
  generations**. Each task therefore ships only what the worker lacks: the
  updated **shared** weights (which changed via the outer step), plus private
  modules / the shard **only the first time** a worker handles a path. Workers
  ship back the shared pseudo-gradient and their (small) private modules.

Locality: the coordinator prefers re-assigning a path to the worker that owns it
warm, so optimizer/private/shard state stays cached. Fault tolerance is unchanged
-- a dead worker drops its socket, its lease times out, the path is re-queued and
**fails over cold** to another worker (its Adam state resets; the coordinator
ships it the current private modules). New workers connect any time; a worker
whose connection blips **reconnects** and resumes (its warm caches survive).

Wire format: the pickle-free framed codec in ``wire.py`` (8-byte length prefix,
JSON structure + raw tensor blobs decoded against a dtype allowlist -- nothing on
the wire is unpickled). Every lease carries a unique token that the worker must
echo on submit/nack/heartbeat; a submission whose token doesn't match the current
lease is dropped, so a reclaimed-and-re-leased path can't be hijacked by a zombie
worker (and the fresher result silently discarded).
"""

from __future__ import annotations

import math
import os
import socket
import ssl
import threading
import time
import uuid
from collections import OrderedDict

import torch

from ..backend.local import LocalBackend
from ..checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from ..data.spec import materialize_shard, spec_fingerprint, verify_routing
from ..optim.diloco import apply_outer_grads, make_outer_optimizer
from ..topology import is_private_key
from ..train.loop import _optimizer_state_to_cpu, _state_to_cpu
from .compress import (
    check_mode,
    compress_delta,
    compress_shard,
    compress_state,
    maybe_dequantize,
    restore_shard,
)
from .guard import clip_norm_, contribution_ok
from .reactor import TransportMetrics, _ReactorServer  # noqa: F401 (re-exported)
from .scheduler import AsyncScheduler
from .wire import DEFAULT_MAX_MSG_BYTES, client_handshake, recv_msg, send_msg



# -- coordinator -------------------------------------------------------------


class CoordinatorServer(_ReactorServer):
    """Single-node async coordinator: holds the whole bank and serves tasks.

    Construct it around an ``AsyncScheduler`` (which owns the engine/bank), then
    :meth:`start` the server and call :meth:`fit` to drive the run. Worker
    processes call :func:`run_worker` pointing at ``(host, port)``. For models too
    big for one node, see the sharded :class:`~opendipaco.schedule.Scheduler` +
    :class:`~opendipaco.schedule.ParameterServer` instead.
    """

    def __init__(
        self,
        scheduler: AsyncScheduler,
        corpus,
        batch_size: int,
        *,
        host: str = "0.0.0.0",
        port: int = 0,
        heartbeat_timeout: float | None = None,
        auth_key: str | bytes | None = None,
        accept_keys=None,
        admitted_peers=None,
        max_msg_bytes: int = DEFAULT_MAX_MSG_BYTES,
        staleness_bound: int | None = None,
        staleness_weight: str = "inverse",
        rescale_by_sqrt_sharing: bool = False,
        max_update_norm: float | None = None,
        compress: str = "none",
        idle_backoff: float | None = None,
        io_threads: int = 4,
        max_connections: int = 1024,
        tls=None,
    ):
        super().__init__(host=host, port=port, auth_key=auth_key, accept_keys=accept_keys,
                         admitted_peers=admitted_peers, max_msg_bytes=max_msg_bytes,
                         io_threads=io_threads, max_connections=max_connections, tls=tls)
        self.sched = scheduler
        self.engine = scheduler.engine
        self.corpus = corpus
        self.batch_size = batch_size
        # Async (bounded-staleness): a pseudo-gradient computed against weights that
        # have since advanced by > ``staleness_bound`` updates is rejected; else it
        # is applied with weight 1/(1+staleness) ("inverse") or 1.0 ("none"). The
        # √P rescale is the sync-only aggregation correction (off by default here).
        n_paths = len(self.engine.topology.paths())
        self.staleness_bound = staleness_bound if staleness_bound is not None else 2 * n_paths
        self.staleness_weight = staleness_weight
        self.rescale_by_sqrt_sharing = rescale_by_sqrt_sharing
        # Guard rails on what a worker may apply to the bank: non-finite
        # contributions are always rejected; a norm cap (off by default) clips
        # oversized pseudo-gradients per module (see ``guard.py``).
        self.max_update_norm = max_update_norm
        # Wire compression policy; stamped on every task so workers follow it
        # (see ``compress.py``). Receivers are self-describing either way.
        self.compress = check_mode(compress)
        # Server-driven idle pacing: when set, idle replies carry ``retry_in``
        # and workers wait that long before polling again (instead of their own
        # tight ``poll_interval``) -- a swarm of surplus workers stops hammering.
        self.idle_backoff = idle_backoff
        self._worker_caps: dict = {}  # worker_id -> advertised capability profile
        self.heartbeat_timeout = (
            heartbeat_timeout if heartbeat_timeout is not None else scheduler.lease_timeout
        )

        self._lock = threading.Lock()       # guards scheduling state + the bank
        self._serving = False               # whether the drive loop is handing out tasks
        self._versions: dict[str, int] = {
            k: 0 for k in self.engine.topology.module_keys() if not is_private_key(k)
        }
        self._owner: dict = {}
        self._T = 0                         # accepted outer updates so far (the async clock)
        self._target = 0
        self._completed: dict = {}
        self._inflight: dict = {}           # path -> heartbeat deadline (one lease per path)
        self._issued: dict = {}             # path -> _T at lease time (for staleness)
        self._lease: dict = {}              # path -> current lease token (fences submits)
        self._outer_opts: dict = {}         # shared key -> per-module SGD+Nesterov

    def _handle(self, msg: dict, nbytes: int, peer_id: str | None = None):
        kind = msg.get("type")
        if kind == "request":
            return self._next_task(msg)
        if kind == "submit":
            # ``applied`` tells the worker whether the update reached the bank,
            # so it can commit its error-feedback residual only for updates that
            # actually landed (a rejected/stale submit must not leak into the
            # next generation's carry).
            return {"type": "ack", "applied": self._receive(msg)}
        if kind == "nack":
            self._nack(msg)
        elif kind == "heartbeat":
            self._heartbeat(msg)
        return None

    def shutdown(self) -> None:
        with self._lock:
            self._serving = False
        super().shutdown()

    def simulate_crash(self) -> None:
        with self._lock:
            self._serving = False
        super().simulate_crash()

    # -- async task lease / collect ------------------------------------------
    def _idle(self) -> dict:
        msg = {"type": "idle"}
        if self.idle_backoff is not None:
            msg["retry_in"] = self.idle_backoff
        return msg

    def _task_batch_size(self, caps: dict) -> int:
        """The task's batch size, capped by the worker's advertised maximum.

        Lets a memory-constrained volunteer train smaller batches instead of
        OOM-crash-looping. Note this is a *semantic* knob: a capped worker takes
        the same number of inner steps on smaller batches, so its paths see
        different batch statistics (acceptable in async mode, opt-in).
        """
        cap = (caps or {}).get("max_batch")
        return max(1, min(self.batch_size, int(cap))) if cap else self.batch_size

    def _next_task(self, req: dict) -> dict:
        wid = req.get("worker_id")
        warm = {tuple(p) for p in req.get("warm_paths", [])}
        cached = {tuple(p) for p in req.get("cached_shards", [])}
        have_ver = req.get("have_shared", {})
        caps = req.get("capabilities") or {}
        e = self.engine

        with self._lock:  # one lock guards both scheduling state and the bank
            if caps:
                self._worker_caps[wid] = caps
            if not self._serving or self._T >= self._target:
                return {"type": "stop"} if self._stop else self._idle()
            self._reclaim_inflight_locked()
            # Eligible = paths not already in flight. Pick the *least-completed* one
            # (balances progress, bounds staleness); tie-break toward a path this
            # worker holds warm (data locality).
            eligible = [p for p in self._completed if p not in self._inflight]
            if not eligible:
                return self._idle()
            path = min(eligible, key=lambda p: (self._completed[p], p not in warm, p))
            lease = uuid.uuid4().hex  # unique per lease; fences submit/nack/heartbeat
            self._owner[path] = wid
            self._inflight[path] = time.monotonic() + self.heartbeat_timeout
            self._issued[path] = self._T
            self._lease[path] = lease
            generation = self._completed[path]
            versions = dict(self._versions)
            shared_keys = [k for k in e.topology.path_module_keys(path) if not is_private_key(k)]
            private_keys = [k for k in e.topology.path_module_keys(path) if is_private_key(k)]
            # Consistent read of bank weights (the lock also serializes outer steps).
            shared_weights = {
                k: compress_state(_state_to_cpu(e.bank[k].state_dict()), self.compress)
                for k in shared_keys if have_ver.get(k) != versions[k]
            }
            private_weights = (
                {k: compress_state(_state_to_cpu(e.bank[k].state_dict()), self.compress)
                 for k in private_keys}
                if path not in warm else None
            )
        # Data plane: ship the shard bytes -- or, for a spec corpus, just the
        # recipe (the worker materializes the shard locally; see data/spec.py).
        shard, shard_spec = None, None
        if path not in cached:
            if hasattr(self.corpus, "spec"):
                shard_spec = {"path_index": e.topology.path_index(path),
                              "spec": self.corpus.spec}
            else:
                shard = self.corpus.shard(e.topology.path_index(path))
        return {
            "type": "task",
            "gen_id": generation,            # this path's update count (informational)
            "lease": lease,                  # echoed on submit/nack/heartbeat; fences the lease
            "generation": generation,        # this path's update count -> inner LR schedule
            "path": path,
            "compress": self.compress,       # uplink encoding the worker should use
            "shared_weights": shared_weights,
            "shared_versions": {k: versions[k] for k in shared_keys},
            "private_weights": private_weights,
            "shard": compress_shard(shard, self.compress),
            "shard_spec": shard_spec,
            "batch_size": self._task_batch_size(caps),
            "total_rounds": e.total_rounds,
            "seed": self.sched.seed,
        }

    def _receive(self, msg: dict) -> bool:
        """Apply one submitted contribution; return True iff it reached the bank."""
        path = msg["path"]
        e = self.engine
        # Decode + validate outside the lock (O(params) CPU work must not stall
        # task serving). One applied NaN poisons the bank permanently (faulty
        # consumer hardware produces them; see guard.py). Keys are filtered to
        # what they claim to be -- an unknown key must not crash the server, and
        # a "private" payload must not be able to overwrite a shared module.
        try:
            shared = {k: maybe_dequantize(v)
                      for k, v in (msg.get("shared_grad") or {}).items()
                      if k in self._versions}
        except (TypeError, KeyError, ValueError):
            shared = None  # malformed encoding -> invalid contribution
        private = {k: v for k, v in (msg.get("private_weights") or {}).items()
                   if is_private_key(k) and k in e.bank}
        valid = shared is not None and contribution_ok(
            msg.get("loss"), msg.get("empty"), shared, private
        )
        with self._lock:  # serialize with scheduling + other outer steps
            if path not in self._inflight or msg.get("lease") != self._lease.get(path):
                return False  # stale / duplicate / reclaimed / not the lease holder
            staleness = self._T - self._issued.get(path, self._T)
            self._inflight.pop(path, None)
            self._lease.pop(path, None)
            if staleness > self.staleness_bound:
                self.metrics.record_stale_reject()  # too stale -> discard, re-eligible
                return False
            if not valid:
                self.metrics.record_invalid_reject()  # dropped; the path is re-eligible
                return False
            # Restrict to the submitting path's own modules (path is known-valid
            # past the fence): a lease on path A must not be able to write
            # another path's private modules or unrelated shared ones. (The
            # sharded path gets the same property from the grant's key list.)
            path_keys = set(e.topology.path_module_keys(path))
            shared = {k: v for k, v in shared.items() if k in path_keys}
            private = {k: v for k, v in private.items() if k in path_keys}
            if self.max_update_norm is not None:
                for delta in shared.values():
                    if clip_norm_(delta, self.max_update_norm) > self.max_update_norm:
                        self.metrics.record_norm_clip()
            self._T += 1
            self._completed[path] = self._completed.get(path, 0) + 1
            # Apply the (damped) per-contribution outer step for this path.
            damp = 1.0 / (1.0 + staleness) if self.staleness_weight == "inverse" else 1.0
            for key, sd in private.items():
                e.bank[key].load_state_dict({n: v.to(e.device) for n, v in sd.items()})
            path_idx = e.topology.path_index(path)
            for key, delta in shared.items():
                w = e._outer_weight(key, path_idx, self.corpus) * damp
                if self.rescale_by_sqrt_sharing:
                    w *= math.sqrt(e.topology.sharing_count(key))
                apply_outer_grads(e.bank[key], [w * d.to(e.device) for d in delta])
                self._outer_opts[key].step()
                self._outer_opts[key].zero_grad(set_to_none=True)
                self._versions[key] += 1
            self.metrics.record_update(staleness)
            return True

    def _nack(self, msg: dict) -> None:
        path = msg["path"]
        with self._lock:
            # Require a *live* matching lease: a stale nack must not free someone
            # else's lease, and a nack for a never-leased path must not be able
            # to grow ``errors`` with arbitrary keys.
            if path not in self._inflight or msg.get("lease") != self._lease.get(path):
                return
            self.sched.errors[path] = msg.get("error", "worker nack")
            self._inflight.pop(path, None)  # free the lease; the path becomes re-eligible
            self._lease.pop(path, None)

    def _heartbeat(self, msg: dict) -> None:
        path = msg["path"]
        with self._lock:
            if path in self._inflight and msg.get("lease") == self._lease.get(path):
                self._inflight[path] = time.monotonic() + self.heartbeat_timeout

    def _reclaim_inflight_locked(self) -> None:
        """Free in-flight leases whose heartbeat deadline passed (dead workers)."""
        now = time.monotonic()
        for path, deadline in list(self._inflight.items()):
            if now >= deadline:
                del self._inflight[path]
                self._lease.pop(path, None)  # invalidate the token: zombies can't submit
                self._owner[path] = None
                self.metrics.reclaims += 1

    # -- drive (bounded-staleness async) -------------------------------------
    def fit(
        self,
        num_generations: int,
        *,
        total_generations: int | None = None,
        log_every: int = 0,
        reclaim_interval: float = 0.05,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 0,
        resume: bool = False,
    ):
        """Run the fleet asynchronously until each path has had ~``num_generations``
        updates (``target = num_generations * num_paths`` accepted outer steps).

        Workers run ahead out of lockstep; a submission computed against weights
        that have since advanced by more than ``staleness_bound`` is rejected, the
        rest applied with inverse-staleness damping. A slow/dead worker never
        blocks the target. ``checkpoint_dir``/``checkpoint_every`` (in *updates*)
        and ``resume`` work as before; the bank is always current.
        """
        e = self.engine
        restore = None
        if resume and checkpoint_dir and latest_checkpoint(checkpoint_dir):
            restore = load_checkpoint(e, latest_checkpoint(checkpoint_dir)).get("extra")

        if total_generations is not None:
            e.total_rounds = total_generations
        elif e.total_rounds is None:
            e.total_rounds = num_generations

        paths = list(e.topology.paths())
        self.sched._last_batch_size = self.batch_size
        # Per-shared-module outer optimizers so a single path's update moves only the
        # modules it touched (a shared optimizer would drift untouched modules).
        self._outer_opts = {
            k: make_outer_optimizer({k: e.bank[k]}, e.diloco)
            for k in self._versions
        }
        # Restore the coordinator's own state alongside the engine's: the async
        # clock and per-path counts (so the inner LR schedule continues rather
        # than restarting at generation 0), the shared-module versions (so warm
        # workers' caches stay meaningful), and the per-key outer Nesterov
        # momentum (``engine.outer_opt`` is unused on the async path).
        if restore is not None:
            with self._lock:
                self._T = restore["T"]
                self._completed = dict(restore["completed"])
                self._versions.update(restore["versions"])
            for k, sd in restore["outer_opts"].items():
                if k in self._outer_opts:
                    self._outer_opts[k].load_state_dict(sd)
        with self._lock:
            self._completed = {p: self._completed.get(p, 0) for p in paths}
            self._inflight = {}
            self._issued = {}
            self._target = self._T + num_generations * len(paths)
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
                last_ckpt = self._save_cluster_checkpoint(checkpoint_dir)
                if log_every:
                    print(f"[async] T={last_ckpt}/{self._target} "
                          f"{self.metrics.report().splitlines()[0]}", flush=True)
            time.sleep(reclaim_interval)

        with self._lock:
            self._serving = False
        self.metrics._wall += time.monotonic() - t0
        if checkpoint_dir and checkpoint_every:
            self._save_cluster_checkpoint(checkpoint_dir)
        return dict(self._completed)

    def _save_cluster_checkpoint(self, checkpoint_dir: str) -> int:
        """Checkpoint the engine plus the coordinator's async state, consistently.

        Held under ``_lock`` so the snapshot can't interleave with a concurrent
        ``_receive`` outer step (serving pauses for the duration of the write).
        Returns the clock value the checkpoint captured.
        """
        with self._lock:
            extra = {
                "T": self._T,
                "completed": dict(self._completed),
                "versions": dict(self._versions),
                "outer_opts": {
                    k: _optimizer_state_to_cpu(o.state_dict())
                    for k, o in self._outer_opts.items()
                },
            }
            save_checkpoint(
                self.engine, os.path.join(checkpoint_dir, f"upd{self._T:08d}"), extra=extra
            )
            return self._T


# -- worker ------------------------------------------------------------------


def run_worker(
    config,
    diloco,
    host: str,
    port: int,
    *,
    device: str = "cpu",
    seed: int = 0,
    poll_interval: float = 0.02,
    connect_timeout: float = 10.0,
    reconnect: bool = True,
    reconnect_timeout: float = 30.0,
    heartbeat_interval: float = 3.0,
    auth_key: str | bytes | None = None,
    max_msg_bytes: int = DEFAULT_MAX_MSG_BYTES,
    max_tasks: int | None = None,
    fault_hook=None,
    tls=None,
    tls_hostname: str | None = None,
    data_dir: str | None = None,
    data_source=None,
    data_tokenizer=None,
    max_batch_size: int | None = None,
    max_shards: int = 4,
    verify_routing: bool = False,
):
    """Connect to a coordinator and train leased path-tasks until told to stop.

    The worker is **stateful**: it keeps each path's private modules, Adam state,
    and data shard warm across generations, so a task only ships the updated
    shared weights (and private/shard the first time). After training it warms
    those back and submits the shared pseudo-gradient + its private modules.

    While a task is in progress a background thread sends **heartbeats** every
    ``heartbeat_interval`` seconds, so the coordinator can use a short
    liveness/lease timeout (fast dead-worker detection) without reclaiming a
    *slow but alive* task. Keep ``heartbeat_interval`` a few times below the
    coordinator's ``heartbeat_timeout``.

    ``fault_hook(path, attempt_count)`` may raise to simulate a flaky worker.
    ``max_tasks`` makes the worker leave cleanly after that many tasks (elastic
    membership). ``reconnect`` retries a dropped connection (e.g. a coordinator
    restart) for up to ``reconnect_timeout``; warm caches survive reconnects.

    When the coordinator serves a *spec* corpus, tasks carry a shard recipe
    instead of bytes and the worker materializes its shard locally (see
    ``data/spec.py``): ``data_dir`` keeps materialized shards on disk across
    restarts; ``data_source`` / ``data_tokenizer`` override the spec's stream
    and tokenizer (tests, non-C4 corpora).

    ``max_batch_size`` advertises a memory cap: the coordinator ships this
    worker tasks with ``batch_size`` clamped to it, so a small-VRAM volunteer
    trains smaller batches instead of OOM-crash-looping (a semantic knob --
    that worker's paths see different batch statistics). Idle replies may carry
    a server-chosen ``retry_in``, which overrides ``poll_interval``.
    """
    engine = _build_worker_engine(config, diloco, device, seed)
    worker = AsyncScheduler(engine, num_workers=1)
    wid = uuid.uuid4().hex
    warm: set = set()        # paths whose private modules + Adam state are held warm
    shard_cache = _ShardCache(max_shards)   # bounded LRU of materialized shards (W7a)
    versions: dict = {}      # shared key -> version currently held
    attempts: dict = {}
    residuals: dict = {}     # path -> {key: [tensors]}: compression error feedback
    data_ctx = {"dir": data_dir, "source": data_source, "tokenizer": data_tokenizer,
                "verify": verify_routing}
    caps = {"device": str(device)}
    if max_batch_size is not None:
        caps["max_batch"] = int(max_batch_size)
    state = {"done": 0}

    first = True
    backoff = 0.05
    while True:
        try:
            conn = _connect(host, port, connect_timeout if first else reconnect_timeout,
                            tls=tls, server_hostname=tls_hostname)
        except ConnectionError:
            return  # coordinator unreachable; give up
        first = False
        if not client_handshake(conn, auth_key):
            conn.close()
            raise PermissionError("coordinator rejected auth (wrong or missing key)")
        try:
            done = _serve_connection(
                conn, engine, worker, wid, warm, shard_cache, versions, attempts,
                residuals, data_ctx, caps, state, poll_interval, heartbeat_interval,
                max_msg_bytes, max_tasks, fault_hook,
            )
        finally:
            try:
                conn.close()
            except OSError:
                pass
        if done or not reconnect:
            return  # clean stop / budget reached -- not a disconnect
        time.sleep(backoff)  # exponential backoff between reconnect attempts
        backoff = min(backoff * 2, 1.0)


def _serve_connection(conn, engine, worker, wid, warm, shard_cache, versions,
                      attempts, residuals, data_ctx, caps, state, poll_interval,
                      heartbeat_interval, max_msg_bytes, max_tasks, fault_hook) -> bool:
    """Serve tasks on one connection. Returns True on a clean finish (stop /
    budget reached), False on a disconnect (caller may reconnect)."""
    send_lock = threading.Lock()  # heartbeat thread + main thread share the socket

    def safe_send(m) -> None:
        with send_lock:
            send_msg(conn, m)

    while True:
        try:
            safe_send({
                "type": "request", "worker_id": wid,
                "have_shared": versions, "warm_paths": list(warm),
                "cached_shards": list(shard_cache),
                "capabilities": caps,
            })
            msg = recv_msg(conn, max_msg_bytes)
        except (OSError, ValueError):
            return False
        if msg is None:
            return False  # disconnected
        if msg["type"] == "stop":
            return True
        if msg["type"] == "idle":
            time.sleep(msg.get("retry_in") or poll_interval)  # server-paced when set
            continue

        path = msg["path"]
        lease = msg.get("lease")
        worker.seed = msg["seed"]
        engine.total_rounds = msg["total_rounds"]
        # Heartbeat this lease while the (possibly long) task runs.
        stop_beat = threading.Event()
        beat = threading.Thread(
            target=_heartbeat_loop,
            args=(safe_send, stop_beat, heartbeat_interval, wid, lease, path),
            daemon=True,
        )
        beat.start()
        try:
            if fault_hook is not None:
                attempts[path] = attempts.get(path, 0) + 1
                fault_hook(path, attempts[path])
            shard = _apply_task(engine, msg, shard_cache, versions, warm, residuals,
                                data_ctx)
            contrib = worker._train_path(path, shard, msg["batch_size"], msg["generation"])
            # Warm-back: keep this path's Adam state + private modules for next gen.
            engine._opt_state[path] = contrib.opt_state
            _load_private(engine, contrib.private_state)
            warm.add(path)
        except Exception as e:
            stop_beat.set()
            beat.join(timeout=1)
            try:
                safe_send({
                    "type": "nack", "gen_id": msg["gen_id"], "lease": lease,
                    "path": path, "error": repr(e),
                })
            except OSError:
                return False
            continue
        stop_beat.set()
        beat.join(timeout=1)
        mode = msg.get("compress") or "none"
        shared_payload, private_payload, pending_res = _compress_contribution(
            contrib, mode, residuals, path
        )
        try:
            safe_send({
                "type": "submit", "gen_id": msg["gen_id"], "lease": lease, "path": path,
                "loss": contrib.loss, "empty": contrib.empty,
                "shared_grad": shared_payload,
                "private_weights": private_payload,
            })
            ack = recv_msg(conn, max_msg_bytes)
        except OSError:
            return False
        # Acceptance-aware error feedback: adopt the residual only if the
        # coordinator applied the update (rejected/stale/dropped submits keep
        # the previous carry, like the sharded worker's encode-after-commit).
        if ack and ack.get("applied"):
            _commit_residuals(residuals, path, pending_res)
        state["done"] += 1
        if max_tasks is not None and state["done"] >= max_tasks:
            return True


def _heartbeat_loop(safe_send, stop_beat, interval, wid, lease, path) -> None:
    """Ping the coordinator until the task ends or the socket dies."""
    while not stop_beat.wait(interval):
        try:
            safe_send({"type": "heartbeat", "lease": lease, "path": path, "worker_id": wid})
        except OSError:
            return


def _build_worker_engine(config, diloco, device, seed):
    from ..train.loop import DiPaCoEngine

    return DiPaCoEngine(
        config, diloco, LocalBackend(config.build_topology()),
        device=device, seed=seed, materialize="serial",
    )


def _compress_contribution(contrib, mode, residuals, path, density=1.0):
    """Encode a contribution's uplink payloads per the server's compress policy.

    In "bf16"/"int8" mode (or with ``density`` < 1.0 sparsification, W2b) the
    per-(path, module) residual is the error-feedback carry — quantization error
    and/or the dropped top-k mass: it is folded into this delta before encoding,
    and the
    *new* residual is returned as ``pending`` — **not** written back. The caller
    must commit it (:func:`_commit_residuals`) only once the server confirms the
    update was applied; committing for a rejected/stale/lost submit would both
    discard the previous legitimate carry and leak the dead update's rounding
    error into a later accepted one. Returns
    ``(shared_payload, private_payload, pending_residuals | None)``.
    """
    if mode == "none" and density >= 1.0:
        return contrib.shared_delta, contrib.private_state, None
    carry = residuals.get(path) or {}
    shared_payload, pending = {}, {}
    for key, delta in contrib.shared_delta.items():
        payload, res = compress_delta(delta, mode, carry=carry.get(key), density=density)
        pending[key] = res
        shared_payload[key] = payload
    private_payload = {k: compress_state(sd, mode) for k, sd in contrib.private_state.items()}
    return shared_payload, private_payload, pending


def _commit_residuals(residuals, path, pending) -> None:
    """Adopt a contribution's error-feedback residuals after the server applied it."""
    if pending:  # per-key update: an empty contribution must not wipe the carry
        residuals.setdefault(path, {}).update(pending)


class _ShardCache:
    """Bounded LRU of materialized shards (``path -> tensor``), W7a.

    A long-lived worker that fails over / is re-assigned across many paths would
    otherwise keep *every* shard it ever leased resident (the plain dict it
    replaces never evicts). This caps how many it holds; the least-recently-used
    shard is dropped on overflow and re-built on its next lease -- from the spec
    (cheap when the on-disk cache is warm) or from the shipped bytes. Training is
    **byte-identical**: eviction changes only *when* a shard is rebuilt, never its
    contents. The common one-path worker holds a single entry and never evicts.

    Supports exactly the dict operations the worker loops use: ``path in cache``,
    ``cache[path]`` (get marks most-recently-used), ``cache[path] = shard``, and
    ``list(cache)`` for the ``cached_shards`` advertisement.
    """

    def __init__(self, maxsize: int | None = 4):
        maxsize = 4 if maxsize is None else int(maxsize)   # None -> library default
        if maxsize < 1:
            raise ValueError(f"shard cache maxsize must be >= 1, got {maxsize}")
        self._max = maxsize
        self._d: OrderedDict = OrderedDict()

    def __contains__(self, path) -> bool:
        return path in self._d

    def __getitem__(self, path):
        self._d.move_to_end(path)          # touched -> most-recently-used
        return self._d[path]

    def __setitem__(self, path, shard) -> None:
        self._d[path] = shard
        self._d.move_to_end(path)
        while len(self._d) > self._max:
            self._d.popitem(last=False)    # evict least-recently-used

    def __iter__(self):
        # The worker drives the cache from a single serve thread (get/set and the
        # `list(cache)` advertisement never overlap), so iteration is already safe;
        # the cheap tuple() snapshot just keeps it that way if a future caller ever
        # iterates while a get reorders the mapping. It is not a substitute for a
        # lock -- a second *writer* thread would need real synchronization.
        return iter(tuple(self._d))

    def __len__(self) -> int:
        return len(self._d)


def _apply_task(engine, msg, shard_cache, versions, warm, residuals,
                data_ctx=None) -> torch.Tensor:
    """Load a task's shipped (delta) state into the worker bank; return the shard."""
    path = msg["path"]
    for key, sd in msg.get("shared_weights", {}).items():
        _load_into(engine, key, sd)  # load_state_dict casts bf16 -> module dtype
    versions.update(msg.get("shared_versions", {}))
    if msg.get("private_weights"):  # cold: coordinator shipped current private modules
        _load_private(engine, msg["private_weights"])
    if path not in warm:
        engine._opt_state.pop(path, None)  # cold -> reset Adam (reset-on-failover)
        residuals.pop(path, None)          # and drop any stale error-feedback carry
    if msg.get("shard") is not None:
        shard_cache[path] = restore_shard(msg["shard"])
    elif msg.get("shard_spec") is not None and path not in shard_cache:
        shard_cache[path] = _materialize_from_spec(msg["shard_spec"], data_ctx)
    return shard_cache[path]


class RoutingVerificationError(RuntimeError):
    """Raised when ``--verify-routing`` is on and a shipped router does not
    reproduce from its stated public source -- the worker refuses to train (W7c).
    Fatal by design: the worker loops treat it as a hard stop, not a reconnect."""


def _materialize_from_spec(shard_spec, data_ctx) -> torch.Tensor:
    """Build the shard locally from the shipped recipe (no corpus bytes on the wire).

    With ``data_ctx['verify']`` (``--verify-routing``, W7c) the shipped router is
    re-fit from the public source and checked against the shipped centroids the
    first time each spec is seen (memoized by fingerprint, so re-materialization
    after an LRU eviction doesn't re-stream). A mismatch is fatal; a spec without
    reproducible fit metadata can't be verified, so it warns once and proceeds."""
    ctx = data_ctx or {}
    spec = shard_spec["spec"]
    if ctx.get("verify"):
        fp = spec_fingerprint(spec)
        seen = ctx.setdefault("verified", set())
        if fp not in seen:
            try:
                ok = verify_routing(spec, source=ctx.get("source"),
                                    tokenizer=ctx.get("tokenizer"))
            except ValueError as e:
                print(f"WARNING: --verify-routing requested but cannot verify spec "
                      f"{fp}: {e}. Proceeding unverified.", flush=True)
                ok = True
            if not ok:
                raise RoutingVerificationError(
                    f"shipped router for spec {fp} does not reproduce from its public "
                    f"source -- refusing to train (tampered or wrong-corpus routing)")
            seen.add(fp)
    return materialize_shard(
        spec, shard_spec["path_index"],
        source=ctx.get("source"), tokenizer=ctx.get("tokenizer"),
        cache_dir=ctx.get("dir"),
    )


def _load_private(engine, private_state) -> None:
    for key, sd in (private_state or {}).items():
        _load_into(engine, key, sd)


def _load_into(engine, key, sd) -> None:
    engine.bank[key].load_state_dict({n: v.to(engine.device) for n, v in sd.items()})


def _connect(host: str, port: int, timeout: float, *, tls=None,
             server_hostname: str | None = None) -> socket.socket:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if tls is not None:
                s = tls.wrap_socket(s, server_hostname=server_hostname or host)
            return s
        except ssl.SSLError:  # handshake/cert failure is fatal -- don't retry-to-timeout
            s.close()
            raise
        except OSError as e:  # coordinator not up yet
            last = e
            time.sleep(0.05)
    raise ConnectionError(f"could not connect to coordinator at {host}:{port}: {last}")
