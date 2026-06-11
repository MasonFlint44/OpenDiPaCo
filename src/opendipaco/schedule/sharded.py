"""Sharded coordinator: a light Scheduler + K ParameterServers (gap #11).

The single-node :class:`~opendipaco.schedule.CoordinatorServer` holds the whole
bank. For models too big for one node, this module splits the bank across **K
ParameterServers** (each owns a disjoint shard of module keys: weights +
per-module outer optimizers + versions) coordinated by one light **Scheduler**
(task queue + async clock + staleness; *no weights*). A worker leases a path from
the scheduler, **fetches** that path's modules from the ParameterServers that own
them, trains, **commits** to the scheduler (which accepts/rejects on staleness and
returns a damped weight), and **pushes** the pseudo-gradients to the owning
ParameterServers. Model memory *and* weight bandwidth are sharded; the scheduler
stays light.

This reuses the reactor (`reactor.py`), wire/auth (`wire.py`), and the worker's
warm-cache + ``AsyncScheduler._train_path`` machinery. It is scale-only and
unvalidated at toy size (same async-dynamics caveat as the single coordinator).
"""

from __future__ import annotations

import math
import os
import hashlib
import ssl
import threading
import time
import uuid

import torch

from ..model import build_module_bank
from ..optim.diloco import apply_outer_grads, make_outer_optimizer
from ..topology import is_private_key
from ..train.loop import _state_to_cpu
from .distributed import _build_worker_engine, _load_into, _load_private
from .reactor import DEFAULT_MAX_MSG_BYTES, _ReactorServer
from .scheduler import AsyncScheduler
from .wire import client_handshake, recv_msg, send_msg


def assign_shards(keys, num_shards: int) -> dict:
    """Assign module keys to shards round-robin over ``sorted(keys)``.

    Sorted order is stable across processes (unlike per-process-salted ``hash``),
    so scheduler, parameter servers, and workers agree on the routing.
    """
    return {k: i % num_shards for i, k in enumerate(sorted(keys))}


# -- parameter server --------------------------------------------------------


class ParameterServer(_ReactorServer):
    """Owns a shard of module keys: their weights, versions, and outer optimizers.

    ``fetch`` returns the requested owned weights (versioned; private only when the
    worker is cold); ``push`` applies a weighted per-module outer step to owned
    shared modules and stores owned private modules.
    """

    def __init__(self, config, owned_keys, diloco, *, host="0.0.0.0", port=0,
                 auth_key=None, device="cpu", resume_dir=None, **reactor_kw):
        super().__init__(host=host, port=port, auth_key=auth_key, **reactor_kw)
        self.config = config
        self.diloco = diloco
        self.device = torch.device(device)
        self.owned_keys = set(owned_keys)
        # Build the full bank deterministically, then keep only this shard's keys.
        full = build_module_bank(config)
        self.bank = {k: full[k].to(self.device) for k in self.owned_keys}
        self._lock = threading.Lock()
        self._versions = {k: 0 for k in self.owned_keys if not is_private_key(k)}
        self._outer_opts = {
            k: make_outer_optimizer({k: self.bank[k]}, diloco) for k in self._versions
        }
        if resume_dir is not None and os.path.exists(os.path.join(resume_dir, self._shard_name())):
            self.load_shard(resume_dir)  # restart this shard from a checkpoint

    def _handle(self, msg: dict, nbytes: int):
        kind = msg.get("type")
        if kind == "fetch":
            return self._fetch(msg)
        if kind == "push":
            return self._push(msg)
        if kind == "checkpoint":
            self.save_shard(msg["dir"])
            return {"type": "ack"}
        return None

    def _fetch(self, msg: dict) -> dict:
        have = msg.get("have", {})
        cold = msg.get("cold", False)
        weights, versions = {}, {}
        with self._lock:
            for k in msg.get("keys", []):
                if k not in self.owned_keys:
                    continue
                if is_private_key(k):
                    if cold:  # ship the path's private modules only on a cold start
                        weights[k] = _state_to_cpu(self.bank[k].state_dict())
                else:
                    versions[k] = self._versions[k]
                    if have.get(k) != self._versions[k]:  # ship only what's stale
                        weights[k] = _state_to_cpu(self.bank[k].state_dict())
        return {"type": "weights", "weights": weights, "versions": versions}

    def _push(self, msg: dict) -> dict:
        with self._lock:
            for k, upd in (msg.get("updates") or {}).items():
                if k not in self.owned_keys or is_private_key(k):
                    continue
                w = float(upd["weight"])
                apply_outer_grads(self.bank[k], [w * g.to(self.device) for g in upd["grad"]])
                self._outer_opts[k].step()
                self._outer_opts[k].zero_grad(set_to_none=True)
                self._versions[k] += 1
            for k, sd in (msg.get("private") or {}).items():
                if k in self.owned_keys:
                    _load_into(self, k, sd)  # store latest private (authoritative-local)
        return {"type": "ack"}

    def _shard_name(self) -> str:
        # Stable across processes (unlike per-process-salted ``hash``), so a shard
        # saved by one process is found by the restarted one with the same keys.
        digest = hashlib.sha256(",".join(sorted(self.owned_keys)).encode()).hexdigest()[:16]
        return f"shard_{digest}.pt"

    def save_shard(self, dirpath: str) -> None:
        """Persist this shard's weights + versions to ``dir/shard_<stable-hash>.pt``."""
        os.makedirs(dirpath, exist_ok=True)
        with self._lock:
            blob = {"weights": {k: _state_to_cpu(m.state_dict()) for k, m in self.bank.items()},
                    "versions": dict(self._versions)}
        tmp = os.path.join(dirpath, self._shard_name() + ".tmp")
        torch.save(blob, tmp)
        os.replace(tmp, os.path.join(dirpath, self._shard_name()))  # atomic

    def load_shard(self, dirpath: str) -> None:
        blob = torch.load(os.path.join(dirpath, self._shard_name()),
                          map_location=self.device, weights_only=False)
        with self._lock:
            for k, sd in blob["weights"].items():
                if k in self.bank:
                    self.bank[k].load_state_dict({n: v.to(self.device) for n, v in sd.items()})
            self._versions.update({k: v for k, v in blob["versions"].items() if k in self._versions})


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
                 heartbeat_timeout=30.0, ps_tls=None, **reactor_kw):
        super().__init__(host=host, port=port, auth_key=auth_key, **reactor_kw)
        self.ps_tls = ps_tls  # client context for the scheduler's checkpoint RPCs to PSs
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
        self._key_shard = assign_shards(self.topology.module_keys(), len(self.ps_addrs))
        self._routing = {k: list(self.ps_addrs[s]) for k, s in self._key_shard.items()}

        self._lock = threading.Lock()
        self._serving = False
        self._T = 0
        self._target = 0
        self._completed: dict = {}
        self._inflight: dict = {}
        self._issued: dict = {}
        self._owner: dict = {}

    def _handle(self, msg: dict, nbytes: int):
        kind = msg.get("type")
        if kind == "request":
            return self._next_task(msg)
        if kind == "commit":
            return self._commit(msg)
        if kind == "heartbeat":
            self._heartbeat(msg)
        return None

    def _next_task(self, req: dict) -> dict:
        wid = req.get("worker_id")
        warm = {tuple(p) for p in req.get("warm_paths", [])}
        cached = {tuple(p) for p in req.get("cached_shards", [])}
        with self._lock:
            if not self._serving or self._T >= self._target:
                return {"type": "stop"} if self._stop else {"type": "idle"}
            self._reclaim_inflight_locked()
            eligible = [p for p in self._completed if p not in self._inflight]
            if not eligible:
                return {"type": "idle"}
            path = min(eligible, key=lambda p: (self._completed[p], p not in warm, p))
            self._owner[path] = wid
            self._inflight[path] = time.monotonic() + self.heartbeat_timeout
            self._issued[path] = self._T
            generation = self._completed[path]
            keys = self.topology.path_module_keys(path)
            routing = {k: self._routing[k] for k in keys}
        shard = self.corpus.shard(self.topology.path_index(path)) if path not in cached else None
        return {
            "type": "task",
            "gen_id": generation,
            "path": path,
            "routing": routing,
            "shard": shard,
            "batch_size": self.batch_size,
            "total_rounds": self.total_rounds,
            "seed": self.seed,
        }

    def _commit(self, msg: dict) -> dict:
        path = msg["path"]
        with self._lock:
            if path not in self._inflight:
                return {"type": "commit_ack", "accepted": False}  # stale / already freed
            staleness = self._T - self._issued.get(path, self._T)
            self._inflight.pop(path, None)
            if staleness > self.staleness_bound:
                self.metrics.record_stale_reject()
                return {"type": "commit_ack", "accepted": False}
            self._T += 1
            self._completed[path] = self._completed.get(path, 0) + 1
            damp = 1.0 / (1.0 + staleness) if self.staleness_weight == "inverse" else 1.0
            push_weight = self.corpus.shard_weight(self.topology.path_index(path)) * damp
            self.metrics.record_update(staleness)
            return {"type": "commit_ack", "accepted": True, "push_weight": push_weight}

    def _heartbeat(self, msg: dict) -> None:
        path = msg["path"]
        with self._lock:
            if path in self._inflight and self._owner.get(path) == msg.get("worker_id"):
                self._inflight[path] = time.monotonic() + self.heartbeat_timeout

    def _reclaim_inflight_locked(self) -> None:
        now = time.monotonic()
        for path, deadline in list(self._inflight.items()):
            if now >= deadline:
                del self._inflight[path]
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
        """Save the scheduler clock and trigger every parameter server to persist."""
        os.makedirs(dirpath, exist_ok=True)
        for addr in self.ps_addrs:
            try:
                s = _ps_connect(addr, self.auth_key, DEFAULT_MAX_MSG_BYTES, 5.0,
                                tls=self.ps_tls, server_hostname=addr[0])
                _rpc(s, {"type": "checkpoint", "dir": dirpath}, DEFAULT_MAX_MSG_BYTES)
                s.close()
            except OSError:
                pass  # a PS that's momentarily unreachable is checkpointed next time
        with self._lock:
            state = {"T": self._T, "completed": dict(self._completed)}
        tmp = os.path.join(dirpath, "scheduler.pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(dirpath, "scheduler.pt"))

    def _load_state(self, dirpath: str) -> None:
        path = os.path.join(dirpath, "scheduler.pt")
        if not os.path.exists(path):
            return
        state = torch.load(path, weights_only=False)
        with self._lock:
            self._T = state["T"]
            self._completed = dict(state["completed"])

    def shutdown(self) -> None:
        with self._lock:
            self._serving = False
        super().shutdown()


# -- sharded worker ----------------------------------------------------------


def run_sharded_worker(config, diloco, scheduler_addr, *, device="cpu", seed=0,
                       auth_key=None, max_tasks=None, heartbeat_interval=3.0,
                       poll_interval=0.02, max_msg_bytes=DEFAULT_MAX_MSG_BYTES,
                       connect_timeout=10.0, reconnect=False, reconnect_timeout=30.0,
                       fault_hook=None, tls=None, tls_hostname=None):
    """Train path-tasks for a sharded scheduler + parameter servers.

    Per task: lease from the scheduler, fetch the path's modules from the owning
    parameter servers, train, commit (accept/reject + damped weight), and push the
    pseudo-gradients to the owning servers. Warm caches (private modules, Adam
    state, shard) persist across tasks. With ``reconnect`` a dropped scheduler/PS
    connection is retried (e.g. a coordinator restart); warm caches survive.
    """
    engine = _build_worker_engine(config, diloco, device, seed)
    worker = AsyncScheduler(engine, num_workers=1)
    wid = uuid.uuid4().hex
    warm: set = set()
    shard_cache: dict = {}
    versions: dict = {}          # shared key -> held version
    ps_conns: dict = {}          # (host, port) -> connected socket
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
                                   ps_conns, state, auth_key, max_msg_bytes, connect_timeout,
                                   heartbeat_interval, poll_interval, max_tasks, fault_hook,
                                   tls=tls)
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


def _serve_sharded(sch, engine, worker, wid, warm, shard_cache, versions, ps_conns, state,
                   auth_key, max_msg_bytes, connect_timeout, heartbeat_interval,
                   poll_interval, max_tasks, fault_hook, *, tls=None) -> bool:
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
                  "warm_paths": list(warm), "cached_shards": list(shard_cache)})
        task = recv_msg(sch, max_msg_bytes)
        if task is None:
            raise OSError("scheduler disconnected")  # not a clean stop -> reconnect
        if task["type"] == "stop":
            return True
        if task["type"] == "idle":
            time.sleep(poll_interval)
            continue

        path = task["path"]
        worker.seed = task["seed"]
        engine.total_rounds = task["total_rounds"]
        routing = {k: tuple(a) for k, a in task["routing"].items()}
        by_ps: dict = {}
        for k, addr in routing.items():
            by_ps.setdefault(addr, []).append(k)
        cold = path not in warm

        stop_beat = threading.Event()
        beat = threading.Thread(target=_sch_heartbeat,
                                args=(sch_send, stop_beat, heartbeat_interval, wid, path),
                                daemon=True)
        beat.start()
        try:
            if fault_hook is not None:
                fault_hook(path, 1)
            for addr, keys in by_ps.items():
                reply = _rpc(ps_sock(addr), {
                    "type": "fetch", "keys": keys, "cold": cold,
                    "have": {k: versions.get(k) for k in keys if not is_private_key(k)},
                }, max_msg_bytes)
                for k, sd in reply["weights"].items():
                    _load_into(engine, k, sd)
                versions.update(reply.get("versions", {}))
            if cold:
                engine._opt_state.pop(path, None)  # reset Adam on a cold start
            if task.get("shard") is not None:
                shard_cache[path] = task["shard"]
            shard = shard_cache[path]
            contrib = worker._train_path(path, shard, task["batch_size"], task["gen_id"])
        finally:
            stop_beat.set()
            beat.join(timeout=1)

        ack = _rpc_send(sch, send_lock, max_msg_bytes,
                        {"type": "commit", "path": path, "worker_id": wid,
                         "issued_token": task["gen_id"]})
        if ack is None:
            raise OSError("scheduler disconnected during commit")
        if ack.get("accepted"):
            w = ack["push_weight"]
            for addr, keys in by_ps.items():
                updates = {k: {"grad": contrib.shared_delta[k], "weight": w}
                           for k in keys if not is_private_key(k) and k in contrib.shared_delta}
                private = {k: contrib.private_state[k]
                           for k in keys if is_private_key(k) and k in contrib.private_state}
                _rpc(ps_sock(addr), {"type": "push", "updates": updates, "private": private},
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


def _sch_heartbeat(sch_send, stop_beat, interval, wid, path):
    while not stop_beat.wait(interval):
        try:
            sch_send({"type": "heartbeat", "path": path, "worker_id": wid})
        except OSError:
            return
