"""An async, fault-tolerant task-queue scheduler for DiPaCo.

The synchronous engine (:class:`DiPaCoEngine`) sweeps every owned path in order,
then reduces. The paper instead decouples *workers* from *paths*: workers lease a
path-task from a queue, train it for a generation, push back the pseudo-gradient,
and a coordinator aggregates and applies the outer step. A worker that is
preempted or dies just stops submitting -- its lease expires and the path is
re-queued -- so the run tolerates failures.

This module realizes that scheduler **in-process** with worker threads, the same
way ``LocalBackend`` is the single-process reference for the distributed
backend: the coordinator/lease/re-queue logic is the actual fault-tolerance
mechanism and is transport-agnostic, so a real multi-process transport can slot
in behind the same coordinator later. (A future multi-node version would swap the
in-process queue for sockets/files; the coordinator code here is unchanged.)

Design choices that make it faithful *and* testable:

* **Reuses the engine.** The coordinator wraps a :class:`DiPaCoEngine` for its
  module bank, outer optimizer, alpha weighting, and path building; the reduce
  step mirrors ``DiPaCoEngine.run_round`` exactly, so the outer-step math is the
  single source of truth.
* **Per-path optimizer state persists** across generations (the DiLoCo behavior),
  stored on the coordinator and handed to whichever worker leases that path next
  -- exactly the ``materialize="serial"`` regime. Use a serial engine so
  checkpoints capture it.
* **Execution-independent results.** Each path-task trains with its own RNG seeded
  from ``(seed, path, generation)``, and the reduce aggregates in a fixed path
  order, so the result does not depend on worker count or on any pattern of
  (recovered) failures. The map and reduce are each bit-deterministic in
  isolation; what little spread remains is benign floating-point noise from
  running compute in worker threads (PyTorch CPU reductions are not bit-stable
  across threads), so the tests assert numerical agreement rather than
  bit-equality.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field

import torch

from ..data.sharding import ShardedCorpus
from ..model import build_path_model
from ..optim.diloco import apply_outer_grads, make_inner_optimizer, module_delta
from ..topology import Path, is_private_key
from ..train.loop import (
    DiPaCoEngine,
    RoundMetrics,
    _optimizer_state_to_cpu,
    _state_to_cpu,
    run_inner_steps,
)


class TransientFault(Exception):
    """Raised by a fault hook to simulate a recoverable error -> immediate re-queue."""


class Preempt(Exception):
    """Raised by a fault hook to simulate a worker dying -> lease expiry re-queue."""


@dataclass
class Contribution:
    """One path's result for a generation (computed on a worker)."""

    path: Path
    path_idx: int
    loss: float
    shared_delta: dict[str, list[torch.Tensor]]  # shared key -> per-param (global - local)
    private_state: dict[str, dict]               # private key -> trained state_dict (CPU)
    opt_state: dict                              # inner-optimizer state (CPU) to persist
    empty: bool = False                          # empty shard -> no-op contribution


@dataclass
class _OuterAccumulator:
    """Running outer-gradient sums for one generation (enables streaming reduce)."""

    acc_sum: dict[str, list[torch.Tensor]]  # shared key -> per-param weighted sum
    acc_w: dict[str, float]                 # shared key -> sum of weights
    per_path_loss: dict                     # path -> last inner loss


@dataclass
class _Generation:
    """Thread-safe task set for one generation: lease, complete, nack, reclaim."""

    paths: list[Path]
    lease_timeout: float
    max_attempts: int
    min_success: int
    _cv: threading.Condition = field(default_factory=threading.Condition)
    _pending: collections.deque = field(default_factory=collections.deque)
    _leased: dict = field(default_factory=dict)      # path -> lease deadline
    _attempts: collections.Counter = field(default_factory=collections.Counter)
    results: dict = field(default_factory=dict)      # path -> Contribution
    failed: set = field(default_factory=set)         # paths that exhausted attempts
    reclaimed: int = 0                               # leases requeued after timeout (metrics)

    def __post_init__(self):
        self._pending.extend(self.paths)

    # -- internal (lock held) ------------------------------------------------
    def _reclaim_locked(self) -> None:
        now = time.monotonic()
        for path, deadline in list(self._leased.items()):
            if now >= deadline:
                del self._leased[path]
                if path in self.results:
                    continue  # already completed by a (faster) lease; nothing to do
                if self._attempts[path] >= self.max_attempts:
                    self.failed.add(path)
                else:
                    self._pending.append(path)
                    self.reclaimed += 1

    def _done_locked(self) -> bool:
        if self._pending or self._leased:
            return False
        # Nothing in flight: done once every path has succeeded or been given up.
        return len(self.results) + len(self.failed) >= len(self.paths)

    # -- worker API ----------------------------------------------------------
    def lease(self, wait: float = 0.05):
        """Lease the next available path, or ``None`` when the generation is done."""
        with self._cv:
            while True:
                self._reclaim_locked()
                if self._done_locked():
                    return None
                if self._pending:
                    path = self._pending.popleft()
                    self._attempts[path] += 1
                    self._leased[path] = time.monotonic() + self.lease_timeout
                    return path, self._attempts[path]
                self._cv.wait(wait)

    def poll_lease(self, pick=None):
        """Non-blocking lease for a server thread.

        Returns ``(path, attempt)`` if work is available, the string ``"done"``
        if the generation is finished, or ``"idle"`` if it is merely waiting on
        outstanding leases (so the caller should poll again later).

        ``pick(pending)`` optionally chooses *which* pending path to lease (used by
        the distributed coordinator for data-locality: prefer a path the
        requesting worker already holds warm). It returns a path or ``None``
        (nothing suitable right now -> ``"idle"``). Default is FIFO.
        """
        with self._cv:
            self._reclaim_locked()
            if self._done_locked():
                return "done"
            if not self._pending:
                return "idle"
            if pick is None:
                path = self._pending.popleft()
            else:
                path = pick(list(self._pending))
                if path is None:
                    return "idle"
                self._pending.remove(path)
            self._attempts[path] += 1
            self._leased[path] = time.monotonic() + self.lease_timeout
            return path, self._attempts[path]

    def refresh_lease(self, path: Path) -> None:
        """Extend a live lease's deadline (called on a worker heartbeat).

        Lets the lease/reclaim deadline be short (fast dead-worker detection)
        while a *slow but alive* task keeps its lease by heartbeating. No-op if
        the path is not currently leased (a stale heartbeat after reclaim).
        """
        with self._cv:
            if path in self._leased:
                self._leased[path] = time.monotonic() + self.lease_timeout

    def complete(self, path: Path, contrib: Contribution) -> None:
        with self._cv:
            if path in self.results:
                return  # a duplicate from a reclaimed-then-recovered lease; ignore
            self._leased.pop(path, None)
            self.failed.discard(path)  # a late success un-fails a timed-out path
            try:
                self._pending.remove(path)  # idempotent: don't redo a path that just finished
            except ValueError:
                pass
            self.results[path] = contrib
            self._cv.notify_all()

    def nack(self, path: Path) -> None:
        with self._cv:
            self._leased.pop(path, None)
            if path in self.results:
                return  # already done by another lease; ignore the nack
            if self._attempts[path] >= self.max_attempts:
                self.failed.add(path)
            else:
                self._pending.append(path)
            self._cv.notify_all()

    def reclaim(self) -> None:
        with self._cv:
            self._reclaim_locked()
            self._cv.notify_all()

    def is_done(self) -> bool:
        with self._cv:
            return self._done_locked()

    def pending_paths(self) -> list:
        """Paths currently waiting to be (re)leased -- used to expire stale owners."""
        with self._cv:
            self._reclaim_locked()
            return list(self._pending)


class AsyncScheduler:
    """Coordinator that trains a :class:`DiPaCoEngine` via a fault-tolerant queue.

    Parameters
    ----------
    engine:
        The engine whose bank / outer optimizer / config this drives. Construct it
        with ``materialize="serial"`` so checkpoints capture the per-path inner
        optimizer state the scheduler persists.
    num_workers:
        How many worker threads pull tasks concurrently.
    lease_timeout:
        Seconds before an unfinished lease is reclaimed (a worker presumed dead).
    max_attempts:
        How many times a path is retried before it is dropped for the generation.
    min_fraction:
        Fraction of paths that must succeed for a generation to apply its outer
        step; ``1.0`` (default) tries every path to completion, then proceeds with
        whatever survived (graceful degradation). Lower it to tolerate stragglers.
    """

    def __init__(
        self,
        engine: DiPaCoEngine,
        *,
        num_workers: int = 4,
        lease_timeout: float = 30.0,
        max_attempts: int = 3,
        min_fraction: float = 1.0,
    ):
        self.engine = engine
        self.num_workers = num_workers
        self.lease_timeout = lease_timeout
        self.max_attempts = max_attempts
        self.min_fraction = min_fraction
        self.seed = getattr(engine, "_seed", 0)
        self.dropped: set[Path] = set()
        self.errors: dict[Path, str] = {}  # last exception per failing path (debugging)

    # -- map: train one path on a worker ------------------------------------
    def _train_path(self, path: Path, shard: torch.Tensor, batch_size: int, generation: int):
        """Train ``path`` on ``shard`` and return its :class:`Contribution`.

        Takes the shard tensor directly (rather than a corpus) so the same
        compute path serves both the in-process worker and a remote worker that
        received its shard over the wire. Reads the starting weights from
        ``engine.bank`` -- which the remote worker has loaded with the
        coordinator's shipped authoritative weights for this path.
        """
        e = self.engine
        path_idx = e.topology.path_index(path)
        shard = shard.to(e.device)
        if shard.size(0) == 0:
            return Contribution(path, path_idx, float("nan"), {}, {}, {}, empty=True)

        # dedup_private (W3d, off by default) aliases the private modules from the
        # bank to save the copy -- so inner training mutates the warm private
        # cache in place. A warm task whose *shared* commit is later rejected as
        # stale therefore leaves its private training in place (it seeds the next
        # task), unlike the deep-copy path which discards it. This is intended:
        # private is local-authoritative (only the shared pseudo-gradient is
        # stale), and snapshotting to restore would defeat the memory saving. It
        # is part of the §0f-gated warm-private dynamics the dedup arm validates.
        pm = build_path_model(e.config, path, e.bank, deepcopy=True,
                              dedup_private=getattr(e.diloco, "dedup_private", False)).to(e.device)
        opt = make_inner_optimizer(pm, e.diloco)
        prior = e._opt_state.get(path)
        if prior is not None:
            opt.load_state_dict(prior)  # DiLoCo: this path's inner moments persist

        # Per-task RNG so training is independent of worker count / order.
        seed = (self.seed * 0x9E3779B1) ^ (path_idx * 0x85EBCA77) ^ (generation * 0xC2B2AE35)
        gen = torch.Generator().manual_seed(seed & 0x7FFFFFFFFFFFFFFF)

        inner_steps = e.diloco.inner_steps
        total_steps = e.total_rounds * inner_steps if e.total_rounds else None
        loss = run_inner_steps(
            pm, opt, shard, batch_size, gen,
            inner_steps=inner_steps, total_steps=total_steps,
            base_step=generation * inner_steps, diloco=e.diloco, device=e.device,
        )

        shared_delta: dict[str, list[torch.Tensor]] = {}
        private_state: dict[str, dict] = {}
        for key, local_mod in pm.modules_by_key().items():
            if is_private_key(key):
                private_state[key] = _state_to_cpu(local_mod.state_dict())
            else:
                shared_delta[key] = [d.detach().cpu() for d in module_delta(e.bank[key], local_mod)]
        return Contribution(
            path, path_idx, loss, shared_delta, private_state,
            _optimizer_state_to_cpu(opt.state_dict()),
        )

    def _worker(self, gen: _Generation, corpus, batch_size, generation, fault_hook):
        while True:
            leased = gen.lease()
            if leased is None:
                return
            path, attempt = leased
            try:
                if fault_hook is not None:
                    fault_hook(path, attempt)  # may raise TransientFault / Preempt
                shard = corpus.shard(self.engine.topology.path_index(path))
                contrib = self._train_path(path, shard, batch_size, generation)
                gen.complete(path, contrib)
            except Preempt:
                # Simulated death: abandon without ack; the lease will expire and
                # the path is re-queued by reclaim.
                continue
            except Exception as e:
                # Record the error so a path that fails every attempt (and is
                # dropped) is debuggable rather than silently disappearing.
                self.errors[path] = repr(e)
                gen.nack(path)

    def _monitor(self, gen: _Generation):
        # Reclaim expired leases even if every worker is blocked waiting.
        while not gen.is_done():
            time.sleep(max(self.lease_timeout / 2, 0.005))
            gen.reclaim()

    # -- reduce: aggregate contributions and apply the outer step -----------
    #
    # Split into new/fold/finalize so the distributed coordinator can *stream* --
    # fold each contribution on arrival and drop its tensors -- instead of
    # buffering every path's full delta in RAM. The in-process scheduler folds in
    # sorted-path order (bit-stable); the coordinator folds in arrival order
    # (already float-noisy, asserted within NOISE).
    def _new_accumulator(self) -> _OuterAccumulator:
        e = self.engine
        shared_keys = [k for k in e.topology.module_keys() if not is_private_key(k)]
        return _OuterAccumulator(
            acc_sum={k: [torch.zeros_like(p) for p in e.bank[k].parameters()] for k in shared_keys},
            acc_w=dict.fromkeys(shared_keys, 0.0),
            per_path_loss={},
        )

    def _fold_contribution(self, acc: _OuterAccumulator, corpus, c: Contribution,
                           *, persist_opt: bool = True) -> None:
        """Fold one path's contribution into the running outer-grad accumulator."""
        if c.empty:
            return
        e = self.engine
        acc.per_path_loss[c.path] = c.loss
        for key, sd in c.private_state.items():
            e.bank[key].load_state_dict({k: v.to(e.device) for k, v in sd.items()})
        for key, delta in c.shared_delta.items():
            w = e._outer_weight(key, c.path_idx, corpus)
            for accT, d in zip(acc.acc_sum[key], delta):
                accT.add_(w * d.to(e.device))
            acc.acc_w[key] += w
        if persist_opt and c.opt_state:
            e._opt_state[c.path] = c.opt_state  # persist this path's inner moments

    def _finalize_outer_step(self, acc: _OuterAccumulator, corpus, generation) -> RoundMetrics:
        """Apply the outer step from accumulated grads and return metrics."""
        e = self.engine
        topo = e.topology
        delta_norms: list[float] = []
        for key, w in acc.acc_w.items():
            if w == 0.0:
                continue  # no path contributed this key this generation
            if e.diloco.normalize_outer_delta:
                denom = max(w, 1e-12)
                avg = [s / denom for s in acc.acc_sum[key]]
            else:
                avg = acc.acc_sum[key]
            if e.diloco.rescale_by_sqrt_sharing:
                scale = math.sqrt(topo.sharing_count(key))
                avg = [a * scale for a in avg]
            apply_outer_grads(e.bank[key], avg)
            delta_norms.append(math.sqrt(sum(float(a.pow(2).sum()) for a in avg)) if avg else 0.0)

        e.outer_opt.step()
        e.outer_opt.zero_grad(set_to_none=True)

        val_loss = e._track_best(corpus, batch_size=self._last_batch_size) if corpus.has_validation else None
        losses = acc.per_path_loss
        mean_loss = sum(losses.values()) / len(losses) if losses else float("nan")
        return RoundMetrics(
            round=generation,
            inner_loss=mean_loss,
            per_path_loss=dict(losses),
            delta_norm=sum(delta_norms) / len(delta_norms) if delta_norms else 0.0,
            val_loss=val_loss,
        )

    def _apply_generation(self, corpus, contributions, generation) -> RoundMetrics:
        acc = self._new_accumulator()
        # Fixed path order -> aggregation is independent of completion order.
        for c in sorted(contributions, key=lambda c: c.path_idx):
            self._fold_contribution(acc, corpus, c)
        return self._finalize_outer_step(acc, corpus, generation)

    # -- one generation ------------------------------------------------------
    def run_generation(self, corpus, batch_size, generation, *, fault_hook=None) -> RoundMetrics:
        self._last_batch_size = batch_size
        paths = list(self.engine.topology.paths())
        min_success = max(1, math.ceil(self.min_fraction * len(paths)))
        gen = _Generation(paths, self.lease_timeout, self.max_attempts, min_success)

        workers = [
            threading.Thread(
                target=self._worker, args=(gen, corpus, batch_size, generation, fault_hook),
                daemon=True,
            )
            for _ in range(self.num_workers)
        ]
        monitor = threading.Thread(target=self._monitor, args=(gen,), daemon=True)
        for w in workers:
            w.start()
        monitor.start()
        for w in workers:
            w.join()
        monitor.join()

        self.dropped = set(gen.failed)
        return self._apply_generation(corpus, list(gen.results.values()), generation)

    def fit(
        self,
        corpus,
        num_generations: int,
        batch_size: int,
        *,
        total_generations: int | None = None,
        log_every: int = 1,
        fault_hook=None,
    ):
        """Train for ``num_generations`` outer generations via the task queue.

        Like :meth:`DiPaCoEngine.fit`, the cosine LR schedule spans the whole run
        via the engine's persistent generation counter; pass ``total_generations``
        (or set ``engine.total_rounds``) to the full horizon when splitting a run.
        """
        e = self.engine
        if total_generations is not None:
            e.total_rounds = total_generations
        elif e.total_rounds is None:
            e.total_rounds = e._global_round + num_generations

        history: list[RoundMetrics] = []
        for _ in range(num_generations):
            m = self.run_generation(corpus, batch_size, e._global_round, fault_hook=fault_hook)
            e._global_round += 1
            history.append(m)
            if log_every and (m.round % log_every == 0 or _ == num_generations - 1):
                drop = f" dropped={len(self.dropped)}" if self.dropped else ""
                print(
                    f"[gen {m.round:4d}] inner_loss={m.inner_loss:.4f} "
                    f"delta_norm={m.delta_norm:.4f}{drop}"
                )
        return history
