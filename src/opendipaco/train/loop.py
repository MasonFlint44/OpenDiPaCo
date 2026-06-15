"""The DiPaCo training engine.

One *round* = one DiLoCo outer step:

1. each owned path takes ``inner_steps`` AdamW steps on its data shard;
2. each shared module's pseudo-gradient (``global - local``) is weighted by shard
   size and summed across the paths that share it (locally, then across nodes via
   the backend);
3. the outer SGD+Nesterov optimizer applies the averaged delta to the global
   modules;
4. the refreshed global weights are copied back into every path's working copy.

The same code path drives both the single-process simulation (``LocalBackend``)
and the distributed setting (``TorchDistBackend``); only the backend's
``global_reduce`` differs.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ..backend.base import ReducedDelta, SyncBackend
from ..config import DiLoCoConfig, DiPaCoConfig
from ..data.sharding import ShardedCorpus
from ..model import build_module_bank, build_path_model
from ..init import warm_start_modules
from ..optim.diloco import (
    apply_outer_grads,
    inner_lr_at,
    make_inner_optimizer,
    make_outer_optimizer,
    module_delta,
)
from ..topology import Path, is_private_key


@dataclass
class RoundMetrics:
    round: int
    inner_loss: float                       # mean final inner loss across owned paths
    per_path_loss: dict[Path, float] = field(default_factory=dict)
    delta_norm: float = 0.0                 # mean averaged-delta norm across modules
    val_loss: float | None = None           # mean shard-validation loss (if a val split exists)


def _iter_batches(shard: torch.Tensor, batch_size: int, steps: int, gen: torch.Generator):
    """Yield ``steps`` batches by iterating the shard in shuffled epochs.

    Each example is seen once per epoch (sampling *without* replacement); when an
    epoch is exhausted the shard is reshuffled and iteration continues, so the
    inner phase takes ``steps`` proper SGD steps over the shard's data. The
    trailing partial batch of an epoch is dropped for consistent batch sizes.
    """
    n = shard.size(0)
    bs = min(batch_size, n)
    perm = torch.randperm(n, generator=gen)
    pos = 0
    for _ in range(steps):
        if pos + bs > n:  # epoch boundary -> reshuffle
            perm = torch.randperm(n, generator=gen)
            pos = 0
        yield shard[perm[pos : pos + bs]]
        pos += bs


@torch.no_grad()
def _copy_into(dst: nn.Module, src: nn.Module) -> None:
    for d, s in zip(dst.parameters(), src.parameters()):
        d.data.copy_(s.data)


def _optimizer_state_to_cpu(state: dict) -> dict:
    """Move an optimizer ``state_dict``'s tensors to CPU (for offload between rounds)."""
    moved = {}
    for idx, entry in state["state"].items():
        moved[idx] = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in entry.items()
        }
    return {"state": moved, "param_groups": state["param_groups"]}


def _state_to_cpu(sd: dict) -> dict:
    """Deep-copy a module ``state_dict`` onto CPU (so checkpoints are device-free)."""
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def config_fingerprint(config: DiPaCoConfig, diloco: DiLoCoConfig) -> str:
    """A stable hash of the run's config, used to guard checkpoint reloads."""
    payload = repr(config) + "||" + repr(diloco)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_inner_steps(
    pm,
    opt,
    shard,
    batch_size,
    gen,
    *,
    inner_steps,
    total_steps,
    base_step,
    diloco,
    device,
):
    """One path's inner loop: ``inner_steps`` AdamW steps with the cosine LR.

    Factored out of :meth:`DiPaCoEngine._inner_train` so the async scheduler can
    reuse the *exact* same numerics (the LR schedule, grad clip, and step) with
    its own per-task RNG. Returns the last batch's loss.

    With ``diloco.inner_autocast`` the forward/loss runs under bf16 autocast;
    parameters and gradients stay fp32 (bf16 needs no GradScaler), so the
    pseudo-gradient and clipping are unchanged in dtype.
    """
    device_type = torch.device(device).type
    autocast_on = getattr(diloco, "inner_autocast", False)
    last_loss = 0.0
    pm.train()
    pm.checkpoint = getattr(diloco, "activation_checkpoint", False)  # exact; W3b
    pm.loss_chunks = getattr(diloco, "loss_chunks", 1)               # chunked CE; W3c
    for i, batch in enumerate(_iter_batches(shard, batch_size, inner_steps, gen)):
        for group in opt.param_groups:
            group["lr"] = inner_lr_at(base_step + i, total_steps, diloco)
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=autocast_on):
            _, loss = pm(batch, labels=batch)
        loss = loss.float()
        loss.backward()
        if diloco.inner_grad_clip is not None:
            nn.utils.clip_grad_norm_(pm.parameters(), diloco.inner_grad_clip)
        opt.step()
        last_loss = float(loss.detach())
    return last_loss


class DiPaCoEngine:
    def __init__(
        self,
        config: DiPaCoConfig,
        diloco: DiLoCoConfig,
        backend: SyncBackend,
        device: str | torch.device = "cpu",
        seed: int = 0,
        init_from: str | nn.Module | None = None,
        materialize: str = "eager",
    ):
        if materialize not in ("eager", "serial"):
            raise ValueError(f"materialize must be 'eager' or 'serial', got {materialize!r}")
        self.config = config
        self.diloco = diloco
        self.backend = backend
        self.device = torch.device(device)
        self.topology = backend.topology
        # "eager": hold every owned path's working model + optimizer co-resident.
        # "serial": materialize one owned path at a time per round (peak memory ~=
        # the owned module bank + a single path), offloading inner-optimizer state
        # to CPU between rounds -- the paper's many-paths-per-worker regime.
        self.materialize = materialize
        self._seed = seed                      # kept for the async scheduler's per-path RNG
        self.total_rounds: int | None = None  # full-run horizon for the LR schedule
        self._global_round = 0                 # rounds completed so far (persists across fit calls)

        # Deterministic init so every node/path builds *identical* module weights;
        # shared modules must start the same everywhere for averaging to be valid.
        torch.manual_seed(seed)
        full_bank = build_module_bank(config)
        self.bank: dict[str, nn.Module] = {
            k: m.to(self.device) for k, m in full_bank.items() if backend.owns_module(k)
        }
        # Warm-start every module from a pretrained dense model (paper theta-bar).
        if init_from is not None:
            warm_start_modules(self.bank, self.topology, init_from)
        # Private modules are never communicated, so they take no outer step; only
        # shared modules are driven by the outer (DiLoCo) optimizer.
        shared = {k: m for k, m in self.bank.items() if not is_private_key(k)}
        self.outer_opt = make_outer_optimizer(shared, diloco)

        # Working copies. In "eager" mode they are persistent; in "serial" mode they
        # are built on demand each round and ``_opt_state`` holds the CPU-offloaded
        # inner-optimizer state so persistence is preserved.
        self.path_models: dict[Path, nn.Module] = {}
        self.inner_opts: dict[Path, torch.optim.Optimizer] = {}
        self._opt_state: dict[Path, dict] = {}
        if materialize == "eager":
            for path in backend.owned_paths():
                pm = build_path_model(config, path, self.bank, deepcopy=True).to(self.device)
                self.path_models[path] = pm
                self.inner_opts[path] = make_inner_optimizer(pm, diloco)

        self._gen = torch.Generator().manual_seed(seed + 1)

        # Per-path early stopping: best shard-validation loss and a CPU snapshot of
        # that path's full weights at the round where it was lowest (paper: each
        # path keeps the params with the lowest loss on its shard-validation set).
        self.best_val_loss: dict[Path, float] = {}
        self.best_path_state: dict[Path, dict] = {}

    def _outer_weight(self, key: str, path_idx: int, corpus: ShardedCorpus) -> float:
        """Per-path weight for a module's pseudo-gradient.

        * shard_size_weighting -> alpha_i = |D_i| / |D_total| (paper Eq. 2-3);
          summed un-normalized, the alphas carry the normalization.
        * else, normalized mode  -> 1.0 (a mean is taken by dividing by sum-of-w).
        * else, un-normalized    -> 1 / P_{l,e}, so the raw sum is the 1/P mean of
          Algorithm 1 (line 17).
        """
        if self.diloco.shard_size_weighting:
            return corpus.shard_weight(path_idx)
        if self.diloco.normalize_outer_delta:
            return 1.0
        return 1.0 / self.topology.sharing_count(key)

    # -- path materialization ------------------------------------------------
    def _acquire(self, path: Path):
        """Return ``(model, optimizer)`` for ``path`` (persistent or freshly built)."""
        if self.materialize == "eager":
            return self.path_models[path], self.inner_opts[path]
        pm = build_path_model(self.config, path, self.bank, deepcopy=True).to(self.device)
        opt = make_inner_optimizer(pm, self.diloco)
        if path in self._opt_state:
            opt.load_state_dict(self._opt_state[path])  # restores moments to pm's device
        return pm, opt

    def _release(self, path: Path, opt: torch.optim.Optimizer) -> None:
        if self.materialize == "serial":
            self._opt_state[path] = _optimizer_state_to_cpu(opt.state_dict())

    def _path_model_for(self, path: Path) -> nn.Module:
        """A model for ``path`` reflecting the current bank (for validation/snapshot)."""
        if self.materialize == "eager":
            return self.path_models[path]
        return build_path_model(self.config, path, self.bank, deepcopy=True).to(self.device)

    def _inner_train(self, pm, opt, shard, batch_size, round_idx) -> float:
        inner_steps = self.diloco.inner_steps
        total_steps = self.total_rounds * inner_steps if self.total_rounds else None
        return run_inner_steps(
            pm, opt, shard, batch_size, self._gen,
            inner_steps=inner_steps, total_steps=total_steps,
            base_step=round_idx * inner_steps, diloco=self.diloco, device=self.device,
        )

    # -- one outer round -----------------------------------------------------
    def run_round(self, corpus: ShardedCorpus, batch_size: int, round_idx: int = 0) -> RoundMetrics:
        owned_keys = [k for k in self.topology.module_keys() if self.backend.owns_module(k)]
        # Shared modules are averaged + outer-stepped; private modules are never
        # communicated and just keep their locally-trained weights.
        shared_keys = [k for k in owned_keys if not is_private_key(k)]
        acc_sum: dict[str, list[torch.Tensor]] = {
            k: [torch.zeros_like(p) for p in self.bank[k].parameters()] for k in shared_keys
        }
        acc_w: dict[str, float] = dict.fromkeys(shared_keys, 0.0)

        per_path_loss: dict[Path, float] = {}
        for path in self.backend.owned_paths():
            path_idx = self.topology.path_index(path)
            shard = corpus.shard(path_idx).to(self.device)
            if shard.size(0) == 0:
                continue  # empty shard contributes nothing this round

            pm, opt = self._acquire(path)
            per_path_loss[path] = self._inner_train(pm, opt, shard, batch_size, round_idx)

            # Private modules: the locally-trained weights *are* authoritative.
            # Shared modules: accumulate the pseudo-gradient, weighted per the
            # paper's alpha re-weighting (Eq. 2-3).
            for key, local_mod in pm.modules_by_key().items():
                if not self.backend.owns_module(key):
                    continue
                if is_private_key(key):
                    _copy_into(self.bank[key], local_mod)
                    continue
                w = self._outer_weight(key, path_idx, corpus)
                for accT, d in zip(acc_sum[key], module_delta(self.bank[key], local_mod)):
                    accT.add_(w * d)
                acc_w[key] += w

            self._release(path, opt)
            if self.materialize == "serial":
                del pm, opt  # free the working copy before the next path

        delta_norms: list[float] = []
        for key in shared_keys:
            reduced = self.backend.global_reduce(key, ReducedDelta(acc_sum[key], acc_w[key]))
            if self.diloco.normalize_outer_delta:
                denom = max(reduced.weight, 1e-12)
                avg = [s / denom for s in reduced.summed]
            else:
                # Raw weighted sum: alpha terms (or the 1/P weights) already
                # carry the normalization, matching the paper's outer step.
                avg = list(reduced.summed)
            if self.diloco.rescale_by_sqrt_sharing:
                scale = math.sqrt(self.topology.sharing_count(key))
                avg = [a * scale for a in avg]
            apply_outer_grads(self.bank[key], avg)
            delta_norms.append(
                math.sqrt(sum(float(a.pow(2).sum()) for a in avg)) if avg else 0.0
            )

        self.outer_opt.step()
        self.outer_opt.zero_grad(set_to_none=True)

        # Eager mode: redistribute refreshed shared weights into the persistent
        # working copies (private modules already equal them). Serial mode rebuilds
        # path models from the bank next round, so there is nothing to redistribute.
        if self.materialize == "eager":
            for path, pm in self.path_models.items():
                for key, local_mod in pm.modules_by_key().items():
                    if self.backend.owns_module(key) and not is_private_key(key):
                        _copy_into(local_mod, self.bank[key])

        # Per-path early stopping: snapshot each path whose shard-validation loss
        # is the best seen so far.
        val_loss = self._track_best(corpus, batch_size) if corpus.has_validation else None

        mean_loss = (
            sum(per_path_loss.values()) / len(per_path_loss) if per_path_loss else float("nan")
        )
        return RoundMetrics(
            round=round_idx,
            inner_loss=mean_loss,
            per_path_loss=per_path_loss,
            delta_norm=sum(delta_norms) / len(delta_norms) if delta_norms else 0.0,
            val_loss=val_loss,
        )

    @torch.no_grad()
    def _eval_val(self, pm: nn.Module, seqs: torch.Tensor, batch_size: int) -> float:
        """Token-level loss of ``pm`` on a validation shard."""
        pm.eval()
        total, tokens = 0.0, 0
        for start in range(0, seqs.size(0), batch_size):
            batch = seqs[start : start + batch_size].to(self.device)
            logits, _ = pm(batch)
            pred, tgt = logits[:, :-1, :], batch[:, 1:]
            total += float(
                F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="sum")
            )
            tokens += tgt.numel()
        pm.train()
        return total / max(tokens, 1)

    def _track_best(self, corpus: ShardedCorpus, batch_size: int) -> float | None:
        losses: list[float] = []
        for path in self.backend.owned_paths():
            val = corpus.val_shard(self.topology.path_index(path))
            if val is None or val.size(0) == 0:
                continue
            pm = self._path_model_for(path)  # reflects the post-outer-step bank
            loss = self._eval_val(pm, val, batch_size)
            losses.append(loss)
            if path not in self.best_val_loss or loss < self.best_val_loss[path]:
                self.best_val_loss[path] = loss
                self.best_path_state[path] = {
                    k: v.detach().cpu().clone() for k, v in pm.state_dict().items()
                }
            if self.materialize == "serial":
                del pm
        return sum(losses) / len(losses) if losses else None

    def compose_best(self, path: Path) -> nn.Module:
        """A :class:`PathModel` for ``path`` loaded with its best-val checkpoint.

        Falls back to current weights if the path was never validated. Use this
        (rather than the live bank) at inference to honour per-path early stopping.
        """
        pm = build_path_model(self.config, path, self.bank, deepcopy=True).to(self.device)
        if path in self.best_path_state:
            pm.load_state_dict(self.best_path_state[path])
        return pm

    def fit(
        self,
        corpus: ShardedCorpus,
        num_rounds: int,
        batch_size: int,
        log_every: int = 1,
        total_rounds: int | None = None,
    ):
        """Train for ``num_rounds`` outer rounds.

        The inner LR schedule (cosine) spans the whole run via a persistent round
        counter, so it does **not** restart between ``fit`` calls. When training is
        split across several ``fit`` calls (e.g. an EM loop), pass ``total_rounds``
        (or set ``engine.total_rounds``) to the *full* horizon so the cosine decays
        over the entire run rather than just the first chunk.
        """
        if total_rounds is not None:
            self.total_rounds = total_rounds
        elif self.total_rounds is None:
            self.total_rounds = self._global_round + num_rounds  # assume single-run

        history: list[RoundMetrics] = []
        for _ in range(num_rounds):
            m = self.run_round(corpus, batch_size, round_idx=self._global_round)
            self._global_round += 1
            history.append(m)
            if log_every and (m.round % log_every == 0 or _ == num_rounds - 1):
                print(
                    f"[round {m.round:4d}] inner_loss={m.inner_loss:.4f} "
                    f"delta_norm={m.delta_norm:.4f}"
                )
        return history

    # -- access global weights (e.g. for inference / checkpointing) ----------
    def global_modules(self) -> dict[str, nn.Module]:
        return self.bank

    # -- checkpoint / resume -------------------------------------------------
    def _gather_inner_state(self) -> dict[Path, dict]:
        """Inner-optimizer state per owned path, on CPU, regardless of mode.

        Eager keeps live optimizers in ``self.inner_opts``; serial keeps the
        CPU-offloaded state in ``self._opt_state`` (paths not yet touched have
        none yet). Either way we return ``{path: cpu_state_dict}``.
        """
        out: dict[Path, dict] = {}
        if self.materialize == "eager":
            for path, opt in self.inner_opts.items():
                out[path] = _optimizer_state_to_cpu(opt.state_dict())
        else:
            out = {p: s for p, s in self._opt_state.items()}
        return out

    def state_dict(self) -> dict:
        """A device-free snapshot of everything needed to resume this node.

        Captures only the modules this node owns (the per-rank bank), the outer
        and per-path inner optimizer state, the LR-schedule counters, the
        per-path early-stopping snapshots, and RNG state. Combined with the same
        ``config``/``diloco``/``backend`` it resumes bit-for-bit.
        """
        return {
            "format": 1,
            "fingerprint": config_fingerprint(self.config, self.diloco),
            "world_size": self._world_size(),
            "materialize": self.materialize,
            "bank": {k: _state_to_cpu(m.state_dict()) for k, m in self.bank.items()},
            "outer_opt": _optimizer_state_to_cpu(self.outer_opt.state_dict()),
            "inner_opt_state": self._gather_inner_state(),
            "global_round": self._global_round,
            "total_rounds": self.total_rounds,
            "best_val_loss": dict(self.best_val_loss),
            "best_path_state": {
                p: _state_to_cpu(sd) for p, sd in self.best_path_state.items()
            },
            "gen_state": self._gen.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def _world_size(self) -> int:
        sizes = {self.backend.rank_of_path(p) for p in self.topology.paths()}
        return len(sizes)

    def load_state_dict(self, state: dict, *, strict: bool = True) -> None:
        """Restore a snapshot produced by :meth:`state_dict` into this engine.

        With ``strict`` (default) a config-fingerprint or world-size mismatch
        raises, since resuming into a differently-shaped run would silently
        corrupt training.
        """
        if strict:
            fp = config_fingerprint(self.config, self.diloco)
            if state.get("fingerprint") != fp:
                raise ValueError(
                    "checkpoint config fingerprint does not match this engine "
                    f"({state.get('fingerprint')} != {fp}); pass strict=False to override"
                )
            if state.get("world_size") != self._world_size():
                raise ValueError(
                    "checkpoint world_size does not match this engine "
                    f"({state.get('world_size')} != {self._world_size()})"
                )

        for key, sd in state["bank"].items():
            self.bank[key].load_state_dict({k: v.to(self.device) for k, v in sd.items()})
        self.outer_opt.load_state_dict(state["outer_opt"])
        self._global_round = state["global_round"]
        self.total_rounds = state["total_rounds"]
        self.best_val_loss = dict(state["best_val_loss"])
        self.best_path_state = {p: dict(sd) for p, sd in state["best_path_state"].items()}
        self._gen.set_state(state["gen_state"])
        torch.set_rng_state(state["torch_rng_state"])

        inner = state["inner_opt_state"]
        if self.materialize == "serial":
            # State is reloaded lazily by ``_acquire``; just stash it on CPU.
            self._opt_state = {p: s for p, s in inner.items()}
        else:
            # Rebuild working models from the restored bank (redistribute already
            # synced them at the round boundary), then reattach inner-opt state.
            for path in self.backend.owned_paths():
                pm = build_path_model(
                    self.config, path, self.bank, deepcopy=True
                ).to(self.device)
                opt = make_inner_optimizer(pm, self.diloco)
                if path in inner:
                    opt.load_state_dict(inner[path])
                self.path_models[path] = pm
                self.inner_opts[path] = opt
