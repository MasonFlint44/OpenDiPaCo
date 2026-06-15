"""VRAM profiling for the worker training loop (W3a; docs/w3-vram-design.md).

Measure-first: before adding any memory lever, see where the per-round peak goes.
:func:`vram_breakdown` *counts* the path's real parameters (on the ``meta``
device, so it works for models far too big to allocate) and reports the per-round
peak as

    fetched-global + local-params + AdamW(2x) + grads + activations

The parameter/optimizer terms are **exact**; the activation term is a coarse
transformer estimate (long sequences and a big vocab's logits dominate it), so on
CUDA :func:`measure_peak` reports the *true* peak around a real round -- the
estimate is for planning, the measurement is the truth.
"""

from __future__ import annotations

import torch

from ..model import build_module_bank, build_path_model
from ..topology import is_private_key

# Coarse: live intermediate tensors a transformer block holds in the forward
# (residual stream copies, attention, the MLP intermediate). Only used for the
# *estimate*; measure_peak is authoritative on CUDA.
_BLOCK_ACT_FACTOR = 16


def _count_params(config):
    """``(total, private, embed, head)`` parameter counts for one path, deduped
    (tying-safe), computed on the ``meta`` device so a huge model isn't
    allocated just to be counted."""
    with torch.device("meta"):
        bank = build_module_bank(config)
        topo = config.build_topology()
        pm = build_path_model(config, topo.path_from_index(0), bank, deepcopy=False)

    def numel(params, seen):
        n = 0
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
        return n

    total = numel(pm.parameters(), set())
    private = 0
    pseen: set = set()
    for k, mod in pm.modules_by_key().items():   # actual keys (per-path when private)
        if is_private_key(k):
            private += numel(mod.parameters(), pseen)
    # embed/head by module object (the key is path-specific when private).
    embed = numel(pm.embed.parameters(), set())
    head = numel(pm.head.parameters(), set())
    return total, private, embed, head


def vram_breakdown(config, *, batch_size: int, seq_len: int, autocast: bool = False,
                   checkpoint: bool = False, master_bytes: int = 4,
                   chunked_logits: bool = False) -> dict:
    """Estimate one worker round's VRAM peak for ``config`` (bytes per term).

    The peak today stacks the fetched **global** + the **local** working params +
    **AdamW** moments (2x) + **grads**, plus **activations**. Flags model the W3
    levers: ``checkpoint`` (store block inputs, not activations), ``autocast``
    (bf16 activations), ``chunked_logits`` (don't materialize the full
    ``[tokens, vocab]`` logits). Parameter terms are exact; activations coarse.
    """
    total, private, _embed, _head = _count_params(config)
    act_bytes = 2 if autocast else 4
    depth = sum(config.backbone.layers_per_level)
    hidden = config.backbone.hidden_size
    tokens = batch_size * seq_len

    params = total * master_bytes
    glob = total * master_bytes      # today the worker holds a full global copy (D4 -> shared only)
    adam = 2 * total * master_bytes
    grads = total * master_bytes
    block_act = depth * tokens * hidden * act_bytes * (1 if checkpoint else _BLOCK_ACT_FACTOR)
    logits = 0 if chunked_logits else tokens * config.backbone.vocab_size * act_bytes
    activations = block_act + logits
    return {
        "n_params": total, "n_private": private,
        "params": params, "global": glob, "adam": adam, "grads": grads,
        "activations": activations, "logits": logits,
        "total": params + glob + adam + grads + activations,
    }


def fits(breakdown: dict, budget_bytes: int) -> bool:
    return breakdown["total"] <= budget_bytes


def measure_peak(config, diloco, *, device: str, batch_size: int, seq_len: int,
                 seed: int = 0) -> int | None:
    """The **true** peak (bytes) of one real worker round on CUDA, via
    ``torch.cuda.max_memory_allocated``; ``None`` off CUDA (no portable
    device-memory peak there -- use :func:`vram_breakdown` instead)."""
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return None
    from ..backend import LocalBackend
    from ..schedule import AsyncScheduler
    from .loop import DiPaCoEngine

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    engine = DiPaCoEngine(config, diloco, LocalBackend(config.build_topology()),
                          device=device, seed=seed, materialize="serial")
    worker = AsyncScheduler(engine, num_workers=1)
    path = config.build_topology().path_from_index(0)
    g = torch.Generator().manual_seed(seed)
    # Rows are `seq_len` long: the model is called pm(batch, labels=batch) and
    # shifts internally, so a seq_len row matches real training (and never
    # exceeds max_position_embeddings, unlike seq_len + 1).
    shard = torch.randint(0, config.backbone.vocab_size,
                          (max(2 * batch_size, 4), seq_len), generator=g)
    worker._train_path(path, shard, batch_size, 0)
    return int(torch.cuda.max_memory_allocated())
