# W3 design — fit one path in consumer VRAM

Status: **design; no slices landed yet.** W3 (from [viability-roadmap.md](viability-roadmap.md))
removes the third practical wall to consumer-hardware training: a worker holds
**one path**, not the whole model (DiPaCo's premise), but a *large* path can
still exceed consumer VRAM. The dominant chunk for a real vocab is the **private
embedding/head** (vocab × hidden, never communicated), and the per-round training
peak stacks several copies of the path's parameters plus activations.

Three operator calls (§6) fixed the approach: **measure first** (a VRAM profiler
before any lever, so we attack the real peak), **exact levers default-on**
(activation checkpointing + offload are bit-exact — no convergence risk, no §0f
gate — while *quantized training* is the one lossy lever, deferred behind §0f
like W2's compression), and a **12 GB target** (RTX 3060, the volunteer sweet
spot; 8 GB a stretch the same levers extend toward).

## 1. Goal and where the VRAM goes

**Goal.** A useful path **trains within ~12 GB** (8 GB stretch) on a worker.
W3 targets the **worker training loop** (`run_sharded_worker` →
`_build_worker_engine` → `_train_path`); owners/scheduler don't train (no
activations), so they are out of scope.

**The per-round peak** (path params `P`, split shared `S` + private `R`, where
the embedding/head `R` dominates for a real vocab):

| Consumer | Size | Notes |
|---|---|---|
| fetched **global** (for `global − local`) | ~`S` | only *shared* modules need a global copy; private's "global" is its local |
| **local** working params | `P` | the trained copy |
| **AdamW** state (`m`, `v`) | `2P` | on-GPU during the round (offloaded to CPU *between* rounds today, `_opt_state`) |
| **gradients** | `P` | during backward |
| **activations** | `A` | ∝ batch × seq × layers × hidden — the big variable, the checkpointing target |

So the peak is ≈ `4P + S + A`, and for a large vocab `R` (embed/head) dominates
`P` while long sequences make `A` dominate everything. Which term is biggest
*depends on the model+batch+seq* — hence **measure first**.

`diloco.inner_autocast` (bf16 forward, fp32 master) already exists and trims
activation/compute precision; it is a start, not the finish.

## 2. Exactness model — why most of W3 carries no §0f risk

This is the key contrast with W2 (whose levers were all lossy). W3's levers split:

- **Exact (bit-for-bit identical results), default-on, no §0f gate:**
  - **activation checkpointing** — recompute activations in backward instead of
    storing them; the math is unchanged.
  - **offload** — move tensors (optimizer state, embedding rows) between GPU and
    CPU; *where/when* a tensor lives, never its value.
  - **not duplicating the global copy of private modules** — private modules are
    never communicated, so a worker needs only one copy (D4).
  These change peak memory, not numerics, so the deterministic anchor stays
  bit-exact and they can ship **on** wherever they help.
- **Lossy (changes numerics) — §0f-gated, off by default, on-box validated:**
  - **quantized training** — 8-bit AdamW state, optional int8/int4 params. Rides
    the WAN §0f run for its convergence verdict, exactly like W2's compression;
    `examples/validate_dynamics.py` de-risks it on-box.

So W3 ships real VRAM reduction **now** with zero convergence debt, and isolates
the one risky bet.

## 3. Shape of the result

```
  WORKER ROUND (the VRAM peak):
    profile  -> breakdown {global, local, adam, grad, activations} vs a budget (W3a)
    activations:  torch.utils.checkpoint over the body blocks   -> A shrinks      (W3b, exact)
    private copy: keep ONE copy of embed/head (no global dup)    -> ~R saved       (W3b, exact)
    optimizer:    Adam m,v offloaded to CPU w/ prefetch, OR 8-bit Adam            (W3c/W3d)
    embedding:    tied head (R/2), CPU-gathered active rows, chunked head logits   (W3c, exact)
    params:       int8/int4 master (lossy)                                         (W3d, 0f)
```

The owner/scheduler data plane, the W2 compression, and the deterministic anchor
are untouched: W3 is worker-local training-loop memory engineering.

## 4. Decisions

### D1. Measure first — a VRAM profiler drives the priority
Before any lever, W3a ships a profiler: an **analytical calculator** (the §1
breakdown for a given config — params/Adam/activations/embedding, and fit-vs-
budget, à la W2's `bandwidth_budget.py`) **and a real peak measurement**
(`torch.cuda.max_memory_allocated` around a real worker round, with a CPU
fallback that reports the analytical estimate). The profiler names the dominant
term so W3b–W3d attack it in measured priority order, not by guess.

### D2. Exact levers ship default-on; quantized training is the only §0f-gated one
Per §2: checkpointing + offload + the private-copy de-dup change peak memory, not
results, so they default **on** where they help and need no convergence
validation. The lossy 8-bit/int-param training (W3d) is **off by default**,
on-box-validated, WAN-§0f-verdicted. The split keeps the anchor bit-exact.

### D3. Activation checkpointing over the body blocks (W3b, exact)
Wrap each `LlamaDecoderLayer` block of the path's body in
`torch.utils.checkpoint` (non-reentrant) so the forward stores only block
*inputs* and recomputes activations in backward — trading ~one extra forward
(~25–35% step time) for an activation-memory cut that scales with depth. Exact.
A flag (default-on for the worker; off for the tiny in-process anchor where it
only adds compute) controls it; checkpointing must coexist with `inner_autocast`
(recompute under the same autocast context) so they compose.

### D4. Keep one copy of private modules — don't duplicate the global (W3b, exact)
The pseudo-gradient `global − local` is only meaningful for **shared** modules;
private modules (the dominant embed/head) are never communicated, so the worker
needs no separate global copy of them. Building the worker's working path so
private modules are **not** deep-copied from a global (only shared are) saves
~`R` — the largest single exact win for a real vocab. The owner/anchor paths are
unaffected (they already treat private modules as locally authoritative).

### D5. Offload: optimizer state and embedding rows (W3c, exact)
Two exact offloads, applied by measured priority:
- **Optimizer state** — `m, v` (2`P`) offloaded to CPU and prefetched per inner
  step. This is PCIe-bound (touched every inner step, unlike the *between-round*
  `_opt_state` offload that already exists), so it is **opt-in** and pays off
  when Adam dominates and PCIe is fast; the lossy alternative is 8-bit Adam (D7).
- **Embedding** — the lookup touches only the *active* token rows, so the
  embedding table can live on CPU with active rows gathered to GPU per step
  (exact). The **head**'s full-vocab logits matmul needs the whole table, so it
  is handled by **tying** (D6) or **chunked** logit/loss computation, not row
  offload.

### D6. Embedding/head — tie first, then chunk (W3c, exact)
`tie_word_embeddings` (already a config field) makes head = embedᵀ, **halving**
`R` at no cost — the cheapest exact win when untied. For an untied or still-too-
large head, compute the vocab logits + cross-entropy in **chunks** over the vocab
dimension (a standard exact trick) so the full `[batch×seq, vocab]` logit tensor
never materializes — often a large activation term for big vocabs.

### D7. Quantized training (W3d) — custom 8-bit AdamW first, §0f-gated
The lossy lever, off by default. **8-bit AdamW state** (blockwise-quantized
`m, v`) is the biggest quantization win — optimizer moments tolerate 8 bits well
— and cuts `2P → ~0.5P`. Implement a **custom blockwise 8-bit Adam** rather than
take a CUDA-only `bitsandbytes` dependency: it is **CPU-testable** (CI is CPU),
needs no heavy dep, and keeps the lever auditable; note `bitsandbytes` as a
drop-in alternative for production CUDA throughput. Optional int8/int4 master
params come after, lower priority. All changes ride `validate_dynamics.py`
on-box + the WAN §0f verdict.

### D8. Target 12 GB; report fit against a budget
The profiler (D1) reports fit/headroom against a configurable budget, defaulting
to **12 GB** (RTX 3060). 8 GB is a stretch the same levers extend toward
(checkpointing + private-dedup + tied head + 8-bit Adam stacked). The point is a
*useful* path fits, not the largest possible.

### D9. Compatibility and the deterministic anchor
The exact levers (D3–D6) keep training **bit-for-bit identical** — the
synchronous anchor, the TCP/libp2p data plane, W1, and W2 are untouched. W3 is
worker-local. The W2a keyframe baseline already lives CPU/bf16 (off-device),
consistent with W3's offload philosophy. Quantized training (D7) is off by
default; on it changes numerics and is §0f-gated.

### D10. Explicitly deferred / out of scope
- **Cross-worker tensor/FSDP sharding of a single path** — DiPaCo's
  one-path-per-worker *is* the model sharding; intra-path sharding is a different
  regime, out of scope.
- **NVMe/disk offload of params** — CPU offload first; disk is a later tier.
- **bf16/fp16 master weights** — a numerics change (the outer step accumulates on
  the master); defer with the other lossy levers.
- **Production `bitsandbytes` / fused kernels** — measured during the 0f-systems
  GPU run, not assumed here (D7 ships a portable custom path).

## 5. Implementation slices

Each lands green on its own; the profiler comes first (so the rest is
measured-priority), the exact levers before the lossy one.

| Slice | Contents | Key tests |
|---|---|---|
| **W3a** | VRAM profiler (D1): analytical calculator (params/Adam/activations/embedding breakdown + fit-vs-budget) + real `max_memory_allocated` measurement of a worker round (CPU fallback = the estimate). `examples/vram_budget.py`. | The calculator's breakdown sums to the measured peak within tolerance on a small GPU/CPU run; fit-vs-budget reports correctly; CPU fallback returns the estimate. |
| **W3b** | Activation checkpointing over the body (D3) + private-copy de-dup (D4), both exact + default-on for the worker. | Peak activations drop with checkpointing on; training is **bit-identical** with/without (exact); the worker holds one copy of private modules; anchor unchanged. |
| **W3c** | Offload (D5) + embedding tie/chunk (D6), exact, by measured priority. | Optimizer/embedding offload cuts the peak; results bit-identical; tied-head halves `R`; chunked logits match unchunked exactly. |
| **W3d** | Quantized training (D7): custom blockwise 8-bit AdamW, off by default, §0f-gated; on-box `validate_dynamics` arm. | 8-bit Adam round-trips within the blockwise bound; a `quant-optim` dynamics arm converges; peak `2P → ~0.5P`. |

Rough sizing: W3a M, W3b M, W3c M–L, W3d M. M–L overall — worker-local, no new
transport or protocol, mostly memory engineering over the existing train loop.

## 6. Operator decisions (resolved)
1. **Approach (D1): measure first**, then activation checkpointing, staging
   offload/quantization by measured priority — a profiler before levers so we
   attack the real peak.
2. **Numerics (D2): exact levers default-on** (checkpointing + offload + private
   de-dup — no §0f gate, bit-exact), with **quantized training the one lossy,
   §0f-gated, off-by-default** follow-on.
3. **Target (D8): 12 GB** (RTX 3060, the volunteer sweet spot); **8 GB** a
   stretch the same levers extend toward.
