# W3 design — fit one path in consumer VRAM

Status: **W3 complete (W3a–W3d).** W3 (from [viability-roadmap.md](viability-roadmap.md))

**W3d status (the §0f-gated lossy levers).**
- *Blockwise 8-bit AdamW* (`diloco.optim_8bit`, `optim/adam8bit.py`): the inner
  optimizer's moments are stored quantized — symmetric int8 for `m`, **log-domain
  uint8 for `v`** — ~2 B/param vs fp32's 8 (a ~4× cut of the often-dominant
  optimizer term). The param update is full-precision (moments dequantized for the
  step, requantized for storage). *Implementation finding:* linear int8 for `v`
  zeroed small second-moments in mixed-magnitude blocks → `denom = √v + eps`
  collapsed → exploding step; the log domain (relative precision) fixed it. Off by
  default, §0f-gated; tracks fp32 AdamW on-box (`validate_dynamics` arm converges).
- *Private-copy de-dup* (`diloco.dedup_private`, reclassified from D4): aliases
  the worker's private modules (deep-copies only shared, tie-safe) — saves ~the
  embed/head and lets private warm in-place across tasks. Off by default,
  §0f-gated (it changes warm-round private dynamics); `validate_dynamics` arm
  converges. *Known characteristic (Codex review):* because training mutates the
  aliased private in place, a warm task whose **shared** commit is rejected as
  stale leaves its private training in place (seeding the next task) — unlike the
  deep-copy path, which discards it. This is intended: private is local-
  authoritative (only the shared pseudo-gradient is stale), and snapshotting to
  restore would defeat the memory saving; it's part of the validated dynamics.
- *8-bit step transient bounded (Codex review).* The step dequantizes moments to
  fp32 per parameter; processing each parameter in **block-chunks**
  (`_MAX_CHUNK_ELEMS`) keeps that a bounded transient instead of a full fp32 copy
  of a huge embed/head — so the optimizer-step peak actually realizes the int8
  storage win (the binding peak once checkpointing has shrunk activations).
- *Deferred (D5 CPU-offloads):* optimizer CPU-offload (superseded by 8-bit Adam)
  and embedding row-gather (covered by tying) remain unbuilt PCIe-bound
  follow-ups, not needed to hit 12 GB.



**W3c status (exact param/activation levers).**
- *Tied embed/head now actually works (D6 + a real bug fix).* The bank tied the
  weights, but `build_path_model(deepcopy=True)` deep-copied each module
  *separately*, **severing the tie** — a tied worker held two independent copies
  and trained them apart (no memory saving, wrong dynamics). Fixed by
  deep-copying the selection in one call (shared memo preserves the tie);
  bit-identical when nothing is tied. Tying now halves the dominant embed/head.
- *Chunked cross-entropy (D6).* `diloco.loss_chunks > 1` computes the vocab
  logits + loss in token-chunks (`PathModel._chunked_loss`), so the full
  `[tokens, vocab]` logits never materialize — the big activation cut for a large
  vocab. The training path discards logits, so it returns `(None, loss)`;
  callers wanting logits pass `labels=None`. Mathematically the dense loss, but
  the sum runs in chunk order (~1e-7, far below the int8-digest noise), so it's
  **opt-in** (default off keeps the anchor bit-identical).
- *Deferred from D5 (not landed).* **Optimizer-state CPU-offload** is superseded
  by W3d's 8-bit Adam (which cuts the moments on-GPU without per-step PCIe
  traffic). **Embedding row CPU-gather** is largely addressed by tying (D6) +
  (for private embed) the W3d de-dup; the PCIe-bound gather is a later follow-up,
  not needed to fit the 12 GB target alongside checkpointing + chunked CE.



**W3b status (activation checkpointing) + a correction to D4.**
- *Activation checkpointing landed (exact).* `diloco.activation_checkpoint`
  wraps each body block in `torch.utils.checkpoint` (`use_reentrant=False`, so it
  composes with `inner_autocast` and preserves RNG); it's inert outside training
  (no grad / eval). **Bit-exact** — training is identical with it on or off
  (tested) — so the launch config defaults it **on** for real runs while the core
  `DiLoCoConfig` default stays off (fast, byte-identical in-process anchor + unit
  tests).
- *Correction — D4 (private-copy de-dup) is NOT exact, reclassified.* The design
  assumed aliasing the worker's private modules from its bank (instead of
  deep-copying) was a free, bit-exact win. Implementation revealed it is **not**:
  the remote worker's `_train_path` **never writes trained private weights back
  to its bank** (it pushes them to the owner and re-fetches private only on a
  *cold* task). So deep-copy (re-train private from the cold-fetched base each
  warm task) and aliasing (accumulate private in-place across warm tasks) yield
  **different private trajectories** — a behavior/dynamics change, not an exact
  one. It is plausibly a private-*warming improvement*, but it must be validated
  like any dynamics change, so it moves to the **§0f-gated, off-by-default**
  bucket (W3d-adjacent), out of the exact-default-on set. (The *in-process*
  engine does copy private back via `_copy_into`, so there aliasing would be
  exact — but the worker is the memory target, and there it is not.)



**W3a status (VRAM profiler):** `src/opendipaco/train/memory.py` —
`vram_breakdown(config, batch_size, seq_len, ...)` counts a path's real
parameters on the **meta** device (so a model too big to allocate is still
profiled) and returns the per-round peak by term (params / global / Adam(2×) /
grads / activations + the `[tokens, vocab]` logits term), with flags modelling
each lever (`autocast`, `checkpoint`, `chunked_logits`). Parameter/optimizer
terms are exact; activations coarse. `measure_peak(...)` reports the **true**
peak around a real round on CUDA (`max_memory_allocated`), `None` off CUDA.
`examples/vram_budget.py` prints the breakdown, what each lever saves, and
fit-vs-budget — e.g. a ~540M path goes 19 GB → 10.8 GB (checkpointing, fits
12 GB) → 7.3 GB (8-bit Adam). The measured term shows the activation estimate is
a planning aid, not the truth.


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
  These change peak memory, not numerics, so the deterministic anchor stays
  bit-exact and they can ship **on** wherever they help.
- **Behavior/dynamics-changing — §0f-gated, off by default, on-box validated:**
  - **private-copy de-dup / warming** (D4) — aliasing the worker's private
    modules changes the warm-round private trajectory (the worker doesn't write
    trained private back to its bank), so it is a dynamics change, not the exact
    win first assumed; see the status note.
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

### D4. Private-copy de-dup — reclassified as a dynamics change (NOT exact)
Aliasing the worker's private modules (deep-copying only shared) would save ~`R`
(the dominant embed/head). It was first assumed bit-exact, but the worker's
`_train_path` never writes trained private back to its bank, so aliasing
(accumulate private in-place across warm tasks) differs from deep-copy (re-train
private from the cold-fetched base each warm task). That is a **behavior/dynamics
change** — likely a private-*warming improvement*, but it must ride §0f like any
dynamics change, so it is **off by default** and validated on-box, not shipped as
a free exact win. (In the *in-process* engine, the round copies private back via
`_copy_into`, so there aliasing is exact — but the worker is the memory target.)

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
| **W3b** | Activation checkpointing over the body (D3), exact + default-on for real runs. *(Private-copy de-dup, D4, moved to W3d — it turned out to be a dynamics change, not exact.)* | Training **bit-identical** with/without checkpointing; the flag flows from `diloco`; inert outside training; anchor unchanged. |
| **W3c** | Tied embed/head fixed through the working-copy deepcopy + chunked cross-entropy (D6). *(D5 CPU-offloads deferred: optimizer offload → superseded by W3d 8-bit Adam; embedding gather → a PCIe-bound follow-up, tying covers the embedding.)* | Tying survives `deepcopy` (one shared weight, half `R`) + stays tied through training; untied stays bit-identical; chunked CE matches the dense loss to fp tolerance and skips the full logits; chunked CE trains. |
| **W3d** | Dynamics-gated levers (off by default, §0f, on-box-validated): custom blockwise 8-bit AdamW (D7) **and** the private-copy de-dup / warming (D4). | 8-bit Adam round-trips within the blockwise bound; private-warming + `quant-optim` arms converge in `validate_dynamics`; peak `2P → ~0.5P`. |

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
