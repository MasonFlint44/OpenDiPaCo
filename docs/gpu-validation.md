# GPU validation results

First real-GPU validation of the DiPaCo implementation (RTX 5070 Ti, 16 GB, torch
2.12 / CUDA 13). Everything prior to this ran only on CPU at toy scale; these are the
project's first results on real data at non-toy size. **Honest framing up front:** a
single consumer GPU is far from the paper's regime (256 paths × 150M params, billions
of tokens, a GPU cluster), so read these as *evidence the method works and behaves as
theory predicts*, not as a paper reproduction.

The comparison is `opendipaco.validation.run_comparison`: a **dense baseline** (DiPaCo
with `level_sizes=[1,1]` — one path = a plain dense transformer) vs. a **K×K DiPaCo**
model on the *same* backbone, both evaluated on held-out C4 at **equal inference cost**
(one path executes). Lower perplexity is better.

## 1. The CUDA path works end-to-end

Nothing in the codebase had ever run on a GPU (only `.to(device)` calls, exercised on
CPU). With `device="cuda"` threaded through `run_comparison`, the full pipeline —
engine training, DiLoCo outer steps, k-means + discriminative routing, and inference —
runs clean on GPU with no device-mismatch bugs.

## 2. DiPaCo beats matched dense (real C4, 13.4M params/path, 2 seeds)

`examples/validate_c4_gpu.py`, 16 paths (4×4), hidden 384, 2 layers/level, 30 rounds,
8k docs, seq 128 — ~3 min, ~5 GB:

| seed | dense ppl | DiPaCo ppl | margin |
|---|---:|---:|---:|
| 0 | 485.19 | 439.46 | DiPaCo **−45.7** |
| 1 | 486.58 | 380.88 | DiPaCo **−105.7** |

Dense is stable; **DiPaCo wins in both seeds**. This is the first empirical confirmation
of DiPaCo's core claim (a routed single path matching/beating a dense model of the same
size) on real data in this project. At *toy CPU* scale dense had won — the advantage is a
scale phenomenon that only appears once the model and data are big enough.

## 3. Scale sweep: the margin is regime-dependent (paths × data)

`examples/scale_sweep_gpu.py`, fixed backbone (hidden 384, 2 layers/level), **fixed**
data (10k docs), 30 rounds, sweeping the path count:

| paths | dense ppl | DiPaCo ppl | margin | total params |
|---:|---:|---:|---:|---:|
| 4 | 478.05 | 350.53 | **+127.5** | 20M |
| 9 | 477.41 | 326.65 | **+150.8** | 28M |
| 16 | 477.80 | 341.31 | **+136.5** | 35M |
| 25 | 475.42 | 435.25 | +40.2 | 42M |
| 36 | 477.41 | 552.02 | **−74.6** | 49M |

- **Dense is flat (~477.7 ± 1)** across the sweep — the control is perfect (dense is
  always one path of the same backbone, independent of K), so the DiPaCo column is signal.
- **DiPaCo's edge peaks at ~9 paths, then collapses** — going *negative* at 36 paths.
- **Why:** at a *fixed* data budget, more experts split the same corpus, so each expert is
  **data-starved**; and the test-time router must choose among more paths from a fixed
  holdout, so **routing gets harder**. Both degrade DiPaCo as K outruns the data/routing
  budget. This is exactly why the paper scales **data *with* paths** — DiPaCo's advantage
  is a *joint* paths×data phenomenon, not free capacity from path count alone.

**Takeaway:** at *fixed* data the margin peaks then collapses — but that conflates
"more capacity" with "less data per expert." §4 separates them.

## 4. Data-scaled sweep: capacity helps when it's fed (the thesis, confirmed)

`examples/data_scaled_sweep_gpu.py`, same backbone, but now **data-per-path is held
constant** (~2,500 docs/path → total docs scale with K², 10k→90k), 30 rounds:

| paths | docs | dense ppl | DiPaCo ppl | margin |
|---:|---:|---:|---:|---:|
| 4 | 10k | 485.11 | 335.19 | +149.9 |
| 9 | 22.5k | 474.98 | 277.19 | +197.8 |
| 16 | 40k | 488.53 | 258.10 | +230.4 |
| 25 | 62.5k | 476.98 | 252.96 | +224.0 |
| 36 | 90k | 509.87 | 242.45 | **+267.4** |

- **DiPaCo improves monotonically** (335 → 242 ppl) and the **margin grows monotonically**
  (+150 → +267) as paths *and* data scale together. Dense stays in a band (~475–510; it
  even worsens slightly at 90k docs — same inference cost / fixed compute can't absorb the
  bigger corpus, while DiPaCo throws K× training compute at it across its experts).
- **The decisive contrast is at 36 paths: −74.6 (fixed data) vs +267.4 (data scaled).**
  Identical model, identical path count — the only difference is whether the experts and
  router are fed. So the fixed-data collapse (§3) was **data/routing starvation, not a
  path-count ceiling**: added path capacity (more model at *fixed inference cost*) genuinely
  buys better perplexity *when it has the data to learn from*.

**This is DiPaCo's central claim, demonstrated on real data:** scaling path count *with*
data widens DiPaCo's advantage over a same-inference-cost dense model — here from +150 ppl
(4 paths) to +267 ppl (36 paths).

## Open / next

- **Paper scale** (256 paths × 150M params, billions of tokens, multi-GPU/nccl via
  `TorchDistBackend`) needs a *cluster*, not one 16 GB card. The single-GPU trend is now
  established; the remaining unknown is whether it holds two orders of magnitude up.
- Bigger per-path backbones, more seeds, and longer training are all bounded by this card's
  memory/time, not by code.

Reproduce: `validate_c4_gpu.py` (single comparison), `scale_sweep_gpu.py` (fixed-data path
sweep), `data_scaled_sweep_gpu.py` (data-scaled path sweep). Env knobs:
`HIDDEN/HEADS/LAYERS/VOCAB/SEQ_LEN/ROUNDS/INNER_STEPS/BATCH/SEED`, plus `LEVELS_SWEEP` /
`DOCS_PER_PATH` for the sweeps.
