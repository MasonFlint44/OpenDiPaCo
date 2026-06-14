# W2 design — bandwidth: delta encoding + sparsification + sub-int8

Status: **W2a + W2b + W2c landed; inner_steps docs pending.** W2 (from [viability-roadmap.md](viability-roadmap.md))

**W2c status (sub-int8: int4 + per-group scale):**
- *int4 is a value encoding that stacks on every W2 path.* `compress: int4`
  adds a fourth mode: symmetric int4 (`[-7, 7]`) with a **per-group scale** (one
  per 128 elements -- a single per-tensor scale is far too lossy at 4 bits), two
  nibbles packed per byte (`_quantize_int4`/`_dequant_int4`, ~0.5 B/elem +
  scales). It encodes the **up pseudo-gradient** (dense and sparse-kept values)
  and the **down delta** (an int4 owner ships int4 deltas); weights still ship
  bf16 (raw weights at 4 bits are hopeless). Self-describing `{"q4","s","g","n",
  "shape"}` payloads, error feedback throughout, device-safe (no CPU/GPU mix).
- *Byzantine-hardened decode.* `_dequant_int4` validates the packed/scale lengths
  and group size, raising a caught `ValueError` (rejected push) instead of an
  uncaught reshape crash -- same boundary as the int8/sparse paths.
- *Off by default; validated.* `validate_dynamics.py` gained `int4` and a fully
  `W2 stacked` (int4 + delta-down + sparse-up) arm; both converge (~0.7× the
  anchor at toy scale). WAN §0f stays the final verdict.



**W2b status (structured sparsification, up path):**
- *Top-k sparsification works end to end.* `up_density` (∈ (0, 1], 1.0 = dense =
  byte-identical) is stamped on tasks; the worker keeps each pseudo-gradient's
  top `up_density` fraction — **per output-row** for a 2-D weight, flat
  otherwise (`compress._sparsify`) — encodes the kept values via the existing
  `compress` mode, and **error-feeds the dropped mass** through the same residual
  carry (so a `current − reconstruction` residual captures dropped entries +
  any value-quant error; tested: exact mass conservation in `none` mode). The
  payload is self-describing (`{"sp": shape, "i": int32 indices, "v": values}`),
  so
  `maybe_dequantize` scatters it back to dense on the owner and a malformed one
  is refused. Composes with int8 values (a step toward W2c). `validate_dynamics.py`
  gained a `sparse-up` arm (~0.7× the anchor at toy scale — converges).
- *Off by default; sharded only* — `up_density < 1.0` is rejected in coordinator
  mode (it is stamped on sharded tasks), like delta-down.
- *Honest byte tradeoff.* Each kept entry costs an int32 index + the encoded
  value (≈ 5 B with int8 values), vs ~1 B/param for a dense int8 ship — so
  sparsification is a net win only at **low density** (≈ ρ < 0.2 with int8
  values; a bigger win at the ρ ≈ 0.01–0.05 typical of gradient sparsification).
  A future refinement (W2c-era) could shrink the index further (per-row int16, or
  a bitmask) to raise that break-even density. The tests use larger ρ for clear
  assertions, not as a recommended operating point.



**W2a status (delta-down):**
- *Delta-down works end to end.* `down="delta"` (scheduler + owners; default
  `full` is byte-identical): the owner ships `current − keyframe` int8 against the
  exact bytes the worker holds (`_down_payload_locked`, reusing the version ring
  `_history`/`_pinned_state_locked`), falling back to a **full** ship (a new
  keyframe) when the worker's keyframe ages out of the ring. The worker keeps one
  keyframe baseline per shared key, sends the **keyframe** version in `have`, and
  reconstructs `keyframe + dequant(delta)` (`compress.encode_state_delta`/
  `apply_state_delta`); its trained-against version (the push `base`) stays the
  nominal current, so staleness is unaffected. Payloads are self-describing
  (`{"__delta__","base","tensors"}`), so a full-mode owner and a delta-mode worker
  interoperate. `examples/validate_dynamics.py` gained a `delta-down` arm (lands
  ~0.8× the anchor at toy scale — converges).
- *Deviation from D4 — owner-side error feedback is NOT carried (yet).* The design
  imagined folding the down-quant residual into the next delta. But an owner
  serves many workers at **different keyframes**, so a single per-key residual is
  incoherent (it would mix recipients), and a per-(key, keyframe, recipient)
  residual is heavy and short-lived. Since the keyframe scheme already bounds the
  error to a **single, non-accumulating** int8 step (D2), W2a ships without
  owner-side error feedback; revisit only if the §0f run shows the within-window
  error matters. The worker-side *up-path* error feedback is unchanged.


removes the second big *practical* wall to consumer-hardware training:
**bandwidth**. Phase 0c got ~2× down / 4× up with `compress.py` (bf16 weights
down, int8 pseudo-gradients up with error feedback), but the "ship only changed
weights" cache is **structurally defeated in async mode**: every accepted
contribution bumps a shared module's `(epoch, counter)` version, so the worker's
`have` is almost always stale and the owner re-ships the *full* bf16 weights
(`_fetch`, `sharded.py`). For a 150M-param path that is ~300 MB down + ~150 MB up
per round (~10 min/round on a 20 Mbps uplink); for a *large* model over
asymmetric consumer uplinks it is fatal.

Three operator calls (§5) fixed the approach: **delta-down first** (the
structurally-defeated cache, the biggest and lowest-risk win), with
sparsification and sub-int8 staged behind it; **on-box validation, shipped
off-by-default** (extend `examples/validate_dynamics.py`; the WAN §0f run stays
the final convergence verdict); and **owner-side version history with a full
fallback** for the delta baseline.

## 1. Goal and the structurally-defeated cache

**Goal.** Cut per-round bytes — especially **down-traffic** (weights), the worse
offender on asymmetric consumer links and the one the cache should have solved —
enough that a large path is viable on a real home uplink. Three levers, staged by
dynamics risk:

- **W2a — delta-down** (low risk): ship `current − held` instead of `current`,
  quantized; deltas have far smaller dynamic range than raw weights, so int8 (or
  less) captures them at *pseudo-gradient* quality — which is exactly why raw
  weights stay bf16 today (`compress.py`: "quantizing raw weights to int8 is far
  lossier"). The delta makes a cheaper down-encoding *convergent*.
- **W2b — structured sparsification** (dynamics): send only the largest-magnitude
  pseudo-gradient entries (top-k / per-row), error-feeding the rest. Drops
  information per round; rides §0f.
- **W2c — sub-int8 quant** (dynamics): int4 / per-group quantization for the
  down deltas and the up pseudo-gradients.

Plus the **free lever**: DiLoCo's `inner_steps` trades local compute for sync
frequency — more inner steps ⇒ fewer rounds ⇒ proportionally less traffic, no new
code and no dynamics-validation debt. W2 documents it as the first thing to turn
before paying any compression-precision cost.

## 2. Dynamics model — the anchor, and why delta-down is the safe one

`compress="none"` stays **bit-identical** to today (the deterministic anchor),
and so does today's `bf16`/`int8` behavior — every W2 lever is **additive and
off by default**, negotiated by self-describing payloads (a delta/sparse payload
is detected by shape, like int8's `{"q","s"}` is today) so a receiver needs no
configuration and an un-upgraded peer is unaffected.

Every W2 lever **changes numerics** (the worker trains from a reconstructed-
lossy down-weight; the outer step sees a sparser/coarser up-delta), so each is a
§0f-class change. Per the operator call, the convergence verdict is split:

- **on-box now**: `examples/validate_dynamics.py` gains a compression arm — the
  synchronous anchor vs. the async sharded path *with each lever on*, on one
  corpus, checking the deltas converge comparably (the same shape as the
  robustness/int8 arms already there);
- **WAN §0f (deferred)**: the real multi-node verdict, as for every other
  dynamics feature.

**Why delta-down is the lowest-risk lever, made precise.** A delta sent at the
*same* precision as today's full ship is not automatically cheaper (a dense
small outer step changes nearly every bf16 value), so the win is **lossy**: the
delta is quantized more aggressively than the weights could be. The error is
bounded and non-accumulating by construction (§D2 keyframes), and unlike
sparsification it discards *no* coordinate — every weight still moves, just at
coarser precision for one round. That is a milder dynamics perturbation than
dropping coordinates outright, which is why it leads.

## 3. Shape of the result

```
  DOWN (owner -> worker), per shared key:
    worker fetch: have = {key: keyframe_version it holds EXACTLY}
    owner:
      held in version ring?  -> ship  delta = quantize(current_exact - held_exact)   (W2a)
                                       + optional top-k structure                      (W2b/W2c)
      else (aged out / policy)-> ship  full  (a new *keyframe*; resets drift)
    worker: current ~= keyframe + dequantize(delta)   (reconstruct; keep keyframe as baseline)

  UP (worker -> owner), per shared key (already int8+error-feedback today):
    + structured sparsification (top-k, error-fed)     (W2b)
    + sub-int8 (int4 / per-group)                       (W2c)
```

Everything above the encoding — the `have`/version protocol, `_fetch`/push
dispatch, the owner version ring (`version_history`), the worker warm caches,
error feedback (`compress.py`) — is reused; only the bytes on the heavy payloads
change.

## 4. Decisions

### D1. Delta-down against an owner version ring, with a full fallback
The owner already retains recent versions for redundant-execution checks
(`version_history`, `hist[version] = state`). W2a generalizes that into a small
**ring of exact weight snapshots** per owned key. On fetch, if the worker's
`have[key]` version is in the ring, the owner ships `current − held`; otherwise
it ships the **full** weights (a keyframe). The ring size bounds owner memory and
defines how far a slow/absent worker can fall behind before paying a full ship.
`compress="none"` and a worker that sends no `have` always get a full ship
(anchor preserved).

### D2. Keyframes: deltas are non-chained, so quantization error can't accumulate
A quantized delta reconstructs the target only approximately, so **chaining**
deltas (delta-from-last-reconstruction) would let error drift unboundedly across
rounds. W2a avoids this: a worker's **baseline is a keyframe** (the last *full*
weights it received, held exactly), and every delta is computed **against that
keyframe's exact snapshot** (`current_exact − keyframe_exact`), not against the
worker's lossy reconstruction. The worker reconstructs `keyframe + dequant(delta)`
and **keeps the keyframe** (not the reconstruction) as its baseline. Error is
therefore a *single* quantization step relative to the keyframe — bounded, not
accumulating. A **keyframe interval** (refresh after K versions, or when the
delta's norm relative to the keyframe exceeds a bound) caps drift and the per-
delta size; ageing the keyframe out of the ring forces a refresh (D1 fallback).

### D3. Worker baseline cache — one clean copy, memory-aware (W3 interaction)
Delta-decode needs the worker to hold its keyframe **exactly**, separate from the
trained (local) module weights — one extra path-sized copy per resident key.
This collides with W3 (consumer-VRAM fit), so the baseline is held **off the
training device** (CPU) and at the **wire precision** (bf16), not fp32 on GPU;
reconstruction casts on apply (`load_state_dict` already casts). The cache is
keyed by `(key, keyframe_version)` and dropped when the key leaves the path's
resident set. Net worker cost: ~one bf16 copy of the path in host RAM.

### D4. Delta encoding reuses the pseudo-gradient quantizer + error feedback
A delta is "a small-magnitude tensor list," exactly what the int8 quantizer
already handles (symmetric int8, `{"q","s"}`). W2a reuses it via
`encode_state_delta`/`apply_state_delta`; payloads stay self-describing
(`{"__delta__","base","tensors"}`) and `apply_state_delta` refuses malformed
input as today. *(Owner-side error feedback was designed here but is **not**
carried in W2a — an owner serves many workers at different keyframes, so a single
per-key residual is incoherent; the keyframe scheme already bounds the error to a
non-accumulating single step. See the W2a status note above.)*

### D5. Down-compression becomes a negotiated policy, additive to `compress`
`compress` (`none|bf16|int8`) keeps its current meaning (it governs the *up*
path and the *full* down ship). Delta-down is a **separate policy** the
owner/scheduler stamps — e.g. `down="full"|"delta"` with `keyframe_interval` and
ring size — so the two dimensions compose (`compress="int8"` up, `down="delta"`
down) and the default (`down="full"`) is byte-identical to today. Workers follow
server policy and the payloads are self-describing, so mixed-version swarms
interoperate.

### D6. Structured sparsification of pseudo-gradients (W2b)
The up path already ships int8+error-feedback. W2b adds **structured top-k**: per
tensor (or per output-row for 2-D weights), keep the largest-magnitude entries,
send `(indices, values)`, and **error-feed the dropped mass** into the next
round's delta (the residual machinery already exists). Structure (per-row k, or
block-sparse) keeps the index overhead and the kernel cost low and the on-wire
format simple. Off by default; on-box validated; the headline up-path win for
large modules where the pseudo-gradient is compressible.

### D7. Sub-int8 quantization (W2c)
int4 (and a per-group scale, since a single per-tensor scale gets lossy below 8
bits) for the down **delta** and the up **pseudo-gradient**. Two int4 values pack
per byte; per-group scales (e.g. one per 64/128 elements) keep error bounded.
Error feedback throughout. Applies to both the delta (D4) and the sparsified
values (D6), so the levers stack. Most dynamics-aggressive ⇒ last, off by
default, on-box + §0f validated.

### D8. The free lever: `inner_steps` is documented first
More inner AdamW steps between syncs ⇒ fewer syncs ⇒ proportionally less traffic,
with **no precision cost and no new dynamics risk** (it is already a tuned
DiLoCo knob, exercised by the paper and our runs). W2 documents the
bytes-per-token-trained tradeoff and recommends raising `inner_steps` *before*
turning on any lossy compression — the cheapest byte you don't send is the sync
you don't do.

### D9. Compatibility and the deterministic anchor
`compress="none"`, `down="full"`, and the existing `bf16`/`int8` paths stay
**bit-identical**. Every lever is additive, off by default, self-describing on
the wire, and reuses the error-feedback machinery. The replication/recovery
invariant is untouched: **replication pulls never compress** (a version must
identify identical bytes across replicas — `include_state` ships exact
uncompressed state, as today); delta/sparsify/sub-int8 apply only to the
worker-facing fetch/push data plane, never to owner↔owner state transfer.

### D10. Explicitly deferred / out of scope
- **WAN §0f convergence verdict** — the real multi-node proof for every lever;
  on-box `validate_dynamics.py` de-risks but does not replace it.
- **Gradient compression *theory* tuning** (optimal k, group size, error-feedback
  variants) beyond sane defaults — revisit once the §0f run gives real numbers.
- **Compressing replication/gossip** — stays exact (D9 invariant).
- **Entropy coding** (zstd over the quantized stream) — a later, orthogonal,
  lossless win; not in W2.

## 5. Implementation slices

Each lands green on its own; the low-risk structural win (delta-down) comes
first, the dynamics-aggressive levers after, each behind its own off-by-default
policy and on-box validation arm.

| Slice | Contents | Key tests |
|---|---|---|
| **W2a** | Delta-down (D1–D5): owner version ring + keyframe/full fallback; owner-side delta quantize + error feedback; worker keyframe baseline cache (CPU/bf16) + reconstruct; `down` policy negotiated, self-describing payloads; `compress="none"`/`down="full"` byte-identical. | Reconstruct == full-ship target within the quant bound; keyframe refresh resets drift (no accumulation over many versions); aged-out `have` → full fallback; replication pulls still exact; anchor byte-identical; a `validate_dynamics.py` delta arm converges. |
| **W2b** | Structured sparsification of up pseudo-gradients (D6): per-row top-k + error-fed dropped mass; self-describing `(idx, val)` payload; off by default. | Error feedback conserves total mass over rounds (no systematic bias); a sparsified run converges in the on-box arm; malformed sparse payload refused. |
| **W2c** | Sub-int8 (D7): int4 + per-group scale for the down delta and up pseudo-grad, packed; error feedback; stacks on W2a/W2b. | int4 round-trip within the per-group bound; packed size ≈ 0.5 B/elem + scales; on-box arm converges; refuse malformed. |
| **docs** | The `inner_steps` bytes-per-round lever (D8): a roadmap/README note + a worked example of the traffic/round tradeoff. | n/a (doc); a sanity example in `examples/`. |

Rough sizing: W2a M–L (the ring + keyframe + baseline cache is the substance),
W2b M, W2c M. L overall — smaller than W1, since there is no new transport or
external dependency, only encodings over existing payloads.

## 6. Operator decisions (resolved)
1. **Sequencing (D1/§1): delta-down first**, then structured sparsification, then
   sub-int8 — the structurally-defeated cache is the biggest, lowest-risk win, and
   staging keeps a convergence regression bisectable. `inner_steps` is documented
   as the free first lever.
2. **Dynamics-risk posture (§2): on-box validate, ship off-by-default.** Extend
   `examples/validate_dynamics.py` with a compression arm; ship every lever off by
   default like robustness/decentralized; the WAN §0f run stays the final verdict.
3. **Delta baseline (D1–D3): owner-side version ring with a full fallback**, and a
   single CPU/bf16 worker baseline copy — bounded owner memory, graceful
   degradation when a worker falls too far behind, and W3-VRAM-aware on the worker.
