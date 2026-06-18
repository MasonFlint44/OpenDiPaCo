# W7 · Finish data decentralization

Status: **design** (slices a/b/c below land incrementally).

## The gap

Phase 0d (`data.ship: spec`) already removed corpus *bytes* from the wire: the
server ships a few-KB shard **recipe** and each worker `materialize_shard`s its
own shard from the public source (`src/opendipaco/data/spec.py`). But three
central data dependencies remain (per `docs/viability-roadmap.md` §W7):

1. **The operator still holds the whole corpus in RAM to fit the router.**
   `build_spec_corpus` (`launch/roles.py`) does
   `KMeansRouter(...).fit(feat([doc[:seq_len] for doc in docs]))` and then
   `SpecCorpus.from_documents(spec, docs)` for the α-weighting token counts —
   both require *every document resident*. So a "spec" run that ships no bytes
   still needs a node big enough to load the entire corpus at startup. The
   shipped centroids are also **unverifiable**: a worker trusts whatever the
   manifest/spec carries (a poisoned router could herd documents).

2. **The worker shard cache is unbounded in RAM.** `shard_cache: dict = {}`
   (`schedule/sharded.py`) keeps one materialized `[N, seq_len]` shard per
   *distinct path the worker ever leased*. A single-path worker holds one entry
   (fine), but a long-lived worker that fails over / is re-assigned across many
   paths accumulates them all. The disk cache (`materialize_shard(cache_dir=)`)
   bounds *re-streaming*, not *memory*.

3. **EM re-sharding is central** (`em.py`): the E-step scores every doc against
   every path and re-assigns to the lowest-loss path — it needs the full corpus
   *and* the full module bank on one node. **Out of scope for W7's eng slices:**
   it is not wired into the async/sharded loop at all (offline capability), and
   decentralizing it changes *global* assignments and needs a consensus
   mechanism — research-shaped, §0f-gated, owed alongside the WAN run. Recorded
   honestly in the roadmap, not attempted here.

W7 closes (1) and (2): after it, a `ship: spec` run needs **no node that holds
the whole corpus**, the shipped router is **peer-verifiable**, and a worker's
memory is **bounded** regardless of how many paths it churns through.

## Slices

### Slice a — Bound the worker shard cache (RAM) — **landed**

Replace the unbounded `shard_cache` dict with a small LRU keyed by path, capped
by entry count (default a handful; one is the common case). On eviction we drop
the tensor; the next lease for that path re-materializes from the spec (cheap if
the on-disk cache is warm) or reloads the shipped bytes. Training is
**byte-identical** — eviction only changes *when* a shard is rebuilt, never its
contents. Cap is a worker-local knob (`run`/`join`-side), library default
conservative.

- `schedule/sharded.py` + `distributed.py`: a `_ShardCache` (OrderedDict LRU)
  swapped in for the dict; `cached_shards` advertisement reports current keys.
- `launch`: `join --max-shards` / config knob; conservative library default.
- Test: eviction re-materializes byte-identically; cap respected; one-path
  worker never evicts.

### Slice b — Bounded-memory, streaming router fit + token counts — **landed**

The operator must build the spec corpus **without holding the whole corpus**.

- `data/spec.py`: `fit_routing_from_source(spec_source, num_paths, ...,
  sample=N)` streams the public source, featurizes a **bounded deterministic
  prefix sample** (first N docs / reservoir at fixed seed), fits k-means, and
  returns the `kmeans_routing(...)` dict. Deterministic in `(source, sample,
  seed)` → reproducible anywhere.
- `SpecCorpus.build` already streams counts (no doc retention); route
  `build_spec_corpus` through `fit_routing_from_source` + `SpecCorpus.build` so
  the operator never materializes the full doc list.
- Bit-for-bit caveat: fitting on a *sample* changes the centroids vs. fitting on
  all docs, so the resulting assignments differ from today's `ship: spec` run.
  This is a **dynamics-adjacent** change (different shards) → gated behind a
  `data.router_sample` knob; when unset the old "fit on docs in hand" path is
  kept verbatim (byte-identical). Sampled fit is the decentralized default only
  for the `join`/launch spec path, not retrofitted onto byte mode.
- Test: same sample+seed → identical centroids on two independent builds;
  unset knob → byte-identical to today.

### Slice c — Peer-side routing verification

Make the shipped router **checkable** so a worker need not trust it blindly.

- `data/spec.py`: `verify_routing(spec, *, source=None, tokenizer=None,
  atol=...)` re-runs the slice-b deterministic fit from the spec's own
  (public) source + sample params and compares centroids to the shipped ones;
  returns ok/mismatch. A mismatch means the manifest's router was tampered with
  or built from a different corpus.
- `launch/client.py` (`join`): after fetching the manifest, optionally
  `--verify-routing` re-fits and **refuses to train** on mismatch (off by
  default — re-streaming the sample costs bandwidth; opt-in for the paranoid,
  surfaced in the health line either way). Wrong router wastes compute, can't
  poison weights (grant/quorum-gated), so this is belt-and-suspenders, not a
  security boundary on its own.
- Test: untampered spec verifies; centroids perturbed → mismatch detected;
  round-robin spec verifies trivially (no centroids).

## Non-goals / owed

- **Decentralized EM re-sharding** — §0f/research, see gap (3). Tracked in
  `docs/remaining-gaps.md` and the roadmap.
- **Worker-reported token counts under heterogeneous sources** — counts are
  deterministic from the (public) spec, so worker- vs operator-derived counts
  are identical today; only matters if peers see different source bytes, which
  is a W8 trust question, not a W7 eng one.
