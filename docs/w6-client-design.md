# W6 — a consumer client (volunteer-grade `join`)

Status: **landed (a, b, c ✅).** All three slices are built and tested. Notable
deltas from this design, found while building: the bandwidth cap throttles the
worker's **PS sockets only** — the scheduler *control* socket is exempt so a
throttle sleep can't starve the heartbeat under the link lock (a low cap would
otherwise lose the lease), and bytes-mode shard download then rides that
unthrottled socket (use `data.ship: spec` for a fully-capped run); per-worker
tailoring adjusts the **uplink** (`compress`/`up_density`) only, not `down`
(delta-down needs the owner keyframe ring). libp2p/coordinator paths warn that
the cap isn't enforced rather than silently ignore it. Manifest trust is TOFU by
default (channel-dependent) with `--server-pub` pinning; operator-local paths and
the volunteer's own `max_mbps` are stripped.

Original design follows. The `opendipaco` CLI drives every role, but there is no
volunteer-grade client: today "consumer hardware" means "people who can write a
YAML config, know the model architecture, and open a port"
(`docs/viability-roadmap.md` W6). This adds a one-command **`opendipaco join`**
that a non-expert can run: autodetect the GPU, fetch what it needs from the
swarm, honor a bandwidth cap, survive a laptop sleeping, and show health.

Two operator decisions are settled (the rest are recommended defaults):
- **Join surface = flags.** `opendipaco join --tracker HOST:PORT --auth SECRET`
  (or `--scheduler`/`--identity`). No new invite artifact to issue.
- **Bandwidth cap = throttle + advertise.** A hard client-side byte throttle is
  the guaranteed ceiling; the worker also advertises its budget so the server
  sends less and the cap hurts less.

`schedule: central`/`sharded` worker behavior is unchanged — `join` is a new
front door onto the existing `run_worker_role`, not a new training path.

## 1. Goal and the one new requirement

```
opendipaco join --tracker tracker.example.com:9000 --auth <secret> [--max-mbps 5]
  -> autodetect device (cuda > mps > cpu) + check the path fits VRAM
  -> fetch the run manifest (model + diloco + run params) from the swarm
  -> connect as a worker, train, reconnect across drops/sleep
  -> print a health line every few seconds; SIGINT = graceful leave
```

**The one thing flags can't carry: the model architecture.** A worker builds its
`PathModel` from the model/diloco config; a volunteer with only `--tracker
--auth` doesn't have it. So `join` **fetches a signed run manifest** from the
swarm (D2). This is the keystone that makes a flags-only join possible; without
it, "flags" degenerates into "paste the whole model config on the command line."

## 2. Decisions

### D1. Join surface — flags (settled)

A new `join` subcommand, not a slim YAML. Required: `--tracker HOST:PORT` (or
`--scheduler HOST:PORT` for a static sharded run) and a credential (`--auth
SECRET` for HMAC, or `--identity KEY.pem` for per-peer Ed25519). Optional:
`--device`, `--max-mbps`, `--max-tasks`, `--server-pub` (manifest pinning, D2),
`--data-dir` (spec-materialization cache). Everything else is autodetected or
comes from the manifest. `join` builds a `LaunchConfig` in memory from the flags
+ manifest and calls `run_worker_role` — so the actual training loop is the
audited one, reached through a friendlier door.

### D2. Run-manifest fetch (the keystone)

The server publishes a **signed manifest** — the model `BackboneConfig`/
`DiPaCoConfig`, the `DiLoCoConfig`, and the worker-relevant run params
(`sequence_length`, `data.ship`/spec, `compress`/`down` defaults, `schedule.mode`,
`ownership` salt/k) — and the joining worker fetches it before building its
engine. Only public, already-on-the-wire values; **no secrets, no weights.**

- **Where it's served.** From whoever the volunteer already dials: the
  **scheduler** answers a `manifest` RPC (sharded/central); the **tracker**
  serves/relays it for rendezvous + decentralized (the tracker is the rendezvous
  point a flags-join already hits). The operator's server registers the manifest
  at startup (the scheduler holds it; for decentralized the owners/tracker do).
- **Trust.** The manifest is `sign_record`-signed by the run's identity. With
  `--server-pub` the worker pins that key (verify-or-refuse). Without it,
  **TOFU**: accept the first manifest and print its fingerprint loudly so the
  volunteer can verify out-of-band. A manifest only chooses *what the worker
  computes*, never what it trusts on the weight path (grants/quorum still gate
  that), so a wrong manifest wastes the volunteer's compute but can't poison the
  run — TOFU is an acceptable default, pinning is the careful one.

### D3. GPU autodetect + VRAM fit check

Resolve the device unless `--device` is given: `cuda` if available, else `mps`,
else `cpu`. Then run the W3 `vram_breakdown(config, batch_size, seq_len)` against
the detected device's free VRAM. If the path doesn't fit, **auto-enable the W3
fit levers in order** (`inner_autocast` → 8-bit Adam → `dedup_private`) and
re-check; if it still doesn't fit, fall back to CPU with a clear notice (or, with
`--device cuda` forced, refuse with the shortfall). The volunteer never computes
a VRAM budget by hand.

### D4. Bandwidth cap — throttle + advertise (settled)

- **Hard throttle (the ceiling).** A shared **token bucket** over all the
  worker's transport sockets enforces `--max-mbps` on bytes sent+received; when
  the bucket is empty, sends/receives block (natural back-pressure → tasks just
  take longer, never exceed the cap). Wraps the socket I/O seam
  (`wire.send_msg`/`recv_msg` / the connect helpers), so it is transport-level
  and feature-agnostic. Off (`None`) = today's behavior, byte-identical.
- **Advertise (make the cap hurt less).** The worker adds its budget +
  compression capability to the `capabilities` it already sends; the scheduler
  tailors *that worker's* task `compress`/`down`/`up_density` to fit, so less
  data hits the throttle. Server-side per-worker tailoring is the one server
  change; it defaults off and is a no-op when unset. **As built it tailors the
  uplink only** (`compress`/`up_density`) — `down` (delta) needs the owner
  keyframe ring and can't be toggled per worker.

  **Unvalidated §0f dynamics (why it's opt-in, stated plainly).** Tailoring makes
  a shared module's lossy encoding depend on *which* worker fed it, two effects
  no test yet covers: (1) workers push differently-quantized/sparsified
  pseudo-gradients to the same owner — mixed lossy levers per path, like the W2
  arms but *heterogeneous*; (2) the W2b error-feedback residual is **per-worker,
  per-path**, so when a path fails over from a capped (sparse) worker to an
  uncapped (dense) one, the capped worker's accumulated dropped-mass is **dropped,
  not compensated** — a small *biased* (worker-correlated, not zero-mean) error on
  shared modules. The audit still holds (the digest is the raw pre-compression
  delta), but *what lands in the bank* differs per worker. This rides the §0f run;
  the de-risking step is a `validate_dynamics` **het-compress** arm (half the
  workers tailored), analogous to the het-batch arm — owed, not yet built. Hence
  off by default.

### D5. Sleep / resume (laptop reality)

The sharded worker already reconnects across a dropped link (`reconnect=True`).
Add a **monotonic-gap detector**: a light heartbeat thread watches for a wall-
clock jump (the process was suspended — laptop lid). On a detected sleep it
**abandons any in-flight task** (its lease/grant is stale; submitting a
post-sleep commit would be a zombie write the token-fence would reject anyway)
and **rejoins fresh**. `SIGTERM`/`SIGINT` → graceful leave is already wired (W4);
`join` surfaces it as "leaving cleanly."

### D6. Health / contribution surface

A periodic status line (default every few seconds, to stderr): tasks committed +
accept rate, tokens/sec, bytes up/down and current Mbps (vs the cap), peers /
connection state, device + whether fit levers engaged. Emitted through one
`_HealthReporter` fed by counters the worker already has (task acks) plus the
D4 byte accounting — structured (a dict per tick) so a later GUI/daemon can
consume the same stream. `--quiet` silences it; `--json-status` emits JSONL.

### D7. Compatibility / scope

`run_worker_role` and the training loop are unchanged; `join` only assembles
config and wraps it. Explicitly out of scope (later/never): a desktop GUI
(CLI-only here); a contribution-reward economy (W8 incentives); the decentralized
multi-writer convergence (Phase 4 frontier). The manifest publish path reuses the
existing signed-record machinery; no new crypto.

## 3. Implementation slices

| Slice | Contents | Key tests |
|---|---|---|
| **a** | `opendipaco join` (D1): build config from flags, **fetch + verify the run manifest** (D2, server-side publish + worker-side fetch), GPU autodetect + VRAM fit check (D3), the health surface (D6), and sleep/resume (D5). Calls `run_worker_role`. | manifest round-trips (publish→fetch→verify; pinned-key refuse on mismatch, TOFU prints a fingerprint); device resolves cuda>mps>cpu and falls back when a path won't fit; a flags-only `join` against an in-process run trains to a task budget; a simulated monotonic gap abandons the in-flight task and rejoins. |
| **b** | Transport **byte throttle** (D4a): a shared token bucket on the worker's socket I/O honoring `--max-mbps`, with byte accounting feeding the health line. Off = byte-identical. | a capped worker's measured throughput stays ≤ the cap (+slack) while still completing tasks; bytes counted match what crossed the wire; `None` cap is a no-op. |
| **c** | **Advertise + server tailoring** (D4b): the worker advertises budget/compression in `capabilities`; the scheduler tailors that worker's task `compress`/`down`/`up_density` to fit. Defaults off. | the scheduler honors an advertised budget (a low-budget worker gets compressed tasks; an uncapped one is unchanged); byte-identical when unadvertised. |

Rough sizing: a M–L (the client core + manifest), b M, c M.

## 4. Open questions (recommendation first)

1. **Manifest trust default** — recommend **TOFU with a printed fingerprint**,
   `--server-pub` to pin. A wrong manifest can only waste the volunteer's compute
   (weights stay grant/quorum-gated), so refusing-by-default would add friction
   for little gain; print the fingerprint so a careful volunteer can verify.
2. **Throttle scope** — recommend **one aggregate token bucket** across all the
   worker's connections, so `--max-mbps` is a true total (not per-socket), which
   is what a consumer means by "cap my upload."
3. **Sleep detection cadence** — recommend the existing heartbeat thread carry
   the monotonic-gap check (no new thread); a gap > a few heartbeat intervals is
   a sleep. Revisit if false positives appear under heavy load.
