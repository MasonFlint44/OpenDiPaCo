"""W6 consumer client — ``opendipaco join`` (design ``docs/w6-client-design.md``).

A volunteer-grade front door onto the existing ``run_worker_role``: given only a
``--scheduler``/``--tracker`` address and a credential, it fetches the run
manifest from the swarm (``manifest.py``), autodetects the GPU and checks the
path fits VRAM, builds a :class:`LaunchConfig` in memory, and trains — printing a
periodic health line and surviving a laptop sleeping (the existing reconnect +
lease-token fence handle the correctness; the watchdog just surfaces it). The
actual training loop is the audited one; this only assembles config and wraps it.
"""

from __future__ import annotations

import threading
import time

import torch

from .config import dipaco_config
from .manifest import manifest_fingerprint, manifest_to_config, verify_manifest
from .roles import run_worker_role
from ..schedule import PeerIdentity
from ..schedule.throttle import TokenBucket, rate_from_mbps
from ..schedule.tracker import tracker_rpc


def _split_addr(addr) -> tuple[str, int]:
    """``"host:port"`` (or a ``(host, port)`` pair) -> ``(host, int(port))``."""
    if isinstance(addr, (tuple, list)):
        return str(addr[0]), int(addr[1])
    host, _, port = str(addr).rpartition(":")
    if not host or not port:
        raise ValueError(f"address must be host:port, got {addr!r}")
    return host, int(port)


def resolve_device(config, *, requested=None, batch_size: int, seq_len: int):
    """Resolve and fit-check the training device (D3).

    Returns ``(device, diloco_overrides, notes)``. Autodetects ``cuda > mps >
    cpu`` unless ``requested`` is an explicit device; on CUDA, estimates the
    per-round VRAM peak (W3 ``vram_breakdown``) against free memory and engages
    the W3 fit levers in order (autocast -> activation checkpoint -> chunked
    logits) before falling back to CPU. ``requested="cuda"`` forced past a
    shortfall raises rather than silently downgrading."""
    def _mps_available():
        return (getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available())

    forced = requested not in (None, "auto")
    if forced:
        dev = requested
        # Reject a device we don't recognize (a typo like "gpu", or "CUDA") early
        # rather than hand torch a bad string mid-training. cuda:N / mps:N are fine.
        if not (dev == "cpu" or dev.startswith("cuda") or dev.startswith("mps")):
            raise SystemExit(f"--device {dev!r} not recognized (use cpu, cuda[:N], or mps)")
        # A forced accelerator that isn't present must fail early + clearly, not
        # be accepted and blow up mid-training (the fit-check below only runs when
        # the device is actually available, so it would otherwise slip through).
        if dev.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA device is available "
                             "(use --device cpu, or drop --device to autodetect)")
        if dev.startswith("mps") and not _mps_available():
            raise SystemExit("--device mps requested but MPS is unavailable "
                             "(use --device cpu, or drop --device to autodetect)")
    elif torch.cuda.is_available():
        dev = "cuda"
    elif _mps_available():
        dev = "mps"
    else:
        dev = "cpu"
    notes = [f"using {dev}" + ("" if forced or dev == "cpu" else " (autodetected)")]
    diloco_overrides: dict = {}
    if dev.startswith("cuda") and torch.cuda.is_available():
        from ..train.memory import fits, vram_breakdown
        # mem_get_info can raise on a quirky build/driver (no current device, MIG,
        # ROCm) even when is_available() is True. The whole point of the ladder is
        # to degrade, not crash -- so a probe failure falls back to cpu (unless the
        # user forced cuda, in which case surface the real error).
        try:
            free_total = torch.cuda.mem_get_info()[0]
        except Exception as e:  # noqa: BLE001
            if forced:
                raise SystemExit(f"--device {dev} but CUDA memory probe failed: {e}") from e
            return "cpu", {}, notes + [f"CUDA memory probe failed ({e}); using cpu"]
        # Leave headroom: the activation term is a coarse estimate and a consumer
        # GPU also feeds a desktop, so budgeting 100% of free OOMs in practice.
        free = int(free_total * 0.9)
        # Engage fit levers in increasing-cost order; stop at the first that fits.
        ladder = [
            ({}, "no levers"),
            ({"inner_autocast": True}, "bf16 inner loop"),
            ({"inner_autocast": True, "activation_checkpoint": True}, "+ activation checkpoint"),
            ({"inner_autocast": True, "activation_checkpoint": True, "loss_chunks": 4},
             "+ chunked logits"),
        ]
        for over, label in ladder:
            bd = vram_breakdown(config, batch_size=batch_size, seq_len=seq_len,
                                autocast=over.get("inner_autocast", False),
                                checkpoint=over.get("activation_checkpoint", False),
                                chunked_logits=over.get("loss_chunks", 1) > 1)
            if fits(bd, free):
                diloco_overrides = over
                if over:
                    notes.append(f"VRAM fit: enabled {label} (~{bd['total'] / 1e9:.1f} GB "
                                 f"vs {free / 1e9:.1f} GB budget)")
                break
        else:
            need = vram_breakdown(config, batch_size=batch_size, seq_len=seq_len,
                                  autocast=True, checkpoint=True, chunked_logits=True)["total"]
            if forced:
                raise SystemExit(
                    f"path won't fit {dev}: needs ~{need / 1e9:.1f} GB even with all fit "
                    f"levers, only ~{free / 1e9:.1f} GB usable. Use a smaller batch or "
                    f"--device cpu.")
            dev = "cpu"
            notes.append(f"VRAM too small (~{need / 1e9:.1f} GB needed, ~{free / 1e9:.1f} GB "
                         f"usable) -> falling back to cpu")
    return dev, diloco_overrides, notes


def fetch_manifest(addr, *, auth_key=None, tls=None, timeout: float = 10.0) -> dict:
    """Fetch the run manifest from whoever the volunteer dials (scheduler or
    tracker). Raises if the server published none (operator hasn't enabled it)."""
    host, port = _split_addr(addr)
    reply = tracker_rpc((host, port), {"type": "manifest"}, auth_key=auth_key, tls=tls,
                        timeout=timeout)
    manifest = (reply or {}).get("manifest")
    if manifest is None:
        raise SystemExit(
            f"{host}:{port} published no run manifest -- the operator must enable it "
            f"(serve_manifest); or join with a config file via `opendipaco worker`.")
    return manifest


class HealthReporter:
    """Periodic one-line health surface (D6), to stderr, plus a **sleep detector**
    (D5): a *wall-clock* jump between ticks means the process was suspended (laptop
    lid). It must use ``time.time()``, not ``time.monotonic()`` -- on Linux the
    monotonic clock does **not** advance across a suspend, so a monotonic gap would
    never fire. Lean for now -- device / target / uptime / sleep events; rich
    per-task metrics (tasks, accept rate, tokens/sec, Mbps) ride a worker progress
    hook + the W6b byte accounting."""

    def __init__(self, *, target, device, interval: float = 10.0, quiet: bool = False,
                 bucket=None, cap_mbps=None):
        self.target, self.device, self.interval, self.quiet = target, device, interval, quiet
        self.bucket, self.cap_mbps = bucket, cap_mbps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.quiet:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _bandwidth_note(self, dt: float, prev: tuple[int, int]) -> tuple[str, tuple[int, int]]:
        """Current Mbps (over the last tick) + cumulative MB, from the bucket's
        byte counters. Returns ``(note, new_prev_counters)``."""
        if self.bucket is None or dt <= 0:
            return "", prev
        sent, recv = self.bucket.sent_bytes, self.bucket.recv_bytes
        mbps = ((sent + recv) - (prev[0] + prev[1])) * 8 / 1e6 / dt
        cap = f"/{self.cap_mbps:g}" if self.cap_mbps else ""
        return (f"  {mbps:.2f}{cap} Mbps  ({(sent + recv) / 1e6:.1f} MB total)",
                (sent, recv))

    def _loop(self) -> None:
        # Wall clock (time.time): it jumps across an OS suspend, whereas
        # time.monotonic does not advance during one (so it can't see the sleep).
        start = last = time.time()
        prev = (self.bucket.sent_bytes, self.bucket.recv_bytes) if self.bucket else (0, 0)
        # A wall gap much larger than the tick is a suspend, not a slow tick.
        gap_threshold = max(self.interval * 3, 30.0)
        while not self._stop.wait(self.interval):
            now = time.time()
            if now - last > gap_threshold:
                print(f"[join] resumed after ~{now - last:.0f}s suspended; reconnecting "
                      f"(stale work is refused by the lease fence)", flush=True)
            bw, prev = self._bandwidth_note(now - last, prev)
            last = now
            print(f"[join] training on {self.device} -> {self.target}  "
                  f"up {now - start:.0f}s{bw}", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


def run_join(*, scheduler=None, tracker=None, auth_key=None, identity_key=None,
             device=None, max_tasks=None, server_pub=None, data_dir=None, max_mbps=None,
             quiet: bool = False, tls=None, stop_event=None) -> None:
    """Join a run from connection flags + the fetched manifest (W6 slice a).

    Exactly one of ``scheduler`` / ``tracker`` is the dial target (and the
    manifest source). ``server_pub`` pins the manifest signer; without it the
    manifest is accepted TOFU with its fingerprint printed."""
    if (scheduler is None) == (tracker is None):
        raise SystemExit("join needs exactly one of --scheduler / --tracker")
    source = scheduler or tracker
    # The fetch must present the SAME credential the worker will use to train:
    # in per-peer-auth deployments (admitted_peers, no shared auth_key) the server
    # only honors the Ed25519 identity, so a fetch sending the (absent) HMAC key
    # would be rejected before the worker ever reaches the training loop.
    try:
        cred = PeerIdentity.load(identity_key) if identity_key else auth_key
    except (OSError, ValueError) as e:
        raise SystemExit(f"could not load --identity {identity_key!r}: {e}") from e
    manifest = fetch_manifest(source, auth_key=cred, tls=tls)
    if not verify_manifest(manifest, server_pub=server_pub):
        raise SystemExit("manifest verification failed (bad signature, or --server-pub mismatch)")
    fp = manifest_fingerprint(manifest)
    if server_pub is not None:
        print(f"[join] manifest verified against pinned --server-pub (content {fp})")
    elif "sig" in manifest:
        # Show the actual signer pubkey -- that's what --server-pub pins (NOT the
        # peer_id or the content fingerprint). TOFU here is only as strong as the
        # channel: a MITM could have re-signed or stripped the signature.
        print(f"[join] TOFU: trusting manifest signer pub={manifest.get('pub', '?')} "
              f"(content {fp}). Re-run with --server-pub <that key> to pin it; "
              f"on an untrusted network, pin or use TLS.")
    else:
        print(f"[join] TOFU: accepting an UNSIGNED manifest (content {fp}). The channel "
              f"is the only protection -- a tampered or sig-stripped manifest is "
              f"indistinguishable here. Prefer a signed run + --server-pub, or TLS.")

    base = manifest_to_config(manifest)
    # join has no TLS-client wiring yet; refuse loudly rather than fail with an
    # opaque socket/SSL error against a TLS-only run (volunteer-internet runs that
    # mandate TLS need `opendipaco worker` with a config until join grows --tls).
    if base.tls.enabled and tls is None:
        raise SystemExit(
            "this run requires TLS, which `opendipaco join` does not configure yet; "
            "use `opendipaco worker --config <file>` with the TLS settings for now.")
    dev, diloco_overrides, notes = resolve_device(
        dipaco_config(base.model), requested=device,
        batch_size=base.run.batch_size, seq_len=base.model.sequence_length)
    for n in notes:
        print(f"[join] {n}")

    overrides: dict = {
        "run": {"device": dev, "max_tasks": max_tasks},
        "transport": {"auth_key": auth_key, "identity_key": identity_key},
        "diloco": diloco_overrides,
    }
    if data_dir is not None:
        overrides["data"] = {"shard_cache_dir": data_dir}
    host, port = _split_addr(source)
    if scheduler is not None:
        overrides["transport"].update(connect_host=host, port=port)
    else:
        overrides["tracker"] = {"connect_host": host, "port": port}
    cfg = manifest_to_config(manifest, overrides=overrides)

    # One shared bandwidth bucket for the whole worker (hard --max-mbps ceiling on
    # bytes sent+received); the reporter reads its counters for the live Mbps line.
    rate = rate_from_mbps(max_mbps)
    bucket = TokenBucket(rate) if rate else None
    if bucket is not None:
        print(f"[join] bandwidth cap: {max_mbps:g} Mbps (hard ceiling on send+recv)")
    reporter = HealthReporter(target=f"{host}:{port}", device=dev, quiet=quiet,
                              bucket=bucket, cap_mbps=max_mbps)
    reporter.start()
    try:
        run_worker_role(cfg, max_tasks=max_tasks, stop_event=stop_event, bucket=bucket)
    finally:
        reporter.stop()
