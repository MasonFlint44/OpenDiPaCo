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
    forced = requested not in (None, "auto")
    if forced:
        dev = requested
    elif torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    notes = [f"using {dev}" + ("" if forced or dev == "cpu" else " (autodetected)")]
    diloco_overrides: dict = {}
    if dev.startswith("cuda") and torch.cuda.is_available():
        from ..train.memory import fits, vram_breakdown
        free = torch.cuda.mem_get_info()[0]
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
                                 f"vs {free / 1e9:.1f} GB free)")
                break
        else:
            need = vram_breakdown(config, batch_size=batch_size, seq_len=seq_len,
                                  autocast=True, checkpoint=True, chunked_logits=True)["total"]
            if forced:
                raise SystemExit(
                    f"path won't fit {dev}: needs ~{need / 1e9:.1f} GB even with all fit "
                    f"levers, only {free / 1e9:.1f} GB free. Use a smaller batch or --device cpu.")
            dev = "cpu"
            notes.append(f"VRAM too small (~{need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB "
                         f"free) -> falling back to cpu")
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
    """Periodic one-line health surface (D6), to stderr, plus a monotonic-gap
    **sleep detector** (D5): a wall-clock jump means the process was suspended
    (laptop lid). Lean for now -- device / target / uptime / sleep events; rich
    per-task metrics (tasks, accept rate, tokens/sec, Mbps) ride a worker
    progress hook + the W6b byte accounting."""

    def __init__(self, *, target, device, interval: float = 10.0, quiet: bool = False):
        self.target, self.device, self.interval, self.quiet = target, device, interval, quiet
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.quiet:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        start = last = time.monotonic()
        # A gap much larger than the tick is a suspend, not a slow tick.
        gap_threshold = max(self.interval * 3, 30.0)
        while not self._stop.wait(self.interval):
            now = time.monotonic()
            if now - last > gap_threshold:
                print(f"[join] resumed after ~{now - last:.0f}s suspended; reconnecting "
                      f"(stale work is refused by the lease fence)", flush=True)
            last = now
            print(f"[join] training on {self.device} -> {self.target}  "
                  f"up {now - start:.0f}s", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


def run_join(*, scheduler=None, tracker=None, auth_key=None, identity_key=None,
             device=None, max_tasks=None, server_pub=None, data_dir=None,
             quiet: bool = False, tls=None, stop_event=None) -> None:
    """Join a run from connection flags + the fetched manifest (W6 slice a).

    Exactly one of ``scheduler`` / ``tracker`` is the dial target (and the
    manifest source). ``server_pub`` pins the manifest signer; without it the
    manifest is accepted TOFU with its fingerprint printed."""
    if (scheduler is None) == (tracker is None):
        raise SystemExit("join needs exactly one of --scheduler / --tracker")
    source = scheduler or tracker
    manifest = fetch_manifest(source, auth_key=auth_key, tls=tls)
    if not verify_manifest(manifest, server_pub=server_pub):
        raise SystemExit("manifest verification failed (bad signature, or --server-pub mismatch)")
    fp = manifest_fingerprint(manifest)
    if server_pub is not None:
        print(f"[join] manifest verified against pinned key (fingerprint {fp})")
    elif "sig" in manifest:
        print(f"[join] accepting signed manifest from {manifest.get('peer_id', '?')[:12]}… "
              f"(fingerprint {fp}); pass --server-pub to pin it.")
    else:
        print(f"[join] accepting UNSIGNED manifest (fingerprint {fp}); the operator ran "
              f"without an identity -- verify out-of-band or require --server-pub.")

    base = manifest_to_config(manifest)
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

    reporter = HealthReporter(target=f"{host}:{port}", device=dev, quiet=quiet)
    reporter.start()
    try:
        run_worker_role(cfg, max_tasks=max_tasks, stop_event=stop_event)
    finally:
        reporter.stop()
