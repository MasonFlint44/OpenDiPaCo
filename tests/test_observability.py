"""Tests for streaming the transport metrics (Prometheus export + structured log).

These cover the metrics serializer and per-worker liveness directly, the HTTP
exporter (a real GET over a socket), the periodic logger, and one integration where
a live coordinator's ``/metrics`` endpoint reflects a worker actually training.
"""

import json
import threading
import time
from urllib.request import urlopen

import torch

from opendipaco import (
    AsyncScheduler,
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
)
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import (
    CoordinatorServer,
    MetricsExporter,
    MetricsLogger,
    TransportMetrics,
    run_worker,
)

BATCH = 8


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1], max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _diloco():
    return DiLoCoConfig(inner_steps=4, inner_lr=1e-3)


def _docs():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(8)]


def _corpus(cfg):
    assign = torch.tensor([i % cfg.num_paths for i in range(32)])
    return ShardedCorpus.from_assignments(_docs(), assign, cfg.num_paths, cfg.sequence_length)


def _engine(cfg, seed=0):
    return DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                        seed=seed, materialize="serial")


# -- metrics serializer + liveness -------------------------------------------


def test_prometheus_exposition_format():
    m = TransportMetrics()
    m.record_update(2)
    m.record_update(4)
    m.record_stale_reject()
    text = m.prometheus()
    # Counters carry a _total suffix + a TYPE line; the value reflects recorded state.
    assert "# TYPE opendipaco_transport_accepted_updates_total counter" in text
    assert "opendipaco_transport_accepted_updates_total 2" in text
    assert "opendipaco_transport_stale_rejected_total 1" in text
    # Derived gauges are gauges (no _total), and max_staleness reflects the worst seen.
    assert "# TYPE opendipaco_transport_mean_staleness gauge" in text
    assert "opendipaco_transport_max_staleness 4" in text
    # Every metric line has a preceding TYPE line.
    assert text.count("# TYPE ") == sum(1 for ln in text.splitlines() if ln and not ln.startswith("#"))


def test_active_workers_liveness_window():
    m = TransportMetrics()
    m.record_worker("a")
    m.record_worker("b")
    assert m.active_workers() == 2
    assert m.summary()["active_workers"] == 2
    # Age 'a' beyond the window -> only 'b' counts as recently seen.
    m._worker_seen["a"] -= 1000.0
    assert m.active_workers(window=60.0) == 1
    m.record_worker("a")  # a fresh message revives it
    assert m.active_workers() == 2


# -- HTTP exporter -----------------------------------------------------------


def test_metrics_exporter_serves_routes():
    m = TransportMetrics()
    m.record_update(1)
    m.record_worker("w0")
    exp = MetricsExporter(m, host="127.0.0.1", port=0).start()
    try:
        metrics = urlopen(f"http://127.0.0.1:{exp.port}/metrics", timeout=5).read().decode()
        assert "opendipaco_transport_accepted_updates_total 1" in metrics
        assert "opendipaco_transport_active_workers 1" in metrics
        health = urlopen(f"http://127.0.0.1:{exp.port}/healthz", timeout=5).read().decode()
        assert health.strip() == "ok"
        root = urlopen(f"http://127.0.0.1:{exp.port}/", timeout=5).read().decode()
        assert "accepted=" in root  # the human report
    finally:
        exp.stop()


def test_metrics_exporter_stops():
    exp = MetricsExporter(TransportMetrics(), host="127.0.0.1", port=0).start()
    port = exp.port
    exp.stop()
    time.sleep(0.1)
    try:
        urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1)
        raised = False
    except Exception:  # noqa: BLE001 - connection refused once stopped
        raised = True
    assert raised


# -- structured logger -------------------------------------------------------


def test_metrics_logger_emits_structured_snapshots():
    m = TransportMetrics()
    m.record_update(3)
    seen = []
    lg = MetricsLogger(m, interval=0.03, sink=seen.append).start()
    try:
        deadline = time.monotonic() + 2.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        lg.stop()
    assert seen, "logger emitted no snapshot"
    snap = seen[-1]
    assert snap["accepted_updates"] == 1 and "active_workers" in snap
    json.dumps(snap)  # snapshots are JSON-serializable (log-friendly)


# -- integration: live endpoint reflects a training worker -------------------


def test_live_endpoint_reflects_training_worker():
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0)
    server.start()
    exp = server.start_metrics_server(host="127.0.0.1")
    logs = []
    server.start_metrics_logging(interval=0.05, sink=logs.append)
    w = threading.Thread(target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
                         kwargs=dict(seed=0, reconnect=False), daemon=True)
    w.start()
    server.fit(num_generations=2, total_generations=2, log_every=0)
    # Scrape before shutdown: a worker really connected and updates were applied.
    scrape = urlopen(f"http://127.0.0.1:{exp.port}/metrics", timeout=5).read().decode()
    server.shutdown()
    w.join(timeout=10)
    assert "opendipaco_transport_accepted_updates_total" in scrape
    assert server.metrics.active_workers() >= 1
    assert any(d.get("accepted_updates", 0) > 0 for d in logs)  # the logger saw progress
    # The exporter is stopped by shutdown().
    time.sleep(0.1)
    try:
        urlopen(f"http://127.0.0.1:{exp.port}/metrics", timeout=1)
        stopped = False
    except Exception:  # noqa: BLE001
        stopped = True
    assert stopped
