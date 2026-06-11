"""Stream the transport metrics while a run is in flight.

``TransportMetrics`` (reactor.py) is updated live; this module makes it *operable*
rather than snapshot-only:

- :class:`MetricsExporter` -- a tiny stdlib HTTP server exposing ``/metrics``
  (Prometheus exposition), ``/`` (the human ``report``), and ``/healthz``. Point a
  Prometheus scraper or ``curl`` at it to watch a live coordinator/scheduler.
- :class:`MetricsLogger` -- a background thread that periodically emits a structured
  (JSON) snapshot, for log-based dashboards / debugging a run.

Both read a ``TransportMetrics`` by reference (its accessors are lock-guarded), so
they reflect the live state with no coupling back into the reactor. A server gets
them for free via ``server.start_metrics_server()`` / ``start_metrics_logging()``,
which also stop them on ``shutdown()``.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .reactor import TransportMetrics

logger = logging.getLogger("opendipaco.metrics")


class MetricsExporter:
    """Serve a :class:`TransportMetrics` over HTTP on its own port (separate from
    the transport socket). Routes: ``/metrics`` (Prometheus), ``/healthz``, ``/``."""

    def __init__(self, metrics: TransportMetrics, *, host: str = "0.0.0.0", port: int = 0,
                 namespace: str = "opendipaco_transport"):
        self.metrics = metrics
        self.namespace = namespace
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _make_handler(self):
        exporter = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence the default stderr access log
                pass

            def do_GET(self):  # noqa: N802 (http.server API)
                if self.path.startswith("/metrics"):
                    body = exporter.metrics.prometheus(exporter.namespace).encode()
                    ctype = "text/plain; version=0.0.4; charset=utf-8"
                elif self.path.startswith("/healthz"):
                    body, ctype = b"ok\n", "text/plain; charset=utf-8"
                else:
                    body = (exporter.metrics.report() + "\n").encode()
                    ctype = "text/plain; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return _Handler

    def start(self) -> "MetricsExporter":
        self._thread.start()
        return self

    def stop(self) -> None:
        try:
            self._httpd.shutdown()  # must be called from another thread (it is)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        try:
            self._httpd.server_close()
        except Exception:  # noqa: BLE001
            pass


class MetricsLogger:
    """Emit a structured snapshot of a :class:`TransportMetrics` every ``interval``
    seconds. Default ``sink`` logs one JSON line via ``logging``; pass your own
    callable (e.g. to push to a queue / file / metrics backend)."""

    def __init__(self, metrics: TransportMetrics, *, interval: float = 10.0, sink=None):
        self.metrics = metrics
        self.interval = interval
        self.sink = sink or (lambda d: logger.info("transport_metrics %s", json.dumps(d)))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        # wait() returns True only when stopped -> emit on each interval until then.
        while not self._stop.wait(self.interval):
            try:
                self.sink(self.metrics.summary())
            except Exception:  # noqa: BLE001 - a bad sink must not kill the run
                logger.exception("metrics sink failed")

    def start(self) -> "MetricsLogger":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
