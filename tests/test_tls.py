"""Tests for optional TLS on the scheduler transport.

The wire format is pickle-free and HMAC auth proves key possession, but neither
encrypts. These tests check the TLS wrapper that closes that gap: a TLS coordinator
trains a TLS worker end-to-end, server identity is actually verified against a CA,
a plaintext client can't talk to a TLS server, TLS composes with the HMAC auth, and
the sharded (Scheduler + ParameterServer) path works over TLS too.

A throwaway self-signed cert (SAN: localhost, 127.0.0.1) is generated per test.
"""

import socket
import ssl
import threading

import pytest
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
    ParameterServer,
    Scheduler,
    assign_shards,
    client_context,
    generate_selfsigned_cert,
    run_sharded_worker,
    run_worker,
    server_context,
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


@pytest.fixture
def certs(tmp_path):
    return generate_selfsigned_cert(str(tmp_path))  # (certfile, keyfile)


# -- context builders --------------------------------------------------------


def test_client_context_insecure_disables_verification():
    ctx = client_context(insecure=True)
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.check_hostname is False


def test_require_client_cert_needs_cafile(certs):
    certfile, keyfile = certs
    with pytest.raises(ValueError):
        server_context(certfile, keyfile, require_client_cert=True)  # no cafile


def test_verifying_client_context_enforces_a_ca(certs):
    certfile, _ = certs
    ctx = client_context(cafile=certfile)
    assert ctx.verify_mode == ssl.CERT_REQUIRED


# -- coordinator over TLS ----------------------------------------------------


def _run_coordinator(certs, *, client_tls, auth_key=None, hostname=None, gens=2):
    certfile, keyfile = certs
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(
        AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg), batch_size=BATCH,
        host="127.0.0.1", port=0, auth_key=auth_key,
        tls=server_context(certfile, keyfile),
    )
    server.start()
    w = threading.Thread(
        target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
        kwargs=dict(seed=0, reconnect=False, auth_key=auth_key,
                    tls=client_tls, tls_hostname=hostname),
        daemon=True,
    )
    w.start()
    server.fit(num_generations=gens, total_generations=gens, log_every=0)
    server.shutdown()
    w.join(timeout=15)
    return server, cfg


def test_coordinator_over_tls_completes(certs):
    """An encrypted worker trains the bank end-to-end (server not verified)."""
    server, cfg = _run_coordinator(certs, client_tls=client_context(insecure=True))
    assert server._T >= server._target
    assert server.metrics.accepted_updates >= cfg.num_paths * 2  # real work happened


def test_tls_client_verifies_server_against_ca(certs):
    """Pinning the self-signed cert as the CA + hostname check still completes."""
    certfile, _ = certs
    ctx = client_context(cafile=certfile, check_hostname=True)
    server, cfg = _run_coordinator(certs, client_tls=ctx, hostname="127.0.0.1")
    assert server._T >= server._target


def test_tls_client_rejects_untrusted_server(certs, tmp_path):
    """A client pinned to a *different* CA refuses the handshake (no work served)."""
    other_cert, _ = generate_selfsigned_cert(str(tmp_path / "other"))
    certfile, keyfile = certs
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(
        AsyncScheduler(eng, lease_timeout=2.0), _corpus(cfg), batch_size=BATCH,
        host="127.0.0.1", port=0, tls=server_context(certfile, keyfile),
    )
    server.start()
    err = {}

    def run():
        try:
            run_worker(cfg, _diloco(), "127.0.0.1", server.port, seed=0, reconnect=False,
                       tls=client_context(cafile=other_cert), tls_hostname="127.0.0.1")
        except ssl.SSLError as e:
            err["e"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=10)
    server.shutdown()
    assert isinstance(err.get("e"), ssl.SSLError)  # cert verification failed
    assert server.metrics.accepted_updates == 0


def test_plaintext_client_cannot_talk_to_tls_server(certs):
    """A non-TLS worker against a TLS server does no work (handshake mismatch)."""
    certfile, keyfile = certs
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(
        AsyncScheduler(eng, lease_timeout=2.0), _corpus(cfg), batch_size=BATCH,
        host="127.0.0.1", port=0, tls=server_context(certfile, keyfile),
    )
    server.start()
    # Plaintext worker: its framed bytes look like a malformed TLS record -> the
    # server drops it; the worker sees the socket close and returns without work.
    w = threading.Thread(
        target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
        kwargs=dict(seed=0, reconnect=False), daemon=True,
    )
    w.start()
    w.join(timeout=8)
    server.shutdown()
    assert server.metrics.accepted_updates == 0


def test_tls_and_auth_compose(certs):
    """TLS for confidentiality + HMAC auth for identity work together."""
    server, cfg = _run_coordinator(certs, client_tls=client_context(insecure=True),
                                   auth_key="s3cret")
    assert server._T >= server._target
    assert server.metrics.accepted_updates >= cfg.num_paths * 2


# -- sharded path over TLS ---------------------------------------------------


def test_sharded_over_tls_completes(certs):
    """Scheduler + parameter servers + worker all over TLS train the bank."""
    certfile, keyfile = certs
    cfg = _cfg()
    dl = _diloco()
    srv_ctx = server_context(certfile, keyfile)
    cli_ctx = client_context(insecure=True)
    keys = assign_shards(cfg.build_topology().module_keys(), 2)
    shards = [[k for k, s in keys.items() if s == i] for i in range(2)]
    pss = [ParameterServer(cfg, sk, dl, host="127.0.0.1", port=0, tls=srv_ctx) for sk in shards]
    for ps in pss:
        ps.start()
    sched = Scheduler(cfg, _corpus(cfg), [("127.0.0.1", ps.port) for ps in pss], dl,
                      batch_size=BATCH, host="127.0.0.1", port=0, tls=srv_ctx, ps_tls=cli_ctx)
    sched.start()
    w = threading.Thread(
        target=run_sharded_worker, args=(cfg, dl, ("127.0.0.1", sched.port)),
        kwargs=dict(seed=0, heartbeat_interval=1.0, tls=cli_ctx), daemon=True,
    )
    w.start()
    completed = sched.fit(num_generations=2, total_generations=2)
    sched.shutdown()
    for ps in pss:
        ps.shutdown()
    w.join(timeout=15)
    assert sum(completed.values()) >= cfg.num_paths * 2
    # The shards genuinely moved over the encrypted channel.
    assert any(ps._versions and max(ps._versions.values()) > 0 for ps in pss)
