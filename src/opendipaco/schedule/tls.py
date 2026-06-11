"""Optional TLS for the scheduler transport.

The wire format unpickles nothing and the HMAC handshake proves key possession,
but **neither encrypts** -- anyone on-path can read the weights and gradients that
cross the wire. Wrapping the sockets in TLS closes that confidentiality gap; it is
the documented answer to "auth proves possession but does not encrypt".

This module just builds the ``ssl.SSLContext`` objects the servers/clients need
(and, for tests/dev, a throwaway self-signed cert). The transport stays the same:
servers pass a server context into the reactor (``CoordinatorServer(tls=...)``,
``Scheduler``/``ParameterServer(tls=...)``) and clients pass a client context into
``run_worker`` / ``run_sharded_worker``.

Both ends are our own code over the same OpenSSL, so the defaults are strict
(TLS 1.2+). For real deployments supply a CA and verify the peer (and
``require_client_cert=True`` for mutual TLS); for a quick encrypted-but-unverified
channel on an otherwise-trusted network, ``client_context(insecure=True)`` still
beats plaintext (it encrypts; it just doesn't authenticate the server).
"""

from __future__ import annotations

import datetime
import ipaddress
import os
import ssl


def server_context(certfile: str, keyfile: str, *, cafile: str | None = None,
                   require_client_cert: bool = False, password: str | None = None) -> ssl.SSLContext:
    """Build a server-side context from a cert + private key (PEM).

    Pass ``cafile`` + ``require_client_cert=True`` for **mutual TLS** (the server
    then also verifies the worker's certificate).
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile, keyfile, password=password)
    if cafile:
        ctx.load_verify_locations(cafile)
    if require_client_cert:
        if not cafile:
            raise ValueError("require_client_cert needs a cafile to verify the worker cert")
        ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def client_context(*, cafile: str | None = None, certfile: str | None = None,
                   keyfile: str | None = None, check_hostname: bool = False,
                   insecure: bool = False) -> ssl.SSLContext:
    """Build a client-side context.

    - ``cafile`` -- CA that signed the server cert (verification on). With no
      ``cafile`` and not ``insecure``, the system trust store is used.
    - ``check_hostname`` -- also verify the server's hostname against its cert SAN.
    - ``certfile``/``keyfile`` -- this worker's own cert, for mutual TLS.
    - ``insecure`` -- encrypt but do **not** verify the server (self-signed/dev).
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    if insecure:
        ctx.check_hostname = False  # must precede verify_mode = CERT_NONE
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx.check_hostname = check_hostname
        if cafile:
            ctx.load_verify_locations(cafile)
        else:
            ctx.load_default_certs()
    if certfile:
        ctx.load_cert_chain(certfile, keyfile)
    return ctx


def generate_selfsigned_cert(dirpath: str, *, common_name: str = "localhost",
                             hosts=("localhost", "127.0.0.1"), days: int = 365) -> tuple[str, str]:
    """Write a throwaway self-signed cert + key into ``dirpath`` (dev/test only).

    Returns ``(certfile, keyfile)``. The cert lists ``hosts`` in its SAN so
    ``check_hostname`` works against them. Requires the ``cryptography`` package.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "generate_selfsigned_cert needs the 'cryptography' package (dev/test only)"
        ) from e

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    sans = []
    for h in hosts:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .sign(key, hashes.SHA256())
    )
    os.makedirs(dirpath, exist_ok=True)
    certfile = os.path.join(dirpath, "cert.pem")
    keyfile = os.path.join(dirpath, "key.pem")
    with open(certfile, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(keyfile, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    return certfile, keyfile
