"""Pickle-free wire format + auth for the scheduler transport.

The transport previously framed messages with ``torch.save``/``torch.load``,
which **pickles** -- so receiving a message ran arbitrary code. This module
replaces that with a typed, code-free serializer plus an optional auth handshake
and a message-size cap, so the transport can run between hosts that don't fully
trust each other.

Format of one message body::

    [u32 json_len][json structure][u32 num_tensors]
    repeated num_tensors times:
        [u8 dtype_code][u8 ndim][u64 * ndim shape][u64 nbytes][raw little-endian bytes]

The *structure* is JSON (parsed with ``json.loads`` -- no code execution): plain
dicts (string keys only), lists, and scalars, with tensors replaced by
``{"$t": idx}`` and tuples tagged ``{"$tup": [...]}`` so paths round-trip. Tensors
are reconstructed from raw bytes against an explicit dtype **allowlist** and the
declared shape -- never executed.

``send_msg``/``recv_msg`` keep the 8-byte length framing (so byte accounting and
``_recv_n`` are unchanged) but use ``encode``/``decode`` for the body, and
``recv_msg_sized`` enforces ``max_bytes`` against the length prefix.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct

import torch

# 4 GiB default cap: comfortably above a single 150M-param path's weights, but
# bounds a garbage/oversized length prefix.
DEFAULT_MAX_MSG_BYTES = 4 * 1024 ** 3

_HEADER = struct.Struct(">Q")
_U32 = struct.Struct(">I")
_TENSOR_HDR = struct.Struct(">BB")
_U64 = struct.Struct(">Q")

# dtype <-> stable byte code (allowlist; unknown dtypes are rejected on both ends).
_DTYPES = {
    torch.float32: 0, torch.float64: 1, torch.float16: 2, torch.bfloat16: 3,
    torch.int64: 4, torch.int32: 5, torch.int16: 6, torch.int8: 7,
    torch.uint8: 8, torch.bool: 9,
}
_CODES = {v: k for k, v in _DTYPES.items()}


# -- (de)serialization -------------------------------------------------------


def _to_structure(obj, blobs: list) -> object:
    if torch.is_tensor(obj):
        blobs.append(obj.detach().cpu().contiguous())
        return {"$t": len(blobs) - 1}
    if isinstance(obj, tuple):
        return {"$tup": [_to_structure(x, blobs) for x in obj]}
    if isinstance(obj, list):
        return [_to_structure(x, blobs) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"wire dict keys must be str, got {type(k)}")
            out[k] = _to_structure(v, blobs)
        return out
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f"cannot serialize {type(obj)} over the wire")


def _from_structure(s, blobs: list):
    if isinstance(s, dict):
        if len(s) == 1 and "$t" in s:
            return blobs[s["$t"]]
        if len(s) == 1 and "$tup" in s:
            return tuple(_from_structure(x, blobs) for x in s["$tup"])
        return {k: _from_structure(v, blobs) for k, v in s.items()}
    if isinstance(s, list):
        return [_from_structure(x, blobs) for x in s]
    return s


def encode(obj) -> bytes:
    blobs: list = []
    structure = json.dumps(_to_structure(obj, blobs)).encode("utf-8")
    out = bytearray()
    out += _U32.pack(len(structure))
    out += structure
    out += _U32.pack(len(blobs))
    for t in blobs:
        if t.dtype not in _DTYPES:
            raise TypeError(f"unsupported tensor dtype {t.dtype}")
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        out += _TENSOR_HDR.pack(_DTYPES[t.dtype], t.dim())
        for d in t.shape:
            out += _U64.pack(d)
        out += _U64.pack(len(raw))
        out += raw
    return bytes(out)


def decode(data: bytes):
    off = 0
    (slen,) = _U32.unpack_from(data, off); off += _U32.size
    structure = json.loads(bytes(data[off:off + slen])); off += slen
    (ntensors,) = _U32.unpack_from(data, off); off += _U32.size
    blobs: list = []
    for _ in range(ntensors):
        code, ndim = _TENSOR_HDR.unpack_from(data, off); off += _TENSOR_HDR.size
        if code not in _CODES:
            raise ValueError(f"unknown tensor dtype code {code}")
        shape = []
        for _ in range(ndim):
            (d,) = _U64.unpack_from(data, off); off += _U64.size
            shape.append(d)
        (nbytes,) = _U64.unpack_from(data, off); off += _U64.size
        raw = data[off:off + nbytes]; off += nbytes
        if len(raw) != nbytes:
            raise ValueError("truncated tensor blob")
        dtype = _CODES[code]
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(shape).clone()
        blobs.append(t)
    return _from_structure(structure, blobs)


# -- framing -----------------------------------------------------------------


def send_msg(sock: socket.socket, obj) -> int:
    """Send a framed message; return the number of bytes written (for metrics)."""
    data = encode(obj)
    sock.sendall(_HEADER.pack(len(data)) + data)
    return len(data) + _HEADER.size


def _recv_n(sock: socket.socket, n: int) -> bytes | None:
    chunks: list[bytes] = []
    got = 0
    while got < n:
        b = sock.recv(min(n - got, 1 << 20))
        if not b:
            return None  # peer closed
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def recv_msg_sized(sock: socket.socket, max_bytes: int = DEFAULT_MAX_MSG_BYTES):
    """Receive a framed message; return ``(obj, nbytes)`` (``(None, 0)`` on close).

    Raises ``ValueError`` if the declared length exceeds ``max_bytes`` (so a
    garbage/oversized prefix can't drive an unbounded read).
    """
    header = _recv_n(sock, _HEADER.size)
    if header is None:
        return None, 0
    (n,) = _HEADER.unpack(header)
    if n > max_bytes:
        raise ValueError(f"incoming message of {n} bytes exceeds cap {max_bytes}")
    body = _recv_n(sock, n)
    if body is None:
        return None, 0
    return decode(body), n + _HEADER.size


def recv_msg(sock: socket.socket, max_bytes: int = DEFAULT_MAX_MSG_BYTES):
    return recv_msg_sized(sock, max_bytes)[0]


# -- auth handshake (HMAC challenge-response over a shared secret) ------------


def _key_bytes(key) -> bytes:
    return key.encode("utf-8") if isinstance(key, str) else bytes(key)


def coerce_keys(auth_key) -> list[bytes] | None:
    """Normalize ``None | str | bytes | iterable[str|bytes]`` to ``list[bytes]|None``.

    A single secret stays a one-element list; a collection becomes the set of keys a
    *server* will accept (key rotation: list old+new during the migration window;
    per-worker identity: one distinct secret per worker, revoke by dropping it).
    """
    if auth_key is None:
        return None
    if isinstance(auth_key, (str, bytes, bytearray)):
        return [_key_bytes(auth_key)]
    keys = [_key_bytes(k) for k in auth_key]
    return keys or None


def acceptable_keys(auth_key, accept_keys=None) -> list[bytes] | None:
    """The de-duplicated set of keys a server accepts: ``auth_key`` ∪ ``accept_keys``.

    ``auth_key`` is also the node's own *client* identity (a single secret); the
    extra ``accept_keys`` exist only to accept additional workers' keys.
    """
    out: list[bytes] = []
    for src in (auth_key, accept_keys):
        for k in coerce_keys(src) or []:
            if k not in out:
                out.append(k)
    return out or None


def _verify_mac(mac: str, nonce: bytes, keys: list[bytes]) -> bool:
    """Constant-time check that ``mac`` proves possession of *some* accepted key."""
    ok = False
    for k in keys:  # check all keys (no early-out) so timing doesn't leak which matched
        expected = hmac.new(k, nonce, hashlib.sha256).hexdigest()
        ok |= hmac.compare_digest(mac, expected)
    return ok


def server_handshake(sock: socket.socket, key) -> bool:
    """Coordinator side: challenge the worker; return True if it proves an accepted key.

    ``key`` may be a single secret or a collection of acceptable secrets.
    """
    keys = coerce_keys(key)
    if keys is None:
        return True
    nonce = os.urandom(32)
    try:
        send_msg(sock, {"type": "challenge", "nonce": nonce.hex()})
        reply = recv_msg(sock, max_bytes=4096)
    except (OSError, ValueError):
        return False
    if not reply or reply.get("type") != "auth":
        return False
    if not _verify_mac(reply.get("mac", ""), nonce, keys):
        return False
    try:
        send_msg(sock, {"type": "welcome"})
    except OSError:
        return False
    return True


def client_handshake(sock: socket.socket, key) -> bool:
    """Worker side: answer the coordinator's challenge.

    ``key`` is either a shared secret (``str``/``bytes`` -> HMAC reply, as
    before) or a :class:`~opendipaco.schedule.identity.PeerIdentity` (-> Ed25519
    signature reply; the server must list the peer's public key in
    ``admitted_peers``).
    """
    if key is None:
        return True
    try:
        challenge = recv_msg(sock, max_bytes=4096)
        if not challenge or challenge.get("type") != "challenge":
            return False
        nonce = bytes.fromhex(challenge["nonce"])
        if hasattr(key, "auth_response"):  # a PeerIdentity (duck-typed; no import)
            send_msg(sock, {"type": "auth", **key.auth_response(nonce)})
        else:
            mac = hmac.new(_key_bytes(key), nonce, hashlib.sha256).hexdigest()
            send_msg(sock, {"type": "auth", "mac": mac})
        welcome = recv_msg(sock, max_bytes=4096)
    except (OSError, ValueError):
        return False
    return bool(welcome and welcome.get("type") == "welcome")
