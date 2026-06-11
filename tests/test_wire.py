"""Unit tests for the pickle-free wire format + auth handshake (wire.py)."""

import math
import socket
import struct
import threading

import pytest
import torch

from opendipaco.schedule.wire import (
    client_handshake,
    decode,
    encode,
    recv_msg_sized,
    send_msg,
    server_handshake,
)


def _example_message():
    return {
        "type": "task", "gen_id": 3, "path": (0, 1), "loss": float("nan"),
        "empty": False, "total_rounds": None, "seed": 12345,
        "shared_weights": {"embed": {"weight": torch.randn(4, 3)},
                           "L0E0": {"b": torch.zeros(2, dtype=torch.int64)}},
        "shared_grad": {"head": [torch.randn(3), torch.ones(2, dtype=torch.bfloat16)]},
        "flag": torch.tensor([True, False]),
        "warm_paths": [(0, 0), (1, 1)], "versions": {"embed": 2},
    }


def test_roundtrip_dtypes_containers_and_tuples():
    msg = _example_message()
    out = decode(encode(msg))
    assert out["path"] == (0, 1) and isinstance(out["path"], tuple)
    assert out["warm_paths"] == [(0, 0), (1, 1)]
    assert out["total_rounds"] is None and math.isnan(out["loss"])
    assert torch.equal(out["shared_weights"]["embed"]["weight"],
                       msg["shared_weights"]["embed"]["weight"])
    assert out["shared_weights"]["L0E0"]["b"].dtype == torch.int64
    g = out["shared_grad"]["head"]
    assert g[1].dtype == torch.bfloat16 and torch.equal(g[1], msg["shared_grad"]["head"][1])
    assert out["flag"].dtype == torch.bool and out["flag"].tolist() == [True, False]


def test_not_pickle():
    # The body is a length prefix followed by JSON, never a pickle opcode stream.
    data = encode({"type": "ping", "x": 1})
    assert data[4:6] == b'{"'  # JSON structure right after the 4-byte length


def test_unsupported_types_rejected():
    with pytest.raises(TypeError):
        encode({"s": {1, 2, 3}})       # a set
    with pytest.raises(TypeError):
        encode({"o": object()})        # arbitrary object
    with pytest.raises(TypeError):
        encode({(0, 0): 1})            # non-str dict key


def test_decode_rejects_garbage():
    with pytest.raises(Exception):
        decode(b"definitely not a valid wire message")


def test_send_recv_over_socketpair():
    a, b = socket.socketpair()
    try:
        send_msg(a, {"type": "task", "t": torch.arange(5)})
        obj, n = recv_msg_sized(b)
        assert torch.equal(obj["t"], torch.arange(5)) and n > 0
    finally:
        a.close()
        b.close()


def test_oversize_message_rejected():
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack(">Q", 10 ** 12) + b"x")  # header claims ~1 TB
        with pytest.raises(ValueError):
            recv_msg_sized(b, max_bytes=1024)
    finally:
        a.close()
        b.close()


def _handshake_pair(server_key, client_key):
    a, b = socket.socketpair()
    res = {}

    def server():
        res["server"] = server_handshake(a, server_key)
        if not res["server"]:
            a.close()  # real flow closes on failure, which unblocks the client

    t = threading.Thread(target=server)
    t.start()
    res["client"] = client_handshake(b, client_key)
    t.join(timeout=5)
    a.close()
    b.close()
    return res


def test_auth_matching_key_succeeds():
    res = _handshake_pair("secret", "secret")
    assert res["server"] and res["client"]


def test_auth_wrong_key_fails():
    res = _handshake_pair("secret", "wrong")
    assert not res["server"] and not res["client"]
