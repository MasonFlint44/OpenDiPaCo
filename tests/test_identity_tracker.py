"""Tests for peer identity + the rendezvous tracker (internet-scale plan, Phase 1).

Identity: Ed25519 keypairs, challenge auth in the reactor handshake (alongside
HMAC), runtime admit/revoke, and self-certifying signed records. Tracker:
register/heartbeat/TTL/deregister, reachability tiers, enrollment, and the
bootstrap-a-replacement-from-a-peer's-cache property that makes tracker loss
degrade rather than halt.
"""

import threading
import time

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
    PeerIdentity,
    Tracker,
    fetch_directory,
    import_records,
    make_peer_record,
    peer_id_of,
    register_peer,
    run_worker,
    sign_record,
    verify_record,
)
from opendipaco.schedule.identity import verify_auth
from opendipaco.schedule.tracker import deregister_peer

BATCH = 8


def _cfg():
    bb = BackboneConfig(
        vocab_size=48, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
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


def _engine(cfg):
    return DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                        seed=0, materialize="serial")


# -- identity ---------------------------------------------------------------------


def test_identity_roundtrip_and_key_file_permissions(tmp_path):
    ident = PeerIdentity.generate()
    path = ident.save(tmp_path / "peer.pem")
    loaded = PeerIdentity.load(path)
    assert loaded.peer_id == ident.peer_id == peer_id_of(ident.public_key_hex)
    assert loaded.public_key_hex == ident.public_key_hex
    import os
    assert (os.stat(path).st_mode & 0o777) == 0o600   # private key is owner-only


def test_challenge_auth_verifies_and_rejects():
    ident = PeerIdentity.generate()
    nonce = b"n" * 32
    resp = ident.auth_response(nonce)
    assert verify_auth(resp["pub"], resp["sig"], nonce)
    assert not verify_auth(resp["pub"], resp["sig"], b"x" * 32)     # different nonce
    other = PeerIdentity.generate()
    assert not verify_auth(other.public_key_hex, resp["sig"], nonce)  # wrong key
    assert not verify_auth(resp["pub"], "zz-not-hex", nonce)          # garbage sig


def test_signed_records_are_self_certifying():
    ident = PeerIdentity.generate()
    rec = sign_record(ident, {"kind": "peer", "roles": ["worker"], "issued_at": 1.0})
    assert verify_record(rec)
    tampered = dict(rec, roles=["owner"])                 # any field change breaks it
    assert not verify_record(tampered)
    # Claiming someone else's peer_id with your own key fails (id is key-derived).
    other = PeerIdentity.generate()
    forged = sign_record(other, {"kind": "peer", "issued_at": 1.0})
    forged["peer_id"] = ident.peer_id
    assert not verify_record(forged)
    assert not verify_record({"kind": "peer"})            # unsigned
    assert not verify_record(None)


def test_peer_record_reachability_validation():
    ident = PeerIdentity.generate()
    pub = make_peer_record(ident, reachability="public", addr=("1.2.3.4", 29700),
                           roles=["owner", "relay"], capabilities={"device": "cuda"})
    assert verify_record(pub) and pub["addr"] == ["1.2.3.4", 29700]
    nat = make_peer_record(ident, reachability="nat", roles=["worker"])
    assert verify_record(nat) and nat["addr"] is None
    with pytest.raises(ValueError):
        make_peer_record(ident, reachability="public")            # public needs addr
    with pytest.raises(ValueError):
        make_peer_record(ident, reachability="nat", addr=("h", 1))  # nat is dial-out
    with pytest.raises(ValueError):
        make_peer_record(ident, reachability="carrier-pigeon")


# -- reactor handshake with identities ----------------------------------------------


def _fit_with_worker(server, cfg, auth, gens=2):
    server.start()
    err = {}

    def run():
        try:
            run_worker(cfg, _diloco(), "127.0.0.1", server.port, seed=0,
                       reconnect=False, auth_key=auth)
        except Exception as e:  # noqa: BLE001
            err["e"] = e

    w = threading.Thread(target=run, daemon=True)
    w.start()
    server.fit(num_generations=gens, total_generations=gens, log_every=0)
    server.shutdown()
    w.join(timeout=15)
    return err.get("e")


def test_identity_handshake_end_to_end():
    """A worker authenticating by Ed25519 identity (no shared secret anywhere)
    is admitted and does real work."""
    cfg = _cfg()
    ident = PeerIdentity.generate()
    server = CoordinatorServer(AsyncScheduler(_engine(cfg), lease_timeout=5.0),
                               _corpus(cfg), batch_size=BATCH, host="127.0.0.1",
                               port=0, admitted_peers=[ident])
    assert _fit_with_worker(server, cfg, ident) is None
    assert server._T >= server._target
    assert server.metrics.accepted_updates >= cfg.num_paths * 2


def test_unadmitted_and_revoked_identities_are_refused():
    cfg = _cfg()
    admitted = PeerIdentity.generate()
    outsider = PeerIdentity.generate()
    server = CoordinatorServer(AsyncScheduler(_engine(cfg), lease_timeout=5.0),
                               _corpus(cfg), batch_size=BATCH, host="127.0.0.1",
                               port=0, admitted_peers=[admitted])
    server.start()
    err = {}

    def run(ident):
        try:
            run_worker(cfg, _diloco(), "127.0.0.1", server.port, seed=0,
                       reconnect=False, auth_key=ident)
        except Exception as e:  # noqa: BLE001
            err["e"] = e

    t = threading.Thread(target=run, args=(outsider,))
    t.start()
    t.join(timeout=10)
    assert isinstance(err.get("e"), PermissionError)      # not admitted
    assert server.metrics.tasks_sent == 0

    server.revoke_peer(admitted)                          # revocation: drop the key
    err.clear()
    t = threading.Thread(target=run, args=(admitted,))
    t.start()
    t.join(timeout=10)
    assert isinstance(err.get("e"), PermissionError)
    server.shutdown()


def test_hmac_and_identity_auth_coexist():
    """One server, both mechanisms: an HMAC worker (enrollment-token style) and
    an identity worker are each admitted."""
    cfg = _cfg()
    ident = PeerIdentity.generate()
    server = CoordinatorServer(AsyncScheduler(_engine(cfg), lease_timeout=5.0),
                               _corpus(cfg), batch_size=BATCH, host="127.0.0.1",
                               port=0, auth_key="token", admitted_peers=[ident])
    server.start()
    errs = {}

    def run(name, auth):
        try:
            run_worker(cfg, _diloco(), "127.0.0.1", server.port, seed=0,
                       reconnect=False, auth_key=auth)
        except Exception as e:  # noqa: BLE001
            errs[name] = e

    ws = [threading.Thread(target=run, args=("hmac", "token"), daemon=True),
          threading.Thread(target=run, args=("ident", ident), daemon=True)]
    for w in ws:
        w.start()
    server.fit(num_generations=2, total_generations=2, log_every=0)
    server.shutdown()
    for w in ws:
        w.join(timeout=15)
    assert errs == {}
    assert server._T >= server._target


# -- tracker -------------------------------------------------------------------------


def test_tracker_register_fetch_filter_and_heartbeat():
    a, b = PeerIdentity.generate(), PeerIdentity.generate()
    tracker = Tracker(host="127.0.0.1", port=0, ttl=30.0, open_enrollment=True)
    tracker.start()
    try:
        addr = ("127.0.0.1", tracker.port)
        r = register_peer(addr, a, reachability="public", peer_addr=("10.0.0.1", 29700),
                          roles=["owner", "relay"], capabilities={"device": "cuda"})
        assert r["type"] == "registered" and r["ttl"] == 30.0
        register_peer(addr, b, roles=["worker"])           # nat, dial-out-only

        recs = fetch_directory(addr)
        assert {x["peer_id"] for x in recs} == {a.peer_id, b.peer_id}
        owners = fetch_directory(addr, roles=["owner"])
        assert [x["peer_id"] for x in owners] == [a.peer_id]
        public = fetch_directory(addr, reachability="public")
        assert [x["peer_id"] for x in public] == [a.peer_id]

        # Heartbeat = re-register with a fresh issued_at; a *stale* copy is refused.
        old = make_peer_record(a, reachability="public", addr=("10.0.0.1", 29700))
        time.sleep(0.01)
        assert register_peer(addr, a, reachability="public",
                             peer_addr=("10.0.0.1", 29701))["type"] == "registered"
        stale = import_records(addr, [old])
        assert stale["accepted"] == 0                      # older issued_at -> refused
        (rec,) = fetch_directory(addr, reachability="public")
        assert rec["addr"] == ["10.0.0.1", 29701]          # the newer one won
    finally:
        tracker.shutdown()


def test_tracker_ttl_expiry_and_deregister():
    ident = PeerIdentity.generate()
    tracker = Tracker(host="127.0.0.1", port=0, ttl=0.2, open_enrollment=True)
    tracker.start()
    try:
        addr = ("127.0.0.1", tracker.port)
        register_peer(addr, ident, roles=["worker"])
        assert len(fetch_directory(addr)) == 1
        time.sleep(0.3)                                    # no heartbeat -> expired
        assert fetch_directory(addr) == []

        # Deregister leaves a tombstone: an old cached record can't be re-imported.
        rec = make_peer_record(ident, roles=["worker"])
        time.sleep(0.01)
        assert import_records(addr, [rec])["accepted"] == 1
        assert deregister_peer(addr, ident)["type"] == "deregistered"
        assert fetch_directory(addr) == []
        assert import_records(addr, [rec])["accepted"] == 0   # tombstone blocks it
    finally:
        tracker.shutdown()


def test_tracker_enrollment_gate():
    member, outsider = PeerIdentity.generate(), PeerIdentity.generate()
    tracker = Tracker(host="127.0.0.1", port=0, enroll_peers=[member.public_key_hex])
    tracker.start()
    try:
        addr = ("127.0.0.1", tracker.port)
        assert register_peer(addr, member, roles=["worker"])["type"] == "registered"
        refused = register_peer(addr, outsider, roles=["worker"])
        assert refused == {"type": "refused", "reason": "not enrolled"}

        tracker.enroll(outsider)                           # runtime enrollment
        assert register_peer(addr, outsider, roles=["worker"])["type"] == "registered"

        tracker.expel(member)                              # revocation drops the record
        assert {r["peer_id"] for r in fetch_directory(addr)} == {outsider.peer_id}
        again = register_peer(addr, member, roles=["worker"])
        assert again["reason"] == "not enrolled"
    finally:
        tracker.shutdown()


def test_tracker_refuses_unsigned_and_tampered_records():
    ident = PeerIdentity.generate()
    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True)
    tracker.start()
    try:
        addr = ("127.0.0.1", tracker.port)
        rec = make_peer_record(ident, reachability="nat", roles=["worker"])
        tampered = dict(rec, roles=["owner"])              # claim a role it didn't sign
        assert import_records(addr, [tampered])["accepted"] == 0
        assert import_records(addr, [{"kind": "peer"}])["accepted"] == 0
        assert fetch_directory(addr) == []
    finally:
        tracker.shutdown()


def test_tracker_bootstrap_from_a_peers_cache():
    """Tracker loss degrades, not halts: a client's cached (signed) directory
    bootstraps a fresh tracker, which re-verifies every record itself."""
    peers = [PeerIdentity.generate() for _ in range(3)]
    a = Tracker(host="127.0.0.1", port=0, ttl=60.0, open_enrollment=True)
    a.start()
    addr_a = ("127.0.0.1", a.port)
    for i, p in enumerate(peers):
        register_peer(addr_a, p, reachability="public", peer_addr=("10.0.0.1", 29700 + i),
                      roles=["owner"])
    cache = fetch_directory(addr_a)                        # any client's cached copy
    a.shutdown()                                           # the tracker dies

    b = Tracker(host="127.0.0.1", port=0, ttl=60.0, open_enrollment=True)
    b.start()
    try:
        addr_b = ("127.0.0.1", b.port)
        assert import_records(addr_b, cache)["accepted"] == 3
        assert ({r["peer_id"] for r in fetch_directory(addr_b)}
                == {p.peer_id for p in peers})
    finally:
        b.shutdown()


def test_tracker_behind_enrollment_token():
    """Transport-level HMAC composes underneath: without the token you can't
    even talk to the tracker."""
    ident = PeerIdentity.generate()
    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True, auth_key="join-token")
    tracker.start()
    try:
        addr = ("127.0.0.1", tracker.port)
        ok = register_peer(addr, ident, roles=["worker"], auth_key="join-token")
        assert ok["type"] == "registered"
        with pytest.raises(PermissionError):
            register_peer(addr, ident, roles=["worker"], auth_key="wrong")
    finally:
        tracker.shutdown()


# -- launch plumbing -----------------------------------------------------------------


def test_cli_gen_identity_and_config_sections(tmp_path, capsys):
    from opendipaco.launch import LaunchConfig
    from opendipaco.launch.cli import main
    from opendipaco.launch.roles import _worker_auth

    key = tmp_path / "peer.pem"
    assert main(["gen-identity", "--out", str(key)]) == 0
    out = capsys.readouterr().out
    assert key.exists() and "peer id:" in out and "pubkey:" in out

    ident = PeerIdentity.load(key)
    cfg = LaunchConfig.from_dict({
        "transport": {"identity_key": str(key),
                      "admitted_peers": [ident.public_key_hex]},
        "tracker": {"port": 0, "ttl": 5.0, "open_enrollment": True},
    })
    auth = _worker_auth(cfg)
    assert isinstance(auth, PeerIdentity) and auth.peer_id == ident.peer_id
    assert cfg.tracker.ttl == 5.0


def test_launch_tracker_role(tmp_path):
    from opendipaco.launch import LaunchConfig
    from opendipaco.launch.roles import run_tracker

    cfg = LaunchConfig.from_dict({"tracker": {"host": "127.0.0.1", "port": 0,
                                              "open_enrollment": True}})
    box, ready, stop = {}, threading.Event(), threading.Event()

    def on_start(t):
        box["t"] = t
        ready.set()

    th = threading.Thread(target=run_tracker, args=(cfg,),
                          kwargs=dict(on_start=on_start, stop_event=stop), daemon=True)
    th.start()
    assert ready.wait(timeout=10)
    ident = PeerIdentity.generate()
    addr = ("127.0.0.1", box["t"].port)
    assert register_peer(addr, ident, roles=["worker"])["type"] == "registered"
    assert len(fetch_directory(addr)) == 1
    stop.set()
    th.join(timeout=10)
