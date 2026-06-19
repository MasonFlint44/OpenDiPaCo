"""Run one cluster role from a :class:`LaunchConfig`.

Each ``run_*`` builds the right objects (model, diloco, corpus, TLS, auth, metrics)
from the config and drives that role: the coordinator/scheduler train to the update
budget and return; a parameter server serves until stopped; a worker leases and
trains. ``on_start`` / ``stop_event`` / address overrides exist so the all-in-one
``run`` command can wire ephemeral ports in-process; the per-role CLI commands use
the configured addresses.
"""

from __future__ import annotations

import os
import signal
import threading
import time

import torch

from ..backend import LocalBackend
from ..data import ShardedCorpus
from ..data.spec import (
    SpecCorpus,
    c4_source,
    fit_routing_from_source,
    kmeans_routing,
    make_shard_spec,
    round_robin_routing,
    synthetic_documents,
    synthetic_source,
)
from ..routing import BagOfTokensFeaturizer, KMeansRouter
from ..schedule import (
    AsyncScheduler,
    CoordinatorServer,
    ParameterServer,
    PeerIdentity,
    RateLimiter,
    Reputation,
    Scheduler,
    Tracker,
    assign_shards,
    client_context,
    derive_epoch,
    make_peer_record,
    owners_for,
    path_primary,
    run_decentralized_worker,
    run_sharded_worker,
    run_worker,
    server_context,
)
from ..schedule.throttle import TokenBucket, rate_from_mbps
from ..train import DiPaCoEngine
from .config import LaunchConfig, dipaco_config, diloco_config
from .manifest import build_manifest


# -- shared builders ---------------------------------------------------------


def build_tls_server(cfg: LaunchConfig):
    t = cfg.tls
    if not t.enabled:
        return None
    return server_context(t.certfile, t.keyfile, cafile=t.cafile,
                          require_client_cert=t.require_client_cert)


def build_tls_client(cfg: LaunchConfig):
    t = cfg.tls
    if not t.enabled:
        return None
    return client_context(cafile=t.cafile, insecure=t.insecure)


def _accept_keys(cfg: LaunchConfig):
    return list(cfg.transport.accept_keys) or None


def _server_kw(cfg: LaunchConfig, extra_admitted=None) -> dict:
    kw = dict(auth_key=cfg.transport.auth_key, accept_keys=_accept_keys(cfg),
              tls=build_tls_server(cfg))
    admitted = list(cfg.transport.admitted_peers) + list(extra_admitted or [])
    if admitted:
        kw["admitted_peers"] = admitted
    if cfg.transport.max_msg_bytes is not None:
        kw["max_msg_bytes"] = cfg.transport.max_msg_bytes
    return kw


def _scheduler_robustness_kw(cfg: LaunchConfig) -> dict:
    """Phase 3 knobs the scheduler owns: reputation, rate limiting, redundant
    execution, and the private-module policy."""
    r = cfg.robustness
    return dict(
        reputation=Reputation(floor=r.reputation_floor, credit=r.reputation_credit,
                              debit=r.reputation_debit, decay_halflife=r.reputation_halflife),
        rate_limiter=RateLimiter(capacity=r.rate_capacity,
                                 refill_per_sec=r.rate_refill_per_sec),
        min_owner_reputation=r.min_owner_reputation,
        redundancy=r.redundancy, redundancy_rate=r.redundancy_rate,
        audit_timeout=r.audit_timeout, private_policy=r.private_policy)


def _ps_robustness_kw(cfg: LaunchConfig) -> dict:
    """Phase 3 knobs the owner/parameter-server owns: robust aggregation,
    version history (for pinned redundant checks), and the private policy."""
    r = cfg.robustness
    # Pinned redundant checks need retained versions; default a small history
    # when redundancy is on and the operator hasn't set one.
    history = r.version_history
    if history <= 1 and r.redundancy_rate > 0:
        history = 4
    return dict(robustness=r.mode, aggregate=r.aggregate, quorum_target=r.quorum_target,
                quorum_timeout=r.quorum_timeout, version_history=history,
                private_policy=r.private_policy, private_quorum=r.private_quorum)


def _decentralized_owner_kw(cfg: LaunchConfig) -> dict:
    """Phase 4 knobs the owner gains in ``schedule.mode: decentralized``: it
    becomes the path coordinator (mints grants, version-fences), hosts the
    reputation / rate-limit gates, and cross-checks co-owners (quorum reads).
    Empty in ``central`` mode — the scheduler keeps those jobs."""
    if cfg.schedule.mode != "decentralized":
        return {}
    r, s, own = cfg.robustness, cfg.schedule, cfg.ownership
    return dict(
        schedule_mode="decentralized", salt=own.salt, k=own.k,
        lease_ttl=s.lease_ttl, read_quorum=s.read_quorum, directory_ttl=cfg.tracker.ttl,
        reputation=Reputation(floor=r.reputation_floor, credit=r.reputation_credit,
                              debit=r.reputation_debit, decay_halflife=r.reputation_halflife),
        rate_limiter=RateLimiter(capacity=r.rate_capacity,
                                 refill_per_sec=r.rate_refill_per_sec),
        min_owner_reputation=r.min_owner_reputation)


def _publish_manifest(server, cfg: LaunchConfig, identity) -> None:
    """Build + serve the W6 run manifest, warning the operator when it is
    unsigned (no identity) -- joiners then can't pin it (``--server-pub``), so the
    run is TOFU-only. Signing needs ``transport.identity_key``."""
    manifest = build_manifest(cfg, identity=identity)
    if "sig" not in manifest:
        print("NOTE: serving an UNSIGNED run manifest (this node has no identity); "
              "`opendipaco join` clients cannot pin it with --server-pub. Set "
              "transport.identity_key to sign it.", flush=True)
    server.serve_manifest(manifest)


def _advertise_host(cfg: LaunchConfig) -> str:
    """The address other peers dial for this owner: the explicit
    ``ownership.advertise_host``, else ``transport.connect_host``, else the
    bind host when it is a real address (same defaulting as ``connect_addr``
    -- only a wildcard bind falls back to loopback)."""
    t = cfg.transport
    return (cfg.ownership.advertise_host or t.connect_host
            or (t.host if t.host not in ("0.0.0.0", "::") else "127.0.0.1"))


def _serve_libp2p(server, cfg: LaunchConfig, identity=None):
    """When ``transport.kind == 'libp2p'``, serve this server's RPC surface over a
    *parallel* libp2p host (the TCP reactor stays the byte-for-byte anchor, W1
    D10), reserve any configured relays so a NAT'd node is reachable, and return
    the transport. ``None`` on the TCP path. The libp2p host key derives from the
    node's Ed25519 identity (D4), so it must be configured for a libp2p run."""
    if cfg.transport.kind != "libp2p":
        return None
    from ..schedule.p2p import serve_over_libp2p

    ident = _node_identity(cfg, identity, generate=True)
    # Access control over libp2p is *identity-based* (Noise authenticates the
    # peer; admitted_peers authorizes it). The HMAC auth_key -- the TCP bootstrap
    # secret -- does NOT gate the libp2p path, so a server with no allowlist
    # accepts any authenticated peer. Warn loudly rather than silently open up.
    if getattr(server, "admitted_peers", None) is None:
        print("WARNING: transport.kind: libp2p with no transport.admitted_peers -- "
              "any authenticated peer is accepted (the HMAC auth_key does not gate "
              "libp2p). Set transport.admitted_peers to restrict access.", flush=True)
    t = serve_over_libp2p(
        server, identity=ident, listen_addrs=tuple(cfg.transport.libp2p_listen),
        require_identity=True, dcutr=cfg.transport.dcutr,
        max_msg_bytes=cfg.transport.max_msg_bytes)
    # Reserve each configured relay independently (k>=2 for failover, D6). A relay
    # is an unreliable external peer: a down/refusing one must not crash the node
    # (reserve_on raises on connect failure, returns None on refusal), so isolate
    # each and proceed on the survivors. If relays were configured but NONE took,
    # a NAT'd owner has no reachable address -- fail loud rather than serve a
    # silent zombie.
    relays = cfg.transport.relays
    if relays:
        ok = 0
        for relay in relays:
            try:
                if t.reserve_on(relay) is not None:
                    ok += 1
                else:
                    print(f"WARNING: relay refused reservation: {relay}", flush=True)
            except Exception as e:  # noqa: BLE001 -- a down/bad relay is not fatal on its own
                print(f"WARNING: could not reserve on relay {relay}: {e}", flush=True)
        if ok == 0:
            t.close()
            raise RuntimeError(
                f"no relay reservation succeeded ({len(relays)} configured); a NAT'd "
                "owner would be unreachable. Check transport.relays / relay liveness.")
        if ok < len(relays):
            print(f"NOTE: reserved on {ok}/{len(relays)} relays (k>=2 recommended "
                  "for failover).", flush=True)
    return t


def _libp2p_routes(cfg: LaunchConfig) -> bool:
    """Whether this role should actually serve + route over libp2p. Rendezvous
    routing derives owner addresses from the tracker/epoch records, which carry
    TCP addresses today (not multiaddrs), so serving libp2p there would be inert.
    libp2p routing is wired for **static** sharded mode (manual multiaddrs);
    libp2p rendezvous (tracker-multiaddr discovery) rides the 0f WAN run."""
    if cfg.transport.kind != "libp2p":
        return False
    if cfg.ownership.mode == "rendezvous":
        print("NOTE: transport.kind: libp2p + ownership.mode: rendezvous -- routing "
              "still uses TCP addresses from the tracker (libp2p multiaddr discovery "
              "is not yet wired); use static sharded mode for libp2p routing.",
              flush=True)
        return False
    return True


def _node_identity(cfg: LaunchConfig, identity=None, *, generate=False):
    """This node's Ed25519 identity: an explicit object, the configured key
    file, or (rendezvous in-process smoke) a freshly generated one."""
    if identity is not None:
        return identity
    if cfg.transport.identity_key:
        return PeerIdentity.load(cfg.transport.identity_key)
    return PeerIdentity.generate() if generate else None


def _worker_auth(cfg: LaunchConfig):
    """The credential a worker presents: its Ed25519 identity when configured
    (per-peer, revocable), else the shared HMAC secret."""
    if cfg.transport.identity_key:
        return PeerIdentity.load(cfg.transport.identity_key)
    return cfg.transport.auth_key


def build_documents(cfg: LaunchConfig):
    """Load (or synthesize) the training documents named by the data config."""
    d = cfg.data
    if d.source == "synthetic":
        # Single-sourced with spec materialization, so a worker regenerating from
        # a shard spec sees byte-identical documents.
        return synthetic_documents(
            vocab_size=cfg.model.vocab_size, num_documents=d.num_documents,
            doc_len=d.synthetic_doc_len, topics=d.synthetic_topics, seed=cfg.run.seed)
    if d.source == "c4":
        from ..data import load_c4_documents
        return load_c4_documents(num_documents=d.num_documents,
                                 tokenizer_name=d.tokenizer or "t5-base",
                                 max_doc_tokens=d.max_doc_tokens, min_doc_tokens=d.min_doc_tokens,
                                 cache_path=d.cache_path)
    raise ValueError(f"unknown data source {d.source!r} (use 'c4' or 'synthetic')")


def build_server_corpus(cfg: LaunchConfig, model):
    """The scheduler/coordinator corpus, choosing the lowest-memory build.

    In spec mode with ``data.router_sample`` set the server **streams** the public
    source (bounded RAM: a sampled router fit, then a counts pass) and never holds
    the full corpus (W7b). Otherwise it loads every document and builds in hand --
    bytes mode needs the sequences to ship, and the unsampled spec fit needs all
    documents (byte-identical to before)."""
    if cfg.data.ship == "spec" and cfg.data.router_sample is not None:
        return build_spec_corpus_streaming(cfg, model)
    return build_corpus(cfg, model, build_documents(cfg))


def _spec_source(cfg: LaunchConfig) -> dict:
    """The document-source recipe for a spec corpus (synthetic or C4)."""
    d = cfg.data
    if d.source == "synthetic":
        return synthetic_source(
            vocab_size=cfg.model.vocab_size, num_documents=d.num_documents,
            doc_len=d.synthetic_doc_len, topics=d.synthetic_topics, seed=cfg.run.seed)
    if d.source == "c4":
        return c4_source(num_documents=d.num_documents,
                         tokenizer=d.tokenizer or "t5-base",
                         max_doc_tokens=d.max_doc_tokens, min_doc_tokens=d.min_doc_tokens)
    raise ValueError(f"unknown data source {d.source!r}")


def build_corpus(cfg: LaunchConfig, model, docs):
    """Build the server-side corpus: packed shards, or (``data.ship: spec``) a
    shard recipe + token counts with workers materializing shards locally."""
    if cfg.data.ship == "spec":
        return build_spec_corpus(cfg, model, docs)
    if cfg.data.ship != "bytes":
        raise ValueError(f"data.ship must be 'bytes' or 'spec', got {cfg.data.ship!r}")
    num_paths, seq_len = model.num_paths, model.sequence_length
    if cfg.data.routing == "round_robin" or len(docs) < num_paths:
        assign = torch.tensor([i % num_paths for i in range(len(docs))])
        return ShardedCorpus.from_assignments(docs, assign, num_paths, seq_len)
    feat = BagOfTokensFeaturizer(cfg.model.vocab_size,
                                 feature_dim=min(256, cfg.model.vocab_size))
    router = KMeansRouter(num_paths, seed=cfg.data.router_seed).fit(
        feat([d[:seq_len] for d in docs]))
    return ShardedCorpus.from_documents(docs, router, feat, num_paths, seq_len)


def build_spec_corpus(cfg: LaunchConfig, model, docs) -> SpecCorpus:
    """Fit the router on the docs in hand, then keep only the recipe + counts."""
    d = cfg.data
    num_paths, seq_len = model.num_paths, model.sequence_length
    source = _spec_source(cfg)
    if cfg.data.routing == "round_robin" or len(docs) < num_paths:
        routing = round_robin_routing()
    else:
        feature_dim = min(256, cfg.model.vocab_size)
        feat = BagOfTokensFeaturizer(cfg.model.vocab_size, feature_dim=feature_dim)
        router = KMeansRouter(num_paths, seed=d.router_seed).fit(
            feat([doc[:seq_len] for doc in docs]))
        routing = kmeans_routing(router.centroids, vocab_size=cfg.model.vocab_size,
                                 feature_dim=feature_dim)
    spec = make_shard_spec(source=source, routing=routing,
                           num_paths=num_paths, seq_len=seq_len)
    return SpecCorpus.from_documents(spec, docs)


def build_spec_corpus_streaming(cfg: LaunchConfig, model) -> SpecCorpus:
    """Build the spec corpus by **streaming** the public source twice -- a sampled
    router fit, then a token-count pass -- so the operator never holds the full
    corpus (W7b). Used when ``data.ship: spec`` and ``data.router_sample`` is set;
    ``SpecCorpus.build`` does the counts pass holding no sequences."""
    d = cfg.data
    num_paths, seq_len = model.num_paths, model.sequence_length
    source = _spec_source(cfg)
    if d.routing == "round_robin":
        routing = round_robin_routing()
    else:
        routing = fit_routing_from_source(
            source, num_paths=num_paths, vocab_size=cfg.model.vocab_size,
            seq_len=seq_len, sample=d.router_sample,
            feature_dim=min(256, cfg.model.vocab_size), router_seed=d.router_seed)
    spec = make_shard_spec(source=source, routing=routing,
                           num_paths=num_paths, seq_len=seq_len)
    return SpecCorpus.build(spec)


def _attach_metrics(server, cfg: LaunchConfig):
    t = cfg.transport
    if t.metrics_port is not None:
        server.start_metrics_server(host=t.metrics_host, port=t.metrics_port)
    if t.metrics_log_interval and t.metrics_log_interval > 0:
        server.start_metrics_logging(interval=t.metrics_log_interval)


def _wait_for_signal() -> threading.Event:
    """An Event set on SIGINT/SIGTERM, so a server role blocks until told to stop."""
    ev = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: ev.set())
        except (ValueError, OSError):  # not on the main thread (e.g. tests)
            pass
    return ev


# A closing node (laptop lid, SIGTERM) must hand off cleanly but never *hang*:
# the graceful path (owner drain + deregister) is internally bounded, but a hung
# peer could still stall a synchronous RPC. This is the hard backstop (design D5
# open-Q #2: a few-second budget, then abrupt).
_SHUTDOWN_DEADLINE = 10.0


def _bounded_graceful_shutdown(server, *, deadline: float = _SHUTDOWN_DEADLINE) -> None:
    """Run ``server.shutdown(graceful=True)`` under a hard deadline: if the
    handoff overruns (e.g. a hung successor/tracker), force-exit so the node
    still leaves -- the abrupt fallback the TTL+grace path already tolerates.
    Fast shutdowns cancel the timer well before it fires."""
    timer = threading.Timer(deadline, lambda: os._exit(0))
    timer.daemon = True
    timer.start()
    try:
        server.shutdown(graceful=True)
    finally:
        timer.cancel()


# -- roles -------------------------------------------------------------------


def run_coordinator(cfg: LaunchConfig, *, on_start=None):
    """Single-node async coordinator: train to the budget, then return completions."""
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    corpus = build_server_corpus(cfg, model)
    engine = DiPaCoEngine(model, diloco, LocalBackend(model.build_topology()),
                          seed=cfg.run.seed, materialize="serial")
    scheduler = AsyncScheduler(engine, lease_timeout=cfg.transport.heartbeat_timeout)
    server = CoordinatorServer(
        scheduler, corpus, batch_size=cfg.run.batch_size,
        host=cfg.transport.host, port=cfg.transport.port,
        heartbeat_timeout=cfg.transport.heartbeat_timeout,
        staleness_bound=cfg.transport.staleness_bound,
        staleness_weight=cfg.transport.staleness_weight,
        max_update_norm=cfg.transport.max_update_norm,
        compress=cfg.transport.compress,
        idle_backoff=cfg.transport.idle_backoff, **_server_kw(cfg))
    server.start()
    _attach_metrics(server, cfg)
    if on_start:
        on_start(server)
    completed = server.fit(
        num_generations=cfg.run.generations, total_generations=cfg.run.generations,
        log_every=0, checkpoint_dir=cfg.run.checkpoint_dir,
        checkpoint_every=cfg.run.checkpoint_every)
    server.shutdown()
    return server, completed


def run_scheduler(cfg: LaunchConfig, *, ps_addrs=None, on_start=None, identity=None,
                  tracker_addr=None, extra_admitted=None):
    """Sharded scheduler (no weights): drive the run against the parameter servers.

    With ``ownership.mode: rendezvous`` there are no fixed parameter servers:
    the scheduler signs owner-set epochs from tracker liveness
    (``watch_tracker``) and routing derives from the current epoch.
    """
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    corpus = build_server_corpus(cfg, model)
    own = cfg.ownership
    rendezvous = own.mode == "rendezvous"
    if own.mode not in ("static", "rendezvous"):
        raise ValueError(f"ownership.mode must be 'static' or 'rendezvous', got {own.mode!r}")
    ident = _node_identity(cfg, identity, generate=rendezvous)
    if rendezvous:
        addrs = []
    elif ps_addrs is not None:
        addrs = ps_addrs
    elif cfg.transport.kind == "libp2p":
        # libp2p PS addrs are multiaddr strings (each owner prints its own); pass
        # them through untouched -- tupling them would shred the string.
        addrs = list(cfg.sharded.parameter_servers)
        if not addrs:
            raise ValueError("sharded libp2p mode needs sharded.parameter_servers (multiaddrs)")
    else:
        addrs = [tuple(a) for a in cfg.sharded.parameter_servers]
        if not addrs:
            raise ValueError("sharded mode needs sharded.parameter_servers (or ps_addrs)")
    scheduler = Scheduler(
        model, corpus, addrs, diloco, batch_size=cfg.run.batch_size,
        host=cfg.transport.host, port=cfg.transport.port, seed=cfg.run.seed,
        staleness_bound=cfg.transport.staleness_bound,
        staleness_weight=cfg.transport.staleness_weight,
        heartbeat_timeout=cfg.transport.heartbeat_timeout,
        ps_tls=build_tls_client(cfg), grant_key=cfg.transport.grant_key,
        identity=ident, compress=cfg.transport.compress, down=cfg.transport.down,
        up_density=cfg.transport.up_density, idle_backoff=cfg.transport.idle_backoff,
        task_seconds=cfg.run.task_seconds, park_factor=cfg.run.park_factor,
        min_task_rate=cfg.run.min_task_rate, tailor_bandwidth=cfg.transport.tailor_bandwidth,
        **_scheduler_robustness_kw(cfg), **_server_kw(cfg, extra_admitted))
    scheduler.start()
    _publish_manifest(scheduler, cfg, ident)  # W6: flags-only `join`
    lp = _serve_libp2p(scheduler, cfg, ident) if _libp2p_routes(cfg) else None
    if lp is not None:
        print(f"scheduler libp2p addrs: {lp.addrs}", flush=True)
    _attach_metrics(scheduler, cfg)
    if rendezvous:
        if cfg.run.resume and cfg.run.checkpoint_dir:
            # Load the clock + epoch floor + manifest *before* the first epoch
            # is published: a resumed run must never re-flag bootstrap nor
            # restart epoch numbering below what live owners hold.
            scheduler._load_state(cfg.run.checkpoint_dir)
        scheduler.watch_tracker(
            tracker_addr or cfg.tracker_connect_addr(), k=own.k, salt=own.salt,
            owner_grace=own.owner_grace, min_epoch_interval=own.min_epoch_interval,
            poll_interval=own.epoch_poll_interval, tracker_auth=cfg.transport.auth_key,
            tracker_tls=build_tls_client(cfg))
    if on_start:
        on_start(scheduler)
    completed = scheduler.fit(
        num_generations=cfg.run.generations, total_generations=cfg.run.generations,
        checkpoint_dir=cfg.run.checkpoint_dir, checkpoint_every=cfg.run.checkpoint_every,
        resume=cfg.run.resume)
    scheduler.shutdown()
    if lp is not None:
        lp.close()
    return scheduler, completed


def _ps_port(cfg: LaunchConfig, shard_id: int, port) -> int:
    if port is not None:
        return port
    servers = cfg.sharded.parameter_servers
    if shard_id < len(servers):
        return int(servers[shard_id][1])
    return 0


def run_parameter_server(cfg: LaunchConfig, shard_id: int = 0, *, port=None,
                         on_start=None, stop_event=None, identity=None,
                         scheduler_addr=None, tracker_addr=None, scheduler_pub=None,
                         extra_admitted=None):
    """One parameter server: own module keys; serve until stopped.

    Static mode owns a fixed ``assign_shards`` slice (``shard_id``). With
    ``ownership.mode: rendezvous`` the node is a dynamic **owner**: it
    heartbeats the tracker, polls the scheduler for owner-set epochs, and its
    keys (with replication, promotion, and per-key checkpoints) follow from
    them -- ``shard_id`` is ignored.
    """
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    own = cfg.ownership
    resume_dir = cfg.run.checkpoint_dir if cfg.run.resume else None
    common = dict(
        host=cfg.transport.host, device=cfg.run.device,
        grant_key=cfg.transport.grant_key,
        max_update_norm=cfg.transport.max_update_norm, compress=cfg.transport.compress,
        down=cfg.transport.down, resume_dir=resume_dir, **_ps_robustness_kw(cfg))
    node_ident = None
    if own.mode == "rendezvous":
        ident = node_ident = _node_identity(cfg, identity, generate=True)
        decentralized = cfg.schedule.mode == "decentralized"
        # Decentralized: no scheduler to poll for epochs (they're gossip-derived).
        sched_addr = None if decentralized else (scheduler_addr or cfg.connect_addr())
        ps = ParameterServer(
            model, [], diloco, port=port if port is not None else 0,
            identity=ident,
            scheduler_pub=scheduler_pub or cfg.transport.scheduler_pub,
            scheduler_addr=sched_addr,
            replicate_interval=own.replicate_interval, bank_seed=own.bank_seed,
            peer_tls=build_tls_client(cfg),
            **_decentralized_owner_kw(cfg),
            **common, **_server_kw(cfg, extra_admitted))
        ps.start()
        ps.start_tracker_heartbeat(
            tracker_addr or cfg.tracker_connect_addr(), _advertise_host(cfg),
            interval=own.heartbeat_interval, auth_key=cfg.transport.auth_key,
            tls=build_tls_client(cfg))
    else:
        keys = sorted(model.build_topology().module_keys())
        assignment = assign_shards(keys, cfg.sharded.num_shards)
        owned = [k for k, s in assignment.items() if s == shard_id]
        ps = ParameterServer(model, owned, diloco, port=_ps_port(cfg, shard_id, port),
                             **common, **_server_kw(cfg, extra_admitted))
        ps.start()
    try:
        lp = _serve_libp2p(ps, cfg, node_ident or identity) if _libp2p_routes(cfg) else None
    except Exception:
        ps.shutdown()   # don't leak the started TCP reactor if libp2p bring-up fails
        raise
    if lp is not None:
        addrs = lp.circuit_addrs or lp.addrs   # advertise relay addrs for a NAT'd owner
        print(f"owner libp2p addrs: {addrs}", flush=True)
    _attach_metrics(ps, cfg)
    if on_start:
        on_start(ps)
    (stop_event or _wait_for_signal()).wait()
    # Graceful leave (W4d): drain to rank-1 + signed deregister so failover skips
    # owner_grace and loses ~nothing. (No-op in static mode: no epoch, no tracker.)
    # The os._exit deadline backstop applies only to the real CLI signal path;
    # an injected stop_event (run_local / tests) returns to its caller normally.
    if stop_event is None:
        _bounded_graceful_shutdown(ps)
    else:
        ps.shutdown(graceful=True)
    if lp is not None:
        lp.close()
    return ps


def run_worker_role(cfg: LaunchConfig, *, addr=None, scheduler_addr=None, max_tasks=None,
                    stop_event=None, bucket=None):
    """A worker: connect to the coordinator (or scheduler in sharded mode) and train.

    ``bucket`` (W6b) is a shared bandwidth :class:`TokenBucket` throttling the
    worker's sockets; when not passed it is built from ``transport.max_mbps`` (so
    `opendipaco worker` honors the cap too). ``opendipaco join`` passes its own so
    the health line can read the byte counters."""
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    mt = max_tasks if max_tasks is not None else cfg.run.max_tasks
    data_dir = cfg.data.shard_cache_dir  # spec mode: cache materialized shards here
    auth = _worker_auth(cfg)
    if bucket is None and cfg.transport.max_mbps:
        rate = rate_from_mbps(cfg.transport.max_mbps)
        bucket = TokenBucket(rate) if rate else None
    # SIGTERM/SIGINT -> graceful leave (W4d): the sharded worker nacks its in-flight
    # lease so the path re-leases at once instead of waiting out the lease timeout.
    # Only installed on the sharded paths that *consume* the event -- the
    # coordinator path (run_worker) has no stop hook, so installing a handler
    # there would swallow SIGTERM and make the worker unkillable but by SIGKILL.
    if cfg.transport.kind == "libp2p":
        # libp2p worker: dials the scheduler's multiaddr over Noise streams
        # (NAT-traversing via relays/DCUtR). Sharded/owner topology only -- the
        # single-coordinator path has no libp2p worker loop.
        if cfg.mode != "sharded":
            raise ValueError("transport.kind: libp2p is supported for sharded mode only")
        if bucket is not None:
            print("WARNING: --max-mbps / transport.max_mbps is NOT enforced on the libp2p "
                  "transport yet (the cap throttles the TCP path only); your bandwidth is "
                  "uncapped on this run.", flush=True)
        target = scheduler_addr or cfg.transport.connect_libp2p
        if not target:
            raise ValueError("libp2p worker needs transport.connect_libp2p (scheduler multiaddr)")
        run_sharded_worker(
            model, diloco, target, device=cfg.run.device, seed=cfg.run.seed,
            max_tasks=mt, reconnect=True,
            heartbeat_interval=cfg.transport.heartbeat_interval,
            data_dir=data_dir, max_batch_size=cfg.run.worker_max_batch,
            max_shards=cfg.run.worker_max_shards,
            verify_routing=bool(cfg.run.verify_routing),
            transport="libp2p", identity=_node_identity(cfg, generate=True),
            stop_event=stop_event or _wait_for_signal())
        return
    if cfg.schedule.mode == "decentralized":
        # No scheduler: self-assign from the gossiped directory. The worker builds
        # the corpus locally (the shard source) and needs an identity (it is
        # HRW-scored by peer_id and derives epochs).
        own = cfg.ownership
        corpus = build_corpus(cfg, model, build_documents(cfg))
        run_decentralized_worker(
            model, diloco, tuple(cfg.tracker_connect_addr()), corpus,
            identity=_node_identity(cfg, generate=True), device=cfg.run.device,
            seed=cfg.run.seed, auth_key=_worker_auth(cfg), k=own.k, salt=own.salt,
            read_quorum=cfg.schedule.read_quorum, lease_ttl=cfg.schedule.lease_ttl,
            batch_size=cfg.run.batch_size, total_rounds=cfg.run.generations,
            max_tasks=mt, heartbeat_interval=own.heartbeat_interval,
            tls=build_tls_client(cfg), stop_event=stop_event or _wait_for_signal(),
            bucket=bucket)
        return
    if cfg.mode == "sharded":
        target = scheduler_addr or cfg.connect_addr()
        run_sharded_worker(
            model, diloco, tuple(target), device=cfg.run.device, seed=cfg.run.seed,
            auth_key=auth, max_tasks=mt, reconnect=True,
            heartbeat_interval=cfg.transport.heartbeat_interval,
            tls=build_tls_client(cfg), tls_hostname=cfg.tls.server_hostname,
            data_dir=data_dir, max_batch_size=cfg.run.worker_max_batch,
            max_shards=cfg.run.worker_max_shards,
            verify_routing=bool(cfg.run.verify_routing),
            stop_event=stop_event or _wait_for_signal(), bucket=bucket)
    else:
        if bucket is not None:
            print("WARNING: transport.max_mbps is NOT enforced in coordinator mode (the "
                  "single-node worker path is not throttle-wired); your bandwidth is "
                  "uncapped. Use sharded/decentralized mode for the cap.", flush=True)
        host, port = addr or cfg.connect_addr()
        run_worker(
            model, diloco, host, port, device=cfg.run.device, seed=cfg.run.seed,
            auth_key=auth, max_tasks=mt,
            heartbeat_interval=cfg.transport.heartbeat_interval,
            tls=build_tls_client(cfg), tls_hostname=cfg.tls.server_hostname,
            data_dir=data_dir, max_batch_size=cfg.run.worker_max_batch,
            max_shards=cfg.run.worker_max_shards,
            verify_routing=bool(cfg.run.verify_routing))


def run_local(cfg: LaunchConfig):
    """Stand up the *whole* cluster in one process (for local runs / smoke tests).

    Coordinator mode: a coordinator + ``run.local_workers`` workers. Sharded mode: the
    scheduler + ``sharded.num_shards`` parameter servers + workers. Ephemeral ports are
    wired automatically. Returns ``(server, completed)`` for the driving server.
    """
    if cfg.schedule.mode == "decentralized":
        return _run_local_decentralized(cfg)
    return _run_local_sharded(cfg) if cfg.mode == "sharded" else _run_local_coordinator(cfg)


def _run_local_coordinator(cfg: LaunchConfig):
    box, ready, result = {}, threading.Event(), {}

    def on_start(server):
        box["server"] = server
        ready.set()

    ct = threading.Thread(
        target=lambda: result.__setitem__("c", run_coordinator(cfg, on_start=on_start)),
        daemon=True)
    ct.start()
    if not ready.wait(timeout=120):
        raise RuntimeError("coordinator did not start")
    port = box["server"].port
    workers = [threading.Thread(target=run_worker_role,
                                kwargs=dict(cfg=cfg, addr=("127.0.0.1", port)), daemon=True)
               for _ in range(cfg.run.local_workers)]
    for w in workers:
        w.start()
    ct.join()                       # coordinator trains to budget, then shuts down
    for w in workers:
        w.join(timeout=5)           # daemons; they exit once the coordinator is gone
    return result["c"]


def _run_local_sharded(cfg: LaunchConfig):
    if cfg.ownership.mode == "rendezvous":
        return _run_local_rendezvous(cfg)
    n = cfg.sharded.num_shards
    stops = [threading.Event() for _ in range(n)]
    readies = [threading.Event() for _ in range(n)]
    boxes: list[dict] = [{} for _ in range(n)]

    def ps_target(i):
        def on_start(ps):
            boxes[i]["ps"] = ps
            readies[i].set()
        return lambda: run_parameter_server(cfg, i, port=0, on_start=on_start, stop_event=stops[i])

    ps_threads = [threading.Thread(target=ps_target(i), daemon=True) for i in range(n)]
    for t in ps_threads:
        t.start()
    for r in readies:
        if not r.wait(timeout=60):
            raise RuntimeError("a parameter server did not start")
    ps_addrs = [("127.0.0.1", boxes[i]["ps"].port) for i in range(n)]

    sbox, sready, sresult = {}, threading.Event(), {}

    def on_sched_start(s):
        sbox["s"] = s
        sready.set()

    st = threading.Thread(
        target=lambda: sresult.__setitem__(
            "r", run_scheduler(cfg, ps_addrs=ps_addrs, on_start=on_sched_start)), daemon=True)
    st.start()
    if not sready.wait(timeout=60):
        raise RuntimeError("scheduler did not start")
    sport = sbox["s"].port
    workers = [threading.Thread(
        target=run_worker_role,
        kwargs=dict(cfg=cfg, scheduler_addr=("127.0.0.1", sport)), daemon=True)
        for _ in range(cfg.run.local_workers)]
    for w in workers:
        w.start()
    st.join()                       # scheduler trains to budget, then shuts down
    for s in stops:
        s.set()                     # tell the parameter servers to stop
    for t in ps_threads:
        t.join(timeout=10)
    for w in workers:
        w.join(timeout=5)
    return sresult["r"]


def _run_local_rendezvous(cfg: LaunchConfig):
    """Whole rendezvous cluster in one process: tracker + scheduler (epoch
    authority) + ``sharded.num_shards`` dynamic owners + workers, with
    ephemeral ports and freshly generated identities."""
    import os as _os

    n = cfg.sharded.num_shards
    if cfg.transport.auth_key is None:
        # The owners' identity auth makes servers challenge everyone; give the
        # workers/scheduler a shared secret so they can answer too.
        cfg.transport.auth_key = _os.urandom(16).hex()
    sched_id = PeerIdentity.generate()
    owner_ids = [PeerIdentity.generate() for _ in range(n)]
    owner_pubs = [i.public_key_hex for i in owner_ids]

    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True,
                      ttl=cfg.tracker.ttl, auth_key=cfg.transport.auth_key)
    tracker.start()
    taddr = ("127.0.0.1", tracker.port)

    sbox, sready, sresult = {}, threading.Event(), {}

    def on_sched_start(s):
        sbox["s"] = s
        sready.set()

    st = threading.Thread(
        target=lambda: sresult.__setitem__("r", run_scheduler(
            cfg, on_start=on_sched_start, identity=sched_id, tracker_addr=taddr,
            extra_admitted=owner_pubs)), daemon=True)
    st.start()
    if not sready.wait(timeout=60):
        raise RuntimeError("scheduler did not start")
    saddr = ("127.0.0.1", sbox["s"].port)

    stops = [threading.Event() for _ in range(n)]
    ps_threads = [threading.Thread(target=run_parameter_server, kwargs=dict(
        cfg=cfg, port=0, identity=owner_ids[i], scheduler_addr=saddr,
        tracker_addr=taddr, scheduler_pub=sched_id.public_key_hex,
        extra_admitted=[p for j, p in enumerate(owner_pubs) if j != i],
        stop_event=stops[i]), daemon=True) for i in range(n)]
    for t in ps_threads:
        t.start()
    workers = [threading.Thread(target=run_worker_role,
                                kwargs=dict(cfg=cfg, scheduler_addr=saddr), daemon=True)
               for _ in range(cfg.run.local_workers)]
    for w in workers:
        w.start()
    st.join()                       # scheduler trains to budget, then shuts down
    for s in stops:
        s.set()
    for t in ps_threads:
        t.join(timeout=10)
    tracker.shutdown()
    for w in workers:
        w.join(timeout=5)
    return sresult["r"]


_DECENTRALIZED_RUN_TIMEOUT = 120.0  # safety: a stalled owner mustn't hang the harness


class _DecentralizedCluster:
    """Return handle for a local decentralized run: a representative ``metrics``
    (so the CLI ``_report`` works like the other roles) plus the merged bank
    snapshotted from the owners before shutdown."""

    def __init__(self, metrics, bank):
        self.metrics = metrics
        self._bank = bank

    def merged_bank(self) -> dict:
        """``{key: cpu state_dict}`` gathered across the owners' primaries."""
        return self._bank


def _gather_merged_bank(owner_by_id, epoch, topology) -> dict:
    bank = {}
    for key in topology.module_keys():
        owners = owners_for(key, epoch)
        ps = owner_by_id.get(owners[0]["peer_id"]) if owners else None
        if ps is None or key not in ps.bank:
            continue
        with ps._lock:
            bank[key] = {n: v.detach().cpu().clone()
                         for n, v in ps.bank[key].state_dict().items()}
    return bank


def _run_local_decentralized(cfg: LaunchConfig):
    """Whole decentralized swarm in one process (design D7): tracker (seed) +
    ``sharded.num_shards`` gossip-derived owners (no scheduler) + self-assigning
    workers, trained to a per-path generation target.

    The owners cold-start from a **bootstrap epoch** built from their own started
    addresses, so every owner boot-serves its seeded ``(0, 0)`` bank immediately
    (a fresh decentralized cluster has nobody to sync from). ``derive_epoch`` sets
    a ``members_sig`` so each owner's gossip loop re-derives the *same* record and
    membership stays stable until a real change (e.g. a Byzantine eviction)."""
    n = cfg.sharded.num_shards
    own = cfg.ownership
    if cfg.transport.auth_key is None:
        # Owners challenge by identity; a shared secret lets the (HMAC) workers
        # answer too -- same bootstrap as the rendezvous path.
        cfg.transport.auth_key = os.urandom(16).hex()
    auth = cfg.transport.auth_key
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    corpus = build_server_corpus(cfg, model)
    topo = model.build_topology()

    owner_ids = [PeerIdentity.generate() for _ in range(n)]
    owner_pubs = [i.public_key_hex for i in owner_ids]
    tracker = Tracker(host="127.0.0.1", port=0, open_enrollment=True,
                      ttl=cfg.tracker.ttl, auth_key=auth)
    tracker.start()
    taddr = ("127.0.0.1", tracker.port)

    weights = {p: corpus.shard_weight(p) for p in range(model.num_paths)}
    common = dict(host="127.0.0.1", device=cfg.run.device, grant_key=cfg.transport.grant_key,
                  max_update_norm=cfg.transport.max_update_norm,
                  compress=cfg.transport.compress, down=cfg.transport.down,
                  **_ps_robustness_kw(cfg))
    owners = []
    for i in range(n):
        extra = [p for j, p in enumerate(owner_pubs) if j != i]
        ps = ParameterServer(
            model, [], diloco, port=0, identity=owner_ids[i],
            replicate_interval=own.replicate_interval, bank_seed=own.bank_seed,
            corpus_weights=weights, peer_tls=build_tls_client(cfg),
            **_decentralized_owner_kw(cfg), **common, **_server_kw(cfg, extra))
        ps.start()
        owners.append(ps)
    owner_by_id = {ps.peer_id: ps for ps in owners}

    # Bootstrap epoch from the started owners' real addresses; every owner serves
    # its (0, 0) bank at once. members_sig makes the gossip re-derive idempotent.
    recs = [make_peer_record(owner_ids[i], reachability="public",
                             addr=("127.0.0.1", owners[i].port), roles=("owner",))
            for i in range(n)]
    epoch0 = derive_epoch(recs, k=own.k, salt=own.salt, prev=None)
    for ps in owners:
        ps.apply_epoch(epoch0, bootstrap=True)
        ps.start_tracker_heartbeat(taddr, "127.0.0.1",
                                   interval=own.heartbeat_interval, auth_key=auth)

    stop = threading.Event()

    def worker_target():
        run_decentralized_worker(
            model, diloco, taddr, corpus, identity=PeerIdentity.generate(),
            device=cfg.run.device, seed=cfg.run.seed, auth_key=auth,
            k=own.k, salt=own.salt, read_quorum=cfg.schedule.read_quorum,
            lease_ttl=cfg.schedule.lease_ttl, batch_size=cfg.run.batch_size,
            total_rounds=cfg.run.generations,
            heartbeat_interval=own.heartbeat_interval, stop_event=stop)

    workers = [threading.Thread(target=worker_target, daemon=True)
               for _ in range(cfg.run.local_workers)]
    for w in workers:
        w.start()

    target = cfg.run.generations
    deadline = time.monotonic() + _DECENTRALIZED_RUN_TIMEOUT
    completed: dict = {}
    try:
        while time.monotonic() < deadline:
            done = True
            for path in topo.paths():
                prim = path_primary(topo.path_module_keys(path), epoch0)
                ps = owner_by_id.get(prim["peer_id"]) if prim else None
                if ps is None:        # primary remapped off the bootstrap epoch
                    done = False
                    continue
                with ps._lock:
                    g = ps._gen.get(path, [0])[0]
                completed[topo.path_index(path)] = g
                if g < target:
                    done = False
            if done:
                break
            time.sleep(0.05)
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=10)
        merged = _gather_merged_bank(owner_by_id, epoch0, topo)  # before shutdown
        for ps in owners:
            ps.shutdown(graceful=True)
        tracker.shutdown()
    if any(completed.get(p, 0) < target for p in range(model.num_paths)):
        # Don't pretend a stalled run succeeded: a partial result here means an
        # owner/worker wedged (the safety timeout fired) -- surface it loudly.
        print(f"WARNING: decentralized run_local hit the {_DECENTRALIZED_RUN_TIMEOUT:.0f}s "
              f"timeout before every path reached generation {target}: {completed}",
              flush=True)
    return _DecentralizedCluster(owners[0].metrics, merged), completed


def run_relay(cfg: LaunchConfig, *, on_start=None, stop_event=None):
    """A Circuit Relay v2 **relay** node (W1 D6): a public peer that forwards
    other peers' traffic so NAT'd owners are reachable. It runs ``allow_hop`` and
    no RPC handler -- relayed streams are Noise-secured end-to-end, so it only
    ever sees ciphertext (D7). Prints its dialable multiaddrs; wire one (k>=2 for
    failover) into each NAT'd peer's ``transport.relays``. Needs the ``[nat]``
    extra and ``transport.identity_key`` (the relay's libp2p id derives from it)."""
    from ..schedule.p2p import Libp2pTransport

    ident = _node_identity(cfg, generate=True)
    relay = Libp2pTransport(ident, relay=True,
                            listen_addrs=tuple(cfg.transport.libp2p_listen),
                            dcutr=cfg.transport.dcutr).start()
    print(f"relay peer id: {ident.peer_id}", flush=True)
    for a in relay.addrs:
        print(f"relay addr: {a}", flush=True)
    if on_start:
        on_start(relay)
    (stop_event or _wait_for_signal()).wait()
    relay.close()
    return relay


def run_tracker(cfg: LaunchConfig, *, on_start=None, stop_event=None):
    """The rendezvous directory (Phase 1): peers register self-certifying signed
    records; serve until stopped. Enrollment/auth from the ``tracker`` and
    ``transport`` config sections."""
    t = cfg.tracker
    tracker = Tracker(host=t.host, port=t.port, ttl=t.ttl,
                      open_enrollment=t.open_enrollment,
                      enroll_peers=list(t.enroll_peers), **_server_kw(cfg))
    tracker.start()
    # W6: serve the run manifest so a decentralized volunteer can `opendipaco join`
    # with just this tracker's address (signed by the node identity when set).
    _publish_manifest(tracker, cfg, _node_identity(cfg))
    _attach_metrics(tracker, cfg)
    if on_start:
        on_start(tracker)
    (stop_event or _wait_for_signal()).wait()
    tracker.shutdown()
    return tracker


def run_ingest(cfg: LaunchConfig, shard_id: int):
    """Resumably ingest this host's data shard to ``data.shard_cache_dir``."""
    from ..data import ingest_c4_shard, load_tokenizer

    d = cfg.data
    if not d.shard_cache_dir:
        raise ValueError("ingest needs data.shard_cache_dir")
    if d.source != "c4":
        raise ValueError("ingest is for the 'c4' data source")
    tok = load_tokenizer(d.tokenizer or "t5-base")
    return ingest_c4_shard(
        d.shard_cache_dir, shard_id=shard_id, num_shards=cfg.sharded.num_shards,
        target_docs=d.num_documents, tokenizer=tok, max_doc_tokens=d.max_doc_tokens,
        min_doc_tokens=d.min_doc_tokens)
