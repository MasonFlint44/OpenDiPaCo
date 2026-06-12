"""Run one cluster role from a :class:`LaunchConfig`.

Each ``run_*`` builds the right objects (model, diloco, corpus, TLS, auth, metrics)
from the config and drives that role: the coordinator/scheduler train to the update
budget and return; a parameter server serves until stopped; a worker leases and
trains. ``on_start`` / ``stop_event`` / address overrides exist so the all-in-one
``run`` command can wire ephemeral ports in-process; the per-role CLI commands use
the configured addresses.
"""

from __future__ import annotations

import signal
import threading

import torch

from ..backend import LocalBackend
from ..data import ShardedCorpus
from ..data.spec import (
    SpecCorpus,
    c4_source,
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
    Scheduler,
    Tracker,
    assign_shards,
    client_context,
    run_sharded_worker,
    run_worker,
    server_context,
)
from ..train import DiPaCoEngine
from .config import LaunchConfig, dipaco_config, diloco_config


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
    if d.source == "synthetic":
        source = synthetic_source(
            vocab_size=cfg.model.vocab_size, num_documents=d.num_documents,
            doc_len=d.synthetic_doc_len, topics=d.synthetic_topics, seed=cfg.run.seed)
    elif d.source == "c4":
        source = c4_source(num_documents=d.num_documents,
                           tokenizer=d.tokenizer or "t5-base",
                           max_doc_tokens=d.max_doc_tokens,
                           min_doc_tokens=d.min_doc_tokens)
    else:
        raise ValueError(f"unknown data source {d.source!r}")
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


# -- roles -------------------------------------------------------------------


def run_coordinator(cfg: LaunchConfig, *, on_start=None):
    """Single-node async coordinator: train to the budget, then return completions."""
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    docs = build_documents(cfg)
    corpus = build_corpus(cfg, model, docs)
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
    docs = build_documents(cfg)
    corpus = build_corpus(cfg, model, docs)
    own = cfg.ownership
    rendezvous = own.mode == "rendezvous"
    if own.mode not in ("static", "rendezvous"):
        raise ValueError(f"ownership.mode must be 'static' or 'rendezvous', got {own.mode!r}")
    ident = _node_identity(cfg, identity, generate=rendezvous)
    if rendezvous:
        addrs = []
    else:
        addrs = (ps_addrs if ps_addrs is not None
                 else [tuple(a) for a in cfg.sharded.parameter_servers])
        if not addrs:
            raise ValueError("sharded mode needs sharded.parameter_servers (or ps_addrs)")
    scheduler = Scheduler(
        model, corpus, addrs, diloco, batch_size=cfg.run.batch_size,
        host=cfg.transport.host, port=cfg.transport.port, seed=cfg.run.seed,
        staleness_bound=cfg.transport.staleness_bound,
        staleness_weight=cfg.transport.staleness_weight,
        heartbeat_timeout=cfg.transport.heartbeat_timeout,
        ps_tls=build_tls_client(cfg), grant_key=cfg.transport.grant_key,
        identity=ident, compress=cfg.transport.compress,
        idle_backoff=cfg.transport.idle_backoff, **_server_kw(cfg, extra_admitted))
    scheduler.start()
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
        resume_dir=resume_dir)
    if own.mode == "rendezvous":
        ident = _node_identity(cfg, identity, generate=True)
        ps = ParameterServer(
            model, [], diloco, port=port if port is not None else 0,
            identity=ident,
            scheduler_pub=scheduler_pub or cfg.transport.scheduler_pub,
            scheduler_addr=scheduler_addr or cfg.connect_addr(),
            replicate_interval=own.replicate_interval, bank_seed=own.bank_seed,
            peer_tls=build_tls_client(cfg),
            **common, **_server_kw(cfg, extra_admitted))
        ps.start()
        advertise = own.advertise_host or cfg.transport.connect_host or "127.0.0.1"
        ps.start_tracker_heartbeat(
            tracker_addr or cfg.tracker_connect_addr(), advertise,
            interval=own.heartbeat_interval, auth_key=cfg.transport.auth_key,
            tls=build_tls_client(cfg))
    else:
        keys = sorted(model.build_topology().module_keys())
        assignment = assign_shards(keys, cfg.sharded.num_shards)
        owned = [k for k, s in assignment.items() if s == shard_id]
        ps = ParameterServer(model, owned, diloco, port=_ps_port(cfg, shard_id, port),
                             **common, **_server_kw(cfg, extra_admitted))
        ps.start()
    _attach_metrics(ps, cfg)
    if on_start:
        on_start(ps)
    (stop_event or _wait_for_signal()).wait()
    ps.shutdown()
    return ps


def run_worker_role(cfg: LaunchConfig, *, addr=None, scheduler_addr=None, max_tasks=None):
    """A worker: connect to the coordinator (or scheduler in sharded mode) and train."""
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    mt = max_tasks if max_tasks is not None else cfg.run.max_tasks
    data_dir = cfg.data.shard_cache_dir  # spec mode: cache materialized shards here
    auth = _worker_auth(cfg)
    if cfg.mode == "sharded":
        target = scheduler_addr or cfg.connect_addr()
        run_sharded_worker(
            model, diloco, tuple(target), device=cfg.run.device, seed=cfg.run.seed,
            auth_key=auth, max_tasks=mt, reconnect=True,
            heartbeat_interval=cfg.transport.heartbeat_interval,
            tls=build_tls_client(cfg), tls_hostname=cfg.tls.server_hostname,
            data_dir=data_dir, max_batch_size=cfg.run.worker_max_batch)
    else:
        host, port = addr or cfg.connect_addr()
        run_worker(
            model, diloco, host, port, device=cfg.run.device, seed=cfg.run.seed,
            auth_key=auth, max_tasks=mt,
            heartbeat_interval=cfg.transport.heartbeat_interval,
            tls=build_tls_client(cfg), tls_hostname=cfg.tls.server_hostname,
            data_dir=data_dir, max_batch_size=cfg.run.worker_max_batch)


def run_local(cfg: LaunchConfig):
    """Stand up the *whole* cluster in one process (for local runs / smoke tests).

    Coordinator mode: a coordinator + ``run.local_workers`` workers. Sharded mode: the
    scheduler + ``sharded.num_shards`` parameter servers + workers. Ephemeral ports are
    wired automatically. Returns ``(server, completed)`` for the driving server.
    """
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


def run_tracker(cfg: LaunchConfig, *, on_start=None, stop_event=None):
    """The rendezvous directory (Phase 1): peers register self-certifying signed
    records; serve until stopped. Enrollment/auth from the ``tracker`` and
    ``transport`` config sections."""
    t = cfg.tracker
    tracker = Tracker(host=t.host, port=t.port, ttl=t.ttl,
                      open_enrollment=t.open_enrollment,
                      enroll_peers=list(t.enroll_peers), **_server_kw(cfg))
    tracker.start()
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
