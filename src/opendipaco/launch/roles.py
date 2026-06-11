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
from ..routing import BagOfTokensFeaturizer, KMeansRouter
from ..schedule import (
    AsyncScheduler,
    CoordinatorServer,
    ParameterServer,
    Scheduler,
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


def _server_kw(cfg: LaunchConfig) -> dict:
    kw = dict(auth_key=cfg.transport.auth_key, accept_keys=_accept_keys(cfg),
              tls=build_tls_server(cfg))
    if cfg.transport.max_msg_bytes is not None:
        kw["max_msg_bytes"] = cfg.transport.max_msg_bytes
    return kw


def build_documents(cfg: LaunchConfig):
    """Load (or synthesize) the training documents named by the data config."""
    d = cfg.data
    if d.source == "synthetic":
        g = torch.Generator().manual_seed(cfg.run.seed)
        vocab, topics = cfg.model.vocab_size, d.synthetic_topics
        span = max(1, vocab // topics)
        per = max(1, d.num_documents // topics)
        return [torch.randint(t * span, min((t + 1) * span, vocab), (d.synthetic_doc_len,),
                              generator=g)
                for t in range(topics) for _ in range(per)]
    if d.source == "c4":
        from ..data import load_c4_documents
        return load_c4_documents(num_documents=d.num_documents,
                                 tokenizer_name=d.tokenizer or "t5-base",
                                 max_doc_tokens=d.max_doc_tokens, min_doc_tokens=d.min_doc_tokens,
                                 cache_path=d.cache_path)
    raise ValueError(f"unknown data source {d.source!r} (use 'c4' or 'synthetic')")


def build_corpus(cfg: LaunchConfig, model, docs) -> ShardedCorpus:
    """Assign documents to paths (k-means routing or round-robin) and pack them."""
    num_paths, seq_len = model.num_paths, model.sequence_length
    if cfg.data.routing == "round_robin" or len(docs) < num_paths:
        assign = torch.tensor([i % num_paths for i in range(len(docs))])
        return ShardedCorpus.from_assignments(docs, assign, num_paths, seq_len)
    feat = BagOfTokensFeaturizer(cfg.model.vocab_size,
                                 feature_dim=min(256, cfg.model.vocab_size))
    router = KMeansRouter(num_paths, seed=cfg.data.router_seed).fit(
        feat([d[:seq_len] for d in docs]))
    return ShardedCorpus.from_documents(docs, router, feat, num_paths, seq_len)


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
        staleness_weight=cfg.transport.staleness_weight, **_server_kw(cfg))
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


def run_scheduler(cfg: LaunchConfig, *, ps_addrs=None, on_start=None):
    """Sharded scheduler (no weights): drive the run against the parameter servers."""
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    docs = build_documents(cfg)
    corpus = build_corpus(cfg, model, docs)
    addrs = ps_addrs if ps_addrs is not None else [tuple(a) for a in cfg.sharded.parameter_servers]
    if not addrs:
        raise ValueError("sharded mode needs sharded.parameter_servers (or ps_addrs)")
    scheduler = Scheduler(
        model, corpus, addrs, diloco, batch_size=cfg.run.batch_size,
        host=cfg.transport.host, port=cfg.transport.port, seed=cfg.run.seed,
        staleness_bound=cfg.transport.staleness_bound,
        staleness_weight=cfg.transport.staleness_weight,
        heartbeat_timeout=cfg.transport.heartbeat_timeout,
        ps_tls=build_tls_client(cfg), **_server_kw(cfg))
    scheduler.start()
    _attach_metrics(scheduler, cfg)
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


def run_parameter_server(cfg: LaunchConfig, shard_id: int, *, port=None,
                         on_start=None, stop_event=None):
    """One parameter-server shard: own a disjoint slice of keys; serve until stopped."""
    model, diloco = dipaco_config(cfg.model), diloco_config(cfg.diloco)
    keys = sorted(model.build_topology().module_keys())
    assignment = assign_shards(keys, cfg.sharded.num_shards)
    owned = [k for k, s in assignment.items() if s == shard_id]
    ps = ParameterServer(
        model, owned, diloco, host=cfg.transport.host, port=_ps_port(cfg, shard_id, port),
        device=cfg.run.device,
        resume_dir=cfg.run.checkpoint_dir if cfg.run.resume else None, **_server_kw(cfg))
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
    if cfg.mode == "sharded":
        target = scheduler_addr or cfg.connect_addr()
        run_sharded_worker(
            model, diloco, tuple(target), device=cfg.run.device, seed=cfg.run.seed,
            auth_key=cfg.transport.auth_key, max_tasks=mt, reconnect=True,
            heartbeat_interval=cfg.transport.heartbeat_interval,
            tls=build_tls_client(cfg), tls_hostname=cfg.tls.server_hostname)
    else:
        host, port = addr or cfg.connect_addr()
        run_worker(
            model, diloco, host, port, device=cfg.run.device, seed=cfg.run.seed,
            auth_key=cfg.transport.auth_key, max_tasks=mt,
            heartbeat_interval=cfg.transport.heartbeat_interval,
            tls=build_tls_client(cfg), tls_hostname=cfg.tls.server_hostname)


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
