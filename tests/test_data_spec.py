"""Tests for shard specs (data/spec.py; internet-scale plan §0d).

The contract under test: a worker materializing shard ``i`` from a spec produces
*exactly* the shard the server-side ``ShardedCorpus`` would have shipped as
bytes — same documents, same routing, same packing — and the servers in spec
mode put zero corpus bytes on the wire while training still runs end to end.
"""

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
from opendipaco.data import ShardedCorpus, SpecCorpus, materialize_shard, spec_fingerprint
from opendipaco.data.spec import (
    c4_source,
    fit_routing_from_source,
    iter_spec_documents,
    kmeans_routing,
    make_shard_spec,
    round_robin_routing,
    synthetic_documents,
    synthetic_source,
    verify_routing,
)
from opendipaco.routing import BagOfTokensFeaturizer, KMeansRouter
from opendipaco.schedule import (
    CoordinatorServer,
    ParameterServer,
    Scheduler,
    assign_shards,
    run_sharded_worker,
    run_worker,
)

BATCH = 8
GENS = 3
VOCAB, SEQ, PATHS = 48, 16, 4


def _cfg():
    bb = BackboneConfig(
        vocab_size=VOCAB, hidden_size=32, num_attention_heads=4, intermediate_size=64,
        layers_per_level=[1, 1], max_position_embeddings=64,
    )
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=SEQ)


def _diloco():
    return DiLoCoConfig(inner_steps=4, inner_lr=1e-3)


def _source_spec():
    return synthetic_source(vocab_size=VOCAB, num_documents=32, doc_len=48,
                            topics=4, seed=0)


def _docs():
    return synthetic_documents(vocab_size=VOCAB, num_documents=32, doc_len=48,
                               topics=4, seed=0)


def _kmeans_spec(docs, prefix_len=32):
    feat = BagOfTokensFeaturizer(VOCAB, feature_dim=VOCAB)
    router = KMeansRouter(PATHS, seed=0).fit(feat([d[:SEQ] for d in docs]))
    return make_shard_spec(
        source=_source_spec(),
        routing=kmeans_routing(router.centroids, vocab_size=VOCAB, feature_dim=VOCAB),
        num_paths=PATHS, seq_len=SEQ, prefix_len=prefix_len,
    ), feat, router


def _engine(cfg, seed=0):
    return DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                        seed=seed, materialize="serial")


def _snap(bank):
    return {k: {n: p.detach().clone() for n, p in m.named_parameters()}
            for k, m in bank.items()}


def _maxdiff(a, b):
    return max(float((a[k][n] - b[k][n]).abs().max()) for k in a for n in a[k])


# -- determinism + parity with the bytes path ------------------------------------


def test_synthetic_documents_deterministic():
    a, b = _docs(), _docs()
    assert len(a) == len(b) == 32
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_materialized_shards_match_sharded_corpus_kmeans():
    """Worker-side materialization reproduces the server's ShardedCorpus shards
    bit-for-bit: same docs, same routing, same packing."""
    docs = _docs()
    spec, feat, router = _kmeans_spec(docs)
    reference = ShardedCorpus.from_documents(docs, router, feat, PATHS, SEQ)
    corpus = SpecCorpus.from_documents(spec, docs)
    for p in range(PATHS):
        assert torch.equal(materialize_shard(spec, p), reference.shard(p))
        assert corpus.shard_weight(p) == pytest.approx(reference.shard_weight(p))


def test_materialized_shards_match_round_robin():
    docs = _docs()
    spec = make_shard_spec(source=_source_spec(), routing=round_robin_routing(),
                           num_paths=PATHS, seq_len=SEQ)
    assign = torch.tensor([i % PATHS for i in range(len(docs))])
    reference = ShardedCorpus.from_assignments(docs, assign, PATHS, SEQ)
    for p in range(PATHS):
        assert torch.equal(materialize_shard(spec, p), reference.shard(p))


def test_spec_corpus_build_streams_counts():
    """``SpecCorpus.build`` (server never holds docs) matches ``from_documents``."""
    docs = _docs()
    spec, *_ = _kmeans_spec(docs)
    a = SpecCorpus.from_documents(spec, docs)
    b = SpecCorpus.build(spec)            # streams from the spec's source
    assert a.token_counts == b.token_counts


def test_fit_routing_from_source_is_deterministic():
    """W7b: a sampled router fit reproduces byte-identical centroids from the same
    public source + params -- the basis for slice c's verification."""
    kw = dict(num_paths=PATHS, vocab_size=VOCAB, seq_len=SEQ, sample=24,
              feature_dim=VOCAB, router_seed=0)
    a = fit_routing_from_source(_source_spec(), **kw)
    b = fit_routing_from_source(_source_spec(), **kw)
    assert a["kind"] == "kmeans"
    assert torch.equal(a["centroids"], b["centroids"])


def test_fit_routing_from_source_falls_back_to_round_robin_when_sample_too_small():
    """Fewer streamed docs than num_paths can't seed k-means; degrade to round-robin
    (mirroring the in-hand builder) instead of crashing at server startup."""
    routing = fit_routing_from_source(_source_spec(), num_paths=PATHS, vocab_size=VOCAB,
                                      seq_len=SEQ, sample=PATHS - 1, feature_dim=VOCAB)
    assert routing == round_robin_routing()


def test_fit_routing_from_source_matches_full_fit_when_sample_covers_corpus():
    """With sample >= corpus size the streamed fit sees every doc in stream order,
    so it equals the in-hand fit (the unsampled path) bit-for-bit."""
    docs = _docs()
    full = fit_routing_from_source(_source_spec(), num_paths=PATHS, vocab_size=VOCAB,
                                   seq_len=SEQ, sample=10_000, feature_dim=VOCAB)
    feat = BagOfTokensFeaturizer(VOCAB, feature_dim=VOCAB)
    router = KMeansRouter(PATHS, seed=0).fit(feat([d[:SEQ] for d in docs]))
    assert torch.equal(full["centroids"], router.centroids)


def test_build_server_corpus_streams_with_router_sample():
    """``build_server_corpus`` with data.router_sample set builds a SpecCorpus by
    streaming -- a valid kmeans recipe + normalized counts, never holding the
    corpus."""
    from opendipaco.launch import LaunchConfig
    from opendipaco.launch.config import dipaco_config
    from opendipaco.launch.roles import build_server_corpus

    cfg = LaunchConfig.from_dict({
        "model": {"vocab_size": VOCAB, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "layers_per_level": [1, 1],
                  "level_sizes": [2, 2], "sequence_length": SEQ,
                  "max_position_embeddings": 64},
        "data": {"source": "synthetic", "num_documents": 32, "ship": "spec",
                 "synthetic_topics": 4, "synthetic_doc_len": 48, "router_sample": 24},
    })
    model = dipaco_config(cfg.model)
    corpus = build_server_corpus(cfg, model)
    assert isinstance(corpus, SpecCorpus)
    assert corpus.spec["routing"]["kind"] == "kmeans"
    assert abs(sum(corpus.shard_weight(p) for p in range(model.num_paths)) - 1.0) < 1e-6


def test_build_server_corpus_unset_router_sample_is_unchanged():
    """Default (no router_sample) routes through the in-hand build -- identical
    token counts to the existing path (byte-identical baseline preserved)."""
    from opendipaco.launch import LaunchConfig
    from opendipaco.launch.config import dipaco_config
    from opendipaco.launch.roles import build_corpus, build_documents, build_server_corpus

    spec_cfg = {
        "model": {"vocab_size": VOCAB, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "layers_per_level": [1, 1],
                  "level_sizes": [2, 2], "sequence_length": SEQ,
                  "max_position_embeddings": 64},
        "data": {"source": "synthetic", "num_documents": 32, "ship": "spec",
                 "synthetic_topics": 4, "synthetic_doc_len": 48},
    }
    cfg = LaunchConfig.from_dict(spec_cfg)
    model = dipaco_config(cfg.model)
    streamed = build_server_corpus(cfg, model)
    in_hand = build_corpus(cfg, model, build_documents(cfg))
    assert streamed.token_counts == in_hand.token_counts


def _sampled_spec(sample=24):
    """A spec whose kmeans router was fit via the streaming sampled path (so it
    carries the `fit` metadata verify_routing needs)."""
    routing = fit_routing_from_source(_source_spec(), num_paths=PATHS, vocab_size=VOCAB,
                                      seq_len=SEQ, sample=sample, feature_dim=VOCAB)
    return make_shard_spec(source=_source_spec(), routing=routing,
                           num_paths=PATHS, seq_len=SEQ)


def test_verify_routing_accepts_untampered_sampled_spec():
    assert verify_routing(_sampled_spec()) is True


def test_verify_routing_detects_tampered_centroids():
    spec = _sampled_spec()
    spec["routing"]["centroids"] = spec["routing"]["centroids"] + 5.0   # poison the router
    assert verify_routing(spec) is False


def test_verify_routing_detects_tampered_featurizer_seed():
    # Keep the legit centroids but flip ONLY the featurizer seed: materialization
    # routes through a different projection (different shards), so verification
    # must reproduce with the shipped seed and refuse the mismatch -- not pass
    # because the centroids are unchanged.
    spec = _sampled_spec()
    spec["routing"]["featurizer"]["seed"] = 7
    assert verify_routing(spec) is False


def test_verify_routing_cant_verify_unknown_featurizer_kind():
    # A future / tampered featurizer kind can't be reproduced by the bag-of-tokens
    # re-fit -> report can't-verify, don't falsely reject by fitting the wrong one.
    spec = _sampled_spec()
    spec["routing"]["featurizer"]["kind"] = "ngram_hash"
    with pytest.raises(ValueError, match="featurizer kind"):
        verify_routing(spec)


def test_verify_routing_trivial_for_round_robin():
    spec = make_shard_spec(source=_source_spec(), routing=round_robin_routing(),
                           num_paths=PATHS, seq_len=SEQ)
    assert verify_routing(spec) is True


def test_verify_routing_raises_without_fit_metadata():
    # An in-hand kmeans spec has no reproducible `fit` block -> can't verify.
    spec, *_ = _kmeans_spec(_docs())
    with pytest.raises(ValueError, match="fit metadata"):
        verify_routing(spec)


def test_verify_routing_cant_verify_when_refit_degrades_to_round_robin():
    # Shipped k-means, but the recorded sample is now too small to re-seed the fit
    # (e.g. the live source shrank). That's inability-to-reproduce, not tampering:
    # raise "cannot verify" (caller proceeds) rather than refuse.
    spec = _sampled_spec()
    spec["routing"]["fit"]["sample"] = PATHS - 1     # re-fit will degrade to round-robin
    with pytest.raises(ValueError, match="cannot verify"):
        verify_routing(spec)


def test_materialize_from_spec_verify_rejects_tampered():
    from opendipaco.schedule.distributed import RoutingVerificationError, _materialize_from_spec
    spec = _sampled_spec()
    spec["routing"]["centroids"] = spec["routing"]["centroids"] + 5.0
    with pytest.raises(RoutingVerificationError):
        _materialize_from_spec({"spec": spec, "path_index": 0}, {"verify": True})


def test_materialize_from_spec_verify_accepts_good_and_memoizes():
    from opendipaco.schedule.distributed import _materialize_from_spec
    spec = _sampled_spec()
    ctx = {"verify": True}
    shard = _materialize_from_spec({"spec": spec, "path_index": 0}, ctx)
    assert shard is not None
    assert spec_fingerprint(spec) in ctx["verified"]      # verified once, memoized


def test_materialize_from_spec_verify_warns_and_proceeds_without_fit_metadata():
    from opendipaco.schedule.distributed import _materialize_from_spec
    spec, *_ = _kmeans_spec(_docs())                       # in-hand spec, no `fit`
    shard = _materialize_from_spec({"spec": spec, "path_index": 0}, {"verify": True})
    assert shard is not None                               # can't-verify -> proceed, not crash


def test_spec_corpus_refuses_to_serve_bytes():
    spec, *_ = _kmeans_spec(_docs())
    corpus = SpecCorpus.from_documents(spec, _docs())
    assert corpus.has_validation is False and corpus.val_shard(0) is None
    with pytest.raises(RuntimeError):
        corpus.shard(0)


def test_spec_fingerprint_tracks_content():
    spec, *_ = _kmeans_spec(_docs())
    assert spec_fingerprint(spec) == spec_fingerprint(spec)
    other = dict(spec, seq_len=SEQ * 2)
    assert spec_fingerprint(other) != spec_fingerprint(spec)
    bent = dict(spec, routing=dict(spec["routing"],
                                   centroids=spec["routing"]["centroids"] + 1.0))
    assert spec_fingerprint(bent) != spec_fingerprint(spec)


# -- the c4-kind source (stub tokenizer; no network) -------------------------------


class _StubTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % (VOCAB - 1) + 1 for c in text]


def test_c4_source_row_budget_and_filters():
    """The c4-kind source counts stream *rows* (like load_c4_documents), applies
    the same tokenize/truncate/filter rules, and accepts HF-style dict rows."""
    rows = [{"text": "a" * 10}, {"text": ""}, {"text": "b" * 100}, {"text": "c" * 7},
            {"text": "never-reached"}]
    src = c4_source(num_documents=4, tokenizer="unused", max_doc_tokens=20,
                    min_doc_tokens=2)
    spec = make_shard_spec(source=src, routing=round_robin_routing(),
                           num_paths=2, seq_len=4)
    docs = list(iter_spec_documents(spec, source=rows, tokenizer=_StubTokenizer()))
    # Row budget 4: the empty row is filtered, "never-reached" is past the budget.
    assert len(docs) == 3
    assert docs[0].numel() == 11                    # 10 chars + eos
    assert docs[1].numel() == 20                    # truncated to the cap (incl. eos)
    assert all(d.dtype == torch.long for d in docs)


def test_materialize_with_cache_dir_survives_source_loss(tmp_path):
    """A cached materialization is served from disk — the second call gets no
    source at all (which would otherwise hit the network for a c4 spec)."""
    rows = ["x" * 30, "y" * 30, "z" * 30, "w" * 30]
    src = c4_source(num_documents=4, tokenizer="unused")
    spec = make_shard_spec(source=src, routing=round_robin_routing(),
                           num_paths=2, seq_len=4)
    first = materialize_shard(spec, 0, source=rows, tokenizer=_StubTokenizer(),
                              cache_dir=str(tmp_path))
    again = materialize_shard(spec, 0, cache_dir=str(tmp_path))  # no source needed
    assert torch.equal(first, again)


# -- end to end: zero corpus bytes on the wire -------------------------------------


def test_coordinator_spec_mode_trains_without_shipping_shards():
    """A coordinator serving a SpecCorpus reaches its target with workers that
    materialize their own shards; no shard bytes ever travel."""
    cfg = _cfg()
    docs = _docs()
    spec, *_ = _kmeans_spec(docs)
    corpus = SpecCorpus.from_documents(spec, docs)
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=10.0), corpus,
                               batch_size=BATCH, host="127.0.0.1", port=0,
                               heartbeat_timeout=10.0)
    tasks = []
    orig = server._next_task
    server._next_task = lambda req: (lambda t: (tasks.append(t) if t.get("type") == "task"
                                                else None, t)[1])(orig(req))
    before = _snap(eng.bank)
    server.start()
    ws = [threading.Thread(target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
                           kwargs=dict(seed=0, reconnect=False, heartbeat_interval=1.0),
                           daemon=True)
          for _ in range(2)]
    for w in ws:
        w.start()
    server.fit(num_generations=GENS, total_generations=GENS, log_every=0)
    server.shutdown()
    for w in ws:
        w.join(timeout=15)

    assert server._T >= server._target
    assert _maxdiff(before, _snap(eng.bank)) > 1e-4      # worker-built shards train
    assert server.metrics.bytes_shard == 0               # the point: no shard bytes
    assert server.metrics.tasks_with_shard == 0
    assert all(t["shard"] is None for t in tasks)
    cold = [t for t in tasks if t["shard_spec"] is not None]
    assert cold                                          # recipes shipped instead
    assert all(t["shard_spec"]["spec"]["version"] == 1 for t in cold)


def test_sharded_spec_mode_trains_without_shipping_shards():
    cfg = _cfg()
    dl = _diloco()
    docs = _docs()
    spec, *_ = _kmeans_spec(docs)
    corpus = SpecCorpus.from_documents(spec, docs)
    ks = assign_shards(cfg.build_topology().module_keys(), 2)
    shards = [[k for k, s in ks.items() if s == i] for i in range(2)]
    pss = [ParameterServer(cfg, sk, dl, host="127.0.0.1", port=0) for sk in shards]
    for ps in pss:
        ps.start()
    befores = [_snap(ps.bank) for ps in pss]
    sched = Scheduler(cfg, corpus, [("127.0.0.1", ps.port) for ps in pss], dl,
                      batch_size=BATCH, host="127.0.0.1", port=0)
    sched.start()
    ws = [threading.Thread(target=run_sharded_worker,
                           args=(cfg, dl, ("127.0.0.1", sched.port)),
                           kwargs=dict(seed=0, heartbeat_interval=1.0), daemon=True)
          for _ in range(2)]
    for w in ws:
        w.start()
    sched.fit(num_generations=GENS, total_generations=GENS)
    sched.shutdown()
    for ps in pss:
        ps.shutdown()
    for w in ws:
        w.join(timeout=10)

    assert sched._T >= sched._target
    assert sched.metrics.bytes_shard == 0
    for ps, before in zip(pss, befores):
        assert _maxdiff(before, _snap(ps.bank)) > 1e-4


# -- launch plumbing ---------------------------------------------------------------


def test_launch_builds_spec_corpus_from_config():
    from opendipaco.launch import LaunchConfig
    from opendipaco.launch.config import dipaco_config
    from opendipaco.launch.roles import build_corpus, build_documents

    cfg = LaunchConfig.from_dict({
        "model": {"vocab_size": VOCAB, "hidden_size": 32, "num_attention_heads": 4,
                  "intermediate_size": 64, "layers_per_level": [1, 1],
                  "level_sizes": [2, 2], "sequence_length": SEQ,
                  "max_position_embeddings": 64},
        "data": {"source": "synthetic", "num_documents": 32, "ship": "spec",
                 "synthetic_topics": 4, "synthetic_doc_len": 48},
    })
    model = dipaco_config(cfg.model)
    docs = build_documents(cfg)
    corpus = build_corpus(cfg, model, docs)
    assert isinstance(corpus, SpecCorpus)
    assert corpus.spec["source"]["kind"] == "synthetic"
    assert abs(sum(corpus.shard_weight(p) for p in range(model.num_paths)) - 1.0) < 1e-6
    # And the spec regenerates the launcher's own documents bit-for-bit.
    regen = list(iter_spec_documents(corpus.spec))
    assert all(torch.equal(a, b) for a, b in zip(docs, regen))

    bad = LaunchConfig.from_dict({"data": {"ship": "tarball"}})
    with pytest.raises(ValueError):
        build_corpus(bad, model, docs)
