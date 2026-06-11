"""Three small-but-real polish items (the §7/§6 bundle):

1. ``DiPaCoConfig.eval_seq_len`` surfaces the train/eval sequence-length split as a
   config knob (the paper evaluates at a longer context than it trains on).
2. Auth-key **rotation / per-worker identity**: a server can accept several secrets,
   so old+new keys both work during a rotation and each worker can hold its own key.
3. Warm-start is verified **end-to-end**: a path warm-started from a real Llama
   computes the *same logits* as that Llama -- not just weight-equal, behavior-equal.
"""

import threading

import torch

from opendipaco import (
    AsyncScheduler,
    BackboneConfig,
    DiLoCoConfig,
    DiPaCoConfig,
    DiPaCoEngine,
    LocalBackend,
    build_module_bank,
    build_path_model,
    warm_start_modules,
)
from opendipaco.data import ShardedCorpus
from opendipaco.schedule import CoordinatorServer, run_worker
from opendipaco.schedule.wire import acceptable_keys, coerce_keys

BATCH = 8


# -- 1. eval-sequence-length knob --------------------------------------------


def test_eval_seq_len_defaults_to_sequence_length():
    cfg = DiPaCoConfig(level_sizes=[2, 2], sequence_length=256)
    assert cfg.eval_sequence_length is None
    assert cfg.eval_seq_len == 256  # falls back to the training length


def test_eval_seq_len_is_independent_when_set():
    cfg = DiPaCoConfig(level_sizes=[2, 2], sequence_length=256, eval_sequence_length=1024)
    assert cfg.eval_seq_len == 1024 and cfg.sequence_length == 256


def test_eval_seq_len_rejects_non_positive():
    for bad in (0, -5):
        try:
            DiPaCoConfig(level_sizes=[2, 2], eval_sequence_length=bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


# -- 2. auth-key rotation / per-worker identity ------------------------------


def test_coerce_and_acceptable_keys():
    assert coerce_keys(None) is None
    assert coerce_keys("a") == [b"a"]
    assert coerce_keys(["a", b"b"]) == [b"a", b"b"]
    # auth_key is unioned with accept_keys, de-duplicated, order preserved.
    assert acceptable_keys("a", ["b", "a"]) == [b"a", b"b"]
    assert acceptable_keys(None, None) is None


def _cfg():
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1], max_position_embeddings=64)
    return DiPaCoConfig(backbone=bb, level_sizes=[2, 2], sequence_length=16)


def _docs():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(t * 12, t * 12 + 12, (48,), generator=g)
            for t in range(4) for _ in range(8)]


def _corpus(cfg):
    assign = torch.tensor([i % cfg.num_paths for i in range(32)])
    return ShardedCorpus.from_assignments(_docs(), assign, cfg.num_paths, cfg.sequence_length)


def _diloco():
    return DiLoCoConfig(inner_steps=4, inner_lr=1e-3)


def _engine(cfg, seed=0):
    return DiPaCoEngine(cfg, _diloco(), LocalBackend(cfg.build_topology()),
                        seed=seed, materialize="serial")


def test_server_accepts_any_of_several_keys():
    """Two workers with *different* secrets both authenticate (per-worker identity);
    rotation works the same way (list old+new). A non-listed key is rejected."""
    cfg = _cfg()
    eng = _engine(cfg)
    # The server holds no single shared secret -- only a per-worker accept-list.
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=5.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0,
                               accept_keys=["worker-old", "worker-new"])
    server.start()
    ws = [threading.Thread(target=run_worker, args=(cfg, _diloco(), "127.0.0.1", server.port),
                           kwargs=dict(seed=0, reconnect=False, auth_key=k), daemon=True)
          for k in ("worker-old", "worker-new")]
    for w in ws:
        w.start()
    server.fit(num_generations=2, total_generations=2, log_every=0)
    server.shutdown()
    for w in ws:
        w.join(timeout=10)
    assert server._T >= server._target
    assert server.metrics.accepted_updates >= cfg.num_paths * 2


def test_server_rejects_unlisted_key():
    cfg = _cfg()
    eng = _engine(cfg)
    server = CoordinatorServer(AsyncScheduler(eng, lease_timeout=2.0), _corpus(cfg),
                               batch_size=BATCH, host="127.0.0.1", port=0,
                               accept_keys=["good-1", "good-2"])
    server.start()
    err = {}

    def run():
        try:
            run_worker(cfg, DiLoCoConfig(inner_steps=4, inner_lr=1e-3), "127.0.0.1", server.port,
                       seed=0, reconnect=False, auth_key="not-listed")
        except Exception as e:  # noqa: BLE001
            err["e"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=10)
    server.shutdown()
    assert isinstance(err.get("e"), PermissionError)
    assert server.metrics.accepted_updates == 0


# -- 3. warm-start end-to-end (behavior, not just weights) -------------------


def _pretrained(bb):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=bb.vocab_size, hidden_size=bb.hidden_size,
        intermediate_size=bb.intermediate_size, num_hidden_layers=bb.total_layers,
        num_attention_heads=bb.num_attention_heads, num_key_value_heads=bb.kv_heads(),
        max_position_embeddings=bb.max_position_embeddings,
        rms_norm_eps=bb.rms_norm_eps, tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg).eval()


def test_warm_started_path_matches_pretrained_logits():
    """A single warm-started DiPaCo path *is* the pretrained model: same logits.

    level_sizes=[1, 1] -> one path == a plain dense transformer, so its composed
    forward must reproduce the source Llama's logits to float tolerance. This
    exercises the whole warm-start -> compose -> forward pipeline end to end.
    """
    bb = BackboneConfig(vocab_size=48, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, layers_per_level=[1, 1], max_position_embeddings=64)
    cfg = DiPaCoConfig(backbone=bb, level_sizes=[1, 1])
    llama = _pretrained(bb)

    bank = build_module_bank(cfg)
    warm_start_modules(bank, cfg.build_topology(), llama)
    path = cfg.build_topology().paths()[0]
    pm = build_path_model(cfg, path, bank, deepcopy=False).eval()

    ids = torch.randint(0, bb.vocab_size, (2, 16))
    with torch.no_grad():
        logits, _ = pm(ids)
        ref = llama(ids).logits
    assert logits.shape == ref.shape
    assert torch.allclose(logits, ref, atol=1e-4), \
        f"max abs diff {float((logits - ref).abs().max()):.2e}"
