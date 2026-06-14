"""Composing modules into a single executable path.

A :class:`PathModel` is what one DiPaCo worker trains: a concrete network for a
single path. It owns its own copy of the modules along that path (embed, the
ordered body blocks/experts, head) so it can take independent inner-optimizer
steps before the shared modules are averaged back together.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from .config import DiPaCoConfig
from .modules import BlockModule, EmbeddingModule, HeadModule, to_llama_config
from .topology import Path, PathTopology, embed_key, head_key


def _build_module(topo: PathTopology, bb, key: str) -> nn.Module:
    role = topo.role_of_key(key)
    if role == "embed":
        return EmbeddingModule(bb)
    if role == "head":
        return HeadModule(bb)
    return BlockModule(bb, topo.layers_of_key(key), topo.offset_of_key(key))


def build_module_bank(config: DiPaCoConfig, *, seed: int | None = None) -> dict[str, nn.Module]:
    """Create one authoritative instance of every module in the topology.

    With ``identical_expert_init`` (the default, matching the paper's ``θ̄``), all
    instances of a module *position* -- the experts of a routing level, or the
    per-path copies of a private module -- start identical and diverge only via
    training. Otherwise each gets an independent random init. The returned dict is
    the canonical set of weights the outer optimizer maintains.

    ``seed`` makes the build a pure function of (config, seed): two processes
    that pass the same seed get bit-identical banks. Replicated module owners
    rely on this -- version ``(0, 0)`` must mean the same bytes on every owner
    (``schedule/sharded.py``). ``None`` keeps today's ambient-RNG behavior.
    """
    if seed is not None:
        with torch.random.fork_rng(devices=[]):  # don't disturb the caller's RNG
            torch.manual_seed(seed)
            return build_module_bank(config)
    bb = config.backbone
    topo = config.build_topology()
    bank: dict[str, nn.Module] = {}
    templates: dict[int, nn.Module] = {}  # segment index -> first-built module
    for key in topo.module_keys():
        seg = topo._segment_index_of_key(key)
        if seg not in templates:
            templates[seg] = _build_module(topo, bb, key)
            bank[key] = templates[seg]
        elif config.identical_expert_init:
            bank[key] = copy.deepcopy(templates[seg])
        else:
            bank[key] = _build_module(topo, bb, key)
    if config.tie_word_embeddings and embed_key() in bank and head_key() in bank:
        bank[head_key()].lm_head.weight = bank[embed_key()].embed_tokens.weight
    return bank


class PathModel(nn.Module):
    """A single composed path: embed -> body blocks/experts -> head."""

    def __init__(self, config: DiPaCoConfig, path: Path, modules: dict[str, nn.Module]):
        super().__init__()
        self.config = config
        self.path = path
        self.topology = config.build_topology()
        keys = self.topology.path_module_keys(path)

        self.embed: nn.Module | None = None
        self.head: nn.Module | None = None
        self.body = nn.ModuleList()
        self._body_keys: list[str] = []
        self._embed_key: str | None = None
        self._head_key: str | None = None
        for key in keys:
            role = self.topology.role_of_key(key)
            mod = modules[key]
            if role == "embed":
                self.embed, self._embed_key = mod, key
            elif role == "head":
                self.head, self._head_key = mod, key
            else:
                self.body.append(mod)
                self._body_keys.append(key)
        if self.embed is None or self.head is None:
            raise ValueError("a path must contain an embedding and a head segment")

        self.rotary = LlamaRotaryEmbedding(to_llama_config(config.backbone))
        # Activation checkpointing (W3b): when True, body blocks recompute their
        # activations in backward instead of storing them. Set by the trainer
        # (off for inference/encode). Bit-exact -- only changes what is stored.
        self.checkpoint = False
        # Chunked cross-entropy (W3c): >1 computes logits+loss in N token-chunks so
        # the full [tokens, vocab] logits never materialize. Set by the trainer.
        self.loss_chunks = 1

    def modules_by_key(self) -> dict[str, nn.Module]:
        """Map topology key -> the live submodule instance in this path."""
        out: dict[str, nn.Module] = {self._embed_key: self.embed, self._head_key: self.head}
        out.update(zip(self._body_keys, self.body))
        return out

    def _backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        """Run embed -> body blocks, returning the final hidden states (pre-head)."""
        _, seqlen = input_ids.shape
        hidden = self.embed(input_ids)
        position_ids = torch.arange(seqlen, device=input_ids.device).unsqueeze(0)
        cos_sin = self.rotary(hidden, position_ids)
        causal = create_causal_mask(
            config=to_llama_config(self.config.backbone),
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        # Checkpoint only when it can help: training with grad on (under no_grad /
        # eval it would be a wasteful no-op). use_reentrant=False preserves the
        # autocast context on recompute, so it composes with inner_autocast.
        ckpt = self.checkpoint and self.training and torch.is_grad_enabled()
        for block in self.body:
            if ckpt:
                hidden = checkpoint(
                    block, hidden, position_embeddings=cos_sin,
                    attention_mask=causal, position_ids=position_ids,
                    use_reentrant=False,
                )
            else:
                hidden = block(
                    hidden,
                    position_embeddings=cos_sin,
                    attention_mask=causal,
                    position_ids=position_ids,
                )
        return hidden

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Contextual representation of the input (final hidden states, normalized).

        Used for routing -- the paper's ``z`` features come from the model itself.
        """
        hidden = self._backbone(input_ids, attention_mask)
        norm = getattr(self.head, "norm", None)
        return norm(hidden) if norm is not None else hidden

    def _chunked_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy without materializing the full [tokens, vocab] logits:
        norm the hidden states (per-token, so it's safe to do whole), then apply
        the vocab projection + CE in token-chunks and accumulate. Mathematically
        the same loss as the dense path (fp summation order differs ~1e-7)."""
        norm = getattr(self.head, "norm", None)
        normed = norm(hidden) if norm is not None else hidden
        h = normed[:, :-1, :].reshape(-1, normed.size(-1))   # shifted, flattened tokens
        y = labels[:, 1:].reshape(-1)
        # Accumulate the loss sum in fp32 even under bf16 autocast (the dense path
        # reduces in fp32 too) so chunking doesn't lose precision per chunk.
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        count = 0
        for hc, yc in zip(h.chunk(self.loss_chunks), y.chunk(self.loss_chunks)):
            total = total + F.cross_entropy(self.head.lm_head(hc), yc,
                                            ignore_index=-100, reduction="sum").float()
            count += int((yc != -100).sum())
        return total / max(count, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        hidden = self._backbone(input_ids, attention_mask)
        # Chunked CE (W3c): when on and we only need the loss (training), skip the
        # full logits tensor entirely. Callers that want logits pass labels=None.
        if labels is not None and self.loss_chunks > 1:
            return None, self._chunked_loss(hidden, labels)

        logits = self.head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


def build_path_model(
    config: DiPaCoConfig, path: Path, bank: dict[str, nn.Module], deepcopy: bool = True
) -> PathModel:
    """Instantiate a :class:`PathModel` from the module bank.

    ``deepcopy=True`` (default) gives the worker independent weights so its inner
    steps don't mutate the shared bank; pass ``False`` to alias the bank
    directly (useful for a single-worker / inference setup).
    """
    topo = config.build_topology()
    keys = topo.path_module_keys(path)
    missing = [k for k in keys if k not in bank]
    if missing:
        raise KeyError(
            f"bank is missing modules {missing} needed for path {path}. Building a "
            "path needs the full module bank; in distributed training each rank holds "
            "only its own path's modules, so gather the complete bank first with "
            "opendipaco.gather_full_bank(backend, engine.bank, config)."
        )
    selected = {k: bank[k] for k in keys}
    if deepcopy:
        # Deep-copy the whole selection in ONE call so cross-module shared tensors
        # (tied embed/head weights, W3c/D6) stay shared in the copy -- per-module
        # deepcopy would sever the tie, doubling memory and training the two copies
        # apart. Identical to per-module copy when nothing is tied.
        selected = copy.deepcopy(selected)
    return PathModel(config, path, selected)
