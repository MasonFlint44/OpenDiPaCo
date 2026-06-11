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


def build_module_bank(config: DiPaCoConfig) -> dict[str, nn.Module]:
    """Create one authoritative instance of every module in the topology.

    With ``identical_expert_init`` (the default, matching the paper's ``θ̄``), all
    instances of a module *position* -- the experts of a routing level, or the
    per-path copies of a private module -- start identical and diverge only via
    training. Otherwise each gets an independent random init. The returned dict is
    the canonical set of weights the outer optimizer maintains.
    """
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
        for block in self.body:
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        hidden = self._backbone(input_ids, attention_mask)
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
        selected = {k: copy.deepcopy(v) for k, v in selected.items()}
    return PathModel(config, path, selected)
