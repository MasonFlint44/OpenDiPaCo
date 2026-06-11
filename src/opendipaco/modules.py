"""The transformer pieces a path is built from.

Each *module* is a plain ``nn.Module`` whose ``state_dict`` is what DiLoCo syncs:

* ``EmbeddingModule`` -- token embedding (a shared module).
* ``LevelModule``     -- a stack of HF Llama decoder layers (one expert at a level).
* ``HeadModule``      -- final RMSNorm + LM head (a shared module).

The rotary position embedding is parameter-free and lives on the composed
``PathModel`` (see model.py), so it is never synced.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

from .config import BackboneConfig


def to_llama_config(bb: BackboneConfig) -> LlamaConfig:
    """Build the HF ``LlamaConfig`` used to instantiate decoder layers/norms."""
    cfg = LlamaConfig(
        vocab_size=bb.vocab_size,
        hidden_size=bb.hidden_size,
        intermediate_size=bb.intermediate_size,
        num_hidden_layers=bb.total_layers,
        num_attention_heads=bb.num_attention_heads,
        num_key_value_heads=bb.kv_heads(),
        max_position_embeddings=bb.max_position_embeddings,
        rope_theta=bb.rope_theta,
        rms_norm_eps=bb.rms_norm_eps,
    )
    # Required so the decoder layers dispatch attention correctly when used as
    # standalone modules (see transformers AttentionInterface).
    cfg._attn_implementation = bb.attn_implementation
    return cfg


class EmbeddingModule(nn.Module):
    def __init__(self, bb: BackboneConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(bb.vocab_size, bb.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class BlockModule(nn.Module):
    """A body block group: a contiguous stack of Llama decoder layers.

    Used for both routing experts and (shared/private) trunk blocks.
    """

    def __init__(self, bb: BackboneConfig, num_layers: int, layer_offset: int):
        super().__init__()
        cfg = to_llama_config(bb)
        # layer_idx is used by HF only for KV-cache addressing; we give each
        # layer a globally-unique index (within a path) so caching works.
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(cfg, layer_offset + i) for i in range(num_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
        return hidden_states


# Backwards-compatible alias (a routing expert is just a body block group).
LevelModule = BlockModule


class HeadModule(nn.Module):
    def __init__(self, bb: BackboneConfig):
        super().__init__()
        self.norm = LlamaRMSNorm(bb.hidden_size, eps=bb.rms_norm_eps)
        self.lm_head = nn.Linear(bb.hidden_size, bb.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))
