"""Warm-starting from a pretrained dense model.

The DiPaCo paper initialises every path from a pretrained dense transformer
(``theta-bar``) rather than from scratch. Each path is a dense Llama-style model,
so a pretrained model of the *same shape* maps cleanly onto the module bank:

* its token embedding   -> every embedding module,
* its decoder layers    -> the body block at the matching layer offset (all
  experts of a routing level copy the *same* pretrained layers, so they start
  identical and then diverge),
* its final norm + head -> every head module.

The pretrained model must match ``BackboneConfig`` (hidden size, heads, vocab,
intermediate size) and have at least ``topology.total_layers`` decoder layers.
"""

from __future__ import annotations

import torch
from torch import nn

from .topology import PathTopology


def load_pretrained(source: str | nn.Module) -> nn.Module:
    """Resolve ``source`` to a model (loads via ``AutoModelForCausalLM`` if a name)."""
    if isinstance(source, str):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(source)
    return source


def _llama_parts(model: nn.Module):
    """Extract (embed_tokens, layers, norm, lm_head) from a Llama-style CausalLM."""
    base = getattr(model, "model", model)
    try:
        embed = base.embed_tokens
        layers = base.layers
        norm = base.norm
    except AttributeError as e:  # pragma: no cover - clear error for odd models
        raise ValueError(
            "warm-start expects a Llama-style model exposing model.embed_tokens, "
            "model.layers and model.norm"
        ) from e
    lm_head = getattr(model, "lm_head", None)
    return embed, layers, norm, lm_head


@torch.no_grad()
def warm_start_modules(
    bank: dict[str, nn.Module], topology: PathTopology, source: str | nn.Module
) -> None:
    """Copy a pretrained dense model's weights into every module in ``bank`` in place."""
    model = load_pretrained(source).eval()
    embed, layers, norm, lm_head = _llama_parts(model)
    if len(layers) < topology.total_layers:
        raise ValueError(
            f"pretrained model has {len(layers)} layers but the path needs "
            f"{topology.total_layers}"
        )
    for key, mod in bank.items():
        role = topology.role_of_key(key)
        if role == "embed":
            mod.embed_tokens.load_state_dict(embed.state_dict())
        elif role == "head":
            mod.norm.load_state_dict(norm.state_dict())
            if lm_head is not None:
                mod.lm_head.load_state_dict(lm_head.state_dict())
        else:  # body block group
            offset = topology.offset_of_key(key)
            for i, layer in enumerate(mod.layers):
                layer.load_state_dict(layers[offset + i].state_dict())
