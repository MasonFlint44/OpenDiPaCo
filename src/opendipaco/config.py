"""Configuration dataclasses for DiPaCo.

These describe (a) the per-path transformer backbone, (b) how the backbone is
carved into shared modules across levels, and (c) the DiLoCo inner/outer
optimization used to keep shared modules in sync.

Reference: Douillard et al., "DiPaCo: Distributed Path Composition",
https://arxiv.org/abs/2403.10616
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from .topology import PathTopology, Segment, Sharing


@dataclass
class BackboneConfig:
    """Shape of the (Llama-style) transformer that a single path realises.

    A *path* is the full network: shared token embedding -> one expert module
    per level -> shared head. ``layers_per_level`` says how many transformer
    decoder layers live inside each level's expert module, so the depth of a
    path is ``sum(layers_per_level)``.
    """

    vocab_size: int = 32000
    hidden_size: int = 896
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None  # defaults to num_attention_heads
    intermediate_size: int = 2432
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attn_implementation: str = "sdpa"

    # One entry per level: how many decoder layers each level's expert holds.
    layers_per_level: list[int] = field(default_factory=lambda: [6, 6])

    def kv_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def num_levels(self) -> int:
        return len(self.layers_per_level)

    @property
    def total_layers(self) -> int:
        return sum(self.layers_per_level)


@dataclass
class SegmentSpec:
    """A body segment for the advanced architecture API (see ``DiPaCoConfig.body``).

    * ``num_experts > 1`` -> a routing level (a path picks one; averaged within
      its sharing group). ``sharing`` is ignored (routing levels are shared).
    * ``num_experts == 1`` -> a single block group, either ``"shared"`` (averaged
      across all paths) or ``"private"`` (a per-path copy, never communicated).
    """

    layers: int
    num_experts: int = 1
    sharing: str = "shared"


@dataclass
class DiPaCoConfig:
    """Full DiPaCo model description.

    Simple API: ``level_sizes = [K_1, ..., K_L]`` gives one shared routing level
    per entry (paired with ``backbone.layers_per_level``); the number of paths is
    ``prod(level_sizes)`` (e.g. 16x16 -> 256 paths). ``embedding`` and ``head``
    select whether those components are ``"shared"`` (averaged across all paths)
    or ``"private"`` (per-path, never communicated -- as in the paper's 16x16).

    Advanced API: set ``body`` to an explicit ``list[SegmentSpec]`` to interleave
    private/shared trunk blocks with routing levels; this overrides ``level_sizes``
    and ``backbone.layers_per_level``.
    """

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    level_sizes: list[int] = field(default_factory=lambda: [4, 4])
    embedding: str = "shared"   # "shared" | "private"
    head: str = "shared"        # "shared" | "private"
    tie_word_embeddings: bool = False
    body: list[SegmentSpec] | None = None
    # Paper: every path is initialised from the same model (theta-bar), so all
    # experts/private copies of a module start identical and diverge only through
    # training on different shards. True reproduces that even from scratch; set
    # False to give each expert/private copy an independent random init.
    identical_expert_init: bool = True

    sequence_length: int = 1024
    # Eval can use a longer context than training (the paper evaluates at a longer
    # sequence length than it trains on). ``None`` -> reuse ``sequence_length``; read
    # it via the ``eval_seq_len`` property so callers don't repeat the fallback.
    eval_sequence_length: int | None = None

    def __post_init__(self) -> None:
        if self.body is None and len(self.level_sizes) != self.backbone.num_levels:
            raise ValueError(
                f"level_sizes has {len(self.level_sizes)} levels but backbone "
                f"layers_per_level has {self.backbone.num_levels}"
            )
        if self.body is None and any(k < 1 for k in self.level_sizes):
            raise ValueError("each level must have at least one expert")
        if self.eval_sequence_length is not None and self.eval_sequence_length < 1:
            raise ValueError("eval_sequence_length must be >= 1")

    @property
    def eval_seq_len(self) -> int:
        """Sequence length used for evaluation (defaults to ``sequence_length``)."""
        return self.eval_sequence_length or self.sequence_length

    # -- architecture resolution --------------------------------------------
    def segments(self) -> list[Segment]:
        """Resolve the full ordered segment list (embed -> body -> head)."""
        segs = [Segment("embed", 0, 1, Sharing(self.embedding))]
        if self.body is None:
            for size, layers in zip(self.level_sizes, self.backbone.layers_per_level):
                segs.append(Segment("body", layers, size, Sharing.SHARED))
        else:
            for spec in self.body:
                segs.append(
                    Segment("body", spec.layers, spec.num_experts, Sharing(spec.sharing))
                )
        segs.append(Segment("head", 0, 1, Sharing(self.head)))
        return segs

    def build_topology(self) -> PathTopology:
        return PathTopology(segments=self.segments())

    @property
    def num_paths(self) -> int:
        if self.body is None:
            return prod(self.level_sizes)
        return prod(s.num_experts for s in self.body if s.num_experts > 1) or 1


@dataclass
class DiLoCoConfig:
    """Inner/outer optimization (DiLoCo) hyper-parameters.

    The inner optimizer (AdamW) takes ``inner_steps`` local steps on each path's
    data shard; the outer optimizer (SGD with Nesterov momentum) then applies
    the averaged pseudo-gradient to the shared module weights.
    """

    inner_steps: int = 150           # H / tau: local steps between syncs (paper's main run)
    inner_lr: float = 4e-4           # peak inner learning rate
    inner_betas: tuple[float, float] = (0.9, 0.95)
    inner_weight_decay: float = 0.1
    inner_grad_clip: float | None = 1.0

    # Inner LR schedule over the *whole* run (paper: cosine, peak 4e-4). Needs the
    # total number of rounds, which ``DiPaCoEngine.fit`` supplies automatically;
    # when rounds are driven manually, set ``engine.total_rounds`` (else the LR
    # stays constant).
    inner_lr_schedule: str = "cosine"     # "constant" | "cosine"
    inner_warmup_steps: int = 0           # linear warmup, in inner steps
    inner_min_lr_ratio: float = 0.1       # cosine floor = inner_lr * this

    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    outer_nesterov: bool = True

    # --- Outer-delta aggregation (DiPaCo defaults reproduce the paper) -------
    # Weight each path's pseudo-gradient by its shard size (the paper's alpha
    # re-weighting, alpha_i = |D_i| / |D_total|; Eq. 2-3).
    shard_size_weighting: bool = True
    # If True, divide the summed delta by the total contributing weight (a
    # scale-stable weighted *mean*). If False (paper), use the raw weighted sum:
    # with shard_size_weighting the alpha terms already carry the normalization
    # (Eq. 2-3), and the unweighted case falls back to the 1/P mean of Algo. 1.
    normalize_outer_delta: bool = False
    # Rescale a module's outer gradient by sqrt(num paths sharing it), to account
    # for its larger effective batch (paper sec. 3).
    rescale_by_sqrt_sharing: bool = True
