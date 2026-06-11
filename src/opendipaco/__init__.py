"""opendipaco -- an open implementation of DiPaCo (Distributed Path Composition).

Quick start (single-process simulation)::

    from opendipaco import (
        BackboneConfig, DiPaCoConfig, DiLoCoConfig,
        PathTopology, LocalBackend, DiPaCoEngine,
    )
    from opendipaco.routing import BagOfTokensFeaturizer, KMeansRouter
    from opendipaco.data import ShardedCorpus

    config = DiPaCoConfig(level_sizes=[2, 2])
    topo = PathTopology(tuple(config.level_sizes))
    featurizer = BagOfTokensFeaturizer(config.backbone.vocab_size)
    router = KMeansRouter(config.num_paths).fit(featurizer(prefixes))
    corpus = ShardedCorpus.from_documents(docs, router, featurizer,
                                          config.num_paths, config.sequence_length)
    engine = DiPaCoEngine(config, DiLoCoConfig(), LocalBackend(topo))
    engine.fit(corpus, num_rounds=100, batch_size=16)
"""

from .backend import LocalBackend, SyncBackend
from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .config import BackboneConfig, DiLoCoConfig, DiPaCoConfig, SegmentSpec
from .distributed import gather_full_bank
from .em import (
    assign_paths_by_loss,
    fit_discriminative_router,
    path_losses,
    reshard_by_loss,
)
from .init import warm_start_modules
from .model import PathModel, build_module_bank, build_path_model
from .schedule import AsyncScheduler
from .topology import PathTopology, Segment, Sharing
from .train import DiPaCoEngine, RoundMetrics

__all__ = [
    "BackboneConfig",
    "DiPaCoConfig",
    "DiLoCoConfig",
    "SegmentSpec",
    "Segment",
    "Sharing",
    "PathTopology",
    "PathModel",
    "build_module_bank",
    "build_path_model",
    "SyncBackend",
    "LocalBackend",
    "DiPaCoEngine",
    "RoundMetrics",
    "warm_start_modules",
    "path_losses",
    "assign_paths_by_loss",
    "reshard_by_loss",
    "fit_discriminative_router",
    "gather_full_bank",
    "save_checkpoint",
    "load_checkpoint",
    "latest_checkpoint",
    "AsyncScheduler",
]

try:
    from .backend import TorchDistBackend  # noqa: F401

    __all__.append("TorchDistBackend")
except Exception:  # pragma: no cover
    pass

__version__ = "0.1.0"
