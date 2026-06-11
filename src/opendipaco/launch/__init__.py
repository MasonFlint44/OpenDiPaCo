"""Config-driven launcher for an opendipaco cluster.

``LaunchConfig`` + ``load_config`` describe a run; the ``run_*`` role functions stand
up each role (coordinator / scheduler / parameter server / worker / ingest) from it;
``opendipaco.launch.cli:main`` is the ``opendipaco`` console script.
"""

from .config import (
    LaunchConfig,
    backbone_config,
    dipaco_config,
    diloco_config,
    load_config,
)
from .roles import (
    build_corpus,
    build_documents,
    run_coordinator,
    run_ingest,
    run_local,
    run_parameter_server,
    run_scheduler,
    run_worker_role,
)

__all__ = [
    "LaunchConfig",
    "load_config",
    "backbone_config",
    "dipaco_config",
    "diloco_config",
    "build_documents",
    "build_corpus",
    "run_coordinator",
    "run_scheduler",
    "run_parameter_server",
    "run_worker_role",
    "run_ingest",
    "run_local",
]
