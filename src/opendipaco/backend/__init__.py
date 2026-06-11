"""Synchronization backends.

The DiPaCo training engine is backend-agnostic: it produces, per shared module,
a *locally* weighted-summed pseudo-gradient and a weight, then asks the backend
to reduce these across every path that shares the module (possibly on other
processes/hosts). Two backends ship today:

* :class:`LocalBackend`   -- single process simulating all paths (no comm).
* :class:`TorchDistBackend` -- one process per path, averaging shared modules via
  ``torch.distributed`` process subgroups.

A ``hivemind`` / DHT backend can be added behind the same interface.
"""

from .base import ReducedDelta, SyncBackend
from .local import LocalBackend

__all__ = ["SyncBackend", "ReducedDelta", "LocalBackend"]

try:  # torch.distributed is optional at import time
    from .torch_dist import TorchDistBackend  # noqa: F401

    __all__.append("TorchDistBackend")
except Exception:  # pragma: no cover
    pass
