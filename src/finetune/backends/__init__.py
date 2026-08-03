"""Backend resolution.

Backends are constructed lazily. ``get_backend`` imports only the requested
module, so listing capabilities or resolving a config never pays the cost of
importing torch or mlx, and works when neither is installed.
"""

from __future__ import annotations

from ..capabilities import Backend
from .base import Plan, TrainingBackend, TrainingResult


def get_backend(backend: Backend) -> TrainingBackend:
    """Instantiate the implementation for ``backend``.

    Importing happens here rather than at module scope so that a missing
    optional dependency surfaces as :class:`~finetune.errors.BackendUnavailableError`
    from ``preflight`` with installation instructions, rather than as an
    ``ImportError`` at package import time.
    """
    if backend is Backend.MLX:
        from .mlx_backend import MlxBackend

        return MlxBackend()
    if backend is Backend.TORCH:
        from .torch_backend import TorchBackend

        return TorchBackend()
    raise ValueError(f"unknown backend: {backend}")  # pragma: no cover - enum is exhaustive


__all__ = ["Plan", "TrainingBackend", "TrainingResult", "get_backend"]
