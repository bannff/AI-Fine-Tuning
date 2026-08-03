"""Fine-tune causal language models with LoRA, DoRA, QLoRA or full updates.

Two backends sit behind one config surface: ``mlx`` for Apple Silicon and
``torch`` for everywhere else. Which combinations exist is declared in
:mod:`finetune.capabilities` and enforced, so the documented support table and
the runtime behaviour cannot drift apart.

Importing this package is cheap. It pulls in PyYAML and nothing else; training
dependencies live behind the ``torch`` and ``apple`` extras and are imported
only when a backend actually runs.
"""

from __future__ import annotations

from .capabilities import CAPABILITIES, Backend, Method, installed_backends, supported_methods
from .config import AdapterConfig, QuantizationConfig, TrainingConfig
from .config_io import dump_run_config, load_run_config
from .errors import (
    BackendUnavailableError,
    ConfigError,
    DatasetSchemaError,
    FinetuneError,
    TrainingFailedError,
    UnsupportedCombinationError,
)
from .run import RunConfig

__version__ = "0.2.0"

__all__ = [
    "CAPABILITIES",
    "AdapterConfig",
    "Backend",
    "BackendUnavailableError",
    "ConfigError",
    "DatasetSchemaError",
    "FinetuneError",
    "Method",
    "QuantizationConfig",
    "RunConfig",
    "TrainingConfig",
    "TrainingFailedError",
    "UnsupportedCombinationError",
    "__version__",
    "dump_run_config",
    "installed_backends",
    "load_run_config",
    "supported_methods",
]
