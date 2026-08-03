"""Apple Silicon backend, driving the ``mlx_lm`` trainer.

Named ``mlx_backend`` rather than ``mlx`` deliberately. A module named for the
package it imports is how this repository previously ended up with a ``peft/``
directory that shadowed the real ``peft`` and made its own headline code path
unimportable.

Two things this fixes relative to the scripts it replaces:

``--fine-tune-type`` is actually passed. The old QLoRA script never set it, so it
silently ran plain LoRA on a full-precision base and differed from the LoRA
script by filename only.

QLoRA quantises. In MLX, quantisation is a ``mlx_lm convert`` step rather than a
trainer flag, so this backend runs convert first and trains against its output.

Argv and trainer-config construction live in :mod:`._mlx_args`, which imports
nothing from mlx, so the command this backend will run is testable on any
platform.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import ClassVar

import yaml

from ..capabilities import Backend, Method, is_installed, resolve
from ..data.layout import validate_split_dir
from ..errors import BackendUnavailableError, TrainingFailedError
from ..run import RunConfig
from . import _mlx_args as mlx_args
from ._process import ProcessOutcome, failure_detail, run_streaming
from .base import Plan, TrainingResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MlxBackend:
    """Fine-tunes via ``python -m mlx_lm lora``."""

    backend: ClassVar[Backend] = Backend.MLX

    def preflight(self, config: RunConfig) -> None:
        """Validate the capability pair, dependency presence and dataset."""
        resolve(config.method, self.backend)
        if not is_installed(self.backend):
            raise BackendUnavailableError(
                backend=str(self.backend), extra="apple", missing="mlx_lm"
            )
        validate_split_dir(config.data_dir)

    def plan(self, config: RunConfig) -> Plan:
        """Describe the run without touching the filesystem."""
        return Plan(
            backend=self.backend,
            method=config.method,
            summary=(
                f"mlx_lm {mlx_args.FINE_TUNE_TYPE[config.method]} on {config.model} "
                f"-> {mlx_args.adapter_dir(config)}"
            ),
            stages=mlx_args.describe_stages(config),
            argv=mlx_args.train_argv(config.output_dir / mlx_args.CONFIG_FILENAME),
        )

    def train(self, config: RunConfig) -> TrainingResult:
        config.output_dir.mkdir(parents=True, exist_ok=True)

        if config.method is Method.QLORA:
            self._run_stage("quantise base", mlx_args.quantize_argv(config))

        config_path = config.output_dir / mlx_args.CONFIG_FILENAME
        config_path.write_text(
            yaml.safe_dump(mlx_args.trainer_config(config), sort_keys=False), encoding="utf-8"
        )
        logger.info("wrote resolved mlx_lm config to %s", config_path)

        label = f"mlx_lm {mlx_args.FINE_TUNE_TYPE[config.method]}"
        outcome = self._run_stage(label, mlx_args.train_argv(config_path))
        self._warn_if_diverged(config, outcome.last_train_loss)

        metrics: dict[str, float] = {}
        if outcome.last_val_loss is not None:
            metrics["val_loss"] = outcome.last_val_loss
        return TrainingResult(
            backend=self.backend,
            method=config.method,
            output_dir=mlx_args.adapter_dir(config),
            succeeded=True,
            steps=outcome.last_step,
            final_loss=outcome.last_train_loss,
            metrics=metrics,
            detail=f"resolved config: {config_path}",
        )

    @staticmethod
    def _run_stage(label: str, argv: tuple[str, ...]) -> ProcessOutcome:
        """Run one subprocess stage, raising with output context on failure."""
        outcome = run_streaming(argv, label=label)
        if not outcome.succeeded:
            raise TrainingFailedError(failure_detail(label, outcome))
        return outcome

    @staticmethod
    def _warn_if_diverged(config: RunConfig, loss: float | None) -> None:
        """Warn on a non-finite loss. mlx_lm exits 0 even when the loss is NaN.

        Two causes dominate, and they have opposite fixes, so name both. A learning
        rate too high for the method diverges; prompt masking combined with
        truncation short enough to cut the completion leaves nothing to train on,
        which shows up as NaN with zero trained tokens.
        """
        if loss is None or math.isfinite(loss):
            return
        logger.warning(
            "final training loss is %s. Common causes: learning rate %g too high for "
            "method '%s' (full fine-tuning wants ~100x less than LoRA), or "
            "max_seq_length %d truncating away the completion while mask_prompt is "
            "on, leaving no tokens in the loss.",
            loss,
            config.learning_rate,
            config.method,
            config.training.max_seq_length,
        )


__all__ = ["MlxBackend"]
