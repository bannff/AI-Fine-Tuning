"""The sections of a run configuration.

Every field here is read by at least one backend. The previous version of this
repository shipped YAML files that no code path ever opened, so anything that
cannot be honoured is rejected at load time rather than ignored.

Defaults encode current practice rather than 2023 folklore. In particular
``AdapterConfig.target_modules`` defaults to ``"all-linear"``: Schulman et al.
(2025), reproduced in TRL's own documentation, found that restricting LoRA to
attention projections underperforms and that raising the rank does not
compensate. See docs/lora-practice.md.

:class:`~finetune.run.RunConfig` composes these into a whole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ConfigError


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """Low-rank adapter geometry. Ignored when ``method`` is ``full``."""

    rank: int = 16
    #: LoRA scaling numerator. Deliberately *not* tied to rank: `alpha = 2 * rank`
    #: is folklore, and decoupling the two is what lets rank change without
    #: re-tuning the learning rate.
    alpha: int = 32
    dropout: float = 0.0
    #: ``"all-linear"`` targets every linear projection, including the MLP.
    #: A tuple of module-name suffixes restricts it. Attention-only targeting is
    #: possible but is the documented anti-pattern.
    target_modules: str | tuple[str, ...] = "all-linear"
    #: Rank-stabilised scaling (alpha / sqrt(rank)) instead of alpha / rank.
    #: Keeps the update magnitude stable as rank grows. torch backend only.
    use_rslora: bool = False
    #: MLX applies adapters to the last N layers; -1 means all of them.
    num_layers: int = -1

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ConfigError(f"adapter.rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ConfigError(f"adapter.alpha must be positive, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"adapter.dropout must be in [0, 1), got {self.dropout}")
        if isinstance(self.target_modules, str) and self.target_modules != "all-linear":
            raise ConfigError(
                "adapter.target_modules must be 'all-linear' or a list of module "
                f"name suffixes, got {self.target_modules!r}"
            )
        if isinstance(self.target_modules, tuple) and not self.target_modules:
            raise ConfigError("adapter.target_modules list must not be empty")
        if self.num_layers == 0 or self.num_layers < -1:
            raise ConfigError(f"adapter.num_layers must be -1 or positive, got {self.num_layers}")

    @property
    def scale(self) -> float:
        """The multiplier applied to the low-rank update.

        Standard LoRA divides by the rank, so the update shrinks as rank grows
        and the learning rate needs re-tuning. rsLoRA divides by its square root
        instead, which keeps the magnitude roughly stable across ranks.
        """
        if self.use_rslora:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Base-model quantisation. Required by, and only used by, ``qlora``."""

    bits: int = 4
    group_size: int = 64

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4, 6, 8):
            raise ConfigError(f"quantization.bits must be one of 2/3/4/6/8, got {self.bits}")
        if self.group_size <= 0 or self.group_size % 16 != 0:
            raise ConfigError(
                f"quantization.group_size must be a positive multiple of 16, got {self.group_size}"
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimiser and schedule settings shared by both backends."""

    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    #: Total optimiser steps. Mutually exclusive with ``epochs``.
    iters: int | None = None
    #: Passes over the dataset. Mutually exclusive with ``iters``.
    epochs: int | None = None
    learning_rate: float | None = None
    max_seq_length: int = 2048
    seed: int = 42
    grad_checkpoint: bool = False
    #: Exclude prompt tokens from the loss, training only on completions. Both
    #: backends support this; leaving it off trains the model to predict the
    #: user's own turn.
    mask_prompt: bool = True
    steps_per_report: int = 10
    steps_per_eval: int = 100
    save_every: int = 100
    val_batches: int = 25

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_seq_length",
            "steps_per_report",
            "steps_per_eval",
            "save_every",
        ):
            value: int = getattr(self, name)
            if value <= 0:
                raise ConfigError(f"training.{name} must be positive, got {value}")
        if self.iters is not None and self.epochs is not None:
            raise ConfigError("training.iters and training.epochs are mutually exclusive")
        if self.iters is None and self.epochs is None:
            raise ConfigError("set exactly one of training.iters or training.epochs")
        if self.iters is not None and self.iters <= 0:
            raise ConfigError(f"training.iters must be positive, got {self.iters}")
        if self.epochs is not None and self.epochs <= 0:
            raise ConfigError(f"training.epochs must be positive, got {self.epochs}")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ConfigError(f"training.learning_rate must be positive, got {self.learning_rate}")
        if self.val_batches < -1 or self.val_batches == 0:
            raise ConfigError(
                f"training.val_batches must be -1 (all) or positive, got {self.val_batches}"
            )

    @property
    def effective_batch_size(self) -> int:
        """Batch size actually seen by the optimiser between updates."""
        return self.batch_size * self.gradient_accumulation_steps


__all__ = ["AdapterConfig", "QuantizationConfig", "TrainingConfig"]
