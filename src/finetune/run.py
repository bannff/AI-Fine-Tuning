"""The complete run configuration.

Split from :mod:`finetune.config`, which owns the individual sections, so that
each file stays under the repository's line ceiling. This module is about the
*whole*: cross-section coherence, resolved defaults, and advisories.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .capabilities import Backend, Method
from .config import AdapterConfig, QuantizationConfig, TrainingConfig
from .errors import ConfigError

#: Learning rates by method. Low-rank updates tolerate -- and want -- a rate
#: roughly two orders of magnitude above full fine-tuning; the 1/rank scaling
#: makes the optimum close to rank-independent. See docs/lora-practice.md.
DEFAULT_LEARNING_RATE: dict[Method, float] = {
    Method.LORA: 2e-4,
    Method.DORA: 2e-4,
    Method.QLORA: 2e-4,
    Method.FULL: 2e-6,
}

#: Effective batch size above which low-rank training is known to degrade
#: relative to full fine-tuning. Drives an advisory, not a hard limit.
LORA_BATCH_ADVISORY = 32


@dataclass(frozen=True, slots=True)
class RunConfig:
    """A complete, validated fine-tuning run."""

    model: str
    data_dir: Path
    output_dir: Path
    method: Method = Method.LORA
    backend: Backend = Backend.MLX
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigError("model must be a non-empty HF repo id or local path")
        if self.method is Method.QLORA and self.quantization is None:
            raise ConfigError(
                "method 'qlora' requires a quantization block; add "
                "'quantization: {bits: 4, group_size: 64}' or choose method 'lora'"
            )
        if self.method is Method.FULL and self.quantization is not None:
            raise ConfigError("method 'full' cannot be combined with quantization")

    @property
    def learning_rate(self) -> float:
        """The resolved learning rate, defaulted by method if unset."""
        if self.training.learning_rate is not None:
            return self.training.learning_rate
        return DEFAULT_LEARNING_RATE[self.method]

    @property
    def uses_adapter(self) -> bool:
        """Whether this run produces an adapter rather than full weights."""
        return self.method is not Method.FULL

    def advisories(self) -> tuple[str, ...]:
        """Non-fatal warnings about the configuration.

        Separate from validation so unusual-but-intentional setups stay runnable
        while still telling the operator what is unusual about them.
        """
        notes: list[str] = []
        if self.uses_adapter and self.training.effective_batch_size > LORA_BATCH_ADVISORY:
            notes.append(
                f"effective batch size {self.training.effective_batch_size} exceeds "
                f"{LORA_BATCH_ADVISORY}; low-rank training tolerates large batches "
                "less well than full fine-tuning, and raising the rank does not fix it"
            )
        if self.uses_adapter and self.adapter.target_modules != "all-linear":
            notes.append(
                "adapter.target_modules is restricted; targeting all linear layers "
                "generally outperforms attention-only adaptation at equal parameter "
                "count (see docs/lora-practice.md)"
            )
        if self.method is Method.FULL and not self.training.grad_checkpoint:
            notes.append(
                "full fine-tuning without training.grad_checkpoint will use "
                "substantially more memory"
            )
        return tuple(notes)

    def with_overrides(self, **changes: object) -> RunConfig:
        """Return a copy with top-level fields replaced. Re-validates."""
        return replace(self, **changes)  # type: ignore[arg-type]


__all__ = ["DEFAULT_LEARNING_RATE", "LORA_BATCH_ADVISORY", "RunConfig"]
