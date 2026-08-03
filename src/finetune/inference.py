"""Generate from a trained adapter on Apple Silicon.

Enough to check that a run produced something, without pretending to be a
serving layer — ``mlx_lm.server`` already exists for that.

The sampler is built with ``mlx_lm.sample_utils.make_sampler`` rather than
composing ``apply_min_p`` / ``apply_top_p`` / softmax by hand. The hand-rolled
version this replaces applied temperature *after* min-p and top-p filtering,
which changes the distribution those cutoffs were computed against.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BackendUnavailableError, FinetuneError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Decoding parameters.

    Defaults follow mlx_lm's own guidance: min-p adapts the cutoff to model
    confidence, with top-p as a backstop, and a light repetition penalty.
    """

    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.05
    repetition_penalty: float = 1.05
    repetition_context_size: int = 64
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise FinetuneError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise FinetuneError(f"top_p must be in [0, 1], got {self.top_p}")
        if not 0.0 <= self.min_p <= 1.0:
            raise FinetuneError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.max_tokens <= 0:
            raise FinetuneError(f"max_tokens must be positive, got {self.max_tokens}")


def _require_mlx() -> None:
    import importlib.util

    if importlib.util.find_spec("mlx_lm") is None:
        raise BackendUnavailableError(backend="mlx", extra="apple", missing="mlx_lm")


def load_adapted(model: str, adapter_path: str | Path | None = None) -> tuple[Any, Any]:
    """Load a base model with an optional adapter applied.

    Returns:
        ``(model, tokenizer)`` as mlx_lm returns them.
    """
    _require_mlx()
    from mlx_lm import load

    if adapter_path is not None:
        resolved = Path(adapter_path)
        if not resolved.exists():
            raise FinetuneError(
                f"adapter path not found: {resolved}. Train one first with 'finetune train'."
            )
        loaded = load(model, adapter_path=str(resolved))
    else:
        loaded = load(model)
    # mlx_lm.load returns a 3-tuple when return_config is set; we never set it.
    return loaded[0], loaded[1]


def build_sampler(config: SamplingConfig) -> tuple[Any, list[Any]]:
    """Build ``(sampler, logits_processors)`` for :func:`mlx_lm.generate`."""
    _require_mlx()
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    sampler = make_sampler(temp=config.temperature, top_p=config.top_p, min_p=config.min_p)
    processors = make_logits_processors(
        repetition_penalty=config.repetition_penalty,
        repetition_context_size=config.repetition_context_size,
    )
    return sampler, processors


def complete(
    model: Any,
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    config: SamplingConfig | None = None,
) -> str:
    """Generate one assistant reply for a chat history.

    Applies the tokenizer's own chat template, so the prompt format matches
    whatever the model was trained with instead of a hand-built string.
    """
    _require_mlx()
    from mlx_lm import generate

    sampling = config or SamplingConfig()
    sampler, processors = build_sampler(sampling)
    prompt = tokenizer.apply_chat_template(
        list(messages), add_generation_prompt=True, tokenize=False
    )
    return str(
        generate(
            model,
            tokenizer,
            prompt,
            max_tokens=sampling.max_tokens,
            sampler=sampler,
            logits_processors=processors,
            verbose=False,
        )
    )


__all__ = ["SamplingConfig", "build_sampler", "complete", "load_adapted"]
