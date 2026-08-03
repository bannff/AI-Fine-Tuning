"""The capability matrix: which method each backend actually implements.

This module is the single source of truth behind the support table in the
README. It exists because the previous version of this repository advertised
methods it did not implement; encoding the matrix in code means the claim and
the behaviour cannot drift apart, and ``finetune capabilities`` prints the same
facts the docs do.

Only implemented combinations appear here. Methods that exist upstream but are
not wrapped by this package (preference optimisation, reward modelling) are
absent on purpose rather than present as stubs.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import Enum


class Method(str, Enum):
    """A fine-tuning method, as the user selects it."""

    LORA = "lora"
    """Low-rank adapters on frozen full-precision weights."""

    DORA = "dora"
    """Weight-decomposed LoRA: direction via low-rank update, magnitude learned."""

    QLORA = "qlora"
    """Low-rank adapters on a quantised frozen base, for memory-bound machines."""

    FULL = "full"
    """Every parameter is updated. No adapter."""

    def __str__(self) -> str:
        return self.value


class Backend(str, Enum):
    """A training engine."""

    MLX = "mlx"
    """Apple Silicon, via the ``mlx_lm`` trainer."""

    TORCH = "torch"
    """Portable, via ``trl`` (SFT loop) and ``peft`` (adapters)."""

    def __str__(self) -> str:
        return self.value


#: Which optional extra provides each backend.
BACKEND_EXTRA: dict[Backend, str] = {
    Backend.MLX: "apple",
    Backend.TORCH: "torch",
}

#: The import each backend needs before it can run.
BACKEND_PROBE: dict[Backend, str] = {
    Backend.MLX: "mlx_lm",
    Backend.TORCH: "trl",
}


@dataclass(frozen=True, slots=True)
class Capability:
    """One implemented (method, backend) pair, with its real constraints."""

    method: Method
    backend: Backend
    note: str
    #: Extra imports required beyond the backend's own extra, if any.
    extra_requirements: tuple[str, ...] = ()


#: Every combination this package implements. Anything absent is unsupported,
#: and :func:`resolve` will say so rather than quietly substituting a default.
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        Method.LORA,
        Backend.MLX,
        "mlx_lm --fine-tune-type lora. Adapters saved as MLX safetensors.",
    ),
    Capability(
        Method.DORA,
        Backend.MLX,
        "mlx_lm --fine-tune-type dora.",
    ),
    Capability(
        Method.FULL,
        Backend.MLX,
        "mlx_lm --fine-tune-type full. Gradient checkpointing on by default; "
        "memory scales with the whole model, not the adapter.",
    ),
    Capability(
        Method.QLORA,
        Backend.MLX,
        "Base model is quantised first with mlx_lm.convert, then adapted with "
        "LoRA. Quantisation is a separate step in MLX, not a trainer flag.",
    ),
    Capability(
        Method.LORA,
        Backend.TORCH,
        "trl SFTTrainer with a peft LoraConfig. Runs on CUDA, MPS or CPU.",
    ),
    Capability(
        Method.DORA,
        Backend.TORCH,
        "As LoRA, with peft use_dora=True. Slower per step than plain LoRA.",
    ),
    Capability(
        Method.QLORA,
        Backend.TORCH,
        "4-bit base via bitsandbytes. bitsandbytes is CUDA-only, so this path "
        "does not run on Apple Silicon; use the mlx backend there instead.",
        extra_requirements=("bitsandbytes",),
    ),
)

#: Methods deliberately not implemented, and where to get them. Surfaced by the
#: CLI so the omission reads as a decision rather than an oversight.
NOT_IMPLEMENTED: dict[str, str] = {
    "dpo/orpo/kto": (
        "Preference optimisation. Available upstream via trl's DPOTrainer and "
        "ORPOTrainer on the torch backend; mlx_lm core has no preference "
        "objective (third-party mlx-lm-lora or SiLLM add one)."
    ),
    "grpo/rloo": (
        "Online RL. Needs a reward function and many generations per step; "
        "realistically wants vLLM, which is out of scope for a single machine."
    ),
}


def supported_methods(backend: Backend) -> tuple[Method, ...]:
    """Return the methods ``backend`` implements, in declaration order."""
    return tuple(c.method for c in CAPABILITIES if c.backend == backend)


def supporting_backends(method: Method) -> tuple[Backend, ...]:
    """Return the backends that implement ``method``, in declaration order."""
    return tuple(c.backend for c in CAPABILITIES if c.method == method)


def resolve(method: Method, backend: Backend) -> Capability:
    """Look up a capability, or raise if the combination is not implemented.

    Raises:
        UnsupportedCombinationError: if the pair is absent from the matrix.
    """
    from .errors import UnsupportedCombinationError

    for capability in CAPABILITIES:
        if capability.method == method and capability.backend == backend:
            return capability
    raise UnsupportedCombinationError(
        method=str(method),
        backend=str(backend),
        supported_by=tuple(str(b) for b in supporting_backends(method)),
    )


def is_installed(backend: Backend) -> bool:
    """Whether ``backend``'s dependencies are importable right now.

    Uses :func:`importlib.util.find_spec` so probing a backend never triggers
    the multi-second import of torch or mlx.
    """
    try:
        return importlib.util.find_spec(BACKEND_PROBE[backend]) is not None
    except (ImportError, ValueError):
        return False


def installed_backends() -> tuple[Backend, ...]:
    """Return every backend whose dependencies are present."""
    return tuple(b for b in Backend if is_installed(b))


__all__ = [
    "BACKEND_EXTRA",
    "BACKEND_PROBE",
    "CAPABILITIES",
    "NOT_IMPLEMENTED",
    "Backend",
    "Capability",
    "Method",
    "installed_backends",
    "is_installed",
    "resolve",
    "supported_methods",
    "supporting_backends",
]
