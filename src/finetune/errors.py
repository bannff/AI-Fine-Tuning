"""Exception types raised by :mod:`finetune`.

Every error carries enough context to act on without reading a traceback,
because most of them surface through the CLI where a traceback is noise.
"""

from __future__ import annotations


class FinetuneError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(FinetuneError):
    """A run configuration is missing, malformed, or self-contradictory."""


class DatasetSchemaError(FinetuneError):
    """A dataset record does not match any schema the trainers accept."""


class BackendUnavailableError(FinetuneError):
    """A backend was requested but its optional dependencies are not installed."""

    def __init__(self, backend: str, extra: str, missing: str) -> None:
        self.backend = backend
        self.extra = extra
        self.missing = missing
        super().__init__(
            f"backend {backend!r} needs the {extra!r} extra "
            f"(missing import: {missing}). Install it with: "
            f'pip install "finetune[{extra}]"'
        )


class UnsupportedCombinationError(FinetuneError):
    """A (method, backend) pair is not implemented.

    Raised instead of silently degrading to something the caller did not ask
    for. The message names the backends that *do* support the method so the
    error is actionable.
    """

    def __init__(self, method: str, backend: str, supported_by: tuple[str, ...]) -> None:
        self.method = method
        self.backend = backend
        self.supported_by = supported_by
        alternatives = ", ".join(supported_by) if supported_by else "no backend in this package"
        super().__init__(
            f"method {method!r} is not supported by backend {backend!r}. "
            f"Supported by: {alternatives}."
        )


class TrainingFailedError(FinetuneError):
    """A training run started but did not complete successfully."""


__all__ = [
    "BackendUnavailableError",
    "ConfigError",
    "DatasetSchemaError",
    "FinetuneError",
    "TrainingFailedError",
    "UnsupportedCombinationError",
]
