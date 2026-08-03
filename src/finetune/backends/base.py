"""The backend seam: one Protocol, two engines.

Every backend is split into three stages so that everything except the last is
testable without installing torch or mlx:

``preflight``  validate the config against this backend's real constraints
``plan``       describe exactly what will run, without running it
``train``      execute

Keeping ``plan`` separate is what makes ``finetune train --dry-run`` honest and
what lets the test suite assert on a constructed ``mlx_lm`` argv rather than
mocking a subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from ..capabilities import Backend, Method
from ..run import RunConfig


@dataclass(frozen=True, slots=True)
class Plan:
    """What a backend intends to do for a given config."""

    backend: Backend
    method: Method
    summary: str
    #: Ordered, human-readable stages. QLoRA on MLX has two; LoRA has one.
    stages: tuple[str, ...] = ()
    #: The exact argv for backends that shell out. ``None`` for in-process ones.
    argv: tuple[str, ...] | None = None

    def render(self) -> str:
        """Format the plan for terminal output."""
        lines = [self.summary]
        for index, stage in enumerate(self.stages, start=1):
            lines.append(f"  {index}. {stage}")
        if self.argv is not None:
            lines.append("  command: " + " ".join(self.argv))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The outcome of a completed (or failed) run."""

    backend: Backend
    method: Method
    output_dir: Path
    succeeded: bool
    detail: str = ""
    steps: int | None = None
    final_loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        status = "completed" if self.succeeded else "failed"
        parts = [f"{self.method} on {self.backend} {status}"]
        if self.steps is not None:
            parts.append(f"{self.steps} steps")
        if self.final_loss is not None:
            parts.append(f"final loss {self.final_loss:.4f}")
        parts.append(f"output {self.output_dir}")
        return ", ".join(parts)


@runtime_checkable
class TrainingBackend(Protocol):
    """A fine-tuning engine.

    Implementations must not import their heavy dependencies at module scope;
    :mod:`finetune.backends` resolves a backend before its extra is known to be
    installed, and ``finetune capabilities`` must work with neither installed.
    """

    backend: ClassVar[Backend]

    def preflight(self, config: RunConfig) -> None:
        """Raise if this backend cannot run ``config``.

        Checks the capability matrix, dataset layout and any backend-specific
        constraint. Must not mutate anything or start work.
        """
        ...

    def plan(self, config: RunConfig) -> Plan:
        """Describe the run without executing it. Assumes preflight passed."""
        ...

    def train(self, config: RunConfig) -> TrainingResult:
        """Run training to completion.

        Raises:
            TrainingFailedError: if the run starts but does not finish cleanly.
        """
        ...


__all__ = ["Plan", "TrainingBackend", "TrainingResult"]
