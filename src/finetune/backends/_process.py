"""Subprocess execution for backends that drive an external trainer.

Output is streamed rather than buffered so a long training run shows progress,
and captured at the same time so the caller can pull metrics out of it.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from ..errors import TrainingFailedError

logger = logging.getLogger(__name__)

#: mlx_lm reports progress as e.g.
#: "Iter 10: Train loss 2.417, Learning Rate 1.000e-05, ..."
#:
#: ``nan`` and ``inf`` are matched deliberately. A diverged run is the single most
#: important thing to report, and a number-only pattern would silently drop it,
#: leaving the caller with no step count and no loss rather than a clear signal.
_NUMBER = r"(nan|-?inf|-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
_ITER_LOSS = re.compile(rf"Iter\s+(\d+):\s+Train loss\s+{_NUMBER}", re.IGNORECASE)
_VAL_LOSS = re.compile(rf"Iter\s+(\d+):\s+Val loss\s+{_NUMBER}", re.IGNORECASE)

#: Cap on retained output. Training logs can be very long; only the tail is
#: useful for diagnosing a failure.
_MAX_RETAINED_LINES = 200


def module_argv(module: str, subcommand: str | None = None) -> tuple[str, ...]:
    """Build the interpreter-anchored prefix for invoking a module's CLI.

    Uses ``sys.executable -m`` rather than a console script. Both forms work for
    ``mlx_lm``, but the console script resolves through ``PATH`` and so can
    silently target a different environment than the one running this code.
    """
    argv = (sys.executable, "-m", module)
    return (*argv, subcommand) if subcommand else argv


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """Result of a completed subprocess."""

    returncode: int
    tail: tuple[str, ...]
    last_step: int | None = None
    last_train_loss: float | None = None
    last_val_loss: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def run_streaming(argv: Sequence[str], *, label: str) -> ProcessOutcome:
    """Run ``argv``, echoing output while scraping progress metrics from it.

    Args:
        argv: A fully-formed argument list. Never a shell string.
        label: Human name for the stage, used in error messages.

    Raises:
        TrainingFailedError: if the executable is missing. A non-zero exit is
            reported in the outcome instead, so callers can decide.
    """
    command = [str(part) for part in argv]
    logger.info("%s: %s", label, " ".join(command))

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise TrainingFailedError(
            f"{label}: could not execute {command[0]!r}. "
            "Is the environment for this backend installed?"
        ) from exc

    tail: list[str] = []
    last_step: int | None = None
    last_train: float | None = None
    last_val: float | None = None

    assert process.stdout is not None  # noqa: S101 - guaranteed by stdout=PIPE
    with process:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            print(line, flush=True)
            tail.append(line)
            if len(tail) > _MAX_RETAINED_LINES:
                del tail[0]
            if match := _ITER_LOSS.search(line):
                last_step, last_train = int(match.group(1)), float(match.group(2))
            elif match := _VAL_LOSS.search(line):
                last_val = float(match.group(2))

    return ProcessOutcome(
        returncode=process.returncode,
        tail=tuple(tail),
        last_step=last_step,
        last_train_loss=last_train,
        last_val_loss=last_val,
    )


def failure_detail(label: str, outcome: ProcessOutcome, *, lines: int = 15) -> str:
    """Build an error message from a failed outcome's output tail."""
    excerpt = "\n".join(outcome.tail[-lines:]) or "(no output captured)"
    return f"{label} exited with code {outcome.returncode}.\nLast output:\n{excerpt}"


__all__ = ["ProcessOutcome", "failure_detail", "module_argv", "run_streaming"]
