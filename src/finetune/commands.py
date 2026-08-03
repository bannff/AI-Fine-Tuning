"""Command implementations.

Separated from :mod:`finetune.cli`, which owns argument parsing and dispatch, so
each stays under the line ceiling and so every command is callable without going
through argparse.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .backends import get_backend
from .capabilities import (
    BACKEND_EXTRA,
    CAPABILITIES,
    NOT_IMPLEMENTED,
    Backend,
    Method,
    installed_backends,
    is_installed,
    resolve,
)
from .config_io import dump_run_config, load_run_config
from .data.io import read_jsonl
from .data.layout import validate_split_dir
from .data.split import write_splits
from .run import RunConfig

logger = logging.getLogger("finetune")


def capabilities() -> int:
    """Print the support matrix, install state, and deliberate omissions."""
    installed = set(installed_backends())
    print("Supported method/backend pairs\n")
    width = max(len(str(c.method)) for c in CAPABILITIES)
    for backend in Backend:
        if backend in installed:
            mark = "installed"
        else:
            mark = f"not installed (pip install 'finetune[{BACKEND_EXTRA[backend]}]')"
        print(f"{backend}  [{mark}]")
        for capability in (c for c in CAPABILITIES if c.backend == backend):
            print(f"  {capability.method!s:<{width}}  {capability.note}")
        print()
    print("Not implemented here, on purpose:")
    for name, reason in NOT_IMPLEMENTED.items():
        print(f"  {name}: {reason}")
    return 0


def validate_data(data_dir: Path) -> int:
    """Validate a prepared split directory and report record counts."""
    schemas = validate_split_dir(data_dir)
    schema = next(iter(schemas.values()))
    print(f"{data_dir}: valid, schema '{schema}'")
    for name in sorted(schemas):
        count = sum(1 for _ in read_jsonl(data_dir / name))
        print(f"  {name}: {count} records")
    return 0


def prepare_data(args: argparse.Namespace) -> int:
    """Split one JSONL file into train/valid/test, validating as it goes."""
    source: Any = (
        read_jsonl(args.dataset) if args.limit is None else _take(args.dataset, args.limit)
    )
    counts = write_splits(
        source,
        args.output_dir,
        ratio=tuple(args.ratio),
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    print(f"wrote {counts.total} records to {args.output_dir}")
    for name, count in counts.as_dict().items():
        if count:
            print(f"  {name}.jsonl: {count}")
    return 0


def _take(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in read_jsonl(path):
        if len(records) >= limit:
            break
        records.append(record)
    return records


def plan(args: argparse.Namespace) -> int:
    """Resolve a config and print what would run, without running it."""
    config = _load_with_overrides(args)
    _report_advisories(config)

    # Whether a combination is supported is a static fact about this package, so
    # it is checked even when the backend is not installed. Only the dependency
    # and dataset checks below are conditional.
    resolve(config.method, config.backend)

    backend = get_backend(config.backend)
    if not is_installed(config.backend):
        logger.warning(
            "backend '%s' is not installed; showing the plan without validating the dataset",
            config.backend,
        )
    else:
        backend.preflight(config)

    print(backend.plan(config).render())
    print("\nResolved configuration:\n")
    print(dump_run_config(config))
    return 0


def train(args: argparse.Namespace) -> int:
    """Run a fine-tuning job to completion."""
    config = _load_with_overrides(args)
    _report_advisories(config)
    backend = get_backend(config.backend)
    backend.preflight(config)

    logger.info("%s", backend.plan(config).render())
    result = backend.train(config)
    print(result.summary())
    return 0 if result.succeeded else 1


def _load_with_overrides(args: argparse.Namespace) -> RunConfig:
    """Load the config file, then apply any command-line overrides."""
    config = load_run_config(args.config)
    changes: dict[str, object] = {}
    if args.method:
        changes["method"] = Method(args.method)
    if args.backend:
        changes["backend"] = Backend(args.backend)
    if args.output_dir:
        changes["output_dir"] = args.output_dir.resolve()
    return config.with_overrides(**changes) if changes else config


def _report_advisories(config: RunConfig) -> None:
    for note in config.advisories():
        logger.warning("%s", note)


__all__ = ["capabilities", "plan", "prepare_data", "train", "validate_data"]
