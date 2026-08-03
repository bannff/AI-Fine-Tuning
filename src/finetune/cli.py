"""Command-line interface: argument parsing and dispatch.

Five commands, each a thin shell over library code so everything the CLI can do
is also callable:

``capabilities``   print the support matrix and what is installed
``validate-data``  check a prepared split directory against the schemas
``prepare-data``   split one JSONL into train/valid/test
``plan``           resolve a config and show what would run, without running it
``train``          run it

The implementations live in :mod:`finetune.commands`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import commands
from .capabilities import Backend, Method
from .data.split import DEFAULT_SPLIT
from .errors import FinetuneError

logger = logging.getLogger("finetune")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so tests can introspect it."""
    parser = argparse.ArgumentParser(
        prog="finetune",
        description="LoRA / DoRA / QLoRA / full fine-tuning across MLX and PyTorch.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log at DEBUG level")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("capabilities", help="show which method/backend pairs are supported")

    validate = sub.add_parser("validate-data", help="validate a train/valid/test directory")
    validate.add_argument("data_dir", type=Path)

    prepare = sub.add_parser("prepare-data", help="split a JSONL into train/valid/test")
    prepare.add_argument("dataset", type=Path, help="input .jsonl")
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument(
        "--ratio",
        type=float,
        nargs=3,
        default=list(DEFAULT_SPLIT),
        metavar=("TRAIN", "VALID", "TEST"),
        help="split fractions, must sum to 1.0 (default: 0.8 0.1 0.1)",
    )
    prepare.add_argument("--seed", type=int, default=42, help="shuffle seed (default: 42)")
    prepare.add_argument(
        "--no-shuffle",
        action="store_true",
        help="keep input order; use only when the order is meaningful",
    )
    prepare.add_argument("--limit", type=int, help="use only the first N records")

    for name, help_text in (
        ("plan", "resolve a config and print what would run"),
        ("train", "run a fine-tuning job"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("config", type=Path, help="path to a YAML run config")
        cmd.add_argument("--method", choices=[str(m) for m in Method], help="override method")
        cmd.add_argument("--backend", choices=[str(b) for b in Backend], help="override backend")
        cmd.add_argument("--output-dir", type=Path, help="override output_dir")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises for user error."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        if args.command == "capabilities":
            return commands.capabilities()
        if args.command == "validate-data":
            return commands.validate_data(args.data_dir)
        if args.command == "prepare-data":
            return commands.prepare_data(args)
        if args.command == "plan":
            return commands.plan(args)
        if args.command == "train":
            return commands.train(args)
    except FinetuneError as exc:
        # Expected failure modes get a clean message, not a traceback.
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
