# Development workflow

## Setup

```bash
uv sync --extra all          # or --extra apple / --extra torch
```

`uv.lock` is committed. Use `uv sync --locked` to install exactly what CI does.

## Before opening a PR

Everything CI checks, in the order it fails fastest:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
bash scripts/verify_repo.sh
```

`pytest -m "not slow"` skips the training smoke tests when iterating.

## Training workflow

```bash
finetune prepare-data raw.jsonl data/mydata   # split, validating as it goes
finetune validate-data data/mydata            # explicit re-check
finetune plan configs/my-run.yaml             # what would happen, no side effects
finetune train configs/my-run.yaml
```

Always `plan` before `train`. It prints the resolved config including defaults you
did not set, the exact command it will invoke, and any advisories. It writes
nothing.

## Adding a method or backend

1. Add the enum member to `capabilities.py` **and** a `Capability` entry with a
   note stating its real constraints. A member without an entry is a stub.
2. Implement `preflight` / `plan` / `train`. Keep `plan` free of side effects and
   of heavy imports.
3. Add a `configs/` example that `tests/test_config.py` will load.
4. Add construction tests that run without the backend's extra installed, plus a
   `slow`-marked smoke test that trains for a few steps.
5. Update the README support matrix. The table and the matrix must agree.

## Changing a default

Defaults are load-bearing and cited. Update `docs/lora-practice.md` with the
reasoning and a source in the same change, and update the test that pins the old
value so the change is visible in the diff rather than silent.

## Branching and PRs

- Branch names: `feature/`, `bugfix/`, `chore/`, `docs/`.
- Never commit to `main` directly. `main` is the published branch.
- Fill out `pull_request_template.md`.
- CI must be green, including `verify_repo.sh`.

## Issue tracking

Beads, local-first:

```bash
bd list
bd create "Add ORPO on the torch backend"
bd update <id> --claim
```

## What not to commit

Model weights, adapters, checkpoints, tokenizers, training logs, prepared
datasets. `data/` and `outputs/` are gitignored; `verify_repo.sh` fails on tracked
files over 512 KB and on empty tracked files.
