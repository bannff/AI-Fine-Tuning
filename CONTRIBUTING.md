# Contributing

## Setup

```bash
uv sync --extra all     # or --extra apple / --extra torch
```

`uv.lock` is committed. `uv sync --locked` installs exactly what CI does.

## The gates

Everything CI checks, ordered so the fastest failure comes first:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
bash scripts/verify_repo.sh
```

`uv run pytest -m "not slow"` skips the training smoke tests while iterating.

The `apple` extra has no Linux wheels, so the MLX training tests are skipped in
CI and only run locally on Apple Silicon. Their command construction is covered
by platform-independent unit tests in `tests/test_mlx_backend.py`.

## Conventions

Full contract in [.agents/steering/dev-principles.md](.agents/steering/dev-principles.md).
The rules that reviewers actually enforce:

- **Every file in `src/` stays under 200 lines.** Checked by `verify_repo.sh`.
- **Never name a module after a package it imports.** `mlx_backend.py`, not
  `mlx.py`. A top-level `peft/` once shadowed the real dependency here and made
  the main code path unimportable.
- **The core stays import-light.** PyYAML only at module scope. Anything from
  torch, trl, peft, transformers, datasets, mlx or mlx-lm is imported inside the
  function that needs it, and CI asserts `import finetune` pulls in neither.
- **Fail loudly.** Unknown config keys are errors, unsupported method/backend
  pairs raise, datasets are validated before training. Reporting success while
  doing something else is the failure mode to avoid.
- **Test behaviour, not the implementation restated.** An assertion that
  recomputes what the code does proves nothing.
- **Code only.** No weights, adapters, checkpoints or logs in git.

## Adding a method or backend

1. Add the `Method`/`Backend` member **and** a `Capability` entry in
   `capabilities.py` stating its real constraints. A member with no entry is a
   stub.
2. Implement `preflight` / `plan` / `train`. Keep `plan` free of side effects and
   heavy imports.
3. Add a `configs/` example; `tests/test_config.py` loads all of them.
4. Add construction tests that pass without the backend's extra, plus a
   `slow`-marked smoke test that trains a few steps.
5. Update the README support matrix so it agrees with `capabilities.py`.

## Changing a default

Defaults are cited, not chosen by taste. Update `docs/lora-practice.md` with the
reasoning and a source in the same change, and update the test that pinned the
old value so the change shows up in the diff.

## PRs

- Branch names: `feature/`, `bugfix/`, `chore/`, `docs/`.
- Fill out the pull request template.
- CI must be green.

By contributing you agree your work is licensed under the MIT License.
