# Dev principles

The engineering contract for this repository. Audit work against it before
finishing.

## Architecture

- **One config surface.** All run settings enter through `RunConfig`. Backends
  read the config; they do not accept loose keyword arguments or read the
  environment.
- **Behind a Protocol.** Backends implement `TrainingBackend` and are resolved
  lazily by `backends.get_backend`. Adding a backend must not require editing
  anything except `capabilities.py` and that resolver.
- **Import-light core.** Only PyYAML at module scope. Anything from torch, trl,
  peft, transformers, datasets, mlx or mlx-lm is imported inside the function that
  needs it. CI asserts `import finetune` stays clean.
- **No module named after its dependency.** `mlx_backend.py`, not `mlx.py`. A
  top-level `peft/` once shadowed the real dependency here and broke the main code
  path; `verify_repo.sh` checks for recurrences.

## Code quality

- **Under 200 lines per file.** Enforced by `verify_repo.sh` over `src/` and
  `tests/`. Split by responsibility, not by line count.
- **Single responsibility.** `data/` does not know about backends; `backends/` do
  not parse YAML; `config_io.py` does not train.
- **Typed and linted.** `ruff check`, `ruff format --check`, and `mypy` in strict
  mode all pass. Relax a mypy rule per-module with a comment explaining why, never
  globally.
- **Errors are actionable.** Raise from the `FinetuneError` hierarchy with a
  message that names the field, file and line at fault, and says what to do. The
  CLI turns these into a message and exit code 2, not a traceback.

## Fail loudly

The failure mode to avoid is reporting success while doing something else. Three
concrete rules:

- **Unknown config keys are errors**, with a did-you-mean suggestion. Not warnings.
- **Unsupported (method, backend) pairs raise.** Never fall back to a default the
  caller did not ask for.
- **Validate datasets before training**, not after the model has loaded, and name
  the offending line.

## Fine-tuning specifics

- **Defaults follow published practice**, and the reasoning lives in
  `docs/lora-practice.md` with a citation. `target_modules: all-linear`,
  `alpha` independent of `rank`, `mask_prompt: true`, learning rate by method.
- **Record provenance.** Every run writes its fully resolved config, defaults
  included, next to the output.
- **Deterministic where it can be.** Seeded splitting, seeded training, exact
  split allocation.
- **A method is implemented or absent.** No stubs, no `NotImplementedError`
  placeholders in the capability matrix.

## Tests

- **Test behaviour, not the implementation restated.** An assertion that
  recomputes what the code does proves nothing. The suite this replaced asserted
  `train_len == int(total * 0.8)` against an implementation that computed exactly
  that.
- **Green must mean working.** The previous suite passed *because* the real
  dependencies were absent from the venv it built. Tests that need an extra are
  skipped explicitly and CI runs a job that installs it.
- **Test the rejections.** Most of the value here is in refusing bad input, so
  most tests assert that bad input is refused.
- **No network in tests.** The tiny-model fixture builds a real model and
  tokenizer on disk.

## Documentation

- **Every README claim is backed by something that runs.** If it is deleted from
  the code, delete it from the docs in the same change.
- **Explain why, not what.** Comments justify a choice or record a constraint;
  they do not narrate the next line.
- **Correct docs when practice changes.** `docs/lora-practice.md` previously
  recommended attention-only targeting; superseded guidance gets rewritten and
  says so.

## Repository hygiene

- **Code only.** No weights, adapters, checkpoints, tokenizers or training logs.
  `verify_repo.sh` rejects tracked files over 512 KB and empty tracked files.
- **Reproducible environment.** `pyproject.toml` declares floors, `uv.lock` pins
  exactly, both are committed.
