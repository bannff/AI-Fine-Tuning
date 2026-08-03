# Project overview

Fine-tuning causal language models with LoRA, DoRA, QLoRA or full-parameter
updates, across an MLX backend (Apple Silicon) and a PyTorch backend (portable),
behind one YAML config surface.

## Structure

```
src/finetune/
├── capabilities.py     supported (method, backend) pairs, enforced at runtime
├── config.py           the config sections, with field validation
├── run.py              RunConfig: coherence across sections, resolved defaults
├── config_io.py        strict YAML loading; unknown keys are errors
├── errors.py           the exception hierarchy
├── inference.py        generate from a trained adapter (MLX)
├── cli.py              argument parsing and dispatch
├── commands.py         command implementations
├── data/
│   ├── schema.py       the three record schemas both trainers accept
│   ├── layout.py       train/valid/test directory validation
│   ├── split.py        deterministic, exact train/valid/test allocation
│   └── io.py           JSONL read/write with line-accurate errors
└── backends/
    ├── base.py         the TrainingBackend Protocol
    ├── _process.py     subprocess execution and metric scraping
    ├── mlx_backend.py  drives `python -m mlx_lm lora`
    ├── _mlx_args.py    its argv/config construction; imports no mlx
    ├── torch_backend.py  drives trl SFTTrainer + peft
    └── _torch_args.py  its PEFT and TRL argument construction
configs/                one worked example per method
docs/                   LoRA background and configuration guidance
scripts/verify_repo.sh  convention checks; exits non-zero on failure
tests/
```

The `_*_args.py` split exists so argument construction is testable without the
backend's extra installed, and so every file stays under the 200-line ceiling.

## Design rules

**The capability matrix is the support claim.** `capabilities.py` declares which
method/backend pairs exist. Anything absent raises
`UnsupportedCombinationError` naming the backends that do support the method. Do
not add an enum member without an implementation, and do not let the README claim
something the matrix does not.

**The core package stays import-light.** Only PyYAML. Training dependencies live
behind the `torch` and `apple` extras and are imported inside functions, never at
module scope. CI asserts that `import finetune` pulls in neither torch nor mlx.

**Backends are three stages.** `preflight` validates, `plan` describes without
executing, `train` runs. Keeping `plan` pure is what makes `finetune plan` honest
and lets the suite assert on a constructed command instead of mocking a
subprocess.

**Never name a module after a package it imports.** `mlx_backend.py`, not
`mlx.py`. A top-level `peft/` directory previously shadowed the real `peft`
dependency and made this repository's headline code path unimportable;
`verify_repo.sh` now checks for recurrences.

**Config that cannot be honoured is an error.** Unknown YAML keys fail the load
with a suggestion. Silently ignoring a key means reporting success while doing
something else.

**Code only.** No weights, adapters, checkpoints or training logs in git. Those
belong in releases or on the Hub.

## Backend constraints

- `mlx`/`mlx-lm` have no Linux wheels, so the MLX backend cannot run in CI. Its
  command construction is unit-tested without mlx installed.
- `bitsandbytes` is CUDA-only, so PyTorch QLoRA does not run on Apple Silicon.
  The matrix records this.
- `mlx_lm` takes rank, dropout and scale only through a YAML config file, not CLI
  flags, so `mlx_backend` writes one per run. It doubles as run provenance.
