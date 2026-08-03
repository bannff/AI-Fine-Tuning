# finetune

[![CI](https://github.com/bannff/AI-Fine-Tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/bannff/AI-Fine-Tuning/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fine-tune causal language models with LoRA, DoRA, QLoRA or full-parameter
updates, across two backends, from one config file.

- **MLX** for Apple Silicon, driving the `mlx_lm` trainer.
- **PyTorch** for everywhere else, driving TRL's `SFTTrainer` with PEFT adapters.

The support table below is generated from the same code the runtime enforces, so
a combination that appears here works and one that does not is rejected with an
error naming the backend that does support it.

## Support matrix

| Method  | MLX | PyTorch | Notes |
|---------|-----|---------|-------|
| `lora`  | yes | yes     | Low-rank adapters on frozen full-precision weights. |
| `dora`  | yes | yes     | Weight-decomposed LoRA. `--fine-tune-type dora` on MLX, `use_dora=True` on PEFT. |
| `qlora` | yes | yes\*   | Quantised frozen base. On MLX this is a real `mlx_lm convert -q` stage before training. |
| `full`  | yes | no      | Every parameter updated. Not implemented on the PyTorch backend. |

\* PyTorch QLoRA needs `bitsandbytes`, which is CUDA-only. On Apple Silicon use
the MLX backend for quantised training.

Run `finetune capabilities` to see this table plus which backends are installed
locally.

### Deliberately not included

- **Preference optimisation** (DPO, ORPO, KTO). Available upstream through TRL
  on the PyTorch backend. `mlx_lm` core has no preference objective; third-party
  [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) and
  [SiLLM](https://github.com/armbues/SiLLM) add one.
- **Online RL** (GRPO, RLOO). Needs a reward function and many generations per
  step, which realistically wants vLLM. Out of scope for a single machine.

These are absent rather than stubbed. `finetune capabilities` prints the same
list, so the omission is visible from the tool and not just the docs.

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/bannff/AI-Fine-Tuning.git
cd AI-Fine-Tuning

uv sync --extra apple      # Apple Silicon: mlx + mlx-lm
uv sync --extra torch      # portable: torch + transformers + trl + peft
uv sync --extra all        # both
```

The core package depends on PyYAML alone. Dataset validation, splitting, config
resolution and the capability matrix all work with no ML dependencies installed,
and `import finetune` does not pull in torch or mlx. CI asserts that.

## Use

```bash
# 1. Split a JSONL dataset into train/valid/test.
finetune prepare-data mydata.jsonl data/mydata

# 2. Check it against the schemas both trainers accept.
finetune validate-data data/mydata

# 3. See exactly what will run, without running it.
finetune plan configs/tinyllama-lora-mlx.yaml

# 4. Run it.
finetune train configs/tinyllama-lora-mlx.yaml
```

`plan` prints the resolved configuration including defaults you did not set, and
the exact `mlx_lm` command it will invoke. It never creates or writes anything.

### Dataset schemas

`mlx_lm` and TRL both accept exactly three record shapes. That intersection is
what this package validates, one shape per file:

```jsonc
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "completion": "..."}
{"text": "..."}
```

Validation happens before training starts and reports the file and line number of
the first bad record, rather than failing after the model has loaded.

### Configuration

One YAML file per run. See [`configs/`](configs/) for a worked example of each
method.

```yaml
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
method: lora
backend: mlx

data_dir: ../data/tinyllama
output_dir: ../outputs/tinyllama-lora

adapter:
  rank: 16
  alpha: 32
  target_modules: all-linear

training:
  batch_size: 4
  gradient_accumulation_steps: 2
  iters: 600
  mask_prompt: true
```

Two properties worth knowing:

**Unknown keys are errors.** A misspelled or unsupported key fails the load with
a suggestion, rather than being silently ignored. A config that reports success
while doing something else is worse than no config at all.

**Relative paths resolve against the config file**, not the working directory, so
a committed config works from anywhere in the tree.

## Defaults

Defaults follow current published practice rather than convention inherited from
2023. The substantive ones:

`target_modules: all-linear` — every linear projection, MLP included. Restricting
LoRA to attention projections underperforms, and increasing the rank does not
compensate for it. Configure something narrower and `finetune plan` will warn you.

`alpha` is independent of `rank`. `alpha = 2 * rank` is folklore; holding alpha
fixed means changing rank does not require re-tuning the learning rate.

`mask_prompt: true` — loss on the completion only. Training on the prompt as well
optimises the model to reproduce the user's own turn.

Learning rate defaults by method: `2e-4` for the low-rank methods, `2e-6` for
`full`. Low-rank updates want roughly two orders of magnitude more than full
fine-tuning. Omit `learning_rate` to accept the default; `plan` shows the
resolved value.

Effective batch size above 32 raises an advisory. Low-rank training tolerates
large batches less well than full fine-tuning, and raising the rank does not fix
that either.

Background and sources: [docs/lora-practice.md](docs/lora-practice.md).

## Layout

```
src/finetune/
  capabilities.py     the support matrix, enforced at runtime
  config.py           the config sections, with field validation
  run.py              RunConfig: cross-section coherence, resolved defaults
  config_io.py        strict YAML loading
  errors.py           the exception hierarchy
  data/
    schema.py         the three record schemas
    layout.py         train/valid/test directory validation
    split.py          deterministic, exact allocation
    io.py             JSONL with line-accurate errors
  backends/
    base.py           the Protocol: preflight / plan / train
    _process.py       subprocess execution and metric scraping
    mlx_backend.py    drives `python -m mlx_lm lora`
    _mlx_args.py      its command and config construction (imports no mlx)
    torch_backend.py  drives trl SFTTrainer + peft
    _torch_args.py    its PEFT and TRL argument construction
  inference.py        generate from a trained adapter (MLX)
  cli.py              parsing and dispatch
  commands.py         command implementations
configs/              one worked example per method
docs/                 LoRA background and practice notes
tests/
```

Every file in `src/` is under 200 lines, checked by `scripts/verify_repo.sh`.

Backends are split into `preflight` / `plan` / `train` so that everything except
execution is testable without installing torch or mlx. That is what makes
`plan` honest and lets the suite assert on a constructed `mlx_lm` command rather
than mocking a subprocess.

Neither backend module is named after the library it imports. A local module
that shadows its own dependency is a real failure mode, and this repository
previously shipped one.

## Development

```bash
uv sync --extra all
uv run pytest              # everything, including training smoke tests
uv run pytest -m "not slow" # unit tests only, no model construction
uv run ruff check . && uv run ruff format --check .
uv run mypy                # strict
bash scripts/verify_repo.sh
```

The training smoke tests build a small Llama and a byte-level BPE tokenizer on
disk, so they exercise the real code path in a couple of seconds with no network
access and no model download. They are the reason CI can prove the trainers
actually run: the previous suite passed only because the real dependencies were
absent from the environment it created.

`scripts/verify_repo.sh` checks conventions that are easy to regress: no
top-level directory shadowing a dependency, no tracked file over 512 KB, no empty
tracked files, and a per-file line ceiling. CI runs it twice, once against this
repository and once against an empty one, to confirm it can still fail.

This repository ships code only. Model weights, adapters and training logs belong
in releases or on the Hub.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: run the five gate commands
above before opening a PR.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
