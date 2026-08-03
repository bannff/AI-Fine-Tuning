"""Command and config construction for the MLX backend.

Pure functions, split from :mod:`finetune.backends.mlx_backend` so the argv and
trainer-config builders stay independently testable and each file stays under the
line ceiling. Nothing here imports mlx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..capabilities import Method
from ..errors import TrainingFailedError
from ..run import RunConfig
from ._process import module_argv

#: Directory inside output_dir holding the quantised base for a QLoRA run.
QUANTIZED_SUBDIR = "quantized-base"

#: The resolved mlx_lm config written for the run. Doubles as provenance.
CONFIG_FILENAME = "mlx-lora-config.yaml"

#: mlx_lm's own name for each of our methods. QLoRA is LoRA over a base that has
#: already been quantised by a separate convert step.
FINE_TUNE_TYPE: dict[Method, str] = {
    Method.LORA: "lora",
    Method.DORA: "dora",
    Method.FULL: "full",
    Method.QLORA: "lora",
}


def adapter_dir(config: RunConfig) -> Path:
    """Where mlx_lm writes adapters, or full weights for method ``full``."""
    return config.output_dir / ("weights" if config.method is Method.FULL else "adapters")


def base_model(config: RunConfig) -> str:
    """The model the trainer sees: the quantised copy for QLoRA, else as given."""
    if config.method is Method.QLORA:
        return str(config.output_dir / QUANTIZED_SUBDIR)
    return config.model


def resolve_iters(config: RunConfig) -> int:
    """Resolve the iteration count.

    mlx_lm counts iterations only, so an epoch-based config has to be converted
    against the actual training set size.
    """
    if config.training.iters is not None:
        return config.training.iters
    train_file = config.data_dir / "train.jsonl"
    rows = sum(1 for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip())
    per_epoch = max(1, rows // config.training.effective_batch_size)
    epochs = config.training.epochs
    if epochs is None:  # pragma: no cover - RunConfig guarantees one is set
        raise TrainingFailedError("neither training.iters nor training.epochs is set")
    return max(1, per_epoch * epochs)


def quantize_argv(config: RunConfig) -> tuple[str, ...]:
    """Build the ``mlx_lm convert`` argv that produces the quantised base.

    In MLX, quantisation is a distinct step rather than a trainer flag. Passing no
    quantisation option to ``mlx_lm lora`` yields plain LoRA on a full-precision
    base, which is what the script this replaces actually did.
    """
    if config.quantization is None:  # pragma: no cover - guarded by RunConfig
        raise TrainingFailedError("method 'qlora' requires a quantization block")
    return (
        *module_argv("mlx_lm", "convert"),
        "--hf-path",
        config.model,
        "--mlx-path",
        str(config.output_dir / QUANTIZED_SUBDIR),
        "--quantize",
        "--q-bits",
        str(config.quantization.bits),
        "--q-group-size",
        str(config.quantization.group_size),
    )


def train_argv(config_path: Path) -> tuple[str, ...]:
    """Build the ``mlx_lm lora`` argv. Every setting comes from the config file."""
    return (*module_argv("mlx_lm", "lora"), "--config", str(config_path))


def trainer_config(config: RunConfig) -> dict[str, Any]:
    """Build the mlx_lm YAML config for this run.

    Rank, dropout and scale are only settable through the config file; ``mlx_lm
    lora`` has no CLI flag for them. ``scale`` is the multiplier applied to the
    low-rank update, which is what the alpha/rank ratio (or alpha/sqrt(rank)
    under rsLoRA) resolves to.
    """
    payload: dict[str, Any] = {
        "model": base_model(config),
        "train": True,
        "data": str(config.data_dir),
        "adapter_path": str(adapter_dir(config)),
        "fine_tune_type": FINE_TUNE_TYPE[config.method],
        "optimizer": "adamw",
        "batch_size": config.training.batch_size,
        "grad_accumulation_steps": config.training.gradient_accumulation_steps,
        "iters": resolve_iters(config),
        "learning_rate": config.learning_rate,
        "max_seq_length": config.training.max_seq_length,
        "seed": config.training.seed,
        "steps_per_report": config.training.steps_per_report,
        "steps_per_eval": config.training.steps_per_eval,
        "save_every": config.training.save_every,
        "val_batches": config.training.val_batches,
        "mask_prompt": config.training.mask_prompt,
        "grad_checkpoint": config.training.grad_checkpoint,
    }
    if config.uses_adapter:
        payload["num_layers"] = config.adapter.num_layers
        payload["lora_parameters"] = {
            "rank": config.adapter.rank,
            "dropout": config.adapter.dropout,
            "scale": config.adapter.scale,
        }
    else:
        # mlx_lm applies num_layers to full fine-tuning too, and its default is
        # 16. Omitting it would train only the last 16 layers and call that
        # "full", and would hard-error on any model with fewer than 16.
        payload["num_layers"] = -1
    return payload


def describe_stages(config: RunConfig) -> tuple[str, ...]:
    """Human-readable stages for the plan output."""
    stages: list[str] = []
    quantization = config.quantization
    if config.method is Method.QLORA and quantization is not None:
        stages.append(
            f"quantise {config.model} to {quantization.bits}-bit "
            f"(group size {quantization.group_size}) via mlx_lm convert"
        )
    if config.method is Method.FULL:
        target = "all weights"
    else:
        layers = (
            "all layers"
            if config.adapter.num_layers == -1
            else f"last {config.adapter.num_layers} layers"
        )
        target = f"rank-{config.adapter.rank} adapters, scale {config.adapter.scale:.3f}, {layers}"
    stages.append(
        f"train {target} for {resolve_iters(config)} iters at lr {config.learning_rate:g}"
    )
    return tuple(stages)


__all__ = [
    "CONFIG_FILENAME",
    "FINE_TUNE_TYPE",
    "QUANTIZED_SUBDIR",
    "adapter_dir",
    "base_model",
    "describe_stages",
    "quantize_argv",
    "resolve_iters",
    "train_argv",
    "trainer_config",
]
