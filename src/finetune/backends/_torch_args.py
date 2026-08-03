"""Argument construction for the PyTorch backend.

Split from :mod:`finetune.backends.torch_backend` so the PEFT and TRL argument
builders are testable in isolation and each file stays under the line ceiling.

:func:`peft_kwargs` and :func:`describe_stages` import nothing heavy, so they run
without the ``torch`` extra installed. :func:`sft_kwargs` needs TRL only to read
the installed ``SFTConfig`` field names.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
from typing import Any

from ..capabilities import Method
from ..run import RunConfig

logger = logging.getLogger(__name__)


def importable(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def peft_kwargs(config: RunConfig) -> dict[str, Any]:
    """Build the ``LoraConfig`` kwargs for this run.

    ``target_modules`` passes through as given, which for the default config is
    the string ``"all-linear"``. PEFT expands that to every linear layer,
    including the MLP projections that attention-only targeting misses.
    """
    modules = config.adapter.target_modules
    return {
        "r": config.adapter.rank,
        "lora_alpha": config.adapter.alpha,
        "lora_dropout": config.adapter.dropout,
        "target_modules": modules if isinstance(modules, str) else list(modules),
        "use_dora": config.method is Method.DORA,
        "use_rslora": config.adapter.use_rslora,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def sft_config_cls() -> type:
    """Return the installed ``trl.SFTConfig``."""
    from trl import SFTConfig

    return SFTConfig


def precision_kwargs() -> dict[str, Any]:
    """Resolve mixed precision and device placement from actual hardware.

    This cannot be left to the defaults. TRL's ``SFTConfig`` sets ``bf16=True``,
    and transformers then raises "Your setup doesn't support bf16/gpu" on any
    machine without a bf16-capable accelerator -- so a CPU-only box fails before
    training starts. Apple Silicon hides the problem, because MPS satisfies the
    check while CPU-only Linux does not.

    bf16 is requested only where it is actually supported. Everywhere else the
    run is fp32, which is slower but correct, and ``use_cpu`` is set explicitly
    when there is no accelerator at all.
    """
    import torch

    if torch.cuda.is_available():
        from transformers.utils import is_torch_bf16_gpu_available

        bf16 = bool(is_torch_bf16_gpu_available())
        # Pre-Ampere CUDA has no bf16; fp16 is the available alternative there.
        return {"bf16": bf16, "fp16": not bf16}
    if torch.backends.mps.is_available():
        return {"bf16": False, "fp16": False}
    return {"use_cpu": True, "bf16": False, "fp16": False}


def sft_kwargs(config: RunConfig) -> dict[str, Any]:
    """Build ``SFTConfig`` kwargs, filtered to fields the installed TRL accepts.

    TRL renames and adds arguments between releases. Filtering against the
    installed dataclass keeps a wide version floor working, and logs what was
    dropped rather than silently changing behaviour.

    Notably absent: ``use_mps_device``. It was removed in transformers 5, where
    passing it raises ``TypeError``, and it is unnecessary because accelerate
    selects MPS on Apple Silicon by itself.
    """
    desired: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "max_length": config.training.max_seq_length,
        "seed": config.training.seed,
        "logging_steps": config.training.steps_per_report,
        "save_steps": config.training.save_every,
        "gradient_checkpointing": config.training.grad_checkpoint,
        # Train on the assistant turn only. Without this the model is also
        # optimised to reproduce the user's prompt.
        "completion_only_loss": config.training.mask_prompt,
        "report_to": [],
        "save_strategy": "steps",
    }
    if config.training.iters is not None:
        desired["max_steps"] = config.training.iters
    if config.training.epochs is not None:
        desired["num_train_epochs"] = config.training.epochs
    desired.update(precision_kwargs())
    return filter_to_dataclass(desired, sft_config_cls())


def quantization_kwargs(config: RunConfig) -> dict[str, Any]:
    """Build ``model_init_kwargs`` carrying a 4-bit base for QLoRA."""
    if config.method is not Method.QLORA or config.quantization is None:
        return {}
    import torch
    from transformers import BitsAndBytesConfig

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=config.quantization.bits == 4,
            load_in_8bit=config.quantization.bits == 8,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    }


def filter_to_dataclass(desired: dict[str, Any], cls: type) -> dict[str, Any]:
    """Drop keys the installed dataclass does not declare, logging each one."""
    if not dataclasses.is_dataclass(cls):  # pragma: no cover - TRL configs are dataclasses
        return desired
    valid = {f.name for f in dataclasses.fields(cls)}
    kept = {k: v for k, v in desired.items() if k in valid}
    for dropped in sorted(set(desired) - valid):
        logger.warning("%s does not accept %r in this version; skipping it", cls.__name__, dropped)
    return kept


def describe_stages(config: RunConfig) -> tuple[str, ...]:
    """Human-readable stages for the plan output."""
    stages: list[str] = []
    quantization = config.quantization
    if config.method is Method.QLORA and quantization is not None:
        stages.append(f"load base in {quantization.bits}-bit via bitsandbytes")
    if config.uses_adapter:
        modules = config.adapter.target_modules
        described = modules if isinstance(modules, str) else ", ".join(modules)
        kind = "DoRA" if config.method is Method.DORA else "LoRA"
        stages.append(
            f"attach rank-{config.adapter.rank} {kind} adapters to {described}"
            + (" with rsLoRA scaling" if config.adapter.use_rslora else "")
        )
    budget = (
        f"{config.training.iters} steps"
        if config.training.iters is not None
        else f"{config.training.epochs} epochs"
    )
    stages.append(
        f"train with trl SFTTrainer for {budget} at lr {config.learning_rate:g}, "
        f"effective batch {config.training.effective_batch_size}, "
        f"{'completion-only' if config.training.mask_prompt else 'full-sequence'} loss"
    )
    return tuple(stages)


__all__ = [
    "describe_stages",
    "filter_to_dataclass",
    "importable",
    "peft_kwargs",
    "quantization_kwargs",
    "sft_config_cls",
    "sft_kwargs",
]
