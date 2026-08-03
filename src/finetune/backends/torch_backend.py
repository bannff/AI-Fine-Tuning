"""Portable backend, driving TRL's ``SFTTrainer`` with PEFT adapters.

Named ``torch_backend`` rather than ``torch`` for the same reason the MLX module
is not named ``mlx``: a local module that shadows its own dependency is the bug
this package was restructured to eliminate.

Why TRL rather than a hand-rolled ``Trainer`` loop: prompt masking, chat-template
application and packing already live in ``SFTTrainer`` and are easy to get subtly
wrong by hand. The loop this replaces tokenised to ``padding="max_length"`` and
then set ``labels = input_ids`` with no ``-100`` masking, so it spent most of its
gradient budget learning to predict ``<pad>``.

Argument construction lives in :mod:`._torch_args`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ..capabilities import Backend, Method, is_installed, resolve
from ..data.io import read_jsonl
from ..data.layout import validate_split_dir
from ..errors import BackendUnavailableError, TrainingFailedError
from ..run import RunConfig
from . import _torch_args as torch_args
from .base import Plan, TrainingResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TorchBackend:
    """Fine-tunes in-process with ``trl.SFTTrainer`` and ``peft``."""

    backend: ClassVar[Backend] = Backend.TORCH

    def preflight(self, config: RunConfig) -> None:
        """Validate the capability pair, dependency presence and dataset."""
        resolve(config.method, self.backend)
        if not is_installed(self.backend):
            raise BackendUnavailableError(backend=str(self.backend), extra="torch", missing="trl")
        if config.method is Method.QLORA and not torch_args.importable("bitsandbytes"):
            # bitsandbytes is CUDA-only, so this is the expected failure on
            # Apple Silicon. The matrix records that the mlx backend is the
            # route to quantised training there.
            raise BackendUnavailableError(
                backend=str(self.backend), extra="torch", missing="bitsandbytes"
            )
        validate_split_dir(config.data_dir)

    def plan(self, config: RunConfig) -> Plan:
        """Describe the run without loading a model."""
        return Plan(
            backend=self.backend,
            method=config.method,
            summary=f"trl SFTTrainer on {config.model} -> {config.output_dir}",
            stages=torch_args.describe_stages(config),
        )

    def load_splits(self, config: RunConfig) -> tuple[Dataset, Dataset]:
        """Load train and validation splits as HF datasets."""
        from datasets import Dataset

        train = Dataset.from_list([dict(r) for r in read_jsonl(config.data_dir / "train.jsonl")])
        valid = Dataset.from_list([dict(r) for r in read_jsonl(config.data_dir / "valid.jsonl")])
        return train, valid

    def train(self, config: RunConfig) -> TrainingResult:
        from peft import LoraConfig
        from trl import SFTTrainer

        config.output_dir.mkdir(parents=True, exist_ok=True)
        train_ds, eval_ds = self.load_splits(config)

        args = torch_args.sft_config_cls()(**torch_args.sft_kwargs(config))
        model_kwargs = torch_args.quantization_kwargs(config)
        if model_kwargs and hasattr(args, "model_init_kwargs"):
            args.model_init_kwargs = model_kwargs

        peft_config = LoraConfig(**torch_args.peft_kwargs(config)) if config.uses_adapter else None

        trainer = SFTTrainer(
            model=config.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_config,
        )
        try:
            output = trainer.train()
        except Exception as exc:
            raise TrainingFailedError(f"trl SFTTrainer failed: {exc}") from exc

        trainer.save_model(str(config.output_dir))
        metrics = dict(getattr(output, "metrics", {}) or {})
        return TrainingResult(
            backend=self.backend,
            method=config.method,
            output_dir=config.output_dir,
            succeeded=True,
            steps=int(getattr(output, "global_step", 0)) or None,
            final_loss=metrics.get("train_loss"),
            metrics={k: float(v) for k, v in metrics.items() if isinstance(v, int | float)},
        )


__all__ = ["TorchBackend"]
