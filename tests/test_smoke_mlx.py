"""MLX backend smoke tests: a real ``mlx_lm`` training run.

Skipped without the ``apple`` extra, and therefore skipped in CI, because mlx has
no Linux wheels. That is exactly why the command construction in
``_mlx_args`` is unit-tested separately and platform-independently -- but running
the real trainer at least once locally is what proves the two agree.

Uses the same locally-built tiny model as the torch smoke tests, so no network
access and no model download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finetune.backends import _mlx_args as mlx_args
from finetune.backends import get_backend
from finetune.capabilities import Backend, Method
from finetune.config import AdapterConfig, TrainingConfig
from finetune.run import RunConfig
from tests.conftest import HAS_MLX

requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="needs the 'apple' extra")

# 256 fits the tiny model's 512-token context and, importantly, leaves the
# assistant turn intact. Truncating below the completion length while
# mask_prompt is on leaves zero tokens in the loss and yields NaN.
SMOKE_TRAINING = TrainingConfig(
    iters=4,
    batch_size=2,
    max_seq_length=256,
    steps_per_report=1,
    steps_per_eval=4,
    save_every=4,
    val_batches=2,
)


def run_config(model: Path, data_dir: Path, out: Path, method: Method = Method.LORA) -> RunConfig:
    return RunConfig(
        model=str(model),
        data_dir=data_dir,
        output_dir=out,
        method=method,
        backend=Backend.MLX,
        adapter=AdapterConfig(rank=8, alpha=16),
        training=SMOKE_TRAINING,
    )


@pytest.mark.slow
@requires_mlx
def test_mlx_lora_trains_and_saves_adapters(
    tiny_model: Path, chat_dataset: Path, tmp_path: Path
) -> None:
    backend = get_backend(Backend.MLX)
    config = run_config(tiny_model, chat_dataset, tmp_path / "out")

    backend.preflight(config)
    result = backend.train(config)

    assert result.succeeded
    assert (result.output_dir / "adapters.safetensors").is_file()


@pytest.mark.slow
@requires_mlx
def test_metrics_are_scraped_from_trainer_output(
    tiny_model: Path, chat_dataset: Path, tmp_path: Path
) -> None:
    """Step count and losses come from parsing mlx_lm's stdout, so parse them."""
    import math

    backend = get_backend(Backend.MLX)
    result = backend.train(run_config(tiny_model, chat_dataset, tmp_path / "out"))

    assert result.steps == 4
    assert result.final_loss is not None
    assert math.isfinite(result.final_loss)
    assert "val_loss" in result.metrics, "validation loss should be captured too"


@pytest.mark.slow
@requires_mlx
def test_resolved_config_is_written_for_provenance(
    tiny_model: Path, chat_dataset: Path, tmp_path: Path
) -> None:
    """Every run records the settings it actually used, defaults included."""
    import yaml

    backend = get_backend(Backend.MLX)
    config = run_config(tiny_model, chat_dataset, tmp_path / "out")
    backend.train(config)

    written = yaml.safe_load(
        (config.output_dir / mlx_args.CONFIG_FILENAME).read_text(encoding="utf-8")
    )

    assert written["fine_tune_type"] == "lora"
    assert written["mask_prompt"] is True
    assert written["lora_parameters"] == {"rank": 8, "dropout": 0.0, "scale": 2.0}
    assert written["learning_rate"] == pytest.approx(2e-4)


@pytest.mark.slow
@requires_mlx
def test_dora_runs_through_the_same_path(
    tiny_model: Path, chat_dataset: Path, tmp_path: Path
) -> None:
    """DoRA is a real mlx_lm fine-tune type, not an alias for LoRA."""
    import yaml

    backend = get_backend(Backend.MLX)
    config = run_config(tiny_model, chat_dataset, tmp_path / "out", method=Method.DORA)
    result = backend.train(config)

    assert result.succeeded
    written = yaml.safe_load(
        (config.output_dir / mlx_args.CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    assert written["fine_tune_type"] == "dora"


@pytest.mark.slow
@requires_mlx
def test_full_finetuning_writes_weights_not_adapters(
    tiny_model: Path, chat_dataset: Path, tmp_path: Path
) -> None:
    backend = get_backend(Backend.MLX)
    config = run_config(tiny_model, chat_dataset, tmp_path / "out", method=Method.FULL)

    result = backend.train(config)

    assert result.succeeded
    assert result.output_dir.name == "weights"
