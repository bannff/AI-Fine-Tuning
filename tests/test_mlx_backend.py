"""MLX backend tests.

Regression tests for the bugs the previous scripts shipped. They assert on the
constructed command and trainer config rather than mocking a subprocess, which
is why ``plan``/``trainer_config``/``quantize_argv`` are separate from ``train``.

None of these need mlx installed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from finetune.backends import _mlx_args as mlx_args
from finetune.backends._mlx_args import QUANTIZED_SUBDIR
from finetune.backends.mlx_backend import MlxBackend
from finetune.capabilities import Backend, Method
from finetune.config import AdapterConfig, QuantizationConfig, TrainingConfig
from finetune.run import RunConfig
from tests.conftest import HAS_MLX


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    for name in ("train.jsonl", "valid.jsonl"):
        rows = "\n".join(f'{{"text": "row {i}"}}' for i in range(20))
        (directory / name).write_text(rows + "\n", encoding="utf-8")
    return directory


def build(
    data_dir: Path,
    tmp_path: Path,
    method: Method = Method.LORA,
    *,
    adapter: AdapterConfig | None = None,
    training: TrainingConfig | None = None,
) -> RunConfig:
    return RunConfig(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_dir=data_dir,
        output_dir=tmp_path / "out",
        method=method,
        backend=Backend.MLX,
        quantization=QuantizationConfig() if method is Method.QLORA else None,
        adapter=adapter or AdapterConfig(),
        training=training or TrainingConfig(iters=100),
    )


backend = MlxBackend()


# ------------------------------------------------------------- fine-tune type


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (Method.LORA, "lora"),
        (Method.DORA, "dora"),
        (Method.FULL, "full"),
        (Method.QLORA, "lora"),
    ],
)
def test_fine_tune_type_is_always_set(
    data_dir: Path, tmp_path: Path, method: Method, expected: str
) -> None:
    """The old QLoRA script never passed --fine-tune-type, so it silently ran LoRA."""
    config = build(data_dir, tmp_path, method)
    assert mlx_args.trainer_config(config)["fine_tune_type"] == expected


def test_dora_and_lora_produce_different_commands(data_dir: Path, tmp_path: Path) -> None:
    lora = mlx_args.trainer_config(build(data_dir, tmp_path, Method.LORA))
    dora = mlx_args.trainer_config(build(data_dir, tmp_path, Method.DORA))
    assert lora["fine_tune_type"] != dora["fine_tune_type"]


# ------------------------------------------------------------------ quantising


def test_qlora_quantises_before_training(data_dir: Path, tmp_path: Path) -> None:
    """QLoRA must actually quantise; in MLX that is a convert step, not a flag."""
    config = build(data_dir, tmp_path, Method.QLORA)
    argv = mlx_args.quantize_argv(config)

    assert argv[:4] == (sys.executable, "-m", "mlx_lm", "convert")
    assert "--quantize" in argv
    assert "--q-bits" in argv
    assert argv[argv.index("--q-bits") + 1] == "4"
    assert argv[argv.index("--q-group-size") + 1] == "64"


def test_qlora_trains_against_the_quantised_copy(data_dir: Path, tmp_path: Path) -> None:
    config = build(data_dir, tmp_path, Method.QLORA)
    assert mlx_args.base_model(config).endswith(QUANTIZED_SUBDIR)


def test_non_qlora_trains_against_the_original_model(data_dir: Path, tmp_path: Path) -> None:
    config = build(data_dir, tmp_path, Method.LORA)
    assert mlx_args.base_model(config) == config.model


def test_qlora_plan_shows_both_stages(data_dir: Path, tmp_path: Path) -> None:
    plan = backend.plan(build(data_dir, tmp_path, Method.QLORA))
    assert len(plan.stages) == 2
    assert "quantise" in plan.stages[0]
    assert "4-bit" in plan.stages[0]


def test_lora_plan_has_a_single_stage(data_dir: Path, tmp_path: Path) -> None:
    plan = backend.plan(build(data_dir, tmp_path, Method.LORA))
    assert len(plan.stages) == 1


def test_quantization_bits_reach_the_command(data_dir: Path, tmp_path: Path) -> None:
    config = build(data_dir, tmp_path, Method.QLORA)
    config = config.with_overrides(quantization=QuantizationConfig(bits=8, group_size=32))
    argv = mlx_args.quantize_argv(config)
    assert argv[argv.index("--q-bits") + 1] == "8"
    assert argv[argv.index("--q-group-size") + 1] == "32"


# ---------------------------------------------------------------- invocation


def test_trainer_is_invoked_through_the_running_interpreter(data_dir: Path, tmp_path: Path) -> None:
    """`sys.executable -m` cannot resolve to a different environment; a bare
    console script on PATH can."""
    argv = mlx_args.train_argv(tmp_path / "cfg.yaml")

    assert argv[0] == sys.executable
    assert argv[1:4] == ("-m", "mlx_lm", "lora")


def test_train_argv_points_at_the_written_config(data_dir: Path, tmp_path: Path) -> None:
    config_path = tmp_path / "mlx.yaml"
    argv = mlx_args.train_argv(config_path)
    assert argv[-2:] == ("--config", str(config_path))


# ------------------------------------------------------------ adapter geometry


def test_adapter_settings_reach_the_trainer_config(data_dir: Path, tmp_path: Path) -> None:
    """rank/dropout/scale are only settable via the config file; mlx_lm has no flags."""
    config = build(
        data_dir, tmp_path, adapter=AdapterConfig(rank=32, alpha=64, dropout=0.05, num_layers=8)
    )
    payload = mlx_args.trainer_config(config)

    assert payload["lora_parameters"] == {"rank": 32, "dropout": 0.05, "scale": 2.0}
    assert payload["num_layers"] == 8


def test_rslora_changes_the_scale_that_mlx_receives(data_dir: Path, tmp_path: Path) -> None:
    plain = mlx_args.trainer_config(
        build(data_dir, tmp_path, adapter=AdapterConfig(rank=64, alpha=32))
    )
    stabilised = mlx_args.trainer_config(
        build(data_dir, tmp_path, adapter=AdapterConfig(rank=64, alpha=32, use_rslora=True))
    )
    assert stabilised["lora_parameters"]["scale"] > plain["lora_parameters"]["scale"]


def test_full_finetuning_sends_no_adapter_settings(data_dir: Path, tmp_path: Path) -> None:
    payload = mlx_args.trainer_config(build(data_dir, tmp_path, Method.FULL))
    assert "lora_parameters" not in payload


def test_full_finetuning_trains_every_layer(data_dir: Path, tmp_path: Path) -> None:
    """`full` must mean full.

    mlx_lm applies num_layers to full fine-tuning as well, defaulting to 16.
    Omitting it would train only the last 16 layers and still call the run
    "full", and would hard-error on a model with fewer layers than that.
    """
    payload = mlx_args.trainer_config(build(data_dir, tmp_path, Method.FULL))
    assert payload["num_layers"] == -1


def test_full_finetuning_writes_weights_not_adapters(data_dir: Path, tmp_path: Path) -> None:
    assert mlx_args.adapter_dir(build(data_dir, tmp_path, Method.FULL)).name == "weights"
    assert mlx_args.adapter_dir(build(data_dir, tmp_path, Method.LORA)).name == "adapters"


# ------------------------------------------------------------- training config


def test_prompt_masking_is_plumbed_through(data_dir: Path, tmp_path: Path) -> None:
    """Without mask_prompt the model is also trained to predict the user turn."""
    on = mlx_args.trainer_config(
        build(data_dir, tmp_path, training=TrainingConfig(iters=5, mask_prompt=True))
    )
    off = mlx_args.trainer_config(
        build(data_dir, tmp_path, training=TrainingConfig(iters=5, mask_prompt=False))
    )
    assert on["mask_prompt"] is True
    assert off["mask_prompt"] is False


def test_gradient_accumulation_is_plumbed_through(data_dir: Path, tmp_path: Path) -> None:
    payload = mlx_args.trainer_config(
        build(data_dir, tmp_path, training=TrainingConfig(iters=5, gradient_accumulation_steps=4))
    )
    assert payload["grad_accumulation_steps"] == 4


def test_method_learning_rate_default_reaches_the_trainer(data_dir: Path, tmp_path: Path) -> None:
    lora = mlx_args.trainer_config(build(data_dir, tmp_path, Method.LORA))
    full = mlx_args.trainer_config(build(data_dir, tmp_path, Method.FULL))
    assert lora["learning_rate"] == pytest.approx(2e-4)
    assert full["learning_rate"] == pytest.approx(2e-6)


def test_epochs_are_converted_to_iterations(data_dir: Path, tmp_path: Path) -> None:
    """mlx_lm counts iterations only, so epochs must be resolved from the dataset."""
    config = build(data_dir, tmp_path, training=TrainingConfig(epochs=2, batch_size=5))
    # 20 train rows / effective batch 5 = 4 steps per epoch, times 2 epochs.
    assert mlx_args.trainer_config(config)["iters"] == 8


def test_explicit_iters_are_passed_verbatim(data_dir: Path, tmp_path: Path) -> None:
    config = build(data_dir, tmp_path, training=TrainingConfig(iters=777))
    assert mlx_args.trainer_config(config)["iters"] == 777


def test_trainer_config_uses_only_keys_mlx_lm_declares(data_dir: Path, tmp_path: Path) -> None:
    """Guards against inventing config keys mlx_lm would silently ignore."""
    known = {
        "model",
        "train",
        "data",
        "adapter_path",
        "fine_tune_type",
        "optimizer",
        "batch_size",
        "grad_accumulation_steps",
        "iters",
        "learning_rate",
        "max_seq_length",
        "seed",
        "steps_per_report",
        "steps_per_eval",
        "save_every",
        "val_batches",
        "mask_prompt",
        "grad_checkpoint",
        "num_layers",
        "lora_parameters",
    }
    for method in Method:
        payload = mlx_args.trainer_config(build(data_dir, tmp_path, method))
        assert set(payload) <= known, f"{method}: unknown keys {set(payload) - known}"


# ------------------------------------------------------------------ preflight


@pytest.mark.skipif(not HAS_MLX, reason="dataset check is only reached once mlx is installed")
def test_preflight_rejects_a_dataset_without_a_validation_split(tmp_path: Path) -> None:
    from finetune.errors import DatasetSchemaError

    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"text": "only train"}\n', encoding="utf-8")

    with pytest.raises(DatasetSchemaError, match=re.escape("required split 'valid.jsonl' missing")):
        backend.preflight(build(data, tmp_path))


@pytest.mark.skipif(HAS_MLX, reason="only meaningful when the apple extra is absent")
def test_preflight_reports_the_missing_extra_before_checking_data(tmp_path: Path) -> None:
    """Dependency check precedes the dataset check, and says how to fix it.

    Validating a dataset for a backend that cannot run would be wasted work and a
    confusing error, so the order is deliberate.
    """
    from finetune.errors import BackendUnavailableError

    data = tmp_path / "data"
    data.mkdir()

    with pytest.raises(BackendUnavailableError) as excinfo:
        backend.preflight(build(data, tmp_path))

    assert excinfo.value.extra == "apple"
    assert "finetune[apple]" in str(excinfo.value)


def test_plan_renders_without_executing_anything(data_dir: Path, tmp_path: Path) -> None:
    rendered = backend.plan(build(data_dir, tmp_path)).render()
    assert "mlx_lm lora" in rendered
    assert not (tmp_path / "out").exists(), "plan must not create the output directory"
