"""PyTorch backend tests.

The pure config-construction tests run everywhere. Anything that needs TRL or
PEFT installed is skipped when the ``torch`` extra is absent, so the light CI
job stays fast without pretending to have covered the heavy path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from finetune.backends import _torch_args as torch_args
from finetune.backends.torch_backend import TorchBackend
from finetune.capabilities import Backend, Method
from finetune.config import AdapterConfig, QuantizationConfig, TrainingConfig
from finetune.run import RunConfig

HAS_TRL = importlib.util.find_spec("trl") is not None
requires_trl = pytest.mark.skipif(not HAS_TRL, reason="needs the 'torch' extra")

backend = TorchBackend()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    rows = "\n".join(
        f'{{"messages": [{{"role": "user", "content": "q{i}"}},'
        f' {{"role": "assistant", "content": "a{i}"}}]}}'
        for i in range(12)
    )
    for name in ("train.jsonl", "valid.jsonl"):
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
        backend=Backend.TORCH,
        quantization=QuantizationConfig() if method is Method.QLORA else None,
        adapter=adapter or AdapterConfig(),
        training=training or TrainingConfig(iters=10),
    )


# ------------------------------------------------------------- adapter config


def test_target_modules_defaults_to_every_linear_layer(data_dir: Path, tmp_path: Path) -> None:
    """Attention-only targeting underperforms and raising rank does not fix it.

    The code this replaces hardcoded Phi-3 names (`qkv_proj`, `o_proj`); against
    a Llama model only `o_proj` matched, so it silently trained one projection.
    """
    kwargs = torch_args.peft_kwargs(build(data_dir, tmp_path))
    assert kwargs["target_modules"] == "all-linear"


def test_explicit_target_modules_become_a_list(data_dir: Path, tmp_path: Path) -> None:
    config = build(data_dir, tmp_path, adapter=AdapterConfig(target_modules=("q_proj", "v_proj")))
    kwargs = torch_args.peft_kwargs(config)
    assert kwargs["target_modules"] == ["q_proj", "v_proj"]


def test_use_dora_is_set_only_for_the_dora_method(data_dir: Path, tmp_path: Path) -> None:
    assert torch_args.peft_kwargs(build(data_dir, tmp_path, Method.DORA))["use_dora"] is True
    assert torch_args.peft_kwargs(build(data_dir, tmp_path, Method.LORA))["use_dora"] is False
    assert torch_args.peft_kwargs(build(data_dir, tmp_path, Method.QLORA))["use_dora"] is False


def test_rslora_is_forwarded_to_peft(data_dir: Path, tmp_path: Path) -> None:
    config = build(data_dir, tmp_path, adapter=AdapterConfig(use_rslora=True))
    assert torch_args.peft_kwargs(config)["use_rslora"] is True


def test_rank_and_alpha_stay_independent(data_dir: Path, tmp_path: Path) -> None:
    config = build(data_dir, tmp_path, adapter=AdapterConfig(rank=64, alpha=16))
    kwargs = torch_args.peft_kwargs(config)
    assert kwargs["r"] == 64
    assert kwargs["lora_alpha"] == 16


def test_task_type_is_causal_lm(data_dir: Path, tmp_path: Path) -> None:
    assert torch_args.peft_kwargs(build(data_dir, tmp_path))["task_type"] == "CAUSAL_LM"


# ------------------------------------------------------------------ SFT config


@requires_trl
def test_use_mps_device_is_never_passed(data_dir: Path, tmp_path: Path) -> None:
    """It was removed in transformers 5 and raised TypeError; accelerate picks MPS itself."""
    assert "use_mps_device" not in torch_args.sft_kwargs(build(data_dir, tmp_path))


@requires_trl
def test_completion_only_loss_follows_mask_prompt(data_dir: Path, tmp_path: Path) -> None:
    on = torch_args.sft_kwargs(
        build(data_dir, tmp_path, training=TrainingConfig(iters=5, mask_prompt=True))
    )
    off = torch_args.sft_kwargs(
        build(data_dir, tmp_path, training=TrainingConfig(iters=5, mask_prompt=False))
    )
    assert on["completion_only_loss"] is True
    assert off["completion_only_loss"] is False


@requires_trl
def test_sft_kwargs_are_all_accepted_by_the_installed_trl(data_dir: Path, tmp_path: Path) -> None:
    """Every key must exist on SFTConfig, so construction cannot raise TypeError."""
    import dataclasses

    from trl import SFTConfig

    valid = {f.name for f in dataclasses.fields(SFTConfig)}
    for method in (Method.LORA, Method.DORA, Method.QLORA):
        kwargs = torch_args.sft_kwargs(build(data_dir, tmp_path, method))
        assert set(kwargs) <= valid, f"{method}: {set(kwargs) - valid}"


@requires_trl
def test_sft_config_actually_constructs(data_dir: Path, tmp_path: Path) -> None:
    from trl import SFTConfig

    config = SFTConfig(**torch_args.sft_kwargs(build(data_dir, tmp_path)))
    assert config.output_dir == str(tmp_path / "out")


# ------------------------------------------------------------------- precision
#
# TRL defaults SFTConfig to bf16=True, and transformers then raises
# "Your setup doesn't support bf16/gpu" on any machine without a bf16-capable
# accelerator. That shipped broken: it passed on Apple Silicon, where MPS
# satisfies the check, and failed on every CPU-only box including CI.
#
# Parametrised rather than hardware-gated so all three branches are covered on
# whatever machine happens to run the suite.


@requires_trl
@pytest.mark.parametrize(
    ("cuda", "bf16_supported", "mps", "expected"),
    [
        (True, True, False, {"bf16": True, "fp16": False}),
        (True, False, False, {"bf16": False, "fp16": True}),
        (False, False, True, {"bf16": False, "fp16": False}),
        (False, False, False, {"use_cpu": True, "bf16": False, "fp16": False}),
    ],
    ids=["cuda-ampere", "cuda-pre-ampere", "apple-mps", "cpu-only"],
)
def test_precision_follows_available_hardware(
    monkeypatch: pytest.MonkeyPatch,
    cuda: bool,
    bf16_supported: bool,
    mps: bool,
    expected: dict[str, bool],
) -> None:
    import torch
    import transformers.utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(transformers.utils, "is_torch_bf16_gpu_available", lambda: bf16_supported)

    assert torch_args.precision_kwargs() == expected


@requires_trl
def test_cpu_only_config_constructs(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, tmp_path: Path
) -> None:
    """The regression CI caught: SFTConfig must build with no accelerator at all."""
    import torch
    from trl import SFTConfig

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    config = SFTConfig(**torch_args.sft_kwargs(build(data_dir, tmp_path)))

    assert config.use_cpu is True
    assert config.bf16 is False


@requires_trl
def test_iters_and_epochs_map_to_different_trl_fields(data_dir: Path, tmp_path: Path) -> None:
    steps = torch_args.sft_kwargs(build(data_dir, tmp_path, training=TrainingConfig(iters=42)))
    epochs = torch_args.sft_kwargs(build(data_dir, tmp_path, training=TrainingConfig(epochs=3)))
    assert steps["max_steps"] == 42
    assert "max_steps" not in epochs
    assert epochs["num_train_epochs"] == 3


@requires_trl
def test_unknown_keys_are_dropped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A wide version floor means TRL may not have every field; dropping must be loud."""
    import logging

    from finetune.backends._torch_args import filter_to_dataclass

    with caplog.at_level(logging.WARNING):
        from trl import SFTConfig

        kept = filter_to_dataclass({"output_dir": "x", "not_a_real_field": 1}, SFTConfig)

    assert kept == {"output_dir": "x"}
    assert "not_a_real_field" in caplog.text


# ------------------------------------------------------------------- planning


def test_plan_names_the_method_and_loss_shape(data_dir: Path, tmp_path: Path) -> None:
    plan = backend.plan(build(data_dir, tmp_path, Method.DORA))
    rendered = plan.render()
    assert "DoRA" in rendered
    assert "all-linear" in rendered
    assert "completion-only loss" in rendered


def test_plan_for_qlora_mentions_bitsandbytes(data_dir: Path, tmp_path: Path) -> None:
    plan = backend.plan(build(data_dir, tmp_path, Method.QLORA))
    assert any("bitsandbytes" in stage for stage in plan.stages)


def test_plan_has_no_argv_for_an_in_process_backend(data_dir: Path, tmp_path: Path) -> None:
    assert backend.plan(build(data_dir, tmp_path)).argv is None


def test_full_finetuning_is_rejected_on_torch(data_dir: Path, tmp_path: Path) -> None:
    from finetune.errors import UnsupportedCombinationError

    config = build(data_dir, tmp_path).with_overrides(method=Method.FULL)
    with pytest.raises(UnsupportedCombinationError, match="Supported by: mlx"):
        backend.preflight(config)


@requires_trl
def test_splits_load_as_datasets(data_dir: Path, tmp_path: Path) -> None:
    train, valid = backend.load_splits(build(data_dir, tmp_path))
    assert len(train) == 12
    assert len(valid) == 12
    assert "messages" in train.column_names
