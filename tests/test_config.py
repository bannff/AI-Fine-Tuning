"""Config validation and YAML loading tests.

The behaviour under test is mostly *rejection*. The repository previously
shipped YAML that nothing read, so the important guarantee is that a config
which cannot be honoured fails loudly instead of being ignored.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from finetune.capabilities import Backend, Method
from finetune.config import AdapterConfig, QuantizationConfig, TrainingConfig
from finetune.config_io import dump_run_config, load_run_config, parse_run_config, to_dict
from finetune.errors import ConfigError
from finetune.run import DEFAULT_LEARNING_RATE, RunConfig

MINIMAL = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "data_dir": "data",
    "output_dir": "out",
    "training": {"iters": 10},
}


def write_config(tmp_path: Path, payload: Mapping[str, Any], name: str = "run.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# ------------------------------------------------------- dataclass validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank": 0}, "rank must be positive"),
        ({"rank": -4}, "rank must be positive"),
        ({"alpha": 0}, "alpha must be positive"),
        ({"dropout": 1.0}, r"dropout must be in \[0, 1\)"),
        ({"dropout": -0.1}, r"dropout must be in \[0, 1\)"),
        ({"target_modules": "attention"}, "must be 'all-linear' or a list"),
        ({"target_modules": ()}, "must not be empty"),
        ({"num_layers": 0}, "num_layers must be -1 or positive"),
        ({"num_layers": -2}, "num_layers must be -1 or positive"),
    ],
)
def test_adapter_config_rejects_nonsense(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        AdapterConfig(**kwargs)


def test_adapter_scale_is_alpha_over_rank() -> None:
    assert AdapterConfig(rank=16, alpha=32).scale == pytest.approx(2.0)


def test_rslora_uses_sqrt_rank_scaling() -> None:
    """The whole point of rsLoRA: scale stops collapsing as rank grows."""
    plain = AdapterConfig(rank=64, alpha=32).scale
    stabilised = AdapterConfig(rank=64, alpha=32, use_rslora=True).scale

    assert plain == pytest.approx(0.5)
    assert stabilised == pytest.approx(4.0)
    assert stabilised > plain


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one of training.iters or training.epochs"),
        ({"iters": 10, "epochs": 2}, "mutually exclusive"),
        ({"iters": 0}, "iters must be positive"),
        ({"epochs": -1}, "epochs must be positive"),
        ({"iters": 5, "batch_size": 0}, "batch_size must be positive"),
        ({"iters": 5, "learning_rate": 0}, "learning_rate must be positive"),
        ({"iters": 5, "val_batches": 0}, r"val_batches must be -1 \(all\) or positive"),
        ({"iters": 5, "max_seq_length": -1}, "max_seq_length must be positive"),
    ],
)
def test_training_config_rejects_nonsense(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        TrainingConfig(**kwargs)


def test_effective_batch_size_accounts_for_accumulation() -> None:
    training = TrainingConfig(iters=1, batch_size=4, gradient_accumulation_steps=8)
    assert training.effective_batch_size == 32


@pytest.mark.parametrize(("bits", "group"), [(5, 64), (16, 64), (4, 0), (4, 33)])
def test_quantization_config_rejects_invalid_settings(bits: int, group: int) -> None:
    with pytest.raises(ConfigError):
        QuantizationConfig(bits=bits, group_size=group)


# --------------------------------------------------------- run-level coherence


def test_qlora_without_a_quantization_block_is_rejected() -> None:
    with pytest.raises(ConfigError, match="requires a quantization block"):
        RunConfig(
            model="m",
            data_dir=Path("d"),
            output_dir=Path("o"),
            method=Method.QLORA,
            training=TrainingConfig(iters=1),
        )


def test_full_finetuning_with_quantization_is_rejected() -> None:
    with pytest.raises(ConfigError, match="cannot be combined with quantization"):
        RunConfig(
            model="m",
            data_dir=Path("d"),
            output_dir=Path("o"),
            method=Method.FULL,
            quantization=QuantizationConfig(),
            training=TrainingConfig(iters=1),
        )


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ConfigError, match="non-empty"):
        RunConfig(
            model="", data_dir=Path("d"), output_dir=Path("o"), training=TrainingConfig(iters=1)
        )


@pytest.mark.parametrize("method", list(Method))
def test_learning_rate_defaults_by_method(method: Method) -> None:
    """Low-rank methods want ~100x the full fine-tuning rate; that is encoded, not guessed."""
    quant = QuantizationConfig() if method is Method.QLORA else None
    config = RunConfig(
        model="m",
        data_dir=Path("d"),
        output_dir=Path("o"),
        method=method,
        quantization=quant,
        training=TrainingConfig(iters=1),
    )
    assert config.learning_rate == DEFAULT_LEARNING_RATE[method]


def test_explicit_learning_rate_wins() -> None:
    config = RunConfig(
        model="m",
        data_dir=Path("d"),
        output_dir=Path("o"),
        training=TrainingConfig(iters=1, learning_rate=5e-5),
    )
    assert config.learning_rate == 5e-5


def test_uses_adapter_is_false_only_for_full() -> None:
    def build(method: Method) -> RunConfig:
        return RunConfig(
            model="m",
            data_dir=Path("d"),
            output_dir=Path("o"),
            method=method,
            quantization=QuantizationConfig() if method is Method.QLORA else None,
            training=TrainingConfig(iters=1),
        )

    assert not build(Method.FULL).uses_adapter
    assert all(build(m).uses_adapter for m in (Method.LORA, Method.DORA, Method.QLORA))


# ------------------------------------------------------------------ advisories


def test_large_effective_batch_raises_an_advisory_not_an_error() -> None:
    config = RunConfig(
        model="m",
        data_dir=Path("d"),
        output_dir=Path("o"),
        training=TrainingConfig(iters=1, batch_size=16, gradient_accumulation_steps=8),
    )
    assert any("effective batch size 128" in note for note in config.advisories())


def test_attention_only_targeting_raises_an_advisory() -> None:
    config = RunConfig(
        model="m",
        data_dir=Path("d"),
        output_dir=Path("o"),
        adapter=AdapterConfig(target_modules=("q_proj", "v_proj")),
        training=TrainingConfig(iters=1),
    )
    assert any("target_modules is restricted" in note for note in config.advisories())


def test_a_sensible_config_has_no_advisories() -> None:
    config = RunConfig(
        model="m",
        data_dir=Path("d"),
        output_dir=Path("o"),
        training=TrainingConfig(iters=100, batch_size=4, gradient_accumulation_steps=2),
    )
    assert config.advisories() == ()


# ------------------------------------------------------------------ YAML layer


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, {**MINIMAL, "learning_rate": 1e-4})
    with pytest.raises(ConfigError, match="unknown config key 'learning_rate'"):
        load_run_config(path)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    """`lora_alpha: 64` in a repo that reads `alpha` is exactly the old bug."""
    path = write_config(tmp_path, {**MINIMAL, "adapter": {"lora_alpha": 64}})
    with pytest.raises(ConfigError, match=re.escape("unknown config key 'adapter.lora_alpha'")):
        load_run_config(path)


def test_typos_get_a_suggestion(tmp_path: Path) -> None:
    path = write_config(tmp_path, {**MINIMAL, "adapter": {"rnak": 8}})
    with pytest.raises(ConfigError, match="did you mean 'rank'"):
        load_run_config(path)


def test_relative_paths_resolve_against_the_config_file(tmp_path: Path) -> None:
    """So a committed config works regardless of the working directory."""
    nested = tmp_path / "configs"
    nested.mkdir()
    path = write_config(nested, {**MINIMAL, "data_dir": "../data/tiny"})

    config = load_run_config(path)

    assert config.data_dir == (tmp_path / "data" / "tiny").resolve()
    assert config.data_dir.is_absolute()


def test_absolute_paths_are_left_alone(tmp_path: Path) -> None:
    path = write_config(tmp_path, {**MINIMAL, "data_dir": "/srv/datasets/train"})
    assert load_run_config(path).data_dir == Path("/srv/datasets/train")


def test_target_modules_list_becomes_a_tuple(tmp_path: Path) -> None:
    path = write_config(tmp_path, {**MINIMAL, "adapter": {"target_modules": ["q_proj", "v_proj"]}})
    assert load_run_config(path).adapter.target_modules == ("q_proj", "v_proj")


def test_bad_method_lists_the_valid_options(tmp_path: Path) -> None:
    path = write_config(tmp_path, {**MINIMAL, "method": "sft"})
    with pytest.raises(ConfigError, match="method must be one of: dora, full, lora, qlora"):
        load_run_config(path)


def test_bad_backend_lists_the_valid_options(tmp_path: Path) -> None:
    path = write_config(tmp_path, {**MINIMAL, "backend": "tensorflow"})
    with pytest.raises(ConfigError, match="backend must be one of: mlx, torch"):
        load_run_config(path)


@pytest.mark.parametrize("missing", ["model", "data_dir", "output_dir"])
def test_missing_required_keys_are_named(tmp_path: Path, missing: str) -> None:
    payload = {k: v for k, v in MINIMAL.items() if k != missing}
    path = write_config(tmp_path, payload)
    with pytest.raises(ConfigError, match=f"missing required key '{missing}'"):
        load_run_config(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="config file is empty"):
        load_run_config(path)


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_run_config(tmp_path / "nope.yaml")


def test_malformed_yaml_is_reported_clearly(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_run_config(path)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_run_config(path)


# --------------------------------------------------------------- serialisation


def test_dump_records_the_resolved_learning_rate(tmp_path: Path) -> None:
    config = parse_run_config({**MINIMAL, "method": "full"}, base_dir=tmp_path)
    payload = to_dict(config)

    assert payload["training"]["learning_rate"] == DEFAULT_LEARNING_RATE[Method.FULL]
    assert "adapter" not in payload, "full fine-tuning has no adapter to record"


def test_dump_round_trips_through_yaml(tmp_path: Path) -> None:
    original = parse_run_config(
        {**MINIMAL, "adapter": {"rank": 8, "target_modules": ["q_proj"]}},
        base_dir=tmp_path,
    )
    path = tmp_path / "resolved.yaml"
    path.write_text(dump_run_config(original), encoding="utf-8")

    reloaded = load_run_config(path)

    assert reloaded.model == original.model
    assert reloaded.adapter.rank == 8
    assert reloaded.adapter.target_modules == ("q_proj",)
    assert reloaded.learning_rate == original.learning_rate


def test_shipped_configs_all_load() -> None:
    """Every config committed to configs/ must be valid. Guards against rot."""
    configs = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))
    assert configs, "expected shipped example configs"
    for path in configs:
        config = load_run_config(path)
        assert config.model
        assert config.learning_rate > 0


def test_with_overrides_revalidates(tmp_path: Path) -> None:
    config = parse_run_config(MINIMAL, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="requires a quantization block"):
        config.with_overrides(method=Method.QLORA)


def test_with_overrides_changes_backend(tmp_path: Path) -> None:
    config = parse_run_config(MINIMAL, base_dir=tmp_path)
    assert config.with_overrides(backend=Backend.TORCH).backend is Backend.TORCH
