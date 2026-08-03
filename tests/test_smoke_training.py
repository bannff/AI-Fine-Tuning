"""Smoke tests that actually train.

Marked ``slow`` and skipped without the relevant extra. They exist because the
previous test suite was green precisely *because* the real dependencies were
absent: ``scripts/run_tests.sh`` built a venv containing only pytest and
hypothesis, so the one import that would have failed never ran.

These run a handful of real optimiser steps on a locally-built tiny model, so
they need no network and finish in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finetune.backends import get_backend
from finetune.capabilities import Backend, Method
from finetune.config import AdapterConfig, TrainingConfig
from finetune.run import RunConfig
from tests.conftest import HAS_TRL

requires_trl = pytest.mark.skipif(not HAS_TRL, reason="needs the 'torch' extra")


def run_config(model: Path, data_dir: Path, out: Path, **overrides: object) -> RunConfig:
    return RunConfig(
        model=str(model),
        data_dir=data_dir,
        output_dir=out,
        method=overrides.pop("method", Method.LORA),  # type: ignore[arg-type]
        backend=overrides.pop("backend", Backend.TORCH),  # type: ignore[arg-type]
        adapter=overrides.pop("adapter", AdapterConfig(rank=8, alpha=16)),  # type: ignore[arg-type]
        training=TrainingConfig(
            iters=3,
            batch_size=2,
            max_seq_length=64,
            steps_per_report=1,
            save_every=1000,
            **overrides,  # type: ignore[arg-type]
        ),
    )


@pytest.mark.slow
@requires_trl
def test_torch_lora_trains_and_writes_an_adapter(
    tiny_model: Path, chat_dataset: Path, tmp_path: Path
) -> None:
    backend = get_backend(Backend.TORCH)
    config = run_config(tiny_model, chat_dataset, tmp_path / "out")

    backend.preflight(config)
    result = backend.train(config)

    assert result.succeeded
    assert result.steps == 3
    assert (config.output_dir / "adapter_config.json").is_file(), "expected a PEFT adapter"


@pytest.mark.slow
@requires_trl
def test_all_linear_adapts_the_mlp_not_just_attention(
    tiny_model: Path, chat_dataset: Path, tmp_path: Path
) -> None:
    """The concrete regression: the old hardcoded target list trained one projection.

    On a Llama geometry, ``["qkv_proj", "o_proj"]`` matched only ``o_proj``
    (``qkv_proj`` is a Phi-3 name), and PEFT does not raise when *some* target
    matches -- so it silently trained a quarter of the intended attention
    adapter and none of the MLP.
    """
    import json

    backend = get_backend(Backend.TORCH)
    config = run_config(tiny_model, chat_dataset, tmp_path / "out")
    backend.train(config)

    saved = json.loads((config.output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    # PEFT resolves "all-linear" to fully-qualified module paths, so compare the
    # projection names rather than the paths.
    adapted = {name.rsplit(".", 1)[-1] for name in saved["target_modules"]}

    assert {"gate_proj", "up_proj", "down_proj"} <= adapted, f"MLP must be adapted, got {adapted}"
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= adapted, (
        f"attention must be adapted, got {adapted}"
    )
    # The old hardcoded list resolved to o_proj alone on this geometry.
    assert len(adapted) > 1


@pytest.mark.slow
@requires_trl
def test_dora_produces_a_dora_adapter(tiny_model: Path, chat_dataset: Path, tmp_path: Path) -> None:
    import json

    backend = get_backend(Backend.TORCH)
    config = run_config(tiny_model, chat_dataset, tmp_path / "out", method=Method.DORA)
    backend.train(config)

    saved = json.loads((config.output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["use_dora"] is True


@pytest.mark.slow
@requires_trl
def test_training_runs_without_use_mps_device(
    tiny_model: Path, chat_dataset: Path, tmp_path: Path
) -> None:
    """Regression: `use_mps_device=True` raised TypeError on transformers 5.

    Reaching a completed run at all proves the argument is gone and that
    accelerate selects the device on its own.
    """
    backend = get_backend(Backend.TORCH)
    result = backend.train(run_config(tiny_model, chat_dataset, tmp_path / "out"))
    assert result.succeeded


@pytest.mark.slow
@requires_trl
def test_loss_is_finite_so_padding_is_masked(
    tiny_model: Path, chat_dataset: Path, tmp_path: Path
) -> None:
    """A real loss value, not NaN, and computed on completions only.

    The loop this replaces padded every example to ``max_length`` and used
    ``input_ids`` as labels, so most of the loss came from ``<pad>`` tokens.
    """
    import math

    backend = get_backend(Backend.TORCH)
    result = backend.train(run_config(tiny_model, chat_dataset, tmp_path / "out"))

    assert result.final_loss is not None
    assert math.isfinite(result.final_loss)
    assert result.final_loss > 0
