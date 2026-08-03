"""CLI tests.

Every command is exercised through ``main()`` with real argv, asserting on exit
codes. The contract is that user error exits 2 with a message, never a traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune.cli import main


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "raw.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]
        }
        for i in range(60)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def write_config(tmp_path: Path, data_dir: Path, **extra: object) -> Path:
    import yaml

    payload: dict[str, object] = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "backend": "mlx",
        "method": "lora",
        "data_dir": str(data_dir),
        "output_dir": str(tmp_path / "out"),
        "training": {"iters": 10},
    }
    payload.update(extra)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------- capabilities


def test_capabilities_lists_both_backends(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["capabilities"]) == 0
    out = capsys.readouterr().out
    assert "mlx" in out
    assert "torch" in out


def test_capabilities_declares_what_is_not_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The omission of DPO should read as a decision, not an oversight."""
    main(["capabilities"])
    out = capsys.readouterr().out
    assert "Not implemented here, on purpose" in out
    assert "dpo" in out.lower()


# ------------------------------------------------------------------- data prep


def test_prepare_then_validate_round_trip(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "prepared"

    assert main(["prepare-data", str(dataset), str(out_dir)]) == 0
    assert main(["validate-data", str(out_dir)]) == 0

    report = capsys.readouterr().out
    assert "valid, schema 'chat'" in report


def test_prepare_data_honours_limit(dataset: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "prepared"
    assert main(["prepare-data", str(dataset), str(out_dir), "--limit", "12"]) == 0

    from finetune.data.io import read_jsonl

    total = sum(
        len(list(read_jsonl(out_dir / name)))
        for name in ("train.jsonl", "valid.jsonl", "test.jsonl")
    )
    assert total == 12


def test_prepare_data_rejects_a_bad_ratio(dataset: Path, tmp_path: Path) -> None:
    code = main(["prepare-data", str(dataset), str(tmp_path / "p"), "--ratio", "0.5", "0.2", "0.2"])
    assert code == 2


def test_validate_data_reports_a_missing_directory(tmp_path: Path) -> None:
    assert main(["validate-data", str(tmp_path / "absent")]) == 2


def test_validate_data_rejects_unsupported_records(tmp_path: Path) -> None:
    data = tmp_path / "d"
    data.mkdir()
    for name in ("train.jsonl", "valid.jsonl"):
        (data / name).write_text('{"instruction": "unsupported"}\n', encoding="utf-8")
    assert main(["validate-data", str(data)]) == 2


# ----------------------------------------------------------------------- plan


def test_plan_prints_the_command_and_resolved_config(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "prepared"
    main(["prepare-data", str(dataset), str(data_dir)])
    config = write_config(tmp_path, data_dir)

    assert main(["plan", str(config)]) == 0

    out = capsys.readouterr().out
    assert "mlx_lm" in out
    assert "Resolved configuration" in out
    assert "learning_rate: 0.0002" in out


def test_plan_does_not_create_the_output_directory(dataset: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "prepared"
    main(["prepare-data", str(dataset), str(data_dir)])
    main(["plan", str(write_config(tmp_path, data_dir))])
    assert not (tmp_path / "out").exists()


def test_plan_applies_method_override(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "prepared"
    main(["prepare-data", str(dataset), str(data_dir)])
    config = write_config(tmp_path, data_dir)

    assert main(["plan", str(config), "--method", "dora"]) == 0
    assert "dora" in capsys.readouterr().out


def test_plan_rejects_an_unsupported_override(dataset: Path, tmp_path: Path) -> None:
    """full fine-tuning on torch is not implemented, so this must fail cleanly."""
    data_dir = tmp_path / "prepared"
    main(["prepare-data", str(dataset), str(data_dir)])
    config = write_config(tmp_path, data_dir)

    assert main(["plan", str(config), "--method", "full", "--backend", "torch"]) == 2


def test_plan_warns_about_restricted_target_modules(
    dataset: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    data_dir = tmp_path / "prepared"
    main(["prepare-data", str(dataset), str(data_dir)])
    config = write_config(tmp_path, data_dir, adapter={"target_modules": ["q_proj"]})

    with caplog.at_level(logging.WARNING):
        assert main(["plan", str(config)]) == 0

    assert "target_modules is restricted" in caplog.text


# ---------------------------------------------------------------- error paths


def test_missing_config_exits_two_without_a_traceback(tmp_path: Path) -> None:
    assert main(["plan", str(tmp_path / "nope.yaml")]) == 2


def test_unknown_config_key_exits_two(dataset: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "prepared"
    main(["prepare-data", str(dataset), str(data_dir)])
    config = write_config(tmp_path, data_dir, adapter={"lora_alpha": 64})
    assert main(["plan", str(config)]) == 2


def test_unknown_command_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate"])
    assert excinfo.value.code == 2


def test_no_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main([])
