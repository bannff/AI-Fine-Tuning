"""Splitting tests.

The properties asserted here are the ones the old suite lacked: that counts sum
to the input for *any* valid ratio (not just 0.8/0.1/0.1), that no split is
empty, that shuffling actually happens, and that it is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from finetune.data.io import read_jsonl
from finetune.data.split import DEFAULT_SPLIT, allocate, split_records, write_splits
from finetune.errors import DatasetSchemaError


def records(count: int) -> list[dict[str, object]]:
    return [{"text": f"record-{i}", "index": i} for i in range(count)]


# ----------------------------------------------------------------- allocation


@given(
    total=st.integers(min_value=3, max_value=100_000),
    train=st.floats(min_value=0.05, max_value=0.98),
    valid=st.floats(min_value=0.01, max_value=0.9),
)
def test_allocation_is_exact_and_non_empty(total: int, train: float, valid: float) -> None:
    """Counts sum to the input and no requested split is starved, at any ratio."""
    assume(train + valid < 0.999)
    ratio = (train, valid, 1.0 - train - valid)

    counts = allocate(total, ratio)

    assert counts.total == total, "allocation must not lose or invent records"
    assert counts.train >= 1
    assert counts.valid >= 1
    assert counts.test >= 1


@pytest.mark.parametrize("total", [3, 4, 7, 10, 11, 999])
def test_default_ratio_allocates_exactly(total: int) -> None:
    counts = allocate(total, DEFAULT_SPLIT)
    assert counts.total == total


def test_tiny_dataset_still_yields_a_validation_set() -> None:
    """int(3 * 0.1) == 0; the old splitter emitted an unusable empty valid.jsonl."""
    counts = allocate(3, DEFAULT_SPLIT)
    assert counts.train >= 1
    assert counts.valid >= 1
    assert counts.test >= 1


def test_two_way_split_is_allowed() -> None:
    counts = allocate(10, (0.9, 0.1, 0.0))
    assert counts.test == 0
    assert counts.total == 10


@pytest.mark.parametrize(
    ("ratio", "message"),
    [
        ((0.8, 0.1), "exactly 3 parts"),
        ((0.5, 0.2, 0.2), "must sum to 1.0"),
        ((0.9, 0.0, 0.1), "positive share to train and valid"),
        ((1.2, -0.1, -0.1), "non-negative"),
    ],
)
def test_invalid_ratios_are_rejected(ratio: tuple[float, ...], message: str) -> None:
    with pytest.raises(DatasetSchemaError, match=message):
        allocate(100, ratio)


def test_too_few_records_is_rejected() -> None:
    with pytest.raises(DatasetSchemaError, match="need at least 3 records"):
        allocate(2, DEFAULT_SPLIT)


# -------------------------------------------------------------- partitioning


@given(count=st.integers(min_value=3, max_value=2_000))
def test_split_preserves_every_record_exactly_once(count: int) -> None:
    splits = split_records(records(count))
    seen = [r["index"] for part in ("train", "valid", "test") for r in splits[part]]
    assert sorted(seen) == list(range(count)), "records must be partitioned, not copied or dropped"


def test_split_is_reproducible_for_a_given_seed() -> None:
    first = split_records(records(200), seed=7)
    second = split_records(records(200), seed=7)
    assert [r["index"] for r in first["train"]] == [r["index"] for r in second["train"]]


def test_different_seeds_give_different_partitions() -> None:
    a = split_records(records(200), seed=1)["train"]
    b = split_records(records(200), seed=2)["train"]
    assert [r["index"] for r in a] != [r["index"] for r in b]


def test_shuffling_defeats_positional_bias() -> None:
    """The old splitter sliced in order, so a sorted dataset gave a skewed holdout."""
    ordered = split_records(records(100), shuffle=False)
    shuffled = split_records(records(100), shuffle=True, seed=42)

    assert [r["index"] for r in ordered["test"]] == list(range(90, 100))
    assert [r["index"] for r in shuffled["test"]] != list(range(90, 100))


def test_no_shuffle_keeps_input_order() -> None:
    splits = split_records(records(10), shuffle=False)
    assert [r["index"] for r in splits["train"]] == list(range(8))


def test_split_does_not_disturb_global_random_state() -> None:
    import random

    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    split_records(records(50), seed=99)
    assert random.random() == expected


# ------------------------------------------------------------------- writing


def test_write_splits_creates_files_and_counts_match(tmp_path: Path) -> None:
    counts = write_splits(records(100), tmp_path)

    assert counts.total == 100
    for name, expected in counts.as_dict().items():
        rows = list(read_jsonl(tmp_path / f"{name}.jsonl"))
        assert len(rows) == expected


def test_write_splits_validates_before_writing_anything(tmp_path: Path) -> None:
    """A bad dataset must not leave a partial directory behind."""
    bad = [{"text": "fine"}, {"unsupported": "shape"}]

    with pytest.raises(DatasetSchemaError):
        write_splits(bad, tmp_path)

    assert not list(tmp_path.glob("*.jsonl"))


def test_write_splits_omits_empty_test_file(tmp_path: Path) -> None:
    write_splits(records(10), tmp_path, ratio=(0.9, 0.1, 0.0))
    assert not (tmp_path / "test.jsonl").exists()
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "valid.jsonl").exists()


def test_written_splits_pass_the_directory_gate(tmp_path: Path) -> None:
    """End to end: what prepare-data writes is what train accepts."""
    from finetune.data.layout import validate_split_dir

    write_splits(records(50), tmp_path)
    assert validate_split_dir(tmp_path)
