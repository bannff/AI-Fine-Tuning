"""Schema validation tests.

These deliberately assert that *invalid* data is rejected. The suite this
replaces generated arbitrary ``dict[str, str]`` records with Hypothesis and
asserted only on split arithmetic, so it passed on data neither trainer can
load -- the property held while the pipeline was broken.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from finetune.data.layout import validate_split_dir
from finetune.data.schema import (
    VALID_ROLES,
    DatasetSchema,
    detect_schema,
    validate_record,
    validate_records,
)
from finetune.errors import DatasetSchemaError

# ---------------------------------------------------------------- strategies

roles = st.sampled_from(sorted(VALID_ROLES))
prose = st.text(min_size=1).filter(lambda s: bool(s.strip()))

chat_records = st.builds(
    lambda msgs: {"messages": msgs},
    st.lists(
        st.builds(lambda r, c: {"role": r, "content": c}, roles, prose),
        min_size=1,
        max_size=6,
    ),
)
completion_records = st.builds(lambda p, c: {"prompt": p, "completion": c}, prose, prose)
text_records = st.builds(lambda t: {"text": t}, prose)


# ------------------------------------------------------------------ accepted


@given(chat_records)
def test_chat_records_are_accepted(record: dict[str, object]) -> None:
    assert validate_record(record) is DatasetSchema.CHAT


@given(completion_records)
def test_completion_records_are_accepted(record: dict[str, object]) -> None:
    assert validate_record(record) is DatasetSchema.COMPLETIONS


@given(text_records)
def test_text_records_are_accepted(record: dict[str, object]) -> None:
    assert validate_record(record) is DatasetSchema.TEXT


def test_chat_wins_when_a_record_has_several_shapes() -> None:
    """Detection order is fixed, so classification is deterministic."""
    record = {"messages": [{"role": "user", "content": "hi"}], "text": "hi"}
    assert detect_schema(record) is DatasetSchema.CHAT


# ------------------------------------------------------------------ rejected


@given(
    st.dictionaries(
        keys=st.text(min_size=1).filter(
            lambda k: k not in {"messages", "prompt", "completion", "text"}
        ),
        values=st.text(),
        min_size=1,
        max_size=5,
    )
)
def test_arbitrary_dicts_are_rejected(record: dict[str, str]) -> None:
    """The exact input the previous test suite used to prove nothing."""
    with pytest.raises(DatasetSchemaError):
        validate_record(record)


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"prompt": "q"}, "does not match any accepted schema"),
        ({"messages": []}, "must not be empty"),
        ({"messages": "not a list"}, "must be a list"),
        ({"messages": [{"role": "user"}]}, "missing content"),
        ({"messages": [{"role": "wizard", "content": "x"}]}, "expected one of"),
        ({"messages": [{"role": "user", "content": 42}]}, "must be a string"),
        ({"text": ""}, "must not be empty"),
        ({"text": "   "}, "must not be empty"),
        ({"text": 5}, "must be a string"),
        ({"prompt": "q", "completion": ""}, "must not be empty"),
    ],
)
def test_malformed_records_name_the_problem(record: dict[str, object], expected: str) -> None:
    with pytest.raises(DatasetSchemaError, match=expected):
        validate_record(record)


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(DatasetSchemaError, match="contains no records"):
        validate_records([])


def test_mixed_schemas_are_rejected_with_a_line_number() -> None:
    records = [{"text": "a"}, {"text": "b"}, {"prompt": "q", "completion": "a"}]
    with pytest.raises(DatasetSchemaError, match="line 3: schema changed from text to completions"):
        validate_records(records)


def test_error_messages_carry_the_source_and_line() -> None:
    with pytest.raises(DatasetSchemaError, match=r"mydata\.jsonl line 2"):
        validate_records([{"text": "ok"}, {"text": ""}], source="mydata.jsonl")


# ------------------------------------------------------- split directory gate


def test_validate_split_dir_requires_train_and_valid(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"text": "a"}\n', encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match=re.escape("required split 'valid.jsonl' missing")):
        validate_split_dir(tmp_path)


def test_validate_split_dir_rejects_disagreeing_splits(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"text": "a"}\n', encoding="utf-8")
    (tmp_path / "valid.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8"
    )
    with pytest.raises(DatasetSchemaError, match="splits disagree on schema"):
        validate_split_dir(tmp_path)


def test_validate_split_dir_accepts_a_consistent_directory(tmp_path: Path) -> None:
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        (tmp_path / name).write_text('{"text": "sample"}\n', encoding="utf-8")
    schemas = validate_split_dir(tmp_path)
    assert set(schemas) == {"train.jsonl", "valid.jsonl", "test.jsonl"}
    assert set(schemas.values()) == {DatasetSchema.TEXT}


def test_missing_directory_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(DatasetSchemaError, match="data directory not found"):
        validate_split_dir(tmp_path / "nope")
