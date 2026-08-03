"""Validation of a prepared split directory.

Separate from :mod:`finetune.data.schema`, which is about the shape of a single
record. This module is about the shape of a *directory*: which files must exist
and that they agree with each other.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import DatasetSchemaError
from .io import read_jsonl
from .schema import DatasetSchema, validate_records

#: Filenames a prepared split directory may contain, in trainer order.
SPLIT_FILENAMES: tuple[str, ...] = ("train.jsonl", "valid.jsonl", "test.jsonl")

#: Splits a trainer requires. ``test.jsonl`` is only read with --test.
REQUIRED_SPLITS: tuple[str, ...] = ("train.jsonl", "valid.jsonl")


def validate_split_dir(data_dir: Path) -> dict[str, DatasetSchema]:
    """Validate a prepared ``train``/``valid``/``test`` directory.

    Returns:
        Mapping of filename to the schema it uses.

    Raises:
        DatasetSchemaError: if a required split is missing, any record is
            invalid, or the splits disagree on schema.
    """
    if not data_dir.is_dir():
        raise DatasetSchemaError(f"data directory not found: {data_dir}")

    schemas: dict[str, DatasetSchema] = {}
    for filename in REQUIRED_SPLITS:
        path = data_dir / filename
        if not path.is_file():
            raise DatasetSchemaError(
                f"required split {filename!r} missing from {data_dir}. "
                "Run 'finetune prepare-data' to build it."
            )
        schemas[filename] = validate_records(read_jsonl(path), source=str(path))

    optional = data_dir / "test.jsonl"
    if optional.is_file():
        schemas["test.jsonl"] = validate_records(read_jsonl(optional), source=str(optional))

    # The trainers select a dataset class from the first record they see, so a
    # validation split in a different shape than training would be misread.
    distinct = set(schemas.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{name}={schema}" for name, schema in sorted(schemas.items()))
        raise DatasetSchemaError(f"splits disagree on schema ({detail})")
    return schemas


__all__ = ["REQUIRED_SPLITS", "SPLIT_FILENAMES", "validate_split_dir"]
