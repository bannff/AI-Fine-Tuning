"""Dataset handling: schema validation, deterministic splitting, JSONL IO.

Backend-agnostic on purpose. Both trainers this package drives accept the same
three record schemas, so validation belongs here rather than being duplicated
(and drifting) inside each backend.
"""

from __future__ import annotations

from .io import read_jsonl, write_jsonl
from .layout import REQUIRED_SPLITS, SPLIT_FILENAMES, validate_split_dir
from .schema import (
    ACCEPTED_SCHEMAS,
    DatasetSchema,
    detect_schema,
    validate_record,
    validate_records,
)
from .split import DEFAULT_SPLIT, SplitCounts, allocate, split_records, write_splits

__all__ = [
    "ACCEPTED_SCHEMAS",
    "DEFAULT_SPLIT",
    "REQUIRED_SPLITS",
    "SPLIT_FILENAMES",
    "DatasetSchema",
    "SplitCounts",
    "allocate",
    "detect_schema",
    "read_jsonl",
    "split_records",
    "validate_record",
    "validate_records",
    "validate_split_dir",
    "write_jsonl",
    "write_splits",
]
