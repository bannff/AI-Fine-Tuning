"""JSONL reading and writing.

Errors name the file and line number, because a malformed line in a 100k-record
dataset is otherwise very hard to find.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from ..errors import DatasetSchemaError


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each record from a JSONL file, skipping blank lines.

    Streams rather than materialising, so validating a large dataset does not
    require holding it in memory.

    Raises:
        DatasetSchemaError: if the file is unreadable or a line is not valid JSON.
    """
    file_path = Path(path)
    try:
        handle = file_path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise DatasetSchemaError(f"dataset file not found: {file_path}") from None
    except OSError as exc:
        raise DatasetSchemaError(f"could not read {file_path}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(
                    f"{file_path} line {line_number}: invalid JSON ({exc.msg})"
                ) from None


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write records as JSONL, creating parent directories. Returns the count.

    ``ensure_ascii=False`` keeps non-English text readable in the output rather
    than escaping it, which matters when eyeballing a prepared split.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


__all__ = ["read_jsonl", "write_jsonl"]
