"""The record schemas both trainers accept, and validation against them.

``mlx_lm.tuner.datasets`` recognises exactly three shapes, and TRL's
``SFTTrainer`` recognises the same three. That intersection is the contract this
package enforces:

===============  ===========================================================
``chat``         ``{"messages": [{"role": ..., "content": ...}, ...]}``
``completions``  ``{"prompt": ..., "completion": ...}``
``text``         ``{"text": ...}``
===============  ===========================================================

Validating up front matters because the failure mode otherwise is bad: a
splitter that passes arbitrary records through produces files the trainer rejects
only after model loading, and a test that generates arbitrary records passes while
proving nothing.

Directory-level checks live in :mod:`finetune.data.layout`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

from ..errors import DatasetSchemaError

#: Roles permitted in a ``chat`` record. ``tool`` is accepted because chat
#: templates that support tool calling emit it.
VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class DatasetSchema(str, Enum):
    """A supported record shape."""

    CHAT = "chat"
    COMPLETIONS = "completions"
    TEXT = "text"

    def __str__(self) -> str:
        return self.value


#: Required keys per schema, checked in this order so that a record with both
#: ``messages`` and ``text`` is classified as chat rather than ambiguously.
ACCEPTED_SCHEMAS: tuple[tuple[DatasetSchema, frozenset[str]], ...] = (
    (DatasetSchema.CHAT, frozenset({"messages"})),
    (DatasetSchema.COMPLETIONS, frozenset({"prompt", "completion"})),
    (DatasetSchema.TEXT, frozenset({"text"})),
)


def detect_schema(record: Mapping[str, Any]) -> DatasetSchema:
    """Classify a single record.

    Raises:
        DatasetSchemaError: if the record matches none of the three schemas.
    """
    for schema, required in ACCEPTED_SCHEMAS:
        if required <= record.keys():
            return schema
    raise DatasetSchemaError(
        f"record does not match any accepted schema. Found keys "
        f"{sorted(record.keys())}; expected one of "
        "{'messages'}, {'prompt', 'completion'}, or {'text'}"
    )


def _check_chat(record: Mapping[str, Any]) -> None:
    messages = record["messages"]
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes):
        raise DatasetSchemaError("'messages' must be a list of message objects")
    if not messages:
        raise DatasetSchemaError("'messages' must not be empty")
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise DatasetSchemaError(
                f"messages[{index}] must be an object, got {type(message).__name__}"
            )
        missing = {"role", "content"} - message.keys()
        if missing:
            raise DatasetSchemaError(f"messages[{index}] is missing {', '.join(sorted(missing))}")
        role = message["role"]
        if role not in VALID_ROLES:
            raise DatasetSchemaError(
                f"messages[{index}].role is {role!r}; expected one of "
                f"{', '.join(sorted(VALID_ROLES))}"
            )
        if not isinstance(message["content"], str):
            raise DatasetSchemaError(
                f"messages[{index}].content must be a string, got "
                f"{type(message['content']).__name__}"
            )


def _check_non_empty_strings(record: Mapping[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        value = record[key]
        if not isinstance(value, str):
            raise DatasetSchemaError(f"'{key}' must be a string, got {type(value).__name__}")
        if not value.strip():
            raise DatasetSchemaError(f"'{key}' must not be empty or whitespace-only")


def validate_record(record: Mapping[str, Any]) -> DatasetSchema:
    """Validate one record and return the schema it satisfies.

    Raises:
        DatasetSchemaError: with a message naming the offending field.
    """
    schema = detect_schema(record)
    if schema is DatasetSchema.CHAT:
        _check_chat(record)
    elif schema is DatasetSchema.COMPLETIONS:
        _check_non_empty_strings(record, ("prompt", "completion"))
    else:
        _check_non_empty_strings(record, ("text",))
    return schema


def validate_records(
    records: Iterable[Mapping[str, Any]], *, source: str = "<records>"
) -> DatasetSchema:
    """Validate every record and assert they share one schema.

    A mixed-schema file is rejected: the trainers pick a dataset class from the
    first record, so a file that changes shape halfway through would train on
    only part of itself.

    Returns:
        The single schema every record satisfies.

    Raises:
        DatasetSchemaError: on an empty iterable, an invalid record, or a
            schema change partway through.
    """
    first: DatasetSchema | None = None
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise DatasetSchemaError(
                f"{source} line {index}: expected a JSON object, got {type(record).__name__}"
            )
        try:
            schema = validate_record(record)
        except DatasetSchemaError as exc:
            raise DatasetSchemaError(f"{source} line {index}: {exc}") from None
        if first is None:
            first = schema
        elif schema is not first:
            raise DatasetSchemaError(
                f"{source} line {index}: schema changed from {first} to {schema}. "
                "Every record in a file must use the same schema."
            )

    if first is None:
        raise DatasetSchemaError(f"{source} contains no records")
    return first


__all__ = [
    "ACCEPTED_SCHEMAS",
    "VALID_ROLES",
    "DatasetSchema",
    "detect_schema",
    "validate_record",
    "validate_records",
]
