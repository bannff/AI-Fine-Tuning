"""Deterministic train/valid/test splitting.

Three properties the previous implementation lacked:

Shuffling. The old splitter sliced positionally, so a dataset grouped by topic
or label produced a validation set drawn from a different distribution than
training. The shuffle is seeded, so it is reproducible without being positional.

Exact accounting. Counts are allocated by largest remainder, so they sum to the
input length for any ratio rather than dropping records to floor division.

Non-empty splits. ``mlx_lm`` and TRL both need a non-empty validation set; a
ratio that would round one to zero borrows from the largest split instead of
emitting a file the trainer cannot use.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import DatasetSchemaError
from .layout import SPLIT_FILENAMES

#: train / valid / test.
DEFAULT_SPLIT: tuple[float, float, float] = (0.8, 0.1, 0.1)

#: Smallest dataset that can yield three non-empty splits.
MIN_RECORDS = 3

_RATIO_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class SplitCounts:
    """How many records landed in each split."""

    train: int
    valid: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.valid + self.test

    def as_dict(self) -> dict[str, int]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


def _validate_ratio(ratio: Sequence[float]) -> tuple[float, float, float]:
    if len(ratio) != 3:
        raise DatasetSchemaError(
            f"split ratio must have exactly 3 parts (train, valid, test), got {len(ratio)}"
        )
    if any(part < 0 for part in ratio):
        raise DatasetSchemaError(f"split ratio parts must be non-negative, got {tuple(ratio)}")
    if ratio[0] <= 0 or ratio[1] <= 0:
        raise DatasetSchemaError(
            f"split ratio must allocate a positive share to train and valid, got {tuple(ratio)}"
        )
    total = sum(ratio)
    if abs(total - 1.0) > _RATIO_TOLERANCE:
        raise DatasetSchemaError(f"split ratio must sum to 1.0, got {total}")
    return (float(ratio[0]), float(ratio[1]), float(ratio[2]))


def allocate(total: int, ratio: Sequence[float] = DEFAULT_SPLIT) -> SplitCounts:
    """Split ``total`` records into counts that sum exactly to ``total``.

    Uses largest-remainder allocation, then guarantees that every split with a
    positive ratio share receives at least one record.

    Raises:
        DatasetSchemaError: if the ratio is invalid or ``total`` is too small.
    """
    shares = _validate_ratio(ratio)
    wanted = 3 if shares[2] > 0 else 2
    if total < wanted:
        raise DatasetSchemaError(
            f"need at least {wanted} records to build {wanted} non-empty splits, got {total}"
        )

    exact = [total * share for share in shares]
    counts = [int(value) for value in exact]
    remainder = total - sum(counts)
    # Hand the leftover records to whichever splits were rounded down hardest.
    order = sorted(range(3), key=lambda i: exact[i] - counts[i], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1

    # Borrow from the largest split for any share that rounded to zero.
    for index in range(wanted):
        if counts[index] == 0:
            donor = max(range(3), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[index] += 1

    return SplitCounts(train=counts[0], valid=counts[1], test=counts[2])


def split_records(
    records: Iterable[Mapping[str, Any]],
    *,
    ratio: Sequence[float] = DEFAULT_SPLIT,
    seed: int = 42,
    shuffle: bool = True,
) -> dict[str, list[Mapping[str, Any]]]:
    """Partition records into ``train``/``valid``/``test``.

    Args:
        records: Any iterable; materialised because splitting needs the length.
        ratio: Fractions summing to 1.0.
        seed: Shuffle seed. The same seed and input always give the same split.
        shuffle: Set False only when the input order is already meaningful,
            for example a chronological holdout.

    Returns:
        Mapping of split name to records. ``test`` may be empty if its ratio is 0.
    """
    items = list(records)
    counts = allocate(len(items), ratio)

    if shuffle:
        # A local Random keeps the global seed untouched, so calling this does
        # not perturb a caller's own sampling. Not a security context: the point
        # is a reproducible partition, which is exactly what a seeded PRNG gives.
        random.Random(seed).shuffle(items)  # noqa: S311

    train_end = counts.train
    valid_end = train_end + counts.valid
    return {
        "train": items[:train_end],
        "valid": items[train_end:valid_end],
        "test": items[valid_end : valid_end + counts.test],
    }


def write_splits(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    ratio: Sequence[float] = DEFAULT_SPLIT,
    seed: int = 42,
    shuffle: bool = True,
) -> SplitCounts:
    """Split records and write ``train.jsonl``/``valid.jsonl``/``test.jsonl``.

    Every record is validated before anything is written, so a malformed
    dataset fails without leaving a half-built directory behind.
    """
    from .io import write_jsonl
    from .schema import validate_records

    items = [dict(record) for record in records]
    validate_records(items, source="input dataset")

    splits = split_records(items, ratio=ratio, seed=seed, shuffle=shuffle)
    target = Path(output_dir)
    written: dict[str, int] = {}
    for filename in SPLIT_FILENAMES:
        name = filename.removesuffix(".jsonl")
        rows = splits[name]
        if not rows and name == "test":
            continue
        written[name] = write_jsonl(target / filename, rows)

    return SplitCounts(
        train=written.get("train", 0),
        valid=written.get("valid", 0),
        test=written.get("test", 0),
    )


__all__ = [
    "DEFAULT_SPLIT",
    "MIN_RECORDS",
    "SplitCounts",
    "allocate",
    "split_records",
    "write_splits",
]
