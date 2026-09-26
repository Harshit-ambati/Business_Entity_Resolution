"""Streaming TSV readers for challenge sources and truth files (PR T1).

Adheres to CONTRACTS.md:
- read_source(path, expected_prefix) -> Iterator[Record]
- read_truth(path) -> Iterator[TruthRow]
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .contracts import (
    CANDIDATE_PREFIXES,
    SOURCE_PREFIXES,
    Record,
    TruthRow,
    validate_entity_id,
)

SOURCE_HEADERS: tuple[str, ...] = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)
TRUTH_HEADERS: tuple[str, ...] = (
    "source1_entity_id",
    "matched_entity_ids",
)


def read_source(
    path: str | Path,
    expected_prefix: str,
    *,
    check_duplicates: bool = False,
) -> Iterator[Record]:
    """Read a tab-delimited UTF-8 source file and yield Record objects in file order.

    Parameters
    ----------
    path : str | Path
        Path to the source TSV file.
    expected_prefix : str
        Expected entity ID prefix, exactly one of "S1-", "S2-", or "S3-".
    check_duplicates : bool, default False
        If True, track seen entity IDs in memory and raise ValueError on duplicate.
        Kept False by default for streaming multi-million row datasets within 8 GB RAM.
        Full-dataset duplicate audits are available via ber.quality.

    Yields
    ------
    Record
        Validated source record.

    Raises
    ------
    ValueError
        On invalid prefix, missing/wrong header, malformed row, or wrong ID prefix.
    """
    if expected_prefix not in SOURCE_PREFIXES:
        raise ValueError(
            f"Invalid expected_prefix {expected_prefix!r}; must be one of {SOURCE_PREFIXES}"
        )

    file_path = Path(path)
    seen_ids: set[str] | None = set() if check_duplicates else None

    with file_path.open(mode="r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty source file at {file_path}:1") from None

        if tuple(header) != SOURCE_HEADERS:
            raise ValueError(
                f"Invalid source header at {file_path}:1: "
                f"expected {list(SOURCE_HEADERS)}, got {header}"
            )

        for line_num, row in enumerate(reader, start=2):
            if len(row) != 4:
                raise ValueError(
                    f"Malformed row at {file_path}:{line_num}: "
                    f"expected 4 columns, got {len(row)} ({row!r})"
                )

            entity_id, name, address, country = row

            if not (entity_id.startswith(expected_prefix) and len(entity_id) > len(expected_prefix)):
                raise ValueError(
                    f"Invalid entity ID {entity_id!r} at {file_path}:{line_num}; "
                    f"expected prefix {expected_prefix!r} and a nonempty suffix"
                )

            if seen_ids is not None:
                if entity_id in seen_ids:
                    raise ValueError(
                        f"Duplicate entity ID {entity_id!r} at {file_path}:{line_num}"
                    )
                seen_ids.add(entity_id)

            yield Record(
                entity_id=entity_id,
                business_name=name,
                business_address=address,
                country=country,
            )


def read_truth(
    path: str | Path,
    *,
    check_duplicates: bool = True,
) -> Iterator[TruthRow]:
    """Read a tab-delimited UTF-8 truth file and yield TruthRow objects in file order.

    Parameters
    ----------
    path : str | Path
        Path to the ground truth TSV file.
    check_duplicates : bool, default True
        If True, track seen S1 entity IDs and raise ValueError on duplicate row.

    Yields
    ------
    TruthRow
        Validated ground truth row.

    Raises
    ------
    ValueError
        On missing/wrong header, malformed row, invalid S1 ID, duplicate S1 row,
        or invalid/duplicate matched IDs.
    """
    file_path = Path(path)
    seen_s1: set[str] | None = set() if check_duplicates else None

    with file_path.open(mode="r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty truth file at {file_path}:1") from None

        if tuple(header) != TRUTH_HEADERS:
            raise ValueError(
                f"Invalid truth header at {file_path}:1: "
                f"expected {list(TRUTH_HEADERS)}, got {header}"
            )

        for line_num, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(
                    f"Malformed row at {file_path}:{line_num}: "
                    f"expected 2 columns, got {len(row)} ({row!r})"
                )

            s1_id, matched_str = row

            if not (s1_id.startswith("S1-") and len(s1_id) > 3):
                raise ValueError(
                    f"Invalid source1_entity_id {s1_id!r} at {file_path}:{line_num}; "
                    "expected prefix 'S1-' and a nonempty suffix"
                )

            if seen_s1 is not None:
                if s1_id in seen_s1:
                    raise ValueError(
                        f"Duplicate source1_entity_id {s1_id!r} at {file_path}:{line_num}"
                    )
                seen_s1.add(s1_id)

            clean_matched = matched_str.strip()
            if not clean_matched:
                matched_tuple: tuple[str, ...] = ()
            else:
                raw_ids = [item.strip() for item in clean_matched.split(",")]
                for candidate_id in raw_ids:
                    if not candidate_id:
                        raise ValueError(
                            f"Empty matched entity ID in truth list {matched_str!r} at {file_path}:{line_num}"
                        )
                    if not any(
                        candidate_id.startswith(p) and len(candidate_id) > len(p)
                        for p in CANDIDATE_PREFIXES
                    ):
                        raise ValueError(
                            f"Invalid candidate entity ID {candidate_id!r} at {file_path}:{line_num}; "
                            f"expected prefix in {CANDIDATE_PREFIXES} and a nonempty suffix"
                        )
                if len(raw_ids) != len(set(raw_ids)):
                    raise ValueError(
                        f"Duplicate matched IDs in truth row at {file_path}:{line_num}: {matched_str!r}"
                    )
                matched_tuple = tuple(raw_ids)

            yield TruthRow(source1_entity_id=s1_id, matched_entity_ids=matched_tuple)
