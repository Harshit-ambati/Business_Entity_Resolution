"""Temporary data readers — replace with Thulasi's ber.data when available.

Streaming UTF-8 TSV readers for source and truth files, following the
contract in docs/CONTRACTS.md. Validates headers and entity ID prefixes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .contracts import Record, TruthRow

_SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
_TRUTH_HEADER = ["source1_entity_id", "matched_entity_ids"]


def read_source(path: str | Path, expected_prefix: str) -> Iterator[Record]:
    """Yield Record objects from a source TSV file in input order.

    Validates the header and every entity ID prefix. Closes the file
    when the iterator is exhausted or closed.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != _SOURCE_HEADER:
            raise ValueError(
                f"Unexpected header in {path}: {reader.fieldnames!r}, "
                f"expected {_SOURCE_HEADER!r}"
            )
        for row in reader:
            entity_id = row["entity_id"]
            if not entity_id.startswith(expected_prefix):
                raise ValueError(
                    f"Expected prefix {expected_prefix!r} for entity ID "
                    f"{entity_id!r} in {path}"
                )
            yield Record(
                entity_id=entity_id,
                business_name=row["business_name"] or "",
                business_address=row.get("business_address") or "",
                country=row.get("country") or "",
            )


def read_truth(path: str | Path) -> Iterator[TruthRow]:
    """Yield TruthRow objects from a truth TSV file in input order.

    An empty ``matched_entity_ids`` cell yields an empty tuple (zero matches).
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != _TRUTH_HEADER:
            raise ValueError(
                f"Unexpected header in {path}: {reader.fieldnames!r}, "
                f"expected {_TRUTH_HEADER!r}"
            )
        for row in reader:
            matched = row.get("matched_entity_ids") or ""
            ids = tuple(m for m in matched.split(",") if m) if matched else ()
            yield TruthRow(
                source1_entity_id=row["source1_entity_id"],
                matched_entity_ids=ids,
            )
