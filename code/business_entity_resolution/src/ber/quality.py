"""Data quality validation, duplicate checks, and streaming audit helper (PR T1/T3).

Provides streaming quality checks that run within an 8 GB RAM budget:
- validate_source_file
- validate_truth_file
- check_duplicate_ids_partitioned
- audit_dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    CANDIDATE_PREFIXES,
    SOURCE_PREFIXES,
)
from .data import SOURCE_HEADERS, TRUTH_HEADERS


@dataclass(frozen=True, slots=True)
class QualityIssue:
    file_path: str
    line_number: int
    issue_type: str
    message: str


@dataclass(slots=True)
class SourceQualityReport:
    file_path: str
    file_size_bytes: int = 0
    total_rows: int = 0
    valid_rows: int = 0
    malformed_rows: int = 0
    prefix_errors: int = 0
    missing_names: int = 0
    empty_addresses: int = 0
    missing_countries: int = 0
    duplicate_ids: int = 0
    country_counts: dict[str, int] = field(default_factory=dict)
    script_counts: dict[str, int] = field(default_factory=dict)
    errors: list[QualityIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.malformed_rows == 0 and self.prefix_errors == 0 and self.duplicate_ids == 0


@dataclass(slots=True)
class TruthQualityReport:
    file_path: str
    file_size_bytes: int = 0
    total_rows: int = 0
    valid_rows: int = 0
    malformed_rows: int = 0
    prefix_errors: int = 0
    duplicate_s1_ids: int = 0
    duplicate_matched_ids: int = 0
    singleton_rows: int = 0
    single_match_rows: int = 0
    multi_match_rows: int = 0
    total_links: int = 0
    s2_links: int = 0
    s3_links: int = 0
    match_length_histogram: dict[int, int] = field(default_factory=dict)
    errors: list[QualityIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            self.malformed_rows == 0
            and self.prefix_errors == 0
            and self.duplicate_s1_ids == 0
            and self.duplicate_matched_ids == 0
        )


def _detect_script(text: str) -> str:
    """Classify the dominant script of a string for multilingual auditing."""
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "Devanagari"
        if 0x0C00 <= code <= 0x0C7F:
            return "Telugu"
        if 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF:
            return "CJK"
        if 0x0600 <= code <= 0x06FF:
            return "Arabic"
        if 0x0400 <= code <= 0x04FF:
            return "Cyrillic"
    return "Latin/Other"


def validate_source_file(
    path: str | Path,
    expected_prefix: str,
    *,
    check_duplicates: bool = True,
    max_errors: int = 50,
) -> SourceQualityReport:
    """Stream a source TSV file and audit headers, columns, prefixes, and missingness.

    Parameters
    ----------
    path : str | Path
        Path to the source TSV.
    expected_prefix : str
        Expected entity ID prefix ('S1-', 'S2-', 'S3-').
    check_duplicates : bool, default True
        If True, check for duplicate IDs (uses in-memory set; for files > 5M rows
        use check_duplicate_ids_partitioned to stay within 8 GB).
    max_errors : int, default 50
        Maximum individual error diagnostics to collect.
    """
    file_path = Path(path)
    report = SourceQualityReport(file_path=str(file_path))

    if not file_path.exists():
        report.errors.append(
            QualityIssue(str(file_path), 0, "file_not_found", f"File does not exist: {file_path}")
        )
        return report

    report.file_size_bytes = file_path.stat().st_size
    seen_ids: set[str] | None = set() if check_duplicates else None

    with file_path.open(mode="r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            report.errors.append(
                QualityIssue(str(file_path), 1, "empty_file", "File is empty")
            )
            return report

        if tuple(header) != SOURCE_HEADERS:
            report.malformed_rows += 1
            if len(report.errors) < max_errors:
                report.errors.append(
                    QualityIssue(
                        str(file_path),
                        1,
                        "header_mismatch",
                        f"Expected header {list(SOURCE_HEADERS)}, got {header}",
                    )
                )

        for line_num, row in enumerate(reader, start=2):
            report.total_rows += 1
            if len(row) != 4:
                report.malformed_rows += 1
                if len(report.errors) < max_errors:
                    report.errors.append(
                        QualityIssue(
                            str(file_path),
                            line_num,
                            "column_count",
                            f"Expected 4 columns, got {len(row)}: {row!r}",
                        )
                    )
                continue

            entity_id, name, address, country = row

            # Check prefix and non-empty suffix
            if not (entity_id.startswith(expected_prefix) and len(entity_id) > len(expected_prefix)):
                report.prefix_errors += 1
                if len(report.errors) < max_errors:
                    report.errors.append(
                        QualityIssue(
                            str(file_path),
                            line_num,
                            "prefix_error",
                            f"Entity ID {entity_id!r} does not match expected prefix {expected_prefix!r}",
                        )
                    )

            # Check duplicate ID
            if seen_ids is not None:
                if entity_id in seen_ids:
                    report.duplicate_ids += 1
                    if len(report.errors) < max_errors:
                        report.errors.append(
                            QualityIssue(
                                str(file_path),
                                line_num,
                                "duplicate_id",
                                f"Duplicate entity ID {entity_id!r}",
                            )
                        )
                else:
                    seen_ids.add(entity_id)

            # Check missingness (preserving raw rows)
            if not name or not name.strip():
                report.missing_names += 1

            if not address or not address.strip():
                report.empty_addresses += 1

            if not country or not country.strip():
                report.missing_countries += 1
            else:
                c_norm = country.strip().title()
                report.country_counts[c_norm] = report.country_counts.get(c_norm, 0) + 1

            # Check script
            script = _detect_script(name)
            report.script_counts[script] = report.script_counts.get(script, 0) + 1

            report.valid_rows += 1

    return report


def validate_truth_file(
    path: str | Path,
    *,
    check_duplicates: bool = True,
    max_errors: int = 50,
) -> TruthQualityReport:
    """Stream a ground truth TSV file and audit headers, IDs, and match statistics.

    Parameters
    ----------
    path : str | Path
        Path to the truth TSV.
    check_duplicates : bool, default True
        If True, check for duplicate S1 IDs.
    max_errors : int, default 50
        Maximum individual error diagnostics to collect.
    """
    file_path = Path(path)
    report = TruthQualityReport(file_path=str(file_path))

    if not file_path.exists():
        report.errors.append(
            QualityIssue(str(file_path), 0, "file_not_found", f"File does not exist: {file_path}")
        )
        return report

    report.file_size_bytes = file_path.stat().st_size
    seen_s1: set[str] | None = set() if check_duplicates else None

    with file_path.open(mode="r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            report.errors.append(
                QualityIssue(str(file_path), 1, "empty_file", "File is empty")
            )
            return report

        if tuple(header) != TRUTH_HEADERS:
            report.malformed_rows += 1
            if len(report.errors) < max_errors:
                report.errors.append(
                    QualityIssue(
                        str(file_path),
                        1,
                        "header_mismatch",
                        f"Expected header {list(TRUTH_HEADERS)}, got {header}",
                    )
                )

        for line_num, row in enumerate(reader, start=2):
            report.total_rows += 1
            if len(row) != 2:
                report.malformed_rows += 1
                if len(report.errors) < max_errors:
                    report.errors.append(
                        QualityIssue(
                            str(file_path),
                            line_num,
                            "column_count",
                            f"Expected 2 columns, got {len(row)}: {row!r}",
                        )
                    )
                continue

            s1_id, matched_str = row

            if not (s1_id.startswith("S1-") and len(s1_id) > 3):
                report.prefix_errors += 1
                if len(report.errors) < max_errors:
                    report.errors.append(
                        QualityIssue(
                            str(file_path),
                            line_num,
                            "prefix_error",
                            f"S1 ID {s1_id!r} does not match prefix 'S1-'",
                        )
                    )

            if seen_s1 is not None:
                if s1_id in seen_s1:
                    report.duplicate_s1_ids += 1
                    if len(report.errors) < max_errors:
                        report.errors.append(
                            QualityIssue(
                                str(file_path),
                                line_num,
                                "duplicate_s1",
                                f"Duplicate source1_entity_id {s1_id!r}",
                            )
                        )
                else:
                    seen_s1.add(s1_id)

            clean_matched = matched_str.strip()
            if not clean_matched:
                report.singleton_rows += 1
                match_count = 0
            else:
                raw_ids = [item.strip() for item in clean_matched.split(",")]
                match_count = len(raw_ids)

                has_internal_dups = len(raw_ids) != len(set(raw_ids))
                if has_internal_dups:
                    report.duplicate_matched_ids += 1
                    if len(report.errors) < max_errors:
                        report.errors.append(
                            QualityIssue(
                                str(file_path),
                                line_num,
                                "duplicate_matched_id",
                                f"Duplicate IDs in match list: {matched_str!r}",
                            )
                        )

                for candidate_id in raw_ids:
                    if not any(
                        candidate_id.startswith(p) and len(candidate_id) > len(p)
                        for p in CANDIDATE_PREFIXES
                    ):
                        report.prefix_errors += 1
                        if len(report.errors) < max_errors:
                            report.errors.append(
                                QualityIssue(
                                    str(file_path),
                                    line_num,
                                    "candidate_prefix_error",
                                    f"Candidate ID {candidate_id!r} not in {CANDIDATE_PREFIXES}",
                                )
                            )
                    elif candidate_id.startswith("S2-"):
                        report.s2_links += 1
                    elif candidate_id.startswith("S3-"):
                        report.s3_links += 1

                if match_count == 1:
                    report.single_match_rows += 1
                else:
                    report.multi_match_rows += 1

                report.total_links += match_count

            report.match_length_histogram[match_count] = (
                report.match_length_histogram.get(match_count, 0) + 1
            )
            report.valid_rows += 1

    return report


def check_duplicate_ids_partitioned(
    path: str | Path,
    id_column: int = 0,
    *,
    num_partitions: int = 64,
    temp_dir: Path | None = None,
) -> list[str]:
    """Check for duplicate IDs in a large TSV using memory-bounded disk hash partitioning.

    This ensures that auditing 10M to 24M records never exceeds the 8 GB RAM target.

    Parameters
    ----------
    path : str | Path
        Path to the TSV file.
    id_column : int, default 0
        0-based index of the ID column.
    num_partitions : int, default 64
        Number of disk buckets for partitioning.
    temp_dir : Path | None
        Directory for partition scratch files (cleaned up automatically).

    Returns
    -------
    list[str]
        List of duplicate IDs found across the file.
    """
    file_path = Path(path)
    with tempfile.TemporaryDirectory(dir=temp_dir) as scratch:
        scratch_path = Path(scratch)
        partition_files = [
            open(scratch_path / f"part_{i}.txt", "w", encoding="utf-8")
            for i in range(num_partitions)
        ]

        try:
            with file_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, delimiter="\t")
                _ = next(reader, None)  # skip header
                for row in reader:
                    if len(row) > id_column:
                        entity_id = row[id_column]
                        # Hash partition
                        h = int(hashlib.md5(entity_id.encode("utf-8")).hexdigest()[:4], 16)
                        bucket = h % num_partitions
                        partition_files[bucket].write(entity_id + "\n")
        finally:
            for pf in partition_files:
                pf.close()

        # Check each bucket independently in memory
        duplicates: list[str] = []
        for i in range(num_partitions):
            bucket_file = scratch_path / f"part_{i}.txt"
            seen: set[str] = set()
            with bucket_file.open("r", encoding="utf-8") as stream:
                for line in stream:
                    eid = line.rstrip("\n")
                    if eid in seen:
                        if eid not in duplicates:
                            duplicates.append(eid)
                    else:
                        seen.add(eid)

    return duplicates


# Files that must exist when they are present in the archive baseline.
# Any name in this set that cannot be found (directly or in a sub-directory)
# raises FileNotFoundError so the command exits non-zero.
_REQUIRED_CHALLENGE_FILES: frozenset[str] = frozenset({
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
})

# Alternative flat-layout names accepted when the canonical names are absent.
# These are *never* required; they exist only for developer convenience.
_OPTIONAL_ALT_FILES: dict[str, tuple[str | None, str]] = {
    "source1.tsv": ("S1-", "source"),
    "source2.tsv": ("S2-", "source"),
    "source3.tsv": ("S3-", "source"),
    "truth.tsv": (None, "truth"),
}

_ALL_DATASET_FILES: dict[str, tuple[str | None, str]] = {
    "train_source1.tsv": ("S1-", "source"),
    "train_source2.tsv": ("S2-", "source"),
    "train_source3.tsv": ("S3-", "source"),
    "train_ground_truth.tsv": (None, "truth"),
    "test_source1.tsv": ("S1-", "source"),
    "test_source2.tsv": ("S2-", "source"),
    "test_source3.tsv": ("S3-", "source"),
    **_OPTIONAL_ALT_FILES,
}


def audit_dataset(
    data_root: str | Path,
    *,
    output_path: Path | None = None,
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    """Stream and audit all available source and truth files under a dataset directory.

    Reports row counts, country distributions, missingness, truth link distributions,
    file sizes, and execution time.

    Duplicate-ID checks use disk-partitioned hashing (``check_duplicate_ids_partitioned``)
    so that auditing 10-24 million records stays within the 8 GB RAM budget.  The result
    records ``duplicate_check_method: "partitioned_disk"`` for every source file so the
    caller can verify which checker was used.

    Parameters
    ----------
    data_root : str | Path
        Directory containing the challenge TSV files.
    output_path : Path | None
        Optional path to write the JSON report to.
    temp_dir : Path | None
        Directory for disk-partition scratch files used by
        ``check_duplicate_ids_partitioned``.  Defaults to the OS temp directory.
        Set this explicitly when the OS temp directory is low on space.

    Raises
    ------
    FileNotFoundError
        If any required challenge file (train/test source and truth TSVs) cannot be
        located directly under *data_root* or in any of its sub-directories.  Missing
        challenge files must never produce a silent exit-0 with an empty ``"files"`` map.
    """
    root = Path(data_root)
    start_time = time.perf_counter()

    results: dict[str, Any] = {
        "data_root": str(root),
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {},
    }

    for filename, (prefix, ftype) in _ALL_DATASET_FILES.items():
        # Resolve the file path, searching sub-directories as a fallback.
        file_path = root / filename
        if not file_path.exists():
            candidates = list(root.glob(f"**/{filename}"))
            if candidates:
                file_path = candidates[0]
            elif filename in _REQUIRED_CHALLENGE_FILES:
                # Required challenge file is missing — this is a hard error.
                raise FileNotFoundError(
                    f"Required challenge file not found: {filename!r} "
                    f"(searched under {root}).  "
                    "Ensure the organizer archive is extracted to --data-root."
                )
            else:
                # Optional alternate-layout file; skip silently.
                continue

        if ftype == "source" and prefix is not None:
            # Basic streaming validation (no in-memory duplicate set for large files).
            rep = validate_source_file(file_path, prefix, check_duplicates=False)

            # Duplicate-ID check via memory-bounded disk partitioning.
            # This is the only method that stays within 8 GB for 10-24 M rows.
            duplicate_ids = check_duplicate_ids_partitioned(
                file_path, id_column=0, temp_dir=temp_dir
            )

            results["files"][filename] = {
                "path": str(file_path),
                "size_bytes": rep.file_size_bytes,
                "total_rows": rep.total_rows,
                "valid_rows": rep.valid_rows,
                "malformed_rows": rep.malformed_rows,
                "missing_names": rep.missing_names,
                "empty_addresses": rep.empty_addresses,
                "missing_countries": rep.missing_countries,
                "countries": rep.country_counts,
                "scripts": rep.script_counts,
                "duplicate_check_method": "partitioned_disk",
                "duplicate_ids_found": len(duplicate_ids),
                "duplicate_id_sample": duplicate_ids[:20],
                "is_valid": rep.is_valid and len(duplicate_ids) == 0,
            }
        elif ftype == "truth":
            rep_truth = validate_truth_file(file_path, check_duplicates=True)
            results["files"][filename] = {
                "path": str(file_path),
                "size_bytes": rep_truth.file_size_bytes,
                "total_rows": rep_truth.total_rows,
                "valid_rows": rep_truth.valid_rows,
                "singleton_rows": rep_truth.singleton_rows,
                "single_match_rows": rep_truth.single_match_rows,
                "multi_match_rows": rep_truth.multi_match_rows,
                "total_links": rep_truth.total_links,
                "s2_links": rep_truth.s2_links,
                "s3_links": rep_truth.s3_links,
                "is_valid": rep_truth.is_valid,
            }

    results["elapsed_seconds"] = round(time.perf_counter() - start_time, 3)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results
