"""CLI helper for streaming challenge dataset audits (PR T3).

Exit codes
----------
0 : All required files found and all pass validation.
1 : A required challenge file is missing (FileNotFoundError).
2 : One or more files present but contain invalid TSV content
    (bad header, malformed rows, wrong prefix, or duplicate IDs).

Note on peak-RAM claim
----------------------
The audit streams records and uses disk-partitioned duplicate checking,
designed for the 8 GB laptop budget.  The ``duplicate_check_method``
field in the JSON output records which checker ran.  A full run on
the actual challenge archive with a peak-RSS measurement has not yet
been executed; that run and its tracemalloc/resource report belong to
the T3 PR once the extracted TSVs are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ber.quality import audit_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream and audit challenge TSV datasets for data quality (PR T3)"
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing dataset TSVs")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path for the report")
    parser.add_argument(
        "--temp-dir", type=Path, default=None,
        help="Directory for disk-partition scratch files (default: OS temp). "
             "Set this when the OS temp directory is low on space.",
    )
    args = parser.parse_args(argv)

    try:
        report = audit_dataset(args.data_root, output_path=args.output, temp_dir=args.temp_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Exit nonzero when any present file fails content validation so that
    # a CI check or a manual run detects invalid TSV content, not just
    # missing files.  This satisfies the review requirement that invalid
    # TSV contents produce a nonzero exit.
    invalid_files = [
        name
        for name, entry in report.get("files", {}).items()
        if not entry.get("is_valid", True)
    ]
    if invalid_files:
        print(
            f"INVALID: {len(invalid_files)} file(s) failed validation: "
            + ", ".join(invalid_files),
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
