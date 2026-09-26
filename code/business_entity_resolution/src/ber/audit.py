"""CLI helper for streaming challenge dataset audits (PR T3)."""

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
    args = parser.parse_args(argv)

    report = audit_dataset(args.data_root, output_path=args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
