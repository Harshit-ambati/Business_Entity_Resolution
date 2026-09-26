"""Stable command surface; batch stages await teammate pipeline modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


COMMANDS = ("index", "train", "evaluate", "predict", "validate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        stage = subparsers.add_parser(command, help=f"{command} stage")
        stage.add_argument("--data-root", type=Path, required=True, help="Directory containing organizer dataset TSVs")
        stage.add_argument("--work-dir", type=Path, required=True, help="Ignored directory for future indexes/models")
        stage.add_argument("--output-dir", type=Path, required=True, help="Ignored directory for future outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        print("train: H1 pair model API is ready; dataset training awaits merged data, normalization, index and blocking modules", file=sys.stderr)
        return 3
    if args.command == "evaluate":
        print("evaluate: H2 decision and evaluation APIs are ready; real holdout requires merged data, normalization, index and blocking modules", file=sys.stderr)
        return 3
    print(f"{args.command}: Not implemented in H0", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
