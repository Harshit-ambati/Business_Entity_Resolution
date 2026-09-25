"""
cli.py - Command-line interface for BER pipeline.

Owner: Harshit (integration); Suresh adds evaluate/validate/error-analysis commands.

Suresh-owned commands:
  ber evaluate        - compute macro F0.5 from TSV files
  ber candidates      - compute candidate recall, oracle, and distribution
  ber validate        - preflight-validate output TSVs
  ber error-analysis  - classify and report prediction errors

All commands log: config, commit/version, elapsed time, input/output paths.
Non-zero exit code for validation failure.

NOTE: Do not invoke Harshit's train/predict commands here; they are
implemented in Harshit's integration PR. This file provides the CLI skeleton
and Suresh commands. Harshit will integrate the full pipeline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _git_commit() -> str:
    """Return short git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _log(msg: str) -> None:
    print(f"[BER] {msg}", flush=True)


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"


# ---------------------------------------------------------------------------
# ber evaluate
# ---------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Compute macro F0.5 from ground-truth and prediction TSV files."""
    from ber.metrics import evaluate_from_tsv, evaluate_candidates_from_tsv, oracle_sanity_check

    _log(f"command=evaluate commit={_git_commit()}")
    _log(f"truth={args.truth}")
    _log(f"predictions={args.predictions}")

    t0 = time.time()
    result = evaluate_from_tsv(
        args.truth,
        args.predictions,
        store_per_entity=False,
    )
    elapsed = time.time() - t0

    _log(f"elapsed={_fmt(elapsed)}")
    print()
    print("=== Evaluation Results ===")
    print(f"  Macro F0.5:          {result.macro_f0_5:.6f}")
    print(f"  Mean Precision:      {result.mean_precision:.6f}")
    print(f"  Mean Recall:         {result.mean_recall:.6f}")
    print(f"  Evaluated S1:        {result.evaluated_s1_count}")
    print(f"  True-empty count:    {result.true_empty_count}")
    print(f"  Predicted-empty:     {result.predicted_empty_count}")
    print(f"  False positive IDs:  {result.false_positive_count}")
    print(f"  False negative IDs:  {result.false_negative_count}")
    print(f"  Singleton accuracy:  {result.singleton_accuracy:.4f} "
          f"({result.singleton_correct}/{result.singleton_total})")

    if result.country_f0_5:
        print("  Country slices (labeled validation data only):")
        for country, score in sorted(result.country_f0_5.items()):
            print(f"    {country}: {score:.6f}")

    # Oracle sanity check when candidates file is provided
    if args.candidates:
        _log(f"candidates={args.candidates}")
        cand_result = evaluate_candidates_from_tsv(args.truth, args.candidates)
        passed, msg = oracle_sanity_check(result, cand_result)
        print()
        print(msg)
        if not passed:
            return 1

    return 0


# ---------------------------------------------------------------------------
# ber candidates
# ---------------------------------------------------------------------------


def cmd_candidates(args: argparse.Namespace) -> int:
    """Compute candidate retrieval quality metrics."""
    from ber.metrics import evaluate_candidates_from_tsv

    _log(f"command=candidates commit={_git_commit()}")
    _log(f"truth={args.truth}")
    _log(f"candidates={args.candidates}")

    corpus_size = args.corpus_size

    t0 = time.time()
    result = evaluate_candidates_from_tsv(
        args.truth,
        args.candidates,
        corpus_size=corpus_size,
    )
    elapsed = time.time() - t0

    _log(f"elapsed={_fmt(elapsed)}")
    print()
    print("=== Candidate Diagnostics ===")
    print(f"  Evaluated S1:          {result.evaluated_s1_count}")
    print(f"  Total true edges:      {result.total_true_edges}")
    print(f"  True-edge recall:      {result.true_edge_recall:.6f}")
    print(f"  S2 true-edge recall:   {result.s2_true_edge_recall}")
    print(f"  S3 true-edge recall:   {result.s3_true_edge_recall}")
    print(f"  Complete-S1 recall:    {result.complete_s1_recall:.6f}")
    print(f"  Oracle macro F0.5:     {result.oracle_macro_f0_5:.6f}")
    print(f"  Total candidate pairs: {result.total_candidate_pairs}")
    print(f"  Min candidates/S1:     {result.min_candidates}")
    print(f"  Mean candidates/S1:    {result.mean_candidates:.2f}")
    print(f"  Median candidates/S1:  {result.median_candidates:.1f}")
    print(f"  P95 candidates/S1:     {result.p95_candidates:.1f}")
    print(f"  P99 candidates/S1:     {result.p99_candidates:.1f}")
    print(f"  Max candidates/S1:     {result.max_candidates}")
    if result.reduction_ratio is not None:
        print(f"  Reduction ratio:       {result.reduction_ratio:.8f}")
    return 0


# ---------------------------------------------------------------------------
# ber validate
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    """Preflight-validate output TSVs against all formatting rules."""
    from ber.output import validate_outputs, run_organizer_validator

    _log(f"command=validate commit={_git_commit()}")
    _log(f"test_s1={args.test_s1}")
    _log(f"matching={args.matching}")
    _log(f"candidates={args.candidates}")

    t0 = time.time()
    result = validate_outputs(
        test_s1_path=args.test_s1,
        matching_results_path=args.matching,
        candidate_pairs_path=args.candidates,
    )
    elapsed = time.time() - t0

    _log(f"elapsed={_fmt(elapsed)}")
    print()
    print(result.summary())

    # Run organizer validator if requested
    if args.run_organizer:
        print()
        _log("Running organizer validator (utils/validate_submission.py)...")
        _log("NOTE: A PASS means FORMAT correctness only, not ML quality.")
        exit_code, output = run_organizer_validator(
            matching_path=args.matching,
            candidate_path=args.candidates,
            test_dir=os.path.dirname(args.test_s1),
            check_ids=args.check_ids,
        )
        print(output)
        if exit_code != 0:
            _log("Organizer validator: FAIL")
            return 1
        _log("Organizer validator: PASS (format/coverage/consistency only)")

    return 0 if result.passed else 1


# ---------------------------------------------------------------------------
# ber error-analysis
# ---------------------------------------------------------------------------


def cmd_error_analysis(args: argparse.Namespace) -> int:
    """Classify prediction errors into retrieval/scoring/FP categories."""
    from ber.metrics import error_analysis_from_tsv

    _log(f"command=error-analysis commit={_git_commit()}")
    _log(f"truth={args.truth}")
    _log(f"predictions={args.predictions}")
    _log(f"candidates={args.candidates}")

    t0 = time.time()
    report = error_analysis_from_tsv(
        truth_tsv=args.truth,
        predictions_tsv=args.predictions,
        candidates_tsv=args.candidates,
        max_examples=args.max_examples,
    )
    elapsed = time.time() - t0

    _log(f"elapsed={_fmt(elapsed)}")
    print()
    print(report.summary())

    if args.verbose:
        def _show_examples(cat_name: str, cat) -> None:
            if cat.examples:
                print(f"\n  {cat_name} examples (up to {cat.max_examples}):")
                for s1_id, detail in cat.examples:
                    print(f"    S1={s1_id} | {detail}")

        _show_examples("A. Retrieval miss", report.retrieval_miss)
        _show_examples("B. Scoring FN", report.scoring_false_neg)
        _show_examples("C. False positive", report.false_positive)
        _show_examples("D. Wrong singleton", report.wrong_singleton)
        _show_examples("E. Empty-case FP", report.empty_case_fp)
        _show_examples("F. Multi-match partial", report.multi_match_partial)

    return 0


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ber",
        description="Business Entity Resolution CLI (ML Challenge 2026)",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- evaluate ----
    p_eval = sub.add_parser("evaluate", help="Compute macro F0.5")
    p_eval.add_argument("--truth", required=True,
                        help="Path to ground-truth TSV (source1_entity_id, matched_entity_ids)")
    p_eval.add_argument("--predictions", required=True,
                        help="Path to matching_results.tsv")
    p_eval.add_argument("--candidates", default=None,
                        help="Path to candidate_pairs.tsv (for oracle sanity check)")
    p_eval.set_defaults(func=cmd_evaluate)

    # ---- candidates ----
    p_cand = sub.add_parser("candidates", help="Compute candidate recall and oracle")
    p_cand.add_argument("--truth", required=True,
                        help="Path to ground-truth TSV")
    p_cand.add_argument("--candidates", required=True,
                        help="Path to candidate_pairs.tsv")
    p_cand.add_argument("--corpus-size", type=int, default=None,
                        help="Total S2+S3 record count for reduction ratio")
    p_cand.set_defaults(func=cmd_candidates)

    # ---- validate ----
    p_val = sub.add_parser("validate", help="Preflight-validate output TSVs")
    p_val.add_argument("--test-s1", required=True,
                       help="Path to test_source1.tsv")
    p_val.add_argument("--matching", required=True,
                       help="Path to matching_results.tsv")
    p_val.add_argument("--candidates", default=None,
                       help="Path to candidate_pairs.tsv")
    p_val.add_argument("--run-organizer", action="store_true",
                       help="Also run the organizer's validate_submission.py")
    p_val.add_argument("--check-ids", action="store_true",
                       help="Pass --check-ids to organizer validator (memory heavy)")
    p_val.set_defaults(func=cmd_validate)

    # ---- error-analysis ----
    p_err = sub.add_parser("error-analysis", help="Classify prediction errors")
    p_err.add_argument("--truth", required=True,
                       help="Path to ground-truth TSV")
    p_err.add_argument("--predictions", required=True,
                       help="Path to matching_results.tsv")
    p_err.add_argument("--candidates", required=True,
                       help="Path to candidate_pairs.tsv")
    p_err.add_argument("--max-examples", type=int, default=10,
                       help="Max representative examples per error category")
    p_err.add_argument("--verbose", action="store_true",
                       help="Print representative examples")
    p_err.set_defaults(func=cmd_error_analysis)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
