"""Full-scale memory and output validation benchmark for Suresh workstream.

Evaluates memory consumption and runtime on 1,732,544 Source-1 entities (the exact
scale of the challenge test set) to verify strict adherence to the 8 GB machine constraint.

IMPORTANT DISCLAIMERS & SCOPES:
  1. Organizer Validator: Checks submission file formatting, headers, delimiter,
     row counts, non-empty IDs, and prediction-subset-of-candidates constraints.
     It has no ground truth and never measures entity resolution quality or ML accuracy.
  2. Synthetic Holdout for Metric Scaling: Challenge test data has no ground truth labels.
     The candidate oracle and evaluation metrics in this benchmark are computed on a
     synthetic stream to measure memory and runtime scalability under full test volume.
  3. Memory Scope: Process working set and Python traced memory verify that Suresh's output
     writer and evaluation components operate comfortably within the 8 GB machine budget.
     It does not establish memory usage for the upstream candidate generation, feature
     extraction, or model inference pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Generator, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

# Ensure package src is in sys.path
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ber.metrics import evaluate, evaluate_candidates
from ber.output import run_organizer_validator, validate_outputs, write_outputs

try:
    import psutil
except ImportError:
    psutil = None


def find_repo_root() -> Path:
    """Find repository root by walking up ancestors."""
    for parent in THIS_FILE.parents:
        if (parent / ".git").is_dir() or (parent / "student_resource").is_dir():
            return parent
    return SRC_DIR.parent.parent


def get_process_peak_bytes() -> int | None:
    """Return peak process memory in bytes across supported platforms.

    On Windows, uses ``psutil.Process().memory_info().peak_wset``.
    On POSIX (Linux/macOS), uses ``resource.getrusage().ru_maxrss`` which
    gives true peak RSS, even when psutil is installed (psutil's ``rss``
    field is current, not peak).
    """
    if sys.platform == "win32" and psutil is not None:
        info = psutil.Process(os.getpid()).memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            return peak
    # POSIX: resource.getrusage gives true peak RSS (ru_maxrss)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports in KB, macOS in bytes
        return ru * 1024 if sys.platform != "darwin" else ru
    except (ImportError, AttributeError):
        pass
    # Last resort: psutil RSS (current, NOT peak) — caller should be aware
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss
    return None


def get_process_current_rss() -> int | None:
    """Return current process RSS in bytes."""
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss
    return None


def iter_real_test_s1(tsv_path: str) -> Generator[str, None, None]:
    """Stream S1 entity IDs directly from test_source1.tsv."""
    with open(tsv_path, encoding="utf-8", newline="") as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.rstrip("\n")
            if line:
                sid = line.split("\t", 1)[0].strip()
                if sid:
                    yield sid


def get_country_labels_from_s1(tsv_path: str) -> dict[str, str]:
    """Read country mapping from test_source1.tsv for France coverage checking."""
    country_labels = {}
    with open(tsv_path, encoding="utf-8", newline="") as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.rstrip("\n")
            if line:
                parts = line.split("\t")
                if len(parts) >= 4:
                    country_labels[parts[0].strip()] = parts[3].strip()
    return country_labels


def iter_synthetic_s1(count: int) -> Generator[str, None, None]:
    """Stream synthetic S1 entity IDs when dataset is not present."""
    for i in range(count):
        yield f"S1-{i:07d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Suresh full-scale memory and validator benchmark")
    parser.add_argument("--test-s1", type=str, default=None,
                        help="Path to test_source1.tsv (defaults to student_resource/dataset/test/test_source1.tsv)")
    parser.add_argument("--output-dir", type=str, default="artifacts/evaluation",
                        help="Directory to write output benchmark files and reports")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Path to test directory containing test sources for organizer validator")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Force synthetic stream even if real test_source1.tsv exists")
    parser.add_argument("--synthetic-count", type=int, default=1_732_544,
                        help="Number of synthetic S1 entities if real test_source1.tsv is absent or --synthetic is set")
    parser.add_argument("--bench-count", type=int, default=None,
                        help="Number of records for Stage 4 evaluate benchmark (defaults to test S1 count)")
    parser.add_argument("--require-organizer", action="store_true", default=False,
                        help="Fail if organizer validator cannot be run")
    args = parser.parse_args()

    repo_root = find_repo_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine S1 data source
    real_test_s1 = None
    if not args.synthetic:
        real_test_s1 = args.test_s1
        if real_test_s1 is None:
            candidate_path = repo_root / "student_resource" / "dataset" / "test" / "test_source1.tsv"
            if candidate_path.is_file():
                real_test_s1 = str(candidate_path)

    use_real = (not args.synthetic) and real_test_s1 is not None and os.path.isfile(real_test_s1)
    dataset_label = f"Real test_source1 ({real_test_s1})" if use_real else f"Synthetic ({args.synthetic_count:,} S1)"

    print("=" * 75)
    print("SURESH FULL-SCALE MEMORY & VALIDATOR BENCHMARK")
    print("Target Machine Budget:     8 GB RAM")
    print(f"Dataset:                   {dataset_label}")
    print(f"Output Directory:          {output_dir}")
    print("=" * 75)
    print("\n[DISCLAIMER] Organizer validate_submission.py checks format, not ML quality.")
    print("[DISCLAIMER] Test set is unlabeled; evaluation metrics use synthetic streams to test scaling.\n")

    report: dict = {
        "benchmark_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_type": "real_test_dataset" if use_real else "synthetic_generator",
        "target_ram_budget_bytes": 8 * 1024 * 1024 * 1024,
        "disclaimers": [
            "Organizer validator checks submission format/coverage, not matching quality.",
            "Test data has no ground truth; evaluation metrics use synthetic streams for scalability.",
            "Memory measurement covers Suresh's writer and evaluator; full pipeline memory depends on retrieval and model stages."
        ],
        "stages": {},
    }

    # -----------------------------------------------------------------------
    # Stage 1: write_outputs()
    # -----------------------------------------------------------------------
    print("[1/4] Running write_outputs() streaming test...")
    print("  Note: Streaming empty candidate and match sets to verify 1.73M write throughput and format compliance.")

    def make_s1_stream():
        return iter_real_test_s1(real_test_s1) if use_real else iter_synthetic_s1(args.synthetic_count)

    def make_empty_stream():
        for sid in make_s1_stream():
            yield sid, ()

    tracemalloc.start()
    t0 = time.perf_counter()
    m_path, c_path = write_outputs(make_s1_stream(), make_empty_stream(), make_empty_stream(), str(output_dir))
    write_wall = time.perf_counter() - t0
    _, write_peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    write_peak_process = get_process_peak_bytes()
    write_rss = get_process_current_rss()

    m_size = os.path.getsize(m_path)
    c_size = os.path.getsize(c_path)

    print(f"  write_outputs completed in {write_wall:.2f}s")
    print(f"  matching_results.tsv size: {m_size:,} bytes")
    print(f"  candidate_pairs.tsv size:  {c_size:,} bytes")
    print(f"  Peak Python traced memory: {write_peak_traced / (1024 * 1024):.2f} MB")
    if write_peak_process:
        print(f"  Peak process memory:       {write_peak_process / (1024 * 1024):.2f} MB")

    report["stages"]["write_outputs"] = {
        "wall_seconds": round(write_wall, 3),
        "peak_traced_bytes": write_peak_traced,
        "peak_process_bytes": write_peak_process,
        "rss_bytes": write_rss,
        "matching_tsv_bytes": m_size,
        "candidate_tsv_bytes": c_size,
    }

    # -----------------------------------------------------------------------
    # Stage 2: validate_outputs() preflight
    # -----------------------------------------------------------------------
    print("\n[2/4] Running validate_outputs() preflight...")
    tracemalloc.start()
    t0 = time.perf_counter()
    country_labels = None
    if use_real:
        country_labels = get_country_labels_from_s1(real_test_s1)
        v_result = validate_outputs(real_test_s1, m_path, c_path, country_labels=country_labels)
    else:
        dummy_s1_path = str(output_dir / "synthetic_test_s1.tsv")
        with open(dummy_s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for sid in iter_synthetic_s1(args.synthetic_count):
                f.write(f"{sid}\tcompany\taddress\tIndia\n")
        v_result = validate_outputs(dummy_s1_path, m_path, c_path)

    val_wall = time.perf_counter() - t0
    _, val_peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    val_peak_process = get_process_peak_bytes()
    val_rss = get_process_current_rss()

    print(f"  validate_outputs completed in {val_wall:.2f}s")
    print(f"  Result passed:             {v_result.passed}")
    print(f"  Total test S1 evaluated:   {v_result.total_test_s1:,}")
    if v_result.france_total > 0:
        print(f"  France S1 coverage:        {v_result.france_in_output:,} / {v_result.france_total:,}")
    print(f"  Errors count:              {len(v_result.errors)}")
    print(f"  Peak Python traced memory: {val_peak_traced / (1024 * 1024):.2f} MB")
    if val_peak_process:
        print(f"  Peak process memory:       {val_peak_process / (1024 * 1024):.2f} MB")

    report["stages"]["validate_outputs"] = {
        "wall_seconds": round(val_wall, 3),
        "passed": v_result.passed,
        "total_test_s1": v_result.total_test_s1,
        "france_total": v_result.france_total,
        "france_in_output": v_result.france_in_output,
        "errors": [str(e) for e in v_result.errors],
        "peak_traced_bytes": val_peak_traced,
        "peak_process_bytes": val_peak_process,
        "rss_bytes": val_rss,
    }

    if not v_result.passed:
        print(f"ERROR: validate_outputs failed: {v_result.errors[:5]}")
        return 1

    # -----------------------------------------------------------------------
    # Stage 3: Official Organizer validate_submission.py
    # -----------------------------------------------------------------------
    print("\n[3/4] Running organizer utils/validate_submission.py...")
    test_dir_path = Path(args.test_dir) if args.test_dir else repo_root / "student_resource" / "dataset" / "test"
    sr_dir = repo_root / "student_resource"
    organizer_exit = None
    organizer_output = ""
    can_run_organizer = (
        (sr_dir / "utils" / "validate_submission.py").is_file()
        and test_dir_path.is_dir()
        and (use_real or args.test_dir is not None)
    )

    if can_run_organizer:
        organizer_exit, organizer_output = run_organizer_validator(
            matching_path=m_path,
            candidate_path=c_path,
            test_dir=str(test_dir_path),
            check_ids=False,
            student_resource_dir=str(sr_dir),
        )
        if organizer_exit == 0:
            organizer_status_display = "PASS (Format only; not ML quality)"
        else:
            organizer_status_display = f"FAIL (Exit code {organizer_exit})"
        print(f"  Organizer validator result: {organizer_status_display}")
        for line in organizer_output.strip().split("\n"):
            safe_line = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
            print(f"    {safe_line}")
    else:
        if not use_real and args.test_dir is None:
            organizer_status_display = "SKIPPED (synthetic mode; organizer test directory requires real test data)"
            organizer_output = "SKIPPED: running with synthetic stream; organizer validation skipped"
        else:
            organizer_status_display = "SKIPPED (no organizer test directory found; format not verified)"
            organizer_output = "SKIPPED: student_resource test directory not found"
        print(f"  Organizer validator: {organizer_status_display}")
        if args.require_organizer:
            print("  ERROR: --require-organizer specified, but organizer validation cannot run!")

    report["stages"]["organizer_validator"] = {
        "exit_code": organizer_exit,
        "status": organizer_status_display,
        "output": organizer_output,
        "passed": organizer_exit == 0 if organizer_exit is not None else False,
        "skipped": organizer_exit is None,
        "note": "PASS confirms submission format, delimiter, headers, and S1 coverage only (not ML quality).",
    }

    # -----------------------------------------------------------------------
    # Stage 4: evaluate() and evaluate_candidates() memory check
    # -----------------------------------------------------------------------
    eval_count = args.bench_count if args.bench_count is not None else (1_732_544 if use_real else args.synthetic_count)
    print(f"\n[4/4] Running evaluate() & evaluate_candidates() on {eval_count:,} synthetic stream entities...")

    def stream_truth():
        for i in range(eval_count):
            sid = f"S1-{i:07d}"
            links = () if i % 18 == 0 else (f"S2-{i:07d}",)
            yield sid, links

    def stream_cands():
        for i in range(eval_count):
            sid = f"S1-{i:07d}"
            cands = () if i % 18 == 0 else (f"S2-{i:07d}", f"S3-{i:07d}")
            yield sid, cands

    tracemalloc.start()
    t0 = time.perf_counter()
    cand_eval = evaluate_candidates(stream_truth(), stream_cands(), corpus_size=9_969_589)
    cand_eval_wall = time.perf_counter() - t0
    _, cand_peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cand_peak_process = get_process_peak_bytes()
    cand_rss = get_process_current_rss()

    print(f"  evaluate_candidates completed in {cand_eval_wall:.2f}s")
    print(f"  True-edge recall:          {cand_eval.true_edge_recall:.4f}")
    print(f"  Oracle macro F0.5:         {cand_eval.oracle_macro_f0_5:.4f}")
    print(f"  Peak Python traced memory: {cand_peak_traced / (1024 * 1024):.2f} MB")
    if cand_peak_process:
        print(f"  Peak process memory:       {cand_peak_process / (1024 * 1024):.2f} MB")

    tracemalloc.start()
    t0 = time.perf_counter()
    eval_res = evaluate(stream_truth(), stream_cands())
    eval_wall = time.perf_counter() - t0
    _, eval_peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    eval_peak_process = get_process_peak_bytes()
    eval_rss = get_process_current_rss()

    print(f"  evaluate completed in {eval_wall:.2f}s")
    print(f"  Macro F0.5:                {eval_res.macro_f0_5:.4f}")
    print(f"  Singleton accuracy:        {eval_res.singleton_accuracy:.4f}")
    print(f"  Peak Python traced memory: {eval_peak_traced / (1024 * 1024):.2f} MB")
    if eval_peak_process:
        print(f"  Peak process memory:       {eval_peak_process / (1024 * 1024):.2f} MB")

    report["stages"]["evaluate_candidates"] = {
        "wall_seconds": round(cand_eval_wall, 3),
        "true_edge_recall": cand_eval.true_edge_recall,
        "oracle_macro_f0_5": cand_eval.oracle_macro_f0_5,
        "peak_traced_bytes": cand_peak_traced,
        "peak_process_bytes": cand_peak_process,
        "rss_bytes": cand_rss,
    }
    report["stages"]["evaluate"] = {
        "wall_seconds": round(eval_wall, 3),
        "macro_f0_5": eval_res.macro_f0_5,
        "singleton_accuracy": eval_res.singleton_accuracy,
        "peak_traced_bytes": eval_peak_traced,
        "peak_process_bytes": eval_peak_process,
        "rss_bytes": eval_rss,
    }

    # Summary table
    max_peak_traced = max(write_peak_traced, val_peak_traced, cand_peak_traced, eval_peak_traced)
    process_peaks = [p for p in (write_peak_process, val_peak_process, cand_peak_process, eval_peak_process) if p is not None]
    max_peak_process = max(process_peaks) if process_peaks else None

    # Budget verification against 8 GB constraint
    # Only process-level peak memory counts; tracemalloc alone cannot prove
    # the process stays under 8 GB because it excludes C-level allocations,
    # memory-mapped files, and interpreter overhead.
    EIGHT_GB = 8 * 1024 * 1024 * 1024
    if max_peak_process is not None:
        budget_compliant: bool | None = max_peak_process < EIGHT_GB
    else:
        # Process peak unavailable — cannot verify the 8 GB claim
        budget_compliant = None

    report["max_peak_traced_bytes"] = max_peak_traced
    report["max_peak_traced_mb"] = round(max_peak_traced / (1024 * 1024), 2)
    report["max_peak_process_bytes"] = max_peak_process
    report["max_peak_process_mb"] = round(max_peak_process / (1024 * 1024), 2) if max_peak_process is not None else None
    report["8gb_budget_compliant"] = budget_compliant
    report["benchmark_classification"] = "Suresh Component & Memory Scaling Benchmark (Synthetic/Empty)"
    report["release_gate_status"] = "PENDING (Final release gate requires real model outputs and verified organizer validator)"

    report_path = output_dir / "suresh_full_scale_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nBenchmark report written to: {report_path}")

    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY & 8 GB MEMORY VERIFICATION")
    print("=" * 75)
    if report["max_peak_process_mb"]:
        print(f"Max Peak Process Memory:                   {report['max_peak_process_mb']} MB")
    print(f"Max Peak Python Traced Memory:             {report['max_peak_traced_mb']} MB")
    if budget_compliant is True:
        budget_display = "PASS (well below 8 GB limit)"
    elif budget_compliant is False:
        budget_display = "FAIL (exceeded 8 GB process memory budget)"
    else:
        budget_display = "UNKNOWN (process peak memory unavailable; cannot verify 8 GB budget)"
    print(f"8 GB Budget Compliance (Writer/Evaluator): {budget_display}")
    print(f"Preflight Output Validation:               {'PASS' if v_result.passed else 'FAIL'}")
    print(f"Organizer Submission Validator:            {organizer_status_display}")
    print("=" * 75)
    print("NOTICE: This benchmark validates writer/evaluator throughput & memory on synthetic/empty data.")
    print("        It does NOT constitute a final model release gate.")
    print("=" * 75)

    if not v_result.passed:
        return 1
    if organizer_exit is not None and organizer_exit != 0:
        return 1
    if args.require_organizer and organizer_exit is None:
        return 1
    if budget_compliant is False:
        print("\nERROR: Process memory exceeded 8 GB budget.")
        return 1
    if budget_compliant is None:
        print("\nWARNING: Process peak memory could not be measured; 8 GB budget unverified.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
