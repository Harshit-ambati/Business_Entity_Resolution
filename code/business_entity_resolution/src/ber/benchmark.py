"""Retrieval benchmark helper: measures index build, candidate retrieval, and candidate coverage.

Usage:
    python -m ber.benchmark [--fixtures] [--source1 PATH] [--source2 PATH] [--source3 PATH] [--truth PATH] [--work-dir DIR]

Measures wall time, peak memory (tracemalloc), candidate volume distribution,
reduction ratio, and, if truth is provided, true-link recall and complete coverage.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

from .blocking import iter_candidates
from .contracts import CandidateGroup
from .index import IndexConfig, build_index, open_index

logger = logging.getLogger(__name__)


def run_benchmark(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    truth_path: Path | None = None,
    work_dir: Path | None = None,
    index_config: IndexConfig | None = None,
    blocking_config: IndexConfig | None = None,
) -> dict[str, Any]:
    """Run full candidate retrieval benchmark and return summary statistics."""
    if work_dir is None:
        work_dir = Path("artifacts/blocking_benchmark")
    work_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    # 1. Index Build Benchmark
    tracemalloc.start()
    t0 = time.perf_counter()
    manifest = build_index(
        source2_path=source2_path,
        source3_path=source3_path,
        work_dir=work_dir,
        config=index_config,
    )
    t_index = time.perf_counter() - t0
    _, peak_index_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results["index_time_sec"] = t_index
    results["index_peak_mem_mb"] = peak_index_mem / (1024 * 1024)
    results["total_records"] = manifest.record_count
    total_candidates_pool = manifest.record_count

    blocking_dir = Path(manifest.work_dir)
    disk_bytes = sum(f.stat().st_size for f in blocking_dir.glob("*") if f.is_file())
    results["index_disk_size_bytes"] = disk_bytes
    results["index_disk_size_mb"] = disk_bytes / (1024 * 1024)

    # 2. Candidate Retrieval Benchmark
    index_store = open_index(manifest)

    tracemalloc.start()
    t1 = time.perf_counter()
    groups: list[CandidateGroup] = list(
        iter_candidates(
            source1_path=source1_path,
            index_store=index_store,
            config=blocking_config,
        )
    )
    t_retrieval = time.perf_counter() - t1
    _, peak_retrieval_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results["retrieval_time_sec"] = t_retrieval
    results["retrieval_peak_mem_mb"] = peak_retrieval_mem / (1024 * 1024)
    results["num_s1_queries"] = len(groups)

    # Candidate statistics
    candidate_counts = [len(g.candidates) for g in groups]
    total_pairs = sum(candidate_counts)
    cartesian_product = len(groups) * total_candidates_pool

    results["total_candidate_pairs"] = total_pairs
    results["candidate_count_mean"] = statistics.mean(candidate_counts) if candidate_counts else 0.0
    results["candidate_count_median"] = statistics.median(candidate_counts) if candidate_counts else 0.0
    results["candidate_count_min"] = min(candidate_counts) if candidate_counts else 0
    results["candidate_count_max"] = max(candidate_counts) if candidate_counts else 0
    results["reduction_ratio"] = 1.0 - (total_pairs / cartesian_product) if cartesian_product > 0 else 0.0

    # 3. Truth Metrics (if provided)
    if truth_path and truth_path.exists():
        truth_map: dict[str, set[str]] = {}
        with open(truth_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            id_idx = header.index("source1_entity_id")
            match_idx = header.index("matched_entity_ids")
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(id_idx, match_idx):
                    continue
                s1_id = parts[id_idx]
                matches = [m.strip() for m in parts[match_idx].split(",") if m.strip()]
                truth_map[s1_id] = set(matches)

        total_true_edges = sum(len(matches) for matches in truth_map.values())
        retrieved_true_edges = 0
        complete_s1_covered = 0
        total_matched_s1 = sum(1 for m in truth_map.values() if len(m) > 0)

        group_map = {g.source1_entity_id: {c.candidate_entity_id for c in g.candidates} for g in groups}

        for s1_id, true_matches in truth_map.items():
            if not true_matches:
                continue
            retrieved = group_map.get(s1_id, set())
            hits = true_matches.intersection(retrieved)
            retrieved_true_edges += len(hits)
            if hits == true_matches:
                complete_s1_covered += 1

        results["true_edge_recall"] = retrieved_true_edges / total_true_edges if total_true_edges > 0 else 1.0
        results["complete_link_coverage"] = complete_s1_covered / total_matched_s1 if total_matched_s1 > 0 else 1.0
        results["total_true_edges"] = total_true_edges
        results["retrieved_true_edges"] = retrieved_true_edges

    return results


def print_report(results: dict[str, Any]) -> None:
    """Pretty-print benchmark results."""
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION — RETRIEVAL BENCHMARK")
    print(f"Total Indexed Records:   {results.get('total_records', 0):,}")
    print(f"S1 Queries:               {results.get('num_s1_queries', 0):,}")
    print("-" * 60)
    print(f"Index Build Time:         {results.get('index_time_sec', 0.0):.4f}s")
    print(f"Index Peak Memory:        {results.get('index_peak_mem_mb', 0.0):.2f} MB")
    print(f"Index Disk Footprint:     {results.get('index_disk_size_mb', 0.0):.2f} MB")
    print(f"Retrieval Time:           {results.get('retrieval_time_sec', 0.0):.4f}s")
    print(f"Retrieval Peak Memory:    {results.get('retrieval_peak_mem_mb', 0.0):.2f} MB")
    print("-" * 60)
    print(f"Total Candidate Pairs:    {results.get('total_candidate_pairs', 0):,}")
    print(f"Mean Candidates/S1:       {results.get('candidate_count_mean', 0.0):.2f}")
    print(f"Median Candidates/S1:     {results.get('candidate_count_median', 0.0):.1f}")
    print(f"Min / Max Candidates/S1:  {results.get('candidate_count_min', 0)} / {results.get('candidate_count_max', 0)}")
    print(f"Reduction Ratio:          {results.get('reduction_ratio', 0.0):.6f}")

    if "true_edge_recall" in results:
        print("-" * 60)
        print(f"True Edge Recall:         {results['true_edge_recall'] * 100:.2f}% ({results['retrieved_true_edges']}/{results['total_true_edges']})")
        print(f"Complete Link Coverage:   {results['complete_link_coverage'] * 100:.2f}%")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Candidate retrieval benchmark")
    parser.add_argument("--source1", type=Path, help="Path to source1.tsv")
    parser.add_argument("--source2", type=Path, help="Path to source2.tsv")
    parser.add_argument("--source3", type=Path, help="Path to source3.tsv")
    parser.add_argument("--truth", type=Path, default=None, help="Path to truth.tsv (optional)")
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory to store intermediate index")
    parser.add_argument("--fixtures", action="store_true", help="Use built-in synthetic test fixtures")

    args = parser.parse_args(argv)

    if args.fixtures or (not args.source1 and not args.source2 and not args.source3):
        fixture_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
        s1 = fixture_dir / "source1.tsv"
        s2 = fixture_dir / "source2.tsv"
        s3 = fixture_dir / "source3.tsv"
        truth = fixture_dir / "truth.tsv"
        work_dir = Path("artifacts/fixtures_blocking")
    else:
        if not (args.source1 and args.source2 and args.source3):
            parser.error("Must provide --source1, --source2, and --source3, or pass --fixtures")
        s1 = args.source1
        s2 = args.source2
        s3 = args.source3
        truth = args.truth
        work_dir = args.work_dir

    results = run_benchmark(
        source1_path=s1,
        source2_path=s2,
        source3_path=s3,
        truth_path=truth,
        work_dir=work_dir,
    )
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
