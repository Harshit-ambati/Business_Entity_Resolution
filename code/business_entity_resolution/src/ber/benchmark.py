"""Retrieval benchmark helper: measures index build, candidate retrieval, and candidate coverage.

Usage:
    python -m ber.benchmark [--sample] [--generate-scale N] [--fixtures] [--source1 PATH --source2 PATH --source3 PATH]

Measures wall time, peak process RAM (Working Set / maxrss), Python heap allocations,
candidate volume distribution, reduction ratio, disk footprint, and true-link recall.
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


def get_peak_process_memory_mb() -> float:
    """Return peak process resident memory (Working Set on Windows, maxrss on POSIX) in MB."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            get_mem = psapi.GetProcessMemoryInfo
            get_mem.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            get_mem.restype = wintypes.BOOL

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if get_mem(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / (1024 * 1024)
        except Exception:
            pass
    else:
        try:
            import resource

            rusage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                return rusage.ru_maxrss / (1024 * 1024)
            return rusage.ru_maxrss / 1024
        except Exception:
            pass

    # Fallback to tracemalloc peak if OS memory query fails
    try:
        _, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)
    except Exception:
        return 0.0


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
    peak_index_rss = get_peak_process_memory_mb()

    results["index_time_sec"] = t_index
    results["index_peak_mem_mb"] = peak_index_mem / (1024 * 1024)
    results["index_peak_rss_mb"] = peak_index_rss
    results["total_records"] = manifest.record_count
    total_candidates_pool = manifest.record_count

    blocking_dir = Path(manifest.work_dir)
    disk_bytes = sum(f.stat().st_size for f in blocking_dir.glob("*") if f.is_file())
    results["index_disk_size_bytes"] = disk_bytes
    results["index_disk_size_mb"] = disk_bytes / (1024 * 1024)

    # 2. Candidate Retrieval Benchmark (Streaming)
    index_store = open_index(manifest)

    # Preload truth if provided
    truth_map: dict[str, set[str]] = {}
    if truth_path and truth_path.exists():
        with open(truth_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            id_idx = header.index("source1_entity_id") if "source1_entity_id" in header else 0
            match_idx = header.index("matched_entity_ids") if "matched_entity_ids" in header else 1
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(id_idx, match_idx):
                    continue
                s1_id = parts[id_idx]
                matches = [m.strip() for m in parts[match_idx].split(",") if m.strip()]
                truth_map[s1_id] = set(matches)

    total_true_edges = sum(len(matches) for matches in truth_map.values())
    total_matched_s1 = sum(1 for m in truth_map.values() if len(m) > 0)
    retrieved_true_edges = 0
    complete_s1_covered = 0

    candidate_counts: list[int] = []
    total_pairs = 0
    num_s1 = 0

    tracemalloc.start()
    t1 = time.perf_counter()

    for group in iter_candidates(
        source1_path=source1_path,
        index_store=index_store,
        config=blocking_config,
    ):
        num_s1 += 1
        count = len(group.candidates)
        total_pairs += count
        candidate_counts.append(count)

        if truth_map and group.source1_entity_id in truth_map:
            true_matches = truth_map[group.source1_entity_id]
            if true_matches:
                retrieved_cids = {c.candidate_entity_id for c in group.candidates}
                hits = true_matches.intersection(retrieved_cids)
                retrieved_true_edges += len(hits)
                if hits == true_matches:
                    complete_s1_covered += 1

    t_retrieval = time.perf_counter() - t1
    _, peak_retrieval_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_retrieval_rss = get_peak_process_memory_mb()

    index_store.close()

    results["retrieval_time_sec"] = t_retrieval
    results["retrieval_peak_mem_mb"] = peak_retrieval_mem / (1024 * 1024)
    results["retrieval_peak_rss_mb"] = peak_retrieval_rss
    results["num_s1_queries"] = num_s1

    # Candidate statistics
    cartesian_product = num_s1 * total_candidates_pool

    results["total_candidate_pairs"] = total_pairs
    results["candidate_count_mean"] = statistics.mean(candidate_counts) if candidate_counts else 0.0
    results["candidate_count_median"] = statistics.median(candidate_counts) if candidate_counts else 0.0
    results["candidate_count_min"] = min(candidate_counts) if candidate_counts else 0
    results["candidate_count_max"] = max(candidate_counts) if candidate_counts else 0
    results["reduction_ratio"] = 1.0 - (total_pairs / cartesian_product) if cartesian_product > 0 else 0.0

    if truth_path and truth_path.exists():
        results["true_edge_recall"] = retrieved_true_edges / total_true_edges if total_true_edges > 0 else 1.0
        results["complete_link_coverage"] = complete_s1_covered / total_matched_s1 if total_matched_s1 > 0 else 1.0
        results["total_true_edges"] = total_true_edges
        results["retrieved_true_edges"] = retrieved_true_edges

    return results


def print_report(results: dict[str, Any]) -> None:
    """Pretty-print benchmark results."""
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION — RETRIEVAL BENCHMARK")
    print(f"Total Indexed Records:     {results.get('total_records', 0):,}")
    print(f"S1 Queries:                 {results.get('num_s1_queries', 0):,}")
    print("-" * 60)
    print(f"Index Build Time:           {results.get('index_time_sec', 0.0):.4f}s")
    print(f"Index Peak Process RAM:     {results.get('index_peak_rss_mb', 0.0):.2f} MB")
    print(f"Index Peak Python Heap:     {results.get('index_peak_mem_mb', 0.0):.2f} MB")
    print(f"Index Disk Footprint:       {results.get('index_disk_size_mb', 0.0):.2f} MB")
    print(f"Retrieval Time:             {results.get('retrieval_time_sec', 0.0):.4f}s")
    print(f"Retrieval Peak Process RAM: {results.get('retrieval_peak_rss_mb', 0.0):.2f} MB")
    print(f"Retrieval Peak Python Heap: {results.get('retrieval_peak_mem_mb', 0.0):.2f} MB")
    print("-" * 60)
    print(f"Total Candidate Pairs:      {results.get('total_candidate_pairs', 0):,}")
    print(f"Mean Candidates/S1:         {results.get('candidate_count_mean', 0.0):.2f}")
    print(f"Median Candidates/S1:       {results.get('candidate_count_median', 0.1):.1f}")
    print(f"Min / Max Candidates/S1:    {results.get('candidate_count_min', 0)} / {results.get('candidate_count_max', 0)}")
    print(f"Reduction Ratio:            {results.get('reduction_ratio', 0.0):.6f}")

    if "true_edge_recall" in results:
        print("-" * 60)
        print(f"True Edge Recall:           {results['true_edge_recall'] * 100:.2f}% ({results['retrieved_true_edges']}/{results['total_true_edges']})")
        print(f"Complete Link Coverage:     {results['complete_link_coverage'] * 100:.2f}%")
    print("=" * 60)


def extract_representative_sample(
    archive_path: Path,
    output_dir: Path,
    n_s2_s3: int = 50000,
    n_s1: int = 1000,
) -> tuple[Path, Path, Path, Path]:
    """Extract a representative real challenge benchmark sample from student_resource.zip."""
    import zipfile

    output_dir.mkdir(parents=True, exist_ok=True)
    s1_path = output_dir / "train_source1.tsv"
    s2_path = output_dir / "train_source2.tsv"
    s3_path = output_dir / "train_source3.tsv"
    truth_path = output_dir / "train_ground_truth.tsv"

    if s1_path.exists() and s2_path.exists() and s3_path.exists() and truth_path.exists():
        return s1_path, s2_path, s3_path, truth_path

    half_pool = max(100, n_s2_s3 // 2)

    with zipfile.ZipFile(archive_path) as z:
        s2_ids: set[str] = set()
        with z.open("student_resource/dataset/train/train_source2.tsv") as f_in, \
             open(s2_path, "w", encoding="utf-8") as f_out:
            header = f_in.readline().decode("utf-8")
            f_out.write(header)
            for _ in range(half_pool):
                line = f_in.readline()
                if not line:
                    break
                decoded = line.decode("utf-8")
                s2_ids.add(decoded.split("\t")[0])
                f_out.write(decoded)

        s3_ids: set[str] = set()
        with z.open("student_resource/dataset/train/train_source3.tsv") as f_in, \
             open(s3_path, "w", encoding="utf-8") as f_out:
            header = f_in.readline().decode("utf-8")
            f_out.write(header)
            for _ in range(half_pool):
                line = f_in.readline()
                if not line:
                    break
                decoded = line.decode("utf-8")
                s3_ids.add(decoded.split("\t")[0])
                f_out.write(decoded)

        pool = s2_ids | s3_ids

        target_s1: dict[str, list[str]] = {}
        target_matched = max(10, int(n_s1 * 0.8))
        target_singletons = max(5, n_s1 - target_matched)

        with z.open("student_resource/dataset/train/train_ground_truth.tsv") as f_in:
            header = f_in.readline().decode("utf-8")
            for line in f_in:
                parts = line.decode("utf-8").rstrip("\r\n").split("\t")
                s1_id = parts[0]
                matches = [m.strip() for m in parts[1].split(",") if m.strip()] if len(parts) > 1 else []
                if matches and all(m in pool for m in matches):
                    target_s1[s1_id] = matches
                    if len([k for k, v in target_s1.items() if v]) >= target_matched:
                        break
                elif not matches and len([k for k, v in target_s1.items() if not v]) < target_singletons:
                    target_s1[s1_id] = []

        with open(truth_path, "w", encoding="utf-8") as f_out:
            f_out.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id, matches in target_s1.items():
                f_out.write(f"{s1_id}\t{','.join(matches)}\n")

        s1_needed = set(target_s1.keys())
        s1_found: dict[str, str] = {}
        with z.open("student_resource/dataset/train/train_source1.tsv") as f_in:
            header = f_in.readline().decode("utf-8")
            for line in f_in:
                decoded = line.decode("utf-8")
                sid = decoded.split("\t")[0]
                if sid in s1_needed:
                    s1_found[sid] = decoded
                    if len(s1_found) == len(s1_needed):
                        break

        with open(s1_path, "w", encoding="utf-8") as f_out:
            f_out.write(header)
            for sid in target_s1:
                if sid in s1_found:
                    f_out.write(s1_found[sid])

    return s1_path, s2_path, s3_path, truth_path


def generate_scale_data(output_dir: Path, n_records: int) -> tuple[Path, Path, Path, Path]:
    """Generate a synthetic benchmark corpus of n_records S2/S3 and n_records//10 S1 queries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    s1_path = output_dir / "source1.tsv"
    s2_path = output_dir / "source2.tsv"
    s3_path = output_dir / "source3.tsv"
    truth_path = output_dir / "truth.tsv"

    n_s1 = max(10, n_records // 10)
    with open(s1_path, "w", encoding="utf-8") as f1, open(truth_path, "w", encoding="utf-8") as ft:
        f1.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        ft.write("source1_entity_id\tmatched_entity_ids\n")
        for i in range(n_s1):
            f1.write(f"S1-{i}\tEnterprise {i} Global Services\t{i} Market Street\tIndia\n")
            if i < int(n_s1 * 0.8):
                ft.write(f"S1-{i}\tS2-{i},S3-{i}\n")
            else:
                ft.write(f"S1-{i}\t\n")

    with open(s2_path, "w", encoding="utf-8") as f2:
        f2.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for i in range(n_records):
            f2.write(f"S2-{i}\tEnterprise {i} Global Services LLC\t{i} Market Street\tIndia\n")

    with open(s3_path, "w", encoding="utf-8") as f3:
        f3.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for i in range(n_records):
            f3.write(f"S3-{i}\tEnterprise {i} Solutions\t{i} Market Street Suite 10\tIndia\n")

    return s1_path, s2_path, s3_path, truth_path


def find_archive(explicit_path: Path | None = None) -> Path | None:
    """Locate student_resource.zip in current or parent directories."""
    if explicit_path and explicit_path.exists():
        return explicit_path.resolve()
    cur = Path.cwd().resolve()
    for base in [cur, cur.parent, cur.parent.parent, cur.parent.parent.parent]:
        candidate = base / "data" / "student_resource.zip"
        if candidate.exists():
            return candidate.resolve()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Candidate retrieval benchmark")
    parser.add_argument("--source1", type=Path, help="Path to source1.tsv")
    parser.add_argument("--source2", type=Path, help="Path to source2.tsv")
    parser.add_argument("--source3", type=Path, help="Path to source3.tsv")
    parser.add_argument("--truth", type=Path, default=None, help="Path to truth.tsv (optional)")
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory to store intermediate index")
    parser.add_argument("--fixtures", action="store_true", help="Use built-in synthetic test fixtures")
    parser.add_argument("--sample", action="store_true", help="Extract and benchmark a representative sample from student_resource.zip")
    parser.add_argument("--sample-archive", type=Path, default=None, help="Path to student_resource.zip archive")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of S2/S3 candidate records in sample (default: 50,000)")
    parser.add_argument("--sample-s1", type=int, default=1000, help="Number of S1 queries in sample (default: 1,000)")
    parser.add_argument("--generate-scale", type=int, default=None, help="Generate and benchmark a large synthetic corpus of N records")

    args = parser.parse_args(argv)
    archive_path = find_archive(args.sample_archive)
    repo_root = archive_path.parent.parent if archive_path else Path.cwd().resolve()

    if args.generate_scale:
        work_dir = args.work_dir or (repo_root / "artifacts" / "scale_blocking")
        scale_data_dir = work_dir / f"scale_{args.generate_scale}"
        s1, s2, s3, truth = generate_scale_data(scale_data_dir, args.generate_scale)
    elif args.sample or (not args.source1 and not args.source2 and not args.source3 and not args.fixtures and archive_path is not None):
        if archive_path is None:
            parser.error("Cannot run sample benchmark: student_resource.zip not found.")
        work_dir = args.work_dir or (repo_root / "artifacts" / "blocking" / "sample_index")
        sample_dir = repo_root / "artifacts" / "blocking" / "sample"
        s1, s2, s3, truth = extract_representative_sample(
            archive_path=archive_path,
            output_dir=sample_dir,
            n_s2_s3=args.sample_size,
            n_s1=args.sample_s1,
        )
    elif args.fixtures or (not args.source1 and not args.source2 and not args.source3):
        fixture_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
        s1 = fixture_dir / "source1.tsv"
        s2 = fixture_dir / "source2.tsv"
        s3 = fixture_dir / "source3.tsv"
        truth = fixture_dir / "truth.tsv"
        work_dir = Path("artifacts/fixtures_blocking")
    else:
        if not (args.source1 and args.source2 and args.source3):
            parser.error("Must provide --source1, --source2, and --source3, or pass --fixtures / --sample / --generate-scale")
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
