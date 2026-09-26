"""Retrieval benchmark helper: measures index build, candidate retrieval, and candidate coverage.

Usage:
    python -m ber.benchmark [--sample] [--sample-size N] [--sample-s1 N] [--generate-scale N] [--fixtures]

Measures wall time, peak process RAM (Working Set / maxrss), Python heap allocations,
candidate volume distribution, reduction ratio, disk footprint, and true-link recall.
"""

from __future__ import annotations

import argparse
import json
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


def get_peak_process_memory_mb() -> float | None:
    """Return peak process resident memory (Working Set on Windows, maxrss on POSIX) in MB, or None if unavailable."""
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
                mb = counters.PeakWorkingSetSize / (1024 * 1024)
                if mb > 0:
                    return mb
        except Exception:
            pass
    else:
        try:
            import resource

            rusage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                mb = rusage.ru_maxrss / (1024 * 1024)
            else:
                mb = rusage.ru_maxrss / 1024
            if mb > 0:
                return mb
        except Exception:
            pass

    return None


def run_benchmark(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    truth_path: Path | None = None,
    work_dir: Path | None = None,
    index_config: IndexConfig | None = None,
    blocking_config: IndexConfig | None = None,
    expected_s1: int | None = None,
) -> dict[str, Any]:
    """Run full candidate retrieval benchmark and return summary statistics.

    Rejects the run if expected_s1 is provided and fewer S1 queries are produced.
    """
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

    if expected_s1 is not None and num_s1 < expected_s1:
        raise ValueError(
            f"Benchmark produced {num_s1} S1 queries, fewer than requested {expected_s1}. Rejecting run."
        )

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
    idx_rss = results.get("index_peak_rss_mb")
    idx_rss_str = f"{idx_rss:.2f} MB" if idx_rss is not None else "unknown"
    ret_rss = results.get("retrieval_peak_rss_mb")
    ret_rss_str = f"{ret_rss:.2f} MB" if ret_rss is not None else "unknown"

    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION — RETRIEVAL BENCHMARK")
    print(f"Total Indexed Records:     {results.get('total_records', 0):,}")
    print(f"S1 Queries:                 {results.get('num_s1_queries', 0):,}")
    print("-" * 60)
    print(f"Index Build Time:           {results.get('index_time_sec', 0.0):.4f}s")
    print(f"Index Peak Process RAM:     {idx_rss_str}")
    print(f"Index Peak Python Heap:     {results.get('index_peak_mem_mb', 0.0):.2f} MB")
    print(f"Index Disk Footprint:       {results.get('index_disk_size_mb', 0.0):.2f} MB")
    print(f"Retrieval Time:             {results.get('retrieval_time_sec', 0.0):.4f}s")
    print(f"Retrieval Peak Process RAM: {ret_rss_str}")
    print(f"Retrieval Peak Python Heap: {results.get('retrieval_peak_mem_mb', 0.0):.2f} MB")
    print("-" * 60)
    print(f"Total Candidate Pairs:      {results.get('total_candidate_pairs', 0):,}")
    print(f"Mean Candidates/S1:         {results.get('candidate_count_mean', 0.0):.2f}")
    print(f"Median Candidates/S1:       {results.get('candidate_count_median', 0.0):.1f}")
    print(f"Min / Max Candidates/S1:    {results.get('candidate_count_min', 0)} / {results.get('candidate_count_max', 0)}")
    print(f"Reduction Ratio:            {results.get('reduction_ratio', 0.0):.6f}")

    if "true_edge_recall" in results:
        print("-" * 60)
        print(f"Sample True Edge Recall:    {results['true_edge_recall'] * 100:.2f}% ({results['retrieved_true_edges']}/{results['total_true_edges']}) [sample result, not full holdout]")
        print(f"Sample Link Coverage:       {results['complete_link_coverage'] * 100:.2f}% [sample result, not full holdout]")
    print("=" * 60)


def extract_representative_sample(
    archive_path: Path,
    output_dir: Path,
    n_s2_s3: int = 50000,
    n_s1: int = 1000,
    force: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Extract a representative real challenge benchmark sample from student_resource.zip.

    Ensures sample files depend on archive status and requested sizes, regenerating
    them when sizes or archive change. Rejects runs where produced S1 queries < n_s1.
    """
    archive_path = Path(archive_path).resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    sample_dir = output_dir / f"sample_s2s3_{n_s2_s3}_s1_{n_s1}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    meta_path = sample_dir / "sample_meta.json"

    s1_path = sample_dir / "train_source1.tsv"
    s2_path = sample_dir / "train_source2.tsv"
    s3_path = sample_dir / "train_source3.tsv"
    truth_path = sample_dir / "train_ground_truth.tsv"

    archive_stat = archive_path.stat()
    if (
        not force
        and meta_path.exists()
        and s1_path.exists()
        and s2_path.exists()
        and s3_path.exists()
        and truth_path.exists()
    ):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("requested_n_s2_s3") == n_s2_s3
                and meta.get("requested_n_s1") == n_s1
                and meta.get("actual_s1_count") == n_s1
                and meta.get("archive_size") == archive_stat.st_size
                and meta.get("archive_mtime") == archive_stat.st_mtime
            ):
                logger.info("Using cached sample at %s matching requested sizes", sample_dir)
                return s1_path, s2_path, s3_path, truth_path
        except Exception:
            pass

    logger.info("Extracting representative sample (S1=%d, S2/S3=%d) from %s...", n_s1, n_s2_s3, archive_path)
    n_s2 = n_s2_s3 // 2
    n_s3 = n_s2_s3 - n_s2

    import zipfile

    with zipfile.ZipFile(archive_path) as z:
        # 1. Read requested n_s1 from train_source1.tsv
        s1_rows: list[str] = []
        s1_ids: set[str] = set()
        with z.open("student_resource/dataset/train/train_source1.tsv") as f:
            header_s1 = f.readline().decode("utf-8")
            for _ in range(n_s1):
                line = f.readline()
                if not line:
                    break
                decoded = line.decode("utf-8")
                s1_rows.append(decoded)
                s1_ids.add(decoded.split("\t")[0])

        if len(s1_rows) < n_s1:
            raise ValueError(
                f"Requested {n_s1} S1 queries, but only found {len(s1_rows)} in archive. Rejecting run."
            )

        # 2. Read truth for those s1_ids and record all true candidate IDs needed
        truth_rows: list[str] = []
        needed_s2: set[str] = set()
        needed_s3: set[str] = set()
        with z.open("student_resource/dataset/train/train_ground_truth.tsv") as f:
            header_truth = f.readline().decode("utf-8")
            for line in f:
                decoded = line.decode("utf-8")
                parts = decoded.rstrip("\r\n").split("\t")
                if parts[0] in s1_ids:
                    truth_rows.append(decoded)
                    matches = [m.strip() for m in parts[1].split(",") if m.strip()] if len(parts) > 1 else []
                    for m in matches:
                        if m.startswith("S2-"):
                            needed_s2.add(m)
                        elif m.startswith("S3-"):
                            needed_s3.add(m)
                    if len(truth_rows) == len(s1_ids):
                        break

        # 3. Read train_source2.tsv for needed_s2 plus distractors up to n_s2
        s2_matched: list[str] = []
        found_needed_s2: set[str] = set()
        distractors_s2: list[str] = []
        with z.open("student_resource/dataset/train/train_source2.tsv") as f:
            header_s2 = f.readline().decode("utf-8")
            for line in f:
                decoded = line.decode("utf-8")
                sid = decoded.split("\t")[0]
                if sid in needed_s2:
                    s2_matched.append(decoded)
                    found_needed_s2.add(sid)
                elif len(distractors_s2) < (n_s2 - len(needed_s2)):
                    distractors_s2.append(decoded)
                if len(found_needed_s2) == len(needed_s2) and len(distractors_s2) >= (n_s2 - len(needed_s2)):
                    break

        all_s2 = s2_matched + distractors_s2

        # 4. Read train_source3.tsv for needed_s3 plus distractors up to n_s3
        s3_matched: list[str] = []
        found_needed_s3: set[str] = set()
        distractors_s3: list[str] = []
        with z.open("student_resource/dataset/train/train_source3.tsv") as f:
            header_s3 = f.readline().decode("utf-8")
            for line in f:
                decoded = line.decode("utf-8")
                sid = decoded.split("\t")[0]
                if sid in needed_s3:
                    s3_matched.append(decoded)
                    found_needed_s3.add(sid)
                elif len(distractors_s3) < (n_s3 - len(needed_s3)):
                    distractors_s3.append(decoded)
                if len(found_needed_s3) == len(needed_s3) and len(distractors_s3) >= (n_s3 - len(needed_s3)):
                    break

        all_s3 = s3_matched + distractors_s3

        # Write sample files
        with open(s1_path, "w", encoding="utf-8") as f:
            f.write(header_s1)
            for row in s1_rows:
                f.write(row)

        with open(truth_path, "w", encoding="utf-8") as f:
            f.write(header_truth)
            for row in truth_rows:
                f.write(row)

        with open(s2_path, "w", encoding="utf-8") as f:
            f.write(header_s2)
            for row in all_s2:
                f.write(row)

        with open(s3_path, "w", encoding="utf-8") as f:
            f.write(header_s3)
            for row in all_s3:
                f.write(row)

        # Write metadata
        meta = {
            "archive_path": str(archive_path),
            "archive_size": archive_stat.st_size,
            "archive_mtime": archive_stat.st_mtime,
            "requested_n_s2_s3": n_s2_s3,
            "requested_n_s1": n_s1,
            "actual_s1_count": len(s1_rows),
            "actual_s2_count": len(all_s2),
            "actual_s3_count": len(all_s3),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

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
    parser.add_argument("--force-sample", action="store_true", help="Force re-extraction of sample files")
    parser.add_argument("--generate-scale", type=int, default=None, help="Generate and benchmark a large synthetic corpus of N records")

    args = parser.parse_args(argv)
    archive_path = find_archive(args.sample_archive)
    repo_root = archive_path.parent.parent if archive_path else Path.cwd().resolve()
    expected_s1 = None

    if args.generate_scale:
        work_dir = args.work_dir or (repo_root / "artifacts" / "scale_blocking")
        scale_data_dir = work_dir / f"scale_{args.generate_scale}"
        s1, s2, s3, truth = generate_scale_data(scale_data_dir, args.generate_scale)
    elif args.sample or (not args.source1 and not args.source2 and not args.source3 and not args.fixtures and archive_path is not None):
        if archive_path is None:
            parser.error("Cannot run sample benchmark: student_resource.zip not found.")
        work_dir = args.work_dir or (repo_root / "artifacts" / "blocking" / f"sample_index_{args.sample_size}_{args.sample_s1}")
        sample_dir = repo_root / "artifacts" / "blocking"
        s1, s2, s3, truth = extract_representative_sample(
            archive_path=archive_path,
            output_dir=sample_dir,
            n_s2_s3=args.sample_size,
            n_s1=args.sample_s1,
            force=args.force_sample,
        )
        expected_s1 = args.sample_s1
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
        expected_s1=expected_s1,
    )
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
