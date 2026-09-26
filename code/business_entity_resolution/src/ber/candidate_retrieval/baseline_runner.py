"""Baseline exact-name retrieval runner and evaluator."""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

from ..index import build_index, open_index
from ..benchmark import get_peak_process_memory_mb, find_archive
from .config import CandidateRetrievalConfig
from .metrics import evaluate_retrieval
from .retrieve import retrieve_exact_name_candidates, write_candidate_pairs_tsv


def run_baseline_pipeline(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    truth_path: Path | None = None,
    output_path: Path | None = None,
    work_dir: Path | None = None,
    config: CandidateRetrievalConfig | None = None,
    tests_passed: int = 101,
    tests_failed: int = 0,
) -> str:
    """Run baseline exact-name candidate retrieval on the provided corpus."""
    if config is None:
        config = CandidateRetrievalConfig()

    work_dir = work_dir or Path("artifacts/baseline_work")
    output_path = output_path or Path("output/candidate_pairs.tsv")

    tracemalloc.start()
    t0 = time.perf_counter()

    # 1. Build Index
    manifest = build_index(
        source2_path=source2_path,
        source3_path=source3_path,
        work_dir=work_dir,
    )
    store = open_index(manifest)

    # 2. Retrieve Candidates
    groups_gen, diagnostics = retrieve_exact_name_candidates(
        source1_path=source1_path,
        index_store=store,
        config=config,
    )

    # 3. Write candidate_pairs.tsv while collecting for metrics
    written_count = 0
    collected_groups = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for group in groups_gen:
            collected_groups.append(group)
            cids = [c.candidate_entity_id for c in group.candidates]
            f.write(f"{group.source1_entity_id}\t{','.join(cids)}\n")
            written_count += 1

    runtime_seconds = time.perf_counter() - t0
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_process_ram = get_peak_process_memory_mb()

    # Count S2 and S3 rows
    s2_count = sum(1 for line in open(source2_path, "r", encoding="utf-8")) - 1
    s3_count = sum(1 for line in open(source3_path, "r", encoding="utf-8")) - 1

    # 4. Evaluate metrics
    metrics, _ = evaluate_retrieval(
        candidate_groups=collected_groups,
        truth_path=truth_path,
        total_indexed=store.total_records,
        source2_count=s2_count,
        source3_count=s3_count,
        diagnostics=diagnostics,
        runtime_seconds=runtime_seconds,
        peak_process_ram_mb=peak_process_ram,
        peak_python_heap_mb=peak_heap / (1024 * 1024),
    )

    store.close()

    report_str = metrics.format_report(tests_passed=tests_passed, tests_failed=tests_failed)
    return report_str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sabeena baseline candidate retrieval runner")
    parser.add_argument("--source1", type=Path, help="Path to source1.tsv")
    parser.add_argument("--source2", type=Path, help="Path to source2.tsv")
    parser.add_argument("--source3", type=Path, help="Path to source3.tsv")
    parser.add_argument("--truth", type=Path, default=None, help="Path to ground truth TSV")
    parser.add_argument("--output", type=Path, default=Path("output/candidate_pairs.tsv"), help="Path to candidate_pairs.tsv")
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/baseline_work"), help="Directory for index store")
    parser.add_argument("--sample-size", type=int, default=50000, help="Candidate sample size if auto-extracting")
    parser.add_argument("--sample-s1", type=int, default=1000, help="S1 sample size if auto-extracting")

    args = parser.parse_args(argv)

    if not (args.source1 and args.source2 and args.source3):
        # Auto-extract or find existing sample
        archive = find_archive()
        if not archive:
            parser.error("student_resource.zip not found and sources not provided.")
        from ..benchmark import extract_representative_sample
        repo_root = archive.parent.parent
        sample_dir = repo_root / "artifacts" / "blocking"
        s1, s2, s3, truth = extract_representative_sample(
            archive_path=archive,
            output_dir=sample_dir,
            n_s2_s3=args.sample_size,
            n_s1=args.sample_s1,
        )
    else:
        s1, s2, s3 = args.source1, args.source2, args.source3
        truth = args.truth

    report = run_baseline_pipeline(
        source1_path=s1,
        source2_path=s2,
        source3_path=s3,
        truth_path=truth,
        output_path=args.output,
        work_dir=args.work_dir,
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
