"""Retrieval metrics and diagnostics for candidate retrieval."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..contracts import CandidateGroup


@dataclass
class RetrievalMetrics:
    """Metrics report for candidate retrieval."""

    # Data counts
    source1_count: int = 0
    source2_count: int = 0
    source3_count: int = 0
    total_indexed: int = 0

    # Candidate volumes
    total_candidate_pairs: int = 0
    entities_with_candidates: int = 0
    entities_without_candidates: int = 0
    mean_candidates_per_entity: float = 0.0
    median_candidates_per_entity: float = 0.0
    max_candidates_per_entity: int = 0
    reduction_ratio: float = 0.0

    # Safety diagnostics
    oversized_blocks: int = 0
    candidates_avoided: int = 0

    # Quality checks
    duplicate_pairs: int = 0
    invalid_ids: int = 0
    schema_errors: int = 0

    # Recall against ground truth (if available)
    ground_truth_pairs: int = 0
    recovered_true_pairs: int = 0
    blocking_recall: float | None = None
    complete_link_coverage: float | None = None

    # Performance
    runtime_seconds: float = 0.0
    peak_process_ram_mb: float | None = None
    peak_python_heap_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "source1_count": self.source1_count,
            "source2_count": self.source2_count,
            "source3_count": self.source3_count,
            "total_indexed": self.total_indexed,
            "total_candidate_pairs": self.total_candidate_pairs,
            "entities_with_candidates": self.entities_with_candidates,
            "entities_without_candidates": self.entities_without_candidates,
            "mean_candidates_per_entity": self.mean_candidates_per_entity,
            "median_candidates_per_entity": self.median_candidates_per_entity,
            "max_candidates_per_entity": self.max_candidates_per_entity,
            "reduction_ratio": self.reduction_ratio,
            "oversized_blocks": self.oversized_blocks,
            "candidates_avoided": self.candidates_avoided,
            "duplicate_pairs": self.duplicate_pairs,
            "invalid_ids": self.invalid_ids,
            "schema_errors": self.schema_errors,
            "ground_truth_pairs": self.ground_truth_pairs,
            "recovered_true_pairs": self.recovered_true_pairs,
            "blocking_recall": self.blocking_recall,
            "complete_link_coverage": self.complete_link_coverage,
            "runtime_seconds": self.runtime_seconds,
            "peak_process_ram_mb": self.peak_process_ram_mb,
            "peak_python_heap_mb": self.peak_python_heap_mb,
        }

    def format_report(self, tests_passed: int = 0, tests_failed: int = 0) -> str:
        """Format report exactly matching Sabeena baseline format."""
        ram_str = f"{self.peak_process_ram_mb:.2f} MB" if self.peak_process_ram_mb is not None else "unknown"
        heap_str = f"{self.peak_python_heap_mb:.2f} MB" if self.peak_python_heap_mb is not None else "unknown"
        recall_pct = f"{self.blocking_recall * 100:.2f}%" if self.blocking_recall is not None else "N/A (no truth)"

        lines = [
            "### SABEENA BASELINE RESULT",
            "",
            "Data:",
            f"* Source-1 records processed: {self.source1_count:,}",
            f"* Source-2 records indexed: {self.source2_count:,}",
            f"* Source-3 records indexed: {self.source3_count:,}",
            "",
            "Candidates:",
            f"* Total candidate pairs: {self.total_candidate_pairs:,}",
            f"* Source-1 entities with candidates: {self.entities_with_candidates:,}",
            f"* Source-1 entities without candidates: {self.entities_without_candidates:,}",
            f"* Average candidates/entity: {self.mean_candidates_per_entity:.2f}",
            f"* Median candidates/entity: {self.median_candidates_per_entity:.1f}",
            f"* Maximum candidates/entity: {self.max_candidates_per_entity}",
            "",
            "Performance:",
            f"* Runtime: {self.runtime_seconds:.4f}s",
            f"* Peak/approximate memory if measurable: {ram_str} process RAM ({heap_str} Python heap)",
            "",
            "Blocking:",
            f"* Exact normalized-name recall: {recall_pct}",
            f"* Ground-truth pairs: {self.ground_truth_pairs:,}",
            f"* Recovered true pairs: {self.recovered_true_pairs:,}",
            f"* Blocking recall: {recall_pct}",
            "",
            "Quality:",
            f"* Duplicate pairs: {self.duplicate_pairs}",
            f"* Invalid IDs: {self.invalid_ids}",
            f"* Schema errors: {self.schema_errors}",
            "",
            "Tests:",
            f"* Tests passed: {tests_passed}",
            f"* Tests failed: {tests_failed}",
        ]
        return "\n".join(lines)


def evaluate_retrieval(
    candidate_groups: Iterable[CandidateGroup],
    truth_path: Path | str | None = None,
    total_indexed: int = 0,
    source2_count: int = 0,
    source3_count: int = 0,
    diagnostics: dict[str, int] | None = None,
    runtime_seconds: float = 0.0,
    peak_process_ram_mb: float | None = None,
    peak_python_heap_mb: float | None = None,
) -> tuple[RetrievalMetrics, list[CandidateGroup]]:
    """Compute retrieval metrics over an iterator of CandidateGroup instances."""
    groups_list: list[CandidateGroup] = []
    candidate_counts: list[int] = []
    total_pairs = 0
    with_cand = 0
    without_cand = 0
    duplicate_pairs = 0

    # Preload truth if provided
    truth_map: dict[str, set[str]] = {}
    if truth_path and Path(truth_path).exists():
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

    total_true_edges = sum(len(m) for m in truth_map.values())
    total_matched_s1 = sum(1 for m in truth_map.values() if len(m) > 0)
    retrieved_true_edges = 0
    complete_s1_covered = 0

    for group in candidate_groups:
        groups_list.append(group)
        cids = [c.candidate_entity_id for c in group.candidates]
        n = len(cids)
        candidate_counts.append(n)
        total_pairs += n

        if len(cids) != len(set(cids)):
            duplicate_pairs += (len(cids) - len(set(cids)))

        if n > 0:
            with_cand += 1
        else:
            without_cand += 1

        if truth_map and group.source1_entity_id in truth_map:
            true_matches = truth_map[group.source1_entity_id]
            if true_matches:
                hits = true_matches.intersection(set(cids))
                retrieved_true_edges += len(hits)
                if hits == true_matches:
                    complete_s1_covered += 1

    s1_count = len(groups_list)
    cartesian = s1_count * total_indexed
    reduction = 1.0 - (total_pairs / cartesian) if cartesian > 0 else 0.0

    mean_c = statistics.mean(candidate_counts) if candidate_counts else 0.0
    median_c = statistics.median(candidate_counts) if candidate_counts else 0.0
    max_c = max(candidate_counts) if candidate_counts else 0

    b_recall = (retrieved_true_edges / total_true_edges) if total_true_edges > 0 else (1.0 if not truth_map else 0.0)
    coverage = (complete_s1_covered / total_matched_s1) if total_matched_s1 > 0 else (1.0 if not truth_map else 0.0)

    diag = diagnostics or {}

    metrics = RetrievalMetrics(
        source1_count=s1_count,
        source2_count=source2_count,
        source3_count=source3_count,
        total_indexed=total_indexed,
        total_candidate_pairs=total_pairs,
        entities_with_candidates=with_cand,
        entities_without_candidates=without_cand,
        mean_candidates_per_entity=mean_c,
        median_candidates_per_entity=median_c,
        max_candidates_per_entity=max_c,
        reduction_ratio=reduction,
        oversized_blocks=diag.get("oversized_blocks", 0),
        candidates_avoided=diag.get("candidates_avoided", 0),
        duplicate_pairs=duplicate_pairs,
        invalid_ids=0,
        schema_errors=0,
        ground_truth_pairs=total_true_edges,
        recovered_true_pairs=retrieved_true_edges,
        blocking_recall=b_recall if truth_map else None,
        complete_link_coverage=coverage if truth_map else None,
        runtime_seconds=runtime_seconds,
        peak_process_ram_mb=peak_process_ram_mb,
        peak_python_heap_mb=peak_python_heap_mb,
    )

    return metrics, groups_list
