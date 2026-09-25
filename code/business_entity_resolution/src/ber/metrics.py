"""
metrics.py - Official evaluation and candidate diagnostics for BER.

Owner: Suresh

Implements:
  evaluate(truth, predictions)           -> EvaluationResult
  evaluate_candidates(truth, candidates) -> CandidateResult
  error_analysis(truth, predictions, candidates) -> ErrorReport
  oracle_sanity_check(eval_result, oracle_result) -> (bool, str)
  france_coverage_check(test_s1_ids, output_s1_ids, country_labels) -> dict

All functions support streaming/batched inputs.  They NEVER load the full
10 M-record corpus into a single in-memory structure.

Formula (per Source-1 entity):
  T = true matched ID set
  P = predicted matched ID set

  Case 1: T == {} and P == {}  ->  score = 1.0
  Case 2: T == {} and P != {}  ->  score = 0.0
  Case 3 (otherwise):
    precision = |T & P| / |P|
    recall    = |T & P| / |T|
    F0.5 = 1.25 * precision * recall / (0.25 * precision + recall)
           (defined as 0 when both precision and recall are 0)

Final:  macro_f0_5 = mean(score over ALL S1 entities)

IMPORTANT:
  - True-empty/predicted-empty cases ARE included (score = 1.0).
  - True-empty/predicted-non-empty cases ARE included (score = 0.0).
  - Micro or pair-level F0.5 is NOT computed here.
  - Do NOT call this "done" without a measured run and recorded evidence.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import (
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Structured result returned by evaluate()."""

    macro_f0_5: float
    mean_precision: float
    mean_recall: float

    # Entity counts
    evaluated_s1_count: int
    true_empty_count: int
    predicted_empty_count: int

    # Error counts
    false_positive_count: int
    false_negative_count: int

    # Special-case counts
    singleton_correct: int
    singleton_total: int

    # Source slices (populated when source info available)
    s2_macro_f0_5: Optional[float] = None
    s3_macro_f0_5: Optional[float] = None

    # Country slices (populated from labeled validation data only)
    country_f0_5: Dict[str, float] = field(default_factory=dict)

    # Per-entity scores for further analysis (populated if requested)
    per_entity_scores: Optional[Dict[str, float]] = None

    @property
    def singleton_accuracy(self) -> float:
        """Fraction of true-empty S1 entities that were also predicted empty.

        Contract definition: a *singleton* is an S1 with **no** true links
        (i.e. the correct output is an empty match row).  singleton_accuracy
        measures how often the model correctly produces an empty prediction for
        these entities.
        """
        if self.singleton_total == 0:
            return float("nan")
        return self.singleton_correct / self.singleton_total


@dataclass
class CandidateResult:
    """Structured result returned by evaluate_candidates()."""

    # Recall metrics
    true_edge_recall: float
    complete_s1_recall: float

    # Oracle
    oracle_macro_f0_5: float

    # Candidate count distribution
    total_candidate_pairs: int
    min_candidates: int
    mean_candidates: float
    median_candidates: float
    p95_candidates: float
    p99_candidates: float
    max_candidates: int

    # Reduction ratio
    reduction_ratio: Optional[float] = None

    # Source-level true-edge recall
    s2_true_edge_recall: Optional[float] = None
    s3_true_edge_recall: Optional[float] = None

    # Evaluated counts
    evaluated_s1_count: int = 0
    total_true_edges: int = 0
    true_edges_recalled: int = 0


@dataclass
class ErrorCategory:
    """Counts and representative examples for one error class."""

    count: int = 0
    examples: List[Tuple[str, str]] = field(default_factory=list)
    max_examples: int = 10

    def add(self, s1_id: str, detail: str) -> None:
        self.count += 1
        if len(self.examples) < self.max_examples:
            self.examples.append((s1_id, detail))


@dataclass
class ErrorReport:
    """Compact error-analysis report.

    Categories:
      A. retrieval_miss       - true ID absent from candidate set C
      B. scoring_false_neg    - true ID in C but not predicted
      C. false_positive       - predicted ID not in truth T
      D. wrong_singleton      - |T|=1 and |P|=1 but prediction wrong
      E. empty_case_fp        - T empty but P non-empty
      F. multi_match_partial  - |T|>1 but at least one true ID not predicted
    """

    retrieval_miss: ErrorCategory = field(default_factory=ErrorCategory)
    scoring_false_neg: ErrorCategory = field(default_factory=ErrorCategory)
    false_positive: ErrorCategory = field(default_factory=ErrorCategory)
    wrong_singleton: ErrorCategory = field(default_factory=ErrorCategory)
    empty_case_fp: ErrorCategory = field(default_factory=ErrorCategory)
    multi_match_partial: ErrorCategory = field(default_factory=ErrorCategory)

    total_s1: int = 0

    def summary(self) -> str:
        lines = [
            "=== Error Analysis Report ===",
            f"Total S1 evaluated: {self.total_s1}",
            "",
            "A. Retrieval miss (true ID absent from candidates): "
            f"{self.retrieval_miss.count}",
            "B. Scoring false negative (in candidates, not predicted): "
            f"{self.scoring_false_neg.count}",
            "C. False positive (predicted ID not in truth): "
            f"{self.false_positive.count}",
            "D. Wrong singleton (|T|=1, |P|=1, mismatch): "
            f"{self.wrong_singleton.count}",
            "E. Empty-case false positive (T empty, P non-empty): "
            f"{self.empty_case_fp.count}",
            "F. Multi-match partial miss (|T|>1, some true IDs missing): "
            f"{self.multi_match_partial.count}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core metric formula
# ---------------------------------------------------------------------------


def _f0_5_score(truth: FrozenSet[str], predicted: FrozenSet[str]) -> float:
    """Compute per-entity F0.5 score according to the official formula.

    Args:
        truth:     set of true matched S2/S3 IDs for this S1 entity.
        predicted: set of predicted matched S2/S3 IDs for this S1 entity.

    Returns:
        float in [0.0, 1.0].
    """
    if not truth and not predicted:
        return 1.0
    if not truth or not predicted:
        return 0.0

    tp = len(truth & predicted)
    if tp == 0:
        return 0.0

    precision = tp / len(predicted)
    recall = tp / len(truth)

    # F_beta with beta=0.5
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return 1.25 * precision * recall / denom


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def evaluate(
    truth: Iterable[Tuple[str, Set[str]]],
    predictions: Iterable[Tuple[str, Set[str]]],
    country_labels: Optional[Dict[str, str]] = None,
    store_per_entity: bool = False,
) -> EvaluationResult:
    """Compute macro F0.5 and diagnostic counts over all S1 entities.

    Args:
        truth:          Iterable of (s1_id, true_id_set). Each ID set may be empty.
        predictions:    Iterable of (s1_id, predicted_id_set).
        country_labels: Optional {s1_id: country}. Use ONLY labeled validation
                        data. Never claim France F0.5 from unlabeled test data.
        store_per_entity: If True, per_entity_scores dict is populated.

    Returns:
        EvaluationResult with all fields populated.
    """
    # Step 1: load truth index
    truth_index: Dict[str, FrozenSet[str]] = {}
    for item in truth:
        if hasattr(item, "source1_entity_id") and hasattr(item, "matched_entity_ids"):
            s1_id = item.source1_entity_id
            true_ids = item.matched_entity_ids
        else:
            s1_id, true_ids = item
        if s1_id in truth_index:
            raise ValueError(f"Duplicate S1 ID in ground truth: {s1_id}")
        truth_index[s1_id] = frozenset(true_ids)

    # Step 2: load prediction index (reject duplicates and require complete alignment)
    pred_index: Dict[str, FrozenSet[str]] = {}
    for item in predictions:
        if hasattr(item, "source1_entity_id") and hasattr(item, "matched_entity_ids"):
            s1_id = item.source1_entity_id
            pred_ids = item.matched_entity_ids
        else:
            s1_id, pred_ids = item
        if s1_id in pred_index:
            raise ValueError(f"Duplicate S1 ID in predictions: {s1_id}")
        pred_index[s1_id] = frozenset(pred_ids)

    missing_in_pred = set(truth_index) - set(pred_index)
    if missing_in_pred:
        sample = sorted(missing_in_pred)[:5]
        raise ValueError(
            f"Missing prediction rows for {len(missing_in_pred)} truth S1 IDs (e.g. {sample})"
        )

    unexpected_in_pred = set(pred_index) - set(truth_index)
    if unexpected_in_pred:
        sample = sorted(unexpected_in_pred)[:5]
        raise ValueError(
            f"Unexpected prediction rows for {len(unexpected_in_pred)} S1 IDs not in truth (e.g. {sample})"
        )

    # Step 3: compute per-entity scores
    scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    true_empty_count = 0
    pred_empty_count = 0
    fp_total = 0
    fn_total = 0
    singleton_correct = 0
    singleton_total = 0

    country_scores: Dict[str, List[float]] = {}
    per_entity: Optional[Dict[str, float]] = {} if store_per_entity else None

    for s1_id, true_ids in truth_index.items():
        pred_ids = pred_index[s1_id]

        score = _f0_5_score(true_ids, pred_ids)
        scores.append(score)
        if per_entity is not None:
            per_entity[s1_id] = score

        # Precision/recall for diagnostics
        if not true_ids and not pred_ids:
            p, r = 1.0, 1.0
        elif not true_ids or not pred_ids:
            p, r = 0.0, 0.0
        else:
            tp = len(true_ids & pred_ids)
            p = tp / len(pred_ids)
            r = tp / len(true_ids)
        precisions.append(p)
        recalls.append(r)

        if not true_ids:
            true_empty_count += 1
        if not pred_ids:
            pred_empty_count += 1
        fp_total += len(pred_ids - true_ids)
        fn_total += len(true_ids - pred_ids)

        # Contract: singleton = S1 with no true links (true_empty case).
        # singleton_correct counts how many were also predicted empty.
        if not true_ids:
            singleton_total += 1
            if not pred_ids:
                singleton_correct += 1

        if country_labels:
            country = country_labels.get(s1_id)
            if country:
                country_scores.setdefault(country, []).append(score)

    macro = statistics.mean(scores) if scores else float("nan")
    mean_p = statistics.mean(precisions) if precisions else float("nan")
    mean_r = statistics.mean(recalls) if recalls else float("nan")
    country_f0_5 = {k: statistics.mean(v) for k, v in country_scores.items()}

    return EvaluationResult(
        macro_f0_5=macro,
        mean_precision=mean_p,
        mean_recall=mean_r,
        evaluated_s1_count=len(truth_index),
        true_empty_count=true_empty_count,
        predicted_empty_count=pred_empty_count,
        false_positive_count=fp_total,
        false_negative_count=fn_total,
        singleton_correct=singleton_correct,
        singleton_total=singleton_total,
        country_f0_5=country_f0_5,
        per_entity_scores=per_entity,
    )


# ---------------------------------------------------------------------------
# Streaming TSV helpers
# ---------------------------------------------------------------------------


def _iter_truth_tsv(path: str) -> Generator[Tuple[str, FrozenSet[str]], None, None]:
    """Yield (s1_id, frozenset(true_ids)) from a ground-truth TSV.

    Expected header: source1_entity_id<TAB>matched_entity_ids
    matched_entity_ids is comma-separated or empty string.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            if not s1_id:
                continue
            raw_ids = parts[1].strip() if len(parts) > 1 else ""
            ids: FrozenSet[str] = (
                frozenset(i.strip() for i in raw_ids.split(",") if i.strip())
                if raw_ids
                else frozenset()
            )
            yield s1_id, ids


def _iter_output_tsv(path: str) -> Generator[Tuple[str, FrozenSet[str]], None, None]:
    """Yield (s1_id, frozenset(ids)) from matching_results.tsv or candidate_pairs.tsv."""
    with open(path, encoding="utf-8", newline="") as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            if not s1_id:
                continue
            raw_ids = parts[1].strip() if len(parts) > 1 else ""
            ids: FrozenSet[str] = (
                frozenset(i.strip() for i in raw_ids.split(",") if i.strip())
                if raw_ids
                else frozenset()
            )
            yield s1_id, ids


def evaluate_from_tsv(
    truth_tsv: str,
    predictions_tsv: str,
    country_labels: Optional[Dict[str, str]] = None,
    store_per_entity: bool = False,
) -> EvaluationResult:
    """Convenience wrapper: evaluate from TSV file paths."""
    return evaluate(
        _iter_truth_tsv(truth_tsv),
        _iter_output_tsv(predictions_tsv),
        country_labels=country_labels,
        store_per_entity=store_per_entity,
    )


# ---------------------------------------------------------------------------
# evaluate_candidates()
# ---------------------------------------------------------------------------


def _percentile(sorted_data: List[int], pct: float) -> float:
    """Nearest-rank percentile for a pre-sorted list."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    k = math.ceil(pct / 100.0 * n) - 1
    k = max(0, min(k, n - 1))
    return float(sorted_data[k])


def evaluate_candidates(
    truth: Iterable[Tuple[str, Set[str]]],
    candidates: Iterable[Tuple[str, Set[str]]],
    corpus_size: Optional[int] = None,
) -> CandidateResult:
    """Compute candidate retrieval quality metrics.

    Args:
        truth:       Iterable of (s1_id, true_id_set). Fully consumed first.
        candidates:  Iterable of (s1_id, candidate_id_set).
        corpus_size: Total |S2 union S3| record count for reduction ratio.

    Memory: candidate count list uses ~14 MB for 1.7M S1 entities (acceptable
    on 8 GB). Switches to t-digest if tighter memory needed (document change).
    """
    # Step 1: load truth index
    truth_index: Dict[str, FrozenSet[str]] = {}
    for item in truth:
        if hasattr(item, "source1_entity_id") and hasattr(item, "matched_entity_ids"):
            s1_id = item.source1_entity_id
            true_ids = item.matched_entity_ids
        else:
            s1_id, true_ids = item
        if s1_id in truth_index:
            raise ValueError(f"Duplicate S1 ID in ground truth: {s1_id}")
        truth_index[s1_id] = frozenset(true_ids)

    total_true_edges = sum(len(v) for v in truth_index.values())
    total_s2_edges = sum(1 for ids in truth_index.values() for tid in ids if tid.startswith("S2-"))
    total_s3_edges = sum(1 for ids in truth_index.values() for tid in ids if tid.startswith("S3-"))

    # Step 2: iterate candidates
    cand_counts: List[int] = []
    true_edges_recalled = 0
    s2_recalled = 0
    s3_recalled = 0
    complete_s1_recalled = 0
    matched_s1_total = 0
    oracle_scores: List[float] = []
    total_candidate_pairs = 0
    seen_s1: Set[str] = set()

    for item in candidates:
        if hasattr(item, "source1_entity_id") and hasattr(item, "candidates"):
            s1_id = item.source1_entity_id
            cand_ids = tuple(c.candidate_entity_id for c in item.candidates)
        else:
            s1_id, cand_ids = item
        if s1_id in seen_s1:
            raise ValueError(f"Duplicate candidate group for S1 ID: {s1_id}")
        seen_s1.add(s1_id)
        if s1_id not in truth_index:
            raise ValueError(f"Unexpected candidate group for S1 ID not in truth: {s1_id}")

        cand_fz = frozenset(cand_ids)
        n = len(cand_fz)
        cand_counts.append(n)
        total_candidate_pairs += n

        true_ids = truth_index[s1_id]
        recalled = true_ids & cand_fz
        true_edges_recalled += len(recalled)
        for rid in recalled:
            if rid.startswith("S2-"):
                s2_recalled += 1
            elif rid.startswith("S3-"):
                s3_recalled += 1

        if true_ids:
            matched_s1_total += 1
            if true_ids <= cand_fz:
                complete_s1_recalled += 1

        oracle_pred = true_ids & cand_fz
        oracle_scores.append(_f0_5_score(true_ids, oracle_pred))

    missing_in_cands = set(truth_index) - seen_s1
    if missing_in_cands:
        sample = sorted(missing_in_cands)[:5]
        raise ValueError(
            f"Missing candidate groups for {len(missing_in_cands)} truth S1 IDs (e.g. {sample})"
        )

    oracle_macro = statistics.mean(oracle_scores) if oracle_scores else float("nan")

    true_edge_recall = (
        true_edges_recalled / total_true_edges if total_true_edges else float("nan")
    )
    s2_recall = s2_recalled / total_s2_edges if total_s2_edges else float("nan")
    s3_recall = s3_recalled / total_s3_edges if total_s3_edges else float("nan")
    complete_recall = (
        complete_s1_recalled / matched_s1_total if matched_s1_total else float("nan")
    )

    if cand_counts:
        cand_sorted = sorted(cand_counts)
        n_s1 = len(cand_sorted)
        min_c = cand_sorted[0]
        max_c = cand_sorted[-1]
        mean_c = total_candidate_pairs / n_s1
        med_c = _percentile(cand_sorted, 50)
        p95 = _percentile(cand_sorted, 95)
        p99 = _percentile(cand_sorted, 99)
    else:
        min_c = max_c = 0
        mean_c = med_c = p95 = p99 = 0.0

    s1_count = len(truth_index)
    reduction = (
        total_candidate_pairs / (s1_count * corpus_size)
        if corpus_size and s1_count
        else None
    )

    return CandidateResult(
        true_edge_recall=true_edge_recall,
        complete_s1_recall=complete_recall,
        oracle_macro_f0_5=oracle_macro,
        total_candidate_pairs=total_candidate_pairs,
        min_candidates=min_c,
        mean_candidates=mean_c,
        median_candidates=med_c,
        p95_candidates=p95,
        p99_candidates=p99,
        max_candidates=max_c,
        reduction_ratio=reduction,
        s2_true_edge_recall=s2_recall,
        s3_true_edge_recall=s3_recall,
        evaluated_s1_count=s1_count,
        total_true_edges=total_true_edges,
        true_edges_recalled=true_edges_recalled,
    )


def evaluate_candidates_from_tsv(
    truth_tsv: str,
    candidates_tsv: str,
    corpus_size: Optional[int] = None,
) -> CandidateResult:
    """Convenience wrapper: evaluate candidates from TSV file paths."""
    return evaluate_candidates(
        _iter_truth_tsv(truth_tsv),
        _iter_output_tsv(candidates_tsv),
        corpus_size=corpus_size,
    )


# ---------------------------------------------------------------------------
# Oracle sanity check
# ---------------------------------------------------------------------------


def oracle_sanity_check(
    eval_result: EvaluationResult,
    oracle_result: CandidateResult,
) -> Tuple[bool, str]:
    """Check that model F0.5 does not exceed the oracle F0.5.

    The oracle predicts exactly T & C for each S1. Any correctly-evaluated
    model restricted to the same candidate set C must score <= oracle.

    Returns:
        (passed, message). passed=False means an evaluation bug.
        The caller MUST NOT continue silently on False.
    """
    model_score = eval_result.macro_f0_5
    oracle_score = oracle_result.oracle_macro_f0_5

    if math.isnan(model_score) or math.isnan(oracle_score):
        return True, "Sanity check skipped: NaN encountered (insufficient data)."

    tolerance = 1e-9
    if model_score > oracle_score + tolerance:
        msg = (
            f"ORACLE SANITY CHECK FAILED: model macro F0.5 ({model_score:.6f}) "
            f"exceeds oracle macro F0.5 ({oracle_score:.6f}). "
            "This indicates an evaluation bug. Investigate before proceeding."
        )
        return False, msg

    msg = (
        f"Oracle sanity check PASSED: model={model_score:.6f} "
        f"<= oracle={oracle_score:.6f}."
    )
    return True, msg


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def error_analysis(
    truth: Iterable[Tuple[str, Set[str]]],
    predictions: Iterable[Tuple[str, Set[str]]],
    candidates: Iterable[Tuple[str, Set[str]]],
    max_examples: int = 10,
) -> ErrorReport:
    """Classify per-entity errors into the six defined categories.

    Distinguishes retrieval misses (true ID absent from C) from scoring false
    negatives (true ID in C but not predicted).

    Memory: O(|S1| IDs) - does NOT store business-name strings.
    """
    truth_index: Dict[str, FrozenSet[str]] = {}
    for s1_id, true_ids in truth:
        truth_index[s1_id] = frozenset(true_ids)

    pred_index: Dict[str, FrozenSet[str]] = {}
    for s1_id, pred_ids in predictions:
        pred_index[s1_id] = frozenset(pred_ids)

    cand_index: Dict[str, FrozenSet[str]] = {}
    for s1_id, cand_ids in candidates:
        cand_index[s1_id] = frozenset(cand_ids)

    report = ErrorReport(total_s1=len(truth_index))
    for cat in (
        report.retrieval_miss,
        report.scoring_false_neg,
        report.false_positive,
        report.wrong_singleton,
        report.empty_case_fp,
        report.multi_match_partial,
    ):
        cat.max_examples = max_examples

    for s1_id, true_ids in truth_index.items():
        pred_ids = pred_index.get(s1_id, frozenset())
        cand_ids = cand_index.get(s1_id, frozenset())

        # E. Empty-case false positive
        if not true_ids and pred_ids:
            report.empty_case_fp.add(s1_id, f"predicted={sorted(pred_ids)[:3]}")
            continue

        # A & B: False negatives
        for tid in true_ids - pred_ids:
            if tid not in cand_ids:
                report.retrieval_miss.add(s1_id, f"missed={tid}")
            else:
                report.scoring_false_neg.add(s1_id, f"in_cands_not_predicted={tid}")

        # C. False positive
        for fid in pred_ids - true_ids:
            report.false_positive.add(s1_id, f"fp={fid}")

        # D. Wrong singleton
        if len(true_ids) == 1 and len(pred_ids) == 1 and pred_ids != true_ids:
            report.wrong_singleton.add(
                s1_id, f"true={list(true_ids)[0]} pred={list(pred_ids)[0]}"
            )

        # F. Multi-match partial miss
        if len(true_ids) > 1 and (true_ids - pred_ids):
            report.multi_match_partial.add(
                s1_id, f"missed={sorted(true_ids - pred_ids)[:3]}"
            )

    return report


def error_analysis_from_tsv(
    truth_tsv: str,
    predictions_tsv: str,
    candidates_tsv: str,
    max_examples: int = 10,
) -> ErrorReport:
    """Convenience wrapper: error analysis from TSV file paths."""
    return error_analysis(
        _iter_truth_tsv(truth_tsv),
        _iter_output_tsv(predictions_tsv),
        _iter_output_tsv(candidates_tsv),
        max_examples=max_examples,
    )


# ---------------------------------------------------------------------------
# France coverage check
# ---------------------------------------------------------------------------


def france_coverage_check(
    test_s1_ids: Iterable[str],
    output_s1_ids: Iterable[str],
    country_labels: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Check that all French S1 rows have output rows (coverage only).

    France is an unseen training country. We CANNOT assign a validation F0.5.
    This function checks COVERAGE only, not prediction quality.

    Args:
        test_s1_ids:    All S1 IDs from the test file.
        output_s1_ids:  All S1 IDs in the output matching file.
        country_labels: Optional {s1_id: country}. Used to count French IDs.

    Returns:
        dict with coverage counts and examples of missing IDs.
    """
    test_ids = set(test_s1_ids)
    out_ids = set(output_s1_ids)
    missing = test_ids - out_ids
    extra = out_ids - test_ids

    france_ids: Set[str] = set()
    if country_labels:
        france_ids = {
            sid
            for sid, c in country_labels.items()
            if c.strip().lower() == "france"
        }

    france_in_output = france_ids & out_ids

    return {
        "total_test_s1": len(test_ids),
        "output_s1_count": len(out_ids),
        "missing_count": len(missing),
        "missing_examples": sorted(missing)[:10],
        "extra_count": len(extra),
        "extra_examples": sorted(extra)[:10],
        "france_total": len(france_ids),
        "france_in_output": len(france_in_output),
        "france_missing": len(france_ids - france_in_output),
    }
