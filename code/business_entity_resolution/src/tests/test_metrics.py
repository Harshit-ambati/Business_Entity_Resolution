"""
Tests for ber.metrics — Suresh workstream.

All fixtures are synthetic (no real business records).

Test cases covered:
  1.  true empty + predicted empty => F0.5 = 1
  2.  true empty + predicted non-empty => F0.5 = 0
  3.  exact singleton match
  4.  incorrect singleton
  5.  multiple true links, all predicted
  6.  multiple true links, partial prediction
  7.  false positive
  8.  false negative
  9.  duplicate prediction IDs (deduplicated before scoring)
  10. duplicate candidate IDs (deduplicated before scoring)
  14. retrieval miss (true ID absent from candidates)
  15. candidate oracle
  16. S2-only true links
  17. S3-only true links
  19. unseen France country fixture (coverage only, no F0.5)
  21. empty candidate list
  24. Unicode/non-Latin business names (IDs only in metrics)
  25. candidate oracle sanity check (oracle >= model)
"""

import math
import pytest

from ber.metrics import (
    EvaluationResult,
    CandidateResult,
    ErrorReport,
    _f0_5_score,
    evaluate,
    evaluate_candidates,
    error_analysis,
    oracle_sanity_check,
    france_coverage_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truth(*pairs):
    """Build truth iterable from (s1_id, [true_ids...]) pairs."""
    return [(s1, set(ids)) for s1, ids in pairs]


def _preds(*pairs):
    """Build prediction iterable from (s1_id, [pred_ids...]) pairs."""
    return [(s1, set(ids)) for s1, ids in pairs]


def _cands(*pairs):
    """Build candidate iterable from (s1_id, [cand_ids...]) pairs."""
    return [(s1, set(ids)) for s1, ids in pairs]


APPROX = pytest.approx


# ---------------------------------------------------------------------------
# _f0_5_score unit tests
# ---------------------------------------------------------------------------


class TestF05ScoreFormula:
    """Direct unit tests for the core per-entity F0.5 formula."""

    def test_both_empty(self):
        assert _f0_5_score(frozenset(), frozenset()) == 1.0

    def test_true_empty_pred_nonempty(self):
        assert _f0_5_score(frozenset(), frozenset(["S2-001"])) == 0.0

    def test_true_nonempty_pred_empty(self):
        assert _f0_5_score(frozenset(["S2-001"]), frozenset()) == 0.0

    def test_exact_singleton_match(self):
        assert _f0_5_score(frozenset(["S2-001"]), frozenset(["S2-001"])) == 1.0

    def test_incorrect_singleton(self):
        assert _f0_5_score(frozenset(["S2-001"]), frozenset(["S2-002"])) == 0.0

    def test_perfect_multi_match(self):
        T = frozenset(["S2-001", "S3-002"])
        P = frozenset(["S2-001", "S3-002"])
        assert _f0_5_score(T, P) == APPROX(1.0)

    def test_partial_multi_match(self):
        # T = {A, B}, P = {A}; precision=1, recall=0.5
        # F0.5 = 1.25 * 1 * 0.5 / (0.25 + 0.5) = 0.625 / 0.75 = 0.8333...
        T = frozenset(["S2-001", "S2-002"])
        P = frozenset(["S2-001"])
        expected = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5)
        assert _f0_5_score(T, P) == APPROX(expected)

    def test_false_positive(self):
        # T = {A}, P = {A, B}; precision=0.5, recall=1
        # F0.5 = 1.25 * 0.5 * 1 / (0.25 * 0.5 + 1) = 0.625 / 1.125 = 0.5555...
        T = frozenset(["S2-001"])
        P = frozenset(["S2-001", "S2-002"])
        expected = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0)
        assert _f0_5_score(T, P) == APPROX(expected)

    def test_false_negative(self):
        # T = {A, B}, P = {} -> 0
        T = frozenset(["S2-001", "S2-002"])
        P = frozenset()
        assert _f0_5_score(T, P) == 0.0

    def test_no_overlap(self):
        T = frozenset(["S2-001"])
        P = frozenset(["S3-099"])
        assert _f0_5_score(T, P) == 0.0

    def test_range_is_zero_to_one(self):
        T = frozenset(["S2-001", "S2-002", "S3-003"])
        P = frozenset(["S2-001", "S2-999"])  # 1 correct, 1 FP
        score = _f0_5_score(T, P)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# evaluate() — macro F0.5 tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for evaluate() macro F0.5 aggregation."""

    def test_case1_both_empty_score_1(self):
        truth = _truth(("S1-001", []))
        preds = _preds(("S1-001", []))
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(1.0)
        assert r.true_empty_count == 1
        assert r.predicted_empty_count == 1
        # true-empty entity correctly predicted empty -> singleton accuracy 1.0
        assert r.singleton_total == 1
        assert r.singleton_correct == 1
        assert r.singleton_accuracy == APPROX(1.0)

    def test_case2_true_empty_pred_nonempty_score_0(self):
        truth = _truth(("S1-001", []))
        preds = _preds(("S1-001", ["S2-001"]))
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(0.0)
        assert r.false_positive_count == 1

    def test_exact_singleton(self):
        """S1 with one true link is NOT a contract singleton (no true links)."""
        truth = _truth(("S1-001", ["S2-001"]))
        preds = _preds(("S1-001", ["S2-001"]))
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(1.0)
        # Contract singleton = true_empty; this entity has 1 link -> not a singleton
        assert r.singleton_total == 0
        assert r.singleton_correct == 0

    def test_incorrect_singleton(self):
        """S1 with one true link: wrong prediction; singleton counts unchanged."""
        truth = _truth(("S1-001", ["S2-001"]))
        preds = _preds(("S1-001", ["S2-002"]))
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(0.0)
        # Contract singleton = true_empty; this entity has 1 link -> not a singleton
        assert r.singleton_total == 0
        assert r.singleton_correct == 0

    def test_singleton_accuracy_true_empty_pred_nonempty_is_zero(self):
        """true_empty S1 predicted non-empty: singleton_correct=0, accuracy=0.0."""
        truth = _truth(("S1-001", []))
        preds = _preds(("S1-001", ["S2-001"]))
        r = evaluate(truth, preds)
        assert r.singleton_total == 1
        assert r.singleton_correct == 0
        assert r.singleton_accuracy == APPROX(0.0)

    def test_singleton_accuracy_nan_when_no_true_empty(self):
        """When there are no true_empty entities, singleton_accuracy is NaN."""
        truth = _truth(("S1-001", ["S2-001"]))
        preds = _preds(("S1-001", ["S2-001"]))
        r = evaluate(truth, preds)
        assert r.singleton_total == 0
        import math
        assert math.isnan(r.singleton_accuracy)


    def test_multiple_true_all_predicted(self):
        truth = _truth(("S1-001", ["S2-001", "S3-002"]))
        preds = _preds(("S1-001", ["S2-001", "S3-002"]))
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(1.0)

    def test_multiple_true_partial_prediction(self):
        truth = _truth(("S1-001", ["S2-001", "S2-002"]))
        preds = _preds(("S1-001", ["S2-001"]))  # recall=0.5, precision=1
        expected = 1.25 * 1.0 * 0.5 / (0.25 + 0.5)
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(expected)
        assert r.false_negative_count == 1

    def test_false_positive(self):
        truth = _truth(("S1-001", ["S2-001"]))
        preds = _preds(("S1-001", ["S2-001", "S2-999"]))
        r = evaluate(truth, preds)
        assert r.false_positive_count == 1
        assert r.macro_f0_5 < 1.0

    def test_macro_includes_all_s1(self):
        """Macro average must include true-empty entities."""
        truth = _truth(
            ("S1-001", []),       # score=1
            ("S1-002", ["S2-001"]),  # score=1 (perfect match)
            ("S1-003", ["S2-002"]),  # score=0 (missed)
        )
        preds = _preds(
            ("S1-001", []),
            ("S1-002", ["S2-001"]),
            ("S1-003", []),
        )
        r = evaluate(truth, preds)
        assert r.evaluated_s1_count == 3
        assert r.macro_f0_5 == APPROX((1.0 + 1.0 + 0.0) / 3)

    def test_duplicate_prediction_ids_deduplicated(self):
        """Duplicate IDs within a prediction set must be deduplicated."""
        truth = _truth(("S1-001", ["S2-001"]))
        # Pass a list with duplicates; frozenset deduplication in evaluate()
        preds = [("S1-001", {"S2-001", "S2-001"})]  # set already deduplicates
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(1.0)

    def test_store_per_entity(self):
        truth = _truth(("S1-001", ["S2-001"]), ("S1-002", []))
        preds = _preds(("S1-001", ["S2-001"]), ("S1-002", []))
        r = evaluate(truth, preds, store_per_entity=True)
        assert r.per_entity_scores is not None
        assert r.per_entity_scores["S1-001"] == APPROX(1.0)
        assert r.per_entity_scores["S1-002"] == APPROX(1.0)

    def test_s2_only_true_links(self):
        truth = _truth(("S1-001", ["S2-001", "S2-002"]))
        preds = _preds(("S1-001", ["S2-001", "S2-002"]))
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(1.0)

    def test_s3_only_true_links(self):
        truth = _truth(("S1-001", ["S3-001", "S3-002"]))
        preds = _preds(("S1-001", ["S3-001", "S3-002"]))
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(1.0)

    def test_country_slice_populated(self):
        truth = _truth(
            ("S1-001", ["S2-001"]),
            ("S1-002", ["S2-002"]),
        )
        preds = _preds(
            ("S1-001", ["S2-001"]),
            ("S1-002", []),
        )
        country_labels = {"S1-001": "India", "S1-002": "India"}
        r = evaluate(truth, preds, country_labels=country_labels)
        assert "India" in r.country_f0_5
        assert r.country_f0_5["India"] == APPROX(0.5)

    def test_unicode_s1_ids(self):
        """IDs are opaque strings; Unicode in S1 IDs must be handled."""
        truth = _truth(("S1-\u4e2d\u6587", ["S2-001"]))
        preds = _preds(("S1-\u4e2d\u6587", ["S2-001"]))
        r = evaluate(truth, preds)
        assert r.macro_f0_5 == APPROX(1.0)

    def test_missing_prediction_raises_value_error(self):
        """S1 with truth but no prediction row must raise ValueError."""
        truth = _truth(("S1-001", ["S2-001"]))
        preds = []  # no predictions at all
        with pytest.raises(ValueError, match="Missing prediction rows"):
            evaluate(truth, preds)

    def test_harshit_probe_missing_singleton_raises(self):
        """Replicates Harshit's probe: missing singleton prediction cannot score 1.0."""
        truth = [("S1-A", set()), ("S1-B", {"S2-X"})]
        predictions = [("S1-B", {"S2-X"})]  # S1-A missing
        with pytest.raises(ValueError, match="Missing prediction rows"):
            evaluate(truth, predictions)

    def test_duplicate_truth_s1_raises(self):
        truth = [("S1-001", set()), ("S1-001", {"S2-001"})]
        preds = [("S1-001", set())]
        with pytest.raises(ValueError, match="Duplicate S1 ID in ground truth"):
            evaluate(truth, preds)

    def test_duplicate_prediction_s1_raises(self):
        truth = [("S1-001", set())]
        preds = [("S1-001", set()), ("S1-001", {"S2-001"})]
        with pytest.raises(ValueError, match="Duplicate S1 ID in predictions"):
            evaluate(truth, preds)

    def test_unexpected_prediction_s1_raises(self):
        truth = [("S1-001", set())]
        preds = [("S1-001", set()), ("S1-EXTRA", set())]
        with pytest.raises(ValueError, match="Unexpected prediction rows"):
            evaluate(truth, preds)


# ---------------------------------------------------------------------------
# evaluate_candidates() tests
# ---------------------------------------------------------------------------


class TestEvaluateCandidates:
    """Tests for candidate recall, oracle, and distribution metrics."""

    def test_perfect_recall(self):
        truth = _truth(("S1-001", ["S2-001", "S3-002"]))
        cands = _cands(("S1-001", ["S2-001", "S3-002", "S2-999"]))
        r = evaluate_candidates(truth, cands)
        assert r.true_edge_recall == APPROX(1.0)
        assert r.complete_s1_recall == APPROX(1.0)
        assert r.oracle_macro_f0_5 == APPROX(1.0)

    def test_partial_recall(self):
        truth = _truth(("S1-001", ["S2-001", "S2-002"]))
        cands = _cands(("S1-001", ["S2-001"]))  # misses S2-002
        r = evaluate_candidates(truth, cands)
        assert r.true_edge_recall == APPROX(0.5)
        assert r.complete_s1_recall == APPROX(0.0)
        # Oracle: predict S2-001 only; precision=1, recall=0.5
        expected_oracle = 1.25 * 1.0 * 0.5 / (0.25 + 0.5)
        assert r.oracle_macro_f0_5 == APPROX(expected_oracle)

    def test_retrieval_miss(self):
        """True ID absent from candidates = retrieval miss."""
        truth = _truth(("S1-001", ["S2-001"]))
        cands = _cands(("S1-001", ["S2-999"]))  # S2-001 not in cands
        r = evaluate_candidates(truth, cands)
        assert r.true_edge_recall == APPROX(0.0)
        assert r.oracle_macro_f0_5 == APPROX(0.0)

    def test_empty_candidates(self):
        """S1 with empty candidate list contributes to distribution."""
        truth = _truth(("S1-001", []))
        cands = _cands(("S1-001", []))
        r = evaluate_candidates(truth, cands)
        assert r.min_candidates == 0
        assert r.max_candidates == 0
        assert r.total_candidate_pairs == 0
        # True-empty oracle: both empty -> score 1
        assert r.oracle_macro_f0_5 == APPROX(1.0)

    def test_s2_s3_recall_distinguished(self):
        truth = _truth(
            ("S1-001", ["S2-001"]),
            ("S1-002", ["S3-002"]),
        )
        cands = _cands(
            ("S1-001", ["S2-001"]),  # S2 recalled
            ("S1-002", ["S2-999"]),  # S3-002 missed
        )
        r = evaluate_candidates(truth, cands)
        assert r.s2_true_edge_recall == APPROX(1.0)
        assert r.s3_true_edge_recall == APPROX(0.0)

    def test_s2_recall_with_shared_targets(self):
        """When multiple S1 link to the same S2 target, recall must never exceed 1.0."""
        truth = _truth(
            ("S1-001", ["S2-001"]),
            ("S1-002", ["S2-001"]),
        )
        cands = _cands(
            ("S1-001", ["S2-001"]),
            ("S1-002", ["S2-001"]),
        )
        r = evaluate_candidates(truth, cands)
        assert r.s2_true_edge_recall == APPROX(1.0)
        assert r.true_edge_recall == APPROX(1.0)

    def test_distribution_stats(self):
        truth = _truth(
            ("S1-001", ["S2-001"]),
            ("S1-002", ["S2-002"]),
            ("S1-003", ["S2-003"]),
        )
        cands = _cands(
            ("S1-001", ["S2-001"]),        # 1 candidate
            ("S1-002", ["S2-002", "S2-999"]),  # 2 candidates
            ("S1-003", []),                # 0 candidates
        )
        r = evaluate_candidates(truth, cands)
        assert r.min_candidates == 0
        assert r.max_candidates == 2
        assert r.total_candidate_pairs == 3
        assert r.mean_candidates == APPROX(1.0)

    def test_duplicate_candidate_ids_deduplicated(self):
        truth = _truth(("S1-001", ["S2-001"]))
        # frozenset() in evaluate_candidates deduplicates
        cands = [("S1-001", {"S2-001"})]
        r = evaluate_candidates(truth, cands)
        assert r.total_candidate_pairs == 1

    def test_reduction_ratio_computed(self):
        truth = _truth(("S1-001", ["S2-001"]))
        cands = _cands(("S1-001", ["S2-001", "S2-002"]))
        r = evaluate_candidates(truth, cands, corpus_size=1000)
        assert r.reduction_ratio == APPROX(2 / (1 * 1000))

    def test_missing_candidate_group_raises(self):
        """Missing candidate group for an S1 entity raises ValueError."""
        # Harshit probe:
        with pytest.raises(ValueError, match="Missing candidate groups"):
            evaluate_candidates([("S1-A", {"S2-X"})], [])

        truth = _truth(
            ("S1-001", ["S2-001"]),
            ("S1-002", ["S2-002"]),
        )
        cands = _cands(("S1-001", ["S2-001"]))  # S1-002 omitted
        with pytest.raises(ValueError, match="Missing candidate groups"):
            evaluate_candidates(truth, cands)

    def test_empty_candidate_group_valid_and_counted(self):
        """Candidate group with explicit empty candidate list is valid and counted."""
        truth = _truth(
            ("S1-001", ["S2-001"]),
            ("S1-002", ["S2-002"]),
        )
        cands = _cands(
            ("S1-001", ["S2-001"]),
            ("S1-002", []),  # explicitly empty
        )
        r = evaluate_candidates(truth, cands)
        assert r.evaluated_s1_count == 2
        assert r.true_edge_recall == APPROX(0.5)


# ---------------------------------------------------------------------------
# error_analysis() tests
# ---------------------------------------------------------------------------


class TestErrorAnalysis:
    """Tests for the error classification report."""

    def test_retrieval_miss_vs_scoring_fn(self):
        """Distinguish: retrieval miss (not in C) vs scoring FN (in C, not predicted)."""
        truth = _truth(("S1-001", ["S2-001", "S2-002"]))
        preds = _preds(("S1-001", []))
        cands = _cands(("S1-001", ["S2-001"]))  # S2-001 in C, S2-002 not

        report = error_analysis(truth, preds, cands)
        # S2-002 absent from candidates -> retrieval miss
        assert report.retrieval_miss.count == 1
        # S2-001 in candidates but not predicted -> scoring FN
        assert report.scoring_false_neg.count == 1

    def test_false_positive_classified(self):
        truth = _truth(("S1-001", ["S2-001"]))
        preds = _preds(("S1-001", ["S2-001", "S2-999"]))
        cands = _cands(("S1-001", ["S2-001", "S2-999"]))
        report = error_analysis(truth, preds, cands)
        assert report.false_positive.count == 1

    def test_wrong_singleton(self):
        truth = _truth(("S1-001", ["S2-001"]))
        preds = _preds(("S1-001", ["S2-002"]))
        cands = _cands(("S1-001", ["S2-001", "S2-002"]))
        report = error_analysis(truth, preds, cands)
        assert report.wrong_singleton.count == 1

    def test_empty_case_false_positive(self):
        truth = _truth(("S1-001", []))
        preds = _preds(("S1-001", ["S2-001"]))
        cands = _cands(("S1-001", ["S2-001"]))
        report = error_analysis(truth, preds, cands)
        assert report.empty_case_fp.count == 1

    def test_multi_match_partial_miss(self):
        truth = _truth(("S1-001", ["S2-001", "S2-002", "S2-003"]))
        preds = _preds(("S1-001", ["S2-001"]))
        cands = _cands(("S1-001", ["S2-001", "S2-002", "S2-003"]))
        report = error_analysis(truth, preds, cands)
        assert report.multi_match_partial.count == 1

    def test_summary_contains_all_categories(self):
        truth = _truth(("S1-001", []))
        preds = _preds(("S1-001", []))
        cands = _cands(("S1-001", []))
        report = error_analysis(truth, preds, cands)
        s = report.summary()
        assert "Retrieval miss" in s
        assert "Scoring false negative" in s
        assert "False positive" in s


# ---------------------------------------------------------------------------
# oracle_sanity_check() tests
# ---------------------------------------------------------------------------


class TestOracleSanityCheck:
    """Test oracle >= model constraint."""

    def test_oracle_ge_model_passes(self):
        """Model score below oracle passes."""
        truth = _truth(("S1-001", ["S2-001"]))
        preds = _preds(("S1-001", []))  # model predicts nothing
        cands = _cands(("S1-001", ["S2-001"]))
        eval_r = evaluate(truth, preds)
        cand_r = evaluate_candidates(truth, cands)
        passed, msg = oracle_sanity_check(eval_r, cand_r)
        assert passed
        assert "PASSED" in msg

    def test_oracle_equals_model_passes(self):
        """Model score equal to oracle passes (it should match oracle when
        all candidates are perfectly predicted)."""
        truth = _truth(("S1-001", ["S2-001"]))
        preds = _preds(("S1-001", ["S2-001"]))
        cands = _cands(("S1-001", ["S2-001"]))
        eval_r = evaluate(truth, preds)
        cand_r = evaluate_candidates(truth, cands)
        passed, msg = oracle_sanity_check(eval_r, cand_r)
        assert passed

    def test_fabricated_model_gt_oracle_fails(self):
        """Simulates an evaluation bug: model score > oracle should fail."""
        eval_r = EvaluationResult(
            macro_f0_5=0.9,
            mean_precision=0.9,
            mean_recall=0.9,
            evaluated_s1_count=1,
            true_empty_count=0,
            predicted_empty_count=0,
            false_positive_count=0,
            false_negative_count=0,
            singleton_correct=0,
            singleton_total=1,
        )
        cand_r = CandidateResult(
            true_edge_recall=0.5,
            complete_s1_recall=0.5,
            oracle_macro_f0_5=0.5,  # oracle lower than "model"
            total_candidate_pairs=1,
            min_candidates=1,
            mean_candidates=1.0,
            median_candidates=1.0,
            p95_candidates=1.0,
            p99_candidates=1.0,
            max_candidates=1,
        )
        passed, msg = oracle_sanity_check(eval_r, cand_r)
        assert not passed
        assert "FAILED" in msg


# ---------------------------------------------------------------------------
# france_coverage_check() tests
# ---------------------------------------------------------------------------


class TestFranceCoverageCheck:
    """France coverage check: coverage only, no F0.5."""

    def test_all_france_covered(self):
        test_ids = ["S1-001", "S1-002", "S1-003"]
        out_ids = ["S1-001", "S1-002", "S1-003"]
        country_labels = {"S1-001": "France", "S1-002": "India", "S1-003": "France"}
        result = france_coverage_check(test_ids, out_ids, country_labels)
        assert result["france_total"] == 2
        assert result["france_in_output"] == 2
        assert result["france_missing"] == 0
        assert result["missing_count"] == 0

    def test_france_missing_from_output(self):
        test_ids = ["S1-001", "S1-002"]
        out_ids = ["S1-001"]
        country_labels = {"S1-001": "France", "S1-002": "France"}
        result = france_coverage_check(test_ids, out_ids, country_labels)
        assert result["france_missing"] == 1
        assert result["missing_count"] == 1

    def test_no_country_labels(self):
        """Without country labels, France totals are 0 (no invented score)."""
        test_ids = ["S1-001"]
        out_ids = ["S1-001"]
        result = france_coverage_check(test_ids, out_ids)
        assert result["france_total"] == 0
        assert result["france_in_output"] == 0

    def test_country_case_insensitive(self):
        test_ids = ["S1-001"]
        out_ids = ["S1-001"]
        country_labels = {"S1-001": "FRANCE"}
        result = france_coverage_check(test_ids, out_ids, country_labels)
        assert result["france_total"] == 1
        assert result["france_in_output"] == 1
