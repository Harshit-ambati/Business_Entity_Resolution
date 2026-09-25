"""
test_e2e_synthetic.py - Small end-to-end synthetic fixture test.

Tests the complete pipeline from write_outputs -> validate_outputs ->
evaluate metrics using only synthetic data. No real business records.

This is the "M0 fixture" that must pass before any full-scale run.
"""

import os
import pytest

from ber.metrics import (
    evaluate,
    evaluate_candidates,
    error_analysis,
    oracle_sanity_check,
    france_coverage_check,
)
from ber.output import write_outputs, validate_outputs


# ---------------------------------------------------------------------------
# Synthetic fixture data
# ---------------------------------------------------------------------------

# Test S1 IDs
TEST_S1 = [
    "S1-001",  # exact singleton (US)
    "S1-002",  # multi-match (India)
    "S1-003",  # true empty (no matches)
    "S1-004",  # missed by model
    "S1-005",  # France (coverage only)
]

# Ground truth
TRUTH = {
    "S1-001": {"S2-001"},
    "S1-002": {"S2-002", "S3-003"},
    "S1-003": set(),
    "S1-004": {"S2-004"},
    "S1-005": {"S2-005"},
}

# Candidate set (what blocking generates)
CANDIDATES = {
    "S1-001": {"S2-001", "S2-999"},   # S2-001 in cands; S2-999 is distractor
    "S1-002": {"S2-002", "S3-003"},   # all true links present
    "S1-003": set(),                   # empty candidates (no matches)
    "S1-004": set(),                   # retrieval miss: S2-004 not in cands
    "S1-005": {"S2-005"},             # France S1
}

# Model decisions (subset of candidates)
DECISIONS = {
    "S1-001": {"S2-001"},             # correct
    "S1-002": {"S2-002"},             # partial (missed S3-003)
    "S1-003": set(),                   # correct empty
    "S1-004": set(),                   # missed (retrieval miss)
    "S1-005": {"S2-005"},             # correct
}

COUNTRY_LABELS = {
    "S1-001": "United States",
    "S1-002": "India",
    "S1-003": "United States",
    "S1-004": "India",
    "S1-005": "France",
}


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


class TestEndToEndSynthetic:
    def test_write_and_validate_outputs(self, tmp_path):
        """write_outputs -> validate_outputs must produce passing result."""
        mpath, cpath = write_outputs(
            TEST_S1, CANDIDATES, DECISIONS, str(tmp_path)
        )

        # Create a synthetic test_source1.tsv
        s1_tsv = str(tmp_path / "test_source1.tsv")
        with open(s1_tsv, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for s1_id in TEST_S1:
                country = COUNTRY_LABELS.get(s1_id, "Unknown")
                f.write(f"{s1_id}\tAcme Inc\t123 Main St\t{country}\n")

        result = validate_outputs(
            s1_tsv, mpath, cpath, country_labels=COUNTRY_LABELS
        )
        assert result.passed, result.summary()
        assert result.total_test_s1 == 5
        assert result.matching_rows == 5
        assert result.candidate_rows == 5
        assert result.france_total == 1
        assert result.france_in_output == 1

    def test_evaluate_metrics(self):
        """Evaluate model predictions against synthetic truth."""
        truth_iter = list(TRUTH.items())
        pred_iter = list(DECISIONS.items())

        result = evaluate(truth_iter, pred_iter)

        # S1-001: perfect -> 1.0
        # S1-002: partial (precision=1, recall=0.5) -> F0.5 = 0.833...
        # S1-003: both empty -> 1.0
        # S1-004: true non-empty, pred empty -> 0.0
        # S1-005: perfect -> 1.0
        expected_s1_001 = 1.0
        expected_s1_002 = 1.25 * 1.0 * 0.5 / (0.25 + 0.5)
        expected_s1_003 = 1.0
        expected_s1_004 = 0.0
        expected_s1_005 = 1.0
        expected_macro = (
            expected_s1_001 + expected_s1_002 + expected_s1_003
            + expected_s1_004 + expected_s1_005
        ) / 5

        assert result.evaluated_s1_count == 5
        assert result.macro_f0_5 == pytest.approx(expected_macro, abs=1e-6)
        assert result.true_empty_count == 1  # S1-003
        # Contract singleton = true_empty. Only S1-003 qualifies (both pred empty -> correct).
        assert result.singleton_total == 1
        assert result.singleton_correct == 1

    def test_candidate_metrics(self):
        """Candidate retrieval diagnostics."""
        truth_iter = list(TRUTH.items())
        cand_iter = list(CANDIDATES.items())

        result = evaluate_candidates(truth_iter, cand_iter)

        # Total true edges: 1+2+0+1+1 = 5
        # Recalled: S1-001:1, S1-002:2, S1-003:0, S1-004:0, S1-005:1 = 4
        assert result.total_true_edges == 5
        assert result.true_edges_recalled == 4
        assert result.true_edge_recall == pytest.approx(4 / 5)

        # Complete-S1 recall: matched S1s with all true in C
        # S1-001: all in cands -> yes; S1-002: all in cands -> yes;
        # S1-004: S2-004 not in cands -> no; S1-005: yes
        # matched_s1_total = 4 (S1-003 excluded as true-empty)
        assert result.complete_s1_recall == pytest.approx(3 / 4)

    def test_oracle_sanity_check_passes(self):
        """Oracle score >= model score in this fixture."""
        truth_iter = list(TRUTH.items())
        pred_iter = list(DECISIONS.items())
        cand_iter = list(CANDIDATES.items())

        eval_r = evaluate(truth_iter, pred_iter)
        cand_r = evaluate_candidates(truth_iter, cand_iter)
        passed, msg = oracle_sanity_check(eval_r, cand_r)
        assert passed, msg

    def test_error_analysis(self):
        """Error classification on the synthetic fixture."""
        truth_iter = list(TRUTH.items())
        pred_iter = list(DECISIONS.items())
        cand_iter = list(CANDIDATES.items())

        report = error_analysis(truth_iter, pred_iter, cand_iter)

        # S1-004: S2-004 not in candidates -> retrieval miss
        assert report.retrieval_miss.count >= 1
        # S1-002: S3-003 is in candidates but not predicted -> scoring FN
        assert report.scoring_false_neg.count >= 1

    def test_france_coverage_fixture(self):
        """France S1 (S1-005) must appear in output coverage check."""
        test_ids = TEST_S1
        out_ids = list(DECISIONS.keys())
        result = france_coverage_check(test_ids, out_ids, COUNTRY_LABELS)
        assert result["france_total"] == 1
        assert result["france_missing"] == 0
