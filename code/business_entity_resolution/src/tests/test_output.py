"""
Tests for ber.output — Suresh workstream.

All fixtures are synthetic (no real business records).

Test cases covered:
  11. prediction outside candidate set -> ValueError
  12. missing S1 output row -> ValidationResult error
  13. extra S1 output row -> ValidationResult error
  20. deterministic output order
  22. comma-separated ID handling
  23. TSV parsing where second column contains commas (valid format)
  + Header validation, encoding, duplicate row detection, subset enforcement,
    France coverage in validation result.
"""

import io
import os
import tempfile
import pytest

from ber.output import (
    validate_outputs,
    write_outputs,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tsv(path: str, rows: list[str]) -> None:
    """Write rows (strings) to a UTF-8 TSV file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def _write_source1(path: str, ids: list[str]) -> None:
    rows = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    for eid in ids:
        rows.append(f"{eid}\tAcme Inc\t123 Main St\tUS")
    _write_tsv(path, rows)


def _write_matching(path: str, rows: list[tuple]) -> None:
    """rows: list of (s1_id, comma_ids_str)"""
    lines = ["source1_entity_id\tmatched_entity_ids"]
    for s1_id, ids_str in rows:
        lines.append(f"{s1_id}\t{ids_str}")
    _write_tsv(path, lines)


def _write_candidates(path: str, rows: list[tuple]) -> None:
    lines = ["source1_entity_id\tcandidate_entity_ids"]
    for s1_id, ids_str in rows:
        lines.append(f"{s1_id}\t{ids_str}")
    _write_tsv(path, lines)


# ---------------------------------------------------------------------------
# write_outputs() tests
# ---------------------------------------------------------------------------


class TestWriteOutputs:
    """Tests for the write_outputs() TSV writer."""

    def test_basic_write(self, tmp_path):
        test_s1 = ["S1-001", "S1-002"]
        candidates = {"S1-001": ["S2-001", "S3-002"], "S1-002": []}
        decisions = {"S1-001": ["S2-001"], "S1-002": []}
        mpath, cpath = write_outputs(test_s1, candidates, decisions, str(tmp_path))

        with open(mpath, encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert len(lines) == 3  # header + 2 rows

        with open(cpath, encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert lines[0] == "source1_entity_id\tcandidate_entity_ids"

    def test_empty_row_preserved(self, tmp_path):
        """S1 with no matches must have a row with empty second column."""
        test_s1 = ["S1-001"]
        candidates = {"S1-001": []}  # explicit empty list required
        decisions = {"S1-001": []}
        mpath, _ = write_outputs(test_s1, candidates, decisions, str(tmp_path))
        with open(mpath, encoding="utf-8") as f:
            lines = f.read().splitlines()
        # Row: "S1-001\t"
        assert lines[1] == "S1-001\t"

    def test_preserves_input_order(self, tmp_path):
        """S1 row order must preserve test_s1 input order; ID lists within rows must be sorted."""
        test_s1 = ["S1-003", "S1-001", "S1-002"]
        candidates = {
            "S1-001": ["S2-002", "S2-001"],
            "S1-002": [],
            "S1-003": ["S3-001"],
        }
        decisions = {
            "S1-001": ["S2-002", "S2-001"],
            "S1-002": [],
            "S1-003": ["S3-001"],
        }
        mpath, _ = write_outputs(test_s1, candidates, decisions, str(tmp_path))
        with open(mpath, encoding="utf-8") as f:
            lines = f.read().splitlines()[1:]  # skip header
        s1_ids = [line.split("\t")[0] for line in lines]
        assert s1_ids == test_s1, "S1 IDs must match input test_s1 order exactly"
        # Check IDs within row sorted
        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) > 1 and parts[1]:
                ids = parts[1].split(",")
                assert ids == sorted(ids), f"IDs in {parts[0]} must be sorted"

    def test_duplicate_ids_raise_in_write_outputs(self, tmp_path):
        """Duplicate IDs in predictions or candidates raise ValueError."""
        test_s1 = ["S1-001"]
        candidates = {"S1-001": ["S2-001", "S2-001", "S2-002"]}
        decisions = {"S1-001": ["S2-001"]}
        with pytest.raises(ValueError, match="Duplicate candidate IDs"):
            write_outputs(test_s1, candidates, decisions, str(tmp_path))

        candidates_ok = {"S1-001": ["S2-001", "S2-002"]}
        decisions_dup = {"S1-001": ["S2-001", "S2-001"]}
        with pytest.raises(ValueError, match="Duplicate decision IDs"):
            write_outputs(test_s1, candidates_ok, decisions_dup, str(tmp_path))

    def test_prediction_outside_candidates_raises(self, tmp_path):
        """Prediction not in candidates raises ValueError."""
        test_s1 = ["S1-001"]
        candidates = {"S1-001": ["S2-001"]}
        decisions = {"S1-001": ["S2-001", "S2-999"]}  # S2-999 not in candidates
        with pytest.raises(ValueError, match="not subset of candidates"):
            write_outputs(test_s1, candidates, decisions, str(tmp_path))

    def test_comma_separated_in_second_column(self, tmp_path):
        """Multiple IDs are comma-separated in the second TSV column."""
        test_s1 = ["S1-001"]
        candidates = {"S1-001": ["S2-001", "S3-002", "S2-003"]}
        decisions = {"S1-001": ["S2-001", "S3-002"]}
        mpath, _ = write_outputs(test_s1, candidates, decisions, str(tmp_path))
        with open(mpath, encoding="utf-8") as f:
            line = f.read().splitlines()[1]
        s1_id, ids_str = line.split("\t")
        ids = ids_str.split(",")
        assert len(ids) == 2
        assert "S2-001" in ids
        assert "S3-002" in ids

    def test_invalid_id_prefix_raises(self, tmp_path):
        """IDs without S2-/S3- prefix raise ValueError."""
        test_s1 = ["S1-001"]
        candidates = {"S1-001": ["S2-001", "S1-BAD"]}
        decisions = {"S1-001": ["S2-001"]}
        with pytest.raises(ValueError, match="Invalid candidate entity ID"):
            write_outputs(test_s1, candidates, decisions, str(tmp_path))

    def test_atomic_write_cleans_up_on_failure(self, tmp_path):
        """Failed write leaves no partial destination or temporary files."""
        test_s1 = ["S1-001"]
        candidates = {"S1-001": ["S2-001"]}
        decisions = {"S1-001": ["S2-002"]}  # invalid subset!
        with pytest.raises(ValueError):
            write_outputs(test_s1, candidates, decisions, str(tmp_path))
        assert not (tmp_path / "matching_results.tsv").exists()
        assert not (tmp_path / "candidate_pairs.tsv").exists()
        assert not (tmp_path / ".matching_results.tsv.tmp").exists()
        assert not (tmp_path / ".candidate_pairs.tsv.tmp").exists()

    def test_utf8_encoding(self, tmp_path):
        """Output files must be valid UTF-8."""
        test_s1 = ["S1-\u4e2d\u6587"]
        candidates = {"S1-\u4e2d\u6587": ["S2-001"]}
        decisions = {"S1-\u4e2d\u6587": ["S2-001"]}
        mpath, cpath = write_outputs(test_s1, candidates, decisions, str(tmp_path))
        # Must be readable as UTF-8
        with open(mpath, encoding="utf-8") as f:
            content = f.read()
        assert "S1-\u4e2d\u6587" in content

    def test_stream_extra_candidate_group_raises(self, tmp_path):
        """Extra candidate group at the end of the candidate stream must raise ValueError."""
        test_s1 = ["S1-001"]
        cand_stream = (item for item in [("S1-001", ["S2-001"]), ("S1-EXTRA", ["S2-002"])])
        dec_stream = (item for item in [("S1-001", ["S2-001"])])

        with pytest.raises(ValueError, match=r"(?i)extra candidate group"):
            write_outputs(test_s1, cand_stream, dec_stream, str(tmp_path))
        assert not (tmp_path / "matching_results.tsv").exists()
        assert not (tmp_path / "candidate_pairs.tsv").exists()

    def test_stream_extra_decision_group_raises(self, tmp_path):
        """Extra decision group at the end of the decision stream must raise ValueError."""
        test_s1 = ["S1-001"]
        cand_stream = (item for item in [("S1-001", ["S2-001"])])
        dec_stream = (item for item in [("S1-001", ["S2-001"]), ("S1-EXTRA", ["S2-001"])])

        with pytest.raises(ValueError, match=r"(?i)extra decision group"):
            write_outputs(test_s1, cand_stream, dec_stream, str(tmp_path))
        assert not (tmp_path / "matching_results.tsv").exists()
        assert not (tmp_path / "candidate_pairs.tsv").exists()

    def test_mapping_extra_candidate_key_raises(self, tmp_path):
        """Extra S1 ID in candidates dict not in test_s1 must raise ValueError."""
        test_s1 = ["S1-001"]
        candidates = {"S1-001": ["S2-001"], "S1-EXTRA": ["S2-002"]}
        decisions = {"S1-001": ["S2-001"]}

        with pytest.raises(ValueError, match=r"(?i)extra candidate group"):
            write_outputs(test_s1, candidates, decisions, str(tmp_path))
        assert not (tmp_path / "matching_results.tsv").exists()
        assert not (tmp_path / "candidate_pairs.tsv").exists()

    def test_mapping_extra_decision_key_raises(self, tmp_path):
        """Extra S1 ID in decisions dict not in test_s1 must raise ValueError."""
        test_s1 = ["S1-001"]
        candidates = {"S1-001": ["S2-001"]}
        decisions = {"S1-001": ["S2-001"], "S1-EXTRA": ["S2-001"]}

        with pytest.raises(ValueError, match=r"(?i)extra decision group"):
            write_outputs(test_s1, candidates, decisions, str(tmp_path))
        assert not (tmp_path / "matching_results.tsv").exists()
        assert not (tmp_path / "candidate_pairs.tsv").exists()

    def test_mapping_missing_candidate_entry_raises(self, tmp_path):
        """S1 ID in test_s1 but absent from candidates dict must raise ValueError."""
        test_s1 = ["S1-001", "S1-002"]
        candidates = {"S1-001": ["S2-001"]}  # S1-002 is missing
        decisions = {"S1-001": ["S2-001"], "S1-002": []}

        with pytest.raises(ValueError, match=r"(?i)candidates mapping missing entry"):
            write_outputs(test_s1, candidates, decisions, str(tmp_path))
        assert not (tmp_path / "matching_results.tsv").exists()
        assert not (tmp_path / "candidate_pairs.tsv").exists()

    def test_mapping_missing_decision_entry_raises(self, tmp_path):
        """S1 ID in test_s1 but absent from decisions dict must raise ValueError."""
        test_s1 = ["S1-001", "S1-002"]
        candidates = {"S1-001": ["S2-001"], "S1-002": ["S2-002"]}
        decisions = {"S1-001": ["S2-001"]}  # S1-002 is missing

        with pytest.raises(ValueError, match=r"(?i)decisions mapping missing entry"):
            write_outputs(test_s1, candidates, decisions, str(tmp_path))
        assert not (tmp_path / "matching_results.tsv").exists()
        assert not (tmp_path / "candidate_pairs.tsv").exists()


# ---------------------------------------------------------------------------
# validate_outputs() tests
# ---------------------------------------------------------------------------


class TestValidateOutputs:
    """Tests for the preflight validator."""

    def test_valid_outputs_pass(self, tmp_path):
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001", "S1-002"])
        _write_matching(m_path, [("S1-001", "S2-001"), ("S1-002", "")])
        _write_candidates(c_path, [("S1-001", "S2-001,S2-002"), ("S1-002", "")])

        result = validate_outputs(s1_path, m_path, c_path)
        assert result.passed, result.summary()
        assert result.total_test_s1 == 2
        assert result.matching_rows == 2

    def test_missing_s1_row_detected(self, tmp_path):
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001", "S1-002"])
        _write_matching(m_path, [("S1-001", "S2-001")])  # S1-002 missing
        _write_candidates(c_path, [("S1-001", "S2-001"), ("S1-002", "")])

        result = validate_outputs(s1_path, m_path, c_path)
        assert not result.passed
        problem_types = [e.problem_type for e in result.errors]
        assert "MISSING_S1_ROWS" in problem_types

    def test_extra_s1_row_detected(self, tmp_path):
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001"])
        _write_matching(m_path, [("S1-001", "S2-001"), ("S1-999", "")])  # S1-999 extra
        _write_candidates(c_path, [("S1-001", "S2-001")])

        result = validate_outputs(s1_path, m_path, c_path)
        assert not result.passed
        problem_types = [e.problem_type for e in result.errors]
        assert "EXTRA_S1_ROWS" in problem_types

    def test_wrong_header_detected(self, tmp_path):
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001"])
        _write_tsv(str(m_path), [
            "entity_id\tresults",  # wrong header
            "S1-001\tS2-001",
        ])
        _write_candidates(c_path, [("S1-001", "S2-001")])

        result = validate_outputs(s1_path, m_path, c_path)
        assert not result.passed
        problem_types = [e.problem_type for e in result.errors]
        assert "WRONG_HEADER" in problem_types

    def test_prediction_not_subset_of_candidates(self, tmp_path):
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001"])
        _write_matching(m_path, [("S1-001", "S2-001,S2-999")])  # S2-999 not in cands
        _write_candidates(c_path, [("S1-001", "S2-001")])

        result = validate_outputs(s1_path, m_path, c_path)
        assert not result.passed
        problem_types = [e.problem_type for e in result.errors]
        assert "PREDICTION_NOT_SUBSET_OF_CANDIDATES" in problem_types

    def test_empty_match_valid(self, tmp_path):
        """Empty match row is valid."""
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001"])
        _write_matching(m_path, [("S1-001", "")])
        _write_candidates(c_path, [("S1-001", "")])

        result = validate_outputs(s1_path, m_path, c_path)
        assert result.passed
        assert result.empty_matching_rows == 1

    def test_invalid_id_prefix_detected(self, tmp_path):
        """IDs without S2-/S3- prefix fail validation."""
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001"])
        _write_matching(m_path, [("S1-001", "S1-BAD")])
        _write_candidates(c_path, [("S1-001", "S1-BAD")])

        result = validate_outputs(s1_path, m_path, c_path)
        assert not result.passed
        problem_types = [e.problem_type for e in result.errors]
        assert "INVALID_ID_PREFIX" in problem_types

    def test_france_coverage_in_result(self, tmp_path):
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001", "S1-002"])
        _write_matching(m_path, [("S1-001", ""), ("S1-002", "")])
        _write_candidates(c_path, [("S1-001", ""), ("S1-002", "")])

        country_labels = {"S1-001": "France", "S1-002": "India"}
        result = validate_outputs(s1_path, m_path, c_path, country_labels=country_labels)
        assert result.passed
        assert result.france_total == 1
        assert result.france_in_output == 1

    def test_missing_s1_file_fails(self, tmp_path):
        result = validate_outputs(
            str(tmp_path / "nonexistent.tsv"),
            str(tmp_path / "matching.tsv"),
        )
        assert not result.passed
        problem_types = [e.problem_type for e in result.errors]
        assert "MISSING_TEST_S1_FILE" in problem_types

    def test_missing_matching_file_fails(self, tmp_path):
        s1_path = str(tmp_path / "test_source1.tsv")
        _write_source1(s1_path, ["S1-001"])
        result = validate_outputs(s1_path, str(tmp_path / "nonexistent.tsv"))
        assert not result.passed

    def test_candidate_missing_file_fails(self, tmp_path):
        """If candidate_pairs_path is specified but does not exist, validation must fail."""
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")

        _write_source1(s1_path, ["S1-001"])
        _write_matching(m_path, [("S1-001", "")])

        result = validate_outputs(
            s1_path, m_path,
            candidate_pairs_path=str(tmp_path / "nonexistent.tsv")
        )
        assert not result.passed
        assert any(e.problem_type == "MISSING_CANDIDATE_FILE" for e in result.errors)

    def test_candidate_mandatory_by_default(self, tmp_path):
        """Omitting candidate_pairs_path must fail preflight validation by default."""
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")

        _write_source1(s1_path, ["S1-001"])
        _write_matching(m_path, [("S1-001", "")])

        result = validate_outputs(s1_path, m_path)
        assert not result.passed
        assert any(e.problem_type == "MISSING_CANDIDATE_FILE" for e in result.errors)

    def test_duplicate_target_ids_in_tsv_detected(self, tmp_path):
        """Duplicate target IDs in matching_results.tsv or candidate_pairs.tsv must fail."""
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001"])
        _write_matching(m_path, [("S1-001", "S2-001,S2-001")])
        _write_candidates(c_path, [("S1-001", "S2-001")])

        result = validate_outputs(s1_path, m_path, c_path)
        assert not result.passed
        problem_types = [e.problem_type for e in result.errors]
        assert "DUPLICATE_IDS" in problem_types

    def test_tsv_with_comma_in_id_list_parsed_correctly(self, tmp_path):
        """Comma inside the second TSV column is the ID separator, not a field
        separator; TSV field separator is TAB."""
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001"])
        # Second column contains comma-separated IDs (valid TSV)
        _write_tsv(str(m_path), [
            "source1_entity_id\tmatched_entity_ids",
            "S1-001\tS2-001,S3-002",
        ])
        _write_tsv(str(c_path), [
            "source1_entity_id\tcandidate_entity_ids",
            "S1-001\tS2-001,S3-002",
        ])

        result = validate_outputs(s1_path, m_path, c_path)
        assert result.passed, result.summary()

    def test_malformed_one_column_row_detected(self, tmp_path):
        """A row missing a tab separator (single column) is malformed and must fail validation."""
        s1_path = str(tmp_path / "test_source1.tsv")
        m_path = str(tmp_path / "matching_results.tsv")
        c_path = str(tmp_path / "candidate_pairs.tsv")

        _write_source1(s1_path, ["S1-001"])
        # Missing \t separator - single column only
        _write_tsv(str(m_path), [
            "source1_entity_id\tmatched_entity_ids",
            "S1-001",
        ])
        _write_candidates(c_path, [("S1-001", "")])

        result = validate_outputs(s1_path, m_path, c_path)
        assert not result.passed
        problem_types = [e.problem_type for e in result.errors]
        assert "MALFORMED_ROW" in problem_types
