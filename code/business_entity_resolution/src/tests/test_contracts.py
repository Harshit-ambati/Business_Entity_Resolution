"""Tests for shared in-memory contracts and deliberately fake fixture rows."""

import csv
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ber import Candidate, CandidateGroup, Decision, NormalizedRecord, Record, TruthRow


FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name: str) -> list[dict[str, str]]:
    with (FIXTURES / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_construct_shared_types_and_immutable_routes():
    raw = Record("S1-MULTI", "Blue Lantern Bakery", "12 River Road", "United States")
    norm = NormalizedRecord(raw, "blue lantern bakery", "12 river road", ("blue", "lantern", "bakery"), ("12", "river", "road"), "united states")
    truth = TruthRow("S1-MULTI", ("S2-MATCH", "S3-MATCH"))
    scores = {"exact_name": 1.0}
    candidate = Candidate("S2-MATCH", ("exact_name",), scores)
    group = CandidateGroup(raw.entity_id, (candidate,))
    decision = Decision(raw.entity_id, truth.matched_entity_ids)
    scores["exact_name"] = 0.0  # Constructor makes a defensive, read-only copy.
    assert candidate.route_scores["exact_name"] == 1.0
    with pytest.raises(TypeError):
        candidate.route_scores["exact_name"] = 0.0
    with pytest.raises(FrozenInstanceError):
        raw.business_name = "changed"
    assert norm.raw is raw
    assert group.candidates == (candidate,)
    assert decision.matched_entity_ids == truth.matched_entity_ids


@pytest.mark.parametrize("bad_id", ["X1-123", "S1-", "S1", ""])
def test_record_bad_prefix(bad_id):
    with pytest.raises(ValueError, match="Invalid entity ID"):
        Record(bad_id, "Name", "", "France")


def test_source_roles_duplicates_and_empty_groups():
    candidate = Candidate("S3-C", ("name",), {"name": 0.5})
    assert CandidateGroup("S1-Q", ()).candidates == ()
    assert Decision("S1-Q", ()).matched_entity_ids == ()
    assert TruthRow("S1-Q", ()).matched_entity_ids == ()
    with pytest.raises(ValueError):
        TruthRow("S2-Q", ())
    with pytest.raises(ValueError):
        Decision("S1-Q", ("S1-WRONG",))
    with pytest.raises(ValueError):
        Candidate("S1-WRONG", ("name",), {})
    with pytest.raises(ValueError, match="Duplicate candidate"):
        CandidateGroup("S1-Q", (candidate, candidate))
    with pytest.raises(ValueError, match="Duplicate matched"):
        TruthRow("S1-Q", ("S2-A", "S2-A"))
    with pytest.raises(ValueError, match="Duplicate matched"):
        Decision("S1-Q", ("S3-A", "S3-A"))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_candidate_rejects_nonfinite_scores(score):
    with pytest.raises(ValueError, match="finite"):
        Candidate("S2-A", ("name",), {"name": score})


def test_candidate_requires_sorted_unique_routes():
    with pytest.raises(ValueError, match="sorted unique"):
        Candidate("S2-A", ("z", "a"), {})
    with pytest.raises(ValueError, match="sorted unique"):
        Candidate("S2-A", ("name", "name"), {})


def test_candidate_route_scores_can_be_omitted():
    candidate = Candidate("S2-A", ("name",))
    assert dict(candidate.route_scores) == {}
    with pytest.raises(TypeError):
        candidate.route_scores["name"] = 1.0


def test_synthetic_fixture_covers_required_cases():
    s1 = _rows("source1.tsv")
    s2 = _rows("source2.tsv")
    s3 = _rows("source3.tsv")
    truth = {row["source1_entity_id"]: row["matched_entity_ids"] for row in _rows("truth.tsv")}
    assert len(s1) == len(truth) == 4
    assert set(truth["S1-MULTI"].split(",")) == {"S2-MATCH", "S3-MATCH"}
    assert truth["S1-SINGLE"] == ""
    assert any(row["entity_id"] == "S2-HARD-NEGATIVE" and "Blue Lantern Bakery" in row["business_name"] for row in s2)
    assert any(row["country"] == "France" for row in s1)
    assert any("さくら" in row["business_name"] for row in s1 + s2)
    assert any(row["business_address"] == "" for row in s1)
    assert {"S2-MATCH", "S2-HARD-NEGATIVE"}.issubset({row["entity_id"] for row in s2})
    assert "S3-MATCH" in {row["entity_id"] for row in s3}


def test_synthetic_fixture_passes_through_contracts():
    records = [Record(**row) for name in ("source1.tsv", "source2.tsv", "source3.tsv") for row in _rows(name)]
    truth_rows = [
        TruthRow(row["source1_entity_id"], tuple(filter(None, row["matched_entity_ids"].split(","))))
        for row in _rows("truth.tsv")
    ]
    by_id = {record.entity_id: record for record in records}
    for truth in truth_rows:
        assert truth.source1_entity_id in by_id
        group = CandidateGroup(
            truth.source1_entity_id,
            tuple(Candidate(entity_id, ("fixture",), {"fixture": 1.0}) for entity_id in truth.matched_entity_ids),
        )
        decision = Decision(truth.source1_entity_id, truth.matched_entity_ids)
        assert set(decision.matched_entity_ids) == {candidate.candidate_entity_id for candidate in group.candidates}
