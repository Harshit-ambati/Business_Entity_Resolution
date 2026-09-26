"""Unit tests for Sabeena candidate retrieval foundation.

Verifies:
1. Exact business name match
2. Different capitalization
3. Extra punctuation
4. Extra whitespace
5. One Source-1 entity with multiple candidates
6. Source-1 entity with no candidates (singleton)
7. Duplicate candidate generation & deduplication
8. Source identity preservation (S2 vs S3)
9. candidate_pairs.tsv contract formatting
10. Large block safeguard (high-frequency name threshold)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from ber.candidate_retrieval import (
    CandidateRetrievalConfig,
    build_index,
    open_index,
    retrieve_exact_name_candidates,
    write_candidate_pairs_tsv,
    evaluate_retrieval,
    deduplicate_candidates,
)
from ber.contracts import Candidate, CandidateGroup


@pytest.fixture
def synthetic_retrieval_corpus(tmp_path: Path):
    """Create synthetic S1, S2, S3, and truth files covering all 7 test cases."""
    s1_path = tmp_path / "source1.tsv"
    s2_path = tmp_path / "source2.tsv"
    s3_path = tmp_path / "source3.tsv"
    truth_path = tmp_path / "truth.tsv"

    # Case 1: Exact business name -> S1-EXACT matches S2-EXACT
    # Case 2: Different capitalization -> S1-CAPS ("ALPHA OMEGA TECH") matches S2-LOWER ("alpha omega tech")
    # Case 3: Extra punctuation & ampersand -> S1-PUNCT ("Bella's & Bob's Cafe, Inc.") matches S3-CLEAN ("Bella's and Bob's Cafe Inc")
    # Case 4: Extra whitespace -> S1-SPACE ("  Delta   Logistics   ") matches S2-NORMAL ("Delta Logistics")
    # Case 5: Multiple candidates -> S1-MULTI ("Global Trade & Co") matches S2-MULTI1 ("global trade and co") and S3-MULTI2 ("Global Trade & Co.")
    # Case 6: No candidates -> S1-EMPTY ("Unique Standalone Boutique") matches nothing
    # Case 7: Duplicate candidates handled via route union/dedup
    s1_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-EXACT\tAcme Corporation\t100 Main St\tUS\n"
        "S1-CAPS\tALPHA OMEGA TECH\t200 Innovation Way\tIndia\n"
        "S1-PUNCT\tBella's & Bob's Cafe, Inc.\t300 Market St\tUS\n"
        "S1-SPACE\t  Delta   Logistics   \t400 Cargo Rd\tIndia\n"
        "S1-MULTI\tGlobal Trade & Co\t500 Harbor Blvd\tUS\n"
        "S1-EMPTY\tUnique Standalone Boutique\t600 Fashion Ave\tFrance\n"
    )
    s1_path.write_text(s1_content, encoding="utf-8")

    s2_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-EXACT\tacme corporation\t100 Main St\tUS\n"
        "S2-LOWER\talpha omega tech\t200 Innovation Way\tIndia\n"
        "S2-NORMAL\tdelta logistics\t400 Cargo Rd\tIndia\n"
        "S2-MULTI1\tglobal trade and co\t500 Harbor Blvd\tUS\n"
    )
    s2_path.write_text(s2_content, encoding="utf-8")

    s3_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-CLEAN\tBella's and Bob's Cafe Inc\t300 Market St\tUS\n"
        "S3-MULTI2\tGlobal Trade & Co.\t500 Harbor Blvd\tUS\n"
    )
    s3_path.write_text(s3_content, encoding="utf-8")

    truth_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-EXACT\tS2-EXACT\n"
        "S1-CAPS\tS2-LOWER\n"
        "S1-PUNCT\tS3-CLEAN\n"
        "S1-SPACE\tS2-NORMAL\n"
        "S1-MULTI\tS2-MULTI1,S3-MULTI2\n"
        "S1-EMPTY\t\n"
    )
    truth_path.write_text(truth_content, encoding="utf-8")

    return s1_path, s2_path, s3_path, truth_path


def test_baseline_exact_name_blocking_all_cases(synthetic_retrieval_corpus, tmp_path: Path):
    """Test all 7 required baseline cases: exact, caps, punct, space, multi, empty, dedup."""
    s1_path, s2_path, s3_path, truth_path = synthetic_retrieval_corpus
    work_dir = tmp_path / "blocking_idx"

    # Build index from S2 and S3
    manifest = build_index(s2_path, s3_path, work_dir)
    store = open_index(manifest)

    config = CandidateRetrievalConfig(max_candidates_per_entity=10)
    groups_gen, diag = retrieve_exact_name_candidates(s1_path, store, config)
    groups = list(groups_gen)

    assert len(groups) == 6
    group_map = {g.source1_entity_id: [c.candidate_entity_id for c in g.candidates] for g in groups}

    # Case 1: Exact match
    assert "S2-EXACT" in group_map["S1-EXACT"]

    # Case 2: Different capitalization
    assert "S2-LOWER" in group_map["S1-CAPS"]

    # Case 3: Punctuation and ampersand normalization
    assert "S3-CLEAN" in group_map["S1-PUNCT"]

    # Case 4: Extra whitespace
    assert "S2-NORMAL" in group_map["S1-SPACE"]

    # Case 5: Multiple candidates for one S1
    assert "S2-MULTI1" in group_map["S1-MULTI"]
    assert "S3-MULTI2" in group_map["S1-MULTI"]
    assert len(group_map["S1-MULTI"]) == 2

    # Case 6: Empty candidate group (singleton)
    assert group_map["S1-EMPTY"] == []

    # Case 7: Deterministic candidate ordering & source identity preservation
    for c in groups[0].candidates:
        assert c.candidate_entity_id.startswith(("S2-", "S3-"))
        assert "exact_name" in c.retrieval_routes

    store.close()


def test_deduplication_removes_duplicate_candidates():
    """Verify deduplicate_candidates removes duplicate candidates deterministically."""
    c1 = Candidate("S2-001", ("exact_name",), {"exact_name": 1.0})
    c2 = Candidate("S2-001", ("exact_name",), {"exact_name": 1.0})
    c3 = Candidate("S3-002", ("exact_name",), {"exact_name": 1.0})

    deduped = deduplicate_candidates([c1, c2, c3], cap=10)
    assert len(deduped) == 2
    assert [c.candidate_entity_id for c in deduped] == ["S2-001", "S3-002"]


def test_write_candidate_pairs_tsv_contract(synthetic_retrieval_corpus, tmp_path: Path):
    """Verify output/candidate_pairs.tsv formatting strictly follows CONTRACTS.md §4."""
    s1_path, s2_path, s3_path, _ = synthetic_retrieval_corpus
    work_dir = tmp_path / "blocking_idx"
    manifest = build_index(s2_path, s3_path, work_dir)
    store = open_index(manifest)

    groups_gen, _ = retrieve_exact_name_candidates(s1_path, store)
    out_tsv = tmp_path / "output" / "candidate_pairs.tsv"
    rows_written = write_candidate_pairs_tsv(groups_gen, out_tsv)

    assert rows_written == 6
    assert out_tsv.exists()

    raw_lines = out_tsv.read_text(encoding="utf-8").splitlines()
    assert raw_lines[0] == "source1_entity_id\tcandidate_entity_ids"

    # Check that empty matches are emitted as empty second cell
    s1_empty_line = [l for l in raw_lines if l.startswith("S1-EMPTY")][0]
    parts = s1_empty_line.split("\t")
    assert parts[0] == "S1-EMPTY"
    assert parts[1] == ""

    # Check multi matches are comma separated
    s1_multi_line = [l for l in raw_lines if l.startswith("S1-MULTI")][0]
    multi_parts = s1_multi_line.split("\t")[1].split(",")
    assert set(multi_parts) == {"S2-MULTI1", "S3-MULTI2"}

    store.close()


def test_large_block_safeguard(tmp_path: Path):
    """Verify large block safety truncates high-frequency normalized names and tracks avoided candidates."""
    s1 = tmp_path / "source1.tsv"
    s2 = tmp_path / "source2.tsv"
    s3 = tmp_path / "source3.tsv"

    s1.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-COMMON\tGeneral Store\t100 Main St\tUS\n", encoding="utf-8")

    # S2 has 20 entities with the same name "General Store"
    s2_rows = ["entity_id\tbusiness_name\tbusiness_address\tcountry\n"]
    for i in range(20):
        s2_rows.append(f"S2-{i}\tGeneral Store\t{i} Market St\tUS\n")
    s2.write_text("".join(s2_rows), encoding="utf-8")
    s3.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n", encoding="utf-8")

    manifest = build_index(s2, s3, tmp_path / "idx")
    store = open_index(manifest)

    # Set max_block_size = 5
    config = CandidateRetrievalConfig(max_block_size=5, max_candidates_per_entity=5)
    groups_gen, diag = retrieve_exact_name_candidates(s1, store, config)
    groups = list(groups_gen)

    assert len(groups[0].candidates) == 5
    assert diag["oversized_blocks"] == 1
    assert diag["candidates_avoided"] == 15

    store.close()


def test_metrics_evaluation(synthetic_retrieval_corpus, tmp_path: Path):
    """Verify evaluate_retrieval accurately computes candidate volumes, recall, and report strings."""
    s1_path, s2_path, s3_path, truth_path = synthetic_retrieval_corpus
    manifest = build_index(s2_path, s3_path, tmp_path / "idx")
    store = open_index(manifest)

    groups_gen, diag = retrieve_exact_name_candidates(s1_path, store)
    metrics, _ = evaluate_retrieval(
        candidate_groups=groups_gen,
        truth_path=truth_path,
        total_indexed=store.total_records,
        source2_count=4,
        source3_count=2,
        diagnostics=diag,
        runtime_seconds=0.05,
        peak_process_ram_mb=45.0,
        peak_python_heap_mb=2.5,
    )

    assert metrics.source1_count == 6
    assert metrics.entities_with_candidates == 5
    assert metrics.entities_without_candidates == 1
    assert metrics.blocking_recall == 1.0  # All true links were exact normalized matches
    assert metrics.duplicate_pairs == 0

    report = metrics.format_report(tests_passed=1, tests_failed=0)
    assert "### SABEENA BASELINE RESULT" in report
    assert "* Source-1 records processed: 6" in report
    assert "* Blocking recall: 100.00%" in report

    store.close()
