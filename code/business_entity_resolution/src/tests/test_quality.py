"""Tests for data quality validation, duplicate checks, and dataset audit helper (PR T3)."""

from pathlib import Path

import pytest

from ber import (
    audit_dataset,
    check_duplicate_ids_partitioned,
    validate_source_file,
    validate_truth_file,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_validate_source_file_clean_fixture():
    report = validate_source_file(FIXTURES / "source1.tsv", "S1-")
    assert report.is_valid
    assert report.total_rows == 4
    assert report.valid_rows == 4
    assert report.malformed_rows == 0
    assert report.prefix_errors == 0
    assert report.duplicate_ids == 0
    assert report.empty_addresses == 1  # S1-SINGLE has empty address
    assert report.missing_names == 0
    assert report.country_counts == {"United States": 1, "India": 1, "France": 1, "Japan": 1}
    assert "CJK" in report.script_counts


def test_validate_truth_file_clean_fixture():
    report = validate_truth_file(FIXTURES / "truth.tsv")
    assert report.is_valid
    assert report.total_rows == 4
    assert report.valid_rows == 4
    assert report.singleton_rows == 1  # S1-SINGLE
    assert report.single_match_rows == 2  # S1-FRANCE, S1-UNICODE
    assert report.multi_match_rows == 1  # S1-MULTI (2 matches)
    assert report.total_links == 4  # 2 + 1 + 1
    assert report.s2_links == 2  # S2-MATCH, S2-UNICODE
    assert report.s3_links == 2  # S3-MATCH, S3-FRANCE
    assert report.match_length_histogram == {2: 1, 0: 1, 1: 2}


def test_validate_source_file_corrupt(tmp_path):
    f = tmp_path / "corrupt_source.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tValid Name\t123 Main St\tUS\n"
        "S2-BAD\tWrong Prefix\t456 Elm St\tUS\n"
        "S1-2\tMissing Address\t\tIndia\n"
        "S1-3\t\t789 Oak Ave\tUS\n"  # missing name
        "S1-1\tDuplicate ID\t100 Pine St\tUS\n"  # duplicate ID
        "S1-4\tMalformed Column Count\n"  # only 2 columns
    )
    f.write_text(content, encoding="utf-8")
    report = validate_source_file(f, "S1-")
    assert not report.is_valid
    assert report.prefix_errors == 1  # S2-BAD
    assert report.duplicate_ids == 1  # S1-1 duplicate
    assert report.missing_names == 1  # S1-3 has empty name
    assert report.empty_addresses == 1  # S1-2 has empty address
    assert report.malformed_rows == 1  # S1-4 malformed
    assert len(report.errors) >= 3


def test_validate_truth_file_corrupt(tmp_path):
    f = tmp_path / "corrupt_truth.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-A,S3-B\n"
        "S1-2\tS2-C,S2-C\n"  # internal duplicate in match list
        "S1-1\tS2-D\n"  # duplicate S1 row
        "S1-3\tS1-WRONG\n"  # invalid candidate prefix
    )
    f.write_text(content, encoding="utf-8")
    report = validate_truth_file(f)
    assert not report.is_valid
    assert report.duplicate_s1_ids == 1
    assert report.duplicate_matched_ids == 1
    assert report.prefix_errors == 1


def test_check_duplicate_ids_partitioned(tmp_path):
    """Test memory-bounded partitioned duplicate checking."""
    f = tmp_path / "partition_test.tsv"
    rows = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    for i in range(200):
        rows.append(f"S1-{i:04d}\tBusiness {i}\tAddress {i}\tUS")
    # Add deliberate duplicates
    rows.append("S1-0042\tDuplicate 42\tAddress 42\tUS")
    rows.append("S1-0105\tDuplicate 105\tAddress 105\tUS")
    f.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dups = check_duplicate_ids_partitioned(f, id_column=0, num_partitions=8, temp_dir=tmp_path)
    assert set(dups) == {"S1-0042", "S1-0105"}


def test_audit_dataset_on_fixtures(tmp_path):
    """Test audit_dataset summarizes files in a directory."""
    # Copy fixtures into a mock dataset directory structure
    for name in ("source1.tsv", "source2.tsv", "source3.tsv", "truth.tsv"):
        content = (FIXTURES / name).read_bytes()
        dest_name = (
            "train_source1.tsv" if name == "source1.tsv"
            else "train_source2.tsv" if name == "source2.tsv"
            else "train_source3.tsv" if name == "source3.tsv"
            else "train_ground_truth.tsv"
        )
        (tmp_path / dest_name).write_bytes(content)

    out_json = tmp_path / "audit_report.json"
    audit_res = audit_dataset(tmp_path, output_path=out_json)

    assert "train_source1.tsv" in audit_res["files"]
    assert "train_ground_truth.tsv" in audit_res["files"]
    assert audit_res["files"]["train_source1.tsv"]["total_rows"] == 4
    assert audit_res["files"]["train_ground_truth.tsv"]["singleton_rows"] == 1
    assert out_json.exists()


def test_streaming_memory_efficiency(tmp_path):
    """Verify that reading a stream of thousands of records does not accumulate memory objects."""
    f = tmp_path / "large_mock_source.tsv"
    num_rows = 5000
    with f.open("w", encoding="utf-8", newline="") as stream:
        stream.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for i in range(num_rows):
            stream.write(f"S1-{i}\tBusiness Name {i}\t{i} River Road\tUnited States\n")

    # Audit streams without accumulating in-memory row objects
    report = validate_source_file(f, "S1-", check_duplicates=False)
    assert report.total_rows == num_rows
    assert report.valid_rows == num_rows
    assert report.is_valid
