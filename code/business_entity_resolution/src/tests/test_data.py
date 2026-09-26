"""Tests for TSV streaming readers and data contracts (PR T1)."""

from io import StringIO
from pathlib import Path

import pytest

from ber import (
    SOURCE_HEADERS,
    TRUTH_HEADERS,
    Record,
    TruthRow,
    read_source,
    read_truth,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_source_fixtures():
    records = list(read_source(FIXTURES / "source1.tsv", "S1-"))
    assert len(records) == 4
    assert all(isinstance(r, Record) for r in records)
    assert records[0].entity_id == "S1-MULTI"
    assert records[0].business_name == "Blue Lantern Bakery"
    assert records[0].business_address == "12 River Road"
    assert records[0].country == "United States"

    # Singletons and empty address handling
    assert records[1].entity_id == "S1-SINGLE"
    assert records[1].business_address == ""
    assert records[1].country == "India"

    # France open-set country
    assert records[2].entity_id == "S1-FRANCE"
    assert records[2].country == "France"

    # Unicode preservation
    assert records[3].entity_id == "S1-UNICODE"
    assert records[3].business_name == "さくら商店"


def test_read_source_sources_2_and_3():
    s2 = list(read_source(FIXTURES / "source2.tsv", "S2-"))
    assert len(s2) == 3
    assert all(r.entity_id.startswith("S2-") for r in s2)

    s3 = list(read_source(FIXTURES / "source3.tsv", "S3-"))
    assert len(s3) == 2
    assert all(r.entity_id.startswith("S3-") for r in s3)


def test_read_truth_fixtures():
    truth = list(read_truth(FIXTURES / "truth.tsv"))
    assert len(truth) == 4
    assert all(isinstance(t, TruthRow) for t in truth)

    by_id = {t.source1_entity_id: t.matched_entity_ids for t in truth}
    assert by_id["S1-MULTI"] == ("S2-MATCH", "S3-MATCH")
    assert by_id["S1-SINGLE"] == ()
    assert by_id["S1-FRANCE"] == ("S3-FRANCE",)
    assert by_id["S1-UNICODE"] == ("S2-UNICODE",)


def test_read_source_invalid_expected_prefix(tmp_path):
    f = tmp_path / "dummy.tsv"
    f.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid expected_prefix"):
        list(read_source(f, "INVALID-"))


def test_read_source_bad_header(tmp_path):
    f = tmp_path / "bad_header.tsv"
    f.write_text("id\tname\taddress\tcountry\nS1-1\tFoo\tBar\tUS\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid source header at .*bad_header.tsv:1"):
        list(read_source(f, "S1-"))


def test_read_source_empty_file(tmp_path):
    f = tmp_path / "empty.tsv"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty source file"):
        list(read_source(f, "S1-"))


def test_read_source_malformed_columns(tmp_path):
    f = tmp_path / "malformed.tsv"
    content = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tName Only\n"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed row at .*malformed.tsv:2: expected 4 columns, got 2"):
        list(read_source(f, "S1-"))


def test_read_source_wrong_prefix_in_row(tmp_path):
    f = tmp_path / "wrong_prefix.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-WRONG\tName\tAddress\tCountry\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid entity ID 'S2-WRONG' at .*wrong_prefix.tsv:2; expected prefix 'S1-'"):
        list(read_source(f, "S1-"))


def test_read_source_empty_suffix(tmp_path):
    f = tmp_path / "empty_suffix.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-\tName\tAddress\tCountry\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty suffix"):
        list(read_source(f, "S1-"))


def test_read_source_duplicate_detection(tmp_path):
    f = tmp_path / "dups.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tFirst\tAddr\tUS\n"
        "S1-1\tSecond\tAddr\tUS\n"
    )
    f.write_text(content, encoding="utf-8")
    # With check_duplicates=True
    with pytest.raises(ValueError, match="Duplicate entity ID 'S1-1' at .*dups.tsv:3"):
        list(read_source(f, "S1-", check_duplicates=True))

    # With check_duplicates=False, both yield without tracking
    records = list(read_source(f, "S1-", check_duplicates=False))
    assert len(records) == 2


def test_read_source_quoted_punctuation_and_commas(tmp_path):
    """Addresses containing commas and quotes must parse without column shift."""
    f = tmp_path / "quoted.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        'S1-Q1\tBlue Lantern Bakery\t12 River Road, Suite "4B"\tUnited States\n'
        'S1-Q2\tAcme & Co.\t"742 Evergreen Terr, Apt 2"\tUnited States\n'
    )
    f.write_text(content, encoding="utf-8")
    records = list(read_source(f, "S1-"))
    assert len(records) == 2
    assert records[0].business_name == "Blue Lantern Bakery"
    assert records[0].business_address == '12 River Road, Suite "4B"'
    assert records[1].business_name == "Acme & Co."
    assert records[1].business_address == "742 Evergreen Terr, Apt 2"



def test_read_truth_bad_header(tmp_path):
    f = tmp_path / "bad_truth_header.tsv"
    f.write_text("s1_id\tmatches\nS1-1\tS2-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid truth header at .*bad_truth_header.tsv:1"):
        list(read_truth(f))


def test_read_truth_empty_file(tmp_path):
    f = tmp_path / "empty_truth.tsv"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty truth file"):
        list(read_truth(f))


def test_read_truth_malformed_columns(tmp_path):
    f = tmp_path / "malformed_truth.tsv"
    f.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-A\tEXTRA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2 columns, got 3"):
        list(read_truth(f))


def test_read_truth_duplicate_s1_row(tmp_path):
    f = tmp_path / "dup_s1.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-A\n"
        "S1-1\tS2-B\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate source1_entity_id 'S1-1' at .*dup_s1.tsv:3"):
        list(read_truth(f))


def test_read_truth_duplicate_matched_id_in_row(tmp_path):
    f = tmp_path / "dup_matched.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-A,S2-A\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate matched IDs in truth row"):
        list(read_truth(f))


def test_read_truth_invalid_candidate_prefix(tmp_path):
    f = tmp_path / "bad_candidate_prefix.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS1-WRONG\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid candidate entity ID 'S1-WRONG'"):
        list(read_truth(f))


def test_read_truth_empty_candidate_in_list(tmp_path):
    f = tmp_path / "empty_candidate.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-A,,S3-B\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Empty matched entity ID in truth list"):
        list(read_truth(f))


def test_read_source_file_handle_closes(tmp_path):
    f = tmp_path / "resource_test.tsv"
    f.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tName 1\tAddr 1\tUS\n"
        "S1-2\tName 2\tAddr 2\tUS\n",
        encoding="utf-8",
    )
    # Open iterator and break early
    gen = read_source(f, "S1-")
    first = next(gen)
    assert first.entity_id == "S1-1"
    gen.close()  # Generator closing releases file handle immediately
    # On Windows, open handles prevent file deletion; so unlink confirms handle was released
    f.unlink()
    assert not f.exists()
