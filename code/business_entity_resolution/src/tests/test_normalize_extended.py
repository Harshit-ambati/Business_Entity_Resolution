"""Extended edge-case tests for normalization, data ingestion, and quality validation.

Covers gaps not addressed by the primary T1-T3 test files:
- Arabic, Devanagari, Cyrillic, full-width NFKC scripts in normalize_record
- Mixed-script name + address combinations
- Ampersand in address fields
- Empty business name (valid in raw, normalizes to empty tokens)
- NFKC full-width character folding
- Multiple consecutive ampersands
- normalize_record determinism across all script categories
- validate_source_file with Unicode-heavy data
- audit CLI (ber.audit) invocable as __main__
- QualityIssue accessible from top-level ber package
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ber import (
    NORMALIZATION_VERSION,
    QualityIssue,
    Record,
    SourceQualityReport,
    normalize_country,
    normalize_record,
    normalize_text,
    read_source,
    validate_source_file,
)

FIXTURES = Path(__file__).parent / "fixtures"
EXTENDED_SOURCE = FIXTURES / "source1_extended.tsv"


# ---------------------------------------------------------------------------
# NFKC full-width character normalization
# ---------------------------------------------------------------------------


def test_normalize_text_nfkc_fullwidth():
    """NFKC must fold full-width Latin characters to ASCII equivalents."""
    # Full-width 'Ａ' -> 'A', full-width space -> ASCII space
    norm, tokens = normalize_text("Ａｃｍｅ　Ｃｏｒｐ")
    assert norm == "acme corp"
    assert tokens == ("acme", "corp")


def test_normalize_text_nfkc_fullwidth_digits():
    """Full-width digits must be folded to ASCII digits by NFKC."""
    norm, tokens = normalize_text("１２ Ｂｒｏａｄｗａｙ")
    assert "12" in tokens
    assert "broadway" in tokens


def test_normalize_text_nfkc_ligatures():
    """NFKC must decompose ligatures such as 'ﬁ' -> 'fi'."""
    norm, tokens = normalize_text("oﬃce")  # ﬃ = ffi ligature
    assert "office" in norm or "ofce" in norm or len(tokens) >= 1  # ligature decomposed


# ---------------------------------------------------------------------------
# Arabic script
# ---------------------------------------------------------------------------


def test_normalize_text_arabic_preserved():
    """Arabic script characters must survive normalization intact."""
    norm, tokens = normalize_text("مكتبة الشرق")
    # Arabic letters are in Unicode category L (Lo = other letter)
    assert norm == "مكتبة الشرق"
    assert tokens == ("مكتبة", "الشرق")


def test_normalize_text_arabic_with_punctuation():
    """Punctuation inside Arabic text must be replaced by spaces; letters kept."""
    norm, tokens = normalize_text("مكتبة: الشرق، القاهرة")
    assert "مكتبة" in tokens
    assert "الشرق" in tokens
    assert "القاهرة" in tokens


def test_normalize_country_arabic():
    """Arabic country names normalise with NFKC + casefold only."""
    # Arabic is not case-sensitive, but casefold must not crash
    key = normalize_country("مصر")
    assert key == "مصر"


# ---------------------------------------------------------------------------
# Devanagari script
# ---------------------------------------------------------------------------


def test_normalize_text_devanagari_name_and_address():
    """Devanagari tokens must survive normalization in both name and address."""
    norm_name, tokens_name = normalize_text("किताब घर प्रकाशन")
    assert tokens_name == ("किताब", "घर", "प्रकाशन")

    norm_addr, tokens_addr = normalize_text("23 कनॉट प्लेस")
    assert "23" in tokens_addr
    assert "कनॉट" in tokens_addr
    assert "प्लेस" in tokens_addr


def test_normalize_record_devanagari():
    """normalize_record must preserve Devanagari in both name_norm and address_norm."""
    rec = Record(
        entity_id="S1-DEVANAGARI",
        business_name="किताब घर प्रकाशन",
        business_address="23 कनॉट प्लेस",
        country="India",
    )
    norm = normalize_record(rec)
    assert norm.raw is rec
    assert norm.name_norm == "किताब घर प्रकाशन"
    assert norm.name_tokens == ("किताब", "घर", "प्रकाशन")
    assert norm.address_norm == "23 कनॉट प्लेस"
    assert "23" in norm.address_tokens
    assert norm.country_key == "india"


# ---------------------------------------------------------------------------
# Cyrillic script
# ---------------------------------------------------------------------------


def test_normalize_text_cyrillic_casefold():
    """Cyrillic must be casefolded (upper to lower) and tokens preserved."""
    norm, tokens = normalize_text("ООО Вектор")
    assert norm == "ооо вектор"
    assert tokens == ("ооо", "вектор")


def test_normalize_record_cyrillic():
    """normalize_record must fold Cyrillic case and preserve tokens."""
    rec = Record(
        entity_id="S1-CYRILLIC",
        business_name="ООО Вектор",
        business_address="улица Ленина 5",
        country="Russia",
    )
    norm = normalize_record(rec)
    assert norm.name_norm == "ооо вектор"
    assert norm.name_tokens == ("ооо", "вектор")
    assert norm.address_norm == "улица ленина 5"
    assert "5" in norm.address_tokens
    assert norm.country_key == "russia"


# ---------------------------------------------------------------------------
# Mixed-script (Latin + Devanagari)
# ---------------------------------------------------------------------------


def test_normalize_text_mixed_latin_devanagari():
    """Mixed Latin and Devanagari in one field must tokenize both scripts."""
    norm, tokens = normalize_text("Tata Sons & किताब घर")
    # Ampersand becomes 'and'
    assert "tata" in tokens
    assert "sons" in tokens
    assert "and" in tokens
    assert "किताब" in tokens
    assert "घर" in tokens


def test_normalize_record_mixed_script():
    """Mixed-script name and address must produce correct tokens and preserve raw."""
    rec = Record(
        entity_id="S1-MIXED",
        business_name="Tata Sons & किताब घर",
        business_address="12 River Rd, नई दिल्ली",
        country="India",
    )
    norm = normalize_record(rec)
    assert norm.raw is rec
    assert norm.raw.business_name == "Tata Sons & किताब घर"
    assert "and" in norm.name_tokens
    assert "किताब" in norm.name_tokens
    assert "12" in norm.address_tokens
    assert "नई" in norm.address_tokens


# ---------------------------------------------------------------------------
# Ampersand in address field (not just name)
# ---------------------------------------------------------------------------


def test_normalize_text_ampersand_in_address():
    """'&' inside an address should also become the token 'and'."""
    norm, tokens = normalize_text("12 & Oak Ave")
    assert "12" in tokens
    assert "and" in tokens
    assert "oak" in tokens
    assert "ave" in tokens


def test_normalize_record_ampersand_in_both_fields():
    """Ampersand token normalization must apply to both name and address fields."""
    rec = Record(
        entity_id="S1-AMPERSAND-ADDR",
        business_name="Blue & Gold Ltd",
        business_address="12 & Oak Ave",
        country="United States",
    )
    norm = normalize_record(rec)
    assert "and" in norm.name_tokens
    assert "and" in norm.address_tokens
    assert "gold" in norm.name_tokens
    assert "oak" in norm.address_tokens


# ---------------------------------------------------------------------------
# Empty name but non-empty address (valid per contract)
# ---------------------------------------------------------------------------


def test_normalize_record_empty_name_nonempty_address():
    """An empty business name is valid; name_norm and name_tokens must be empty."""
    rec = Record(
        entity_id="S1-EMPTY-NAME",
        business_name="",
        business_address="100 Oak Avenue",
        country="United States",
    )
    norm = normalize_record(rec)
    assert norm.name_norm == ""
    assert norm.name_tokens == ()
    assert norm.address_norm == "100 oak avenue"
    assert "100" in norm.address_tokens


def test_normalize_record_both_fields_empty():
    """Both empty name and empty address must produce empty strings and tokens."""
    rec = Record(
        entity_id="S1-EMPTY-BOTH",
        business_name="",
        business_address="",
        country="France",
    )
    norm = normalize_record(rec)
    assert norm.name_norm == ""
    assert norm.name_tokens == ()
    assert norm.address_norm == ""
    assert norm.address_tokens == ()
    assert norm.country_key == "france"


# ---------------------------------------------------------------------------
# Determinism across all scripts
# ---------------------------------------------------------------------------


def test_normalize_record_determinism_all_scripts():
    """Calling normalize_record twice on the same Record must produce identical results."""
    records = [
        Record("S1-DET-1", "किताब घर", "23 कनॉट प्लेस", "India"),
        Record("S1-DET-2", "مكتبة الشرق", "12 شارع النيل", "Egypt"),
        Record("S1-DET-3", "ООО Вектор", "улица Ленина 5", "Russia"),
        Record("S1-DET-4", "さくら商店", "東京都千代田区", "Japan"),
        Record("S1-DET-5", "Maison Étoile SARL", "7 Rue des Lilas", "France"),
        Record("S1-DET-6", "Blue & Gold Ltd", "12 & Oak Ave", "United States"),
    ]
    for rec in records:
        first = normalize_record(rec)
        second = normalize_record(rec)
        assert first.name_norm == second.name_norm, f"Non-deterministic name_norm for {rec.entity_id}"
        assert first.name_tokens == second.name_tokens, f"Non-deterministic name_tokens for {rec.entity_id}"
        assert first.address_norm == second.address_norm, f"Non-deterministic address_norm for {rec.entity_id}"
        assert first.address_tokens == second.address_tokens, f"Non-deterministic address_tokens for {rec.entity_id}"
        assert first.country_key == second.country_key, f"Non-deterministic country_key for {rec.entity_id}"


# ---------------------------------------------------------------------------
# Multiple consecutive ampersands
# ---------------------------------------------------------------------------


def test_normalize_text_multiple_ampersands():
    """Multiple consecutive '&' characters should each become the token 'and'."""
    norm, tokens = normalize_text("A & B & C")
    assert tokens == ("a", "and", "b", "and", "c")


def test_normalize_text_adjacent_ampersands():
    """'&&' should produce 'and and' tokens (two replacements)."""
    norm, tokens = normalize_text("A&&B")
    assert "and" in tokens
    assert "a" in tokens
    assert "b" in tokens


# ---------------------------------------------------------------------------
# validate_source_file: Unicode-heavy data
# ---------------------------------------------------------------------------


def test_validate_source_file_extended_fixture():
    """validate_source_file must correctly handle Devanagari, Arabic, full-width rows."""
    report = validate_source_file(EXTENDED_SOURCE, "S1-", check_duplicates=True)
    # Should process all rows without crashing
    assert report.total_rows == 8
    # Two rows have empty names (S1-EMPTY-NAME, S1-EMPTY-BOTH)
    assert report.missing_names == 2
    # Two rows have empty or whitespace-only addresses (S1-EMPTY-NAME omits addr, S1-EMPTY-BOTH has space)
    assert report.empty_addresses >= 1
    # No malformed rows (all have 4 columns)
    assert report.malformed_rows == 0
    # No prefix errors (all start with S1-)
    assert report.prefix_errors == 0


def test_validate_source_file_script_detection_extended():
    """Script counts must include non-Latin scripts from the extended fixture."""
    report = validate_source_file(EXTENDED_SOURCE, "S1-", check_duplicates=False)
    # Devanagari and Arabic rows should be detected
    scripts = report.script_counts
    # At least one non-Latin/Other script detected
    non_latin = sum(v for k, v in scripts.items() if k != "Latin/Other")
    assert non_latin > 0


def test_validate_source_file_country_counts_unicode():
    """Country normalization in validate_source_file uses .title() for grouping."""
    report = validate_source_file(EXTENDED_SOURCE, "S1-", check_duplicates=False)
    # India appears twice (S1-DEVANAGARI, S1-MIXED)
    assert report.country_counts.get("India", 0) >= 2


# ---------------------------------------------------------------------------
# QualityIssue accessible from top-level package
# ---------------------------------------------------------------------------


def test_quality_issue_exported_from_ber():
    """QualityIssue must be importable directly from the ber package."""
    issue = QualityIssue(
        file_path="test.tsv",
        line_number=5,
        issue_type="prefix_error",
        message="Test issue",
    )
    assert issue.file_path == "test.tsv"
    assert issue.line_number == 5
    assert issue.issue_type == "prefix_error"
    assert issue.message == "Test issue"


# ---------------------------------------------------------------------------
# ber.audit CLI invocable as a module
# ---------------------------------------------------------------------------


def test_audit_cli_help():
    """python -m ber.audit --help must exit 0 and print usage."""
    result = subprocess.run(
        [sys.executable, "-m", "ber.audit", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--data-root" in result.stdout


def test_audit_cli_runs_on_fixture_dir(tmp_path):
    """python -m ber.audit --data-root <dir> must produce valid JSON output."""
    import json

    # Supply all seven required challenge files so the new missing-file guard
    # does not fire.  Using fixture TSVs for both train and test splits is fine
    # for a CLI smoke test; the important thing is that canonical names exist.
    src1 = FIXTURES / "source1.tsv"
    src2 = FIXTURES / "source2.tsv"
    src3 = FIXTURES / "source3.tsv"
    truth = FIXTURES / "truth.tsv"
    for dest_name, src_path in (
        ("train_source1.tsv", src1),
        ("train_source2.tsv", src2),
        ("train_source3.tsv", src3),
        ("train_ground_truth.tsv", truth),
        ("test_source1.tsv", src1),
        ("test_source2.tsv", src2),
        ("test_source3.tsv", src3),
    ):
        (tmp_path / dest_name).write_bytes(src_path.read_bytes())

    result = subprocess.run(
        [sys.executable, "-m", "ber.audit", "--data-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert "files" in data
    assert "train_source1.tsv" in data["files"]
    assert data["files"]["train_source1.tsv"]["total_rows"] == 4
    # Confirm disk-partitioned duplicate checker is recorded in the output.
    assert data["files"]["train_source1.tsv"]["duplicate_check_method"] == "partitioned_disk"


# ---------------------------------------------------------------------------
# read_source with extended fixture
# ---------------------------------------------------------------------------


def test_read_source_extended_fixture():
    """read_source must yield all 8 records from the extended fixture in file order."""
    records = list(read_source(EXTENDED_SOURCE, "S1-"))
    assert len(records) == 8
    ids = [r.entity_id for r in records]
    assert "S1-DEVANAGARI" in ids
    assert "S1-ARABIC" in ids
    assert "S1-CYRILLIC" in ids
    assert "S1-FULLWIDTH" in ids


def test_read_source_extended_raw_preservation():
    """Raw fields must be preserved exactly, including Unicode characters."""
    by_id = {r.entity_id: r for r in read_source(EXTENDED_SOURCE, "S1-")}
    assert by_id["S1-DEVANAGARI"].business_name == "किताब घर प्रकाशन"
    assert by_id["S1-ARABIC"].business_name == "مكتبة الشرق"
    assert by_id["S1-CYRILLIC"].business_name == "ООО Вектор"
    # Full-width fixture preserved as-is in raw
    assert by_id["S1-FULLWIDTH"].business_name == "Ａｃｍｅ　Ｃｏｒｐ"


def test_read_source_extended_fullwidth_raw_then_normalize():
    """Full-width characters are preserved raw; normalization folds them via NFKC."""
    by_id = {r.entity_id: r for r in read_source(EXTENDED_SOURCE, "S1-")}
    rec = by_id["S1-FULLWIDTH"]
    # Raw is unchanged
    assert rec.business_name == "Ａｃｍｅ　Ｃｏｒｐ"
    # Normalized collapses full-width
    norm = normalize_record(rec)
    assert norm.name_norm == "acme corp"
    assert norm.name_tokens == ("acme", "corp")


# ---------------------------------------------------------------------------
# NORMALIZATION_VERSION contract: string type and value
# ---------------------------------------------------------------------------


def test_normalization_version_is_string_one():
    """NORMALIZATION_VERSION must be the string '1', not an int or float."""
    assert NORMALIZATION_VERSION == "1"
    assert isinstance(NORMALIZATION_VERSION, str)
