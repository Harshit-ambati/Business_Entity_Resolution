"""Tests for Unicode-safe text and record normalization (PR T2)."""

import pytest

from ber import (
    NORMALIZATION_VERSION,
    NormalizedRecord,
    Record,
    normalize_country,
    normalize_record,
    normalize_text,
)


def test_normalization_version_exported():
    assert NORMALIZATION_VERSION == "1"


def test_normalize_text_determinism():
    sample = "Blue Lantern Bakery & Cafe, 12 River Road"
    first_norm, first_tokens = normalize_text(sample)
    second_norm, second_tokens = normalize_text(sample)
    assert first_norm == second_norm
    assert first_tokens == second_tokens


def test_normalize_text_casefolding():
    assert normalize_text("UPPERCASE NAME")[0] == "uppercase name"
    assert normalize_text("MixedCase Name")[0] == "mixedcase name"
    # German Eszett expands under Unicode casefold
    norm, tokens = normalize_text("Straße")
    assert norm == "strasse"
    assert tokens == ("strasse",)


def test_normalize_text_ampersand_versus_and():
    norm_amp, tokens_amp = normalize_text("Blue Lantern & Bakery")
    norm_word, tokens_word = normalize_text("Blue Lantern and Bakery")
    assert norm_amp == "blue lantern and bakery"
    assert norm_word == "blue lantern and bakery"
    assert tokens_amp == ("blue", "lantern", "and", "bakery")
    assert tokens_word == ("blue", "lantern", "and", "bakery")

    # Embedded ampersand
    assert normalize_text("AT&T")[0] == "at and t"
    assert normalize_text("AT&T")[1] == ("at", "and", "t")


def test_normalize_text_punctuation_to_spaces():
    raw = 'Blue-Lantern, "Bakery" / Co. (Main); Suite #1!'
    norm, tokens = normalize_text(raw)
    assert norm == "blue lantern bakery co main suite 1"
    assert tokens == ("blue", "lantern", "bakery", "co", "main", "suite", "1")


def test_normalize_text_whitespace_collapsing():
    raw = "   Blue \t\t Lantern \n\n   Bakery   "
    norm, tokens = normalize_text(raw)
    assert norm == "blue lantern bakery"
    assert tokens == ("blue", "lantern", "bakery")


def test_v1_does_not_expand_abbreviations():
    """V1 normalization contract: do not rewrite abbreviations (st != street, etc.)."""
    _, tokens_ltd = normalize_text("Bakery Ltd")
    _, tokens_limited = normalize_text("Bakery Limited")
    assert tokens_ltd == ("bakery", "ltd")
    assert tokens_limited == ("bakery", "limited")
    assert tokens_ltd != tokens_limited

    _, tokens_rd = normalize_text("12 River Rd")
    _, tokens_road = normalize_text("12 River Road")
    assert tokens_rd == ("12", "river", "rd")
    assert tokens_road == ("12", "river", "road")

    _, tokens_st = normalize_text("Main St")
    assert tokens_st == ("main", "st")


def test_v1_preserves_legal_suffixes():
    """Legal suffixes must survive in name_norm without destructive removal."""
    norm_llc, _ = normalize_text("Blue Lantern Bakery LLC")
    assert "llc" in norm_llc.split()

    norm_sarl, _ = normalize_text("Maison Étoile SARL")
    assert "sarl" in norm_sarl.split()

    norm_pvt, _ = normalize_text("Tata Sons Pvt Ltd")
    assert {"pvt", "ltd"}.issubset(set(norm_pvt.split()))


def test_v1_preserves_numbers_and_digits():
    """Informative numbers (house numbers, suite codes) must survive."""
    norm, tokens = normalize_text("12 River Road, Suite 4B")
    assert "12" in tokens
    assert "4b" in tokens

    norm7, tokens7 = normalize_text("7 Rue des Lilas, Apt 101")
    assert "7" in tokens7
    assert "101" in tokens7


def test_v1_preserves_accents_and_diacritics():
    """French and European accents must be preserved in normalized text."""
    norm, tokens = normalize_text("Maison Étoile")
    assert norm == "maison étoile"
    assert tokens == ("maison", "étoile")

    norm_cafe, tokens_cafe = normalize_text("Café de la Gare")
    assert norm_cafe == "café de la gare"
    assert "café" in tokens_cafe

    norm_zurich, tokens_zurich = normalize_text("Zürich Hotel")
    assert norm_zurich == "zürich hotel"


def test_v1_preserves_multilingual_scripts():
    """Devanagari, Telugu, CJK, and other non-Latin scripts must survive intact."""
    # Devanagari (Hindi)
    norm_hi, tokens_hi = normalize_text("किताब घर (Book House)")
    assert norm_hi == "किताब घर book house"
    assert tokens_hi == ("किताब", "घर", "book", "house")

    # Telugu
    norm_te, tokens_te = normalize_text("భారత్ ఎలక్ట్రానిక్స్")
    assert norm_te == "భారత్ ఎలక్ట్రానిక్స్"
    assert tokens_te == ("భారత్", "ఎలక్ట్రానిక్స్")

    # Japanese
    norm_ja, tokens_ja = normalize_text("さくら商店, 東京都千代田区")
    assert norm_ja == "さくら商店 東京都千代田区"
    assert tokens_ja == ("さくら商店", "東京都千代田区")

    # Cyrillic
    norm_ru, tokens_ru = normalize_text("ООО Вектор")
    assert norm_ru == "ооо вектор"
    assert tokens_ru == ("ооо", "вектор")


def test_normalize_country_open_set():
    """Country normalization is open-set: NFKC + casefold + whitespace collapse only."""
    assert normalize_country("United States") == "united states"
    assert normalize_country("  India  ") == "india"
    assert normalize_country("France") == "france"
    # Arbitrary unseen country
    assert normalize_country(" Côte  d'Ivoire ") == "côte d'ivoire"
    assert normalize_country("Deutschland") == "deutschland"
    assert normalize_country("") == ""
    assert normalize_country("   ") == ""


def test_normalize_empty_and_whitespace_values():
    assert normalize_text("") == ("", ())
    assert normalize_text("   ") == ("", ())
    assert normalize_country("") == ""
    assert normalize_country("   ") == ""


def test_normalize_record_contract():
    raw = Record(
        entity_id="S1-MULTI",
        business_name="Blue Lantern & Bakery LLC",
        business_address="12 River Road, Suite #4B",
        country="United States",
    )
    norm = normalize_record(raw)

    assert isinstance(norm, NormalizedRecord)
    # Raw record is preserved untouched
    assert norm.raw is raw
    assert norm.raw.business_name == "Blue Lantern & Bakery LLC"

    # Normalized fields
    assert norm.name_norm == "blue lantern and bakery llc"
    assert norm.name_tokens == ("blue", "lantern", "and", "bakery", "llc")
    assert norm.address_norm == "12 river road suite 4b"
    assert norm.address_tokens == ("12", "river", "road", "suite", "4b")
    assert norm.country_key == "united states"


def test_normalize_record_empty_address():
    raw = Record("S1-SINGLE", "Quiet Birch Studio", "", "India")
    norm = normalize_record(raw)
    assert norm.address_norm == ""
    assert norm.address_tokens == ()
    assert norm.country_key == "india"


def test_normalize_none_and_unexpected_types():
    assert normalize_text(None) == ("", ())
    assert normalize_country(None) == ""
    # Unexpected types like integers or floats converted safely
    norm, tokens = normalize_text(12345)
    assert norm == "12345"
    assert tokens == ("12345",)
    assert normalize_country(42) == "42"
