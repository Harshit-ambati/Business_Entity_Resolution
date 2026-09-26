"""Temporary v1 normalization shim — replace with Thulasi's ber.normalize when available.

Implements the exact v1 normalization contract from docs/CONTRACTS.md:
- NFKC then casefold
- & -> "and"
- Punctuation/separators -> spaces (preserve Unicode letters/digits)
- Collapse whitespace, tokenize in order
- Country: NFKC/casefold/whitespace collapse only
- Retain raw fields, digits, accents, non-Latin scripts
- Do NOT expand abbreviations, remove legal suffixes, or transliterate
"""

from __future__ import annotations

import re
import unicodedata

from .contracts import NormalizedRecord, Record

NORMALIZATION_VERSION = "1"


def _normalize_text(text: str) -> str:
    """Apply v1 text normalization: NFKC, casefold, & -> and, punct -> space, collapse."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = text.replace("&", " and ")
    # Replace non-letter/non-digit characters with spaces
    chars: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in ("L", "N"):  # Letter or Number
            chars.append(ch)
        else:
            chars.append(" ")
    text = "".join(chars)
    # Collapse whitespace
    text = re.sub(r" +", " ", text).strip()
    return text


def _normalize_country(country: str) -> str:
    """Apply v1 country normalization: NFKC, casefold, whitespace collapse only."""
    if not country:
        return ""
    text = unicodedata.normalize("NFKC", country)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_record(record: Record) -> NormalizedRecord:
    """Normalize a record following the v1 contract specification.

    Pure and deterministic. Preserves the raw record. Never translates or
    deletes entire non-Latin scripts. Empty address produces empty address
    features.
    """
    name_norm = _normalize_text(record.business_name)
    address_norm = _normalize_text(record.business_address)
    return NormalizedRecord(
        raw=record,
        name_norm=name_norm,
        address_norm=address_norm,
        name_tokens=tuple(name_norm.split()) if name_norm else (),
        address_tokens=tuple(address_norm.split()) if address_norm else (),
        country_key=_normalize_country(record.country),
    )
