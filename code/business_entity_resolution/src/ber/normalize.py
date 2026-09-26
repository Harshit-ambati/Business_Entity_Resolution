"""Pure, deterministic, Unicode-safe text and record normalization (PR T2).

Adheres to CONTRACTS.md:
- NORMALIZATION_VERSION = "1"
- normalize_record(record: Record) -> NormalizedRecord
"""

from __future__ import annotations

import unicodedata

from .contracts import NormalizedRecord, Record

NORMALIZATION_VERSION: str = "1"


def normalize_text(text: str) -> tuple[str, tuple[str, ...]]:
    """Normalize business name or address text into canonical string and tokens.

    Applies:
    1. Unicode NFKC normalization.
    2. Unicode casefold().
    3. Replacement of '&' with the token 'and'.
    4. Replacement of punctuation and separators with spaces while preserving
       Unicode letters (L), numbers (N), and combining marks/diacritics (M).
    5. Whitespace collapse and deterministic tokenization in order.

    Parameters
    ----------
    text : str
        Raw input string (name or address).

    Returns
    -------
    tuple[str, tuple[str, ...]]
        (normalized_string, ordered_tokens_tuple)
        Returns ("", ()) for empty or whitespace-only inputs.
    """
    if text is None:
        return "", ()
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return "", ()

    # 1. NFKC normalization
    nfkc_text = unicodedata.normalize("NFKC", text)

    # 2. Unicode case folding
    folded = nfkc_text.casefold()

    # 3. Replace '&' with the token 'and' (surrounded by spaces for clean tokenization)
    with_and = folded.replace("&", " and ")

    # 4. Replace punctuation/separators with spaces while preserving
    # letters (L), numbers (N), and combining marks (M) for Indic/accented scripts.
    chars = [
        c if unicodedata.category(c)[0] in ("L", "N", "M") else " "
        for c in with_and
    ]

    # 5. Collapse whitespace and produce ordered tokens
    tokens = tuple("".join(chars).split())
    norm_text = " ".join(tokens)
    return norm_text, tokens


def normalize_country(country: str) -> str:
    """Normalize open-set country string with NFKC, casefold, and whitespace collapse.

    Does NOT use an allowlist; accepts 'United States', 'India', 'France', and any
    future country label deterministically. Does not strip punctuation or replace words.

    Parameters
    ----------
    country : str
        Raw country string.

    Returns
    -------
    str
        Normalized country key, or "" if empty/whitespace-only.
    """
    if country is None:
        return ""
    if not isinstance(country, str):
        country = str(country)
    if not country:
        return ""

    nfkc_text = unicodedata.normalize("NFKC", country)
    folded = nfkc_text.casefold()
    tokens = folded.split()
    return " ".join(tokens)


def normalize_record(record: Record) -> NormalizedRecord:
    """Deterministically normalize a Record into a NormalizedRecord.

    Preserves the raw Record untouched and creates normalized views and token tuples
    for name and address, plus the normalized open-set country key.

    Parameters
    ----------
    record : Record
        Input raw record.

    Returns
    -------
    NormalizedRecord
        Normalized representation matching the shared team contract.
    """
    name_norm, name_tokens = normalize_text(record.business_name)
    address_norm, address_tokens = normalize_text(record.business_address)
    country_key = normalize_country(record.country)

    return NormalizedRecord(
        raw=record,
        name_norm=name_norm,
        address_norm=address_norm,
        name_tokens=name_tokens,
        address_tokens=address_tokens,
        country_key=country_key,
    )
