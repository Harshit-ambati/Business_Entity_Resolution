"""Fixed H1 pair features. Empty comparisons have zero similarity, never agreement."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .contracts import Candidate, NormalizedRecord

FEATURE_SCHEMA_VERSION = "h1.1"
FEATURE_NAMES = (
    "name_exact", "name_token_jaccard", "name_token_containment",
    "name_char_ratio", "name_length_ratio", "name_prefix_ratio",
    "s1_name_missing", "candidate_name_missing",
    "address_exact", "address_token_jaccard", "address_token_containment",
    "address_char_ratio", "address_length_ratio", "s1_address_missing",
    "candidate_address_missing", "both_addresses_missing",
    "number_overlap_count", "number_jaccard", "number_any_agreement",
    "number_conflict", "is_source2", "is_source3", "retrieval_route_count",
)


def _sets(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[float, float]:
    left, right = set(a), set(b)
    if not left or not right:
        return 0.0, 0.0
    common = len(left & right)
    return common / len(left | right), common / min(len(left), len(right))


def _text(a: str, b: str) -> tuple[float, float, float, float]:
    if not a or not b:
        return 0.0, 0.0, 0.0, 0.0
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return (
        float(a == b),
        SequenceMatcher(None, a, b, autojunk=False).ratio(),
        min(len(a), len(b)) / max(len(a), len(b)),
        prefix / min(len(a), len(b)),
    )


def _numbers(record: NormalizedRecord) -> set[str]:
    # Decimal runs inside already-normalized tokens include 42 in "unit42".
    return set(re.findall(r"\d+", " ".join(record.name_tokens + record.address_tokens)))


def pair_features(s1: NormalizedRecord, candidate: NormalizedRecord,
                  retrieval: Candidate) -> tuple[float, ...]:
    """Return numeric features in FEATURE_NAMES order; IDs are used only for source prefix."""
    if not s1.raw.entity_id.startswith("S1-"):
        raise ValueError("s1 must be a Source 1 record")
    if candidate.raw.entity_id != retrieval.candidate_entity_id:
        raise ValueError("candidate record and retrieval ID disagree")
    name_j, name_cont = _sets(s1.name_tokens, candidate.name_tokens)
    name_exact, name_char, name_len, name_prefix = _text(s1.name_norm, candidate.name_norm)
    address_j, address_cont = _sets(s1.address_tokens, candidate.address_tokens)
    address_exact, address_char, address_len, _ = _text(s1.address_norm, candidate.address_norm)
    left, right = _numbers(s1), _numbers(candidate)
    common = len(left & right)
    number_j = common / len(left | right) if left and right else 0.0
    candidate_id = retrieval.candidate_entity_id
    return (
        name_exact, name_j, name_cont, name_char, name_len, name_prefix,
        float(not s1.name_norm), float(not candidate.name_norm),
        address_exact, address_j, address_cont, address_char, address_len,
        float(not s1.address_norm), float(not candidate.address_norm),
        float(not s1.address_norm and not candidate.address_norm),
        float(common), number_j, float(common > 0),
        float(bool(left and right and left != right)),
        float(candidate_id.startswith("S2-")), float(candidate_id.startswith("S3-")),
        float(len(retrieval.retrieval_routes)),
    )
