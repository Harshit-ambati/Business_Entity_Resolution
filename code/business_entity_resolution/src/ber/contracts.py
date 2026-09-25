"""Small, shared in-memory contracts; full TSV validation belongs to later modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping


SOURCE_PREFIXES = ("S1-", "S2-", "S3-")
CANDIDATE_PREFIXES = ("S2-", "S3-")


def validate_entity_id(entity_id: str, allowed_prefixes: tuple[str, ...] = SOURCE_PREFIXES) -> str:
    """Check only the source prefix and a nonempty opaque suffix."""
    if not isinstance(entity_id, str) or not any(
        entity_id.startswith(prefix) and len(entity_id) > len(prefix)
        for prefix in allowed_prefixes
    ):
        raise ValueError(f"Invalid entity ID {entity_id!r}; expected prefix in {allowed_prefixes} and a nonempty suffix")
    return entity_id


def _validate_unique_ids(ids: tuple[str, ...], label: str) -> None:
    if not isinstance(ids, tuple):
        raise TypeError(f"{label} must be a tuple")
    for entity_id in ids:
        validate_entity_id(entity_id, CANDIDATE_PREFIXES)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {label}")


@dataclass(frozen=True, slots=True)
class Record:
    entity_id: str
    business_name: str
    business_address: str
    country: str

    def __post_init__(self) -> None:
        validate_entity_id(self.entity_id)


@dataclass(frozen=True, slots=True)
class TruthRow:
    source1_entity_id: str
    matched_entity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_entity_id(self.source1_entity_id, ("S1-",))
        _validate_unique_ids(self.matched_entity_ids, "matched IDs")


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    raw: Record
    name_norm: str
    address_norm: str
    name_tokens: tuple[str, ...]
    address_tokens: tuple[str, ...]
    country_key: str


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_entity_id: str
    retrieval_routes: tuple[str, ...]
    route_scores: Mapping[str, float] = field(hash=False)

    def __post_init__(self) -> None:
        validate_entity_id(self.candidate_entity_id, CANDIDATE_PREFIXES)
        routes = self.retrieval_routes
        if not isinstance(routes, tuple) or not routes or any(not isinstance(route, str) or not route for route in routes):
            raise ValueError("retrieval_routes must be a nonempty tuple of route names")
        if tuple(sorted(set(routes))) != routes:
            raise ValueError("retrieval_routes must be a sorted unique tuple")
        scores = dict(self.route_scores)
        if any(not isinstance(route, str) or not route for route in scores):
            raise ValueError("route_scores keys must be nonempty route names")
        if any(route not in routes for route in scores):
            raise ValueError("route_scores keys must belong to retrieval_routes")
        for score in scores.values():
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score):
                raise ValueError("route_scores must be finite numbers")
        object.__setattr__(self, "route_scores", MappingProxyType({route: float(score) for route, score in scores.items()}))


@dataclass(frozen=True, slots=True)
class CandidateGroup:
    source1_entity_id: str
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        validate_entity_id(self.source1_entity_id, ("S1-",))
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple")
        ids = [candidate.candidate_entity_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate IDs")


@dataclass(frozen=True, slots=True)
class Decision:
    source1_entity_id: str
    matched_entity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_entity_id(self.source1_entity_id, ("S1-",))
        _validate_unique_ids(self.matched_entity_ids, "matched IDs")
