"""Deterministic candidate pair deduplication and ranking."""

from __future__ import annotations

from typing import Iterable, Sequence

from ..contracts import Candidate, CandidateGroup, validate_entity_id, CANDIDATE_PREFIXES


def deduplicate_candidates(
    candidates: Iterable[Candidate | str],
    cap: int | None = None,
    default_route: str = "exact_name",
    default_score: float = 1.0,
) -> tuple[Candidate, ...]:
    """Deduplicate candidate records deterministically for an entity.

    Parameters
    ----------
    candidates : Iterable[Candidate | str]
        Candidate objects or raw candidate entity IDs.
    cap : int, optional
        Maximum number of candidates to retain after deduplication.
    default_route : str
        Route name to assign if string IDs are passed.
    default_score : float
        Route score to assign if string IDs are passed.

    Returns
    -------
    tuple[Candidate, ...]
        Deduplicated, deterministically ordered tuple of Candidate instances.
    """
    candidate_map: dict[str, Candidate] = {}

    for item in candidates:
        if isinstance(item, str):
            cid = validate_entity_id(item, CANDIDATE_PREFIXES)
            if cid not in candidate_map:
                candidate_map[cid] = Candidate(
                    candidate_entity_id=cid,
                    retrieval_routes=(default_route,),
                    route_scores={default_route: default_score},
                )
        elif isinstance(item, Candidate):
            cid = item.candidate_entity_id
            if cid in candidate_map:
                existing = candidate_map[cid]
                # Merge routes and scores
                combined_routes = tuple(sorted(set(existing.retrieval_routes) | set(item.retrieval_routes)))
                combined_scores = dict(existing.route_scores)
                combined_scores.update(item.route_scores)
                candidate_map[cid] = Candidate(
                    candidate_entity_id=cid,
                    retrieval_routes=combined_routes,
                    route_scores=combined_scores,
                )
            else:
                candidate_map[cid] = item
        else:
            raise TypeError(f"Expected Candidate or str ID, got {type(item)}")

    # Deterministic ranking:
    # 1. More retrieval routes (descending)
    # 2. Higher cumulative route score (descending)
    # 3. Candidate entity ID alphabetically (ascending) for tie-break stability
    sorted_candidates = sorted(
        candidate_map.values(),
        key=lambda c: (
            -len(c.retrieval_routes),
            -sum(c.route_scores.values()),
            c.candidate_entity_id,
        ),
    )

    if cap is not None and cap >= 0:
        sorted_candidates = sorted_candidates[:cap]

    return tuple(sorted_candidates)
