"""Multi-route candidate retrieval: exact/fuzzy queries, union, dedup, cap.

Provides ``iter_candidates`` which yields one ``CandidateGroup`` per Source-1
record in input order.  Each group contains deduplicated, deterministically
ranked candidates from enabled retrieval routes.

Contract reference: docs/CONTRACTS.md §3 (Sabeena section).

PR S1 routes
~~~~~~~~~~~~
- **Route A — exact_name**: exact normalized business-name lookup, within
  the country partition.
- **Route B — name_token**: discriminative name-token blocking with
  frequency and stopword filtering.

PR S2 routes (planned)
~~~~~~~~~~~~~~~~~~~~~~
- Route C — address: independent address/locality blocking.
- Route D — combined: conjunction blocking keys (e.g. country + name prefix).
- Route E — fuzzy_name: character-gram or compact text index.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from .contracts import Candidate, CandidateGroup

# Prefer Thulasi's modules; fall back to local shims.
try:
    from .normalize import normalize_record  # type: ignore[import-not-found]
except ImportError:
    from .normalize_shim import normalize_record

try:
    from .data import read_source  # type: ignore[import-not-found]
except ImportError:
    from .data_shim import read_source

from .index import IndexConfig, IndexStore, detect_split

# BlockingConfig is an alias for IndexConfig to provide symmetrical configuration naming
BlockingConfig = IndexConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common business terms to exclude from token blocking
# ---------------------------------------------------------------------------

COMMON_BUSINESS_TERMS: frozenset[str] = frozenset({
    # English determiners / prepositions
    "a", "an", "the", "of", "in", "at", "on", "to", "for", "by", "and", "or",
    # Legal suffixes and corporate designators
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "ltd", "limited", "llc", "llp", "plc", "sa", "sarl", "srl", "gmbh",
    "ag", "pvt", "private", "public", "nv", "bv", "pty",
    # Generic business words
    "group", "holding", "holdings", "enterprise", "enterprises",
    "services", "service", "solutions", "solution", "systems", "system",
    "international", "global", "national", "general",
    "store", "stores", "shop", "shops", "mart", "market",
    "restaurant", "restaurants", "hotel", "hotels", "cafe", "bar",
    "industries", "industry", "trading", "traders",
    "associates", "associate", "partners", "partner", "consulting",
    "technology", "technologies", "tech",
})


# ---------------------------------------------------------------------------
# Route A: exact normalized name
# ---------------------------------------------------------------------------

def _route_exact_name(
    s1_norm: "NormalizedRecord",
    index_store: IndexStore,
    config: IndexConfig,
) -> dict[str, float]:
    """Exact normalized business-name lookup within the country partition.

    Returns {candidate_entity_id: score} where score is 1.0.
    Uses country partition when the S1 record has a country; otherwise
    falls back to a global lookup.
    """
    if not s1_norm.name_norm:
        return {}

    country = s1_norm.country_key if s1_norm.country_key else None
    candidates = index_store.lookup_exact_name(s1_norm.name_norm, country_key=country)

    if len(candidates) > config.max_block_size:
        logger.warning(
            "exact_name: S1 %s name %r matched %d entities (> max_block_size=%d), "
            "truncating to %d",
            s1_norm.raw.entity_id,
            s1_norm.name_norm,
            len(candidates),
            config.max_block_size,
            config.max_block_size,
        )
        candidates = candidates[: config.max_block_size]

    return {eid: 1.0 for eid in candidates}


# ---------------------------------------------------------------------------
# Route B: discriminative name tokens
# ---------------------------------------------------------------------------

def _route_name_token(
    s1_norm: "NormalizedRecord",
    index_store: IndexStore,
    config: IndexConfig,
) -> dict[str, float]:
    """Discriminative name-token blocking.

    For each rare, informative name token in the S1 record, retrieves all
    S2/S3 entities sharing that token.  Scores candidates by the fraction
    of the query's rare tokens they share.

    Excludes tokens that are:
    - in ``COMMON_BUSINESS_TERMS``
    - shorter than ``config.token_min_length``
    - above the document-frequency threshold
    - whose posting list exceeds ``config.max_block_size``
    """
    if not s1_norm.name_tokens:
        return {}

    # Compute the document-frequency threshold.
    # Use max(10, ...) to avoid filtering everything on tiny corpora.
    freq_threshold = max(10, int(index_store.total_records * config.token_max_doc_freq_ratio))

    # Identify rare, informative tokens
    rare_tokens: list[str] = []
    for token in s1_norm.name_tokens:
        if len(token) < config.token_min_length:
            continue
        if token in COMMON_BUSINESS_TERMS:
            continue
        freq = index_store.get_token_doc_freq(token)
        if freq == 0:
            continue  # token not in any indexed record
        if freq > freq_threshold:
            logger.debug(
                "name_token: skipping token %r (freq=%d > threshold=%d) for S1 %s",
                token, freq, freq_threshold, s1_norm.raw.entity_id,
            )
            continue
        rare_tokens.append(token)

    if not rare_tokens:
        return {}

    # Collect candidates from all rare tokens
    candidate_token_count: dict[str, int] = {}
    skipped_tokens = 0
    for token in rare_tokens:
        postings = index_store.lookup_token(token)
        if len(postings) > config.max_block_size:
            logger.debug(
                "name_token: token %r has %d postings (> max_block_size=%d), skipping",
                token, len(postings), config.max_block_size,
            )
            skipped_tokens += 1
            continue
        for eid in postings:
            candidate_token_count[eid] = candidate_token_count.get(eid, 0) + 1

    if not candidate_token_count:
        return {}

    # Score: fraction of query's usable rare tokens that the candidate shares
    usable_rare = len(rare_tokens) - skipped_tokens
    if usable_rare <= 0:
        return {}

    return {
        eid: count / usable_rare
        for eid, count in candidate_token_count.items()
    }


# ---------------------------------------------------------------------------
# Route C: address token blocking (placeholder for S2 PR)
# ---------------------------------------------------------------------------

def _route_address(
    s1_norm: "NormalizedRecord",
    index_store: IndexStore,
    config: IndexConfig,
) -> dict[str, float]:
    """Independent address/locality route — S2 PR placeholder."""
    if not s1_norm.address_tokens:
        return {}

    # Require at least 2 shared address tokens for a hit
    candidate_token_count: dict[str, int] = {}
    for token in set(s1_norm.address_tokens):
        if len(token) < config.token_min_length:
            continue
        postings = index_store.lookup_address_token(token)
        if len(postings) > config.max_block_size:
            continue
        for eid in postings:
            candidate_token_count[eid] = candidate_token_count.get(eid, 0) + 1

    total_addr_tokens = sum(
        1 for t in set(s1_norm.address_tokens) if len(t) >= config.token_min_length
    )
    if total_addr_tokens == 0:
        return {}

    return {
        eid: count / total_addr_tokens
        for eid, count in candidate_token_count.items()
        if count >= 2  # require at least 2 shared address tokens
    }


# ---------------------------------------------------------------------------
# Main candidate iterator
# ---------------------------------------------------------------------------

def iter_candidates(
    source1_path: str,
    index_store: IndexStore,
    config: IndexConfig | dict | None = None,
) -> Iterator[CandidateGroup]:
    """Yield one ``CandidateGroup`` per S1 record in input order.

    Queries all enabled retrieval routes, unions results, deduplicates by
    candidate entity ID, ranks deterministically, applies the candidate cap,
    and yields the resulting group.

    An S1 record with no matching candidates yields an empty group (zero
    candidates).  The system does not fabricate candidates.

    Parameters
    ----------
    source1_path : str
        Path to the Source-1 TSV file.
    index_store : IndexStore
        A previously built and opened index of S2/S3 records.
    config : IndexConfig or dict, optional
        Retrieval configuration.  Defaults to ``IndexConfig()``.
    """
    if config is None:
        config = IndexConfig()
    elif isinstance(config, dict):
        config = IndexConfig.from_dict(config)

    # Validate split consistency
    s1_split = detect_split(source1_path)
    idx_split = index_store.manifest.split
    if s1_split != "unspecified" and idx_split != "unspecified" and s1_split != idx_split:
        raise ValueError(
            f"Split mismatch: source1 ({source1_path}) belongs to split {s1_split!r}, "
            f"but candidate index was built for split {idx_split!r}. "
            f"Cross-split retrieval is prohibited."
        )

    # -- Statistics accumulators ---------------------------------------------
    s1_count = 0
    total_candidates = 0
    empty_count = 0
    max_candidates = 0
    route_contribution: dict[str, int] = {}  # route -> number of candidate hits
    oversized_blocks = 0
    t0 = time.time()

    for s1_record in read_source(source1_path, "S1-"):
        s1_norm = normalize_record(s1_record)
        s1_count += 1

        # -- Query each enabled route ----------------------------------------
        route_results: dict[str, dict[str, float]] = {}

        if config.exact_name:
            hits = _route_exact_name(s1_norm, index_store, config)
            if hits:
                route_results["exact_name"] = hits

        if config.name_token:
            hits = _route_name_token(s1_norm, index_store, config)
            if hits:
                route_results["name_token"] = hits

        if config.address:
            hits = _route_address(s1_norm, index_store, config)
            if hits:
                route_results["address"] = hits

        # Future S2 routes: combined, fuzzy_name

        # -- Union and build Candidate objects --------------------------------
        all_candidate_ids: set[str] = set()
        for hits in route_results.values():
            all_candidate_ids.update(hits.keys())

        candidates: list[Candidate] = []
        for cid in all_candidate_ids:
            routes: list[str] = []
            scores: dict[str, float] = {}
            for route_name, hits in route_results.items():
                if cid in hits:
                    routes.append(route_name)
                    scores[route_name] = hits[cid]
            sorted_routes = tuple(sorted(routes))
            candidates.append(
                Candidate(
                    candidate_entity_id=cid,
                    retrieval_routes=sorted_routes,
                    route_scores={r: scores[r] for r in sorted_routes},
                )
            )

        # -- Deterministic ranking -------------------------------------------
        # Primary: more routes (descending)
        # Secondary: higher total score (descending)
        # Tertiary: entity ID alphabetical (ascending, for stability)
        candidates.sort(
            key=lambda c: (
                -len(c.retrieval_routes),
                -sum(c.route_scores.values()),
                c.candidate_entity_id,
            )
        )

        # -- Apply global cap after union ------------------------------------
        cap = config.max_candidates_per_s1
        if len(candidates) > cap:
            oversized_blocks += 1
        candidates = candidates[:cap]

        # -- Accumulate statistics -------------------------------------------
        n = len(candidates)
        total_candidates += n
        if n > max_candidates:
            max_candidates = n
        if n == 0:
            empty_count += 1
        for c in candidates:
            for r in c.retrieval_routes:
                route_contribution[r] = route_contribution.get(r, 0) + 1

        yield CandidateGroup(
            source1_entity_id=s1_record.entity_id,
            candidates=tuple(candidates),
        )

    # -- Final summary -------------------------------------------------------
    elapsed = time.time() - t0
    avg = total_candidates / max(1, s1_count)
    logger.info("Candidate retrieval complete:")
    logger.info("  S1 records processed:     %d", s1_count)
    logger.info("  Total candidate pairs:    %d", total_candidates)
    logger.info("  Average candidates / S1:  %.1f", avg)
    logger.info("  Max candidates for an S1: %d", max_candidates)
    logger.info("  S1 with zero candidates:  %d (%.1f%%)", empty_count,
                100.0 * empty_count / max(1, s1_count))
    logger.info("  S1 groups exceeding cap:  %d", oversized_blocks)
    logger.info("  Route contributions:      %s", dict(sorted(route_contribution.items())))
    logger.info("  Runtime:                  %.1fs", elapsed)
