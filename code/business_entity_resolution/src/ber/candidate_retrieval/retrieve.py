"""Candidate retrieval execution and candidate_pairs.tsv emission."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from ..contracts import Candidate, CandidateGroup, validate_entity_id, CANDIDATE_PREFIXES
from ..index import IndexStore, detect_split
from .config import CandidateRetrievalConfig
from .deduplicate import deduplicate_candidates

# Prefer canonical modules; fall back to shims
try:
    from ..normalize import normalize_record  # type: ignore[import-not-found]
except ImportError:
    from ..normalize_shim import normalize_record

try:
    from ..data import read_source  # type: ignore[import-not-found]
except ImportError:
    from ..data_shim import read_source

logger = logging.getLogger(__name__)


def retrieve_exact_name_candidates(
    source1_path: Path | str,
    index_store: IndexStore,
    config: CandidateRetrievalConfig | None = None,
) -> tuple[Iterator[CandidateGroup], dict[str, int]]:
    """Retrieve candidate records using normalized exact-name blocking.

    Parameters
    ----------
    source1_path : Path | str
        Path to the Source-1 TSV file.
    index_store : IndexStore
        A previously built and opened index of S2/S3 records.
    config : CandidateRetrievalConfig, optional
        Retrieval configuration with caps and safety cutoffs.

    Returns
    -------
    (candidate_groups_generator, diagnostics_dict)
    """
    if config is None:
        config = CandidateRetrievalConfig()

    s1_split = detect_split(source1_path)
    idx_split = index_store.manifest.split
    if s1_split != "unspecified" and idx_split != "unspecified" and s1_split != idx_split:
        raise ValueError(
            f"Split mismatch: source1 ({source1_path}) is {s1_split!r}, "
            f"but candidate index was built for {idx_split!r}. Cross-split retrieval is prohibited."
        )

    diagnostics = {
        "s1_records": 0,
        "total_pairs_pre_cap": 0,
        "total_pairs_post_cap": 0,
        "oversized_blocks": 0,
        "candidates_avoided": 0,
        "empty_groups": 0,
    }

    def _generator() -> Iterator[CandidateGroup]:
        for s1_rec in read_source(source1_path, "S1-"):
            diagnostics["s1_records"] += 1
            s1_norm = normalize_record(s1_rec)

            matched_ids: list[str] = []
            if s1_norm.name_norm:
                country = s1_norm.country_key if s1_norm.country_key else None
                raw_matches = index_store.lookup_exact_name(s1_norm.name_norm, country_key=country)
                block_len = len(raw_matches)

                # Large-block safeguard (Step 11)
                if block_len > config.max_block_size:
                    diagnostics["oversized_blocks"] += 1
                    avoided = block_len - config.max_block_size
                    diagnostics["candidates_avoided"] += avoided
                    logger.debug(
                        "Oversized block for S1 %s name %r: %d candidates truncated to %d",
                        s1_rec.entity_id,
                        s1_norm.name_norm,
                        block_len,
                        config.max_block_size,
                    )
                    raw_matches = raw_matches[: config.max_block_size]

                matched_ids = raw_matches

            diagnostics["total_pairs_pre_cap"] += len(matched_ids)

            # Deduplicate and cap candidates deterministically
            deduped = deduplicate_candidates(
                matched_ids,
                cap=config.max_candidates_per_entity,
                default_route="exact_name",
                default_score=1.0,
            )

            pair_count = len(deduped)
            diagnostics["total_pairs_post_cap"] += pair_count
            if pair_count == 0:
                diagnostics["empty_groups"] += 1

            yield CandidateGroup(
                source1_entity_id=s1_rec.entity_id,
                candidates=deduped,
            )

    return _generator(), diagnostics


def write_candidate_pairs_tsv(
    candidate_groups: Iterator[CandidateGroup] | Iterable[CandidateGroup],
    output_path: Path | str,
) -> int:
    """Generate candidate_pairs.tsv adhering to CONTRACTS.md §4.

    Header: source1_entity_id<TAB>candidate_entity_ids
    Row for every S1, comma-separated candidate IDs without quotes.

    Returns the number of S1 rows written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    seen_s1: set[str] = set()

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for group in candidate_groups:
            sid = group.source1_entity_id
            validate_entity_id(sid, ("S1-",))
            if sid in seen_s1:
                raise ValueError(f"Duplicate Source-1 entity ID {sid} in candidate stream")
            seen_s1.add(sid)

            # Ensure candidate IDs are valid and unique
            candidate_ids = [c.candidate_entity_id for c in group.candidates]
            for cid in candidate_ids:
                validate_entity_id(cid, CANDIDATE_PREFIXES)
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"Duplicate candidate IDs in group for S1 {sid}")

            joined_cids = ",".join(candidate_ids)
            f.write(f"{sid}\t{joined_cids}\n")
            rows_written += 1

    return rows_written
