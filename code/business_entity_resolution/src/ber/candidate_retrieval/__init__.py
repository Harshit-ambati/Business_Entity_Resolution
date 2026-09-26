"""Candidate retrieval & blocking package (Sabeena).

Provides scalable, bounded, reproducible candidate retrieval for business entity resolution.
"""

from __future__ import annotations

from .config import CandidateRetrievalConfig
from .deduplicate import deduplicate_candidates
from .metrics import RetrievalMetrics, evaluate_retrieval
from .retrieve import retrieve_exact_name_candidates, write_candidate_pairs_tsv

# Expose canonical indexing utilities
from ..index import IndexConfig, IndexManifest, IndexStore, build_index, open_index

__all__ = [
    "CandidateRetrievalConfig",
    "IndexConfig",
    "IndexManifest",
    "IndexStore",
    "RetrievalMetrics",
    "build_index",
    "deduplicate_candidates",
    "evaluate_retrieval",
    "open_index",
    "retrieve_exact_name_candidates",
    "write_candidate_pairs_tsv",
]
