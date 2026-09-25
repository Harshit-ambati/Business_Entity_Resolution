"""Shared interfaces for the Business Entity Resolution pipeline (H0)."""

from .contracts import Candidate, CandidateGroup, Decision, NormalizedRecord, Record, TruthRow
from .split import is_validation_s1

__all__ = [
    "Record",
    "TruthRow",
    "NormalizedRecord",
    "Candidate",
    "CandidateGroup",
    "Decision",
    "is_validation_s1",
]
