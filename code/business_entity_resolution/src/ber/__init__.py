"""Shared interfaces for the Business Entity Resolution pipeline."""

from .contracts import (
    Candidate,
    CandidateGroup,
    Decision,
    NormalizedRecord,
    Record,
    TruthRow,
    validate_entity_id,
)
from .split import is_validation_s1
from .metrics import (
    evaluate,
    evaluate_candidates,
    evaluate_from_tsv,
    evaluate_candidates_from_tsv,
    EvaluationResult,
    CandidateResult,
)
from .output import (
    write_outputs,
    validate_outputs,
    ValidationResult,
)

__version__ = "0.1.0"

__all__ = [
    "Record",
    "TruthRow",
    "NormalizedRecord",
    "Candidate",
    "CandidateGroup",
    "Decision",
    "validate_entity_id",
    "is_validation_s1",
    "evaluate",
    "evaluate_candidates",
    "evaluate_from_tsv",
    "evaluate_candidates_from_tsv",
    "EvaluationResult",
    "CandidateResult",
    "write_outputs",
    "validate_outputs",
    "ValidationResult",
]
