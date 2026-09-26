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
from .data import SOURCE_HEADERS, TRUTH_HEADERS, read_source, read_truth
from .metrics import (
    CandidateResult,
    EvaluationResult,
    evaluate,
    evaluate_candidates,
    evaluate_candidates_from_tsv,
    evaluate_from_tsv,
)
from .normalize import (
    NORMALIZATION_VERSION,
    normalize_country,
    normalize_record,
    normalize_text,
)
from .output import (
    ValidationResult,
    validate_outputs,
    write_outputs,
)
from .quality import (
    SourceQualityReport,
    TruthQualityReport,
    audit_dataset,
    check_duplicate_ids_partitioned,
    validate_source_file,
    validate_truth_file,
)
from .split import is_validation_s1

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
    "read_source",
    "read_truth",
    "SOURCE_HEADERS",
    "TRUTH_HEADERS",
    "NORMALIZATION_VERSION",
    "normalize_record",
    "normalize_text",
    "normalize_country",
    "validate_source_file",
    "validate_truth_file",
    "check_duplicate_ids_partitioned",
    "audit_dataset",
    "SourceQualityReport",
    "TruthQualityReport",
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
