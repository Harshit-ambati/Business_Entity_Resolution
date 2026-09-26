"""Shared interfaces for the Business Entity Resolution pipeline (H0)."""

from .contracts import Candidate, CandidateGroup, Decision, NormalizedRecord, Record, TruthRow
from .data import SOURCE_HEADERS, TRUTH_HEADERS, read_source, read_truth
from .normalize import (
    NORMALIZATION_VERSION,
    normalize_country,
    normalize_record,
    normalize_text,
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

__all__ = [
    "Record",
    "TruthRow",
    "NormalizedRecord",
    "Candidate",
    "CandidateGroup",
    "Decision",
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
]

