"""
output.py - Output writer and preflight validator for BER.

Owner: Suresh

Implements:
  write_outputs(test_s1, candidates, decisions, output_dir)
  validate_outputs(test_s1, matching_results, candidate_pairs) -> ValidationResult

Output schema:

  matching_results.tsv
    Header:  source1_entity_id<TAB>matched_entity_ids
    One row per test S1; comma-separated IDs or empty string for no match.

  candidate_pairs.tsv
    Header:  source1_entity_id<TAB>candidate_entity_ids
    One row per test S1; comma-separated IDs or empty string.

Rules:
  - UTF-8 encoding
  - Tab-separated columns
  - Exactly one row per test S1
  - Comma-separated IDs within the second column
  - No duplicate IDs within a list
  - Final predictions must be a subset of final candidates
  - Deterministic ordering (sorted S1 IDs; sorted ID lists within each row)
  - Valid S2-/S3- ID prefixes only in output ID lists
  - Empty lists are legal (empty string in second column)
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)


MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids"

VALID_MATCH_PREFIXES = ("S2-", "S3-")


# ---------------------------------------------------------------------------
# Validation result data class
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    """One preflight validation finding."""

    s1_id: Optional[str]
    problem_type: str
    invalid_ids: List[str] = field(default_factory=list)
    context: str = ""

    def __str__(self) -> str:
        base = f"[{self.problem_type}]"
        if self.s1_id:
            base += f" S1={self.s1_id}"
        if self.invalid_ids:
            base += f" ids={self.invalid_ids[:5]}"
        if self.context:
            base += f" | {self.context}"
        return base


@dataclass
class ValidationResult:
    """Structured result from validate_outputs()."""

    passed: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Summary statistics
    total_test_s1: int = 0
    matching_rows: int = 0
    candidate_rows: int = 0
    empty_matching_rows: int = 0
    empty_candidate_rows: int = 0
    france_total: int = 0
    france_in_output: int = 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Preflight validation: {status}",
            f"  Test S1 count:      {self.total_test_s1}",
            f"  Matching rows:      {self.matching_rows} "
            f"({self.empty_matching_rows} empty)",
            f"  Candidate rows:     {self.candidate_rows} "
            f"({self.empty_candidate_rows} empty)",
            f"  France coverage:    {self.france_in_output}/{self.france_total}",
            f"  Errors:             {len(self.errors)}",
            f"  Warnings:           {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("Errors:")
            for e in self.errors[:20]:
                lines.append(f"  {e}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings[:10]:
                lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal TSV helpers
# ---------------------------------------------------------------------------


def _read_s1_ids_from_source(path: str) -> List[str]:
    """Return all S1 entity_id values from a source TSV (streaming)."""
    ids: List[str] = []
    with open(path, encoding="utf-8", newline="") as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            entity_id = line.split("\t", 1)[0].strip()
            if entity_id:
                ids.append(entity_id)
    return ids


def _read_output_tsv(
    path: str,
) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    """Read an output TSV (matching or candidate).

    Returns:
        (ordered_s1_ids, s1_to_ids_list_map, raw_header_columns)
    """
    ordered_ids: List[str] = []
    mapping: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        header_line = fh.readline()
        header_cols = [c.strip().lower() for c in header_line.rstrip("\n").split("\t")]
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            if not s1_id:
                continue
            raw_ids = parts[1].strip() if len(parts) > 1 else ""
            if raw_ids:
                ids = [i.strip() for i in raw_ids.split(",")]
            else:
                ids = []
            ordered_ids.append(s1_id)
            mapping[s1_id] = ids
    return ordered_ids, mapping, header_cols


# ---------------------------------------------------------------------------
# write_outputs()
# ---------------------------------------------------------------------------


def write_outputs(
    test_s1: Iterable[str],
    candidates: Union[Dict[str, Iterable[str]], Iterable[Any]],
    decisions: Union[Dict[str, Iterable[str]], Iterable[Any]],
    output_dir: str,
) -> Tuple[str, str]:
    """Write matching_results.tsv and candidate_pairs.tsv.

    Args:
        test_s1:    Iterable of all test S1 entity IDs (preserved in exact input order).
        candidates: Mapping or iterable of candidates.
        decisions:  Mapping or iterable of decisions.
        output_dir: Directory to write output files into.

    Returns:
        (matching_path, candidate_path) absolute paths.

    Rules enforced:
      - Preserves exact test_s1 input sequence (no sorting).
      - Rejects duplicate S1 IDs in test_s1.
      - Strictly validates that all IDs start with S2- or S3- and have non-empty suffix.
      - Rejects duplicate IDs within candidate or decision lists.
      - Enforces that decisions are a subset of candidates for every S1.
      - Atomic write: writes to temporary files and atomically renames only upon full success.
        Cleans up temporary files on any failure, leaving no partial files.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_s1: List[str] = []
    seen_s1: Set[str] = set()
    for item in test_s1:
        s1_id = item.entity_id if hasattr(item, "entity_id") else item
        if not isinstance(s1_id, str) or not s1_id.startswith("S1-") or len(s1_id) <= 3:
            raise ValueError(f"Invalid S1 ID in test_s1: {s1_id!r}")
        if s1_id in seen_s1:
            raise ValueError(f"Duplicate S1 ID in test_s1: {s1_id}")
        seen_s1.add(s1_id)
        all_s1.append(s1_id)

    # Handle candidate intake (mapping vs stream)
    cand_map: Optional[Dict[str, List[str]]] = None
    cand_iter = None
    if isinstance(candidates, Mapping):
        cand_map = {}
        for sid, cids in candidates.items():
            cand_map[sid] = list(cids)
    else:
        cand_iter = iter(candidates)

    # Handle decision intake (mapping vs stream)
    dec_map: Optional[Dict[str, List[str]]] = None
    dec_iter = None
    if isinstance(decisions, Mapping):
        dec_map = {}
        for sid, dids in decisions.items():
            dec_map[sid] = list(dids)
    else:
        dec_iter = iter(decisions)

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_tmp = os.path.join(output_dir, ".matching_results.tsv.tmp")
    candidate_tmp = os.path.join(output_dir, ".candidate_pairs.tsv.tmp")

    try:
        with (
            open(matching_tmp, "w", encoding="utf-8", newline="") as mf,
            open(candidate_tmp, "w", encoding="utf-8", newline="") as cf,
        ):
            mf.write(MATCHING_HEADER + "\n")
            cf.write(CANDIDATE_HEADER + "\n")

            for s1_id in all_s1:
                # Retrieve raw candidate list
                if cand_map is not None:
                    raw_cands = cand_map.get(s1_id, [])
                else:
                    try:
                        cand_item = next(cand_iter)
                    except StopIteration:
                        raise ValueError(f"Candidate stream ended prematurely; missing {s1_id}")
                    if hasattr(cand_item, "source1_entity_id") and hasattr(cand_item, "candidates"):
                        c_sid = cand_item.source1_entity_id
                        raw_cands = [c.candidate_entity_id for c in cand_item.candidates]
                    else:
                        c_sid, raw_cands = cand_item
                    if c_sid != s1_id:
                        raise ValueError(f"Candidate stream misalignment: expected {s1_id}, got {c_sid}")

                # Retrieve raw decision list
                if dec_map is not None:
                    raw_preds = dec_map.get(s1_id, [])
                else:
                    try:
                        dec_item = next(dec_iter)
                    except StopIteration:
                        raise ValueError(f"Decision stream ended prematurely; missing {s1_id}")
                    if hasattr(dec_item, "source1_entity_id") and hasattr(dec_item, "matched_entity_ids"):
                        d_sid = dec_item.source1_entity_id
                        raw_preds = list(dec_item.matched_entity_ids)
                    else:
                        d_sid, raw_preds = dec_item
                    if d_sid != s1_id:
                        raise ValueError(f"Decision stream misalignment: expected {s1_id}, got {d_sid}")

                # Strict validation of candidate IDs
                cand_list = list(raw_cands)
                for cid in cand_list:
                    if not isinstance(cid, str) or not any(cid.startswith(p) and len(cid) > len(p) for p in VALID_MATCH_PREFIXES):
                        raise ValueError(f"Invalid candidate entity ID for {s1_id}: {cid!r}")
                if len(cand_list) != len(set(cand_list)):
                    raise ValueError(f"Duplicate candidate IDs for {s1_id}: {cand_list}")

                # Strict validation of decision IDs
                pred_list = list(raw_preds)
                for pid in pred_list:
                    if not isinstance(pid, str) or not any(pid.startswith(p) and len(pid) > len(p) for p in VALID_MATCH_PREFIXES):
                        raise ValueError(f"Invalid decision entity ID for {s1_id}: {pid!r}")
                if len(pred_list) != len(set(pred_list)):
                    raise ValueError(f"Duplicate decision IDs for {s1_id}: {pred_list}")

                # Strict subset validation
                cand_set = set(cand_list)
                pred_set = set(pred_list)
                violating = pred_set - cand_set
                if violating:
                    raise ValueError(
                        f"write_outputs: predictions not subset of candidates for {s1_id}. "
                        f"Violations: {sorted(violating)}"
                    )

                cand_str = ",".join(sorted(cand_list))
                pred_str = ",".join(sorted(pred_list))

                cf.write(f"{s1_id}\t{cand_str}\n")
                mf.write(f"{s1_id}\t{pred_str}\n")

        # Atomically publish upon complete success
        os.replace(matching_tmp, matching_path)
        os.replace(candidate_tmp, candidate_path)

    except Exception:
        for tmp in (matching_tmp, candidate_tmp):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        raise

    return matching_path, candidate_path


# ---------------------------------------------------------------------------
# validate_outputs()
# ---------------------------------------------------------------------------


def validate_outputs(
    test_s1_path: str,
    matching_results_path: str,
    candidate_pairs_path: Optional[str] = None,
    valid_s2s3_ids: Optional[Set[str]] = None,
    country_labels: Optional[Dict[str, str]] = None,
    require_candidates: bool = False,
) -> ValidationResult:
    """Preflight-validate output files before submission.

    Checks all formatting rules that the organizer scorer enforces,
    plus additional internal consistency rules.

    Args:
        test_s1_path:          Path to test_source1.tsv.
        matching_results_path: Path to matching_results.tsv to validate.
        candidate_pairs_path:  Path to candidate_pairs.tsv. Required for submission.
        valid_s2s3_ids:        Optional set of all valid S2/S3 IDs for
                               existence checking (memory-heavy; load only
                               when --check-ids is desired).
        country_labels:        Optional {s1_id: country} for France coverage.
        require_candidates:    If True, candidate_pairs_path is mandatory even if None.

    Returns:
        ValidationResult with passed=True only if all blocking checks pass.
    """
    result = ValidationResult(passed=True)

    def _add_error(s1_id, problem_type, invalid_ids=None, context=""):
        result.errors.append(
            ValidationError(
                s1_id=s1_id,
                problem_type=problem_type,
                invalid_ids=invalid_ids or [],
                context=context,
            )
        )
        result.passed = False

    # ---- 1. Read test S1 IDs ----
    if not os.path.isfile(test_s1_path):
        _add_error(None, "MISSING_TEST_S1_FILE", context=test_s1_path)
        return result

    required_s1: Set[str] = set()
    with open(test_s1_path, encoding="utf-8", newline="") as fh:
        fh.readline()
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            s1_id = line.split("\t", 1)[0].strip()
            if s1_id:
                required_s1.add(s1_id)
    result.total_test_s1 = len(required_s1)

    # ---- 2. France count from country labels ----
    if country_labels:
        result.france_total = sum(
            1 for c in country_labels.values() if c.strip().lower() == "france"
        )

    # ---- 3. Validate matching_results.tsv ----
    if not os.path.isfile(matching_results_path):
        _add_error(None, "MISSING_MATCHING_FILE", context=matching_results_path)
        return result

    try:
        m_ordered, m_mapping, m_header = _read_output_tsv(matching_results_path)
    except UnicodeDecodeError as e:
        _add_error(None, "ENCODING_ERROR",
                   context=f"{matching_results_path}: {e}")
        return result

    # Check header
    expected_m_header = ["source1_entity_id", "matched_entity_ids"]
    if m_header != expected_m_header:
        _add_error(
            None, "WRONG_HEADER",
            context=f"matching_results.tsv: got {m_header}, expected {expected_m_header}",
        )

    # Check for duplicate S1 rows
    seen_s1: Set[str] = set()
    for s1_id in m_ordered:
        if s1_id in seen_s1:
            _add_error(s1_id, "DUPLICATE_S1_ROW",
                       context="matching_results.tsv")
        seen_s1.add(s1_id)

    result.matching_rows = len(m_mapping)
    result.empty_matching_rows = sum(1 for v in m_mapping.values() if not v)

    # Missing / extra S1 rows
    missing_s1 = required_s1 - seen_s1
    extra_s1 = seen_s1 - required_s1

    if missing_s1:
        _add_error(
            None, "MISSING_S1_ROWS",
            invalid_ids=sorted(missing_s1)[:10],
            context=f"{len(missing_s1)} S1 IDs missing from matching_results.tsv",
        )
    if extra_s1:
        _add_error(
            None, "EXTRA_S1_ROWS",
            invalid_ids=sorted(extra_s1)[:10],
            context=f"{len(extra_s1)} rows for S1 IDs not in test set",
        )

    # ID validity in matching_results.tsv (check duplicate IDs and prefixes)
    for s1_id, pred_ids in m_mapping.items():
        if len(pred_ids) != len(set(pred_ids)):
            _add_error(
                s1_id, "DUPLICATE_IDS",
                invalid_ids=pred_ids,
                context=f"Duplicate entity IDs in matching_results.tsv for {s1_id}: {pred_ids}",
            )
        bad_ids = [
            pid for pid in pred_ids
            if not any(pid.startswith(p) and len(pid) > len(p) for p in VALID_MATCH_PREFIXES)
        ]
        if bad_ids:
            _add_error(s1_id, "INVALID_ID_PREFIX",
                       invalid_ids=bad_ids[:5],
                       context="matching_results.tsv")
        if valid_s2s3_ids is not None:
            unknown = [pid for pid in pred_ids if pid not in valid_s2s3_ids]
            if unknown:
                _add_error(s1_id, "UNKNOWN_S2S3_IDS",
                           invalid_ids=unknown[:5],
                           context="matching_results.tsv")

    # France coverage in matching_results
    if country_labels:
        france_ids = {
            sid for sid, c in country_labels.items()
            if c.strip().lower() == "france"
        }
        france_in_output = france_ids & seen_s1
        result.france_in_output = len(france_in_output)
        france_missing = france_ids - seen_s1
        if france_missing:
            _add_error(
                None, "FRANCE_COVERAGE",
                invalid_ids=sorted(france_missing)[:10],
                context=f"{len(france_missing)} French S1 IDs have no output row",
            )

    # ---- 4. Validate candidate_pairs.tsv (mandatory) ----
    c_mapping: Optional[Dict[str, List[str]]] = None
    if candidate_pairs_path is None:
        if require_candidates:
            _add_error(
                None, "MISSING_CANDIDATE_FILE",
                context="candidate_pairs.tsv is mandatory for submission validation but was not provided",
            )
    elif not os.path.isfile(candidate_pairs_path):
        _add_error(
            None, "MISSING_CANDIDATE_FILE",
            context=f"candidate_pairs.tsv not found at {candidate_pairs_path}. File is required in final submission zip.",
        )
    else:
        try:
            c_ordered, c_mapping, c_header = _read_output_tsv(
                candidate_pairs_path
            )
        except UnicodeDecodeError as e:
            _add_error(None, "ENCODING_ERROR",
                       context=f"{candidate_pairs_path}: {e}")
            c_mapping = None

        if c_mapping is not None:
            expected_c_header = ["source1_entity_id", "candidate_entity_ids"]
            if c_header != expected_c_header:
                _add_error(
                    None, "WRONG_HEADER",
                    context=f"candidate_pairs.tsv: got {c_header}, "
                            f"expected {expected_c_header}",
                )

            result.candidate_rows = len(c_mapping)
            result.empty_candidate_rows = sum(
                1 for v in c_mapping.values() if not v
            )

            c_seen: Set[str] = set()
            for s1_id in c_ordered:
                if s1_id in c_seen:
                    _add_error(s1_id, "DUPLICATE_S1_ROW",
                               context="candidate_pairs.tsv")
                c_seen.add(s1_id)

            c_missing = required_s1 - c_seen
            c_extra = c_seen - required_s1
            if c_missing:
                _add_error(
                    None, "MISSING_S1_ROWS",
                    invalid_ids=sorted(c_missing)[:10],
                    context=f"{len(c_missing)} S1 IDs missing from "
                            "candidate_pairs.tsv",
                )
            if c_extra:
                _add_error(
                    None, "EXTRA_S1_ROWS",
                    invalid_ids=sorted(c_extra)[:10],
                    context=f"{len(c_extra)} extra rows in candidate_pairs.tsv",
                )

            for s1_id, cand_ids in c_mapping.items():
                if len(cand_ids) != len(set(cand_ids)):
                    _add_error(
                        s1_id, "DUPLICATE_IDS",
                        invalid_ids=cand_ids,
                        context=f"Duplicate entity IDs in candidate_pairs.tsv for {s1_id}: {cand_ids}",
                    )
                bad_ids = [
                    cid for cid in cand_ids
                    if not any(cid.startswith(p) and len(cid) > len(p) for p in VALID_MATCH_PREFIXES)
                ]
                if bad_ids:
                    _add_error(s1_id, "INVALID_ID_PREFIX",
                               invalid_ids=bad_ids[:5],
                               context="candidate_pairs.tsv")
                if valid_s2s3_ids is not None:
                    unknown = [
                        cid for cid in cand_ids if cid not in valid_s2s3_ids
                    ]
                    if unknown:
                        _add_error(s1_id, "UNKNOWN_S2S3_IDS",
                                   invalid_ids=unknown[:5],
                                   context="candidate_pairs.tsv")

    # ---- 5. Subset check: predictions must be subset of candidates ----
    if c_mapping is not None and m_mapping:
        for s1_id, pred_ids in m_mapping.items():
            if not pred_ids:
                continue
            cand_ids = set(c_mapping.get(s1_id, []))
            violating = set(pred_ids) - cand_ids
            if violating:
                _add_error(
                    s1_id, "PREDICTION_NOT_SUBSET_OF_CANDIDATES",
                    invalid_ids=sorted(violating)[:5],
                )

    return result


# ---------------------------------------------------------------------------
# Organizer validator integration
# ---------------------------------------------------------------------------


def run_organizer_validator(
    matching_path: str,
    candidate_path: Optional[str],
    test_dir: str,
    check_ids: bool = False,
    student_resource_dir: Optional[str] = None,
) -> Tuple[int, str]:
    """Run the organizer's utils/validate_submission.py and return (exit_code, output).

    Args:
        matching_path:         Path to matching_results.tsv.
        candidate_path:        Path to candidate_pairs.tsv (or None).
        test_dir:              Path to the test dataset directory.
        check_ids:             Whether to pass --check-ids flag.
        student_resource_dir:  Root of the student_resource directory
                               (default: auto-detected from this file's location).

    Returns:
        (exit_code, stdout+stderr combined).
        exit_code == 0 means FORMAT/COVERAGE/CONSISTENCY PASS.
        exit_code != 0 means issues were found.

    IMPORTANT: A PASS here means format correctness only, NOT ML quality.

    Command (run from student_resource_dir):
        python utils/validate_submission.py \\
            --matching <matching> \\
            --candidate <candidate> \\
            --test-dir <test_dir>
    """
    if student_resource_dir is None:
        # Auto-detect: walk up from this file's location to find student_resource/
        this_dir = os.path.dirname(os.path.abspath(__file__))
        search_dir = this_dir
        for _ in range(8):
            candidate_sr = os.path.join(search_dir, "student_resource")
            if os.path.isdir(candidate_sr) and os.path.isfile(
                os.path.join(candidate_sr, "utils", "validate_submission.py")
            ):
                student_resource_dir = candidate_sr
                break
            search_dir = os.path.dirname(search_dir)

    if student_resource_dir is None or not os.path.isdir(student_resource_dir):
        return 1, "ERROR: Could not locate student_resource directory."

    validator_path = os.path.join(student_resource_dir, "utils", "validate_submission.py")
    if not os.path.isfile(validator_path):
        return 1, f"ERROR: Organizer validator not found at {validator_path}"

    cmd = [
        sys.executable,
        validator_path,
        "--matching", matching_path,
        "--test-dir", test_dir,
    ]
    if candidate_path:
        cmd += ["--candidate", candidate_path]
    if check_ids:
        cmd.append("--check-ids")

    try:
        proc = subprocess.run(
            cmd,
            cwd=student_resource_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output
    except Exception as e:
        return 1, f"ERROR running organizer validator: {e}"
