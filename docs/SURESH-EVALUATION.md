# Suresh Workstream — Evaluation, Diagnostics, Output Validation

**Owner:** Suresh | **Branch:** `feat/evaluation-output`

## Official Macro F0.5 Formula

For each Source-1 entity with true match set **T** and predicted match set **P**:

| Case | Score |
|------|-------|
| T = ∅ and P = ∅ | **1.0** |
| T = ∅ and P ≠ ∅ | **0.0** |
| T ≠ ∅ and P = ∅ | **0.0** |
| Otherwise | `1.25 × precision × recall / (0.25 × precision + recall)` |

Where:
- `precision = |T ∩ P| / |P|`
- `recall    = |T ∩ P| / |T|`

**Final score:** `macro_F0.5 = mean(score over ALL S1 entities)`

> ⚠️ This is a **macro** average — every S1 entity contributes equally, including true-empty ones.
> Do not use micro-averaged or pair-level F0.5.

## Empty-Case Behavior

- **Both empty** (T=∅, P=∅): score = 1.0 (correct abstention)
- **True empty, predicted non-empty**: score = 0.0 (false alarm)
- **True non-empty, predicted empty**: score = 0.0 (complete miss)

Training data: 5.6% of S1 entities have no matches. These contribute 1.0 when correctly predicted empty.

## Candidate Metrics

`evaluate_candidates(truth, candidates)` returns:

| Metric | Description |
|--------|-------------|
| `true_edge_recall` | Fraction of true (S1,S2/S3) links present in candidate set C |
| `complete_s1_recall` | Fraction of **matched** S1s where ALL true links are in C |
| `oracle_macro_f0_5` | Macro F0.5 when predicting exactly T ∩ C per S1 |
| `total_candidate_pairs` | Total (S1, S2/S3) candidate pairs |
| `min/mean/median/p95/p99/max_candidates` | Distribution of candidates per S1 |
| `reduction_ratio` | `total_pairs / (|S1| × |S2∪S3|)` when `corpus_size` provided |
| `s2_true_edge_recall` | Source-2-only true-edge recall |
| `s3_true_edge_recall` | Source-3-only true-edge recall |

> ⚠️ True-empty S1 entities are excluded from `complete_s1_recall` (no links to miss).

## Oracle Metric

**Oracle** = predicting exactly T ∩ C per S1. It is the theoretical ceiling for any
model using candidate set C. The oracle F0.5 must always be ≥ the model's actual F0.5.

```python
from ber.metrics import oracle_sanity_check
passed, msg = oracle_sanity_check(eval_result, cand_result)
if not passed:
    raise RuntimeError(msg)  # Evaluation bug — investigate immediately
```

## Error Categories

`error_analysis(truth, predictions, candidates)` classifies errors:

| Code | Category | Description |
|------|----------|-------------|
| A | **Retrieval miss** | True ID absent from candidate set C |
| B | **Scoring false negative** | True ID in C but model did not predict it |
| C | **False positive** | Predicted ID not in truth T |
| D | **Wrong singleton** | &#124;T&#124;=1, &#124;P&#124;=1, but wrong ID predicted |
| E | **Empty-case false positive** | T=∅ but P≠∅ |
| F | **Multi-match partial miss** | &#124;T&#124;>1, at least one true ID not predicted |

- **A → Sabeena** (retrieval needs improvement)
- **B → Harshit** (model/threshold needs improvement)

## Output Schema

### `matching_results.tsv` (scored on leaderboard)

```
source1_entity_id<TAB>matched_entity_ids
S1-001<TAB>S2-123,S3-456
S1-002<TAB>
S1-003<TAB>S2-789
```

### `candidate_pairs.tsv` (required in submission zip)

```
source1_entity_id<TAB>candidate_entity_ids
S1-001<TAB>S2-123,S3-456,S2-999
S1-002<TAB>
S1-003<TAB>S2-789,S3-111
```

**Rules enforced by `write_outputs()` and `validate_outputs()`:**
- UTF-8 encoding
- Tab-separated (`\t`) columns; IDs comma-separated within second column
- Exactly one row per test S1 (including empty rows)
- No duplicate IDs within a comma-separated list
- No duplicate S1 rows
- Valid `S2-` or `S3-` prefixes only
- Final predictions ⊆ final candidates (enforced; raises `ValueError` if violated)
- Deterministic ordering: S1 IDs sorted; IDs within lists sorted

## Validation Rules (Preflight)

`validate_outputs(test_s1_path, matching_results_path, candidate_pairs_path)`:

1. Exact headers (`source1_entity_id`, `matched_entity_ids` / `candidate_entity_ids`)
2. UTF-8 encoding
3. Tab delimiter (not comma)
4. Exactly one row per test S1
5. No duplicate S1 rows
6. No missing S1 rows
7. No unexpected S1 rows
8. Valid `S2-` / `S3-` ID prefixes
9. No duplicate IDs within comma-separated lists
10. Predictions ⊆ candidates (when candidate file provided)
11. Optional: ID existence check against S2/S3 corpus (memory-heavy, off by default)
12. French S1 coverage (coverage only — **no F0.5 for France**)

Returns `ValidationResult` with structured errors, not just True/False.

## Organizer Validator

**Command (run from `student_resource/`):**
```
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Optional ID existence check (memory-heavy):**
```
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

**Via BER CLI:**
```
ber validate \
    --test-s1 student_resource/dataset/test/test_source1.tsv \
    --matching output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv \
    --run-organizer
```

> ⚠️ **A PASS here means FORMAT/COVERAGE/CONSISTENCY PASS only.**
> It does NOT mean ML quality PASS. Never claim a score based only on the organizer validator.

## CLI Commands (Suresh)

```bash
# Compute macro F0.5
ber evaluate \
    --truth data/train_ground_truth.tsv \
    --predictions output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv

# Candidate diagnostics
ber candidates \
    --truth data/train_ground_truth.tsv \
    --candidates output/candidate_pairs.tsv \
    --corpus-size 9969589

# Preflight validation
ber validate \
    --test-s1 student_resource/dataset/test/test_source1.tsv \
    --matching output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv \
    --run-organizer

# Error analysis
ber error-analysis \
    --truth data/train_ground_truth.tsv \
    --predictions output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv \
    --verbose
```

## Synthetic Test Command

```bash
cd code/business_entity_resolution
python -m pytest tests/ -v --tb=short
```

**74 tests, all PASS. Runtime: ~1.1s.**

## France (Unseen Country)

France appears in test data but **not** in training labels.

**DO:**
- Verify all French S1 IDs receive output rows (checked by `france_coverage_check()`)
- Report candidate counts and coverage diagnostics
- Process Unicode/non-Latin business names without modification

**DO NOT:**
- Invent a French F0.5 score
- Use country whitelist that excludes France
- Claim any French prediction quality

## Memory Considerations

| Operation | Memory Strategy |
|-----------|----------------|
| `evaluate()` | O(|S1| IDs) — truth index only |
| `evaluate_candidates()` | O(|S1|) — counts list ~14 MB for 1.7M S1 |
| `error_analysis()` | O(|S1| IDs) — no business name strings |
| `write_outputs()` | O(|S1|) — sorted ID list, row-by-row write |
| `validate_outputs()` | O(|S1| IDs) — streaming read |

The 8 GB machine constraint is respected. No 10M-record corpus is loaded into RAM.

## Limitations

- `evaluate()` loads the full truth index into memory (S1 IDs only, ~100 MB for 1.7M entities).
- Candidate count distribution uses an exact sorted list. On 1.7M S1, this is ~14 MB (acceptable).
  For stricter memory requirements, switch to a t-digest approximation and document.
- Organizer validator `--check-ids` requires loading all S2/S3 IDs (~few GB); use without `--candidate` if memory is tight.
- No full-scale run has been measured yet. This documentation covers the implementation only.
  Harshit must run the complete pipeline and record actual metrics.

## Handoff to Harshit

### Evaluation commands
```bash
ber evaluate --truth <truth_tsv> --predictions <matching_tsv> [--candidates <candidates_tsv>]
ber candidates --truth <truth_tsv> --candidates <candidates_tsv> [--corpus-size <N>]
ber error-analysis --truth <truth_tsv> --predictions <matching_tsv> --candidates <candidates_tsv> --verbose
ber validate --test-s1 <test_source1.tsv> --matching <matching_tsv> --candidates <candidates_tsv> --run-organizer
```

### Report schema
All results are structured dataclasses (`EvaluationResult`, `CandidateResult`, `ErrorReport`) — JSON-serializable if needed.

### Output checksum
```bash
Get-FileHash output\matching_results.tsv -Algorithm SHA256
Get-FileHash output\candidate_pairs.tsv -Algorithm SHA256
```

### Preflight checklist (Suresh's gate)
- [ ] `validate_outputs()` returns `passed=True`
- [ ] Organizer validator returns exit code 0 (format PASS)
- [ ] Oracle sanity check PASSED (model ≤ oracle)
- [ ] All 1,732,544 test S1 rows have output rows
- [ ] All 259,452 French S1 rows have output rows
- [ ] No duplicate IDs in any row
- [ ] Predictions ⊆ candidates for all S1
