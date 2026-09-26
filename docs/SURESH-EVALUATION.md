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

> ⚠️ **A PASS here means FORMAT/COVERAGE/CONSISTENCY PASS only.**
> It does NOT mean ML quality PASS. Never claim a score based only on the organizer validator.

## Integration status

The functions in `ber.metrics` and `ber.output` are importable Python APIs. The shared `ber.cli` command names and path flags remain H0 stubs; this PR does not add runnable evaluation or error-analysis CLI commands. Harshit's later integration work will connect these APIs to the shared pipeline.

## Synthetic Test Command

```bash
cd code/business_entity_resolution
python -m pytest -q
```

**118 tests passed** in the reviewed PR checkout. This is a synthetic test result, not a full challenge run.

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
| `write_outputs()` | O(|S1|) — input-order ID list plus streamed candidate/decision rows |
| `validate_outputs()` | O(|S1| IDs) — streaming read |

The implementation avoids loading the 10M-record candidate corpus into memory. A full-scale run on 1,732,544 test S1 entities was measured on 26 September 2026.

## Measured Full-Scale Benchmark Evidence (1.73M Test S1 Entities)

The full-scale benchmark script is available at `code/business_entity_resolution/src/tests/suresh_memory_benchmark.py`:

```bash
cd code/business_entity_resolution
python src/tests/suresh_memory_benchmark.py
```

### Benchmark Methodology & Setup
1. **Stage 1 & 2 (Writer and Preflight Throughput):** Streams the full 1,732,544 test S1 IDs with empty match and candidate sets (`yield sid, ()`). This isolates and measures file I/O throughput, TSV delimiter formatting, deterministic sorting, UTF-8 compliance, and preflight rule checking on 1.73M rows without requiring precomputed model inference.
2. **Stage 3 (Organizer Submission Validator):** Runs the official organizer `utils/validate_submission.py`. As noted in its script (line 9), it checks submission file formatting, delimiters, headers, row count, non-empty IDs, and prediction-subset-of-candidates constraints; it has no ground truth and never computes a match quality score. If organizer test sources are absent (or running in synthetic mode), the benchmark records `SKIPPED (format not verified)` and exits 0 for CI, unless `--require-organizer` is specified.
3. **Stage 4 (Candidate and Metric Scalability):** Streams 1,732,544 synthetic entities through `evaluate_candidates()` and `evaluate()`. Because the challenge test set contains no ground truth labels, metrics such as oracle $F_{0.5} = 1.000$ and singleton accuracy = $1.000$ are strictly synthetic stress tests to verify algorithmic scaling, in-memory index size, candidate count distributions, and garbage collection under full test volume. They do not represent competition leaderboard scores.
4. **Memory Measurement & Scope:**
   - Peak process memory is tracked per stage using `psutil.Process().memory_info().peak_wset` (on Windows) or `resource.getrusage().ru_maxrss * 1024` (on POSIX), along with stage-end RSS.
   - Python heap memory is tracked per stage via `tracemalloc.get_traced_memory()`.
   - The benchmark's 8 GB budget pass condition tests `max_peak_process_bytes < 8 GiB`. If process peak cannot be measured, the benchmark explicitly reports UNKNOWN and exits non-zero rather than silently passing.
   - **Scope Qualification:** The measured peak process memory applies strictly to Suresh's output writer and evaluation components under 1.73M volume. It does *not* cover memory consumption for upstream candidate retrieval (Sabeena) or model feature extraction/inference (Harshit).
   - **Historical note:** The approximate per-stage numbers below (200 MB – 1.63 GB) were recorded from code that used post-stage RSS on Linux, not true peak RSS. A rerun with the corrected `get_process_peak_bytes()` (which uses `ru_maxrss` on POSIX) is needed to produce verified peak measurements.

### Historical Performance on 1,732,544 S1 Entities (Stage-End RSS Estimates)

> **⚠️ These values are historical stage-end RSS approximations**, not verified peak process memory.
> The code that produced these numbers used `psutil.memory_info().rss` (current, not peak) on Linux.
> A rerun with the corrected peak-tracking code (`resource.getrusage().ru_maxrss` on POSIX) is required
> to produce verified peak measurements.

| Pipeline Stage | Evaluated Entities | Runtime | Peak Python Memory (traced) | Process Memory (est. stage-end RSS) | Status / Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `write_outputs()` | 1,732,544 test S1 | 100.65 s | 169.81 MB | ~200 MB (est.) | **PASS** (1.73M rows written) |
| `validate_outputs()` | 1,732,544 test S1 | 80.79 s | 817.60 MB | ~950 MB (est.) | **PASS** (0 errors, 259,452 France rows verified) |
| Organizer `validate_submission.py` | 1,732,544 test S1 | 15.20 s | < 100 MB | ~150 MB (est.) | **PASS** (format check only, exit code 0) |
| `evaluate_candidates()` | 1,732,544 synthetic S1 | 63.33 s | 858.17 MB | ~1.10 GB (est.) | **PASS** (synthetic oracle F0.5 = 1.000) |
| `evaluate()` | 1,732,544 synthetic S1 | 155.23 s | 1,394.32 MB | ~1.63 GB (est.) | **PASS** (synthetic singleton acc = 1.000) |

> ⚠️ **Summary Disclaimers:**
> 1. **Organizer Validator Checks Format, Not Quality:** Line 9 of `utils/validate_submission.py` states it has no ground truth and never computes a match score. Exit code 0 confirms format and row coverage only.
> 2. **Evaluation Metrics from Synthetic Stream Only:** The challenge test set has no ground truth labels. Stage 4 metrics are synthetic scaling benchmarks, not model performance.
> 3. **Memory Scope:** The stage-end RSS estimates (~1.63 GB largest) reflect Suresh's writer and evaluator components only, not the full candidate retrieval and model pipeline. True peak may differ from stage-end RSS.

**8 GB Machine Target:** Historical estimated stage-end RSS across all stages of the writer/evaluator benchmark is **~1.63 GB** (1.39 GB Python traced). A verified peak measurement from the corrected code is pending.


## Continuous Integration (CI) Workflow

The dataset-free CI workflow is implemented in `.github/workflows/pr-checks.yml`:
- Triggered on PRs and pushes to `main` and feature branches.
- Runs on standard GitHub Actions Ubuntu runners.
- Executes `git diff --check`, packages installation, full synthetic pytest suite (121+ tests), and synthetic smoke runs without requiring external datasets or secrets.

## Limitations

- `evaluate()` loads the full truth index into memory (S1 IDs only, ~100 MB for 1.7M entities).
- Candidate count distribution uses an exact sorted list. On 1.7M S1, this is ~14 MB (acceptable).
- Optional organizer ID-existence checking (`--check-ids`) loads all S2/S3 IDs (~few GB); off by default for memory-constrained environments.

## Handoff to Harshit

### Python APIs awaiting pipeline integration

`ber.metrics.evaluate`, `ber.metrics.evaluate_candidates`, and `ber.metrics.error_analysis` provide score and error reports. `ber.output.write_outputs`, `ber.output.validate_outputs`, and `ber.output.run_organizer_validator` provide output and validation behavior. Call them through their documented signatures until the shared CLI integration PR lands.

### Report schema
All results are structured dataclasses (`EvaluationResult`, `CandidateResult`, `ErrorReport`) — JSON-serializable if needed.

### Milestone R4 Verification (Delivered by Suresh)
- [x] Streaming output writer validated at 1.73M scale (`write_outputs()` streams all 1,732,544 S1 rows without OOM)
- [x] Preflight validator (`validate_outputs()`) verified on 1.73M rows (100% S1 row coverage, French S1 coverage, duplicate detection, UTF-8 TSV compliance)
- [x] Organizer validator integration (`run_organizer_validator`) verified (reports exact output and exit code; explicitly flagged as format check only)
- [x] Candidate & metric streaming evaluators (`evaluate_candidates()`, `evaluate()`) verified at full 1.73M volume with synthetic stream
- [x] Per-stage process memory tracking implemented with corrected peak measurement (`ru_maxrss` on POSIX, `peak_wset` on Windows); budget failure and unavailable measurement both exit non-zero
- [ ] Verified peak process memory from corrected code on a full 1.73M rerun (historical ~1.63 GB was stage-end RSS, not verified peak)
- [x] Dataset-free CI workflow active (`.github/workflows/pr-checks.yml`) and passing (121 tests)

### Final Release Gate (Pending — To be executed with Harshit upon real model inference)
- [ ] Candidate retrieval outputs populated (Sabeena's candidate sets, non-empty)
- [ ] Pair model predictions populated (Harshit's model decisions, non-empty)
- [ ] `matching_results.tsv` and `candidate_pairs.tsv` generated from same inference run with real model predictions
- [ ] `validate_outputs()` returns `passed=True` on actual model predictions and candidate pairs
- [ ] All 1,732,544 test S1 rows populated in final `matching_results.tsv` and `candidate_pairs.tsv`
- [ ] All 259,452 French S1 rows populated with valid candidates and predictions
- [ ] Final predictions ⊆ candidates verified for every test S1 row on real pipeline output
- [ ] Organizer validator `validate_submission.py` executed against unzipped `dataset/test` with `--require-organizer` and returns exit code 0 (`PASS`)
- [ ] Oracle sanity check verified on real validation split (`model_f0_5 <= oracle_f0_5`)
- [ ] Submission zip assembled per organizer specification with code, documentation, and dependencies
- [ ] End-to-end inference + writing pipeline executes within 8 GB RAM and 4-hour runtime budget
