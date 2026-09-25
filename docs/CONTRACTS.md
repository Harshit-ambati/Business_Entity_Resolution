# Implementation contracts for the four workstreams

**Status:** normative planning contract. Harshit's first implementation PR (M0) must create these importable types/functions or amend this document with approval from every affected owner. No teammate should silently substitute a different API, file layout, output schema, or scoring definition. The organizer's materials override this document for competition rules.

## 1. Repository and runtime boundaries

- Python package root: `code/business_entity_resolution/src/ber/`. Imports use `ber.*` after installing the local package or setting `PYTHONPATH` as documented by M0.
- Tests: `code/business_entity_resolution/src/tests/`. Tiny synthetic fixtures may be committed there; challenge source rows must not be committed. Keeping tests under `src/` honors the final package's all-source-under-`src/` rule.
- Scripts used to reproduce the solution must be inside `code/business_entity_resolution/`; all source code shipped in the final zip goes under its `src/` directory.
- Original organizer `dataset/train/` and `dataset/test/` are supplied at runtime. Code must accept a data-root path, not require a developer's absolute Windows path.
- Working indexes, model files, intermediate pair tables, reports, and final outputs go under a configurable ignored work/output directory. They are never committed.
- The default execution target is a CPU laptop with 16 GB RAM for index/model/full inference and 8 GB RAM for data and evaluation development. No CUDA, cloud service, or external data service is assumed.

## 2. Input schema and identity rules

Read UTF-8 TSV with an explicit tab delimiter. Source headers are exactly `entity_id`, `business_name`, `business_address`, `country`; truth headers are exactly `source1_entity_id`, `matched_entity_ids`. Treat IDs as opaque strings except for validating `S1-`, `S2-`, `S3-` prefixes. Never use the numeric suffix as a model feature. `matched_entity_ids` is a comma-separated list inside the second TSV cell; an empty cell means zero matches. Empty address is valid. Preserve input S1 order in generated files. Country is an open-set string; `France` must work without adding a new code branch.

Data validation must reject a bad header, malformed row, wrong source prefix, duplicate entity ID within a source file, duplicate truth S1 ID, or duplicate ID within one truth list. Missing names/country, if encountered, must be reported and handled explicitly rather than silently dropping the S1 row. Because a global duplicate-ID set may exceed 8 GB, the implementation may use sorting or disk-backed checks for full-size validation.

## 3. Shared Python types and functions

M0 creates `ber/contracts.py` with immutable or effectively immutable types equivalent to the following. Field names and meaning are fixed; concrete dataclass details are Harshit's implementation choice in M0.

```python
Record(entity_id: str, business_name: str, business_address: str, country: str)
TruthRow(source1_entity_id: str, matched_entity_ids: tuple[str, ...])
NormalizedRecord(
    raw: Record,
    name_norm: str,
    address_norm: str,
    name_tokens: tuple[str, ...],
    address_tokens: tuple[str, ...],
    country_key: str,
)
Candidate(
    candidate_entity_id: str,
    retrieval_routes: tuple[str, ...],
    route_scores: dict[str, float],
)
CandidateGroup(source1_entity_id: str, candidates: tuple[Candidate, ...])
Decision(source1_entity_id: str, matched_entity_ids: tuple[str, ...])
```

`CandidateGroup` contains one S1's **final candidates that will actually be scored**. Do not write early-stage candidates to `candidate_pairs.tsv` and then silently filter them later. A group may be empty. Candidate order is deterministic: descending retrieval priority/score with `candidate_entity_id` as the final tie break. `retrieval_routes` is a sorted unique tuple. All route scores are finite numbers with route-specific meaning documented by Sabeena; the matcher may ignore a route score until calibrated.

`iter_candidates` emits groups in source1 input order. Harshit's pipeline may join each group with a second `read_source(source1_path, "S1-")` iterator by asserting equal S1 IDs at every step; a mismatch is an error, never a silent skip. Suresh's writer uses the same alignment rule. This allows streaming without requiring all S1 records in memory.

### Thulasi: `ber/data.py`, `ber/normalize.py`

```python
read_source(path, expected_prefix) -> Iterator[Record]
read_truth(path) -> Iterator[TruthRow]
normalize_record(record: Record) -> NormalizedRecord
```

Readers yield input order and close files. `normalize_record` is pure and deterministic. It preserves the raw record and never translates or deletes entire non-Latin scripts. `country_key` is normalized from the supplied country string without an allowlist. Tokens are ordered and deterministic; empty address produces empty address features. The normalization version is an exported constant and is recorded in model/index manifests. Any extra normalized fields require a reviewed contract change.

`expected_prefix` is exactly one of `"S1-"`, `"S2-"`, or `"S3-"` (including the dash). The caller supplies it from the file's role; the reader validates every yielded ID.

Normalization v1 is fixed for the initial PR: NFKC then `casefold`, replace `&` with the token `and`, turn other punctuation/separators into spaces while preserving Unicode letters/digits, collapse whitespace, and tokenize in order. Country uses NFKC/case folding/whitespace collapse only. Retain raw fields, digits, accents, and non-Latin scripts. Do not expand abbreviations, remove legal suffixes, or transliterate in v1. Export `NORMALIZATION_VERSION = "1"`; a changed output requires a new version and index/model rebuild. Thulasi may propose measured variants in a later PR.

### Sabeena: `ber/index.py`, `ber/blocking.py`

```python
build_index(source2_path, source3_path, work_dir, config) -> IndexManifest
open_index(manifest) -> IndexStore
iter_candidates(source1_path, index_store, config) -> Iterator[CandidateGroup]
index_store.get_record(candidate_entity_id) -> NormalizedRecord
```

`IndexManifest` stores paths, source checksums/sizes, normalization version, route configuration, and index version. `IndexStore` may use disk-backed storage; it must not require all raw records as Python objects. `iter_candidates` emits exactly one group per input S1 in input order, with only valid S2/S3 IDs from that split's source files. Training validation builds/queries an index from **training S2/S3**; test inference builds/queries a separate index from **test S2/S3**. No truth labels are inputs to candidate generation. `max_candidates_per_s1 = 32` is the **temporary baseline**, applied after route union; Sabeena compares 16/32/64 on the same holdout before Harshit freezes the final cap. The selected cap is always a named, logged config value.

### Harshit: `ber/features.py`, `ber/model.py`, `ber/decision.py`, `ber/cli.py`

```python
pair_features(s1: NormalizedRecord, candidate: NormalizedRecord,
              retrieval: Candidate) -> fixed-order numeric vector
train_model(training_pairs, validation_pairs, config) -> ModelManifest
score_group(s1, group, index_store, model_manifest) -> pair scores in candidate order
decide_group(source1_entity_id, candidates, pair_scores, config) -> Decision
```

The model manifest contains feature names/order, normalization/index versions, training config/seed, model type/license/version, and artifact path. Scores correspond one-to-one with the group's candidates. `decide_group` may return zero or multiple S2/S3 IDs; it cannot add an ID absent from the candidate group. Thresholds are selected on the fixed validation split using Suresh's evaluator. Exact CLI flags are frozen in M0 and documented in the package README; required commands are `index`, `train`, `evaluate`, `predict`, and `validate`.

### Suresh: `ber/metrics.py`, `ber/output.py`

```python
evaluate(truth_rows, decision_rows, s1_country_lookup=None) -> MetricsReport
evaluate_candidates(truth_rows, candidate_groups) -> CandidateReport
write_outputs(test_s1_rows, candidate_groups, decision_rows, output_dir) -> OutputManifest
```

Evaluation joins by S1 ID, includes **all** truth S1 IDs, and rejects duplicate/missing/unexpected IDs. It must not silently treat a missing prediction row as a correct singleton. Candidate reports include edge recall, matched-S1 complete-link coverage, candidate count distribution, reduction ratio, and oracle macro F0.5. The writer consumes aligned S1/candidate/decision streams or a bounded disk-backed equivalent; it validates every ID and subset relation, writes in test S1 input order, and records file checksums/row counts. It does not need to retain the full test output in RAM.

## 4. Exact output files

`output/matching_results.tsv` header: `source1_entity_id<TAB>matched_entity_ids`.

`output/candidate_pairs.tsv` header: `source1_entity_id<TAB>candidate_entity_ids`.

Both are UTF-8 TSV with one row for **every** test S1 ID, including France and empty lists. Each second cell is zero or more comma-separated IDs with no duplicates and no embedded quotes. All listed IDs exist in the corresponding test S2/S3 files. Final matches are a subset of the final candidate list for that S1. The organizer's `utils/validate_submission.py` must print `PASS` for release; it does not establish ML quality.

## 5. Metric and split contract

For each S1, let `T` be its true set and `P` its predicted set. If both are empty, score 1. If exactly one is empty, score 0. Otherwise compute precision and recall and `F0.5 = 1.25*Prc*Rec/(0.25*Prc+Rec)`. Average over **all** held-out S1 rows. Do not substitute pair-level, micro, or globally pooled F0.5. The blocking oracle predicts `T intersect C` from final candidate set `C` and must score at least as well as any model whose outputs are subsets of `C`.

The M0 split is fixed: compute `sha256(("2026|" + source1_entity_id).encode("utf-8"))`, interpret the first eight digest bytes as an unsigned big-endian integer, and assign the S1 row to validation when that integer modulo 10 equals 0; all other S1 rows are for model fitting. This gives a reproducible approximately 10% S1-level holdout. Harshit records resulting counts by country and singleton status and tests the split function. Validation labels are not available to model fitting; validation experiments and threshold selection are recorded explicitly and compared on the same fixed holdout. S2/S3 records can be indexed as unlabeled retrieval corpus for their respective split; test labels do not exist. A `France` test accuracy is unknowable locally.

## 6. Change control and unresolved choices

The contracts above and temporary starting values are fixed now. The following **final** choices remain experimental: enabled retrieval routes and candidate cap, address parser rules, model type, negative sampling ratio, threshold policy, batch sizes, and cache/index format. Each owner proposes these in their PR with measurements; Harshit records the accepted config. If a teammate needs a new field or behavior, they update this file and request affected-owner review before coding against it. When the organizer's materials are ambiguous, quote the exact conflict in the PR and select the interpretation that satisfies both where possible.
