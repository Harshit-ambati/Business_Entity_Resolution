# Shared contracts (freeze in M0)

This document defines the interfaces teammates implement against. Proposed module names become binding after the M0 contract PR is reviewed. A change to an interface requires a contract PR or an explicitly reviewed change in the same PR before another workstream depends on it.

## Inputs

All input files are UTF-8 TSV, read with `delimiter="\t"`. Source files have exactly these logical columns: `entity_id`, `business_name`, `business_address`, `country`. Ground truth has `source1_entity_id`, `matched_entity_ids`, where the latter is an empty string or comma-separated S2/S3 IDs. Empty address is valid; no country is assumed to be in a closed list. IDs are opaque strings: prefixes identify source, while numeric portions have no predictive meaning.

Suggested package: `code/business_entity_resolution/src/ber/`.

## Record and normalization interface (Thulasi)

`Record`: `entity_id: str`, `business_name: str`, `business_address: str`, `country: str`. A source reader yields records in stable input order and accepts a TSV path; it does not load the complete file into memory.

`normalize(record) -> NormalizedRecord` retains the original record and exposes `name_norm`, `address_norm`, `name_tokens`, `address_tokens`, `country_key`, and extracted optional address components. Missing components use an explicit empty/null representation. Normalization must be deterministic and Unicode-safe, with a version string saved alongside trained artifacts. `country_key` derives from the supplied label without a hard-coded US/India whitelist.

Ownership: `data.py`, `normalize.py`, their tests, and a data-quality summary command or function. Feature additions can request new fields through a contract change.

## Candidate interface (Sabeena)

`build_index(source2, source3, work_dir, config)` prepares only the provided corpus. `iter_candidates(source1, index, config)` yields each S1 ID with a deduplicated, bounded ordered list of `Candidate` records. Each candidate has `source1_entity_id`, `candidate_entity_id`, `retrieval_routes`, and optional route scores/ranks. IDs must exist in the respective input corpus. Output order and tie breaks are deterministic.

The candidate set passed to the model must be the set written to `candidate_pairs.tsv`; any cheap prefilter before inference happens *before* the file is produced. An empty list is valid. Blocking records its configuration, version, pair count, and resource measurements.

Ownership: `blocking.py`, `index.py` or equivalent, retrieval tests, and benchmark notes. Index file format is internal but must be reproducible from TSV inputs and bounded for 16 GB RAM.

## Feature/model/decision interface (Harshit)

`features(source1_record, candidate_record, retrieval_metadata) -> fixed-order numeric feature vector`. Features must be defined identically for train and test, including missing-value behavior. `train(...)` saves model plus feature order and versions. `score(candidate_pairs) -> pair scores`. `decide(scores, config) -> zero-or-more matched IDs per S1`. A shared evaluator chooses thresholds on validation labels using macro F0.5. The scorer may return an empty list and may retain multiple IDs from either source.

Ownership: `features.py`, `model.py`, `decision.py`, `cli.py`, end-to-end orchestration, model licensing review, and release selection. IDs are keys, not predictors.

## Evaluation/output interface (Suresh)

`evaluate(truth, predictions)` returns macro F0.5 and diagnostic counts, treating true-empty/predicted-empty as 1 and true-empty/predicted-nonempty as 0. It reports precision, recall, singleton accuracy, per-source and per-country slices where ground truth exists. `evaluate_candidates(truth, candidates)` returns true-edge recall, complete-S1 recall, candidate counts, reduction ratio, and oracle macro F0.5 ceiling.

`write_outputs(test_s1, candidates, decisions, output_dir)` writes two UTF-8 TSVs with exact headers and one row per test S1 ID, including empty rows. `matching_results.tsv` has `source1_entity_id<TAB>matched_entity_ids`; `candidate_pairs.tsv` has `source1_entity_id<TAB>candidate_entity_ids`. ID lists are comma-separated without duplicates. Final matches are subsets of candidates. It invokes or documents the organizer's `utils/validate_submission.py` as the release gate.

Ownership: `metrics.py`, `output.py`, tests, error-reporting tools, and submission checks. Suresh may prepare documentation evidence; Harshit owns the final artifact and portal submission.

## Command-line entry points (Harshit integrates)

The M0 PR will settle exact flags and paths. Required capabilities: prepare/index, train, evaluate, predict, and validate outputs from the original TSV folder. All commands must log seed, config, version/commit, elapsed time, and output path. A README in `code/business_entity_resolution/` must show the exact commands for a fresh end-to-end reproduction.

## Tests and changes

Each owner supplies small fixtures covering exact match, reordered or abbreviated name, cross-script name with address evidence, changed/missing address, hard negative, singleton, unseen `France` label, and multiple matches. Do not use real business records as external lookup queries. Keep test fixtures tiny and synthetic. Interface changes list affected owners in the PR and require their review before merge.
