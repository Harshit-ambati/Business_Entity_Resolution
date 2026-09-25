# Suresh: exact evaluation, diagnostics, submission correctness

**Machine:** 8 GB RAM. **Starting branch:** `feat/evaluation-output`. **Primary reviewer:** Harshit. **Contract reviewers:** Thulasi for truth/source parsing; Sabeena for final candidate semantics. **Authoritative shared interface:** [CONTRACTS.md](CONTRACTS.md).

## Outcome you own

Make every model comparison trustworthy and make invalid submissions impossible to overlook. You implement the challenge's per-S1 macro F0.5, blocking diagnostics, clear error reports, and deterministic writing/checking of both TSVs. Harshit owns choosing the final model, assembling the zip, and uploading; you provide the release gate and evidence.

## Files and output

| File/artifact | You deliver |
| --- | --- |
| `code/business_entity_resolution/src/ber/metrics.py` | `evaluate`, `evaluate_candidates`, score/slice reports |
| `code/business_entity_resolution/src/ber/output.py` | `write_outputs`, ID/subset/order checks, output manifest/checksums |
| `code/business_entity_resolution/src/tests/test_metrics.py` | Hand-calculated metric and candidate-ceiling cases |
| `code/business_entity_resolution/src/tests/test_output.py` | Exact headers, empty rows, duplicates, invalid IDs, France coverage |
| `code/business_entity_resolution/src/ber/` report helper | Reproducible false-positive/false-negative/singleton summaries |
| `.github/workflows/pr-checks.yml` | Fast fixture tests once M0 test command exists; no dataset/secrets required |
| `artifacts/evaluation/` (ignored) | Holdout score and error reports; final preflight evidence |

Do not implement retrieval routes or train the classifier. Do not silently repair invalid output in the evaluator: expose a clear error so the owner fixes the source. Full metric/output checks should use streaming, sorted joins, or disk-backed state rather than retaining all pair scores on an 8 GB laptop.

## PR R1: score exactly what organizers score

**Input:** truth rows and one decision per S1. **Deliverable:** `evaluate` from [CONTRACTS.md](CONTRACTS.md), including a per-S1 F0.5 function and aggregate report. For nonempty truth `T` and prediction `P`, precision is `|T∩P|/|P|`, recall is `|T∩P|/|T|`, and F0.5 is `1.25*precision*recall/(0.25*precision+recall)`. Empty/empty scores 1; exactly one empty scores 0. Average over **all** S1 rows. Reject missing/duplicate/unexpected S1 rows and duplicate predicted IDs; otherwise singleton mistakes may be hidden.

**Acceptance:** hand-calculated tests include one true singleton predicted empty, one true singleton falsely linked, one matched S1 predicted empty, a perfect multi-link case, and the organizer's 2-of-3 example (approximately 0.714). A three-S1 fixture verifies macro averaging differs from pooled pair-level F0.5. Reports include total S1 count, macro F0.5, singleton count/accuracy, and optional US/India slices. France has no labeled test truth and must never be reported as a measured accuracy.

## PR R2: blocking ceiling and error attribution

**Input:** truth rows and Sabeena's **final** `CandidateGroup` for each held-out S1. **Deliverable:** `evaluate_candidates` with true-edge recall; recall separately for S2/S3; fraction of matched S1 with every true link present; number/mean/p50/p95/p99/max candidates per S1; reduction ratio against all possible S1 x (S2+S3) pairs; and oracle macro F0.5 from `truth ∩ candidates`. Include all true singleton rows in the oracle average. A diagnostic report separates (a) true IDs absent from candidates, (b) true IDs retrieved but rejected by Harshit's decision, (c) false emitted IDs, and (d) wrongly nonempty singleton rows.

**Acceptance:** synthetic fixtures prove each category. Every final matched ID must appear in the corresponding final candidate group. Oracle macro F0.5 is at least the actual score for any decisions restricted to those groups; a violation means a bug in joining/scoring and fails CI. Route contribution reports, if requested, use Sabeena's route names without inventing new definitions.

## PR R3: exact output writer and preflight

**Input:** input-order test S1 iterator, aligned final candidate groups, aligned decisions, and valid test S2/S3 IDs. **Deliverable:** `write_outputs` produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` with the exact two headers in the problem statement, UTF-8, single tab separator, one row per S1, empty second cell for zero IDs, comma-separated IDs without quoting or duplicates. Preserve test S1 input order. Check valid ID prefixes and membership in the **test** S2/S3 files, not training files. Ensure final matches are subsets of that S1's exact scored candidate set. Write a manifest with row count, empty count, per-country S1 coverage, output byte size, and SHA-256 checksums.

**Acceptance:** fixture writer and preflight reject missing/duplicate S1, wrong source ID, ID absent from test sources, duplicate list ID, match absent from candidates, extra columns, and unexpected header. They preserve an S1 row with no candidates and an S1 row with candidates but no final matches. The official organizer `utils/validate_submission.py` prints `PASS` on fixture outputs using fixture test data and later on full outputs. Final full checks expect exactly 1,732,544 S1 rows, including all 259,452 French S1 rows in the supplied archive.

## PR R4: report helper and fast CI

**Input:** predictions, candidate groups, truth, scores/features, and M0 test command. **Deliverable:** bounded error samples with IDs, score, route, key feature values, and failure category; an aggregate comparison table matching [VERIFICATION.md](VERIFICATION.md); and a GitHub Action that runs fixture tests without downloading challenge data or requiring secrets. When records are included in a local error report, keep that report ignored; do not post full challenge rows in GitHub PRs.

**Acceptance:** the action passes on a clean clone with only repository files. A full holdout report can be generated within 8 GB using bounded state or disk-backed joins. Report score and candidate numbers with split, commit, configuration, and denominators. Do not label a sample or fixture score as full holdout.

## Release handoff to Harshit

Provide the exact evaluator and writer commands, report schema, failing-row error format, organizer-validator command/result, and final output checksums. Supply methodology numbers: macro F0.5, singleton accuracy, candidate recall/ceiling, candidate volume, and representative error categories. Say whether full output validation was actually run; a format-valid sample is not proof the 1.73 million-row file is valid. Keep a known-good checksum and do not overwrite it before Harshit records the portal upload.
