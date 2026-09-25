# Suresh | evaluation, diagnostics, output correctness

**Machine:** 8 GB RAM. **Workstream:** exact competition metric, error reporting, output writer, and final validation support. **Starting branch:** `feat/evaluation-output`.

## Mission

Make every improvement measurable and every submission format-safe. Your tools should stream or batch data so a full report can run on 8 GB, while small synthetic tests prove the special cases in the challenge metric.

## Owned files

- `code/business_entity_resolution/src/ber/metrics.py`: per-S1 macro F0.5, candidate recall/ceiling, slice reports.
- `code/business_entity_resolution/src/ber/output.py`: deterministic final TSV writer and preflight checks.
- Tests for scoring, candidate subset, duplicates, missing IDs, singleton rows, country coverage, and output headers.
- Small error-analysis report and final validation checklist; CI test workflow after package/test commands exist.

## PR milestones

1. **Evaluator:** implement the exact per-entity F0.5 formula and special empty/empty case. Compare against hand-calculated fixtures including no match, multiple matches, false merges, and missed links. Report macro score across all S1 rows, not just matched rows.
2. **Candidate diagnostics:** compute true-edge recall, complete-match coverage per S1, oracle macro ceiling, reduction ratio, candidate count distribution, and S2/S3 slices. Distinguish misses caused by retrieval from misses caused by model decisions.
3. **Output/release:** write exact headers and exactly one row per test S1, including empty lists and every French S1. Verify IDs exist, lists have no duplicates, final matches are subsets of candidates, and output is UTF-8 TSV. Run the organizer validator on both files and record `PASS`.
4. **Error report:** produce compact false-positive, false-negative, and singleton examples with scores/features for Harshit and Sabeena to investigate. Provide numbers and representative cases for the methodology template.

## Performance rules

- Use a small fixture for development. For full runs, read sorted/partitioned inputs or batches rather than holding all pair features/predictions in memory.
- Metrics use labeled US/India train holdout. France in test has coverage and output checks only; never assign it a validation accuracy.
- Avoid using portal score as a replacement for local tests. The official validator checks formatting, not prediction quality.

## Acceptance and handoff

- Tests demonstrate correct macro score for true-empty/pred-empty = 1, true-empty/pred-nonempty = 0, and the problem statement's multi-link example.
- Candidate oracle score is always at least the score of a correctly evaluated model restricted to those candidates; investigate any violation.
- A sample output and later a full test output pass both your preflight checks and `utils/validate_submission.py`.
- Harshit receives exact evaluation commands, report schema, error cases, output checksum procedure, and final-package checklist.

## Review partners

Harshit reviews evaluator/output PRs; Thulasi checks source and truth parsing assumptions; Sabeena checks blocking metrics and candidate-file semantics.
