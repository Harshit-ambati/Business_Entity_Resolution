# Working rules for Business Entity Resolution

Read `README.md`, `PLAN.md`, `docs/CONTRACTS.md`, `docs/DECISION-REGISTER.md`, `docs/VERIFICATION.md`, `docs/TEAM-WORKFLOW.md`, and the relevant owner assignment before editing. This repository currently contains a plan; do not report proposed models, experiments, or scores as implemented or measured.

## Scope and ownership

Harshit owns model/decision/integration/release; Sabeena owns candidate retrieval; Thulasi owns data/normalization; Suresh owns metrics/output checks. See the four assignment files for exact modules and handoffs. Work on a branch, open small PRs to `main`, and keep contract changes reviewed by affected owners. Do not force-push shared branches or push directly to `main` after the bootstrap commit.

## Challenge constraints

- Use only provided challenge business records. No external identity lookup, geocoding, registration database, entity-resolution API, or internet data augmentation.
- Every test S1 ID needs exactly one output row. Match/candidate lists contain only valid S2/S3 IDs and no duplicates. Final matches are subsets of final candidates.
- Treat country as open-set text; test includes France although training does not. Preserve Unicode and non-Latin scripts.
- Compute per-S1 macro F0.5 including correct empty matches. Distinguish retrieval ceiling from scorer performance.
- Keep data, generated indexes, model binaries, outputs, and secrets out of Git. Verify final model license and pinned versions.
- The official output validator checks format, not model quality. Never claim a score without the corresponding measured run.

## PR evidence

Include exact test/benchmark commands, results, config, memory/runtime where relevant, and handoff/limitations. At least one teammate other than the author reviews and approves each PR. Harshit integrates and releases; Sabeena reviews Harshit's PRs. Maintain a known-good commit and output for final submission.
