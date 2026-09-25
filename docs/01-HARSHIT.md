# Harshit: ML lead, integration owner, release owner

**Machine:** 16 GB RAM. **Starting branch:** `feat/contracts-and-pipeline`. **Reviewers:** Sabeena for model/integration; Thulasi and Suresh for their affected interfaces. **Authoritative shared interface:** [CONTRACTS.md](CONTRACTS.md). This assignment describes work to be built; none of its model results exists yet.

## Outcome you own

Deliver a reproducible program that learns from training labels, scores Sabeena's final candidates, selects zero or multiple links per S1, reports Suresh's exact macro F0.5 on a fixed holdout, and produces the official test outputs. You decide which *measured* model/config becomes the final submission. Own the complete final zip, methodology, and portal upload log. This is the most complex workstream; do not shift full model training or final release decisions to the 8 GB owners.

## Files and boundaries

| File | You deliver |
| --- | --- |
| `code/business_entity_resolution/src/ber/contracts.py` | M0 shared types agreed with all owners |
| `code/business_entity_resolution/src/ber/features.py` | Fixed-order, train/test-identical pair features |
| `code/business_entity_resolution/src/ber/model.py` | Hard-negative training, model manifest, batched scoring |
| `code/business_entity_resolution/src/ber/decision.py` | Threshold policy and zero/many final links |
| `code/business_entity_resolution/src/ber/cli.py` | Index/train/evaluate/predict/validate orchestration |
| `code/business_entity_resolution/README.md` | Exact fresh-data reproduction commands and expected paths |
| `code/business_entity_resolution/requirements.txt` | Pinned, license-reviewed runtime dependencies |
| `code/business_entity_resolution/src/tests/` | Contract, feature, decision, and end-to-end synthetic tests |
| `artifacts/` (ignored) | Split/config/model/report/output manifests and checksums |

You may update root planning files via PR. Do not implement another owner's module to bypass its contract; request a small interface change or agree on a short integration fix in the PR.

## PR H0: freeze interfaces and make the skeleton importable

**Input:** this plan and organizer file rules. **Deliverables:** package directories, `contracts.py`, tiny synthetic TSV fixture, test runner configuration, CLI entry point with documented subcommands/flags, and an example config. The CLI may report an unimplemented stage clearly; it must not pretend to generate a submission. Select a fixed S1-level validation split algorithm and seed, save the rule in the package README, and make the split reproducible without embedding millions of IDs in Git. Record CPU, available RAM, and free disk on the machine selected for full runs; choose a work directory with enough space for indexes, model, intermediate candidates, and final outputs.

**Acceptance:** Thulasi can import `Record`/`NormalizedRecord`; Sabeena can import `CandidateGroup` and see index signatures; Suresh can import `Decision`; one small synthetic fixture can be parsed and passed through interface stubs. `python -m pytest` (or the documented equivalent) runs on an 8 GB machine. Obtain reviews from every owner whose interface is frozen. Merge this before dependent PRs change public signatures.

## PR H1: feature and training baseline

**Input:** Thulasi's normalized records, Sabeena's candidate groups, training truth, the fixed holdout. **Deliverables:** feature schema and names; a simple interpretable rule baseline; one license-compliant trained pair classifier; deterministic negative sampling from *retrieved, plausible* nonmatches; model save/load and manifest. At minimum test name character/token similarity, address character/token similarity, number/component agreement or conflict, missingness, source, and retrieval-route evidence. Treat a number conflict as evidence, not an automatic veto: the labeled sample contains valid pairs with changed numbers. Keep raw and normalized fields available for error inspection.

**Rules:** fit only on training S1 labels, not held-out labels; use IDs for joins only; use the same feature order and missing-value treatment at training and inference; record package/model license and exact pinned version; score in batches instead of materializing the full candidate matrix in RAM. If some positives are absent from candidates, report that retrieval loss rather than secretly adding them only to training.

**Acceptance:** tests cover exact, reordered, abbreviated, cross-script, missing-address, hard-negative, and unseen-country examples; model reload reproduces scores on a fixture; a training run records seed, split, candidate config, feature list, label counts, runtime, peak RAM, and artifact checksum. Report rule-baseline and classifier macro F0.5 on the *same* held-out S1 IDs. Do not claim a gain from pair accuracy alone.

## PR H2: decision logic and complete holdout

**Input:** batched pair scores and Suresh's evaluator. **Deliverables:** `decide_group` that may emit an empty list or multiple S2/S3 IDs, threshold sweep against per-S1 macro F0.5, diagnostics for false merges/missed links/singletons, and full validation orchestration. Compare at least one global threshold to optional source-specific thresholds; keep extra complexity only if it helps the fixed holdout. Never force one-to-one matching or top-1 output merely because that is convenient.

**Acceptance:** scorer output order matches candidate order; every emitted ID belongs to that S1's final candidate group; macro score includes empty truth rows; report S2/S3 true-link recall, US/India macro scores, singleton accuracy, candidate oracle ceiling, runtime, peak RAM, and representative errors. A full run on the training holdout completes within the available laptop resources. France test coverage is checked, but no French accuracy is invented.

## PR H3: final inference and submission

**Input:** selected frozen model/config, original test TSVs, Suresh's writer and organizer validator. **Deliverables:** full test inference; both TSVs from the same final candidate stream; package README, pinned requirements, filled `Documentation_template.md`, guideline's concise approach summary if requested, final zip, and a submission log with date/time, commit, model/config, TSV checksum, validator result, and portal score/status. Keep a last-known-good output until the challenge closes.

**Acceptance:** `utils/validate_submission.py` prints `PASS`; both outputs have exactly 1,732,544 S1 rows including 259,452 French rows; final IDs are valid subsets of candidates; a fresh reproduction command chain is documented and tested or its full-run execution log is attached; zip paths match organizer requirements; model license and artifact versions are recorded. Only the validated TSV is uploaded to the leaderboard. Do not spend a portal submission on an unvalidated file.

## Decisions you must record rather than assume

Choose and document the candidate cap with Sabeena, model family and license, negative sampling policy, feature set, threshold policy, batch sizes, training/validation split, and final experiment selection. Each selection needs the same-holdout comparison, measured runtime, and a reason. If a contract field is missing, change [CONTRACTS.md](CONTRACTS.md) in a reviewed PR before teammates rely on it.

## Handoff and review behavior

At each integration checkpoint, review open PRs promptly and give a failing command or row-level synthetic reproduction rather than a vague bug report. Ask Sabeena for candidate recall and exact postfilter candidates; Thulasi for normalization version; Suresh for scorer/output checks. Your PR description must state what runs were actually executed, on which split, and what remains unverified. Another teammate must approve your PR; you cannot self-approve.
