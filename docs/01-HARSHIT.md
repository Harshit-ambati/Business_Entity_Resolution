# Harshit | model, decisions, integration, release

**Machine:** 16 GB RAM. **Workstream:** the most complex work: pairwise ML, threshold optimization, cross-module integration, full-scale inference, and final submission. **Starting branch:** `feat/contracts-and-pipeline`.

## Mission

Turn retrieved candidates into accurate zero/many matches under macro F0.5, while keeping a reproducible full pipeline that runs on the challenge data. You own final technical choices, but changes to another owner's interface require review.

## Owned files

- `code/business_entity_resolution/src/ber/features.py`: fixed-order pair features and missing-value behavior.
- `code/business_entity_resolution/src/ber/model.py`: training, artifact persistence, and batch scoring.
- `code/business_entity_resolution/src/ber/decision.py`: score-to-link logic, singleton handling, threshold configuration.
- `code/business_entity_resolution/src/ber/cli.py`: prepare/train/evaluate/predict command integration.
- `code/business_entity_resolution/README.md` and `requirements.txt`: exact reproduction and pinned dependencies.
- Root `PLAN.md`, release documentation, final zip assembly, experiment/submission log.

## PR milestones

1. **M0 contracts/skeleton:** agree on `Record`, `NormalizedRecord`, `Candidate`, feature vector, output writer, CLI, and tiny synthetic fixture. Set up a minimal package/test command so teammates can import interfaces. This PR unblocks everyone.
2. **Baseline model:** implement numeric features for name/address similarity, token overlap, address numbers, missingness, source, and retrieval-route evidence. Train with all known positives among retrieved candidates and hard negatives returned by the same blocking policy. Never include entity-ID number or validation labels as features.
3. **Decision/evaluation integration:** obtain pair scores in batches, permit multiple links, and sweep threshold(s) on the fixed held-out S1 split using the official macro F0.5. Compare to an interpretable rule baseline. Investigate false merges and singletons before increasing model complexity.
4. **Final pipeline:** retrain only after the approach is selected; run complete test inference, generate both outputs from one run, verify package reproducibility and model license, run organizer validator, freeze best commit and checksum, submit.

## Technical decisions to test

- Start with a license-compliant gradient-boosted tabular classifier; verify its exact license/version before finalizing.
- Compare name-only, address-only, and combined features so gains are measurable. Include disagreement features as soft evidence: real labeled pairs can have changed address numbers or cross-script names.
- Select negatives from plausible near neighbors, not mainly random pairs. Keep a separate untouched validation S1 set for model selection and thresholding.
- Evaluate global and optional source-specific thresholds. Do not enforce one match per S1 or a global one-to-one constraint unless holdout data proves it helps.
- Check score calibration and rank margins only if they improve held-out macro F0.5 and do not suppress valid multiple links.
- Ensure France flows through the same pipeline despite having no training labels; do not claim French accuracy from public or synthetic labels.

## Acceptance and handoff

- A fresh command chain can rebuild indexes/artifacts and both TSV outputs from the original train/test TSVs, within recorded hardware limits.
- Holdout report includes actual macro F0.5, candidate oracle ceiling, singleton behavior, per-source/country slices, and representative errors.
- Final model/feature order/config/seed/commit are saved; requirements are pinned and model license is checked.
- Both outputs have exactly one row per test S1, match IDs are subsets of candidate IDs, and official validator reports `PASS`.
- The zip contains the prescribed paths and filled methodology template. All portal uploads have recorded checksums and corresponding commits.

## Review partners

Sabeena reviews model and integration PRs; Thulasi reviews normalization-dependent feature assumptions; Suresh reviews metric/output and release claims. Review their PRs promptly so 8 GB workstreams are not blocked on full-scale runs.
