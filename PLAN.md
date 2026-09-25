# 72-hour execution plan

## Objective and score

Maximize **held-out macro F0.5 per Source 1 entity**, then use limited public-leaderboard feedback without overfitting it. Every S1 entity must have one output row. A row may contain zero, one, or multiple S2/S3 IDs. Correctly empty rows earn credit; a false link on a true singleton earns zero for that entity. The public score is a subset; final ranking uses the private split described in the problem statement.

The portal permits at most five submissions per day. Record the exact commit, configuration, local score, output checksum, portal submission time, and leaderboard result for every upload. Never discard the best known-good output.

## Scope and constraints

- Train files: `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`.
- Test files: `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`; France is an unseen training country.
- Read and write explicit tab-delimited UTF-8 data. Preserve comma-delimited ID lists *inside* the second TSV column.
- Build from provided challenge records only. No external identity lookup, geocoding, business registry, or outside data augmentation.
- Final model license must meet the challenge's MIT/Apache 2.0 rule and its parameter limit. Verify the pinned model and dependency licenses in the release PR.
- The final zip needs both output TSVs, runnable source under `code/business_entity_resolution/src/`, reproduction instructions, pinned requirements, and a completed `Documentation_template.md`.
- The separate guideline also asks for a concise 1-2 page approach summary and commented experiment/training/inference code. Provide that alongside the complete template where the portal requests it.

## Technical strategy

### 1. Data and preprocessing

Stream input rather than loading all approximately 24 million source records into Python objects. Check schema, source prefixes, duplicate IDs, nulls, Unicode, and country distribution. Keep raw strings for traceability. Create normalized name/address views with Unicode-safe case folding, punctuation/whitespace handling, and measured abbreviation rules. Do not remove informative numbers or force transliteration without validation. Country is an open-set string, not a fixed two-class feature.

### 2. Candidate generation

Build complementary retrieval routes, partitioned or indexed to fit a 16 GB machine: exact/near-exact normalized names, discriminative name tokens, address/locality/number signals, and a fuzzy text route. Union and deduplicate routes, then apply a bounded final candidate policy. Record which route found each pair. The emitted `candidate_pairs.tsv` must list exactly the candidates actually sent into model inference.

Measure edge recall, fraction of S1 entities with all true links retrieved, average/p95/p99 candidates per S1, wall time, peak RAM, and the oracle macro F0.5 ceiling. Increase recall before spending time on a better classifier if missed positives dominate.

### 3. Pair scoring

Train on retrieved candidate pairs, including realistic hard negatives. Candidate features should include name and address character/token similarity, agreement or conflict on strong components, missing-value flags, source, and retrieval-route signals. Never use the numeric entity ID as an ML feature. A gradient-boosted tabular classifier is the first model to test; its exact choice is conditional on validation quality, runtime, and license verification. Save the model, feature order, normalization version, split seed, and training command.

### 4. Decision and error analysis

Choose thresholds by sweeping the **official macro F0.5**, including singletons, on held-out S1 entities. Permit multiple links per S1. Check separate S2/S3 and US/India behavior, and inspect missed true links versus wrong merges. France has no labels: keep its processing language-agnostic, inspect its candidate distribution, and do not report an invented French validation score. Add source-specific or query-level rules only when holdout evidence supports them.

## Milestones and PR sequence

| Milestone | Deliverable | Exit evidence |
| --- | --- | --- |
| M0: contracts (Harshit's H0 PR) | Minimal package skeleton and agreed schemas/CLI | Small fixture passes through interfaces |
| M1: baseline | Streaming ingest, one retrieval route, simple scorer, valid output | Full train holdout metric plus one format-valid test output |
| M2: stronger retrieval | Multi-route candidates with bounded RAM | Recall/ceiling and runtime comparison against M1 |
| M3: learned matcher | Model, hard negatives, threshold sweep | Reproducible macro F0.5, slice metrics, error cases |
| M4: final | Best-known-good inference, both TSVs, docs, zip | Official validator PASS; fresh reproduction or recorded full-run evidence |

M0 starts immediately. The role-specific PR identifiers and acceptance checks are in the four assignment files; unresolved choices have named owners in `docs/DECISION-REGISTER.md`. Aim to have M1 before pursuing optional features. Freeze a known-good commit and output early enough to permit a complete final inference run and package check before 27 September 23:59 IST.

## Experiment policy

Use a fixed S1-level holdout; do not tune on its labels while training features or fitting the classifier. The training S2/S3 corpus may be indexed for unlabeled retrieval, but validation labels must stay isolated. Build hard negatives from the same retrieval process used at inference. Compare experiments on the same holdout and record retrieval configuration, candidate budget, model parameters, thresholds, resource use, and errors. Public leaderboard feedback is a secondary check, not the sole optimizer.

## Risks and fallback

| Risk | Response |
| --- | --- |
| 10 million-record retrieval exceeds RAM | Process by country/partition, use compact or disk-backed indexes, bounded batches, and record peak RAM. |
| Normalization loses script or numeric evidence | Preserve raw fields, run multilingual/number fixtures, compare candidate recall. |
| High candidate recall but false merges | Add hard negatives, contradiction features, calibrate threshold on macro F0.5. |
| Strong pair model but poor final score | Inspect singletons, threshold, candidate ceiling, and per-source behavior. |
| France out of training distribution | Use open-set country handling, Unicode-safe text, and proxy robustness checks; avoid claims of measured French accuracy. |
| Late regression or failed full inference | Keep tagged known-good model/config/output; revert to it for submission if the new run is not verified. |

## Out of scope until measured need

No web dashboard, autonomous agent, external business data, oversized language model, or global graph merge is required. Graph consistency and heavier text models are optional experiments only after M1-M3 work end to end and their value can be measured.
