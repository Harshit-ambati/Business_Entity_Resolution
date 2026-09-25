# Verification and experiment record

## Offline validation

Hold out a fixed set of training S1 IDs before training or threshold tuning; save its seed and ID list or reproducible split rule. Train retrieval and pair scoring without using validation labels. The validation candidate search must use the same implementation and limits as test inference. Train on all labeled training entities only after selection is frozen for final test inference.

For a true set `T` and predicted set `P` for one S1 entity:

- If both are empty, score 1.
- If only one is empty, score 0.
- Otherwise `precision = |P intersect T| / |P|`, `recall = |P intersect T| / |T|`, and `F0.5 = 1.25 * precision * recall / (0.25 * precision + recall)`.
- Average this score over **all** validation S1 entities. Do not substitute pair-level F0.5 or micro F0.5.

The candidate oracle predicts exactly `T intersect C`, where `C` is the candidate set. It quantifies the best score possible under current blocking. Report both total true-edge recall and the fraction of matched S1 entities for which *all* true links are in `C`.

## Required comparison table

| Run | Commit/config | Candidate routes and cap | Edge recall | Oracle macro F0.5 | Actual macro F0.5 | Singleton accuracy | Candidate p95 | Runtime / peak RAM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Record US and India separately, as well as S2 and S3 true-link recall, missing-address cases, and cross-script cases when detectable. Test France has no labels; report only coverage and diagnostics, not an accuracy estimate. Inspect at least representative false positives, false negatives caused by blocking, false negatives caused by scoring, and incorrect singleton decisions.

## Acceptance checks per PR

- Exact command and result for relevant tests or benchmark.
- Small synthetic example for the changed behavior and at least one failure/edge case.
- Runtime and peak memory when the change affects full-scale processing.
- Contract changes called out with dependent owners.
- No challenge dataset, secrets, generated indexes, model binaries, or large TSVs staged.

## Final release gate

1. Freeze the selected commit, training split/seed, model/config, feature order, and output checksums.
2. Generate `matching_results.tsv` and `candidate_pairs.tsv` from the same inference run. Confirm every final match is in that row's final candidate set.
3. Run the organizer's `utils/validate_submission.py --matching ... --candidate ... --test-dir ...` and require `PASS` (exit 0). It checks format, not ML quality.
4. Check output row counts equal test S1 count (1,732,544) and inspect country coverage, including all 259,452 French S1 rows.
5. Create the prescribed zip with both outputs, runnable code, pinned requirements, reproduction README, and filled `Documentation_template.md`. Prepare the guideline's 1-2 page approach summary if requested separately.
6. Record the exact uploaded TSV checksum and portal result. Keep the last known-good submission available until the challenge closes.
