# Sabeena | candidate retrieval and blocking

**Machine:** 16 GB RAM. **Workstream:** indexed, high-recall retrieval at approximately 10 million S2/S3 test records. **Starting branch:** `feat/candidate-retrieval`.

## Mission

For each S1 record, produce a bounded list containing as many true S2/S3 links as practical. The matcher cannot recover a true link omitted here. Retrieval must complete within measured RAM and time limits, and its final candidate set must be exactly what the model scores.

## Owned files

- `code/business_entity_resolution/src/ber/blocking.py`: route union, deduplication, candidate caps, ordering.
- `code/business_entity_resolution/src/ber/index.py` or equivalent: compact, reproducible corpus index and batch lookup.
- Retrieval tests and benchmark script under `code/business_entity_resolution/tests/` and `scripts/`.
- Blocking methodology and experiment results supplied to the final documentation.

## PR milestones

1. **Baseline route:** index normalized exact names and useful rare tokens; retrieve both S2 and S3. Demonstrate bounded memory on a sample that contains known positives and distractors. Empty results and unseen country labels work.
2. **Complementary routes:** add a fuzzy name route and address/locality route. Union results before a measured final cap. Preserve route metadata for Harshit's feature model. Do not hard-code only US/India.
3. **Scale run:** build the full training index, query the validation S1 split, and report edge recall, all-links-per-S1 recall, candidate counts, oracle macro F0.5, wall time, and peak RAM. Optimize the largest missed-link categories. Repeat the selected config for test inference.

## Design rules

- Use country as an open-set label for partitioning where beneficial; a `France` partition must be created from test inputs automatically. Verify whether a strict country condition loses any training positives before treating it as a gate.
- Use compact/disk-backed or partitioned indexes and batched queries. A naive full 10 million-record Python dictionary or dense pairwise similarity matrix is not an acceptable scale plan for 16 GB RAM.
- Give address evidence a route independent of name: true links include cross-script and shortened-name pairs. Missing addresses must fall back to name retrieval.
- Keep original raw data and source ID references available to the scorer. Candidate lists may include multiple records from both sources; do not force top-1.
- Stable ordering/tie breaks, deterministic config, no labels used to retrieve validation/test pairs, no outside business data.
- The candidate file reflects post-filter candidates **immediately before scoring**. If a cheap filter prunes a route result, it must happen before the file is written.

## Acceptance and handoff

- Expose the interface in `docs/CONTRACTS.md`; output ID sets are valid and deduplicated.
- Benchmark table compares each added route's incremental true-link recall and extra candidate cost; there is no assumed target score without measurement.
- Full validation run is reproducible with recorded configuration and observed peak RAM no higher than the available machine.
- Harshit receives index build/query commands, index/cache format, route metadata schema, sample output, metrics, runtime, and remaining failure cases.

## Review partners

Harshit reviews retrieval PRs and integrates them into model inference. Thulasi reviews normalization assumptions. Suresh checks candidate metric and output compatibility.
