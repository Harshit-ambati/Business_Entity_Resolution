# Sabeena: large-scale candidate retrieval and blocking

**Machine:** 16 GB RAM. **Starting branch:** `feat/candidate-retrieval`. **Primary reviewer:** Harshit. **Contract reviewers:** Thulasi for normalized fields; Suresh for candidate metrics/output semantics. **Authoritative shared interface:** [CONTRACTS.md](CONTRACTS.md).

## Outcome you own

Given an S1 file plus the S2/S3 files from the *same split*, retrieve a bounded, deduplicated, deterministic set of plausible S2/S3 IDs for **every** S1 row. Your final `CandidateGroup` is exactly the set Harshit's model scores and Suresh's `candidate_pairs.tsv` writer emits. Optimize true-link recall and oracle macro F0.5 under measured time/RAM limits; a high pairwise model score cannot recover a true pair you never retrieved.

## First implementation recipe (run routes in this order)

1. Build a **split-specific** index of normalized S2/S3 records. Store ID-to-record lookup on disk or in a measured compact format. Partition by whatever country labels appear in that split, including `France` in test; do not create a US/India enum. Confirm with labeled training data before making country equality a strict filter.
2. Route A: exact normalized business-name lookup within the country partition. Return all hits up to a documented per-route limit, with deterministic ID tie breaks.
3. Route B: lookup rare normalized name tokens, excluding high-frequency tokens using a documented frequency cutoff. Common words such as `services` must not create unbounded posting lists.
4. Route C: independent address retrieval using a conjunction of informative address tokens and, where present, number/locality evidence. It must be able to retrieve a synthetic cross-script-name pair whose address still agrees. Never require an address when the record has none.
5. Route D: add fuzzy name retrieval using character grams or another measured compact text index, restricted to manageable partitions/posting lists. Do **not** compute a dense S1 x S2/S3 similarity matrix.
6. Union route results, attach all contributing route names/scores, deduplicate by candidate ID, rank deterministically, and then apply the **temporary baseline cap of 32 candidates per S1**. Benchmark 16, 32, and 64 on the same holdout before choosing the final cap with Harshit. An S1 with no hits still emits an empty `CandidateGroup`.

The routes above are an implementation order, not a claim that each improves the score. If a route adds negligible recall but large cost, show the comparison and omit it from the selected final config. The candidate file must reflect the post-cap group, not all raw route hits.

## Files and output

| File/artifact | You deliver |
| --- | --- |
| `code/business_entity_resolution/src/ber/index.py` | Build/open disk-backed or compact source index; ID-to-record lookup |
| `code/business_entity_resolution/src/ber/blocking.py` | Exact/fuzzy route queries, union, deduplication, cap, deterministic ordering |
| `code/business_entity_resolution/src/tests/test_index.py` | Index build/load/lookup and split-isolation tests |
| `code/business_entity_resolution/src/tests/test_blocking.py` | Retrieval routes, cap, empty/Unicode/France/multiple-source tests |
| `code/business_entity_resolution/src/ber/` benchmark helper | Runnable sampled/full retrieval benchmark, with invocation documented |
| `artifacts/blocking/` (ignored) | Index manifest, config, benchmark report, candidate sample, timings |

Do not write the final submission TSV yourself; return `CandidateGroup` to Suresh's output writer. Do not implement Harshit's classifier or mutate Thulasi's normalization without a reviewed contract change. Record any additional dependency and its license for Harshit's requirements review.

## PR S1: working baseline retrieval

**Input:** Thulasi's `read_source`/`normalize_record` or the M0 fixture until her PR lands. **Deliverables:** `build_index`, `open_index`, `index_store.get_record`, `iter_candidates`, one exact normalized-name route, and one discriminative token route. Each route returns valid S2/S3 IDs only; its name is stored in `retrieval_routes`. A named config contains all caps and route switches. The index manifest records source paths/checksums or sizes, normalization version, and index version so a stale index cannot silently be reused with new data.

**Acceptance:** tiny synthetic fixture has an exact match, two matches to one S1, S2 and S3 results, a singleton, duplicate route hits, common-token distractors, and an unseen `France` country label. One candidate group is yielded for every input S1 in input order. Candidate IDs are unique, route labels sorted, tie order stable, cap applied only after route union, and `get_record(id)` returns the correct raw+normalized record. A documented bounded sample run records runtime and peak RAM. No label file is read by retrieval code.

## PR S2: complementary retrieval for noisy data

**Input:** S1 and S2/S3 raw/normalized name, address, country. **Deliverables:** add fuzzy name retrieval and an independent address/locality route so abbreviated, misspelled, or cross-script name pairs have a path into the candidate set. Consider normalized name tokens/character grams, rare address tokens, house-number and locality evidence; select actual routes based on sample recall and scale. Missing address must fall back to name evidence. Country labels are dynamic partitions if used; do not branch on `{US, India}` or omit France. A strict country block needs validation evidence showing it does not remove labeled training links.

**Acceptance:** compare S1 baseline with each added route on the same labeled validation query set, reporting incremental true-edge recall, complete-link S1 coverage, oracle macro F0.5, mean/p95/p99 candidates, and extra runtime/memory. Include at least one labeled-style fixture where name scripts differ but address evidence retrieves the pair. A large common token must not explode candidate output or RAM. Any late cheap pruning happens before `CandidateGroup` is emitted, and the model sees every emitted candidate.

## PR S3: full-scale, reproducible retrieval

**Input:** train S2/S3 index with held-out training S1 queries, then separate test S2/S3 index with test S1 queries. **Deliverables:** full benchmark report, chosen config, index build/query commands, candidate-set checksum or deterministic sample, failure analysis for missed labeled links, and an inference-ready manifest. Full training and test indexes are kept distinct; never mix IDs from one split into another. The final test candidate stream must be consumable in batches and restartable if a long run fails, without duplicate/missing S1 rows.

**Acceptance:** report wall time, peak RAM observed on 16 GB hardware, disk footprint, route contribution, true-edge recall, complete-link coverage, candidate counts, and oracle macro F0.5. The full held-out run completes without swapping the machine into unusability. If it cannot, reduce memory with partitioning/disk-backed lookup and document the measured fix. Harshit can use the manifest and query API without reimplementing retrieval. Suresh can consume the exact final candidate groups for `candidate_pairs.tsv`.

## Measurement definitions and non-goals

True-edge recall is retrieved positive links divided by all labeled links in the query set. Complete-link coverage is the fraction of *matched* S1 queries for which every true ID was retrieved. Oracle macro F0.5 predicts only true IDs present among candidates, averaging over all S1 queries including singletons. A larger cap is not automatically better: compare gain against pair volume and runtime. Do not present a candidate recall number from a sample that excluded hard distractors as if it were full-holdout performance.

No exhaustive all-pairs comparison, single-route-only final design, external entity lookup, global one-to-one assignment, or undocumented top-K value. IDs are opaque; their numeric suffix cannot serve as a retrieval key. Graph or embedding experiments are optional only after a complete baseline and measured benefit.

## Handoff packet for Harshit and Suresh

Each PR provides: exact command, commit/config, input split, index manifest/version, output schema example, route definitions and score meanings, candidate cap, counts and metrics, runtime/peak RAM/disk, and at least five representative misses with the *reason the retrieval route failed*. Share only small synthetic examples in Git; full challenge records and generated indexes stay ignored. Request a contract PR before adding a field Harshit's model or Suresh's writer must consume.

## Failure behavior and definition of done

- Refuse to open an index whose source split, input fingerprint, normalization version, or index version mismatches the query run. Rebuild it; do not silently reuse stale postings.
- If an indexed candidate ID cannot be looked up or has an impossible source prefix, fail with the ID and route; do not drop it from the list and overstate recall.
- If a common-key posting list exceeds its safe bound, skip or narrow **that route** under a documented policy, count how many queries were affected, and measure the resulting recall loss. Do not let one key exhaust 16 GB RAM.
- **Required:** S1-S3, final candidate groups for every S1, full-holdout recall/ceiling/cost report, reproducible train and test indexes.
- **Recommended after required work:** route-specific improvements for measured cross-script, abbreviation, and missing-address misses.
- **Optional only if time and evidence permit:** ANN embeddings or graph expansion. They must improve the same holdout without breaking the 16 GB full run.
