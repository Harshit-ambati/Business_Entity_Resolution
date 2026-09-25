# H1 controlled training record

This is a **synthetic smoke experiment**, not a challenge validation result. The run used the H1 branch working tree based on `main` commit `d24a3ec`. The implementation and script are in the accompanying H1 PR. No organizer row, external business lookup, or test label was used.

| Item | Measured run on 26 September 2026 |
| --- | --- |
| Command | `python src/tests/h1_fixture_run.py --work-dir artifacts/h1-fixture` from `code/business_entity_resolution` |
| Subset | First 16 generated fixture S1 IDs that pass `is_validation_s1`; no match-quality selection |
| Split | SHA-256 of `2026|S1_ID`, first eight bytes big-endian modulo 10; validation iff zero |
| Feature schema | `h1.1`, 23 ordered numeric features |
| Normalization version | `1` declared for synthetic `NormalizedRecord` fixture values; Thulasi's normalizer was not available |
| Candidate/index config | `synthetic-only` (in-memory test double, not Sabeena's pipeline) |
| Seed and negative policy | Seed 73, sample up to 2 of first 3 ranked retrieved nonmatches per training S1 |
| Training pairs | 16 retrieved positives; 32 sampled negatives across 16 S1 groups |
| Retrieval misses | 1 intentionally unretrieved synthetic positive; it was not inserted into candidates |
| Model | LightGBM 4.7.0 binary gradient boosted trees, MIT license; Windows CPU, Python 3.14 |
| Hyperparameters | 8 rounds, 7 leaves, learning rate 0.05, minimum 2 rows per leaf, 2 threads, deterministic column-wise mode, seed 73, no row/feature subsampling |
| Runtime | 17.698 seconds, including pair construction, training, model/manifest writes and Python tracing |
| Peak process working set | 255,180,800 bytes via Windows `psutil.Process().memory_info().peak_wset` |
| Peak traced Python allocations | 93,899,894 bytes; excludes native allocations |
| Model SHA-256 | `1384ec826675c8ae862765dfe0f11eaef809b85e87b66778103fb908f493f68d` |
| Metric | None. Synthetic fixture predictions cannot establish macro F0.5 or leaderboard performance. |

Model and report files stay under ignored `artifacts/h1-fixture/`. Re-running produces the same model checksum with this installed library/version and fixture. Runtime and memory may vary. The real candidate-recall count, rule-versus-model fixed-holdout macro F0.5, and eventual threshold selection require the teammate candidate generator and evaluator. H2 owns thresholds; no threshold was selected here.
