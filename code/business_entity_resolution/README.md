# Business Entity Resolution — Code Package

**ML Challenge 2026** | Team: Harshit, Sabeena, Thulasi, Suresh

## Setup

```bash
git lfs pull
python scripts/prepare_dataset.py  # extract student_resource/

pip install -e code/business_entity_resolution
```

## Module Ownership

| Module | Owner | Status |
|--------|-------|--------|
| `ber/data.py`, `ber/normalize.py` | Thulasi | Planned |
| `ber/blocking.py`, `ber/index.py` | Sabeena | Planned |
| `ber/features.py`, `ber/model.py`, `ber/decision.py` | Harshit | Planned |
| `ber/metrics.py` | **Suresh** | **Implemented** |
| `ber/output.py` | **Suresh** | **Implemented** |
| `ber/cli.py` | Harshit (integration) + Suresh (eval commands) | **Partial** |

## Running Tests

```bash
python -m pytest code/business_entity_resolution/tests/ -v
# Expected: 74 passed, ~1s
```

## Suresh CLI Commands

```bash
# Evaluate macro F0.5
ber evaluate --truth <truth.tsv> --predictions <matching_results.tsv>

# Candidate diagnostics
ber candidates --truth <truth.tsv> --candidates <candidate_pairs.tsv>

# Preflight + organizer validator
ber validate \
    --test-s1 student_resource/dataset/test/test_source1.tsv \
    --matching output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv \
    --run-organizer

# Error analysis
ber error-analysis --truth <truth.tsv> \
    --predictions <matching_results.tsv> \
    --candidates <candidate_pairs.tsv> --verbose
```

## Organizer Validator (direct)

Run from `student_resource/`:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Exit code 0 = format/coverage/consistency PASS. Does NOT measure ML quality.

## See Also

- [docs/SURESH-EVALUATION.md](../../docs/SURESH-EVALUATION.md) — Full evaluation docs
- [docs/CONTRACTS.md](../../docs/CONTRACTS.md) — Interface contracts
- [docs/VERIFICATION.md](../../docs/VERIFICATION.md) — Acceptance criteria
