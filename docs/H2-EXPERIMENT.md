# H2 decision and evaluation record

## Integration state, 26 September 2026

H1 pair scoring is merged: 23 features in schema `h1.1`, `rule_score`, and CPU LightGBM 4.7.0. The model manifest pins feature order, training policy, versions, and artifact SHA-256. Suresh's `evaluate`, `evaluate_candidates`, and oracle sanity check are merged and used directly by H2. The pre-H2 suite passed 121 tests.

`src/ber/data.py`, `normalize.py`, `index.py`, and `blocking.py` were absent from `main` when H2 began. Therefore a real fixed holdout cannot yet be constructed or scored. The `evaluate` CLI exits 3 with this missing-module explanation. There is no measured challenge score, selected production threshold, or production model comparison. The threshold/singleton item in `DECISION-REGISTER.md` remains **Open**.

## Implemented H2 policy

- One immutable global `DecisionConfig(threshold)`; valid finite threshold range is [0, 1]. A pair is emitted on `score >= threshold` in candidate order. Zero, one, and multiple matches are valid, including S2 and S3 together.
- Candidate/score length and S1 identity must agree. Candidate IDs are constrained by existing contracts. Repeated candidate IDs are rejected when forming a `CandidateGroup`.
- Threshold selection uses Suresh's macro per-S1 F0.5, including correct empty singleton rows. The default coarse grid is 0.00–1.00 by 0.05; the fine grid is ±0.05 around the best coarse threshold by 0.01. Exact ties prefer the higher threshold.
- The candidate oracle is computed with `evaluate_candidates` before the sweep. Every evaluated decision must stay at or below that oracle. Repeat-pass SHA-256 fingerprints guard against changed input streams. No source-specific or country-specific thresholds are implemented.
- Diagnostic categories separate unretrieved true links, retrieved/rejected true links, wrong emitted links, and singleton false-positive groups. A 0.05 margin below the threshold divides retrieved/rejected links into **provisional** scoring and threshold-sensitive decision buckets; this is a diagnostic convention, not a proven causal explanation. The report accepts a feature provider for compact error examples. Detailed real challenge rows must remain in ignored artifacts.

## Invented-data smoke run

Command from `code/business_entity_resolution`:

```powershell
python src/tests/h2_fixture_run.py --work-dir ../../artifacts/h2-fixture
```

The run trained on 16 invented, fixed-split training S1 groups and 48 sampled pairs. Both scorers used the same five invented validation S1 groups, seven candidates, truth, and synthetic versions (`fixture-1`, `fixture-index`, `fixture-candidates`). The model artifact checksum was `1384ec826675c8ae862765dfe0f11eaef809b85e87b66778103fb908f493f68d`. These numbers are fixture behavior only; they say nothing about leaderboard or real holdout quality.

| Synthetic fixture measure | Rule | LightGBM |
|---|---:|---:|
| Best global threshold | 1.00 | 0.55 |
| Macro F0.5 | 0.966667 | 0.966667 |
| Candidate oracle F0.5 | 0.966667 | 0.966667 |
| True edge recall | 0.80 | 0.80 |
| Complete-match S1 recall | 0.666667 | 0.666667 |
| S2 true-edge recall | 1.00 | 1.00 |
| S3 true-edge recall | 0.50 | 0.50 |
| Singleton correct / total | 2 / 2 | 2 / 2 |
| US macro F0.5 | 1.00 | 1.00 |
| India macro F0.5 | 0.916667 | 0.916667 |
| Retrieval-miss true links | 1 | 1 |
| Retrieved/rejected true links | 0 | 0 |
| Wrong emitted links | 0 | 0 |
| Singleton false-positive groups | 0 | 0 |

The oracle gap is zero for both scorers on this tiny fixture. There are no French validation labels and no France accuracy is claimed. The US/India slices above refer only to invented labels. The maximum confidence false-merge review cannot be meaningfully performed because this synthetic run has zero false merges.

The command reported 7.60 seconds of wall time, 181,993,472 bytes of process RSS **at completion**, and a 19,308-byte JSON report. The RSS number is not peak memory. It scored seven validation candidate pairs after fitting 48 synthetic training pairs. Results are saved locally at `artifacts/h2-fixture/h2_synthetic_report.json`, which is ignored by Git. No real score cache or restart checkpoint was needed for this five-group fixture.

## Next integration gate

After Thulasi's readers/normalization and Sabeena's retrieval modules are merged, connect the `evaluate` CLI to a bounded, versioned scored-group stream; record model/candidate/normalization/index versions and check it against the model manifest. Then run the fixed validation split, compare rule and LightGBM on identical final candidates, save the complete holdout report and representative errors locally, measure peak RAM/disk/runtime, and select a threshold only from that evidence. H3 inference and submission packaging are outside this PR.
