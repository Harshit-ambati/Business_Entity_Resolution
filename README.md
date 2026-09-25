# Business Entity Resolution | Amazon ML Challenge 2026

Team: **Harshit, Sabeena, Thulasi, Suresh**
Challenge window: **25 September 2026, 00:00 IST to 27 September 2026, 23:59 IST**

This repository is the team's working plan and, as PRs land, the reproducible solution. It is not yet an implemented model or a measured leaderboard result. The source of truth for competition rules is the organiser-provided `student_resource` archive and challenge PDFs. If a rule here conflicts with those materials, correct this plan in a PR before implementing the affected behavior.

## The problem

For every deduplicated Source 1 business, identify **all** records in Sources 2 and 3 that describe the same real-world business, including the valid answer of no matches. Input and output are TSV. The leaderboard scores `output/matching_results.tsv` using macro-averaged F0.5 per Source 1 entity. The final package also requires `output/candidate_pairs.tsv`, containing the exact candidates the matching model scored, runnable code, pinned dependencies, and the filled methodology template. A dashboard or autonomous agent is not a challenge deliverable.

The supplied data has 2,206,821 training Source 1 records and 1,732,544 test Source 1 records. Test retrieval searches 9,969,589 Source 2/3 records. Training has 7,638,365 labeled links; 123,247 Source 1 entities (5.6%) have no matches. France appears in test but not training. These counts were streamed from the provided archive; model quality has not yet been measured.

## Get the challenge data

The organizer-provided [student resource archive](data/student_resource.zip) is tracked with **Git LFS** because it is 1,094,823,222 bytes; the seven TSVs inside total 2,520,573,701 bytes. Install Git LFS before cloning or run `git lfs pull` in an existing clone. The tracked archive's SHA-256 is `2aefd2f8eb6f132b8933fccc1cbb98fa6356756a7581fed5914a5f26704a8bc5`.

```powershell
git lfs install
git lfs pull
python scripts/prepare_dataset.py
```

The helper extracts only the organizer's seven TSVs, validator, template, and README into ignored `student_resource/`. Use `student_resource/dataset` as the data root for the pipeline. It skips macOS metadata and never commits expanded files. See [data/README.md](data/README.md) for contents and integrity checks.

## Solution we will build

```text
TSV input -> validated records and Unicode-safe normalization
          -> multi-route, bounded candidate retrieval
          -> pairwise name/address/source features
          -> trained match scorer
          -> thresholded multi-match or empty decision
          -> evaluated and validator-checked TSV outputs
```

Candidate retrieval determines which true links remain possible; the model determines which candidate links are safe to emit. We will measure both. The first complete run is more valuable than an unfinished sophisticated model. Improvements enter `main` only when held-out macro F0.5, retrieval quality, and runtime evidence support them.

## Read before coding

| Document | Purpose |
| --- | --- |
| [PLAN.md](PLAN.md) | Scope, milestones, technical strategy, and submission gates |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | Shared file, data, and function contracts |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | Offline scoring, blocking checks, output validation, and experiment records |
| [docs/TEAM-WORKFLOW.md](docs/TEAM-WORKFLOW.md) | Branches, PR reviews, integration, and handoffs |
| [Harshit assignment](docs/01-HARSHIT.md) | Model, thresholding, integration, release |
| [Sabeena assignment](docs/02-SABEENA.md) | Scalable candidate retrieval and blocking |
| [Thulasi assignment](docs/03-THULASI.md) | Data contracts, normalization, and quality checks |
| [Suresh assignment](docs/04-SURESH.md) | Metrics, error analysis, output validation, and packaging support |

## Repository layout as work lands

```text
code/business_entity_resolution/
  src/ber/                 # self-contained pipeline code
  tests/                   # fast fixtures and behavioral tests
  README.md                # exact data -> outputs reproduction steps
  requirements.txt         # pinned, license-reviewed dependencies
docs/                     # ownership, contracts, verification
output/                   # generated locally; not committed
```

Only the original archive is tracked with Git LFS. Extracted TSVs, generated indexes, models, and output TSVs stay out of Git. The final submission zip will contain the required code and output files. Do not send business records to outside lookup services, geocoders, or entity-resolution APIs.

## Immediate order of work

1. Harshit merges a small contract/skeleton PR so all branches share the same module and CLI expectations.
2. Thulasi and Suresh implement data and evaluation/output foundations on small fixtures; Sabeena prototypes retrieval on a bounded sample.
3. Harshit trains the first end-to-end matcher against Sabeena's candidates and runs a complete validation pass.
4. Improve the largest measured failure mode, freeze the best reproducible run, validate both outputs, and submit.

All implementation changes after this repository bootstrap use PRs into `main`. See [team workflow](docs/TEAM-WORKFLOW.md).
