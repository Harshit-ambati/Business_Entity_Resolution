# Team workflow and pull requests

The empty-repository documentation commit bootstraps `main`. After that, development happens in short-lived branches and PRs to `main`, mirroring the useful parts of Coldchain Guardian's ownership and review process.

## Owners and equipment

| Person | Laptop | Primary workstream | Starting branch |
| --- | --- | --- | --- |
| Harshit | 16 GB RAM | Contracts, pair features/model, thresholding, end-to-end integration, final release | `feat/contracts-and-pipeline` |
| Sabeena | 16 GB RAM | Scalable candidate retrieval and blocking benchmark | `feat/candidate-retrieval` |
| Thulasi | 8 GB RAM | Streaming ingestion, normalization, quality checks | `feat/data-normalization` |
| Suresh | 8 GB RAM | Metrics, diagnostics, output writer, validator/release checks | `feat/evaluation-output` |

Branches are workstreams, not permanent integration branches. Make small PRs at milestones and sync from `main` after each merge. Do not push directly to `main` after bootstrap or force-push a shared branch. No one needs to copy the multi-gigabyte dataset into Git.

## Dependency order without idle time

1. Harshit opens the M0 contract/skeleton PR first, defining import paths and the tiny synthetic fixture. Sabeena, Thulasi, and Suresh review affected interfaces.
2. While M0 is reviewed, each teammate prototypes only within their owned module using local fixtures and the documented contracts.
3. Thulasi's data/normalization PR and Suresh's evaluator/output PR can merge independently. Sabeena's first retrieval PR uses the shared `NormalizedRecord` contract and includes a bounded sample benchmark.
4. Harshit integrates a baseline scorer, runs the full holdout, then coordinates targeted improvements from measured errors. Full-scale runs are primarily on the 16 GB laptops; 8 GB owners use streamed/sampled checks.

## PR rules

- PRs target `main`; title names the behavior, e.g. `feat(blocking): add bounded name and address retrieval`.
- Fill the PR template with owner, contract impact, commands/results, measured resource use where relevant, limitations, and handoff.
- At least **one other team member** reviews and approves each PR. An author cannot approve their own PR. Sabeena reviews Harshit's model/integration PRs; Harshit reviews the other workstreams; Thulasi and Suresh cross-review data and output contracts where relevant.
- Contract changes require review from every directly affected workstream before dependent code merges. Resolve conversations and pass required checks before merging.
- Harshit performs integration and release decisions. If he authors a PR, a teammate must approve it. Keep `main` runnable and a known-good commit/output available.
- Do not claim a full-scale run or model score based on a fixture test. Record unverified behavior explicitly.

## Daily handoff format

At two short checkpoints each day, each person reports: PR/commit, completed behavior, command and result, measured metric or sample, blockers/contract questions, and next deliverable. Put reproducible failures and exact command output in the PR. Avoid simultaneous edits to another owner's module; request a contract change or pair review first.

## Source of truth

Organizer materials control challenge rules; `docs/CONTRACTS.md` controls interfaces; `PLAN.md` controls project scope; assignment files control ownership; `docs/VERIFICATION.md` controls acceptance. If they disagree, update the documents in a reviewed PR. The code and measured outputs take precedence over unverified claims.
