# Team workflow and pull requests

The empty-repository documentation commit bootstraps `main`. After that, development happens in short-lived branches and PRs to `main`, mirroring the useful parts of Coldchain Guardian's ownership and review process. Read [CONTRACTS.md](CONTRACTS.md) and [DECISION-REGISTER.md](DECISION-REGISTER.md) before implementing a module: the former fixes interfaces; the latter names choices still requiring evidence.

## Owners and equipment

| Person | Laptop | Primary workstream | Starting branch |
| --- | --- | --- | --- |
| Harshit | 16 GB RAM | Contracts, pair features/model, thresholding, end-to-end integration, final release | `feat/contracts-and-pipeline` |
| Sabeena | 16 GB RAM | Scalable candidate retrieval and blocking benchmark | `feat/candidate-retrieval` |
| Thulasi | 8 GB RAM | Streaming ingestion, normalization, quality checks | `feat/data-normalization` |
| Suresh | 8 GB RAM | Metrics, diagnostics, output writer, validator/release checks | `feat/evaluation-output` |

Branches are workstreams, not permanent integration branches. Make small PRs at milestones and sync from `main` after each merge. Do not push directly to `main` after bootstrap or force-push a shared branch. No one needs to copy the multi-gigabyte dataset into Git.

Harshit invites teammates as GitHub collaborators when their account names are known. Until an invitation is accepted, a teammate can work in a fork and open a PR into this repository's `main`. Do not assume every teammate can push to the central repository.

## Dependency order without idle time

1. Harshit opens the H0 contract/skeleton PR first, defining import paths and the tiny synthetic fixture. Sabeena, Thulasi, and Suresh review affected interfaces.
2. While M0 is reviewed, each teammate prototypes only within their owned module using local fixtures and the documented contracts.
3. Thulasi's T1/T2 data/normalization PRs and Suresh's R1 metric PR can merge independently. Sabeena's S1 retrieval PR uses the shared `NormalizedRecord` contract and includes a bounded sample benchmark.
4. Harshit integrates a baseline scorer, runs the full holdout, then coordinates targeted improvements from measured errors. Full-scale runs are primarily on the 16 GB laptops; 8 GB owners use streamed/sampled checks.

## PR rules

- PRs target `main`; title names the behavior, e.g. `feat(blocking): add bounded name and address retrieval`.
- Fill the PR template with owner, contract impact, commands/results, measured resource use where relevant, limitations, and handoff.
- At least **one other team member** reviews and approves each PR. An author cannot approve their own PR. Sabeena reviews Harshit's model/integration PRs; Harshit reviews the other workstreams; Thulasi and Suresh cross-review data and output contracts where relevant.
- Contract changes require review from every directly affected workstream before dependent code merges. Resolve conversations and pass required checks before merging.
- Harshit performs integration and release decisions. If he authors a PR, a teammate must approve it. Keep `main` runnable and a known-good commit/output available.
- Do not claim a full-scale run or model score based on a fixture test. Record unverified behavior explicitly.
- Each assignment names PR-sized deliverables (H0-H3, S1-S3, T1-T3, R1-R4). A branch may be reused after syncing from `main`, but each PR should have one reviewable milestone and exact evidence. A later PR must not depend on an unmerged signature change.
- Harshit coordinates portal uploads and records each team's submission against the daily limit. Teammates do not share login credentials or submit an untracked variant on behalf of the team.

## Daily handoff format

At two short checkpoints each day, each person reports: PR/commit, completed behavior, command and result, measured metric or sample, blockers/contract questions, and next deliverable. Put reproducible failures and exact command output in the PR. Avoid simultaneous edits to another owner's module; request a contract change or pair review first.

## Source of truth

Organizer materials control challenge rules; `docs/CONTRACTS.md` controls interfaces; `docs/DECISION-REGISTER.md` tracks open choices; `PLAN.md` controls project scope; assignment files control ownership; `docs/VERIFICATION.md` controls acceptance. If they disagree, update the documents in a reviewed PR. The code and measured outputs take precedence over unverified claims.
