# Decisions that must not be guessed

This page separates **fixed contracts** from experimental choices. An owner proposes each open choice in the named PR, records the selected value and evidence, then updates this table. Until then, teammates may prototype behind the shared interface but must not make another workstream depend on an undocumented default. Challenge rules and [CONTRACTS.md](CONTRACTS.md) are already fixed.

| Decision | Owner | Resolve by | Current status | Evidence required |
| --- | --- | --- | --- | --- |
| Python package install/test command and CLI flags | Harshit | H0 | Selected in H0: from `code/business_entity_resolution`, `python -m pip install -r requirements.txt`, `python -m pip install -e .`, then `python -m pytest`; invoke `python -m ber.cli <index|train|evaluate|predict|validate> --data-root PATH --work-dir PATH --output-dir PATH` | Synthetic fixture tests and CLI flag/help checks on the 16 GB development laptop; fresh 8 GB clean-clone check remains for teammate review |
| S1-level validation split rule and seed | Harshit | H0 | Selected in contracts: SHA-256 of `2026|S1_ID`, first 8 bytes mod 10 = 0 for validation | Implement/test exact rule; report counts by country/singleton |
| `NormalizedRecord` representation/version | Thulasi with Harshit/Sabeena review | T2 | Selected baseline v1 in contracts; later changes need new version | Unicode and labeled-pair regression tests |
| Index file format and record lookup strategy | Sabeena | S1 | Open | Build/load/lookup test, disk/RAM measurement |
| Retrieval routes and candidate cap | Sabeena with Harshit review | S2/S3 | Temporary route order and cap 32 in Sabeena brief; final config open | Compare caps 16/32/64 on same holdout; recall, oracle F0.5, volume, runtime |
| Exact pair-feature list/order | Harshit | H1 | Open | Manifest, train/test parity test, ablation or error evidence |
| Model family, exact version, license | Harshit | H1/H3 | Open | Held-out score, resource use, license record |
| Hard-negative sampling policy | Harshit | H1 | Open | Reproducible sampled counts and validation comparison |
| Thresholds and singleton decision rule | Harshit | H2 | Open | Macro F0.5 sweep and singleton/error report |
| Full-run batch sizes and restart checkpoints | Harshit/Sabeena | H2/S3 | Open | Complete run without exceeding 16 GB RAM |
| Output join strategy and checksum manifest | Suresh | R3 | Open | 8 GB full-size or bounded stress check; organizer validator PASS |

## How to resolve a decision

In the relevant PR, state the chosen value, alternatives tried, data split, metric/runtime evidence, and any effect on another owner. Update this table from `Open` to `Selected: ...` with a PR link. A choice with no measurement can be an explicitly labeled **temporary baseline**, but it is not a final optimization claim. If a decision changes a public type/function, update [CONTRACTS.md](CONTRACTS.md) and request affected-owner review before merging.
