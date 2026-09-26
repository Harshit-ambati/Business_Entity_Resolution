# Business Entity Resolution package — H1 pair model

This package supplies shared contracts, the fixed Source 1 split, H1 pair features, an interpretable rule score, and a CPU LightGBM pair classifier. It has trained only on invented synthetic records. It has not generated competition outputs or measured competition quality.

## Layout and setup

All Python source, including tests, is under `src/`. `src/ber/contracts.py` defines shared records and candidate/decision types; `src/ber/split.py` defines the holdout; `src/ber/cli.py` names the pipeline stages. Python 3.10 or newer is required. From this package directory, in a normal Python environment:

`src/tests/` is the single test directory specified by `docs/CONTRACTS.md`; later workstreams should add tests there.
`CandidateGroup` carries the S1 ID for its candidates. `Candidate.route_scores` may be omitted when a route has no numeric score; the field then contains an empty read-only mapping.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

The editable install makes `import ber` and `python -m ber.cli` work from any working directory in that environment. H1 pins LightGBM 4.7.0 (MIT); it brings NumPy/SciPy. LightGBM's [PyPI release](https://pypi.org/project/lightgbm/4.7.0/) provides a Windows wheel and lists Python 3.14. The fixtures in `src/tests/fixtures/` and `src/tests/test_h1.py` use invented businesses and run without challenge data. They cover multiple links, hard negatives, a singleton, France, Japanese text, and missing addresses. Tests establish behavior, not competition quality.

## Challenge data and paths

From the repository root, install Git LFS, pull the tracked archive, and extract it:

```powershell
git lfs install
git lfs pull
python scripts/prepare_dataset.py
```

The resulting default **example** data root is `student_resource/dataset`; any equivalent directory can be passed through `--data-root`. The archive and extraction details are in the repository's `data/README.md`. Extracted TSVs and generated artifacts are ignored by Git. `artifacts/` is the proposed ignored work directory on this machine; the selected path can differ on another machine.

## CLI contract

Each subcommand takes all three explicit path flags **after** the command name:

```powershell
python -m ber.cli index --data-root student_resource/dataset --work-dir artifacts --output-dir output
python -m ber.cli train --data-root student_resource/dataset --work-dir artifacts --output-dir output
python -m ber.cli evaluate --data-root student_resource/dataset --work-dir artifacts --output-dir output
python -m ber.cli predict --data-root student_resource/dataset --work-dir artifacts --output-dir output
python -m ber.cli validate --data-root student_resource/dataset --work-dir artifacts --output-dir output
```

`python -m ber.cli --help` and per-command `--help` describe this interface. `train` reports that the H1 model API is ready while dataset training awaits merged data/normalization/index/blocking modules; other stages still report their H0 stub status. All unavailable stages exit with status 3. Missing required flags exit nonzero with an argparse error. No CLI stage creates artifacts or claims success yet.

**Production candidate retrieval, dataset model training, threshold tuning, evaluation, full inference, and official output generation remain pending.** Thulasi owns TSV ingestion/normalization, Sabeena owns indexing/blocking, and Suresh owns metrics/output. H1 model functions consume their shared contracts when merged.

## H1 pair feature schema

`ber.features.pair_features(s1, candidate, retrieval)` returns 23 floats in `FEATURE_NAMES` order: `name_exact`, `name_token_jaccard`, `name_token_containment`, `name_char_ratio`, `name_length_ratio`, `name_prefix_ratio`, `s1_name_missing`, `candidate_name_missing`, `address_exact`, `address_token_jaccard`, `address_token_containment`, `address_char_ratio`, `address_length_ratio`, `s1_address_missing`, `candidate_address_missing`, `both_addresses_missing`, `number_overlap_count`, `number_jaccard`, `number_any_agreement`, `number_conflict`, `is_source2`, `is_source3`, `retrieval_route_count`. Version `h1.1` fixes this order. Training and scoring call the same function.

Token Jaccard and containment use **sets** of already-normalized tokens. If either side is empty, token similarities are 0. Exact/character/length/prefix similarities are 0 if either normalized string is empty, including both empty. Character ratio uses `difflib.SequenceMatcher(autojunk=False)`; length ratio is shorter/longer, and prefix ratio is shared prefix length divided by shorter length. Address missingness has three explicit flags; a missing address does not create an address mismatch penalty in the rule scorer. Decimal runs are extracted from normalized name and address tokens; if either side has no numbers, numeric Jaccard, agreement, and conflict are 0. `number_conflict` means both have numbers but their sets differ, and is never a hard veto. The source features use only `S2-`/`S3-` prefixes; opaque ID suffixes are never features. Route count uses the candidate's sorted route tuple; route score magnitudes are ignored because their route-specific scales are not calibrated.

`ber.model.rule_score` is an **H1 temporary interpretable baseline**, clipped to [0, 1]. Its name blend is 0.45 token Jaccard + 0.25 containment + 0.30 character ratio; the total is 0.75 of that plus 0.20 address blend (half Jaccard, half character ratio) when both addresses exist, plus 0.05 for any number agreement and minus 0.08 for number conflict. These weights were not tuned on the fixed holdout.

## Training and compatibility

`build_training_pairs` accepts aligned normalized S1, final `CandidateGroup`, `TruthRow` triples and a store exposing `get_record(id)`. It skips validation S1 before reading its labels. For each training S1 it includes every **retrieved** positive and samples at most `negatives_per_s1` from the first `top_negative_pool` ranked retrieved nonmatches, using SHA-256 of `seed|S1_ID` as a per-group random seed. Defaults are seed 2026, 4 negatives from the top 12; these are configurable temporary values. Unretrieved truth IDs increment `missed_positives` and are never injected. The returned counts separate S1 groups, retrieved positives, missed positives, sampled negatives, and skipped validation groups. Callers must bound the input S1 subset on a laptop; `train_model` additionally rejects more than 100,000 training pairs by default.

`train_model(training_pairs, validation_pairs, TrainConfig)` checks the H0 split and fits only training pairs; validation pairs are checked for partition membership and never used to fit or tune. CPU LightGBM parameters are binary objective, 40 rounds by default, 7 leaves, 0.05 learning rate, minimum 2 rows per leaf, 2 threads, deterministic column-wise mode, and no row or feature subsampling. The model text file and JSON manifest are saved under the chosen ignored model directory. `load_model` checks feature version/order, library version, artifact path confinement, and SHA-256. `score_group` checks normalization/index/candidate config versions, retrieves each record, and returns one probability per candidate in group order; missing records raise errors. It scores one group at a time, without building a corpus-wide DataFrame. The JSON manifest records model type/library/version/license, seed, feature version/names, normalization/index/candidate versions, fixed split, pair counts, negative policy, artifact path/checksum, timestamp, and training parameters. It contains no threshold.

For the bounded invented-data smoke run, from this package directory execute:

```powershell
python src/tests/h1_fixture_run.py --work-dir artifacts/h1-fixture
```

The script selects the first 16 synthetic S1 IDs that pass `is_validation_s1`, adds one deliberately unretrieved synthetic truth ID, uses seed 73 and 2 negatives from the top 3 per S1, then writes model, manifest, and report under the ignored work directory. It reports runtime and peak Python allocations measured by `tracemalloc`; that figure excludes native LightGBM allocations. If optional `psutil` is installed on Windows, it also reports peak process working-set bytes. The fixture has no valid challenge metric. A real held-out macro F0.5 comparison awaits Suresh's evaluator and actual candidates. H2 threshold selection remains open.

## Suresh Workstream — Metrics & Output

Suresh implements `ber.metrics` and `ber.output` according to `docs/CONTRACTS.md` and `docs/04-SURESH.md`:
- Official Macro $F_{0.5}$ evaluation with empty match handling and edge-level recall (`ber.metrics.evaluate`).
- Candidate retrieval diagnostics and oracle ceiling calculation (`ber.metrics.evaluate_candidates`).
- Prediction error analysis breakdown (`ber.metrics.error_analysis`).
- Output TSV generation preserving exact $S_1$ order, strict ID validation, and atomic writes (`ber.output.write_outputs`).
- Preflight validator enforcing submission formatting, candidate file requirements, and ID integrity (`ber.output.validate_outputs`).
- Organizer validator runner wrapper (`ber.output.run_organizer_validator`).

Full evaluation documentation and verification results are in [docs/SURESH-EVALUATION.md](../../docs/SURESH-EVALUATION.md).

## Fixed validation partition

Call `ber.is_validation_s1(source1_entity_id)`. It rejects non-S1 IDs. It computes `sha256(("2026|" + source1_entity_id).encode("utf-8"))`, interprets the **first eight digest bytes** as an unsigned big-endian integer, and returns `True` if the value modulo 10 equals 0. This is an approximately 10% S1-level holdout, deterministic and independent of file order. It uses no Python process hash and stores no ID list. Training/holdout country and singleton counts require Thulasi's streaming readers and are not measured in H0.

## Machine resource check

Run these from the repository root on Windows before a full run. The disk command measures the volume containing the proposed relative `artifacts/` work directory; substitute the actual work directory if it is on another volume.

```powershell
Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfLogicalProcessors
Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,TotalVisibleMemorySize
python -c "import shutil; from pathlib import Path; p=Path('artifacts'); u=shutil.disk_usage(p.parent); print(f'work_dir={p} free_bytes={u.free} total_bytes={u.total}')"
```

The RAM values above are reported by Windows in KiB; divide by 1,048,576 for GiB. Snapshot on 25 September 2026 for the intended 16 GB full-run laptop: AMD Ryzen 5 7520U, 8 logical processors, 16,023,164 KiB visible RAM (15.28 GiB), 2,943,640 KiB free RAM (2.81 GiB), and 98,870,026,240 free disk bytes (92.08 GiB) on the `artifacts/` volume. Free RAM and disk change over time; this snapshot is not a full-pipeline capacity benchmark.
