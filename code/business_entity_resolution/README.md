# Business Entity Resolution package — H0

This package currently supplies shared in-memory contracts, the fixed Source 1 validation split, a small synthetic fixture, and a command-line surface. It has not run a model or generated competition outputs.

## Layout and setup

All Python source, including tests, is under `src/`. `src/ber/contracts.py` defines shared records and candidate/decision types; `src/ber/split.py` defines the holdout; `src/ber/cli.py` names the pipeline stages. Python 3.10 or newer is required. From this package directory, in a normal Python environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
```

The editable install makes `import ber` and `python -m ber.cli` work from any working directory in that environment. H0 has no runtime dependency outside Python's standard library; the pinned requirement installs pytest for tests. The fixtures in `src/tests/fixtures/` use invented businesses and run without the challenge data. They cover two true links for one S1, a similar-name hard negative, a singleton, France, Japanese text, and an empty address. The tests do not claim training or evaluation quality.

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

`python -m ber.cli --help` and per-command `--help` describe this interface. At H0, each stage accepts the flags, prints `Not implemented in H0`, and exits with status 3. Missing required flags exit nonzero with an argparse error. No stage creates artifacts or claims success yet.

**Candidate retrieval, model training, threshold tuning, evaluation, full inference, and official output generation are not implemented in H0.** Thulasi owns TSV ingestion/normalization, Sabeena owns indexing/blocking, and Suresh owns metrics/output. Later Harshit PRs will add features, model, decisions, and orchestration against their reviewed interfaces.

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
