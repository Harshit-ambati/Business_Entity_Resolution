# T3 audit infrastructure status

This file separates **what is implemented and tested** from **what has not yet been
run** for the T3 data-quality audit work in PR #7 (`feat/data-normalization`).
No organizer row, external lookup, or test label was used.

> **No full-dataset run has been executed.**  Do not treat this as a measured
> 8 GB result.  The claim "memory-bounded audit" refers to the algorithm design
> (disk-partitioned duplicate checker, streaming validators), not a confirmed
> run on the actual challenge archive.

---

## What is implemented and tested (synthetic fixtures only)

| Item | Status |
| --- | --- |
| `read_source`/`read_truth` streaming readers | Implemented; 195 tests pass |
| `normalize_record` Unicode normalizer | Implemented; tested (Latin, Arabic, Devanagari, Cyrillic, NFKC) |
| `validate_source_file` / `validate_truth_file` | Implemented; reports `is_valid`, row counts, country/script distributions |
| `check_duplicate_ids_partitioned` disk-partitioned checker | Implemented; tested on small fixture |
| Missing required file raises `FileNotFoundError`, CLI exits 1 | Implemented; regression tests pass |
| Empty required file sets `is_valid=False`, CLI exits 2 | Fixed in `afaea8d`; reviewer-reproduced case tested |
| Invalid TSV content sets `is_valid=False`, CLI exits 2 | Implemented; regression tests pass |
| CLI `--temp-dir` flag | Implemented |
| `git diff --check` | Clean |

## What is NOT yet measured

| Item | Blocker | Path to resolve |
| --- | --- | --- |
| Full run on the extracted challenge archive (10-24 M rows) | Challenge TSVs not extracted; C: drive at 0 bytes free | Extract `data/student_resource.zip` to D:, run `python -m ber.audit --data-root D:\extracted --temp-dir D:\audit_tmp` |
| Peak RSS during full audit | Same | Wrap run with `psutil.Process().memory_info()` or `/usr/bin/time -v` on Linux |
| Wall-clock time for full run | Same | Captured automatically by `audit_dataset` `elapsed_seconds` field |

## Intended full-run command (once disk space is resolved)

`python -m ber.audit --data-root D:\extracted --temp-dir D:\audit_tmp --output D:\audit_tmp\audit_report.json`

Output will contain `duplicate_check_method: "partitioned_disk"`, `elapsed_seconds`,
and per-file `is_valid`, `total_rows`, `duplicate_ids_found`.
Attach that output to the PR before T3 is declared complete.

## Scope of this PR

This PR delivers the **reader, normalizer, and audit infrastructure**.
The T3 full-run evidence belongs in a follow-on commit once the challenge
archive is extracted and disk space is available.
