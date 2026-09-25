# Thulasi: streaming input, normalization, data quality

**Machine:** 8 GB RAM. **Starting branch:** `feat/data-normalization`. **Primary reviewer:** Harshit. **Contract reviewers:** Sabeena for retrieval fields; Suresh for IDs/truth parsing. **Authoritative shared interface:** [CONTRACTS.md](CONTRACTS.md).

## Outcome you own

Make the organizer's TSV files readable and comparable without dropping records or destroying information. Give all teammates the same deterministic `Record`, `TruthRow`, and `NormalizedRecord` behavior. The 8 GB laptop is sufficient because your readers and audits stream; no task requires holding millions of Python record objects in memory.

## Files and output

| File/artifact | You deliver |
| --- | --- |
| `code/business_entity_resolution/src/ber/data.py` | Explicit-tab UTF-8 readers, schema/prefix checks, truth-list parser, row context on errors |
| `code/business_entity_resolution/src/ber/normalize.py` | Pure Unicode-safe normalizer and exported normalization version |
| `code/business_entity_resolution/src/tests/test_data.py` | Header, quoting, empty cell, malformed row, prefix, duplicate fixture tests |
| `code/business_entity_resolution/src/tests/test_normalize.py` | Case/punctuation/Unicode/script/number/empty-address regression tests |
| `code/business_entity_resolution/src/ber/` audit helper | Streaming count/missingness/country/prefix report; invocation documented |
| `artifacts/data_audit/` (ignored) | Local reports and sample comparisons, without committing original data |

Do not write the retrieval index, train a model, or generate final outputs. No exact role depends on a specific dataframe library; use a reader that works on 8 GB and accepts original TSV paths on any machine. If you need a dependency, propose it in the PR with pinned-version/licensing impact for Harshit.

## PR T1: source and truth readers

**Input:** seven organizer TSVs, but develop with tiny synthetic fixtures. **Deliverables:** `read_source(path, expected_prefix)` and `read_truth(path)` from [CONTRACTS.md](CONTRACTS.md). Source reads four columns; truth reads two columns. Preserve input order, raw business strings, empty address, and empty `matched_entity_ids` as an empty tuple. Validate exact logical headers and expected prefixes. A malformed record must report file and row number rather than quietly skipping it. Ensure file handles close when iteration finishes.

**Acceptance:** tests cover tabs versus commas, quoted punctuation in an address, empty address, empty truth list, multiple S2/S3 IDs, wrong header, wrong prefix, duplicate IDs within a truth list, duplicate S1 truth row, and malformed column count. Full-file duplicate source-ID checking may use disk-backed sort/index, because an in-memory set of 10 million strings can exceed 8 GB. The PR states which checks run during normal reading and which require a separate full audit; never claim duplicate validation if it was not run. All S1 rows must remain present.

## PR T2: normalization with retained evidence

**Input:** `Record`. **Deliverable:** pure `normalize_record(record)` returning raw record, normalized strings, ordered tokens, open-set `country_key`, and a `NORMALIZATION_VERSION` constant. Start with Unicode NFKC, case folding, whitespace/punctuation normalization, and a small documented set of generic abbreviations. Keep name and address separate. Preserve digits, informative tokens, accents/non-Latin evidence, and original raw text. Do not transliterate scripts or remove legal suffixes destructively unless held-out data shows a benefit; if an alternate canonical view is wanted, request a new contract field rather than overwriting the raw view.

**Acceptance:** exact input gives the same normalized output across runs; no `US`/`India` allowlist exists; `France` and an arbitrary fourth country yield valid keys; empty address yields empty normalized address/tokens. Synthetic fixtures cover ampersand versus `and`, punctuation, `Ltd`/`Limited`, `Rd`/`Road`, reordered tokens, accents, Devanagari, Telugu, mixed scripts, and house numbers. Tests assert important tokens survive normalization rather than only comparing to the implementation's own output.

## PR T3: streaming data audit and documented findings

**Input:** original train/test TSVs plus truth. **Deliverables:** a command that streams counts by file and country, missing-name/address/country counts, prefix/schema errors, and truth match-list length distribution. Record run time, peak RAM, and selected input file checksums/sizes. Compare a bounded sample of labeled positive pairs and plausible negatives before proposing additional normalization rules. Report patterns with denominators; label small-sample estimates as such.

**Acceptance:** the report can complete on an 8 GB laptop without storing all records. File counts match the organizer archive baseline unless the inputs differ: train S1 2,206,821; test S1 1,732,544; train truth 2,206,821. A mismatch stops downstream work until explained. France S1 test count is 259,452 in the provided archive. Do not publish original business rows or query external services to enrich them.

## Interface handoff

Give Sabeena and Harshit one synthetic `NormalizedRecord` example and the exported version string. Document exactly which abbreviations are applied, whether punctuation is replaced or deleted, token order, what happens to empty strings, and what is intentionally **not** normalized. Give Suresh the truth-list parsing behavior and any source-data anomalies. If a teammate asks for an address component not in the current contract, propose the field and tests in a reviewed contract PR. Do not add an undocumented field that only one branch knows about.

## Review evidence and limits

Each PR includes exact test command/output and a bounded memory check, plus a list of remaining data anomalies. Use challenge records locally for profiling but commit only synthetic fixtures and aggregate counts. You are not responsible for a full 10 million-record retrieval or model score; those depend on Sabeena and Harshit respectively. When an aggressive normalization improves some pairs and harms others, report both outcomes so the team can compare held-out retrieval and F0.5 before merging it.
