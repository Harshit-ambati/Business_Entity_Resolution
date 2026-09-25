# Thulasi | data ingestion, normalization, quality

**Machine:** 8 GB RAM. **Workstream:** deterministic, streaming data layer and trustworthy text preparation. **Starting branch:** `feat/data-normalization`.

## Mission

Give every workstream one correct view of the raw TSVs and normalized business text. Preserve useful evidence while making names and addresses comparable. All development and tests should fit on 8 GB; full corpus profiling is streamed rather than retained in RAM.

## Owned files

- `code/business_entity_resolution/src/ber/data.py`: TSV readers, schema and prefix checks, ground-truth parsing.
- `code/business_entity_resolution/src/ber/normalize.py`: Unicode-safe normalization, tokens, optional address component extraction.
- Tests for TSV parsing, Unicode, missing fields, unseen countries, determinism, and noisy formats.
- Small data-quality report/command and normalization methodology notes.

## PR milestones

1. **Readers and fixtures:** stream the four-column source files and two-column truth file with explicit tab delimiters. Preserve empty addresses and empty match lists. Report invalid rows with file and row number. Provide tiny synthetic fixtures for all three sources and ground truth.
2. **Normalization v1:** use Unicode normalization/case folding, punctuation and whitespace handling, conservative legal-suffix and common street-abbreviation handling. Keep raw strings; expose normalized strings/tokens without silently dropping accents, non-Latin scripts, or house numbers.
3. **Quality report:** stream counts by source/country, missing-field rates, ID prefix/schema issues, and representative format patterns. Compare normalization on labeled sample pairs and hard negatives before introducing an aggressive rule.

## Edge cases to cover

- TSV addresses contain commas; ID lists contain commas within the second TSV cell.
- Names may be abbreviated, reordered, DBA/trade names, accented French, Devanagari, Telugu, or other Unicode scripts.
- Addresses can be missing, reordered, landmark-based, abbreviated, or contain conflicting numbers. A number conflict is a feature for the matcher, not an automatic nonmatch at ingest.
- Test `France` must survive even though training contains only `US` and `India`.
- S1 IDs begin `S1-`, S2 IDs `S2-`, S3 IDs `S3-`; treat the remainder as opaque.

## Acceptance and handoff

- Iterating the readers does not materialize a whole multi-million-row file. A 100,000-row fixture or bounded real-data scan fits comfortably on 8 GB.
- Same input gives byte-for-byte stable normalized fields, with a documented normalization version.
- Tiny fixtures demonstrate cross-script, accents, ampersand/and, legal suffixes, empty address, and unrecognized country.
- Sabeena and Harshit receive a stable `NormalizedRecord` example and a list of any intentionally unresolved normalization cases.

## Review partners

Harshit reviews data/normalization PRs; Sabeena checks retrieval needs; Suresh checks parsing and ID coverage assumptions.
