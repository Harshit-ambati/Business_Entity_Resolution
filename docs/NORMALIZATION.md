# Data Contracts, Normalization, and Data Quality (Thulasi Workstream)

This document specifies the implementation contracts, Unicode normalization semantics, data quality validation, and streaming audit capabilities delivered in PRs T1–T3.

## 1. Shared Ingestion Contracts (`ber.data`)

### Source Reader: `read_source`
```python
read_source(path: str | Path, expected_prefix: str, *, check_duplicates: bool = False) -> Iterator[Record]
```
- **File Format:** UTF-8 encoded, tab-delimited (`\t`).
- **Logical Header:** Exactly `entity_id`, `business_name`, `business_address`, `country`.
- **Validation:**
  - `expected_prefix` must be strictly `"S1-"`, `"S2-"`, or `"S3-"`.
  - Every `entity_id` is validated to begin with `expected_prefix` and have a non-empty opaque suffix.
  - Exactly 4 tab-delimited columns required per row; malformed rows report file path and 1-indexed line number.
  - Commas and quotes inside addresses are safely parsed without column shifting.
  - Empty address is valid and produces `Record.business_address == ""`.
- **Resource Safety:** Generator pattern ensures underlying file handles are closed cleanly upon completion or early break.
- **Memory Efficiency:** Rows are yielded lazily as `Record` slots dataclasses; no entire dataset is retained in memory. `check_duplicates=False` by default for multi-million row processing on 8 GB RAM machines. Full duplicate audits are delegated to disk-partitioned utilities.

### Truth Reader: `read_truth`
```python
read_truth(path: str | Path, *, check_duplicates: bool = True) -> Iterator[TruthRow]
```
- **Logical Header:** Exactly `source1_entity_id`, `matched_entity_ids`.
- **Validation:**
  - `source1_entity_id` must begin with `"S1-"` and have a non-empty suffix.
  - Duplicate S1 IDs within the truth file raise `ValueError` with path and line number.
  - `matched_entity_ids` is parsed from the comma-separated string inside the second column.
  - An empty cell yields `matched_entity_ids == ()` (singletons).
  - Each matched candidate ID must begin with `"S2-"` or `"S3-"` and have a non-empty suffix.
  - Duplicate candidate IDs within a single truth row raise `ValueError`.

---

## 2. Text and Record Normalization (`ber.normalize`)

### Exported Version
```python
NORMALIZATION_VERSION = "1"
```
Any change that modifies normalized string representation or token output requires bumping this version and coordinating an index/model rebuild with Sabeena and Harshit.

### Normalization Pipeline: `normalize_record`
```python
normalize_record(record: Record) -> NormalizedRecord
```
Given a `Record`, `normalize_record` produces a pure, deterministic `NormalizedRecord` containing:
- `raw`: Original untouched `Record` instance.
- `name_norm`: Normalized business name string.
- `address_norm`: Normalized business address string.
- `name_tokens`: Ordered tuple of normalized name tokens.
- `address_tokens`: Ordered tuple of normalized address tokens.
- `country_key`: Normalized open-set country key.

### Text Normalization Semantics (`normalize_text`)
1. **Unicode NFKC:** Applies `unicodedata.normalize("NFKC", text)` to standardize full-width characters and ligatures.
2. **Unicode Casefold:** Applies `.casefold()` for language-independent case insensitivity (e.g. `ß` -> `ss`).
3. **Ampersand Handling:** Replaces `&` with the token `and` (surrounded by spaces: `" and "`), ensuring `"A & B"`, `"A&B"`, and `"A and B"` generate identical tokens `("a", "and", "b")`.
4. **Punctuation & Separator Handling:** Replaces punctuation, symbols, and formatting separators with space characters while strictly preserving:
   - Unicode Letters (Unicode category `L*`)
   - Unicode Numbers/Digits (Unicode category `N*`)
   - Unicode Combining Marks / Diacritics (Unicode category `M*`)
5. **Whitespace Collapsing & Tokenization:** Collapses consecutive whitespace (spaces, tabs, newlines) into single spaces and splits into an ordered tuple of tokens.

### Intentional Constraints in Normalization v1
- **Preserves Raw Fields:** Normalization never overwrites or destroys source data.
- **No Abbreviation Expansion:** `st` is NOT expanded to `street` or `saint`; `rd` is NOT expanded to `road`; `ltd` is NOT expanded to `limited`. This avoids false-positive merges.
- **Preserves Legal Suffixes:** Suffixes such as `llc`, `sarl`, `ltd`, `pvt` remain in `name_norm` and token tuples.
- **Preserves Informative Digits:** House numbers, suite numbers, and numeric brand elements (e.g., `12`, `7`, `4b`, `101`) are preserved intact.
- **Preserves Accents and Non-Latin Scripts:** Preserves European diacritics (`é`, `ü`, `ç`) and non-Latin scripts:
  - Devanagari (`किताब घर`)
  - Telugu (`భారత్ ఎలక్ట్రానిక్స్`)
  - Japanese / CJK (`さくら商店`)
  - Cyrillic (`ООО Вектор`)
- **Empty / Null Handling:** Empty or whitespace-only name or address returns `""` and an empty token tuple `()`.

### Country Normalization (`normalize_country`)
- **Open-Set:** No country allowlist is used. `United States`, `India`, `France`, and any unseen country (e.g. `Côte d'Ivoire`) produce valid keys.
- **Transformations:** NFKC normalization, case folding, and whitespace collapse only. Punctuation is not removed and words are not substituted.

---

## 3. Data Quality and Audit Utilities (`ber.quality`)

### Streaming File Audits
- `validate_source_file(path, expected_prefix, *, check_duplicates=True, max_errors=50) -> SourceQualityReport`:
  Streams a source TSV and records row counts, schema validity, prefix errors, duplicate IDs, missing names, empty addresses, missing countries, country distribution, and script classification.
- `validate_truth_file(path, *, check_duplicates=True, max_errors=50) -> TruthQualityReport`:
  Streams a ground-truth TSV and audits S1 prefixes, duplicate S1 IDs, duplicate matched IDs, candidate prefixes, singleton counts (0 matches), single matches, multi-matches, and link count distribution.

### Memory-Bounded Duplicate Checking
- `check_duplicate_ids_partitioned(path, id_column=0, num_partitions=64, temp_dir=None) -> list[str]`:
  Uses disk-backed MD5 hash partitioning to split IDs into small buckets. Each bucket is audited independently in memory, guaranteeing that verifying 10M–24M IDs never exceeds the 8 GB RAM target.

### CLI Audit Helper
Run the audit tool against any dataset directory:
```powershell
python -m ber.quality --data-root student_resource/dataset --output artifacts/data_audit_report.json
```
