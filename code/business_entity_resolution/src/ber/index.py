"""Build/open disk-backed or compact source index; ID-to-record lookup.

Provides ``build_index`` to create a persisted index from S2/S3 source files
and ``open_index`` to load it. The ``IndexStore`` exposes inverted-index
lookups for candidate retrieval routes and a ``get_record`` accessor for the
downstream matching model.

Contract reference: docs/CONTRACTS.md §3 (Sabeena section).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pickle
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import CANDIDATE_PREFIXES, NormalizedRecord, validate_entity_id

# Prefer Thulasi's modules; fall back to local shims until her PR lands.
try:
    from .normalize import normalize_record, NORMALIZATION_VERSION  # type: ignore[import-not-found]
except ImportError:
    from .normalize_shim import normalize_record, NORMALIZATION_VERSION

try:
    from .data import read_source  # type: ignore[import-not-found]
except ImportError:
    from .data_shim import read_source

logger = logging.getLogger(__name__)

INDEX_VERSION = "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_file_fingerprint(path: Path | str, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a file in chunks to detect modifications."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def detect_split(path: Path | str) -> str:
    """Infer the dataset split ('train', 'test', or 'fixture') from a file path."""
    p = Path(path)
    parent_parts = [part.lower() for part in p.parts[:-1]]

    # 1. Any fixture directory anywhere in path takes precedence
    if any(part in ("fixture", "fixtures") for part in parent_parts):
        return "fixture"

    # 2. Dataset split directories (strict names: 'test', 'train'; exclude general 'tests' dir)
    for part in reversed(parent_parts):
        if part in ("test",):
            return "test"
        if part in ("train", "training"):
            return "train"

    # 3. Check filename and stem (for flat files e.g. test_source2.tsv, train_source2.tsv)
    stem = p.stem.lower()
    if stem.startswith("test_source") or stem == "test" or stem.startswith("test_") or stem.endswith("_test"):
        return "test"
    if stem.startswith("train_source") or stem == "train" or stem.startswith("train_") or stem.endswith("_train"):
        return "train"
    if stem.startswith("fixture_") or stem == "fixture" or stem.endswith("_fixture") or "fixture" in stem:
        return "fixture"

    # 4. Check final part if no parent matched
    last_part = p.parts[-1].lower() if p.parts else ""
    if last_part in ("test",):
        return "test"
    if last_part in ("train",):
        return "train"
    if last_part in ("fixture", "fixtures"):
        return "fixture"

    return "unspecified"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IndexConfig:
    """Configuration for index building and candidate retrieval.

    Route switches control which retrieval routes ``iter_candidates`` uses.
    Index-level parameters affect both build-time index construction and
    query-time behaviour.
    """

    # Route switches (query-time; indexes are always built for all routes)
    exact_name: bool = True
    name_token: bool = True
    address: bool = False       # S2 PR
    combined: bool = False      # S2 PR
    fuzzy_name: bool = False    # S2 PR

    # Split label
    split: str = "unspecified"

    # Limits
    max_candidates_per_s1: int = 32
    max_block_size: int = 10_000
    token_max_doc_freq_ratio: float = 0.01
    token_min_length: int = 2

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IndexConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class IndexManifest:
    """Provenance record for a built index.

    Stores source paths/sizes/fingerprints, normalization and index versions,
    split, route configuration, and creation timestamp. Persisted as JSON so
    that ``open_index`` can detect stale or incompatible indexes.
    """

    work_dir: str
    source2_path: str
    source3_path: str
    source2_size: int
    source3_size: int
    normalization_version: str
    index_version: str
    route_config: dict[str, Any]
    record_count: int
    created_at: str
    source2_fingerprint: str = ""
    source3_fingerprint: str = ""
    split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IndexManifest:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Index store
# ---------------------------------------------------------------------------

class IndexStore:
    """Disk-backed index store with inverted indexes and record lookup in SQLite.

    Provides:
    - ``get_record``: entity ID → NormalizedRecord lookup (disk-backed via SQLite
      or in-memory fallback, with local caching)
    - ``lookup_exact_name``: exact normalized name → entity IDs (queried from SQLite)
    - ``lookup_token``: name token → entity IDs (queried from SQLite)
    - ``lookup_address_token``: address token → entity IDs (queried from SQLite)
    - ``get_token_doc_freq``: document frequency of a name token (queried from SQLite)
    - ``get_country_entities``: all entity IDs for a country key (queried from SQLite)

    All posting lists are sorted for deterministic iteration. Python heap memory
    is strictly bounded because all postings and records reside in SQLite on disk.
    """

    def __init__(
        self,
        records: dict[str, NormalizedRecord] | None,
        name_exact_index: dict[str, list[str]] | None,
        token_postings: dict[str, list[str]] | None,
        token_doc_freq: dict[str, int] | None,
        country_entities: dict[str, set[str]] | None,
        address_token_postings: dict[str, list[str]] | None,
        total_records: int,
        manifest: IndexManifest,
        db_path: Path | str | None = None,
    ) -> None:
        self._records = records
        self._name_exact = name_exact_index
        self._token_postings = token_postings
        self._token_doc_freq = token_doc_freq
        self._country_entities = country_entities
        self._address_token_postings = address_token_postings
        self._total_records = total_records
        self._manifest = manifest
        self._db_path = Path(db_path) if db_path else None
        self._db_conn: sqlite3.Connection | None = None
        self._cache: dict[str, NormalizedRecord] = {}

        if self._db_path and self._db_path.exists():
            self._db_conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._db_conn.execute("PRAGMA query_only = ON")

    def close(self) -> None:
        """Close SQLite connection if open."""
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None

    def __enter__(self) -> "IndexStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -- Record lookup -------------------------------------------------------

    def get_record(self, entity_id: str) -> NormalizedRecord:
        """Return the NormalizedRecord for *entity_id*.

        Raises ``KeyError`` if the ID is not in the index, or ``ValueError``
        if the ID does not have a valid S2-/S3- prefix.
        """
        validate_entity_id(entity_id, CANDIDATE_PREFIXES)

        # 1. Check in-memory store if present (test fallback)
        if self._records is not None and entity_id in self._records:
            return self._records[entity_id]

        # 2. Check local query cache
        if entity_id in self._cache:
            return self._cache[entity_id]

        # 3. Query disk-backed SQLite store
        if self._db_conn is not None:
            cur = self._db_conn.cursor()
            cur.execute("SELECT data FROM records WHERE entity_id = ?", (entity_id,))
            row = cur.fetchone()
            if row is not None:
                rec: NormalizedRecord = pickle.loads(row[0])
                if len(self._cache) < 16384:
                    self._cache[entity_id] = rec
                return rec

        raise KeyError(
            f"Entity {entity_id!r} not found in index "
            f"({self._total_records} records indexed)"
        )

    # -- Inverted-index lookups -----------------------------------------------

    def lookup_exact_name(
        self,
        name_norm: str,
        country_key: str | None = None,
    ) -> list[str]:
        """Return sorted entity IDs matching *name_norm* exactly.

        If *country_key* is given, restrict to entities in that country.
        """
        if self._db_conn is not None:
            cur = self._db_conn.cursor()
            if country_key is not None:
                cur.execute(
                    "SELECT entity_id FROM postings_exact "
                    "WHERE name_norm = ? AND country_key = ? "
                    "ORDER BY entity_id",
                    (name_norm, country_key),
                )
            else:
                cur.execute(
                    "SELECT entity_id FROM postings_exact "
                    "WHERE name_norm = ? "
                    "ORDER BY entity_id",
                    (name_norm,),
                )
            return [row[0] for row in cur.fetchall()]

        if self._name_exact is not None:
            ids = self._name_exact.get(name_norm, [])
            if country_key is not None and self._country_entities is not None:
                country_set = self._country_entities.get(country_key, set())
                ids = [eid for eid in ids if eid in country_set]
            return ids
        return []

    def lookup_token(self, token: str) -> list[str]:
        """Return sorted entity IDs whose name contains *token*."""
        if self._db_conn is not None:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT entity_id FROM postings_name_token "
                "WHERE token = ? "
                "ORDER BY entity_id",
                (token,),
            )
            return [row[0] for row in cur.fetchall()]

        if self._token_postings is not None:
            return self._token_postings.get(token, [])
        return []

    def lookup_address_token(self, token: str) -> list[str]:
        """Return sorted entity IDs whose address contains *token*."""
        if self._db_conn is not None:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT entity_id FROM postings_address_token "
                "WHERE token = ? "
                "ORDER BY entity_id",
                (token,),
            )
            return [row[0] for row in cur.fetchall()]

        if self._address_token_postings is not None:
            return self._address_token_postings.get(token, [])
        return []

    def get_token_doc_freq(self, token: str) -> int:
        """Return the number of indexed records containing *token* in name."""
        if self._db_conn is not None:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT doc_freq FROM token_doc_freq WHERE token = ?",
                (token,),
            )
            row = cur.fetchone()
            return row[0] if row else 0

        if self._token_doc_freq is not None:
            return self._token_doc_freq.get(token, 0)
        return 0

    def get_country_entities(self, country_key: str) -> set[str]:
        """Return the set of entity IDs with *country_key*."""
        if self._db_conn is not None:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT entity_id FROM country_entities WHERE country_key = ?",
                (country_key,),
            )
            return {row[0] for row in cur.fetchall()}

        if self._country_entities is not None:
            return self._country_entities.get(country_key, set())
        return set()

    # -- Properties -----------------------------------------------------------

    @property
    def total_records(self) -> int:
        """Total number of S2/S3 records in the index."""
        return self._total_records

    @property
    def manifest(self) -> IndexManifest:
        return self._manifest

    @property
    def countries(self) -> list[str]:
        """Sorted list of country keys present in the index."""
        if self._db_conn is not None:
            cur = self._db_conn.cursor()
            cur.execute(
                "SELECT DISTINCT country_key FROM country_entities "
                "WHERE country_key != '' ORDER BY country_key"
            )
            return [row[0] for row in cur.fetchall()]

        if self._country_entities is not None:
            return sorted(self._country_entities.keys())
        return []


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_index(
    source2_path: str | Path,
    source3_path: str | Path,
    work_dir: str | Path,
    config: IndexConfig | dict[str, Any] | None = None,
) -> IndexManifest:
    """Build a candidate index from S2/S3 source files and persist to disk.

    Creates ``{work_dir}/blocking/manifest.json``,
    ``{work_dir}/blocking/records.db`` (disk-backed SQLite records and inverted postings),
    and ``{work_dir}/blocking/index.pkl`` (metadata).
    Returns the manifest for use with ``open_index``.

    Parameters
    ----------
    source2_path : path
        Path to the Source-2 TSV file.
    source3_path : path
        Path to the Source-3 TSV file.
    work_dir : path
        Parent working directory; a ``blocking/`` subdirectory is created.
    config : IndexConfig or dict, optional
        Index configuration. Defaults to ``IndexConfig()``.
    """
    source2_path = Path(source2_path)
    source3_path = Path(source3_path)
    if not source2_path.exists():
        raise FileNotFoundError(f"Source-2 file not found: {source2_path}")
    if not source3_path.exists():
        raise FileNotFoundError(f"Source-3 file not found: {source3_path}")

    work_dir = Path(work_dir)
    blocking_dir = work_dir / "blocking"
    blocking_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = IndexConfig()
    elif isinstance(config, dict):
        config = IndexConfig.from_dict(config)

    # Validate split consistency and reject conflicting overrides
    s2_split = detect_split(source2_path)
    s3_split = detect_split(source3_path)
    if s2_split != "unspecified" and s3_split != "unspecified" and s2_split != s3_split:
        raise ValueError(
            f"Cannot build index with mixed source splits: source2 is {s2_split!r}, "
            f"source3 is {s3_split!r}"
        )
    source_split = s2_split if s2_split != "unspecified" else s3_split

    if config.split != "unspecified" and source_split != "unspecified" and config.split != source_split:
        raise ValueError(
            f"Cannot override detected split {source_split!r} with conflicting config.split {config.split!r}"
        )

    split_name = source_split if source_split != "unspecified" else (
        config.split if config.split != "unspecified" else "train"
    )

    logger.info("Building index from S2/S3 sources (split: %s)...", split_name)
    t0 = time.time()

    # -- 1. Stream & normalize records directly into SQLite disk store -------
    records_db_path = blocking_dir / "records.db"
    if records_db_path.exists():
        records_db_path.unlink()

    conn = sqlite3.connect(str(records_db_path))
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA cache_size = 50000")

    conn.execute("CREATE TABLE records (entity_id TEXT PRIMARY KEY, data BLOB)")
    conn.execute("CREATE TABLE postings_exact (name_norm TEXT NOT NULL, country_key TEXT NOT NULL, entity_id TEXT NOT NULL)")
    conn.execute("CREATE TABLE postings_name_token (token TEXT NOT NULL, entity_id TEXT NOT NULL)")
    conn.execute("CREATE TABLE postings_address_token (token TEXT NOT NULL, entity_id TEXT NOT NULL)")
    conn.execute("CREATE TABLE country_entities (country_key TEXT NOT NULL, entity_id TEXT NOT NULL)")
    conn.execute("CREATE TABLE token_doc_freq (token TEXT PRIMARY KEY, doc_freq INTEGER NOT NULL)")

    seen_ids: set[str] = set()
    total = 0
    token_doc_freq: dict[str, int] = {}

    records_batch: list[tuple[str, bytes]] = []
    exact_batch: list[tuple[str, str, str]] = []
    token_batch: list[tuple[str, str]] = []
    addr_batch: list[tuple[str, str]] = []
    country_batch: list[tuple[str, str]] = []
    batch_size = 5000

    for path, prefix in [(source2_path, "S2-"), (source3_path, "S3-")]:
        for record in read_source(path, prefix):
            eid = record.entity_id
            if eid in seen_ids:
                conn.close()
                raise ValueError(f"Duplicate entity ID: {eid}")
            seen_ids.add(eid)

            nrec = normalize_record(record)
            records_batch.append((eid, pickle.dumps(nrec, protocol=pickle.HIGHEST_PROTOCOL)))

            if nrec.name_norm:
                exact_batch.append((nrec.name_norm, nrec.country_key or "", eid))
            for token in set(nrec.name_tokens):
                token_batch.append((token, eid))
                token_doc_freq[token] = token_doc_freq.get(token, 0) + 1
            for token in set(nrec.address_tokens):
                addr_batch.append((token, eid))
            if nrec.country_key:
                country_batch.append((nrec.country_key, eid))

            total += 1

            if len(records_batch) >= batch_size:
                conn.executemany("INSERT INTO records VALUES (?, ?)", records_batch)
                conn.executemany("INSERT INTO postings_exact VALUES (?, ?, ?)", exact_batch)
                conn.executemany("INSERT INTO postings_name_token VALUES (?, ?)", token_batch)
                conn.executemany("INSERT INTO postings_address_token VALUES (?, ?)", addr_batch)
                conn.executemany("INSERT INTO country_entities VALUES (?, ?)", country_batch)
                conn.commit()
                records_batch.clear()
                exact_batch.clear()
                token_batch.clear()
                addr_batch.clear()
                country_batch.clear()

    if records_batch:
        conn.executemany("INSERT INTO records VALUES (?, ?)", records_batch)
        conn.executemany("INSERT INTO postings_exact VALUES (?, ?, ?)", exact_batch)
        conn.executemany("INSERT INTO postings_name_token VALUES (?, ?)", token_batch)
        conn.executemany("INSERT INTO postings_address_token VALUES (?, ?)", addr_batch)
        conn.executemany("INSERT INTO country_entities VALUES (?, ?)", country_batch)
        records_batch.clear()
        exact_batch.clear()
        token_batch.clear()
        addr_batch.clear()
        country_batch.clear()

    # Insert token doc frequencies
    freq_items = list(token_doc_freq.items())
    token_doc_freq.clear()
    for i in range(0, len(freq_items), batch_size):
        conn.executemany("INSERT INTO token_doc_freq VALUES (?, ?)", freq_items[i:i + batch_size])
    freq_items.clear()
    conn.commit()

    # Create indexes after bulk insert for maximum speed
    conn.execute("CREATE INDEX idx_exact_name_country ON postings_exact (name_norm, country_key, entity_id)")
    conn.execute("CREATE INDEX idx_exact_name ON postings_exact (name_norm, entity_id)")
    conn.execute("CREATE INDEX idx_name_token ON postings_name_token (token, entity_id)")
    conn.execute("CREATE INDEX idx_address_token ON postings_address_token (token, entity_id)")
    conn.execute("CREATE INDEX idx_country ON country_entities (country_key, entity_id)")
    conn.commit()
    conn.close()

    logger.info("Indexed %d S2/S3 records into SQLite disk store in %.1fs", total, time.time() - t0)

    # -- 2. Compute source SHA-256 fingerprints ------------------------------
    s2_fp = compute_file_fingerprint(source2_path)
    s3_fp = compute_file_fingerprint(source3_path)

    # -- 3. Create and persist manifest + index ------------------------------
    manifest = IndexManifest(
        work_dir=str(blocking_dir),
        source2_path=str(source2_path.resolve()),
        source3_path=str(source3_path.resolve()),
        source2_size=source2_path.stat().st_size,
        source3_size=source3_path.stat().st_size,
        source2_fingerprint=s2_fp,
        source3_fingerprint=s3_fp,
        normalization_version=NORMALIZATION_VERSION,
        index_version=INDEX_VERSION,
        split=split_name,
        route_config=config.to_dict(),
        record_count=total,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    manifest_path = blocking_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    index_data = {
        "total_records": total,
    }
    index_path = blocking_dir / "index.pkl"
    with index_path.open("wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    logger.info("Index built in %.1fs, saved to %s", elapsed, blocking_dir)
    return manifest


# ---------------------------------------------------------------------------
# Open
# ---------------------------------------------------------------------------

def open_index(
    manifest: IndexManifest | str | Path,
    expected_split: str | None = None,
) -> IndexStore:
    """Open a previously built index with freshness, version, and split validation.

    Parameters
    ----------
    manifest : IndexManifest, str, or Path
        Either an ``IndexManifest`` returned by ``build_index``, a path to
        the ``manifest.json`` file, or a path to the ``blocking/`` directory.
    expected_split : str, optional
        Expected split name (e.g. 'train', 'test'). If supplied, must match
        manifest.split.

    Raises
    ------
    ValueError
        If the normalization or index version in the manifest does not match
        the current code version, or if the source file size/fingerprint or
        split does not match.
    FileNotFoundError
        If an indexed source file, SQLite records.db store, or index pickle is missing.
    """
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = IndexManifest.from_dict(json.load(f))

    # 1. Validate versions to prevent stale index reuse
    if manifest.normalization_version != NORMALIZATION_VERSION:
        raise ValueError(
            f"Normalization version mismatch: index has "
            f"{manifest.normalization_version!r}, current is "
            f"{NORMALIZATION_VERSION!r}. Rebuild the index."
        )
    if manifest.index_version != INDEX_VERSION:
        raise ValueError(
            f"Index version mismatch: index has {manifest.index_version!r}, "
            f"current is {INDEX_VERSION!r}. Rebuild the index."
        )

    # 2. Validate split
    if expected_split is not None and manifest.split != expected_split:
        raise ValueError(
            f"Index split mismatch: index was built for split {manifest.split!r}, "
            f"but expected {expected_split!r}. Rebuild the index for the target split."
        )

    # 3. Validate source files freshness and fingerprints
    for path_str, exp_size, exp_fp in [
        (manifest.source2_path, manifest.source2_size, manifest.source2_fingerprint),
        (manifest.source3_path, manifest.source3_size, manifest.source3_fingerprint),
    ]:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(
                f"Stale index: indexed source file {path_str!r} not found. "
                f"Rebuild the index."
            )
        cur_size = p.stat().st_size
        if cur_size != exp_size:
            raise ValueError(
                f"Stale index: source file {path_str!r} size changed "
                f"({cur_size} bytes vs {exp_size} bytes in manifest). Rebuild the index."
            )
        if exp_fp:
            cur_fp = compute_file_fingerprint(p)
            if cur_fp != exp_fp:
                raise ValueError(
                    f"Stale index: source file {path_str!r} content changed "
                    f"(fingerprint {cur_fp[:12]}... vs {exp_fp[:12]}... in manifest). "
                    f"Rebuild the index."
                )

    blocking_dir = Path(manifest.work_dir)
    db_path = blocking_dir / "records.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Index record store missing: {db_path} does not exist. Rebuild the index."
        )

    index_path = blocking_dir / "index.pkl"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file missing: {index_path}")

    logger.info("Loading index from %s ...", blocking_dir)
    t0 = time.time()

    with index_path.open("rb") as f:
        data = pickle.load(f)  # noqa: S301

    store = IndexStore(
        records=data.get("records"),
        name_exact_index=data.get("name_exact"),
        token_postings=data.get("token_postings"),
        token_doc_freq=data.get("token_doc_freq"),
        country_entities=data.get("country_entities"),
        address_token_postings=data.get("address_token_postings"),
        total_records=data.get("total_records", manifest.record_count),
        manifest=manifest,
        db_path=db_path,
    )

    logger.info(
        "Index loaded in %.1fs: %d records, %d countries",
        time.time() - t0,
        store.total_records,
        len(store.countries),
    )
    return store
