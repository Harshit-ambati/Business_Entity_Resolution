"""Build/open disk-backed or compact source index; ID-to-record lookup.

Provides ``build_index`` to create a persisted index from S2/S3 source files
and ``open_index`` to load it. The ``IndexStore`` exposes inverted-index
lookups for candidate retrieval routes and a ``get_record`` accessor for the
downstream matching model.

Contract reference: docs/CONTRACTS.md §3 (Sabeena section).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pickle
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

    Stores source paths/sizes, normalization and index versions, route
    configuration, and creation timestamp.  Persisted as JSON so that
    ``open_index`` can detect stale or incompatible indexes.
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
    """In-memory index store with inverted indexes for candidate retrieval.

    Provides:
    - ``get_record``: entity ID → NormalizedRecord lookup
    - ``lookup_exact_name``: exact normalized name → entity IDs
    - ``lookup_token``: name token → entity IDs (posting list)
    - ``lookup_address_token``: address token → entity IDs (posting list)
    - ``get_token_doc_freq``: document frequency of a name token
    - ``get_country_entities``: all entity IDs for a country key

    All posting lists are sorted for deterministic iteration.
    """

    def __init__(
        self,
        records: dict[str, NormalizedRecord],
        name_exact_index: dict[str, list[str]],
        token_postings: dict[str, list[str]],
        token_doc_freq: dict[str, int],
        country_entities: dict[str, set[str]],
        address_token_postings: dict[str, list[str]],
        total_records: int,
        manifest: IndexManifest,
    ) -> None:
        self._records = records
        self._name_exact = name_exact_index
        self._token_postings = token_postings
        self._token_doc_freq = token_doc_freq
        self._country_entities = country_entities
        self._address_token_postings = address_token_postings
        self._total_records = total_records
        self._manifest = manifest

    # -- Record lookup -------------------------------------------------------

    def get_record(self, entity_id: str) -> NormalizedRecord:
        """Return the NormalizedRecord for *entity_id*.

        Raises ``KeyError`` if the ID is not in the index, or ``ValueError``
        if the ID does not have a valid S2-/S3- prefix.
        """
        validate_entity_id(entity_id, CANDIDATE_PREFIXES)
        try:
            return self._records[entity_id]
        except KeyError:
            raise KeyError(
                f"Entity {entity_id!r} not found in index "
                f"({self._total_records} records indexed)"
            ) from None

    # -- Inverted-index lookups -----------------------------------------------

    def lookup_exact_name(
        self,
        name_norm: str,
        country_key: str | None = None,
    ) -> list[str]:
        """Return sorted entity IDs matching *name_norm* exactly.

        If *country_key* is given, restrict to entities in that country.
        """
        ids = self._name_exact.get(name_norm, [])
        if country_key is not None:
            country_set = self._country_entities.get(country_key, set())
            ids = [eid for eid in ids if eid in country_set]
        return ids

    def lookup_token(self, token: str) -> list[str]:
        """Return sorted entity IDs whose name contains *token*."""
        return self._token_postings.get(token, [])

    def lookup_address_token(self, token: str) -> list[str]:
        """Return sorted entity IDs whose address contains *token*."""
        return self._address_token_postings.get(token, [])

    def get_token_doc_freq(self, token: str) -> int:
        """Return the number of indexed records containing *token* in name."""
        return self._token_doc_freq.get(token, 0)

    def get_country_entities(self, country_key: str) -> set[str]:
        """Return the set of entity IDs with *country_key*."""
        return self._country_entities.get(country_key, set())

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
        return sorted(self._country_entities.keys())


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

    Creates ``{work_dir}/blocking/manifest.json`` and
    ``{work_dir}/blocking/index.pkl``.  Returns the manifest for use with
    ``open_index``.

    Parameters
    ----------
    source2_path : path
        Path to the Source-2 TSV file.
    source3_path : path
        Path to the Source-3 TSV file.
    work_dir : path
        Parent working directory; a ``blocking/`` subdirectory is created.
    config : IndexConfig or dict, optional
        Index configuration.  Defaults to ``IndexConfig()``.
    """
    source2_path = Path(source2_path)
    source3_path = Path(source3_path)
    work_dir = Path(work_dir)
    blocking_dir = work_dir / "blocking"
    blocking_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = IndexConfig()
    elif isinstance(config, dict):
        config = IndexConfig.from_dict(config)

    logger.info("Building index from S2/S3 sources...")
    t0 = time.time()

    # -- 1. Read and normalize all S2/S3 records -----------------------------
    records: dict[str, NormalizedRecord] = {}
    seen_ids: set[str] = set()
    for path, prefix in [(source2_path, "S2-"), (source3_path, "S3-")]:
        for record in read_source(path, prefix):
            if record.entity_id in seen_ids:
                raise ValueError(f"Duplicate entity ID: {record.entity_id}")
            seen_ids.add(record.entity_id)
            records[record.entity_id] = normalize_record(record)

    total = len(records)
    logger.info("Loaded and normalized %d S2/S3 records in %.1fs", total, time.time() - t0)

    # -- 2. Count name-token document frequencies ----------------------------
    token_doc_freq: dict[str, int] = {}
    for nrec in records.values():
        for token in set(nrec.name_tokens):
            token_doc_freq[token] = token_doc_freq.get(token, 0) + 1

    # -- 3. Build exact-name index -------------------------------------------
    name_exact: dict[str, list[str]] = {}
    for eid, nrec in records.items():
        if nrec.name_norm:
            name_exact.setdefault(nrec.name_norm, []).append(eid)
    for key in name_exact:
        name_exact[key].sort()

    # -- 4. Build name-token posting lists -----------------------------------
    token_postings: dict[str, list[str]] = {}
    for eid, nrec in records.items():
        for token in set(nrec.name_tokens):
            token_postings.setdefault(token, []).append(eid)
    for key in token_postings:
        token_postings[key].sort()

    # -- 5. Build address-token posting lists --------------------------------
    address_token_postings: dict[str, list[str]] = {}
    for eid, nrec in records.items():
        for token in set(nrec.address_tokens):
            address_token_postings.setdefault(token, []).append(eid)
    for key in address_token_postings:
        address_token_postings[key].sort()

    # -- 6. Build country index ----------------------------------------------
    country_entities: dict[str, set[str]] = {}
    for eid, nrec in records.items():
        if nrec.country_key:
            country_entities.setdefault(nrec.country_key, set()).add(eid)

    # -- 7. Log index statistics ---------------------------------------------
    logger.info("  Name-exact index: %d unique normalized names", len(name_exact))
    logger.info("  Token postings:   %d unique name tokens", len(token_postings))
    logger.info("  Address postings: %d unique address tokens", len(address_token_postings))
    logger.info("  Countries:        %s", sorted(country_entities.keys()))

    # -- 8. Create and persist manifest + index ------------------------------
    manifest = IndexManifest(
        work_dir=str(blocking_dir),
        source2_path=str(source2_path),
        source3_path=str(source3_path),
        source2_size=source2_path.stat().st_size,
        source3_size=source3_path.stat().st_size,
        normalization_version=NORMALIZATION_VERSION,
        index_version=INDEX_VERSION,
        route_config=config.to_dict(),
        record_count=total,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    manifest_path = blocking_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    index_data = {
        "records": records,
        "name_exact": name_exact,
        "token_postings": token_postings,
        "token_doc_freq": token_doc_freq,
        "country_entities": country_entities,
        "address_token_postings": address_token_postings,
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

def open_index(manifest: IndexManifest | str | Path) -> IndexStore:
    """Open a previously built index.

    Parameters
    ----------
    manifest : IndexManifest, str, or Path
        Either an ``IndexManifest`` returned by ``build_index``, a path to
        the ``manifest.json`` file, or a path to the ``blocking/`` directory.

    Raises
    ------
    ValueError
        If the normalization or index version in the manifest does not match
        the current code version.
    FileNotFoundError
        If the index pickle file is missing.
    """
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = IndexManifest.from_dict(json.load(f))

    # Validate versions to prevent stale index reuse
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

    blocking_dir = Path(manifest.work_dir)
    index_path = blocking_dir / "index.pkl"

    logger.info("Loading index from %s ...", index_path)
    t0 = time.time()

    with index_path.open("rb") as f:
        data = pickle.load(f)  # noqa: S301

    store = IndexStore(
        records=data["records"],
        name_exact_index=data["name_exact"],
        token_postings=data["token_postings"],
        token_doc_freq=data["token_doc_freq"],
        country_entities=data["country_entities"],
        address_token_postings=data["address_token_postings"],
        total_records=data["total_records"],
        manifest=manifest,
    )

    logger.info(
        "Index loaded in %.1fs: %d records, %d countries",
        time.time() - t0,
        store.total_records,
        len(store.countries),
    )
    return store
