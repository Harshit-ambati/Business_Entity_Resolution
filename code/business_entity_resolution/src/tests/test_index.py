"""Tests for index building, loading, and lookup.

Uses the existing synthetic fixtures from src/tests/fixtures/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ber.contracts import CANDIDATE_PREFIXES
from ber.index import (
    INDEX_VERSION,
    IndexConfig,
    IndexManifest,
    IndexStore,
    build_index,
    open_index,
)
from ber.normalize_shim import NORMALIZATION_VERSION

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared pytest fixture: build + open an index on the synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def index_store(tmp_path: Path) -> IndexStore:
    """Build and open an index from the synthetic S2/S3 fixtures."""
    config = IndexConfig()
    manifest = build_index(
        FIXTURES / "source2.tsv",
        FIXTURES / "source3.tsv",
        tmp_path,
        config,
    )
    return open_index(manifest)


# ---------------------------------------------------------------------------
# Build / Open lifecycle
# ---------------------------------------------------------------------------

class TestBuildAndOpen:

    def test_build_returns_manifest(self, tmp_path: Path) -> None:
        manifest = build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        assert isinstance(manifest, IndexManifest)
        assert manifest.record_count == 5  # 3 S2 + 2 S3
        assert manifest.normalization_version == NORMALIZATION_VERSION
        assert manifest.index_version == INDEX_VERSION

    def test_manifest_json_written(self, tmp_path: Path) -> None:
        build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        manifest_path = tmp_path / "blocking" / "manifest.json"
        assert manifest_path.exists()
        assert manifest_path.stat().st_size > 0

    def test_index_pickle_written(self, tmp_path: Path) -> None:
        build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        index_path = tmp_path / "blocking" / "index.pkl"
        assert index_path.exists()
        assert index_path.stat().st_size > 0

    def test_open_from_manifest_object(self, tmp_path: Path) -> None:
        manifest = build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        store = open_index(manifest)
        assert isinstance(store, IndexStore)
        assert store.total_records == 5

    def test_open_from_manifest_path(self, tmp_path: Path) -> None:
        build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        manifest_path = tmp_path / "blocking" / "manifest.json"
        store = open_index(manifest_path)
        assert store.total_records == 5

    def test_open_from_blocking_dir(self, tmp_path: Path) -> None:
        build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        blocking_dir = tmp_path / "blocking"
        store = open_index(blocking_dir)
        assert store.total_records == 5

    def test_duplicate_entity_id_raises(self, tmp_path: Path) -> None:
        """Building with overlapping IDs between S2 and S3 should fail."""
        # Create a S3 file that contains an ID also present in source2.tsv
        dup_s3 = tmp_path / "dup_source3.tsv"
        dup_s3.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3-DUP\tSome Name\tSome Addr\tIndia\n"
            "S3-DUP\tAnother Name\tAnother Addr\tIndia\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate entity ID"):
            build_index(
                FIXTURES / "source2.tsv",
                dup_s3,
                tmp_path / "work",
                IndexConfig(),
            )


# ---------------------------------------------------------------------------
# Version validation
# ---------------------------------------------------------------------------

class TestVersionValidation:

    def test_rejects_normalization_version_mismatch(self, tmp_path: Path) -> None:
        manifest = build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        # Tamper with the manifest
        manifest = IndexManifest(**{
            **manifest.to_dict(),
            "normalization_version": "99",
        })
        with pytest.raises(ValueError, match="Normalization version mismatch"):
            open_index(manifest)

    def test_rejects_index_version_mismatch(self, tmp_path: Path) -> None:
        manifest = build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        manifest = IndexManifest(**{
            **manifest.to_dict(),
            "index_version": "99",
        })
        with pytest.raises(ValueError, match="Index version mismatch"):
            open_index(manifest)


# ---------------------------------------------------------------------------
# Record lookup
# ---------------------------------------------------------------------------

class TestGetRecord:

    def test_get_s2_record(self, index_store: IndexStore) -> None:
        rec = index_store.get_record("S2-MATCH")
        assert rec.raw.entity_id == "S2-MATCH"
        assert rec.raw.business_name == "Blue Lantern Bakery LLC"
        assert rec.raw.country == "United States"
        assert rec.name_norm  # non-empty normalized name

    def test_get_s3_record(self, index_store: IndexStore) -> None:
        rec = index_store.get_record("S3-MATCH")
        assert rec.raw.entity_id == "S3-MATCH"
        assert rec.raw.business_name == "Blue Lantern Bakes"

    def test_get_unicode_record(self, index_store: IndexStore) -> None:
        rec = index_store.get_record("S2-UNICODE")
        assert "さくら" in rec.raw.business_name
        assert rec.name_norm  # normalized Japanese preserved

    def test_get_france_record(self, index_store: IndexStore) -> None:
        rec = index_store.get_record("S3-FRANCE")
        assert rec.raw.country == "France"
        assert rec.country_key == "france"

    def test_get_nonexistent_raises(self, index_store: IndexStore) -> None:
        with pytest.raises(KeyError, match="S2-NONEXISTENT"):
            index_store.get_record("S2-NONEXISTENT")

    def test_get_s1_id_raises(self, index_store: IndexStore) -> None:
        """S1 IDs are not in the S2/S3 index."""
        with pytest.raises(ValueError, match="Invalid entity ID"):
            index_store.get_record("S1-MULTI")


# ---------------------------------------------------------------------------
# Inverted-index lookups
# ---------------------------------------------------------------------------

class TestInvertedIndexes:

    def test_exact_name_lookup(self, index_store: IndexStore) -> None:
        """さくら商店 is an exact name match."""
        rec = index_store.get_record("S2-UNICODE")
        ids = index_store.lookup_exact_name(rec.name_norm)
        assert "S2-UNICODE" in ids

    def test_exact_name_no_match(self, index_store: IndexStore) -> None:
        ids = index_store.lookup_exact_name("nonexistent business name")
        assert ids == []

    def test_exact_name_with_country_filter(self, index_store: IndexStore) -> None:
        rec = index_store.get_record("S2-UNICODE")
        ids_japan = index_store.lookup_exact_name(rec.name_norm, country_key="japan")
        ids_us = index_store.lookup_exact_name(rec.name_norm, country_key="united states")
        assert "S2-UNICODE" in ids_japan
        assert "S2-UNICODE" not in ids_us

    def test_token_lookup(self, index_store: IndexStore) -> None:
        """'bakery' token should match S2-MATCH and S2-HARD-NEGATIVE."""
        ids = index_store.lookup_token("bakery")
        assert "S2-MATCH" in ids
        assert "S2-HARD-NEGATIVE" in ids
        assert "S3-MATCH" not in ids  # S3-MATCH has "bakes" not "bakery"

    def test_token_posting_lists_sorted(self, index_store: IndexStore) -> None:
        ids = index_store.lookup_token("blue")
        assert ids == sorted(ids)

    def test_token_doc_freq(self, index_store: IndexStore) -> None:
        assert index_store.get_token_doc_freq("bakery") == 2
        assert index_store.get_token_doc_freq("nonexistent") == 0

    def test_country_entities(self, index_store: IndexStore) -> None:
        us_entities = index_store.get_country_entities("united states")
        assert "S2-MATCH" in us_entities
        assert "S2-HARD-NEGATIVE" in us_entities
        france_entities = index_store.get_country_entities("france")
        assert "S3-FRANCE" in france_entities

    def test_countries_list(self, index_store: IndexStore) -> None:
        countries = index_store.countries
        assert "france" in countries
        assert "japan" in countries
        assert "united states" in countries
        assert countries == sorted(countries)  # sorted

    def test_address_token_lookup(self, index_store: IndexStore) -> None:
        """Address tokens should be indexed."""
        ids = index_store.lookup_address_token("river")
        assert "S2-MATCH" in ids  # "12 River Road"


# ---------------------------------------------------------------------------
# IndexConfig serialization
# ---------------------------------------------------------------------------

class TestIndexConfig:

    def test_round_trip(self) -> None:
        config = IndexConfig(exact_name=True, name_token=False, max_candidates_per_s1=64)
        restored = IndexConfig.from_dict(config.to_dict())
        assert restored.exact_name is True
        assert restored.name_token is False
        assert restored.max_candidates_per_s1 == 64

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {"exact_name": True, "unknown_key": "value"}
        config = IndexConfig.from_dict(d)
        assert config.exact_name is True

    def test_defaults(self) -> None:
        config = IndexConfig()
        assert config.exact_name is True
        assert config.name_token is True
        assert config.address is False
        assert config.max_candidates_per_s1 == 32
