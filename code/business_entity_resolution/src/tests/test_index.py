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
    NORMALIZATION_VERSION,
    build_index,
    open_index,
)

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


# ---------------------------------------------------------------------------
# Stale index and source validation
# ---------------------------------------------------------------------------

class TestStaleIndexValidation:

    def test_detects_source_content_modified(self, tmp_path: Path) -> None:
        """When an indexed source file content changes, open_index must refuse it."""
        s2 = tmp_path / "source2.tsv"
        s3 = tmp_path / "source3.tsv"
        s2.write_text((FIXTURES / "source2.tsv").read_text(encoding="utf-8"), encoding="utf-8")
        s3.write_text((FIXTURES / "source3.tsv").read_text(encoding="utf-8"), encoding="utf-8")

        manifest = build_index(s2, s3, tmp_path / "work", IndexConfig())

        # Modify source2 content without changing length (replace character)
        content = s2.read_text(encoding="utf-8")
        s2.write_text(content.replace("Blue", "Red_"), encoding="utf-8")

        with pytest.raises(ValueError, match="Stale index: source file .* content changed"):
            open_index(manifest)

    def test_detects_source_size_modified(self, tmp_path: Path) -> None:
        """When an indexed source file size changes, open_index must refuse it."""
        s2 = tmp_path / "source2.tsv"
        s3 = tmp_path / "source3.tsv"
        s2.write_text((FIXTURES / "source2.tsv").read_text(encoding="utf-8"), encoding="utf-8")
        s3.write_text((FIXTURES / "source3.tsv").read_text(encoding="utf-8"), encoding="utf-8")

        manifest = build_index(s2, s3, tmp_path / "work", IndexConfig())

        # Append row to source3
        with s3.open("a", encoding="utf-8") as f:
            f.write("S3-EXTRA\tExtra Business\t100 Main\tIndia\n")

        with pytest.raises(ValueError, match="Stale index: source file .* size changed"):
            open_index(manifest)

    def test_detects_source_file_missing(self, tmp_path: Path) -> None:
        """When an indexed source file is deleted, open_index raises FileNotFoundError."""
        s2 = tmp_path / "source2.tsv"
        s3 = tmp_path / "source3.tsv"
        s2.write_text((FIXTURES / "source2.tsv").read_text(encoding="utf-8"), encoding="utf-8")
        s3.write_text((FIXTURES / "source3.tsv").read_text(encoding="utf-8"), encoding="utf-8")

        manifest = build_index(s2, s3, tmp_path / "work", IndexConfig())
        s2.unlink()

        with pytest.raises(FileNotFoundError, match="Stale index: indexed source file .* not found"):
            open_index(manifest)

    def test_open_index_expected_split_validation(self, tmp_path: Path) -> None:
        """open_index rejects opening an index with the wrong expected split."""
        manifest = build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(split="train"),
        )
        assert manifest.split == "train"

        # Matching expected_split passes
        store = open_index(manifest, expected_split="train")
        assert store.total_records == 5

        # Mismatched expected_split fails
        with pytest.raises(ValueError, match="Index split mismatch"):
            open_index(manifest, expected_split="test")

    def test_build_index_mixed_splits_raises(self, tmp_path: Path) -> None:
        """build_index refuses to index source files from conflicting splits."""
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()

        train_s2 = train_dir / "source2.tsv"
        test_s3 = test_dir / "source3.tsv"
        train_s2.write_text((FIXTURES / "source2.tsv").read_text(encoding="utf-8"), encoding="utf-8")
        test_s3.write_text((FIXTURES / "source3.tsv").read_text(encoding="utf-8"), encoding="utf-8")

        with pytest.raises(ValueError, match="Cannot build index with mixed source splits"):
            build_index(train_s2, test_s3, tmp_path / "work", IndexConfig())

    def test_disk_backed_records_db_created(self, tmp_path: Path) -> None:
        """build_index creates records.db SQLite file and open_index reads from it."""
        manifest = build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        db_path = tmp_path / "blocking" / "records.db"
        assert db_path.exists()
        assert db_path.stat().st_size > 0

        # open_index reads via SQLite
        store = open_index(manifest)
        rec = store.get_record("S2-MATCH")
        assert rec.raw.entity_id == "S2-MATCH"
        assert rec.name_norm == "blue lantern bakery llc"
        assert store.get_record("S3-FRANCE").country_key == "france"


# ---------------------------------------------------------------------------
# Normalization version and Unicode safety
# ---------------------------------------------------------------------------

class TestNormalizationVersionSafety:

    def test_telugu_keys_preserve_combining_marks(self) -> None:
        """Telugu vowel signs (Mc) and viramas (Mn) must survive normalization intact."""
        from ber.contracts import Record
        from ber.normalize_shim import normalize_record as shim_norm

        raw = Record(
            entity_id="S2-TELUGU",
            business_name="భారత్ ఎలక్ట్రానిక్స్",
            business_address="హైదరాబాద్ 500001",
            country="India",
        )
        norm = shim_norm(raw)

        # Must not be broken into disjoint consonants
        assert norm.name_norm == "భారత్ ఎలక్ట్రానిక్స్"
        assert norm.name_tokens == ("భారత్", "ఎలక్ట్రానిక్స్")
        assert "హైదరాబాద్" in norm.address_tokens

    def test_temporary_normalizer_version_label(self) -> None:
        """Temporary normalizer shim exports '1-shim' so it cannot masquerade as canonical v1."""
        from ber.normalize_shim import NORMALIZATION_VERSION as SHIM_VERSION
        assert SHIM_VERSION == "1-shim"

    def test_index_rejects_incompatible_normalizer_version(self, tmp_path: Path) -> None:
        """An index built with '1-shim' cannot reopen under canonical '1' without rebuild."""
        manifest = build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path,
            IndexConfig(),
        )
        # If manifest was built with "1-shim" and runtime has canonical "1":
        tampered = IndexManifest(**{
            **manifest.to_dict(),
            "normalization_version": "1",  # simulate runtime expecting "1" while index had "1-shim"
        })
        if NORMALIZATION_VERSION == "1-shim":
            with pytest.raises(ValueError, match="Normalization version mismatch"):
                open_index(tampered)
