"""Tests for multi-route candidate retrieval, deduplication, cap, and ordering.

Uses the existing synthetic fixtures:
- S1-MULTI  → matches S2-MATCH and S3-MATCH (two matches, S2+S3, same US)
- S1-SINGLE → no matches (singleton, empty address)
- S1-FRANCE → matches S3-FRANCE (France, accented name)
- S1-UNICODE→ matches S2-UNICODE (Japanese text)
- S2-HARD-NEGATIVE → similar name to S1-MULTI but different entity
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ber.blocking import COMMON_BUSINESS_TERMS, iter_candidates
from ber.contracts import CandidateGroup
from ber.index import IndexConfig, IndexStore, build_index, open_index

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared fixture: pre-built index
# ---------------------------------------------------------------------------

@pytest.fixture
def index_store(tmp_path: Path) -> IndexStore:
    config = IndexConfig()
    manifest = build_index(
        FIXTURES / "source2.tsv",
        FIXTURES / "source3.tsv",
        tmp_path,
        config,
    )
    return open_index(manifest)


def _collect_groups(
    index_store: IndexStore,
    config: IndexConfig | None = None,
) -> list[CandidateGroup]:
    """Run iter_candidates on the fixture and collect all groups."""
    return list(iter_candidates(
        str(FIXTURES / "source1.tsv"),
        index_store,
        config or IndexConfig(),
    ))


# ---------------------------------------------------------------------------
# Basic coverage
# ---------------------------------------------------------------------------

class TestBasicCoverage:

    def test_one_group_per_s1(self, index_store: IndexStore) -> None:
        """iter_candidates yields exactly one CandidateGroup per S1 record."""
        groups = _collect_groups(index_store)
        assert len(groups) == 4  # 4 S1 records in fixture

    def test_input_order_preserved(self, index_store: IndexStore) -> None:
        """CandidateGroups appear in S1 input order."""
        groups = _collect_groups(index_store)
        s1_ids = [g.source1_entity_id for g in groups]
        assert s1_ids == ["S1-MULTI", "S1-SINGLE", "S1-FRANCE", "S1-UNICODE"]

    def test_all_candidate_ids_valid(self, index_store: IndexStore) -> None:
        """Every candidate ID starts with S2- or S3-."""
        groups = _collect_groups(index_store)
        for group in groups:
            for c in group.candidates:
                assert c.candidate_entity_id.startswith("S2-") or \
                       c.candidate_entity_id.startswith("S3-"), \
                    f"Invalid candidate ID: {c.candidate_entity_id}"


# ---------------------------------------------------------------------------
# Route A: exact name matching
# ---------------------------------------------------------------------------

class TestRouteExactName:

    def test_exact_unicode_match(self, index_store: IndexStore) -> None:
        """S1-UNICODE (さくら商店) should find S2-UNICODE via exact name."""
        groups = _collect_groups(index_store)
        unicode_group = [g for g in groups if g.source1_entity_id == "S1-UNICODE"][0]
        candidate_ids = {c.candidate_entity_id for c in unicode_group.candidates}
        assert "S2-UNICODE" in candidate_ids

    def test_exact_route_label(self, index_store: IndexStore) -> None:
        """Candidates from exact-name route have 'exact_name' in routes."""
        groups = _collect_groups(index_store)
        unicode_group = [g for g in groups if g.source1_entity_id == "S1-UNICODE"][0]
        unicode_candidate = [
            c for c in unicode_group.candidates if c.candidate_entity_id == "S2-UNICODE"
        ][0]
        assert "exact_name" in unicode_candidate.retrieval_routes

    def test_exact_name_only_returns_nothing_for_fuzzy_names(
        self, index_store: IndexStore,
    ) -> None:
        """S1-MULTI should NOT match S2-MATCH via exact name (different suffix)."""
        config = IndexConfig(exact_name=True, name_token=False)
        groups = _collect_groups(index_store, config)
        multi_group = [g for g in groups if g.source1_entity_id == "S1-MULTI"][0]
        # "blue lantern bakery" != "blue lantern bakery llc"
        candidate_ids = {c.candidate_entity_id for c in multi_group.candidates}
        assert "S2-MATCH" not in candidate_ids


# ---------------------------------------------------------------------------
# Route B: name token matching
# ---------------------------------------------------------------------------

class TestRouteNameToken:

    def test_token_route_finds_multi_match(self, index_store: IndexStore) -> None:
        """S1-MULTI should find S2-MATCH and S3-MATCH via shared tokens."""
        groups = _collect_groups(index_store)
        multi_group = [g for g in groups if g.source1_entity_id == "S1-MULTI"][0]
        candidate_ids = {c.candidate_entity_id for c in multi_group.candidates}
        assert "S2-MATCH" in candidate_ids
        assert "S3-MATCH" in candidate_ids

    def test_hard_negative_is_candidate(self, index_store: IndexStore) -> None:
        """S2-HARD-NEGATIVE shares name tokens with S1-MULTI and IS a candidate.

        The blocking system is designed for high recall — it's the downstream
        model's job to reject hard negatives, not the blocker's.
        """
        groups = _collect_groups(index_store)
        multi_group = [g for g in groups if g.source1_entity_id == "S1-MULTI"][0]
        candidate_ids = {c.candidate_entity_id for c in multi_group.candidates}
        assert "S2-HARD-NEGATIVE" in candidate_ids

    def test_france_token_match(self, index_store: IndexStore) -> None:
        """S1-FRANCE should find S3-FRANCE via shared name tokens."""
        groups = _collect_groups(index_store)
        france_group = [g for g in groups if g.source1_entity_id == "S1-FRANCE"][0]
        candidate_ids = {c.candidate_entity_id for c in france_group.candidates}
        assert "S3-FRANCE" in candidate_ids

    def test_token_route_label(self, index_store: IndexStore) -> None:
        """Candidates from token route have 'name_token' in routes."""
        groups = _collect_groups(index_store)
        multi_group = [g for g in groups if g.source1_entity_id == "S1-MULTI"][0]
        s2_match = [c for c in multi_group.candidates if c.candidate_entity_id == "S2-MATCH"][0]
        assert "name_token" in s2_match.retrieval_routes

    def test_common_terms_excluded(self) -> None:
        """Known legal suffixes and generic terms are in the stopword set."""
        assert "llc" in COMMON_BUSINESS_TERMS
        assert "ltd" in COMMON_BUSINESS_TERMS
        assert "inc" in COMMON_BUSINESS_TERMS
        assert "services" in COMMON_BUSINESS_TERMS
        assert "hotel" in COMMON_BUSINESS_TERMS


# ---------------------------------------------------------------------------
# Singleton / empty-match handling
# ---------------------------------------------------------------------------

class TestSingleton:

    def test_singleton_has_empty_or_few_candidates(self, index_store: IndexStore) -> None:
        """S1-SINGLE (no matches) may have an empty candidate group."""
        groups = _collect_groups(index_store)
        single_group = [g for g in groups if g.source1_entity_id == "S1-SINGLE"][0]
        # S1-SINGLE name "Quiet Birch Studio" shares no tokens with any S2/S3
        assert len(single_group.candidates) == 0

    def test_empty_address_handled(self, index_store: IndexStore) -> None:
        """S1-SINGLE has an empty address and does not crash."""
        groups = _collect_groups(index_store)
        single_group = [g for g in groups if g.source1_entity_id == "S1-SINGLE"][0]
        # Just verifying no exception was raised
        assert isinstance(single_group, CandidateGroup)


# ---------------------------------------------------------------------------
# Deduplication and route merging
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_candidates_are_deduplicated(self, index_store: IndexStore) -> None:
        """No duplicate candidate entity IDs within a group."""
        groups = _collect_groups(index_store)
        for group in groups:
            ids = [c.candidate_entity_id for c in group.candidates]
            assert len(ids) == len(set(ids)), f"Duplicates in {group.source1_entity_id}"

    def test_multi_route_candidate_has_all_routes(self, index_store: IndexStore) -> None:
        """S2-UNICODE is found by both exact_name and name_token; both listed."""
        groups = _collect_groups(index_store)
        unicode_group = [g for g in groups if g.source1_entity_id == "S1-UNICODE"][0]
        unicode_candidate = [
            c for c in unicode_group.candidates if c.candidate_entity_id == "S2-UNICODE"
        ][0]
        assert "exact_name" in unicode_candidate.retrieval_routes
        assert "name_token" in unicode_candidate.retrieval_routes

    def test_route_labels_are_sorted(self, index_store: IndexStore) -> None:
        """retrieval_routes must be a sorted tuple (contract requirement)."""
        groups = _collect_groups(index_store)
        for group in groups:
            for c in group.candidates:
                assert c.retrieval_routes == tuple(sorted(c.retrieval_routes))

    def test_route_scores_keys_match_routes(self, index_store: IndexStore) -> None:
        """route_scores keys are a subset of retrieval_routes."""
        groups = _collect_groups(index_store)
        for group in groups:
            for c in group.candidates:
                for key in c.route_scores:
                    assert key in c.retrieval_routes, \
                        f"score key {key!r} not in routes {c.retrieval_routes}"


# ---------------------------------------------------------------------------
# Candidate cap
# ---------------------------------------------------------------------------

class TestCap:

    def test_cap_applied(self, index_store: IndexStore) -> None:
        """Candidate count does not exceed max_candidates_per_s1."""
        config = IndexConfig(max_candidates_per_s1=2)
        groups = _collect_groups(index_store, config)
        for group in groups:
            assert len(group.candidates) <= 2

    def test_cap_1(self, index_store: IndexStore) -> None:
        """Cap of 1 keeps at most 1 candidate."""
        config = IndexConfig(max_candidates_per_s1=1)
        groups = _collect_groups(index_store, config)
        multi_group = [g for g in groups if g.source1_entity_id == "S1-MULTI"][0]
        assert len(multi_group.candidates) <= 1

    def test_cap_is_after_union(self, index_store: IndexStore) -> None:
        """Cap is applied after route union, not per-route."""
        config = IndexConfig(max_candidates_per_s1=32)
        groups = _collect_groups(index_store, config)
        multi_group = [g for g in groups if g.source1_entity_id == "S1-MULTI"][0]
        # S1-MULTI should have 3 candidates: S2-MATCH, S2-HARD-NEGATIVE, S3-MATCH
        assert len(multi_group.candidates) == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_deterministic_output(self, index_store: IndexStore) -> None:
        """Running twice produces identical candidate groups."""
        groups1 = _collect_groups(index_store)
        groups2 = _collect_groups(index_store)

        assert len(groups1) == len(groups2)
        for g1, g2 in zip(groups1, groups2):
            assert g1.source1_entity_id == g2.source1_entity_id
            assert len(g1.candidates) == len(g2.candidates)
            for c1, c2 in zip(g1.candidates, g2.candidates):
                assert c1.candidate_entity_id == c2.candidate_entity_id
                assert c1.retrieval_routes == c2.retrieval_routes
                assert dict(c1.route_scores) == dict(c2.route_scores)


# ---------------------------------------------------------------------------
# Route toggling
# ---------------------------------------------------------------------------

class TestRouteToggling:

    def test_disable_all_routes_gives_empty_groups(self, index_store: IndexStore) -> None:
        """With all routes disabled, every group is empty."""
        config = IndexConfig(exact_name=False, name_token=False)
        groups = _collect_groups(index_store, config)
        for group in groups:
            assert len(group.candidates) == 0

    def test_only_exact_name(self, index_store: IndexStore) -> None:
        """With only exact_name enabled, only exact matches appear."""
        config = IndexConfig(exact_name=True, name_token=False)
        groups = _collect_groups(index_store, config)
        # S1-UNICODE matches S2-UNICODE exactly
        unicode_group = [g for g in groups if g.source1_entity_id == "S1-UNICODE"][0]
        assert any(c.candidate_entity_id == "S2-UNICODE" for c in unicode_group.candidates)
        # S1-MULTI has no exact match
        multi_group = [g for g in groups if g.source1_entity_id == "S1-MULTI"][0]
        assert len(multi_group.candidates) == 0

    def test_only_name_token(self, index_store: IndexStore) -> None:
        """With only name_token enabled, token matches appear."""
        config = IndexConfig(exact_name=False, name_token=True)
        groups = _collect_groups(index_store, config)
        multi_group = [g for g in groups if g.source1_entity_id == "S1-MULTI"][0]
        candidate_ids = {c.candidate_entity_id for c in multi_group.candidates}
        assert "S2-MATCH" in candidate_ids


# ---------------------------------------------------------------------------
# Config via dict
# ---------------------------------------------------------------------------

class TestDictConfig:

    def test_dict_config_accepted(self, index_store: IndexStore) -> None:
        """iter_candidates accepts a plain dict as config."""
        groups = list(iter_candidates(
            str(FIXTURES / "source1.tsv"),
            index_store,
            {"exact_name": True, "name_token": True, "max_candidates_per_s1": 10},
        ))
        assert len(groups) == 4

    def test_none_config_uses_defaults(self, index_store: IndexStore) -> None:
        """Passing None uses default IndexConfig."""
        groups = list(iter_candidates(
            str(FIXTURES / "source1.tsv"),
            index_store,
            None,
        ))
        assert len(groups) == 4


# ---------------------------------------------------------------------------
# Benchmark helper test
# ---------------------------------------------------------------------------

class TestBenchmarkHelper:

    def test_run_benchmark_fixtures(self, tmp_path: Path) -> None:
        """Benchmark helper runs cleanly on synthetic fixtures and returns stats."""
        from ber.benchmark import run_benchmark

        results = run_benchmark(
            source1_path=FIXTURES / "source1.tsv",
            source2_path=FIXTURES / "source2.tsv",
            source3_path=FIXTURES / "source3.tsv",
            truth_path=FIXTURES / "truth.tsv",
            work_dir=tmp_path / "bench_idx",
        )
        assert results["num_s1_queries"] == 4
        assert results["total_records"] == 5
        assert results["true_edge_recall"] == 1.0
        assert results["complete_link_coverage"] == 1.0
        assert results["reduction_ratio"] > 0.0

    def test_iter_candidates_split_mismatch_raises(self, tmp_path: Path) -> None:
        """iter_candidates raises ValueError when query S1 split mismatches index split."""
        # Create an index with split="train"
        manifest = build_index(
            FIXTURES / "source2.tsv",
            FIXTURES / "source3.tsv",
            tmp_path / "train_work",
            IndexConfig(split="train"),
        )
        store = open_index(manifest)

        # Create a query file under a test/ folder
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        test_s1 = test_dir / "source1.tsv"
        test_s1.write_text((FIXTURES / "source1.tsv").read_text(encoding="utf-8"), encoding="utf-8")

        with pytest.raises(ValueError, match="Split mismatch"):
            list(iter_candidates(test_s1, store))
