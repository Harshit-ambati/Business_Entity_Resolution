"""Contract-only H1 fixtures; no teammate production module is impersonated."""

import json
from dataclasses import replace

import pytest

from ber.contracts import Candidate, CandidateGroup, NormalizedRecord, Record, TruthRow
from ber.features import FEATURE_NAMES, pair_features
from ber.model import (SamplingConfig, TrainConfig, build_training_pairs, load_model,
                       rule_score, score_group, train_model)
from ber.split import is_validation_s1


def normalized(entity_id, name, address="", country="India"):
    # Synthetic values are supplied directly; this is not a replacement normalizer.
    return NormalizedRecord(Record(entity_id, name, address, country), name, address,
                            tuple(name.split()), tuple(address.split()), country.casefold())


def features(left, right):
    return dict(zip(FEATURE_NAMES, pair_features(left, right, Candidate(right.raw.entity_id, ("fixture",)))))


def test_feature_schema_and_variations():
    base = normalized("S1-a", "acme pvt ltd", "42 mg road")
    exact = features(base, normalized("S2-a", "acme pvt ltd", "42 mg road"))
    assert len(exact) == len(FEATURE_NAMES)
    assert exact["name_exact"] == exact["address_exact"] == 1
    assert exact["number_any_agreement"] == 1
    assert features(base, normalized("S3-b", "acme private limited"))["name_token_containment"] > 0
    reordered = features(normalized("S1-b", "abc trading company"), normalized("S2-b", "trading company abc"))
    assert reordered["name_token_jaccard"] == 1 and reordered["name_exact"] == 0
    assert features(base, normalized("S2-c", "acme pvt ltd", "91 mg road"))["number_conflict"] == 1
    assert features(base, normalized("S2-d", "acme pvt ltd", "road mg 42"))["address_token_jaccard"] == 1
    assert features(base, normalized("S2-e", "acme pvt ltd"))["candidate_address_missing"] == 1
    assert features(normalized("S1-c", "x"), normalized("S2-f", "x"))["both_addresses_missing"] == 1
    assert features(normalized("S1-d", ""), normalized("S2-g", ""))["name_exact"] == 0
    assert features(normalized("S1-e", "a&b stores"), normalized("S2-h", "a and b stores"))["name_char_ratio"] > .5
    assert features(normalized("S1-f", "acme technolgies"), normalized("S2-i", "acme technologies"))["name_char_ratio"] > .8
    assert features(normalized("S1-g", "株式会社 東京"), normalized("S2-j", "株式会社 東京"))["name_exact"] == 1
    assert features(normalized("S1-h", "café paris", country="France"), normalized("S3-k", "café paris", country="France"))["is_source3"] == 1
    hard = features(normalized("S1-i", "abc technologies", "42 mg road"), normalized("S2-l", "abc technology solutions", "91 broadway"))
    assert hard["name_token_containment"] > 0 and hard["address_token_jaccard"] == 0
    assert rule_score(base, normalized("S2-c", "acme pvt ltd", "91 mg road"), Candidate("S2-c", ("fixture",))) > 0


class Store:
    def __init__(self, records):
        self.records = records

    def get_record(self, entity_id):
        return self.records.get(entity_id)


def fixture_rows():
    rows, records = [], {}
    # Select the first 16 training IDs by the fixed S1 split, not by match ease.
    ids = [f"S1-fixture-{i}" for i in range(100) if not is_validation_s1(f"S1-fixture-{i}")][:16]
    for i, sid in enumerate(ids):
        s1 = normalized(sid, f"business {i} limited", f"{i + 1} main road")
        candidates = []
        for j, (name, address) in enumerate(((f"business {i} limited", f"{i + 1} main road"),
                                              (f"business {i} services", "91 broadway"),
                                              (f"business {i + 1} limited", "42 side street"),
                                              ("other enterprise", "75 avenue"))):
            cid = f"S2-fixture-{i}-{j}"
            records[cid] = normalized(cid, name, address)
            candidates.append(Candidate(cid, ("fixture",)))
        rows.append((s1, CandidateGroup(sid, tuple(candidates)), TruthRow(sid, (candidates[0].candidate_entity_id,))))
    return rows, Store(records)


def test_sampling_and_retrieval_miss():
    rows, store = fixture_rows()
    config = SamplingConfig(seed=73, negatives_per_s1=2, top_negative_pool=3)
    first, counts = build_training_pairs(rows, store, config)
    second, other_counts = build_training_pairs(reversed(rows), store, config)
    by_id = lambda pairs: {p.s1.raw.entity_id: sorted(p.retrieval.candidate_entity_id for p in pairs) for p in pairs}
    assert by_id(first) == by_id(second)
    assert counts == other_counts
    assert (counts.training_s1_groups, counts.retrieved_positives, counts.negative_pairs) == (16, 16, 32)
    s1, group, truth = rows[0]
    missing = TruthRow(s1.raw.entity_id, truth.matched_entity_ids + ("S3-unretrieved",))
    pairs, miss_counts = build_training_pairs([(s1, group, missing)], store, config)
    assert miss_counts.missed_positives == 1
    assert all(p.retrieval.candidate_entity_id != "S3-unretrieved" for p in pairs)
    validation_id = next(f"S1-validation-{i}" for i in range(100) if is_validation_s1(f"S1-validation-{i}"))
    validation = normalized(validation_id, "private company")
    _, skipped = build_training_pairs([(validation, CandidateGroup(validation_id, ()), TruthRow(validation_id, ("S2-secret",)))], store)
    assert skipped.validation_groups_skipped == 1 and skipped.missed_positives == 0


def test_training_manifest_reload_order_and_failures(tmp_path):
    rows, store = fixture_rows()
    pairs, counts = build_training_pairs(rows, store, SamplingConfig(seed=73, negatives_per_s1=2))
    config = TrainConfig(tmp_path, "1", "fixture-index", "fixture-candidates", seed=73,
                         sampling=SamplingConfig(seed=73, negatives_per_s1=2), num_boost_round=8)
    manifest = train_model(pairs, None, config)
    assert (tmp_path / "h1_model.txt").is_file()
    path = tmp_path / "h1_manifest.json"
    assert path.is_file() and manifest.license == "MIT" and manifest.model_library_version
    assert manifest.training_positive_count == counts.retrieved_positives
    loaded = load_model(path)
    s1, group, _ = rows[0]
    kwargs = {"normalization_version": "1", "index_version": "fixture-index", "candidate_config_id": "fixture-candidates"}
    scores = score_group(s1, group, store, loaded, **kwargs)
    assert len(scores) == len(group.candidates) and all(0 <= x <= 1 for x in scores)
    assert scores == pytest.approx(score_group(s1, group, store, load_model(path), **kwargs), abs=1e-12)
    reverse = CandidateGroup(group.source1_entity_id, tuple(reversed(group.candidates)))
    assert score_group(s1, reverse, store, loaded, **kwargs) == pytest.approx(tuple(reversed(scores)))
    with pytest.raises(ValueError, match="normalization"):
        score_group(s1, group, store, loaded, normalization_version="2", index_version="fixture-index", candidate_config_id="fixture-candidates")
    with pytest.raises(ValueError, match="index version"):
        score_group(s1, group, store, loaded, normalization_version="1", index_version="wrong", candidate_config_id="fixture-candidates")
    with pytest.raises(ValueError, match="candidate configuration"):
        score_group(s1, group, store, loaded, normalization_version="1", index_version="fixture-index", candidate_config_id="wrong")
    with pytest.raises(KeyError):
        score_group(s1, group, Store({}), loaded, **kwargs)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ordered_feature_names"] = list(reversed(data["ordered_feature_names"]))
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="feature schema/order"):
        load_model(path)
    data["ordered_feature_names"] = list(FEATURE_NAMES)
    path.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "h1_model.txt").write_bytes((tmp_path / "h1_model.txt").read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_model(path)
    validation_id = next(f"S1-val-{i}" for i in range(100) if is_validation_s1(f"S1-val-{i}"))
    with pytest.raises(ValueError, match="partition"):
        train_model([replace(pairs[0], s1=normalized(validation_id, "x"))], None, config)
