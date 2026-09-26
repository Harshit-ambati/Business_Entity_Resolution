"""H2 decisions and the fixed validation sweep use invented IDs only."""

from dataclasses import FrozenInstanceError

import pytest

from ber.contracts import Candidate, CandidateGroup, TruthRow
from ber.decision import DecisionConfig, ScoredGroup, decide_group, sweep_thresholds
from ber.metrics import evaluate, evaluate_candidates
from ber.split import is_validation_s1


def group(sid, *ids):
    return CandidateGroup(sid, tuple(Candidate(cid, ("fixture",)) for cid in ids))


def validation_ids(count):
    return [sid for i in range(1000) if is_validation_s1(sid := f"S1-h2-{i}")][:count]


def synthetic_holdout():
    a, b, c, d, e = validation_ids(5)
    truth = (
        TruthRow(a, ()),
        TruthRow(b, ("S2-b",)),
        TruthRow(c, ("S2-c", "S3-c")),
        TruthRow(d, ("S2-d", "S3-missing")),
        TruthRow(e, ()),
    )
    scored = (
        ScoredGroup(group(a, "S2-a"), (.44,)),
        ScoredGroup(group(b, "S2-b", "S3-bad"), (.85, .10)),
        ScoredGroup(group(c, "S3-c", "S2-c", "S2-wrong"), (.91, .81, .25)),
        ScoredGroup(group(d, "S2-d", "S3-wrong"), (.65, .15)),
        ScoredGroup(group(e), ()),
    )
    return truth, scored, {a: "US", b: "India", c: "US", d: "India", e: "US"}


def test_decide_empty_below_one_multi_equality_order_and_sources():
    sid = "S1-test"
    assert decide_group(sid, group(sid), (), DecisionConfig(.5)).matched_entity_ids == ()
    candidates = group(sid, "S3-first", "S2-second", "S2-third")
    assert decide_group(sid, candidates, (.49, .2, .3), DecisionConfig(.5)).matched_entity_ids == ()
    assert decide_group(sid, candidates, (.49, .5, .3), DecisionConfig(.5)).matched_entity_ids == ("S2-second",)
    assert decide_group(sid, candidates, (.9, .5, .1), DecisionConfig(.5)).matched_entity_ids == ("S3-first", "S2-second")
    assert decide_group(sid, candidates, (.9, .8, .7), DecisionConfig(.5)).matched_entity_ids == ("S3-first", "S2-second", "S2-third")


@pytest.mark.parametrize("threshold", [-.1, 1.1, float("nan"), float("inf"), True])
def test_bad_threshold(threshold):
    with pytest.raises(ValueError):
        DecisionConfig(threshold)


def test_alignment_and_validity_fail_loudly():
    sid = "S1-test"
    candidates = group(sid, "S2-a", "S3-b")
    with pytest.raises(ValueError, match="count mismatch"):
        decide_group(sid, candidates, (.7,), DecisionConfig(.5))
    with pytest.raises(ValueError, match="count mismatch"):
        decide_group(sid, candidates, (.7, .8, .9), DecisionConfig(.5))
    with pytest.raises(ValueError, match="S1"):
        decide_group("S1-other", candidates, (.7, .8), DecisionConfig(.5))
    with pytest.raises(ValueError, match="pair score"):
        decide_group(sid, candidates, (.7, float("nan")), DecisionConfig(.5))
    with pytest.raises(ValueError, match="Duplicate"):
        group(sid, "S2-a", "S2-a")
    with pytest.raises(FrozenInstanceError):
        DecisionConfig(.5).threshold = .7


def test_official_metric_cases_and_macro():
    ids = validation_ids(6)
    truth = [TruthRow(ids[0], ()), TruthRow(ids[1], ()), TruthRow(ids[2], ("S2-a",)),
             TruthRow(ids[3], ("S2-b", "S3-b")), TruthRow(ids[4], ("S2-c",)),
             TruthRow(ids[5], ("S2-d", "S3-d"))]
    predicted = [(ids[0], ()), (ids[1], ("S2-bad",)), (ids[2], ("S2-a",)),
                 (ids[3], ("S2-b",)), (ids[4], ("S2-c", "S3-extra")),
                 (ids[5], ("S2-d", "S3-d"))]
    result = evaluate(truth, predicted, store_per_entity=True)
    assert result.per_entity_scores[ids[0]] == 1
    assert result.per_entity_scores[ids[1]] == 0
    assert result.per_entity_scores[ids[2]] == 1
    assert result.per_entity_scores[ids[3]] == pytest.approx(5 / 6)
    assert result.per_entity_scores[ids[4]] == pytest.approx(5 / 9)
    assert result.per_entity_scores[ids[5]] == 1
    assert result.macro_f0_5 == pytest.approx((1 + 0 + 1 + 5 / 6 + 5 / 9 + 1) / 6)
    assert (result.singleton_total, result.singleton_correct) == (2, 1)


def test_sweep_finds_optimum_and_oracle_and_diagnostics():
    truth, scored, countries = synthetic_holdout()
    report = sweep_thresholds(lambda: iter(truth), lambda: iter(scored), model_name="explicit",
                              thresholds=(0, .5, .7, .8, 1), country_labels=countries)
    assert report.best_threshold == .5
    assert report.best_macro_f0_5 == pytest.approx((1 + 1 + 1 + 5 / 6 + 1) / 5)
    assert report.candidate.oracle_macro_f0_5 == report.best_macro_f0_5
    assert report.candidate.true_edge_recall == pytest.approx(4 / 5)
    assert report.candidate.complete_s1_recall == pytest.approx(2 / 3)
    assert report.candidate.s2_true_edge_recall == 1
    assert report.candidate.s3_true_edge_recall == .5
    assert report.best_evaluation.singleton_accuracy == 1
    assert report.best_evaluation.country_f0_5["US"] == 1
    assert report.best_evaluation.country_f0_5["India"] == pytest.approx(11 / 12)
    assert report.diagnostics.retrieval_miss_links == 1
    assert report.diagnostics.retrieved_rejected_true_links == 0
    assert report.diagnostics.false_emitted_links == 0
    assert report.to_json_dict()["best_threshold"] == .5


def test_oracle_exceeds_restricted_model():
    sid = validation_ids(1)[0]
    truth = (TruthRow(sid, ("S2-a", "S3-b")),)
    scored = (ScoredGroup(group(sid, "S2-a"), (.2,)),)
    oracle = evaluate_candidates(truth, [(sid, ("S2-a",))])
    model = evaluate(truth, [decide_group(sid, scored[0].group, scored[0].scores, DecisionConfig(.5))])
    assert oracle.oracle_macro_f0_5 == pytest.approx(5 / 6)
    assert model.macro_f0_5 == 0
    assert oracle.oracle_macro_f0_5 >= model.macro_f0_5


def test_error_categories_and_default_fine_grid():
    a, b, c = validation_ids(3)
    truth = (TruthRow(a, ("S2-retrieval-miss", "S3-low", "S2-near")),
             TruthRow(b, ("S2-good",)), TruthRow(c, ()))
    scored = (ScoredGroup(group(a, "S3-low", "S2-near"), (.1, .47)),
              ScoredGroup(group(b, "S2-good", "S3-false"), (.9, .8)),
              ScoredGroup(group(c, "S2-singleton-false"), (.7,)))
    report = sweep_thresholds(lambda: iter(truth), lambda: iter(scored), model_name="diagnostic",
                              thresholds=(.5,), max_examples=10,
                              feature_provider=lambda _sid, _cid: {"name_exact": 1.0})
    errors = report.diagnostics
    assert (errors.retrieval_miss_links, errors.scoring_miss_links,
            errors.decision_miss_links, errors.retrieved_rejected_true_links) == (1, 1, 1, 2)
    assert (errors.false_emitted_links, errors.singleton_false_positive_groups) == (2, 1)
    assert all(example.important_features == {"name_exact": 1.0}
               for example in errors.examples if example.candidate_entity_id)
    fixture_truth, fixture_scored, _ = synthetic_holdout()
    fine = sweep_thresholds(lambda: iter(fixture_truth), lambda: iter(fixture_scored),
                            model_name="fine")
    assert fine.best_macro_f0_5 == pytest.approx(29 / 30)
    assert any(0 < point.threshold % .05 < .05 for point in fine.thresholds)


def test_sweep_rejects_mutating_inputs_and_training_ids():
    truth, scored, _ = synthetic_holdout()
    calls = iter((scored, scored[:-1]))
    with pytest.raises(ValueError):
        sweep_thresholds(lambda: iter(truth), lambda: iter(next(calls)),
                         model_name="changing", thresholds=(.5,))
    with pytest.raises(ValueError, match="non-validation"):
        sweep_thresholds(lambda: iter((TruthRow("S1-train", ()),)),
                         lambda: iter(()), model_name="bad", thresholds=(.5,))
