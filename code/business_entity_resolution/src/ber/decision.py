"""H2 score-to-link decisions and fixed-holdout threshold evaluation.

The only competition metric used here is Suresh's ``ber.metrics.evaluate``.
Factories let a caller reopen a disk-backed score stream for each threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Mapping

from .contracts import Candidate, CandidateGroup, Decision, TruthRow
from .metrics import (CandidateResult, EvaluationResult, evaluate,
                      evaluate_candidates, oracle_sanity_check)
from .split import is_validation_s1


def _probability(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return float(value)


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", _probability(self.threshold, "threshold"))


def decide_group(source1_entity_id: str, candidates: CandidateGroup,
                 pair_scores: Iterable[float], config: DecisionConfig) -> Decision:
    """Emit every candidate with score >= threshold, in candidate order."""
    if candidates.source1_entity_id != source1_entity_id:
        raise ValueError("decision S1 and candidate group disagree")
    if not isinstance(config, DecisionConfig):
        raise TypeError("config must be DecisionConfig")
    scores = tuple(pair_scores)
    if len(scores) != len(candidates.candidates):
        raise ValueError("candidate/score count mismatch")
    scores = tuple(_probability(score, "pair score") for score in scores)
    return Decision(source1_entity_id, tuple(
        candidate.candidate_entity_id
        for candidate, score in zip(candidates.candidates, scores)
        if score >= config.threshold
    ))


@dataclass(frozen=True, slots=True)
class ScoredGroup:
    group: CandidateGroup
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scores, tuple) or len(self.scores) != len(self.group.candidates):
            raise ValueError("candidate/score count mismatch")
        object.__setattr__(self, "scores", tuple(_probability(s, "pair score") for s in self.scores))


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    threshold: float
    macro_f0_5: float
    singleton_accuracy: float
    singleton_total: int
    singleton_correct: int
    false_positive_links: int
    false_negative_links: int


@dataclass(frozen=True, slots=True)
class ErrorExample:
    category: str
    source1_entity_id: str
    candidate_entity_id: str | None
    truth_status: bool | None
    score: float | None
    threshold: float
    retrieval_routes: tuple[str, ...]
    important_features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class H2Diagnostics:
    retrieval_miss_links: int
    scoring_miss_links: int
    decision_miss_links: int
    retrieved_rejected_true_links: int
    false_emitted_links: int
    singleton_false_positive_groups: int
    examples: tuple[ErrorExample, ...]
    decision_margin: float


@dataclass(frozen=True, slots=True)
class SweepReport:
    model_name: str
    candidate: CandidateResult
    thresholds: tuple[ThresholdPoint, ...]
    best_threshold: float
    best_macro_f0_5: float
    best_evaluation: EvaluationResult
    diagnostics: H2Diagnostics

    def to_json_dict(self) -> dict:
        """Produce JSON-compatible report data; unavailable slice values become null."""
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            return value
        return clean(asdict(self))


TruthFactory = Callable[[], Iterable[TruthRow]]
ScoredFactory = Callable[[], Iterable[ScoredGroup]]
FeatureProvider = Callable[[str, str], Mapping[str, float]]


def _iter_truth(factory: TruthFactory, digest: "hashlib._Hash"):
    for row in factory():
        if not is_validation_s1(row.source1_entity_id):
            raise ValueError("truth contains non-validation S1")
        digest.update(json.dumps([row.source1_entity_id, row.matched_entity_ids], separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
        yield row


def _iter_scored(factory: ScoredFactory, digest: "hashlib._Hash"):
    for item in factory():
        sid = item.group.source1_entity_id
        if not is_validation_s1(sid):
            raise ValueError("scores contain non-validation S1")
        digest.update(json.dumps([sid, [c.candidate_entity_id for c in item.group.candidates], item.scores], separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
        yield item


def _grid(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(sorted({_probability(x, "threshold") for x in values}))
    if not result:
        raise ValueError("threshold grid must be nonempty")
    return result


def _point(threshold: float, result: EvaluationResult) -> ThresholdPoint:
    return ThresholdPoint(threshold, result.macro_f0_5, result.singleton_accuracy,
                          result.singleton_total, result.singleton_correct,
                          result.false_positive_count, result.false_negative_count)


def _diagnose(truth_factory: TruthFactory, scored_factory: ScoredFactory,
              threshold: float, *, margin: float, feature_provider: FeatureProvider | None,
              max_examples: int, expected_truth: bytes, expected_scores: bytes) -> H2Diagnostics:
    # Diagnostic counts only; F0.5 continues to come from Suresh's evaluator.
    truth_digest, score_digest = hashlib.sha256(), hashlib.sha256()
    truth = {}
    for row in _iter_truth(truth_factory, truth_digest):
        if row.source1_entity_id in truth:
            raise ValueError("duplicate diagnostic truth S1")
        truth[row.source1_entity_id] = set(row.matched_entity_ids)
    seen = set()
    retrieval = scoring = decision = false_links = singleton_fp = 0
    examples: list[ErrorExample] = []

    def add(category: str, sid: str, candidate: Candidate | None, is_true: bool | None,
            score: float | None) -> None:
        if len(examples) >= max_examples:
            return
        cid = candidate.candidate_entity_id if candidate else None
        features = feature_provider(sid, cid) if feature_provider and cid else {}
        examples.append(ErrorExample(category, sid, cid, is_true, score, threshold,
                                     candidate.retrieval_routes if candidate else (), dict(features)))

    for item in _iter_scored(scored_factory, score_digest):
        sid = item.group.source1_entity_id
        if sid in seen or sid not in truth:
            raise ValueError("duplicate or unexpected diagnostic S1")
        seen.add(sid)
        true_ids = truth[sid]
        lookup = {c.candidate_entity_id: (c, s) for c, s in zip(item.group.candidates, item.scores)}
        selected = set(decide_group(sid, item.group, item.scores, DecisionConfig(threshold)).matched_entity_ids)
        for tid in sorted(true_ids - lookup.keys()):
            retrieval += 1
            add("retrieval_miss", sid, None, True, None)
        for tid in sorted((true_ids & lookup.keys()) - selected):
            candidate, score = lookup[tid]
            if score >= threshold - margin:
                decision += 1
                add("decision_sensitive_miss", sid, candidate, True, score)
            else:
                scoring += 1
                add("scoring_miss", sid, candidate, True, score)
        wrong = selected - true_ids
        false_links += len(wrong)
        if not true_ids and wrong:
            singleton_fp += 1
        for cid in sorted(wrong):
            candidate, score = lookup[cid]
            add("singleton_false_positive" if not true_ids else "false_merge",
                sid, candidate, False, score)
    if seen != truth.keys():
        raise ValueError("missing diagnostic S1 groups")
    if truth_digest.digest() != expected_truth or score_digest.digest() != expected_scores:
        raise ValueError("holdout truth/scores changed during diagnostics")
    return H2Diagnostics(retrieval, scoring, decision, scoring + decision,
                         false_links, singleton_fp, tuple(examples), margin)


def sweep_thresholds(truth_factory: TruthFactory, scored_factory: ScoredFactory,
                     *, model_name: str, thresholds: Iterable[float] | None = None,
                     country_labels: Mapping[str, str] | None = None,
                     corpus_size: int | None = None,
                     decision_margin: float = .05,
                     feature_provider: FeatureProvider | None = None,
                     max_examples: int = 20) -> SweepReport:
    """Sweep official macro F0.5 on a repeatable fixed holdout.

    With no explicit grid, run 0.05 coarse steps then 0.01 steps around its best.
    Ties choose the higher threshold. Both factories must reopen identical inputs.
    The diagnostic score/decision split uses an explicit, provisional margin.
    """
    if not model_name:
        raise ValueError("model_name is required")
    if not isinstance(decision_margin, (int, float)) or isinstance(decision_margin, bool) or not math.isfinite(decision_margin) or not 0 <= decision_margin <= 1:
        raise ValueError("decision_margin must be in [0, 1]")
    if max_examples < 0:
        raise ValueError("max_examples must be nonnegative")
    truth_digest, score_digest = hashlib.sha256(), hashlib.sha256()
    oracle = evaluate_candidates(
        _iter_truth(truth_factory, truth_digest),
        ((item.group.source1_entity_id, tuple(c.candidate_entity_id for c in item.group.candidates))
         for item in _iter_scored(scored_factory, score_digest)),
        corpus_size=corpus_size,
    )
    if oracle.evaluated_s1_count == 0:
        raise ValueError("holdout is empty")
    expected_truth, expected_scores = truth_digest.digest(), score_digest.digest()
    points: dict[float, ThresholdPoint] = {}
    evaluations: dict[float, EvaluationResult] = {}

    def run(grid: tuple[float, ...]) -> None:
        for threshold in grid:
            if threshold in points:
                continue
            td, sd = hashlib.sha256(), hashlib.sha256()
            result = evaluate(
                _iter_truth(truth_factory, td),
                (decide_group(item.group.source1_entity_id, item.group, item.scores,
                              DecisionConfig(threshold))
                 for item in _iter_scored(scored_factory, sd)),
                country_labels=dict(country_labels) if country_labels is not None else None,
            )
            if td.digest() != expected_truth or sd.digest() != expected_scores:
                raise ValueError("holdout truth/scores changed between threshold passes")
            passed, message = oracle_sanity_check(result, oracle)
            if not passed:
                raise ValueError(message)
            points[threshold] = _point(threshold, result)
            evaluations[threshold] = result

    if thresholds is None:
        run(_grid(i / 20 for i in range(21)))
        coarse_best = max(points.values(), key=lambda p: (p.macro_f0_5, p.threshold)).threshold
        run(_grid((round(coarse_best + i / 100, 2) for i in range(-5, 6)
                   if 0 <= coarse_best + i / 100 <= 1)))
    else:
        run(_grid(thresholds))
    best = max(points.values(), key=lambda p: (p.macro_f0_5, p.threshold))
    diagnostics = _diagnose(truth_factory, scored_factory, best.threshold,
                            margin=float(decision_margin), feature_provider=feature_provider,
                            max_examples=max_examples, expected_truth=expected_truth,
                            expected_scores=expected_scores)
    return SweepReport(model_name, oracle, tuple(points[t] for t in sorted(points)),
                       best.threshold, best.macro_f0_5, evaluations[best.threshold], diagnostics)
