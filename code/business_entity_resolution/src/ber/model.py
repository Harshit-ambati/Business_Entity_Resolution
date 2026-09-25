"""H1 candidate-level pair model; no threshold or final decision is made here."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Protocol

from .contracts import Candidate, CandidateGroup, NormalizedRecord, TruthRow
from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, pair_features
from .split import is_validation_s1


class RecordStore(Protocol):
    def get_record(self, candidate_entity_id: str) -> NormalizedRecord: ...


@dataclass(frozen=True)
class TrainingPair:
    s1: NormalizedRecord
    candidate: NormalizedRecord
    retrieval: Candidate
    label: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if self.candidate.raw.entity_id != self.retrieval.candidate_entity_id:
            raise ValueError("pair candidate ID mismatch")


@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 2026
    negatives_per_s1: int = 4
    top_negative_pool: int = 12

    def __post_init__(self) -> None:
        if self.negatives_per_s1 < 1 or self.top_negative_pool < self.negatives_per_s1:
            raise ValueError("invalid hard-negative sampling limits")


@dataclass(frozen=True)
class PairCounts:
    training_s1_groups: int
    retrieved_positives: int
    missed_positives: int
    negative_pairs: int
    validation_groups_skipped: int


def build_training_pairs(
    rows: Iterable[tuple[NormalizedRecord, CandidateGroup, TruthRow]],
    index_store: RecordStore,
    sampling: SamplingConfig = SamplingConfig(),
) -> tuple[list[TrainingPair], PairCounts]:
    """Build a bounded caller-selected sample; never insert an unretrieved truth ID.

    Input groups must already be ranked by retrieval priority. Validation labels are
    neither read nor sampled. Caller must bound the iterable for laptop training.
    """
    pairs: list[TrainingPair] = []
    groups = positives = misses = negatives = skipped = 0
    for s1, group, truth in rows:
        sid = s1.raw.entity_id
        if sid != group.source1_entity_id or sid != truth.source1_entity_id:
            raise ValueError("S1/group/truth alignment mismatch")
        if is_validation_s1(sid):
            skipped += 1
            continue
        groups += 1
        true_ids = set(truth.matched_entity_ids)
        retrieved = {c.candidate_entity_id for c in group.candidates}
        misses += len(true_ids - retrieved)
        positive_candidates = [c for c in group.candidates if c.candidate_entity_id in true_ids]
        negative_pool = [c for c in group.candidates if c.candidate_entity_id not in true_ids][:sampling.top_negative_pool]
        # Stable per-S1 seed means adding or reordering other groups changes nothing.
        digest = sha256(f"{sampling.seed}|{sid}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        picks = set(rng.sample(range(len(negative_pool)), min(sampling.negatives_per_s1, len(negative_pool))))
        selected = positive_candidates + [c for i, c in enumerate(negative_pool) if i in picks]
        for retrieval in selected:
            record = index_store.get_record(retrieval.candidate_entity_id)
            if record is None:
                raise KeyError(retrieval.candidate_entity_id)
            label = int(retrieval.candidate_entity_id in true_ids)
            pairs.append(TrainingPair(s1, record, retrieval, label))
            positives += label
            negatives += 1 - label
    return pairs, PairCounts(groups, positives, misses, negatives, skipped)


_F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def rule_score(s1: NormalizedRecord, candidate: NormalizedRecord,
               retrieval: Candidate) -> float:
    """Temporary H1 interpretable score in [0,1]; weights are not tuned."""
    f = pair_features(s1, candidate, retrieval)
    name = .45 * f[_F["name_token_jaccard"]] + .25 * f[_F["name_token_containment"]] + .30 * f[_F["name_char_ratio"]]
    address = .5 * f[_F["address_token_jaccard"]] + .5 * f[_F["address_char_ratio"]]
    score = .75 * name + (.20 * address if not f[_F["s1_address_missing"]] and not f[_F["candidate_address_missing"]] else 0.0)
    score += .05 * f[_F["number_any_agreement"]]
    score -= .08 * f[_F["number_conflict"]]  # negative evidence, not a veto
    return max(0.0, min(1.0, score))


@dataclass(frozen=True)
class TrainConfig:
    model_dir: Path
    normalization_version: str
    index_version: str | None = None
    candidate_config_id: str | None = None
    seed: int = 2026
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    num_boost_round: int = 40
    max_training_pairs: int = 100_000


@dataclass(frozen=True)
class ModelManifest:
    model_type: str
    model_library: str
    model_library_version: str
    license: str
    training_seed: int
    feature_schema_version: str
    ordered_feature_names: tuple[str, ...]
    normalization_version: str
    index_version: str | None
    candidate_config_id: str | None
    split_definition: str
    training_positive_count: int
    training_negative_count: int
    negative_sampling_config: dict[str, int]
    model_artifact_path: str
    artifact_sha256: str
    training_timestamp_utc: str
    training_config: dict[str, int]


@dataclass(frozen=True)
class LoadedModel:
    manifest: ModelManifest
    booster: object


def _check_pairs(pairs: Iterable[TrainingPair], validation: bool) -> None:
    for pair in pairs:
        if is_validation_s1(pair.s1.raw.entity_id) != validation:
            raise ValueError("pair in wrong fixed S1 partition")


def train_model(training_pairs: Iterable[TrainingPair],
                validation_pairs: Iterable[TrainingPair] | None,
                config: TrainConfig) -> ModelManifest:
    """Fit CPU LightGBM on a caller-bounded sample; validation is checked but never fit."""
    import lightgbm as lgb
    import numpy as np

    if config.max_training_pairs < 1:
        raise ValueError("max_training_pairs must be positive")
    train = []
    for pair in training_pairs:
        if len(train) == config.max_training_pairs:
            raise ValueError("training pair budget exceeded")
        train.append(pair)
    _check_pairs(train, False)
    if validation_pairs is not None:
        _check_pairs(validation_pairs, True)
    pos = sum(pair.label for pair in train)
    neg = len(train) - pos
    if not pos or not neg:
        raise ValueError("training needs retrieved positives and negatives")
    if config.num_boost_round < 1:
        raise ValueError("num_boost_round must be positive")
    matrix = np.asarray([pair_features(p.s1, p.candidate, p.retrieval) for p in train], dtype=np.float32)
    dataset = lgb.Dataset(matrix, label=[p.label for p in train], feature_name=list(FEATURE_NAMES), free_raw_data=True)
    params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
              "seed": config.seed, "num_threads": 2, "deterministic": True,
              "force_col_wise": True, "num_leaves": 7, "min_data_in_leaf": 2,
              "learning_rate": .05, "feature_fraction": 1.0, "bagging_fraction": 1.0}
    booster = lgb.train(params, dataset, num_boost_round=config.num_boost_round)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    artifact = config.model_dir / "h1_model.txt"
    booster.save_model(str(artifact))
    checksum = sha256(artifact.read_bytes()).hexdigest()
    manifest = ModelManifest(
        "binary_gradient_boosted_trees", "lightgbm", lgb.__version__, "MIT",
        config.seed, FEATURE_SCHEMA_VERSION, FEATURE_NAMES, config.normalization_version,
        config.index_version, config.candidate_config_id,
        'sha256("2026|" + S1_ID) first 8 bytes big-endian mod 10 == 0',
        pos, neg, asdict(config.sampling), artifact.name, checksum,
        datetime.now(timezone.utc).isoformat(),
        {"num_boost_round": config.num_boost_round, "max_training_pairs": config.max_training_pairs,
         "num_threads": 2, "num_leaves": 7, "min_data_in_leaf": 2},
    )
    (config.model_dir / "h1_manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def load_model(manifest_path: Path) -> LoadedModel:
    import lightgbm as lgb

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["ordered_feature_names"] = tuple(data["ordered_feature_names"])
    manifest = ModelManifest(**data)
    if manifest.feature_schema_version != FEATURE_SCHEMA_VERSION or manifest.ordered_feature_names != FEATURE_NAMES:
        raise ValueError("feature schema/order mismatch")
    if manifest.model_type != "binary_gradient_boosted_trees" or manifest.license != "MIT":
        raise ValueError("model type/license mismatch")
    if manifest.model_library != "lightgbm" or manifest.model_library_version != lgb.__version__:
        raise ValueError("model library/version mismatch")
    artifact = manifest_path.parent / manifest.model_artifact_path
    if Path(manifest.model_artifact_path).name != manifest.model_artifact_path:
        raise ValueError("artifact path must be a filename within manifest directory")
    if sha256(artifact.read_bytes()).hexdigest() != manifest.artifact_sha256:
        raise ValueError("model artifact checksum mismatch")
    return LoadedModel(manifest, lgb.Booster(model_file=str(artifact)))


def score_group(s1: NormalizedRecord, group: CandidateGroup,
                index_store: RecordStore, model_manifest: LoadedModel,
                *, normalization_version: str,
                index_version: str | None = None,
                candidate_config_id: str | None = None) -> tuple[float, ...]:
    """Score exactly one candidate group in its original order."""
    m = model_manifest.manifest
    if group.source1_entity_id != s1.raw.entity_id:
        raise ValueError("S1 and candidate group disagree")
    if m.feature_schema_version != FEATURE_SCHEMA_VERSION or m.ordered_feature_names != FEATURE_NAMES:
        raise ValueError("feature schema/order mismatch")
    if m.normalization_version != normalization_version:
        raise ValueError("normalization version mismatch")
    if m.index_version is not None and m.index_version != index_version:
        raise ValueError("index version mismatch")
    if m.candidate_config_id is not None and m.candidate_config_id != candidate_config_id:
        raise ValueError("candidate configuration mismatch")
    if not group.candidates:
        return ()
    import numpy as np

    rows = []
    for retrieval in group.candidates:
        candidate = index_store.get_record(retrieval.candidate_entity_id)
        if candidate is None:
            raise KeyError(retrieval.candidate_entity_id)
        rows.append(list(pair_features(s1, candidate, retrieval)))
    return tuple(float(x) for x in model_manifest.booster.predict(np.asarray(rows, dtype=np.float32), num_threads=2))
