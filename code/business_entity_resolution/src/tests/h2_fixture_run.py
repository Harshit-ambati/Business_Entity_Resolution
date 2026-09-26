"""Run a bounded, invented-data H2 smoke comparison; no challenge data."""

import argparse
import json
import os
import time
from pathlib import Path

from ber.contracts import Candidate, CandidateGroup, TruthRow
from ber.decision import ScoredGroup, sweep_thresholds
from ber.model import (SamplingConfig, TrainConfig, build_training_pairs, load_model,
                       rule_score, score_group, train_model)
from test_h1 import Store, fixture_rows, normalized
from test_decision import validation_ids


def validation_fixture():
    ids = validation_ids(5)
    definitions = (
        ("solo bakery", "1 east road", (), (("S2-v0", "solo market", "9 west road"),)),
        ("north tools", "2 main road", ("S2-v1",), (("S2-v1", "north tools", "2 main road"),
                                                   ("S3-v1bad", "north services", "8 side street"))),
        ("river pharma", "3 lake road", ("S2-v2", "S3-v2"), (("S2-v2", "river pharma", "3 lake road"),
                                                          ("S3-v2", "river pharma", "3 lake road"),
                                                          ("S2-v2bad", "river trading", "9 north road"))),
        ("garden supply", "4 park road", ("S2-v3", "S3-v3miss"), (("S2-v3", "garden supply", "4 park road"),)),
        ("empty studio", "5 lane", (), ()),
    )
    rows, records = [], {}
    for i, (name, address, true_ids, candidate_specs) in enumerate(definitions):
        sid = ids[i]
        s1 = normalized(sid, name, address, "US" if i % 2 == 0 else "India")
        candidates = []
        for cid, cname, caddress in candidate_specs:
            records[cid] = normalized(cid, cname, caddress, s1.raw.country)
            candidates.append(Candidate(cid, ("fixture",)))
        rows.append((s1, CandidateGroup(sid, tuple(candidates)), TruthRow(sid, true_ids)))
    return rows, Store(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_store = fixture_rows()
    pairs, counts = build_training_pairs(train_rows, train_store,
                                         SamplingConfig(seed=73, negatives_per_s1=2, top_negative_pool=3))
    model_dir = args.work_dir / "model"
    manifest = train_model(pairs, None, TrainConfig(model_dir, "fixture-1", "fixture-index",
                                                    "fixture-candidates", seed=73,
                                                    sampling=SamplingConfig(seed=73, negatives_per_s1=2,
                                                                            top_negative_pool=3),
                                                    num_boost_round=8))
    loaded = load_model(model_dir / "h1_manifest.json")
    rows, store = validation_fixture()
    truth = tuple(truth_row for _, _, truth_row in rows)
    countries = {s1.raw.entity_id: s1.raw.country for s1, _, _ in rows}
    rule, lightgbm = [], []
    for s1, group, _ in rows:
        rule.append(ScoredGroup(group, tuple(rule_score(s1, store.get_record(c.candidate_entity_id), c)
                                             for c in group.candidates)))
        lightgbm.append(ScoredGroup(group, score_group(s1, group, store, loaded,
                                                        normalization_version="fixture-1",
                                                        index_version="fixture-index",
                                                        candidate_config_id="fixture-candidates")))
    reports = {
        "rule": sweep_thresholds(lambda: iter(truth), lambda: iter(rule), model_name="rule",
                                 country_labels=countries),
        "lightgbm": sweep_thresholds(lambda: iter(truth), lambda: iter(lightgbm), model_name="lightgbm",
                                     country_labels=countries),
    }
    # Optional Windows working-set measurement; process RSS at end is not a peak.
    try:
        import psutil
        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        rss_bytes = None
    result = {
        "kind": "synthetic_fixture_only",
        "training_s1": counts.training_s1_groups,
        "training_pairs": len(pairs),
        "validation_s1": len(rows),
        "validation_pairs": sum(len(group.candidates) for _, group, _ in rows),
        "model_checksum": manifest.artifact_sha256,
        "normalization_version": "fixture-1",
        "index_version": "fixture-index",
        "candidate_config": "fixture-candidates",
        "reports": {key: value.to_json_dict() for key, value in reports.items()},
        "wall_seconds": time.perf_counter() - started,
        "rss_bytes_at_end": rss_bytes,
    }
    report_path = args.work_dir / "h2_synthetic_report.json"
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result["report_bytes"] = report_path.stat().st_size
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, indent=2))
    for key, report in reports.items():
        print(f"{key}: threshold={report.best_threshold:.2f} macro_f0_5={report.best_macro_f0_5:.6f} "
              f"oracle={report.candidate.oracle_macro_f0_5:.6f} singleton={report.best_evaluation.singleton_accuracy:.3f}")


if __name__ == "__main__":
    main()
