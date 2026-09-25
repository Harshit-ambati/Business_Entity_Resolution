"""Run a bounded invented-data H1 smoke experiment; never use challenge rows."""

import argparse
import json
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

from ber.contracts import TruthRow
from ber.features import FEATURE_SCHEMA_VERSION
from ber.model import SamplingConfig, TrainConfig, build_training_pairs, train_model
from test_h1 import fixture_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, store = fixture_rows()
    # One intentionally missed synthetic truth ID tests retrieval-miss accounting.
    s1, group, truth = rows[0]
    rows[0] = (s1, group, TruthRow(truth.source1_entity_id, truth.matched_entity_ids + ("S3-unretrieved",)))
    sampling = SamplingConfig(seed=73, negatives_per_s1=2, top_negative_pool=3)
    start = time.perf_counter()
    tracemalloc.start()
    pairs, counts = build_training_pairs(rows, store, sampling)
    config = TrainConfig(args.work_dir, "1", "synthetic-only", "synthetic-only", seed=73,
                         sampling=sampling, num_boost_round=8)
    manifest = train_model(pairs, None, config)
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    try:
        import psutil  # optional measurement only; not an H1 runtime dependency
        peak_ram_bytes = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except ImportError:
        peak_ram_bytes = None
    report = {
        "data": "invented synthetic records only", "subset": "first 16 fixture S1 IDs passing fixed training split",
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "counts": asdict(counts),
        "seed": 73, "sampling": asdict(sampling), "model_library_version": manifest.model_library_version,
        "license": manifest.license, "artifact_sha256": manifest.artifact_sha256,
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "peak_python_traced_bytes": peak_python_bytes,
        "peak_ram_bytes": peak_ram_bytes,
        "metric": "none; synthetic fixture is not a competition holdout",
    }
    (args.work_dir / "h1_fixture_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
