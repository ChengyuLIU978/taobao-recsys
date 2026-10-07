"""Thin, restart-safe entry point for the existing Recall -> Rank pipeline.

Usage from the project root:

    python scripts/run_end_to_end.py --reuse-retrievers

This entry point never retrains a Two-Tower.  It reuses complete recall and
ranking artifacts, builds a missing recall stage with script 07, and builds a
missing ranking stage with script 08.  A partially present stage is treated as
an error rather than overwritten.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECALL_DIR = PROJECT_ROOT / "artifacts" / "multistage_recall"
RANKING_DIR = PROJECT_ROOT / "artifacts" / "ranking"

RECALL_REQUIRED = {
    "validation_candidates.parquet",
    "test_candidates.parquet",
    "validation_metrics.json",
    "test_metrics.json",
    "config.json",
}
RANKING_REQUIRED = {
    "validation_ranking_dataset.parquet",
    "test_ranking_dataset.parquet",
    "feature_manifest.json",
    "config.json",
    "ranker.pkl",
    "test_metrics.csv",
    "test_recommendations.parquet",
    "feature_importance.csv",
    "summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-retrievers",
        action="store_true",
        help="reuse the existing ItemCF, Two-Tower checkpoint and Popularity retrievers",
    )
    return parser.parse_args()


def missing_files(directory: Path, names: set[str]) -> list[str]:
    return sorted(name for name in names if not (directory / name).is_file())


def run_stage(script_name: str) -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def ensure_stage(directory: Path, required: set[str], script_name: str) -> None:
    missing = missing_files(directory, required)
    if not missing:
        print(f"Reusing complete stage: {directory.relative_to(PROJECT_ROOT)}")
        return
    if directory.exists():
        raise RuntimeError(
            f"Refusing to overwrite partial stage {directory}; missing files: {missing}"
        )
    print(f"Building missing stage with scripts/{script_name}")
    run_stage(script_name)
    remaining = missing_files(directory, required)
    if remaining:
        raise RuntimeError(f"Stage did not produce required files: {remaining}")


def validate_and_report() -> None:
    summary = json.loads(
        (RANKING_DIR / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "completed":
        raise AssertionError("Ranking summary does not report completed status")
    if not all(summary["leakage_checks"].values()):
        raise AssertionError("A persisted leakage check is false")
    if not all(summary["baseline_reproduction_checks"].values()):
        raise AssertionError("A persisted baseline reproduction check is false")

    model = joblib.load(RANKING_DIR / "ranker.pkl")
    manifest = json.loads(
        (RANKING_DIR / "feature_manifest.json").read_text(encoding="utf-8")
    )
    features = [record["name"] for record in manifest["features"]]
    sample = pd.read_parquet(
        RANKING_DIR / "test_ranking_dataset.parquet", columns=features
    ).head(10)
    predictions = model.predict(sample)
    if len(predictions) != 10:
        raise AssertionError("Reloaded model did not score the expected rows")

    validation_rows = pq.ParquetFile(
        RANKING_DIR / "validation_ranking_dataset.parquet"
    ).metadata.num_rows
    test_rows = pq.ParquetFile(
        RANKING_DIR / "test_ranking_dataset.parquet"
    ).metadata.num_rows
    metrics = pd.read_csv(RANKING_DIR / "test_metrics.csv")
    ranker50 = metrics.loc[metrics.Method.eq("Ranker") & metrics.K.eq(50)].iloc[0]
    print("Recall -> Rank pipeline is complete and reloadable.")
    print(f"  validation ranking rows: {validation_rows:,}")
    print(f"  test ranking rows: {test_rows:,}")
    print(f"  Ranker Recall@50: {ranker50.Recall:.6f}")
    print(f"  Ranker NDCG@50: {ranker50.NDCG:.6f}")
    print(
        "  Candidate oracle Recall: "
        f"{summary['test_evaluation']['candidate_oracle_warm']['Recall']:.6f}"
    )


def main() -> None:
    args = parse_args()
    if not args.reuse_retrievers:
        raise SystemExit(
            "This project protects trained retrievers. Rerun with --reuse-retrievers."
        )
    ensure_stage(RECALL_DIR, RECALL_REQUIRED, "07_build_multistage_recall.py")
    ensure_stage(RANKING_DIR, RANKING_REQUIRED, "08_train_ranker.py")
    validate_and_report()


if __name__ == "__main__":
    main()
