"""Independent checks for persisted Ranking + Recall -> Rank artifacts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dev"
RANKING_DIR = PROJECT_ROOT / "artifacts" / "ranking"


class RankingArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (RANKING_DIR / "summary.json").read_text(encoding="utf-8")
        )
        cls.config = json.loads(
            (RANKING_DIR / "config.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(
            (RANKING_DIR / "feature_manifest.json").read_text(encoding="utf-8")
        )

    def test_required_artifacts_and_model_reload(self) -> None:
        required = [
            "validation_ranking_dataset.parquet",
            "test_ranking_dataset.parquet",
            "feature_manifest.json",
            "config.json",
            "ranker.txt",
            "ranker.pkl",
            "test_metrics.csv",
            "test_recommendations.parquet",
            "feature_importance.csv",
            "summary.json",
            "ablation_metrics.csv",
        ]
        self.assertTrue(all((RANKING_DIR / name).is_file() for name in required))
        model = joblib.load(RANKING_DIR / "ranker.pkl")
        features = [record["name"] for record in self.manifest["features"]]
        sample = pd.read_parquet(
            RANKING_DIR / "test_ranking_dataset.parquet", columns=features
        ).head(100)
        prediction = model.predict(sample)
        self.assertEqual(prediction.shape, (100,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_feature_manifest_excludes_identifiers_and_labels(self) -> None:
        records = self.manifest["features"]
        features = [record["name"] for record in records]
        self.assertEqual(len(features), 52)
        self.assertEqual(len(features), len(set(features)))
        self.assertFalse({"user_id", "item_id", "label", "ranking_score"} & set(features))
        aggregate_records = [
            record for record in records if record["source"] == "train.csv aggregate"
        ]
        self.assertTrue(aggregate_records)
        self.assertTrue(all(record["train_only"] for record in aggregate_records))

    def test_time_boundary_and_label_access_contract(self) -> None:
        train = pd.read_csv(DATA_DIR / "train.csv", usecols=["timestamp"])
        validation = pd.read_csv(DATA_DIR / "validation.csv", usecols=["timestamp"])
        test = pd.read_csv(DATA_DIR / "test.csv", usecols=["timestamp"])
        self.assertLess(int(train.timestamp.max()), int(validation.timestamp.min()))
        self.assertLess(int(validation.timestamp.max()), int(test.timestamp.min()))
        self.assertEqual(self.config["training_period_for_ranker"], "validation")
        self.assertEqual(self.config["final_evaluation_period"], "test")
        self.assertTrue(all(self.summary["leakage_checks"].values()))

    def test_group_alignment_and_candidate_uniqueness(self) -> None:
        training = self.summary["training"]
        self.assertEqual(training["group_sum"], training["training_rows"])
        self.assertTrue(training["all_training_queries_have_positive"])
        for split, expected_rows in (("validation", 3_304_863), ("test", 3_294_775)):
            path = RANKING_DIR / f"{split}_ranking_dataset.parquet"
            self.assertEqual(pq.ParquetFile(path).metadata.num_rows, expected_rows)
            pairs = pd.read_parquet(path, columns=["user_id", "item_id"])
            self.assertFalse(pairs.duplicated(["user_id", "item_id"]).any())

    def test_persisted_labels_metrics_and_topk(self) -> None:
        validation_labels = pd.read_parquet(
            RANKING_DIR / "validation_ranking_dataset.parquet", columns=["label"]
        ).label
        test_labels = pd.read_parquet(
            RANKING_DIR / "test_ranking_dataset.parquet", columns=["label"]
        ).label
        self.assertEqual(set(validation_labels.unique()), {0, 1})
        self.assertEqual(set(test_labels.unique()), {0, 1})
        self.assertEqual(int(validation_labels.sum()), 659)
        self.assertEqual(int(test_labels.sum()), 413)

        metrics = pd.read_csv(RANKING_DIR / "test_metrics.csv")
        self.assertEqual(
            set(metrics.Method),
            {"Popularity", "Two-Tower", "ItemCF", "RRF", "ItemCF-first", "Ranker"},
        )
        self.assertEqual(set(metrics.K), {10, 20, 50})
        self.assertTrue(metrics[["Recall", "HitRate", "NDCG"]].notna().all().all())
        self.assertTrue(all(self.summary["baseline_reproduction_checks"].values()))

        recommendations = pd.read_parquet(
            RANKING_DIR / "test_recommendations.parquet",
            columns=["user_id", "item_id", "rank"],
        )
        self.assertFalse(recommendations.duplicated(["user_id", "item_id"]).any())
        self.assertLessEqual(int(recommendations["rank"].max()), 50)
        self.assertEqual(int(recommendations.groupby("user_id").size().max()), 50)


if __name__ == "__main__":
    unittest.main()
