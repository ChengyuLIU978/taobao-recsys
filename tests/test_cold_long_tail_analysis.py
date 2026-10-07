"""Independent partition and denominator checks for analysis artifacts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dev"
ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "analysis"


class ColdLongTailAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.purchases = pd.read_parquet(
            ANALYSIS_DIR / "test_purchase_segments.parquet"
        )
        cls.segments = pd.read_parquet(
            ANALYSIS_DIR / "item_popularity_segments.parquet"
        )
        cls.metrics = pd.read_csv(ANALYSIS_DIR / "long_tail_metrics.csv")
        cls.distribution = pd.read_csv(
            ANALYSIS_DIR / "long_tail_target_distribution.csv"
        )
        cls.exposure = pd.read_csv(ANALYSIS_DIR / "exposure_metrics.csv")
        cls.config = json.loads(
            (ANALYSIS_DIR / "config.json").read_text(encoding="utf-8")
        )
        cls.summary = json.loads(
            (ANALYSIS_DIR / "analysis_summary.json").read_text(encoding="utf-8")
        )

    def test_cold_bucket_partition(self) -> None:
        self.assertEqual(set(self.purchases.bucket), {"A", "C"})
        expected = np.select(
            [
                self.purchases.user_seen_in_train
                & self.purchases.target_item_seen_in_train_catalog,
                ~self.purchases.user_seen_in_train
                & self.purchases.target_item_seen_in_train_catalog,
                self.purchases.user_seen_in_train
                & ~self.purchases.target_item_seen_in_train_catalog,
                ~self.purchases.user_seen_in_train
                & ~self.purchases.target_item_seen_in_train_catalog,
            ],
            ["A", "B", "C", "D"],
            default="ERROR",
        )
        self.assertTrue(np.array_equal(self.purchases.bucket.to_numpy(), expected))
        self.assertEqual(len(self.purchases), 2_512)

    def test_popularity_partition(self) -> None:
        self.assertEqual(len(self.segments), 317_721)
        self.assertFalse(self.segments.item_id.duplicated().any())
        self.assertEqual(set(self.segments.segment.astype(str)), {"Head", "Torso", "Tail"})
        expected = self.summary["segment_metadata"]
        counts = self.segments.segment.astype(str).value_counts()
        self.assertEqual(int(counts["Head"]), expected["head_items"])
        self.assertEqual(int(counts["Torso"]), expected["torso_items"])
        self.assertEqual(int(counts["Tail"]), expected["tail_items"])
        self.assertEqual(int(counts.sum()), expected["total_train_known_items"])

    def test_cold_not_tail(self) -> None:
        cold = self.purchases.loc[
            ~self.purchases.target_item_seen_in_train_catalog
        ]
        self.assertTrue(cold.segment.eq("Cold").all())
        tail = self.purchases.loc[self.purchases.segment.eq("Tail")]
        self.assertTrue(tail.target_item_seen_in_train_catalog.all())

    def test_train_only_popularity(self) -> None:
        self.assertTrue(self.config["analysis_only"])
        self.assertFalse(self.config["model_training_performed"])
        self.assertEqual(
            self.config["sources"]["train_only_popularity"],
            "data/processed/dev/train.csv",
        )
        train = pd.read_csv(DATA_DIR / "train.csv", usecols=["item_id", "timestamp"])
        actual = train.groupby("item_id").size().rename("actual")
        persisted = self.segments.set_index("item_id").train_interaction_count
        self.assertEqual(int(train.timestamp.max()), 1_512_143_999)
        self.assertTrue(actual.index.equals(persisted.index.sort_values()))
        aligned = persisted.reindex(actual.index)
        self.assertTrue(np.array_equal(actual.to_numpy(), aligned.to_numpy()))

    def test_segment_metric_denominator(self) -> None:
        target_counts = (
            self.purchases.loc[self.purchases.segment.isin(["Head", "Torso", "Tail"])]
            .groupby("segment", observed=True)
            .size()
        )
        for row in self.metrics.itertuples():
            denominator = int(target_counts[row.segment])
            self.assertEqual(int(row.target_interactions), denominator)
            for k in (10, 20, 50):
                self.assertAlmostEqual(
                    getattr(row, f"recall_at_{k}"),
                    getattr(row, f"hits_at_{k}") / denominator,
                    places=14,
                )

    def test_target_distribution_total(self) -> None:
        self.assertEqual(
            set(self.distribution.segment), {"Head", "Torso", "Tail", "Cold"}
        )
        self.assertEqual(int(self.distribution.target_interactions.sum()), 2_512)
        actual = self.purchases.segment.value_counts()
        for row in self.distribution.itertuples():
            self.assertEqual(int(row.target_interactions), int(actual[row.segment]))

    def test_exposure_partition(self) -> None:
        partition = (
            self.exposure.head_count
            + self.exposure.torso_count
            + self.exposure.tail_count
            + self.exposure.unmapped_recommendation_count
        )
        self.assertTrue(partition.eq(self.exposure.recommendation_count).all())
        self.assertTrue(self.exposure.unmapped_recommendation_count.eq(0).all())
        share_sum = (
            self.exposure.head_share
            + self.exposure.torso_share
            + self.exposure.tail_share
        )
        self.assertTrue(np.allclose(share_sum, 1.0))
        self.assertTrue(
            self.exposure.gini_coefficient_full_train_catalog.between(0, 1).all()
        )


if __name__ == "__main__":
    unittest.main()
