from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "15_train_sequence_two_tower.py"
SPEC = importlib.util.spec_from_file_location("v3_sequence", SCRIPT)
v3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v3
SPEC.loader.exec_module(v3)


def timestamp(day: str, hour: int = 0) -> int:
    return int(pd.Timestamp(f"{day} {hour:02d}:00:00", tz="Asia/Shanghai").timestamp())


def frame() -> pd.DataFrame:
    rows = [
        (1, 10, 1, "pv", timestamp("2017-01-01", 1)),
        (1, 11, 1, "buy", timestamp("2017-01-01", 2)),
        (2, 12, 1, "pv", timestamp("2017-01-01", 1)),
        (2, 13, 1, "buy", timestamp("2017-01-01", 2)),
        (1, 10, 1, "pv", timestamp("2017-01-02", 1)),
        (1, 11, 1, "buy", timestamp("2017-01-02", 2)),
        (1, 10, 1, "pv", timestamp("2017-01-03", 1)),
        (1, 11, 1, "buy", timestamp("2017-01-03", 2)),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "category_id", "behavior_type", "timestamp"])


class SequenceTwoTowerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = frame()
        self.train, self.select, self.evaluation, _ = v3.build_temporal_split(self.raw)
        self.user_map, self.item_map, self.user_to_idx, self.item_to_idx = v3.stable_mappings(self.train)

    def test_01_temporal_split_has_no_overlap(self) -> None:
        self.assertFalse(set(self.train.index) & set(self.select.index))
        self.assertFalse(set(self.train.index) & set(self.evaluation.index))
        self.assertLess(self.train.timestamp.max(), self.select.timestamp.min())
        self.assertLess(self.select.timestamp.max(), self.evaluation.timestamp.min())

    def test_02_mapping_comes_only_from_v3_train(self) -> None:
        self.assertEqual(set(self.item_map.item_id), {10, 11, 12, 13})
        self.assertEqual(set(self.user_map.user_id), {1, 2})

    def test_03_future_only_target_not_in_mapping(self) -> None:
        changed = self.raw.copy()
        changed.loc[changed.index[-1], "item_id"] = 999
        train, _, evaluation, _ = v3.build_temporal_split(changed)
        _, item_map, _, _ = v3.stable_mappings(train)
        self.assertNotIn(999, set(item_map.item_id))
        self.assertIn(999, set(evaluation.item_id))

    def test_04_training_history_timestamp_is_strict(self) -> None:
        examples, stats, _ = v3.build_training_examples(
            self.train, self.user_to_idx, self.item_to_idx, 5
        )
        self.assertTrue(np.all(examples.max_history_timestamps < examples.target_timestamps))
        self.assertTrue(stats["max_history_timestamp_strictly_before_target"])

    def test_05_same_timestamp_event_is_not_history(self) -> None:
        t = timestamp("2017-01-01", 1)
        synthetic = pd.DataFrame(
            [(1, 10, 1, "pv", t), (1, 11, 1, "buy", t), (1, 12, 1, "pv", t + 1), (1, 13, 1, "buy", t + 2)],
            columns=self.raw.columns,
        )
        _, _, users, items = v3.stable_mappings(synthetic)
        examples, stats, _ = v3.build_training_examples(synthetic, users, items, 5)
        self.assertEqual(stats["non_empty_strict_history_examples"], 1)
        self.assertEqual(int(examples.target_items[0]), items[13])

    def test_06_padding_mask_excludes_padding(self) -> None:
        events = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]]])
        mask = torch.tensor([[True, True, False]])
        pooled = v3.masked_mean(events, mask)
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 0.0]]))

    def test_07_mean_pooling_shape(self) -> None:
        pooled = v3.masked_mean(torch.randn(4, 7, 64), torch.ones(4, 7, dtype=torch.bool))
        self.assertEqual(tuple(pooled.shape), (4, 64))

    def test_08_empty_history_rejected(self) -> None:
        with self.assertRaises(ValueError):
            v3.masked_mean(torch.randn(1, 3, 4), torch.zeros(1, 3, dtype=torch.bool))

    def test_09_inbatch_logits_are_b_by_b(self) -> None:
        users = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        items = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        logits, _, _ = v3.inbatch_logits_and_labels(
            users, items, torch.arange(4), torch.arange(4), torch.full((4,), 0.25), 0.1
        )
        self.assertEqual(tuple(logits.shape), (4, 4))

    def test_10_diagonal_positive_labels(self) -> None:
        vectors = torch.eye(3)
        _, labels, mask = v3.inbatch_logits_and_labels(
            vectors, vectors, torch.arange(3), torch.arange(3), torch.full((3,), 1 / 3), 0.1
        )
        self.assertEqual(labels.tolist(), [0, 1, 2])
        self.assertFalse(bool(torch.diagonal(mask).any()))

    def test_11_duplicate_target_false_negative_mask(self) -> None:
        mask = v3.false_negative_mask(torch.tensor([1, 2, 3]), torch.tensor([9, 9, 10]))
        self.assertTrue(bool(mask[0, 1]))
        self.assertTrue(bool(mask[1, 0]))

    def test_12_same_user_false_negative_mask(self) -> None:
        mask = v3.false_negative_mask(torch.tensor([1, 1, 3]), torch.tensor([8, 9, 10]))
        self.assertTrue(bool(mask[0, 1]))
        self.assertTrue(bool(mask[1, 0]))

    def test_13_logq_logits_are_finite(self) -> None:
        vectors = torch.nn.functional.normalize(torch.randn(3, 5), dim=-1)
        logits, _, _ = v3.inbatch_logits_and_labels(
            vectors, vectors, torch.arange(3), torch.arange(3), torch.tensor([0.0, 0.2, 0.8]), 0.1
        )
        self.assertTrue(torch.isfinite(torch.diagonal(logits)).all())

    def test_14_model_outputs_are_l2_normalized(self) -> None:
        model = v3.IDOnlyDualEncoder(4, 6)
        output = model.encode_users(torch.arange(4))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_15_chunked_exact_equals_dense(self) -> None:
        rng = np.random.default_rng(42)
        users = rng.normal(size=(5, 8)).astype("float32")
        items = rng.normal(size=(13, 8)).astype("float32")
        users /= np.linalg.norm(users, axis=1, keepdims=True)
        items /= np.linalg.norm(items, axis=1, keepdims=True)
        scores, indices = v3.exact_chunked_topk(users, items, 5, user_batch_size=2, item_chunk_size=4)
        dense_indices = np.argsort(users @ items.T, axis=1)[:, ::-1][:, :5]
        np.testing.assert_array_equal(indices, dense_indices)
        np.testing.assert_allclose(scores, np.take_along_axis(users @ items.T, dense_indices, axis=1), atol=1e-6)

    def test_16_eval_population_is_common_and_consistent(self) -> None:
        population = v3.build_query_population(
            "SELECT", self.select, self.train, self.user_to_idx, self.item_to_idx, 5
        )
        self.assertEqual(population.stats["eligible_users"], len(population.user_ids))
        self.assertEqual(set(population.user_ids), set(population.ground_truth))
        self.assertEqual(population.history_items.shape[0], len(population.user_ids))

    def test_17_pre_eval_freeze_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertFalse(v3.is_evaluation_complete(path))
            (path / "final_summary.json").write_text(json.dumps({"evaluation_completed": True}), encoding="utf-8")
            self.assertTrue(v3.is_evaluation_complete(path))


if __name__ == "__main__":
    unittest.main()
