from __future__ import annotations

import importlib.util
import inspect
import math
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "07_build_multistage_recall.py"
)
SPEC = importlib.util.spec_from_file_location("multistage_recall", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def source_frame(source: str, rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["user_id", "item_id", f"{source}_rank", f"{source}_score"],
    )


class MultiStageRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.itemcf = source_frame("itemcf", [(1, 10, 2, 0.8), (1, 13, 1, 0.9)])
        self.two_tower = source_frame(
            "two_tower", [(1, 10, 8, 0.7), (1, 11, 1, 0.95)]
        )
        self.popularity = source_frame(
            "popularity", [(1, 12, 1, 100.0), (1, 10, 4, 90.0)]
        )
        self.fused = MODULE.fuse_source_frames(
            self.itemcf, self.two_tower, self.popularity
        )

    def test_candidate_union_deduplicates_user_item(self) -> None:
        self.assertEqual(len(self.fused), 4)
        self.assertFalse(self.fused.duplicated(["user_id", "item_id"]).any())
        self.assertEqual(len(self.fused.query("user_id == 1 and item_id == 10")), 1)

    def test_source_flags_and_source_count_are_merged(self) -> None:
        row = self.fused.query("user_id == 1 and item_id == 10").iloc[0]
        self.assertEqual(int(row.from_itemcf), 1)
        self.assertEqual(int(row.from_two_tower), 1)
        self.assertEqual(int(row.from_popularity), 1)
        self.assertEqual(int(row.recall_source_count), 3)

    def test_rrf_matches_hand_calculation(self) -> None:
        row = self.fused.query("user_id == 1 and item_id == 10").iloc[0]
        expected = 1 / (60 + 2) + 1 / (60 + 8) + 1 / (60 + 4)
        self.assertTrue(math.isclose(float(row.rrf_score), expected, abs_tol=1e-15))

    def test_generated_ranks_are_one_based(self) -> None:
        frame = MODULE.records_to_source_frame(
            99, [(101, 1.0), (102, 0.5)], "itemcf"
        )
        self.assertEqual(frame.itemcf_rank.tolist(), [1, 2])
        self.assertEqual(int(frame.itemcf_rank.min()), 1)

    def test_candidate_generation_interface_has_no_label_input(self) -> None:
        parameters = inspect.signature(MODULE.generate_split_candidates).parameters
        forbidden = [
            name
            for name in parameters
            if "label" in name.lower()
            or "ground_truth" in name.lower()
            or name == "gt"
        ]
        self.assertEqual(forbidden, [])
        source = inspect.getsource(MODULE.generate_split_candidates)
        self.assertNotIn("TEST_GT_PATH", source)
        self.assertNotIn("test_ground_truth", source)


if __name__ == "__main__":
    unittest.main()
