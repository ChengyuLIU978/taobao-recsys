from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from generative.semantic_ids import assign_collision_suffix, sid_to_tokens, tokens_to_sid


ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "artifacts" / "generative"


class SemanticIDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = pd.read_parquet(GEN / "semantic_ids.parquet")
        cls.mapping = pd.read_csv(ROOT / "artifacts" / "two_tower" / "item_mapping.csv").sort_values("item_idx")
        with (GEN / "token_manifest.json").open("r", encoding="utf-8") as handle:
            cls.manifest = json.load(handle)

    def test_every_active_item_has_unique_final_sid(self) -> None:
        self.assertEqual(len(self.frame), len(self.mapping))
        self.assertEqual(self.frame.item_id.nunique(), len(self.frame))
        self.assertFalse(self.frame.duplicated(["c0", "c1", "c2", "suffix"]).any())

    def test_suffix_is_deterministic_by_item_id(self) -> None:
        items = np.asarray([9, 3, 5, 7], dtype=np.int64)
        codes = np.asarray([[1, 2, 3], [1, 2, 3], [0, 0, 0], [1, 2, 3]], dtype=np.int16)
        suffix = assign_collision_suffix(items, codes)
        self.assertEqual(suffix.tolist(), [2, 0, 0, 1])

    def test_item_sid_roundtrip_and_mapping_alignment(self) -> None:
        aligned = self.frame.set_index("item_id").loc[self.mapping.item_id.head(1000)]
        codes = aligned[["c0", "c1", "c2"]].to_numpy(dtype=np.int64)
        suffix = aligned.suffix.to_numpy(dtype=np.int64)
        tokens = sid_to_tokens(codes, suffix, self.manifest)
        decoded = np.asarray([tokens_to_sid(tuple(map(int, row)), self.manifest) for row in tokens])
        expected = np.column_stack([codes, suffix])
        np.testing.assert_array_equal(decoded, expected)

    def test_no_missing_mapping_and_no_cold_injection(self) -> None:
        self.assertEqual(set(self.frame.item_id.astype(int)), set(self.mapping.item_id.astype(int)))
        test_segments = pd.read_parquet(ROOT / "artifacts" / "analysis" / "test_purchase_segments.parquet")
        cold_items = set(test_segments.loc[test_segments.segment.eq("Cold"), "item_id"].astype(int))
        self.assertTrue(cold_items.isdisjoint(set(self.frame.item_id.astype(int))))


if __name__ == "__main__":
    unittest.main()

