from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generative.beam_search import constrained_beam_search_batch
from generative.transformer import GenerativeRetriever
from generative.trie import SemanticIDTrie


ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "artifacts" / "generative"


class TrieAndGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.trie = SemanticIDTrie(eos_token_id=2)
        self.paths = [(8, 12, 16, 20), (8, 12, 17, 20), (9, 13, 18, 21)]
        for item, path in enumerate(self.paths, start=100):
            self.trie.insert(path, item)

    def test_trie_accepts_valid_paths_and_rejects_invalid_prefix(self) -> None:
        for path in self.paths:
            self.assertTrue(self.trie.accepts((*path, 2)))
            self.assertEqual(self.trie.valid_children(path), (2,))
        self.assertEqual(self.trie.valid_children(()), (8, 9))
        self.assertEqual(self.trie.valid_children((99,)), ())
        self.assertFalse(self.trie.accepts((8, 12, 16, 99, 2)))

    def test_transformer_shapes_masks_forward_and_backward(self) -> None:
        model = GenerativeRetriever(32, 16, 4, 1, 1, 32, 0.0, max_history_items=3)
        histories = torch.tensor([
            [[4, 8, 12, 16, 20, 3], [5, 9, 13, 18, 21, 3], [0, 0, 0, 0, 0, 0]],
            [[6, 8, 12, 17, 20, 3], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        ])
        decoder = torch.tensor([[1, 8, 12], [1, 9, 13]])
        memory, padding = model.encode(histories)
        self.assertEqual(tuple(memory.shape), (2, 3, 16))
        self.assertEqual(tuple(padding.shape), (2, 3))
        mask = model.causal_mask(3, torch.device("cpu"))
        self.assertTrue(mask[0, 1]); self.assertFalse(mask[1, 0])
        logits = model(histories, decoder)
        self.assertEqual(tuple(logits.shape), (2, 3, 32))
        labels = torch.tensor([[8, 12, 16], [9, 13, 18]])
        loss = F.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1))
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertGreater(float(model.token_embedding.weight.grad.abs().sum()), 0.0)

    def test_beam_search_valid_sorted_unique_topk_and_deterministic(self) -> None:
        model = GenerativeRetriever(32, 16, 4, 1, 1, 32, 0.0, max_history_items=2)
        histories = torch.tensor([[[4, 8, 12, 16, 20, 3], [0, 0, 0, 0, 0, 0]]])
        users = np.asarray([1], dtype=np.int64)
        first, first_stats = constrained_beam_search_batch(model, histories, users, self.trie, 1, 2)
        second, _ = constrained_beam_search_batch(model, histories, users, self.trie, 1, 2)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 2)
        self.assertEqual(len({row["item_id"] for row in first}), len(first))
        self.assertTrue(all(self.trie.accepts((*row["token_path"], 2)) for row in first))
        self.assertEqual([row["beam_score"] for row in first], sorted([row["beam_score"] for row in first], reverse=True))
        self.assertEqual(first_stats["invalid_or_incomplete_paths"], 0)

    def test_no_leakage_contract_and_checkpoint_reload(self) -> None:
        with (GEN / "generative_dataset_stats.json").open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        self.assertEqual(stats["history_source"], "train_only")
        self.assertFalse(stats["leakage_checks"]["validation_rows_used_as_training_labels"])
        test_access = stats["leakage_checks"]["test_rows_or_labels_read"]
        self.assertIn(
            test_access,
            (False, "only after config/checkpoint freeze and label-free generations saved"),
        )
        arrays = np.load(GEN / "train_sequences.npz")
        self.assertTrue(np.all(arrays["max_history_timestamps"] < arrays["target_timestamps"]))
        with (GEN / "pre_test_freeze.json").open("r", encoding="utf-8") as handle:
            freeze = json.load(handle)
        if freeze["test_evaluated"]:
            self.assertEqual(freeze["test_evaluation_status"], "completed_once")
            self.assertIn("test_metrics_sha256", freeze)
        else:
            self.assertEqual(freeze["status"], "configuration_and_validation_frozen_before_test")
        with (GEN / "token_manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        checkpoint = torch.load(GEN / "generator_best.pt", map_location="cpu", weights_only=True)
        model = GenerativeRetriever(
            vocab_size=manifest["vocab_size"], d_model=128, nhead=4, encoder_layers=2,
            decoder_layers=2, dim_feedforward=256, dropout=0.1, max_history_items=30,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()
