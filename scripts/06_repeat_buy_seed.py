"""Repeat TT_V1_BUY_UNIFORM once with seed=2027.

This script is deliberately standalone: it reconstructs every dataset, model,
validation, and retrieval object from persisted project files.  It does not
depend on notebook memory and refuses to overwrite an existing replication.

Run from the project root:

    python scripts/06_repeat_buy_seed.py

Use ``--preflight-only`` to validate and print the controlled configuration
without creating artifacts or training a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dev"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "two_tower"
EXPERIMENT_ROOT = ARTIFACT_ROOT / "experiments"
ORIGINAL_V1_DIR = EXPERIMENT_ROOT / "TT_V1_BUY_UNIFORM"
OUTPUT_DIR = EXPERIMENT_ROOT / "TT_V1_BUY_UNIFORM_SEED2027"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_GT_PATH = DATA_DIR / "validation_ground_truth.csv"
USER_MAPPING_PATH = ARTIFACT_ROOT / "user_mapping.csv"
ITEM_MAPPING_PATH = ARTIFACT_ROOT / "item_mapping.csv"
MAPPING_METADATA_PATH = ARTIFACT_ROOT / "mapping_metadata.json"
ORIGINAL_V1_CONFIG_PATH = ORIGINAL_V1_DIR / "config.json"

RUN_NAME = "TT_V1_BUY_UNIFORM_SEED2027"
RUN_SEED = 2027
VALIDATION_NEGATIVE_SEED = 2026
EXPECTED_WARM_USERS = 1_170
EXPECTED_WARM_INTERACTIONS = 1_496
EXPECTED_BUY_PAIRS = 11_844
EXPECTED_BUY_USERS = 5_141
EXPECTED_BUY_ITEMS = 10_819

EXPECTED_V1_SETTINGS = {
    "seed": 42,
    "epochs": 3,
    "epoch_samples": 521_287,
    "batch_size": 1_024,
    "num_negatives": 4,
    "learning_rate": 0.001,
    "embedding_dim": 64,
    "hidden_dim": 128,
    "output_dim": 64,
    "temperature": 0.1,
    "candidate_count": 302_016,
    "evaluation": "warm validation purchase exact retrieval; no history filtering",
    "experiment_name": "TT_V1_BUY_UNIFORM",
    "positive_definition": "train-period behavior_type == 'buy'; unique mapped user-item pairs",
    "negative_strategy": "uniform random; 4 distinct negatives; exclude objective-positive items",
    "sampler": "uniform replacement",
    "positive_pair_count": 11_844,
    "positive_user_count": 5_141,
    "behavior_weights": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate configuration and inputs without creating output or training",
    )
    return parser.parse_args()


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_manifest() -> dict[str, str]:
    """Hash every protected historical artifact, not just the checkpoints."""
    protected = [ARTIFACT_ROOT / "two_tower_best.pt"]
    for directory_name in ("TT_V1_BUY_UNIFORM", "TT_V2_MULTI_UNIFORM"):
        directory = EXPERIMENT_ROOT / directory_name
        protected.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    missing = [str(path) for path in protected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected artifact missing: {missing}")
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in protected
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.user_embedding(ids)), p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.item_embedding(ids)), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(
            num_users, embedding_dim, hidden_dim, output_dim
        )
        self.item_tower = ItemTower(
            num_items, embedding_dim, hidden_dim, output_dim
        )
        self.temperature = temperature

    def encode_users(self, ids: torch.Tensor) -> torch.Tensor:
        return self.user_tower(ids)

    def encode_items(self, ids: torch.Tensor) -> torch.Tensor:
        return self.item_tower(ids)

    def forward(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_vectors = self.encode_users(users)
        positive_vectors = self.encode_items(positives)
        negative_vectors = self.encode_items(negatives)
        positive_logits = (
            (user_vectors * positive_vectors).sum(-1) / self.temperature
        )
        negative_logits = (
            (user_vectors.unsqueeze(1) * negative_vectors).sum(-1)
            / self.temperature
        )
        return positive_logits, negative_logits


def sampled_bce_loss(
    positive_logits: torch.Tensor, negative_logits: torch.Tensor
) -> torch.Tensor:
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits, torch.ones_like(positive_logits)
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_logits, torch.zeros_like(negative_logits)
    )
    return (positive_loss + negative_loss) / 2


class UniformNegativeDataset(Dataset):
    def __init__(
        self,
        pairs: np.ndarray,
        positive_by_user: dict[int, set[int]],
        num_items: int,
        num_negatives: int,
        seed: int,
    ) -> None:
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.positive_by_user = positive_by_user
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_idx, positive_idx = self.pairs[index]
        excluded = self.positive_by_user[int(user_idx)]
        negatives: list[int] = []
        while len(negatives) < self.num_negatives:
            candidate = int(self.rng.integers(0, self.num_items))
            if (
                candidate != positive_idx
                and candidate not in excluded
                and candidate not in negatives
            ):
                negatives.append(candidate)
        return {
            "user": torch.tensor(user_idx),
            "positive_item": torch.tensor(positive_idx),
            "negative_items": torch.tensor(negatives),
        }


def build_positive_sets(pairs: np.ndarray) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for user_idx, item_idx in np.asarray(pairs, dtype=np.int64):
        result[int(user_idx)].add(int(item_idx))
    return result


def make_model(config: dict[str, Any], num_users: int, num_items: int) -> TwoTowerModel:
    return TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["output_dim"]),
        temperature=float(config["temperature"]),
    )


def recall_at_k(recommendations: list[int], ground_truth: set[int], k: int) -> float:
    return (
        len(set(recommendations[:k]) & ground_truth) / len(ground_truth)
        if ground_truth
        else 0.0
    )


def hitrate_at_k(
    recommendations: list[int], ground_truth: set[int], k: int
) -> float:
    return float(bool(set(recommendations[:k]) & ground_truth)) if ground_truth else 0.0


def ndcg_at_k(recommendations: list[int], ground_truth: set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, item in enumerate(recommendations[:k], 1)
        if item in ground_truth
    )
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, min(len(ground_truth), k) + 1)
    )
    return dcg / idcg if idcg else 0.0


def evaluate_recommendations(
    recommendations: dict[int, list[int]],
    warm_ground_truth: dict[int, set[int]],
    ks: tuple[int, ...] = (10, 20, 50),
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in ks:
        rows.append(
            {
                "K": k,
                "Recall": float(
                    np.mean(
                        [
                            recall_at_k(recommendations[user], ground_truth, k)
                            for user, ground_truth in warm_ground_truth.items()
                        ]
                    )
                ),
                "HitRate": float(
                    np.mean(
                        [
                            hitrate_at_k(recommendations[user], ground_truth, k)
                            for user, ground_truth in warm_ground_truth.items()
                        ]
                    )
                ),
                "NDCG": float(
                    np.mean(
                        [
                            ndcg_at_k(recommendations[user], ground_truth, k)
                            for user, ground_truth in warm_ground_truth.items()
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def deterministic_validation_loss(
    model: TwoTowerModel,
    validation_user_indices: np.ndarray,
    validation_item_indices: np.ndarray,
    fixed_validation_negatives: np.ndarray,
    batch_size: int,
) -> float:
    losses: list[tuple[float, int]] = []
    count = len(validation_user_indices)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        users = torch.from_numpy(validation_user_indices[start:end])
        positives = torch.from_numpy(validation_item_indices[start:end])
        negatives = torch.from_numpy(fixed_validation_negatives[start:end])
        positive_logits, negative_logits = model(users, positives, negatives)
        loss = sampled_bce_loss(positive_logits, negative_logits)
        losses.append((float(loss.item()), end - start))
    return sum(loss * size for loss, size in losses) / count


@torch.inference_mode()
def encode_all(
    model: TwoTowerModel,
    num_users: int,
    num_items: int,
    batch_size: int = 8_192,
) -> tuple[np.ndarray, np.ndarray]:
    users = [
        model.encode_users(torch.arange(start, min(start + batch_size, num_users)))
        .numpy()
        for start in range(0, num_users, batch_size)
    ]
    items = [
        model.encode_items(torch.arange(start, min(start + batch_size, num_items)))
        .numpy()
        for start in range(0, num_items, batch_size)
    ]
    return (
        np.concatenate(users).astype("float32"),
        np.concatenate(items).astype("float32"),
    )


def exact_chunked_topk(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    user_indices: np.ndarray,
    num_items: int,
    top_k: int = 50,
    user_batch_size: int = 64,
    item_chunk_size: int = 50_000,
) -> np.ndarray:
    item_tensor = torch.from_numpy(item_embeddings)
    result_indices: list[np.ndarray] = []
    for user_start in range(0, len(user_indices), user_batch_size):
        batch_indices = user_indices[user_start : user_start + user_batch_size]
        user_tensor = torch.from_numpy(user_embeddings[batch_indices])
        current_batch_size = len(batch_indices)
        best_scores = torch.full((current_batch_size, top_k), -torch.inf)
        best_indices = torch.full(
            (current_batch_size, top_k), -1, dtype=torch.long
        )
        for item_start in range(0, num_items, item_chunk_size):
            item_end = min(item_start + item_chunk_size, num_items)
            scores = user_tensor @ item_tensor[item_start:item_end].T
            local_scores, local_indices = torch.topk(
                scores, min(top_k, item_end - item_start), dim=1
            )
            local_indices += item_start
            merged_scores = torch.cat([best_scores, local_scores], dim=1)
            merged_indices = torch.cat([best_indices, local_indices], dim=1)
            best_scores, positions = torch.topk(merged_scores, top_k, dim=1)
            best_indices = torch.gather(merged_indices, 1, positions)
        result_indices.append(best_indices.numpy())
    return np.vstack(result_indices)


def evaluate_model(
    model: TwoTowerModel,
    num_users: int,
    num_items: int,
    warm_user_ids: np.ndarray,
    warm_user_indices: np.ndarray,
    candidate_item_ids: np.ndarray,
    warm_ground_truth: dict[int, set[int]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    user_embeddings, item_embeddings = encode_all(model, num_users, num_items)
    top_indices = exact_chunked_topk(
        user_embeddings,
        item_embeddings,
        warm_user_indices,
        num_items,
    )
    recommendations = {
        int(user_id): candidate_item_ids[top_indices[row]].astype(int).tolist()
        for row, user_id in enumerate(warm_user_ids)
    }
    metrics = evaluate_recommendations(recommendations, warm_ground_truth)
    top1_counts = pd.Series(
        [items[0] for items in recommendations.values()]
    ).value_counts()
    top50_union = {
        item for items in recommendations.values() for item in items
    }
    concentration = {
        "unique_top1": int(len(top1_counts)),
        "dominant_top1_item": int(top1_counts.index[0]),
        "dominant_top1_share": float(
            top1_counts.iloc[0] / len(recommendations)
        ),
        "unique_top50": int(len(top50_union)),
        "catalog_coverage_at_50": float(len(top50_union) / num_items),
    }
    return metrics, concentration


def print_config_comparison(
    original: dict[str, Any], replication: dict[str, Any]
) -> None:
    print("\nORIGINAL V1 CONFIG vs SEED2027 CONFIG")
    print("-" * 92)
    print(f"{'setting':<32} {'original V1':<28} {'seed2027':<28} status")
    print("-" * 92)
    for key in original:
        original_value = original[key]
        replication_value = replication[key]
        status = "CHANGED (required)" if key == "seed" else "SAME"
        print(
            f"{key:<32} {str(original_value):<28} "
            f"{str(replication_value):<28} {status}"
        )
    print("-" * 92)
    print(f"Original output directory: {ORIGINAL_V1_DIR}")
    print(f"New output directory:      {OUTPUT_DIR}")
    print("Scientific setting differences: seed only (42 -> 2027)")
    print("Output identity difference: independent SEED2027 directory")


def preflight() -> dict[str, Any]:
    required_paths = [
        TRAIN_PATH,
        VALIDATION_GT_PATH,
        USER_MAPPING_PATH,
        ITEM_MAPPING_PATH,
        MAPPING_METADATA_PATH,
        ORIGINAL_V1_CONFIG_PATH,
        ORIGINAL_V1_DIR / "checkpoint_best.pt",
        EXPERIMENT_ROOT / "TT_V2_MULTI_UNIFORM" / "checkpoint_best.pt",
        ARTIFACT_ROOT / "two_tower_best.pt",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required project files are missing: {missing}")

    original_config = json_read(ORIGINAL_V1_CONFIG_PATH)
    for key, expected_value in EXPECTED_V1_SETTINGS.items():
        actual_value = original_config.get(key)
        if actual_value != expected_value:
            raise AssertionError(
                f"Original V1 config mismatch for {key}: "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    if original_config["mapping"]["user_mapping"] != "../../user_mapping.csv":
        raise AssertionError("Unexpected original V1 user mapping reference")
    if original_config["mapping"]["item_mapping"] != "../../item_mapping.csv":
        raise AssertionError("Unexpected original V1 item mapping reference")

    mapping_metadata = json_read(MAPPING_METADATA_PATH)
    train_sha256 = sha256_file(TRAIN_PATH)
    expected_sha256 = original_config["mapping"]["train_csv_sha256"]
    if train_sha256 != expected_sha256:
        raise AssertionError(
            f"train.csv SHA-256 changed: expected={expected_sha256}, actual={train_sha256}"
        )
    if mapping_metadata["train_csv_sha256"] != train_sha256:
        raise AssertionError("Mapping metadata and train.csv SHA-256 disagree")

    replication_config = json.loads(json.dumps(original_config))
    replication_config["seed"] = RUN_SEED
    scientific_differences = {
        key: (original_config[key], replication_config[key])
        for key in original_config
        if original_config[key] != replication_config[key]
    }
    if scientific_differences != {"seed": (42, RUN_SEED)}:
        raise AssertionError(
            f"Unexpected scientific configuration differences: {scientific_differences}"
        )
    print_config_comparison(original_config, replication_config)
    print("\nPreflight file/config verification: OK")
    print(f"train.csv SHA-256: {train_sha256}")
    print("Checkpoint selection: maximize exact Recall@50; tie-break by NDCG@50")
    print(f"Validation negative seed remains fixed at {VALIDATION_NEGATIVE_SEED}")
    print("No protected output has been opened for writing.")

    return {
        "original_config": original_config,
        "replication_config": replication_config,
        "mapping_metadata": mapping_metadata,
        "train_sha256": train_sha256,
    }


def main() -> None:
    args = parse_args()
    preflight_state = preflight()
    if args.preflight_only:
        print("\nPREFLIGHT ONLY: training was not started and no output directory was created.")
        return

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing replication directory: {OUTPUT_DIR}"
        )

    original_config = preflight_state["original_config"]
    replication_config = preflight_state["replication_config"]
    mapping_metadata = preflight_state["mapping_metadata"]
    protected_before = protected_manifest()

    train_df = pd.read_csv(TRAIN_PATH)
    validation_gt = pd.read_csv(VALIDATION_GT_PATH)
    user_mapping = pd.read_csv(USER_MAPPING_PATH).sort_values("user_idx")
    item_mapping = pd.read_csv(ITEM_MAPPING_PATH).sort_values("item_idx")

    if user_mapping["user_idx"].tolist() != list(range(len(user_mapping))):
        raise AssertionError("user_idx is not a contiguous zero-based mapping")
    if item_mapping["item_idx"].tolist() != list(range(len(item_mapping))):
        raise AssertionError("item_idx is not a contiguous zero-based mapping")
    if user_mapping["user_id"].duplicated().any():
        raise AssertionError("Duplicate user_id found in persisted mapping")
    if item_mapping["item_id"].duplicated().any():
        raise AssertionError("Duplicate item_id found in persisted mapping")

    user2idx = dict(
        zip(user_mapping.user_id.astype(int), user_mapping.user_idx.astype(int))
    )
    item2idx = dict(
        zip(item_mapping.item_id.astype(int), item_mapping.item_idx.astype(int))
    )
    candidate_item_ids = item_mapping.item_id.to_numpy(dtype=np.int64).copy()
    num_users = len(user_mapping)
    num_items = len(item_mapping)
    if (num_users, num_items) != (9_836, 302_016):
        raise AssertionError(
            f"Vocabulary mismatch: users={num_users}, items={num_items}"
        )
    if mapping_metadata["num_users"] != num_users:
        raise AssertionError("User vocabulary disagrees with mapping metadata")
    if mapping_metadata["num_items"] != num_items:
        raise AssertionError("Item vocabulary disagrees with mapping metadata")

    user_seen = validation_gt.user_id.isin(user2idx)
    item_seen = validation_gt.item_id.isin(item2idx)
    warm_gt_df = (
        validation_gt.loc[
            user_seen & item_seen, ["user_id", "item_id"]
        ]
        .drop_duplicates()
        .copy()
    )
    warm_ground_truth = (
        warm_gt_df.groupby("user_id").item_id.apply(set).to_dict()
    )
    warm_user_ids = np.array(sorted(warm_ground_truth), dtype=np.int64)
    warm_user_indices = np.array(
        [user2idx[int(user_id)] for user_id in warm_user_ids], dtype=np.int64
    )
    validation_user_indices = (
        warm_gt_df.user_id.map(user2idx).to_numpy(dtype=np.int64).copy()
    )
    validation_item_indices = (
        warm_gt_df.item_id.map(item2idx).to_numpy(dtype=np.int64).copy()
    )
    if len(warm_ground_truth) != EXPECTED_WARM_USERS:
        raise AssertionError(
            f"Warm evaluation users changed: {len(warm_ground_truth)}"
        )
    if len(warm_gt_df) != EXPECTED_WARM_INTERACTIONS:
        raise AssertionError(
            f"Warm evaluation interactions changed: {len(warm_gt_df)}"
        )

    mapped_train = train_df.loc[
        train_df.user_id.isin(user2idx) & train_df.item_id.isin(item2idx),
        ["user_id", "item_id", "behavior_type"],
    ].copy()
    mapped_train["user_idx"] = mapped_train.user_id.map(user2idx).astype(int)
    mapped_train["item_idx"] = mapped_train.item_id.map(item2idx).astype(int)
    buy_pairs_df = (
        mapped_train.loc[
            mapped_train.behavior_type.eq("buy"), ["user_idx", "item_idx"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    buy_pairs = buy_pairs_df[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
    buy_positive_sets = build_positive_sets(buy_pairs)
    observed_buy_items = int(buy_pairs_df.item_idx.nunique())
    if len(buy_pairs) != EXPECTED_BUY_PAIRS:
        raise AssertionError(f"Mapped buy positive count changed: {len(buy_pairs)}")
    if len(buy_positive_sets) != EXPECTED_BUY_USERS:
        raise AssertionError(f"Mapped buy user count changed: {len(buy_positive_sets)}")
    if observed_buy_items != EXPECTED_BUY_ITEMS:
        raise AssertionError(f"Mapped buy item count changed: {observed_buy_items}")

    known_positive_by_user: dict[int, set[int]] = defaultdict(set)
    mapped_known = train_df.loc[
        train_df.user_id.isin(user2idx) & train_df.item_id.isin(item2idx),
        ["user_id", "item_id"],
    ].drop_duplicates()
    for row in mapped_known.itertuples(index=False):
        known_positive_by_user[user2idx[int(row.user_id)]].add(
            item2idx[int(row.item_id)]
        )
    for user_idx, item_idx in zip(
        validation_user_indices, validation_item_indices
    ):
        known_positive_by_user[int(user_idx)].add(int(item_idx))

    fixed_rng = np.random.default_rng(VALIDATION_NEGATIVE_SEED)
    fixed_validation_negatives = np.empty(
        (len(warm_gt_df), int(replication_config["num_negatives"])),
        dtype=np.int64,
    )
    for row, (user_idx, positive_idx) in enumerate(
        zip(validation_user_indices, validation_item_indices)
    ):
        negatives: list[int] = []
        while len(negatives) < int(replication_config["num_negatives"]):
            candidate = int(fixed_rng.integers(0, num_items))
            if (
                candidate not in known_positive_by_user[int(user_idx)]
                and candidate not in negatives
            ):
                negatives.append(candidate)
        fixed_validation_negatives[row] = negatives
    if fixed_validation_negatives.shape != (EXPECTED_WARM_INTERACTIONS, 4):
        raise AssertionError(
            f"Validation negative shape changed: {fixed_validation_negatives.shape}"
        )
    if not all(len(set(row)) == 4 for row in fixed_validation_negatives):
        raise AssertionError("Validation negatives contain duplicates within a row")
    if not all(
        int(negative) not in known_positive_by_user[int(user)]
        for user, row in zip(
            validation_user_indices, fixed_validation_negatives
        )
        for negative in row
    ):
        raise AssertionError("A fixed validation negative is a known positive")

    if int(replication_config["epoch_samples"]) != 521_287:
        raise AssertionError("Epoch positive draw budget changed")
    if int(replication_config["num_negatives"]) != 4:
        raise AssertionError("Training negative count changed")
    if replication_config["sampler"] != "uniform replacement":
        raise AssertionError("Buy-only positive sampler changed")

    saved_config = json.loads(json.dumps(replication_config))
    saved_config["experiment_name"] = RUN_NAME
    saved_config["source_experiment"] = original_config["experiment_name"]
    saved_config["output_dir"] = str(OUTPUT_DIR.relative_to(PROJECT_ROOT)).replace(
        "\\", "/"
    )
    saved_config["validation_negative_seed"] = VALIDATION_NEGATIVE_SEED
    saved_config["validation_negative_shape"] = [
        int(value) for value in fixed_validation_negatives.shape
    ]
    saved_config["checkpoint_selection_rule"] = (
        "maximize exact Recall@50; break ties with exact NDCG@50"
    )
    saved_config["mapped_buy_item_count"] = observed_buy_items
    saved_config["warm_evaluation_users"] = len(warm_ground_truth)
    saved_config["warm_evaluation_interactions"] = len(warm_gt_df)

    print("\nDATA AND PROTOCOL CHECKS")
    print(f"Shared mapping: {num_users:,} users / {num_items:,} items")
    print(
        f"Mapped buy positives: {len(buy_pairs):,} pairs / "
        f"{len(buy_positive_sets):,} users / {observed_buy_items:,} items"
    )
    print(
        f"Epoch positive draw budget: {replication_config['epoch_samples']:,} "
        f"({math.ceil(replication_config['epoch_samples'] / replication_config['batch_size'])} batches)"
    )
    print(
        f"Training negatives: {replication_config['num_negatives']} distinct uniform "
        "negatives excluding objective positives"
    )
    print(
        f"Fixed validation negatives: {fixed_validation_negatives.shape}, "
        f"seed={VALIDATION_NEGATIVE_SEED}, known positives excluded"
    )
    print(
        f"Warm evaluation population: {len(warm_ground_truth):,} users / "
        f"{len(warm_gt_df):,} interactions"
    )
    print("Checkpoint selection: exact Recall@50, then exact NDCG@50")
    print("All pre-training checks passed. Starting the one authorized seed=2027 run.")

    OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
    json_write(OUTPUT_DIR / "config.json", saved_config)

    seed_everything(RUN_SEED)
    model = make_model(replication_config, num_users, num_items)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(replication_config["learning_rate"])
    )
    dataset = UniformNegativeDataset(
        buy_pairs,
        buy_positive_sets,
        num_items,
        int(replication_config["num_negatives"]),
        RUN_SEED,
    )
    sampler_generator = torch.Generator().manual_seed(RUN_SEED)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=int(replication_config["epoch_samples"]),
        generator=sampler_generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(replication_config["batch_size"]),
        sampler=sampler,
        num_workers=0,
    )

    history: list[dict[str, Any]] = []
    best_key = (-1.0, -1.0)
    best_path = OUTPUT_DIR / "best.pt"
    for epoch in range(1, int(replication_config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        started = time.perf_counter()
        for batch_id, batch in enumerate(loader, 1):
            optimizer.zero_grad()
            positive_logits, negative_logits = model(
                batch["user"].long(),
                batch["positive_item"].long(),
                batch["negative_items"].long(),
            )
            loss = sampled_bce_loss(positive_logits, negative_logits)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            if batch_id % 100 == 0:
                print(
                    f"{RUN_NAME} epoch {epoch} batch {batch_id}/{len(loader)}",
                    flush=True,
                )

        train_loss = total_loss / len(loader)
        model.eval()
        fixed_loss = deterministic_validation_loss(
            model,
            validation_user_indices,
            validation_item_indices,
            fixed_validation_negatives,
            int(replication_config["batch_size"]),
        )
        epoch_metrics, epoch_concentration = evaluate_model(
            model,
            num_users,
            num_items,
            warm_user_ids,
            warm_user_indices,
            candidate_item_ids,
            warm_ground_truth,
        )
        metrics_at_50 = epoch_metrics.loc[epoch_metrics.K.eq(50)].iloc[0]
        recall_at_50 = float(metrics_at_50.Recall)
        hitrate_at_50 = float(metrics_at_50.HitRate)
        ndcg_at_50 = float(metrics_at_50.NDCG)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "fixed_purchase_sampled_loss": fixed_loss,
            "Recall@50": recall_at_50,
            "HitRate@50": hitrate_at_50,
            "NDCG@50": ndcg_at_50,
            "seconds": time.perf_counter() - started,
            **epoch_concentration,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(OUTPUT_DIR / "history.csv", index=False)
        print(
            f"{RUN_NAME} epoch {epoch}: train={train_loss:.6f} "
            f"fixed_val={fixed_loss:.6f} R@50={recall_at_50:.6f} "
            f"HR@50={hitrate_at_50:.6f} N@50={ndcg_at_50:.6f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
        selection_key = (recall_at_50, ndcg_at_50)
        if selection_key > best_key:
            best_key = selection_key
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_users": num_users,
                    "num_items": num_items,
                    "embedding_dim": int(replication_config["embedding_dim"]),
                    "hidden_dim": int(replication_config["hidden_dim"]),
                    "output_dim": int(replication_config["output_dim"]),
                    "temperature": float(replication_config["temperature"]),
                    "best_epoch": epoch,
                    "best_recall_at_50": recall_at_50,
                    "best_hitrate_at_50": hitrate_at_50,
                    "best_ndcg_at_50": ndcg_at_50,
                    "config": saved_config,
                    "mapping": saved_config["mapping"],
                },
                best_path,
            )

    saved_checkpoint = torch.load(
        best_path, map_location="cpu", weights_only=True
    )
    strict_model = make_model(replication_config, num_users, num_items)
    strict_model.load_state_dict(saved_checkpoint["model_state_dict"], strict=True)
    strict_model.eval()
    final_metrics, final_concentration = evaluate_model(
        strict_model,
        num_users,
        num_items,
        warm_user_ids,
        warm_user_indices,
        candidate_item_ids,
        warm_ground_truth,
    )
    final_metrics.insert(0, "Model", RUN_NAME)
    final_metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    json_write(OUTPUT_DIR / "concentration.json", final_concentration)

    v0_metrics = pd.read_csv(
        EXPERIMENT_ROOT / "TT_V0_PV_UNIFORM" / "metrics.csv"
    )
    original_v1_metrics = pd.read_csv(ORIGINAL_V1_DIR / "metrics.csv")
    v0_r50 = float(v0_metrics.loc[v0_metrics.K.eq(50), "Recall"].iloc[0])
    original_v1_r50 = float(
        original_v1_metrics.loc[original_v1_metrics.K.eq(50), "Recall"].iloc[0]
    )
    new_r50 = float(final_metrics.loc[final_metrics.K.eq(50), "Recall"].iloc[0])
    conclusion = (
        "Buy-only improvement replicated directionally under an alternate random seed."
        if new_r50 > v0_r50
        else "The Buy-only improvement appears sensitive to random initialization / sampling seed."
    )

    protected_after = protected_manifest()
    if protected_after != protected_before:
        changed = sorted(
            set(protected_before)
            | set(protected_after)
        )
        changed = [
            path
            for path in changed
            if protected_before.get(path) != protected_after.get(path)
        ]
        raise RuntimeError(f"Protected historical artifacts changed: {changed}")

    history_check = pd.read_csv(OUTPUT_DIR / "history.csv")
    metrics_check = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    config_check = json_read(OUTPUT_DIR / "config.json")
    concentration_check = json_read(OUTPUT_DIR / "concentration.json")
    if len(history_check) != int(replication_config["epochs"]):
        raise AssertionError("history.csv does not contain all epochs")
    required_history_columns = {
        "epoch",
        "train_loss",
        "fixed_purchase_sampled_loss",
        "Recall@50",
        "HitRate@50",
        "NDCG@50",
    }
    if not required_history_columns.issubset(history_check.columns):
        raise AssertionError("history.csv is missing required columns")
    if set(metrics_check.K.astype(int)) != {10, 20, 50}:
        raise AssertionError("metrics.csv is missing a required K")
    if config_check["seed"] != RUN_SEED:
        raise AssertionError("Saved config does not record seed=2027")
    if config_check["mapping"] != original_config["mapping"]:
        raise AssertionError("Saved mapping reference differs from original V1")
    if concentration_check != final_concentration:
        raise AssertionError("Saved concentration does not match final evaluation")

    summary = {
        "experiment": RUN_NAME,
        "source_experiment": "TT_V1_BUY_UNIFORM",
        "seed": RUN_SEED,
        "best_epoch": int(saved_checkpoint["best_epoch"]),
        "checkpoint_selection_rule": saved_config["checkpoint_selection_rule"],
        "final_metrics": {
            str(int(row.K)): {
                "Recall": float(row.Recall),
                "HitRate": float(row.HitRate),
                "NDCG": float(row.NDCG),
            }
            for row in final_metrics.itertuples(index=False)
        },
        "concentration": final_concentration,
        "evaluation_population": {
            "warm_users": len(warm_ground_truth),
            "warm_interactions": len(warm_gt_df),
            "candidate_items": num_items,
        },
        "comparison": {
            "v0_recall_at_50": v0_r50,
            "original_v1_seed42_recall_at_50": original_v1_r50,
            "seed2027_recall_at_50": new_r50,
            "ratio_vs_v0": new_r50 / v0_r50,
            "absolute_delta_vs_v0": new_r50 - v0_r50,
            "absolute_delta_vs_seed42": new_r50 - original_v1_r50,
            "relative_change_vs_seed42": (
                (new_r50 - original_v1_r50) / original_v1_r50
            ),
        },
        "conclusion": conclusion,
        "strict_checkpoint_load": True,
        "protected_historical_artifacts_unchanged": True,
        "protected_artifact_sha256": protected_after,
        "validation": {
            "history_readable_and_complete": True,
            "metrics_readable_and_complete": True,
            "config_records_seed_2027": True,
            "mapping_reference_matches_original_v1": True,
        },
    }
    json_write(OUTPUT_DIR / "summary.json", summary)

    print("\nRUN COMPLETE")
    print(f"Best epoch: {summary['best_epoch']}")
    print(final_metrics.to_string(index=False))
    print(f"Concentration: {final_concentration}")
    print(f"Ratio vs V0 Recall@50: {summary['comparison']['ratio_vs_v0']:.6f}x")
    print(
        "Absolute delta vs original seed42 V1 Recall@50: "
        f"{summary['comparison']['absolute_delta_vs_seed42']:+.9f}"
    )
    print(conclusion)
    print("Strict checkpoint load: OK")
    print("Protected V0/V1/V2 historical artifacts unchanged: OK")
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
