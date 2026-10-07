"""Build and evaluate the three-source multi-stage recall candidate union.

Run from the project root:

    python scripts/07_build_multistage_recall.py

The script is standalone and reads all data, mappings, and checkpoints from
disk.  Candidate generation uses split user IDs from validation.csv/test.csv;
ground-truth labels are loaded only after both candidate artifacts are saved.
Historical-item filtering is intentionally unchanged to preserve protocol
comparability.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dev"
TWO_TOWER_DIR = PROJECT_ROOT / "artifacts" / "two_tower"
EXPERIMENT_DIR = TWO_TOWER_DIR / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "multistage_recall"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"
VALIDATION_GT_PATH = DATA_DIR / "validation_ground_truth.csv"
TEST_GT_PATH = DATA_DIR / "test_ground_truth.csv"
USER_MAPPING_PATH = TWO_TOWER_DIR / "user_mapping.csv"
ITEM_MAPPING_PATH = TWO_TOWER_DIR / "item_mapping.csv"
MAPPING_METADATA_PATH = TWO_TOWER_DIR / "mapping_metadata.json"
V1_DIR = EXPERIMENT_DIR / "TT_V1_BUY_UNIFORM"
V1_CONFIG_PATH = V1_DIR / "config.json"
V1_CHECKPOINT_PATH = V1_DIR / "checkpoint_best.pt"

ITEMCF_QUOTA = 200
TWO_TOWER_QUOTA = 100
POPULARITY_QUOTA = 50
MAX_CANDIDATES_PER_USER = ITEMCF_QUOTA + TWO_TOWER_QUOTA + POPULARITY_QUOTA
ITEMCF_MAX_HISTORY = 50
ITEMCF_NEIGHBORS_PER_SEED = 200
RRF_CONSTANT = 60
USER_BATCH_SIZE = 64
ITEM_CHUNK_SIZE = 50_000
ENCODE_BATCH_SIZE = 8_192

SOURCE_NAMES = ("itemcf", "two_tower", "popularity")
SOURCE_QUOTAS = {
    "itemcf": ITEMCF_QUOTA,
    "two_tower": TWO_TOWER_QUOTA,
    "popularity": POPULARITY_QUOTA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate inputs and print the locked configuration without writing artifacts",
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_manifest() -> dict[str, str]:
    protected_paths = [
        TWO_TOWER_DIR / "two_tower_best.pt",
        USER_MAPPING_PATH,
        ITEM_MAPPING_PATH,
        MAPPING_METADATA_PATH,
        V1_DIR / "checkpoint_best.pt",
        V1_DIR / "config.json",
        V1_DIR / "training_history.csv",
        V1_DIR / "metrics.csv",
        EXPERIMENT_DIR / "TT_V2_MULTI_UNIFORM" / "checkpoint_best.pt",
        EXPERIMENT_DIR / "TT_V2_MULTI_UNIFORM" / "config.json",
        EXPERIMENT_DIR / "TT_V2_MULTI_UNIFORM" / "training_history.csv",
        EXPERIMENT_DIR / "TT_V2_MULTI_UNIFORM" / "metrics.csv",
        EXPERIMENT_DIR / "TT_V1_BUY_UNIFORM_SEED2027" / "best.pt",
        EXPERIMENT_DIR / "TT_V1_BUY_UNIFORM_SEED2027" / "config.json",
        EXPERIMENT_DIR / "TT_V1_BUY_UNIFORM_SEED2027" / "history.csv",
        EXPERIMENT_DIR / "TT_V1_BUY_UNIFORM_SEED2027" / "metrics.csv",
    ]
    missing = [str(path) for path in protected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected historical artifacts missing: {missing}")
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in protected_paths
    }


class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
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
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
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


def source_columns(source: str) -> list[str]:
    return ["user_id", "item_id", f"{source}_rank", f"{source}_score"]


def empty_source_frame(source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "item_id": pd.Series(dtype="int64"),
            f"{source}_rank": pd.Series(dtype="int32"),
            f"{source}_score": pd.Series(dtype="float32"),
        }
    )


def records_to_source_frame(
    user_id: int, ranked_items: Iterable[tuple[int, float]], source: str
) -> pd.DataFrame:
    """Create a source frame with the public one-based rank convention."""
    records = list(ranked_items)
    if not records:
        return empty_source_frame(source)
    return pd.DataFrame(
        {
            "user_id": np.full(len(records), int(user_id), dtype=np.int64),
            "item_id": np.array([item for item, _ in records], dtype=np.int64),
            f"{source}_rank": np.arange(1, len(records) + 1, dtype=np.int32),
            f"{source}_score": np.array(
                [score for _, score in records], dtype=np.float32
            ),
        }
    )


def normalize_source_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    expected = source_columns(source)
    if list(frame.columns) != expected:
        frame = frame[expected].copy()
    else:
        frame = frame.copy()
    if frame.duplicated(["user_id", "item_id"]).any():
        raise AssertionError(f"{source} contains duplicate user-item pairs")
    if len(frame) and (frame[f"{source}_rank"] < 1).any():
        raise AssertionError(f"{source} uses a non-positive rank")
    frame["user_id"] = frame["user_id"].astype("int64")
    frame["item_id"] = frame["item_id"].astype("int64")
    frame[f"{source}_rank"] = frame[f"{source}_rank"].astype("int32")
    frame[f"{source}_score"] = frame[f"{source}_score"].astype("float32")
    return frame


def fuse_source_frames(
    itemcf: pd.DataFrame,
    two_tower: pd.DataFrame,
    popularity: pd.DataFrame,
    rrf_constant: int = RRF_CONSTANT,
) -> pd.DataFrame:
    """Outer-union three unique source frames and preserve all provenance."""
    frames = {
        "itemcf": normalize_source_frame(itemcf, "itemcf"),
        "two_tower": normalize_source_frame(two_tower, "two_tower"),
        "popularity": normalize_source_frame(popularity, "popularity"),
    }
    merged = frames["itemcf"].merge(
        frames["two_tower"],
        how="outer",
        on=["user_id", "item_id"],
        validate="one_to_one",
    )
    merged = merged.merge(
        frames["popularity"],
        how="outer",
        on=["user_id", "item_id"],
        validate="one_to_one",
    )
    rrf_score = np.zeros(len(merged), dtype=np.float64)
    flag_columns: list[str] = []
    for source in SOURCE_NAMES:
        rank_column = f"{source}_rank"
        score_column = f"{source}_score"
        flag_column = f"from_{source}"
        present = merged[rank_column].notna()
        merged[flag_column] = present.astype("int8")
        merged[rank_column] = merged[rank_column].astype("Int32")
        merged[score_column] = merged[score_column].astype("Float32")
        rrf_score[present.to_numpy()] += 1.0 / (
            rrf_constant
            + merged.loc[present, rank_column].astype("float64").to_numpy()
        )
        flag_columns.append(flag_column)
    merged["recall_source_count"] = (
        merged[flag_columns].sum(axis=1).astype("int8")
    )
    merged["rrf_score"] = rrf_score
    ordered_columns = [
        "user_id",
        "item_id",
        "from_itemcf",
        "from_two_tower",
        "from_popularity",
        "itemcf_rank",
        "two_tower_rank",
        "popularity_rank",
        "itemcf_score",
        "two_tower_score",
        "popularity_score",
        "recall_source_count",
        "rrf_score",
    ]
    merged = merged[ordered_columns].sort_values(
        ["user_id", "rrf_score", "item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return merged.reset_index(drop=True)


def validate_candidate_table(
    candidates: pd.DataFrame, expected_users: np.ndarray
) -> dict[str, Any]:
    if candidates.duplicated(["user_id", "item_id"]).any():
        raise AssertionError("Candidate union contains duplicate user-item rows")
    actual_users = np.sort(candidates.user_id.unique())
    if not np.array_equal(actual_users, np.sort(expected_users.astype(np.int64))):
        raise AssertionError("Candidate users differ from split interaction users")
    counts = candidates.groupby("user_id", sort=False).size()
    if int(counts.max()) > MAX_CANDIDATES_PER_USER:
        raise AssertionError("Candidate count exceeds the 350-source-quota ceiling")
    flags = candidates[
        ["from_itemcf", "from_two_tower", "from_popularity"]
    ]
    if (flags.sum(axis=1) < 1).any():
        raise AssertionError("A candidate has no recall source")
    if not np.array_equal(
        flags.sum(axis=1).to_numpy(dtype=np.int8),
        candidates.recall_source_count.to_numpy(dtype=np.int8),
    ):
        raise AssertionError("recall_source_count disagrees with source flags")
    if not np.isfinite(candidates.rrf_score.to_numpy()).all():
        raise AssertionError("RRF contains inf or NaN")
    if (candidates.rrf_score <= 0).any():
        raise AssertionError("RRF must be positive for every recalled candidate")

    recomputed_rrf = np.zeros(len(candidates), dtype=np.float64)
    source_checks: dict[str, Any] = {}
    for source in SOURCE_NAMES:
        flag = candidates[f"from_{source}"].eq(1)
        rank = candidates[f"{source}_rank"]
        score = candidates[f"{source}_score"]
        if not flag.equals(rank.notna()):
            raise AssertionError(f"{source} flag/rank alignment failed")
        if not flag.equals(score.notna()):
            raise AssertionError(f"{source} flag/score alignment failed")
        if flag.any() and (rank.loc[flag] < 1).any():
            raise AssertionError(f"{source} contains a non-positive rank")
        if flag.any() and (rank.loc[flag] > SOURCE_QUOTAS[source]).any():
            raise AssertionError(f"{source} rank exceeds its quota")
        source_rows = candidates.loc[flag, ["user_id", f"{source}_rank"]]
        if source_rows.duplicated(["user_id", f"{source}_rank"]).any():
            raise AssertionError(f"{source} has duplicate per-user ranks")
        recomputed_rrf[flag.to_numpy()] += 1.0 / (
            RRF_CONSTANT + rank.loc[flag].astype("float64").to_numpy()
        )
        source_checks[source] = {
            "candidate_pairs": int(flag.sum()),
            "users_with_candidates": int(candidates.loc[flag, "user_id"].nunique()),
            "max_rank": int(rank.loc[flag].max()) if flag.any() else 0,
        }
    if not np.allclose(
        recomputed_rrf, candidates.rrf_score.to_numpy(dtype=np.float64)
    ):
        raise AssertionError("Saved RRF differs from the rank-based formula")
    return {
        "unique_user_item_pairs": True,
        "source_flags_valid": True,
        "source_count_valid": True,
        "rrf_finite_and_recomputed": True,
        "positive_one_based_ranks": True,
        "score_rank_alignment": True,
        "users": int(len(counts)),
        "candidate_pairs": int(len(candidates)),
        "max_candidates_per_user": int(counts.max()),
        "within_quota_ceiling": True,
        "sources": source_checks,
    }


def build_itemcf_context(
    train_df: pd.DataFrame, target_user_ids: np.ndarray
) -> tuple[dict[int, list[int]], dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    itemcf_train = (
        train_df.sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "item_id"], keep="last")
        .sort_values(["user_id", "timestamp"])
        .groupby("user_id", group_keys=False)
        .tail(ITEMCF_MAX_HISTORY)
    )
    itemcf_user_history = (
        itemcf_train.groupby("user_id")["item_id"].apply(list).to_dict()
    )
    user_codes, itemcf_user_ids = pd.factorize(itemcf_train.user_id, sort=True)
    item_codes, itemcf_item_ids = pd.factorize(itemcf_train.item_id, sort=True)
    matrix = csr_matrix(
        (
            np.ones(len(itemcf_train), dtype=np.float32),
            (user_codes, item_codes),
        ),
        shape=(len(itemcf_user_ids), len(itemcf_item_ids)),
    )
    item_to_col = {
        int(item_id): column for column, item_id in enumerate(itemcf_item_ids)
    }
    seed_items = sorted(
        {
            int(item)
            for user_id in target_user_ids
            for item in itemcf_user_history.get(int(user_id), [])
            if int(item) in item_to_col
        }
    )
    seed_cols = np.array([item_to_col[item] for item in seed_items], dtype=np.int32)
    item_user_counts = np.asarray(matrix.sum(axis=0)).ravel()
    print(
        f"ItemCF matrix={matrix.shape}, nnz={matrix.nnz:,}, "
        f"target seed items={len(seed_items):,}",
        flush=True,
    )
    co_matrix = (matrix[:, seed_cols].T @ matrix).tocsr()
    print(
        f"ItemCF co-occurrence matrix={co_matrix.shape}, "
        f"nnz={co_matrix.nnz:,}",
        flush=True,
    )

    neighbors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed_position, seed_item in enumerate(seed_items):
        start = co_matrix.indptr[seed_position]
        end = co_matrix.indptr[seed_position + 1]
        candidate_cols = co_matrix.indices[start:end]
        co_counts = co_matrix.data[start:end]
        seed_col = seed_cols[seed_position]
        keep = candidate_cols != seed_col
        candidate_cols = candidate_cols[keep]
        co_counts = co_counts[keep]
        if len(candidate_cols) == 0:
            neighbors[seed_item] = (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
            continue
        similarities = co_counts / np.sqrt(
            item_user_counts[seed_col] * item_user_counts[candidate_cols]
        )
        if len(similarities) > ITEMCF_NEIGHBORS_PER_SEED:
            positions = np.argpartition(
                similarities, -ITEMCF_NEIGHBORS_PER_SEED
            )[-ITEMCF_NEIGHBORS_PER_SEED :]
            positions = positions[np.argsort(similarities[positions])[::-1]]
        else:
            positions = np.argsort(similarities)[::-1]
        neighbors[seed_item] = (
            np.asarray(itemcf_item_ids[candidate_cols[positions]], dtype=np.int64),
            np.asarray(similarities[positions], dtype=np.float32),
        )
    stats = {
        "interactions": int(len(itemcf_train)),
        "users": int(itemcf_train.user_id.nunique()),
        "items": int(itemcf_train.item_id.nunique()),
        "matrix_shape": [int(value) for value in matrix.shape],
        "matrix_nnz": int(matrix.nnz),
        "target_seed_items": int(len(seed_items)),
        "cooccurrence_nnz": int(co_matrix.nnz),
    }
    return itemcf_user_history, neighbors, stats


def build_itemcf_source_frame(
    user_ids: np.ndarray,
    itemcf_user_history: dict[int, list[int]],
    neighbors: dict[int, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    user_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for position, user_id in enumerate(user_ids, 1):
        accumulated: dict[int, float] = defaultdict(float)
        for history_item in itemcf_user_history.get(int(user_id), []):
            neighbor_items, neighbor_scores = neighbors.get(
                int(history_item),
                (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)),
            )
            for item_id, score in zip(neighbor_items, neighbor_scores):
                accumulated[int(item_id)] += float(score)
        ranked = sorted(
            accumulated.items(), key=lambda pair: pair[1], reverse=True
        )[:ITEMCF_QUOTA]
        if ranked:
            length = len(ranked)
            user_parts.append(np.full(length, int(user_id), dtype=np.int64))
            item_parts.append(
                np.fromiter((item for item, _ in ranked), dtype=np.int64)
            )
            rank_parts.append(np.arange(1, length + 1, dtype=np.int32))
            score_parts.append(
                np.fromiter((score for _, score in ranked), dtype=np.float32)
            )
        if position % 2_000 == 0:
            print(
                f"ItemCF recommendations: {position:,}/{len(user_ids):,} users",
                flush=True,
            )
    if not user_parts:
        return empty_source_frame("itemcf")
    return pd.DataFrame(
        {
            "user_id": np.concatenate(user_parts),
            "item_id": np.concatenate(item_parts),
            "itemcf_rank": np.concatenate(rank_parts),
            "itemcf_score": np.concatenate(score_parts),
        }
    )


@torch.inference_mode()
def encode_all(
    encoder: nn.Module, count: int, batch_size: int = ENCODE_BATCH_SIZE
) -> np.ndarray:
    parts = []
    for start in range(0, count, batch_size):
        indices = torch.arange(start, min(start + batch_size, count))
        parts.append(encoder(indices).numpy().astype("float32", copy=False))
    return np.concatenate(parts)


def exact_chunked_topk(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    user_indices: np.ndarray,
    top_k: int,
    user_batch_size: int = USER_BATCH_SIZE,
    item_chunk_size: int = ITEM_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    item_tensor = torch.from_numpy(item_embeddings)
    result_scores: list[np.ndarray] = []
    result_indices: list[np.ndarray] = []
    for user_start in range(0, len(user_indices), user_batch_size):
        batch_indices = user_indices[user_start : user_start + user_batch_size]
        user_tensor = torch.from_numpy(user_embeddings[batch_indices])
        batch_size = len(batch_indices)
        best_scores = torch.full((batch_size, top_k), -torch.inf)
        best_indices = torch.full((batch_size, top_k), -1, dtype=torch.long)
        for item_start in range(0, len(item_embeddings), item_chunk_size):
            item_end = min(item_start + item_chunk_size, len(item_embeddings))
            scores = user_tensor @ item_tensor[item_start:item_end].T
            local_scores, local_indices = torch.topk(
                scores, min(top_k, item_end - item_start), dim=1
            )
            local_indices += item_start
            merged_scores = torch.cat([best_scores, local_scores], dim=1)
            merged_indices = torch.cat([best_indices, local_indices], dim=1)
            best_scores, positions = torch.topk(merged_scores, top_k, dim=1)
            best_indices = torch.gather(merged_indices, 1, positions)
        result_scores.append(best_scores.numpy())
        result_indices.append(best_indices.numpy())
        completed = min(user_start + user_batch_size, len(user_indices))
        if completed % 1_024 < user_batch_size or completed == len(user_indices):
            print(
                f"Two-Tower exact Top-{top_k}: {completed:,}/{len(user_indices):,} users",
                flush=True,
            )
    return np.vstack(result_scores), np.vstack(result_indices)


def build_two_tower_source_frame(
    target_user_ids: np.ndarray,
    user_mapping: pd.DataFrame,
    item_mapping: pd.DataFrame,
    v1_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    user2idx = dict(
        zip(user_mapping.user_id.astype(int), user_mapping.user_idx.astype(int))
    )
    known_user_ids = np.array(
        [int(user_id) for user_id in target_user_ids if int(user_id) in user2idx],
        dtype=np.int64,
    )
    known_user_indices = np.array(
        [user2idx[int(user_id)] for user_id in known_user_ids], dtype=np.int64
    )
    checkpoint = torch.load(
        V1_CHECKPOINT_PATH, map_location="cpu", weights_only=True
    )
    model = TwoTowerModel(
        int(checkpoint["num_users"]),
        int(checkpoint["num_items"]),
        int(checkpoint["embedding_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["output_dim"]),
        float(checkpoint["temperature"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    user_embeddings = encode_all(model.encode_users, int(checkpoint["num_users"]))
    item_embeddings = encode_all(model.encode_items, int(checkpoint["num_items"]))
    scores, item_indices = exact_chunked_topk(
        user_embeddings,
        item_embeddings,
        known_user_indices,
        TWO_TOWER_QUOTA,
    )
    candidate_item_ids = item_mapping.sort_values("item_idx").item_id.to_numpy(
        dtype=np.int64
    )
    frame = pd.DataFrame(
        {
            "user_id": np.repeat(known_user_ids, TWO_TOWER_QUOTA),
            "item_id": candidate_item_ids[item_indices.reshape(-1)],
            "two_tower_rank": np.tile(
                np.arange(1, TWO_TOWER_QUOTA + 1, dtype=np.int32),
                len(known_user_ids),
            ),
            "two_tower_score": scores.reshape(-1).astype(np.float32),
        }
    )
    stats = {
        "checkpoint": str(V1_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "checkpoint_sha256": sha256_file(V1_CHECKPOINT_PATH),
        "checkpoint_strict_load": True,
        "model_seed": int(v1_config["seed"]),
        "target_users": int(len(target_user_ids)),
        "known_users": int(len(known_user_ids)),
        "cold_users": int(len(target_user_ids) - len(known_user_ids)),
        "user_embedding_shape": list(user_embeddings.shape),
        "item_embedding_shape": list(item_embeddings.shape),
    }
    return frame, stats


def build_popularity_template(train_df: pd.DataFrame) -> pd.DataFrame:
    popularity = train_df.groupby("item_id").size().sort_values(ascending=False)
    top = popularity.head(POPULARITY_QUOTA)
    return pd.DataFrame(
        {
            "item_id": top.index.to_numpy(dtype=np.int64),
            "popularity_rank": np.arange(1, len(top) + 1, dtype=np.int32),
            "popularity_score": top.to_numpy(dtype=np.float32),
        }
    )


def build_popularity_source_frame(
    user_ids: np.ndarray, template: pd.DataFrame
) -> pd.DataFrame:
    count = len(template)
    return pd.DataFrame(
        {
            "user_id": np.repeat(user_ids.astype(np.int64), count),
            "item_id": np.tile(template.item_id.to_numpy(dtype=np.int64), len(user_ids)),
            "popularity_rank": np.tile(
                template.popularity_rank.to_numpy(dtype=np.int32), len(user_ids)
            ),
            "popularity_score": np.tile(
                template.popularity_score.to_numpy(dtype=np.float32), len(user_ids)
            ),
        }
    )


def generate_split_candidates(
    user_ids: np.ndarray,
    itemcf_source: pd.DataFrame,
    two_tower_source: pd.DataFrame,
    popularity_template: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate candidates without accepting or reading any label argument."""
    user_ids = np.sort(np.unique(user_ids.astype(np.int64)))
    user_set = set(user_ids.tolist())
    itemcf = itemcf_source.loc[itemcf_source.user_id.isin(user_set)].copy()
    two_tower = two_tower_source.loc[
        two_tower_source.user_id.isin(user_set)
    ].copy()
    popularity = build_popularity_source_frame(user_ids, popularity_template)
    candidates = fuse_source_frames(itemcf, two_tower, popularity)
    checks = validate_candidate_table(candidates, user_ids)
    return candidates, checks


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
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(recommendations[:k], 1)
        if item in ground_truth
    )
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(ground_truth), k) + 1)
    )
    return dcg / idcg if idcg else 0.0


def evaluate_ranked_recommendations(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    ks: Iterable[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in ks:
        recalls = []
        hitrates = []
        ndcgs = []
        for user_id, targets in ground_truth.items():
            recs = recommendations.get(int(user_id), [])
            recalls.append(recall_at_k(recs, targets, int(k)))
            hitrates.append(hitrate_at_k(recs, targets, int(k)))
            ndcgs.append(ndcg_at_k(recs, targets, int(k)))
        rows.append(
            {
                "K": int(k),
                "Recall": float(np.mean(recalls)),
                "HitRate": float(np.mean(hitrates)),
                "NDCG": float(np.mean(ndcgs)),
            }
        )
    return rows


def recommendations_for_source(
    candidates: pd.DataFrame, source: str, evaluation_users: set[int]
) -> dict[int, list[int]]:
    subset = candidates.loc[
        candidates.user_id.isin(evaluation_users)
        & candidates[f"from_{source}"].eq(1),
        ["user_id", "item_id", f"{source}_rank"],
    ].sort_values(["user_id", f"{source}_rank"])
    return {
        int(user_id): group.item_id.astype(int).tolist()
        for user_id, group in subset.groupby("user_id", sort=False)
    }


def recommendations_for_fusion(
    candidates: pd.DataFrame, method: str, evaluation_users: set[int]
) -> dict[int, list[int]]:
    subset = candidates.loc[
        candidates.user_id.isin(evaluation_users)
    ].copy()
    if method == "rrf":
        subset = subset.sort_values(
            ["user_id", "rrf_score", "item_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    elif method == "itemcf_first":
        subset["_not_itemcf"] = 1 - subset.from_itemcf
        subset["_itemcf_rank"] = subset.itemcf_rank.fillna(
            ITEMCF_QUOTA + 1
        )
        subset = subset.sort_values(
            ["user_id", "_not_itemcf", "_itemcf_rank", "rrf_score", "item_id"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    return {
        int(user_id): group.item_id.astype(int).tolist()
        for user_id, group in subset.groupby("user_id", sort=False)
    }


def oracle_candidate_metrics(
    recommendations: dict[int, list[int]], ground_truth: dict[int, set[int]]
) -> dict[str, Any]:
    recalls = []
    hitrates = []
    hit_interactions = 0
    total_interactions = 0
    for user_id, targets in ground_truth.items():
        recalled = set(recommendations.get(int(user_id), []))
        hits = len(recalled & targets)
        recalls.append(hits / len(targets) if targets else 0.0)
        hitrates.append(float(hits > 0))
        hit_interactions += hits
        total_interactions += len(targets)
    return {
        "Recall": float(np.mean(recalls)),
        "HitRate": float(np.mean(hitrates)),
        "hit_interactions": int(hit_interactions),
        "total_interactions": int(total_interactions),
        "interaction_recall": (
            float(hit_interactions / total_interactions)
            if total_interactions
            else 0.0
        ),
    }


def source_overlap(candidates: pd.DataFrame) -> dict[str, Any]:
    source_counts = {
        source: int(candidates[f"from_{source}"].sum()) for source in SOURCE_NAMES
    }
    result: dict[str, Any] = {
        "candidate_pairs_by_source": source_counts,
        "union_candidate_pairs": int(len(candidates)),
        "pairwise": {},
    }
    pairs = [
        ("itemcf", "two_tower"),
        ("itemcf", "popularity"),
        ("two_tower", "popularity"),
    ]
    for first, second in pairs:
        intersection = int(
            (
                candidates[f"from_{first}"].eq(1)
                & candidates[f"from_{second}"].eq(1)
            ).sum()
        )
        union = source_counts[first] + source_counts[second] - intersection
        result["pairwise"][f"{first}__{second}"] = {
            "candidate_pair_count": intersection,
            "fraction_of_first": (
                intersection / source_counts[first] if source_counts[first] else 0.0
            ),
            "fraction_of_second": (
                intersection / source_counts[second]
                if source_counts[second]
                else 0.0
            ),
            "jaccard": intersection / union if union else 0.0,
        }
    return result


def source_contribution(
    candidates: pd.DataFrame, ground_truth_frame: pd.DataFrame
) -> dict[str, Any]:
    flags = candidates[
        [
            "user_id",
            "item_id",
            "from_itemcf",
            "from_two_tower",
            "from_popularity",
        ]
    ]
    joined = ground_truth_frame[["user_id", "item_id"]].drop_duplicates().merge(
        flags, how="left", on=["user_id", "item_id"], validate="one_to_one"
    )
    for column in ("from_itemcf", "from_two_tower", "from_popularity"):
        joined[column] = joined[column].fillna(0).astype("int8")

    i = joined.from_itemcf.eq(1)
    t = joined.from_two_tower.eq(1)
    p = joined.from_popularity.eq(1)
    categories = {
        "only_itemcf": i & ~t & ~p,
        "only_two_tower": ~i & t & ~p,
        "only_popularity": ~i & ~t & p,
        "itemcf_two_tower": i & t & ~p,
        "itemcf_popularity": i & ~t & p,
        "two_tower_popularity": ~i & t & p,
        "all_three": i & t & p,
        "not_recalled": ~i & ~t & ~p,
    }
    counts = {name: int(mask.sum()) for name, mask in categories.items()}
    total = len(joined)
    if sum(counts.values()) != total:
        raise AssertionError("Ground-truth contribution categories are not exhaustive")
    tt_unique = int((t & ~i).sum())
    popularity_unique = int((p & ~i & ~t).sum())
    itemcf_misses = int((~i).sum())
    return {
        "evaluation_interactions": int(total),
        "exclusive_combination_counts": counts,
        "exclusive_combination_rates": {
            name: count / total if total else 0.0 for name, count in counts.items()
        },
        "two_tower_unique_gt_hits": tt_unique,
        "two_tower_unique_gt_hit_rate": tt_unique / total if total else 0.0,
        "two_tower_unique_rate_among_itemcf_misses": (
            tt_unique / itemcf_misses if itemcf_misses else 0.0
        ),
        "only_popularity_gt_hits": popularity_unique,
        "only_popularity_gt_hit_rate": popularity_unique / total if total else 0.0,
    }


def candidate_statistics(
    candidates: pd.DataFrame, expected_users: np.ndarray
) -> dict[str, Any]:
    counts = candidates.groupby("user_id").size().reindex(expected_users, fill_value=0)
    itemcf_users = set(
        candidates.loc[candidates.from_itemcf.eq(1), "user_id"].astype(int)
    )
    tt_users = set(
        candidates.loc[candidates.from_two_tower.eq(1), "user_id"].astype(int)
    )
    expected = set(expected_users.astype(int))
    popularity_only = expected - (itemcf_users | tt_users)
    missing_itemcf = expected - itemcf_users
    missing_tt = expected - tt_users
    missing_either = missing_itemcf | missing_tt
    return {
        "users": int(len(expected_users)),
        "candidate_pairs": int(len(candidates)),
        "average_candidates_per_user": float(counts.mean()),
        "median_candidates_per_user": float(counts.median()),
        "p90_candidates_per_user": float(counts.quantile(0.90)),
        "p95_candidates_per_user": float(counts.quantile(0.95)),
        "min_candidates_per_user": int(counts.min()),
        "max_candidates_per_user": int(counts.max()),
        "theoretical_max_candidates_per_user": MAX_CANDIDATES_PER_USER,
        "popularity_only_fallback_users": int(len(popularity_only)),
        "popularity_only_fallback_fraction": float(
            len(popularity_only) / len(expected) if expected else 0.0
        ),
        "users_missing_itemcf": int(len(missing_itemcf)),
        "users_missing_two_tower": int(len(missing_tt)),
        "users_missing_either_personalized_source": int(len(missing_either)),
    }


def ground_truth_dict(frame: pd.DataFrame) -> dict[int, set[int]]:
    return frame.groupby("user_id").item_id.apply(set).to_dict()


def evaluate_split(
    split_name: str,
    candidates: pd.DataFrame,
    ground_truth_frame: pd.DataFrame,
    user2idx: dict[int, int],
    item2idx: dict[int, int],
    train_item_ids: set[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ground_truth_frame = ground_truth_frame[["user_id", "item_id"]].drop_duplicates()
    all_ground_truth = ground_truth_dict(ground_truth_frame)
    user_seen = ground_truth_frame.user_id.isin(user2idx)
    item_seen = ground_truth_frame.item_id.isin(item2idx)
    warm_frame = ground_truth_frame.loc[user_seen & item_seen].copy()
    warm_ground_truth = ground_truth_dict(warm_frame)
    evaluation_users = set(int(user) for user in warm_ground_truth)

    source_metrics: dict[str, Any] = {}
    source_recommendations: dict[str, dict[int, list[int]]] = {}
    for source in SOURCE_NAMES:
        recs = recommendations_for_source(candidates, source, evaluation_users)
        source_recommendations[source] = recs
        defined_ks = [
            k for k in (50, 100, 200) if k <= SOURCE_QUOTAS[source]
        ]
        ranked_metrics = evaluate_ranked_recommendations(
            recs, warm_ground_truth, defined_ks
        )
        users_with_candidates = len(set(recs) & evaluation_users)
        unique_items = len({item for items in recs.values() for item in items})
        source_metrics[source] = {
            "candidate_quota": SOURCE_QUOTAS[source],
            "ranked_metrics": ranked_metrics,
            "full_quota_oracle": oracle_candidate_metrics(recs, warm_ground_truth),
            "user_coverage": users_with_candidates / len(evaluation_users),
            "users_with_candidates": users_with_candidates,
            "unique_candidate_items": unique_items,
            "catalog_coverage_vs_train_items": unique_items / len(train_item_ids),
        }

    union_recommendations = {
        int(user_id): group.item_id.astype(int).tolist()
        for user_id, group in candidates.loc[
            candidates.user_id.isin(set(all_ground_truth))
        ].groupby("user_id", sort=False)
    }
    warm_union_recommendations = {
        user_id: union_recommendations.get(user_id, [])
        for user_id in warm_ground_truth
    }
    union_oracle = oracle_candidate_metrics(
        warm_union_recommendations, warm_ground_truth
    )
    all_gt_oracle = oracle_candidate_metrics(
        union_recommendations, all_ground_truth
    )

    itemcf_ranked = evaluate_ranked_recommendations(
        source_recommendations["itemcf"], warm_ground_truth, (10, 20, 50)
    )
    rrf_recommendations = recommendations_for_fusion(
        candidates, "rrf", evaluation_users
    )
    itemcf_first_recommendations = recommendations_for_fusion(
        candidates, "itemcf_first", evaluation_users
    )
    fusion_metrics = {
        "itemcf": itemcf_ranked,
        "rrf": evaluate_ranked_recommendations(
            rrf_recommendations, warm_ground_truth, (10, 20, 50)
        ),
        "itemcf_first": evaluate_ranked_recommendations(
            itemcf_first_recommendations, warm_ground_truth, (10, 20, 50)
        ),
    }

    cold_target_mask = ~ground_truth_frame.item_id.isin(train_item_ids)
    cold_target_count = int(cold_target_mask.sum())
    tt_unseen_target_count = int((~item_seen).sum())
    metrics = {
        "split": split_name,
        "protocol": "warm purchase exact evaluation; no historical-item filtering",
        "ground_truth": {
            "all_users": int(ground_truth_frame.user_id.nunique()),
            "all_interactions": int(len(ground_truth_frame)),
            "warm_users": int(len(warm_ground_truth)),
            "warm_interactions": int(len(warm_frame)),
            "warm_user_coverage": float(
                len(warm_ground_truth) / ground_truth_frame.user_id.nunique()
            ),
            "warm_interaction_coverage": float(
                len(warm_frame) / len(ground_truth_frame)
            ),
            "unretrievable_cold_target_interactions": cold_target_count,
            "unretrievable_cold_target_fraction": float(
                cold_target_count / len(ground_truth_frame)
            ),
            "targets_outside_two_tower_vocabulary": tt_unseen_target_count,
            "targets_outside_two_tower_vocabulary_fraction": float(
                tt_unseen_target_count / len(ground_truth_frame)
            ),
        },
        "source_performance": source_metrics,
        "union_oracle_warm": union_oracle,
        "union_oracle_all_gt": all_gt_oracle,
        "fusion_ranking": fusion_metrics,
    }
    contribution = source_contribution(candidates, warm_frame)
    return metrics, contribution


def find_metric(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    return next(row for row in rows if int(row["K"]) == int(k))


def preflight() -> dict[str, Any]:
    required = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
        VALIDATION_GT_PATH,
        TEST_GT_PATH,
        USER_MAPPING_PATH,
        ITEM_MAPPING_PATH,
        MAPPING_METADATA_PATH,
        V1_CONFIG_PATH,
        V1_CHECKPOINT_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIR}")

    mapping_metadata = json_read(MAPPING_METADATA_PATH)
    v1_config = json_read(V1_CONFIG_PATH)
    train_sha = sha256_file(TRAIN_PATH)
    if train_sha != mapping_metadata["train_csv_sha256"]:
        raise AssertionError("train.csv differs from persisted mapping metadata")
    if train_sha != v1_config["mapping"]["train_csv_sha256"]:
        raise AssertionError("train.csv differs from V1 checkpoint config reference")
    locked_expectations = {
        "seed": 42,
        "experiment_name": "TT_V1_BUY_UNIFORM",
        "candidate_count": 302_016,
        "embedding_dim": 64,
        "hidden_dim": 128,
        "output_dim": 64,
        "temperature": 0.1,
    }
    for key, expected in locked_expectations.items():
        if v1_config.get(key) != expected:
            raise AssertionError(
                f"Unexpected original V1 {key}: {v1_config.get(key)!r}"
            )

    generation_parameters = inspect.signature(generate_split_candidates).parameters
    forbidden = [
        name
        for name in generation_parameters
        if "label" in name.lower() or "ground_truth" in name.lower() or name == "gt"
    ]
    if forbidden:
        raise AssertionError(
            f"Candidate generation interface accepts label inputs: {forbidden}"
        )

    print("MULTI-STAGE RECALL PREFLIGHT")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Train SHA-256: {train_sha}")
    print(f"ItemCF quota: {ITEMCF_QUOTA}")
    print(f"Two-Tower quota: {TWO_TOWER_QUOTA}")
    print(f"Popularity quota: {POPULARITY_QUOTA}")
    print(f"RRF constant: {RRF_CONSTANT}")
    print("Two-Tower checkpoint: TT_V1_BUY_UNIFORM seed=42")
    print("Historical-item filtering: unchanged / disabled")
    print("Candidate users: validation.csv/test.csv users, not label files")
    print("Test labels: loaded only after candidate Parquet files are saved")
    print("Output directory is new and historical artifacts are read-only")
    return {
        "mapping_metadata": mapping_metadata,
        "v1_config": v1_config,
        "train_sha256": train_sha,
    }


def main() -> None:
    args = parse_args()
    state = preflight()
    if args.preflight_only:
        print("PREFLIGHT ONLY: no candidates were generated and no files were written.")
        return

    started = time.perf_counter()
    protected_before = protected_manifest()
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)
    user_mapping = pd.read_csv(USER_MAPPING_PATH).sort_values("user_idx")
    item_mapping = pd.read_csv(ITEM_MAPPING_PATH).sort_values("item_idx")
    if user_mapping.user_idx.tolist() != list(range(len(user_mapping))):
        raise AssertionError("Persisted user mapping is not contiguous")
    if item_mapping.item_idx.tolist() != list(range(len(item_mapping))):
        raise AssertionError("Persisted item mapping is not contiguous")
    if (len(user_mapping), len(item_mapping)) != (9_836, 302_016):
        raise AssertionError("Persisted Two-Tower vocabulary size changed")

    validation_users = np.sort(
        validation_df.user_id.unique().astype(np.int64)
    )
    test_users = np.sort(test_df.user_id.unique().astype(np.int64))
    all_target_users = np.union1d(validation_users, test_users)
    print(
        f"Candidate user populations: validation={len(validation_users):,}, "
        f"test={len(test_users):,}, union={len(all_target_users):,}",
        flush=True,
    )

    popularity_template = build_popularity_template(train_df)
    itemcf_history, itemcf_neighbors, itemcf_stats = build_itemcf_context(
        train_df, all_target_users
    )
    itemcf_all = build_itemcf_source_frame(
        all_target_users, itemcf_history, itemcf_neighbors
    )
    print(f"ItemCF source pairs: {len(itemcf_all):,}", flush=True)
    two_tower_all, two_tower_stats = build_two_tower_source_frame(
        all_target_users, user_mapping, item_mapping, state["v1_config"]
    )
    print(f"Two-Tower source pairs: {len(two_tower_all):,}", flush=True)

    OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
    config = {
        "stage": "multi-stage recall candidate union",
        "sources": {
            "itemcf": {
                "quota": ITEMCF_QUOTA,
                "max_history": ITEMCF_MAX_HISTORY,
                "neighbors_per_seed": ITEMCF_NEIGHBORS_PER_SEED,
                "similarity": "binary user-item cosine-style co-occurrence",
                "history_data": "train.csv only; all behaviors; last 50 unique items per user",
            },
            "two_tower": {
                "quota": TWO_TOWER_QUOTA,
                "experiment": "TT_V1_BUY_UNIFORM",
                "seed": 42,
                "checkpoint": str(V1_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
                "exact_retrieval": {
                    "user_batch_size": USER_BATCH_SIZE,
                    "item_chunk_size": ITEM_CHUNK_SIZE,
                    "candidate_items": int(len(item_mapping)),
                },
            },
            "popularity": {
                "quota": POPULARITY_QUOTA,
                "score": "train.csv all-behavior interaction count",
            },
        },
        "rrf": {
            "constant": RRF_CONSTANT,
            "formula": "sum(1 / (60 + source_rank)) over present sources only",
        },
        "rank_convention": "one-based; missing ranks/scores are null",
        "history_filtering": False,
        "history_filtering_note": "Historical-item filtering is intentionally unchanged to preserve protocol comparability.",
        "candidate_user_sources": {
            "validation": "unique user_id from validation.csv",
            "test": "unique user_id from test.csv",
        },
        "label_usage": {
            "candidate_generation_reads_validation_labels": False,
            "candidate_generation_reads_test_labels": False,
            "ground_truth_loaded_after_both_candidate_files_saved": True,
            "test_labels_used_for_evaluation_only": True,
        },
        "evaluation": {
            "primary": "warm purchase GT: user and target item both in persisted Two-Tower train-PV vocabulary",
            "secondary": "all purchase GT oracle with cold-target limitations reported",
            "fusion_methods": ["RRF", "ItemCF-first"],
        },
        "data": {
            "train": str(TRAIN_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "train_sha256": state["train_sha256"],
            "validation_users": int(len(validation_users)),
            "test_users": int(len(test_users)),
        },
        "mapping": state["v1_config"]["mapping"],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    json_write(OUTPUT_DIR / "config.json", config)

    validation_candidates, validation_checks = generate_split_candidates(
        validation_users, itemcf_all, two_tower_all, popularity_template
    )
    validation_candidate_path = OUTPUT_DIR / "validation_candidates.parquet"
    validation_candidates.to_parquet(
        validation_candidate_path, index=False, engine="pyarrow"
    )
    print(
        f"Saved validation candidates: {len(validation_candidates):,} rows",
        flush=True,
    )

    test_candidates, test_checks = generate_split_candidates(
        test_users, itemcf_all, two_tower_all, popularity_template
    )
    test_candidate_path = OUTPUT_DIR / "test_candidates.parquet"
    test_candidates.to_parquet(test_candidate_path, index=False, engine="pyarrow")
    print(f"Saved test candidates: {len(test_candidates):,} rows", flush=True)
    print(
        "Both candidate files are saved. Loading validation/test labels now for evaluation only.",
        flush=True,
    )

    validation_gt = pd.read_csv(VALIDATION_GT_PATH)
    test_gt = pd.read_csv(TEST_GT_PATH)
    user2idx = dict(
        zip(user_mapping.user_id.astype(int), user_mapping.user_idx.astype(int))
    )
    item2idx = dict(
        zip(item_mapping.item_id.astype(int), item_mapping.item_idx.astype(int))
    )
    train_item_ids = set(train_df.item_id.astype(int).unique())

    validation_metrics, validation_contribution = evaluate_split(
        "validation",
        validation_candidates,
        validation_gt,
        user2idx,
        item2idx,
        train_item_ids,
    )
    test_metrics, test_contribution = evaluate_split(
        "test",
        test_candidates,
        test_gt,
        user2idx,
        item2idx,
        train_item_ids,
    )

    # Notebook 04 persisted these baselines only at six-decimal display
    # precision for Popularity/ItemCF.  Reproduction is therefore checked
    # against the published values with a 1e-6 absolute tolerance; the exact
    # values produced in this run are retained in the metrics artifacts.
    expected_validation_r50 = {
        "popularity": 0.007123,
        "itemcf": 0.371696,
        "two_tower": 0.011966,
    }
    reproduced_validation_r50: dict[str, Any] = {}
    for source, expected in expected_validation_r50.items():
        actual = float(
            find_metric(
                validation_metrics["source_performance"][source]["ranked_metrics"],
                50,
            )["Recall"]
        )
        reproduced_validation_r50[source] = {
            "published_reference": expected,
            "actual": actual,
            "absolute_delta": actual - expected,
            "absolute_tolerance": 1e-6,
        }
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
            raise AssertionError(
                f"{source} validation Recall@50 changed: expected={expected}, actual={actual}"
            )

    overlap = {
        "validation": source_overlap(validation_candidates),
        "test": source_overlap(test_candidates),
    }
    contributions = {
        "validation": validation_contribution,
        "test": test_contribution,
    }
    statistics = {
        "validation": candidate_statistics(
            validation_candidates, validation_users
        ),
        "test": candidate_statistics(test_candidates, test_users),
        "itemcf_build": itemcf_stats,
        "two_tower_build": two_tower_stats,
        "validation_checks": validation_checks,
        "test_checks": test_checks,
    }

    json_write(OUTPUT_DIR / "validation_metrics.json", validation_metrics)
    json_write(OUTPUT_DIR / "test_metrics.json", test_metrics)
    json_write(OUTPUT_DIR / "source_overlap.json", overlap)
    json_write(OUTPUT_DIR / "source_contribution.json", contributions)
    json_write(OUTPUT_DIR / "candidate_statistics.json", statistics)

    validation_warm_mask = validation_gt.user_id.isin(user2idx) & validation_gt.item_id.isin(
        item2idx
    )
    validation_warm_gt = ground_truth_dict(
        validation_gt.loc[validation_warm_mask, ["user_id", "item_id"]]
    )
    sample_user_ids = np.random.default_rng(42).choice(
        np.array(sorted(validation_warm_gt), dtype=np.int64), size=5, replace=False
    )
    sample_rrf = recommendations_for_fusion(
        validation_candidates, "rrf", set(sample_user_ids.astype(int))
    )
    sample_rows = []
    for user_id in sample_user_ids:
        user_rows = validation_candidates.loc[
            validation_candidates.user_id.eq(int(user_id))
        ]
        sample_rows.append(
            {
                "user_id": int(user_id),
                "candidate_count": int(len(user_rows)),
                "rrf_top10": sample_rrf[int(user_id)][:10],
                "actual_purchase_gt": sorted(validation_warm_gt[int(user_id)]),
                "itemcf_candidates": int(user_rows.from_itemcf.sum()),
                "two_tower_candidates": int(user_rows.from_two_tower.sum()),
                "popularity_candidates": int(user_rows.from_popularity.sum()),
            }
        )
    pd.DataFrame(sample_rows).to_csv(
        OUTPUT_DIR / "sample_candidates.csv", index=False
    )

    validation_roundtrip = pd.read_parquet(validation_candidate_path)
    test_roundtrip = pd.read_parquet(test_candidate_path)
    if len(validation_roundtrip) != len(validation_candidates):
        raise AssertionError("Validation Parquet row count changed on round-trip")
    if len(test_roundtrip) != len(test_candidates):
        raise AssertionError("Test Parquet row count changed on round-trip")
    if validation_roundtrip.duplicated(["user_id", "item_id"]).any():
        raise AssertionError("Validation Parquet contains duplicates")
    if test_roundtrip.duplicated(["user_id", "item_id"]).any():
        raise AssertionError("Test Parquet contains duplicates")

    protected_after = protected_manifest()
    if protected_after != protected_before:
        changed = sorted(
            path
            for path in set(protected_before) | set(protected_after)
            if protected_before.get(path) != protected_after.get(path)
        )
        raise RuntimeError(f"Protected historical artifacts changed: {changed}")

    execution_summary = {
        "completed": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "candidate_files_roundtrip_readable": True,
        "historical_artifacts_unchanged": True,
        "protected_artifact_sha256": protected_after,
        "validation_baseline_reproduction": reproduced_validation_r50,
        "no_test_leakage": {
            "candidate_user_ids_from_test_csv": True,
            "test_ground_truth_loaded_after_candidate_artifact_saved": True,
            "test_labels_used_for_rules_or_parameters": False,
        },
    }
    json_write(OUTPUT_DIR / "execution_summary.json", execution_summary)

    val_union = validation_metrics["union_oracle_warm"]
    test_union = test_metrics["union_oracle_warm"]
    val_stats = statistics["validation"]
    val_tt_unique = validation_contribution["two_tower_unique_gt_hits"]
    print("\nMULTI-STAGE RECALL COMPLETE")
    print(
        f"Validation warm union oracle Recall={val_union['Recall']:.6f}, "
        f"HitRate={val_union['HitRate']:.6f}"
    )
    print(
        f"Test warm union oracle Recall={test_union['Recall']:.6f}, "
        f"HitRate={test_union['HitRate']:.6f}"
    )
    print(
        f"Validation candidates/user mean={val_stats['average_candidates_per_user']:.3f}, "
        f"median={val_stats['median_candidates_per_user']:.1f}, "
        f"p95={val_stats['p95_candidates_per_user']:.1f}, "
        f"max={val_stats['max_candidates_per_user']}"
    )
    print(f"Validation Two-Tower unique GT hits beyond ItemCF={val_tt_unique}")
    print("Historical-item filtering is intentionally unchanged to preserve protocol comparability.")
    print("Protected historical artifacts unchanged: OK")
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
