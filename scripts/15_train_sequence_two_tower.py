"""Train the post-freeze V3 controlled retrieval extension.

This script is intentionally restricted to data/processed/dev/train.csv.  It
creates an internal three-way temporal protocol, trains an ID-only and a
sequence-aware dual encoder with the same in-batch logQ objective, selects
checkpoints on the internal selection day, freezes configuration, and then
evaluates the internal held-forward day exactly once.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "dev" / "train.csv"
OUTPUT_DIR = ROOT / "artifacts" / "v3_sequence_retrieval"
A_DIR = OUTPUT_DIR / "V3A"
B_DIR = OUTPUT_DIR / "V3B"

SEED = 42
EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
TEMPERATURE = 0.1
HISTORY_LENGTH = 50
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 64
BEHAVIOR_EMBEDDING_DIM = 64
TOP_K = 50
USER_BATCH_SIZE = 64
ITEM_CHUNK_SIZE = 50_000
ENCODE_BATCH_SIZE = 8_192
ITEMCF_NEIGHBORS_PER_SEED = 200
LOGQ_EPS = 1e-12
BEHAVIOR_TO_INDEX = {"pv": 1, "fav": 2, "cart": 3, "buy": 4}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    columns = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
    values = pd.util.hash_pandas_object(frame[columns], index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_shanghai_date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["calendar_date"] = (
        pd.to_datetime(result.timestamp, unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%d")
    )
    return result


def date_statistics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("calendar_date", sort=True):
        counts = group.behavior_type.value_counts()
        rows.append(
            {
                "date": str(date),
                "rows": int(len(group)),
                "users": int(group.user_id.nunique()),
                "items": int(group.item_id.nunique()),
                "pv": int(counts.get("pv", 0)),
                "fav": int(counts.get("fav", 0)),
                "cart": int(counts.get("cart", 0)),
                "buy": int(counts.get("buy", 0)),
            }
        )
    return rows


def build_temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dated = add_shanghai_date(frame)
    dates = sorted(dated.calendar_date.unique().tolist())
    if len(dates) < 3:
        raise AssertionError("V3 requires at least three distinct Asia/Shanghai calendar dates")
    train_dates = dates[:-2]
    select_date = dates[-2]
    eval_date = dates[-1]
    train = dated.loc[dated.calendar_date.isin(train_dates)].copy()
    select = dated.loc[dated.calendar_date.eq(select_date)].copy()
    evaluation = dated.loc[dated.calendar_date.eq(eval_date)].copy()
    if set(train.index) & set(select.index) or set(train.index) & set(evaluation.index) or set(select.index) & set(evaluation.index):
        raise AssertionError("Internal temporal split indices overlap")
    if not (train.timestamp.max() < select.timestamp.min() <= select.timestamp.max() < evaluation.timestamp.min()):
        raise AssertionError("Internal temporal split is not strictly chronological")
    manifest = {
        "timezone": "Asia/Shanghai",
        "rule": "all dates except last two -> V3_TRAIN; penultimate date -> V3_SELECT; last date -> V3_EVAL",
        "original_train_path": "data/processed/dev/train.csv",
        "original_train_sha256": sha256_file(TRAIN_PATH) if TRAIN_PATH.exists() else None,
        "original_min_timestamp": int(dated.timestamp.min()),
        "original_max_timestamp": int(dated.timestamp.max()),
        "date_statistics": date_statistics(dated),
        "v3_train_dates": train_dates,
        "v3_select_date": select_date,
        "v3_eval_date": eval_date,
        "splits": {
            "train": {"rows": int(len(train)), "buy_rows": int(train.behavior_type.eq("buy").sum()), "sha256": frame_sha256(train)},
            "select": {"rows": int(len(select)), "buy_rows": int(select.behavior_type.eq("buy").sum()), "sha256": frame_sha256(select)},
            "eval": {"rows": int(len(evaluation)), "buy_rows": int(evaluation.behavior_type.eq("buy").sum()), "sha256": frame_sha256(evaluation)},
        },
    }
    return train, select, evaluation, manifest


def stable_mappings(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int], dict[int, int]]:
    user_ids = pd.unique(train.user_id).astype(np.int64, copy=False)
    item_ids = pd.unique(train.item_id).astype(np.int64, copy=False)
    user_frame = pd.DataFrame({"user_id": user_ids, "user_idx": np.arange(len(user_ids), dtype=np.int64)})
    item_frame = pd.DataFrame({"item_id": item_ids, "item_idx": np.arange(len(item_ids), dtype=np.int64)})
    user_to_idx = dict(zip(user_frame.user_id.astype(int), user_frame.user_idx.astype(int)))
    item_to_idx = dict(zip(item_frame.item_id.astype(int), item_frame.item_idx.astype(int)))
    return user_frame, item_frame, user_to_idx, item_to_idx


@dataclass
class TrainingExamples:
    user_indices: np.ndarray
    history_items: np.ndarray
    history_behaviors: np.ndarray
    history_lengths: np.ndarray
    target_items: np.ndarray
    target_timestamps: np.ndarray
    max_history_timestamps: np.ndarray


def build_training_examples(
    train: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    history_length: int,
) -> tuple[TrainingExamples, dict[str, Any], np.ndarray]:
    pad_item_idx = len(item_to_idx)
    user_indices: list[int] = []
    histories: list[np.ndarray] = []
    behaviors: list[np.ndarray] = []
    lengths: list[int] = []
    targets: list[int] = []
    target_times: list[int] = []
    max_history_times: list[int] = []
    target_frequency = np.zeros(len(item_to_idx), dtype=np.int64)
    raw_purchases = int(train.behavior_type.eq("buy").sum())
    mapped_purchases = 0
    users_with_examples: set[int] = set()
    unique_targets: set[int] = set()

    ordered = train.assign(_source_row=train.index.to_numpy()).sort_values(
        ["user_id", "timestamp", "_source_row"], kind="mergesort"
    )
    for user_id, group in ordered.groupby("user_id", sort=False):
        history_item_list: list[int] = []
        history_behavior_list: list[int] = []
        history_time_list: list[int] = []
        for timestamp, same_time in group.groupby("timestamp", sort=True):
            timestamp = int(timestamp)
            purchase_rows = same_time.loc[same_time.behavior_type.eq("buy")]
            for row in purchase_rows.itertuples(index=False):
                target_idx = item_to_idx.get(int(row.item_id))
                if target_idx is None:
                    continue
                mapped_purchases += 1
                target_frequency[target_idx] += 1
                if not history_item_list:
                    continue
                selected_items = history_item_list[-history_length:]
                selected_behaviors = history_behavior_list[-history_length:]
                selected_times = history_time_list[-history_length:]
                item_array = np.full(history_length, pad_item_idx, dtype=np.int64)
                behavior_array = np.zeros(history_length, dtype=np.int64)
                item_array[: len(selected_items)] = selected_items
                behavior_array[: len(selected_behaviors)] = selected_behaviors
                user_indices.append(user_to_idx[int(user_id)])
                histories.append(item_array)
                behaviors.append(behavior_array)
                lengths.append(len(selected_items))
                targets.append(target_idx)
                target_times.append(timestamp)
                max_history_times.append(max(selected_times))
                users_with_examples.add(int(user_id))
                unique_targets.add(target_idx)
            # Timestamp batching: no event at this timestamp is available to a
            # purchase target at the same timestamp.
            for row in same_time.itertuples(index=False):
                item_idx = item_to_idx.get(int(row.item_id))
                behavior_idx = BEHAVIOR_TO_INDEX.get(str(row.behavior_type))
                if item_idx is not None and behavior_idx is not None:
                    history_item_list.append(item_idx)
                    history_behavior_list.append(behavior_idx)
                    history_time_list.append(timestamp)

    if not histories:
        raise AssertionError("No non-empty strict-history V3 training examples")
    arrays = TrainingExamples(
        user_indices=np.asarray(user_indices, dtype=np.int64),
        history_items=np.stack(histories),
        history_behaviors=np.stack(behaviors),
        history_lengths=np.asarray(lengths, dtype=np.int64),
        target_items=np.asarray(targets, dtype=np.int64),
        target_timestamps=np.asarray(target_times, dtype=np.int64),
        max_history_timestamps=np.asarray(max_history_times, dtype=np.int64),
    )
    strict = bool(np.all(arrays.max_history_timestamps < arrays.target_timestamps))
    if not strict:
        raise AssertionError("V3 training history timestamp is not strictly before target")
    if np.any(arrays.history_lengths <= 0):
        raise AssertionError("An empty history entered the shared training subset")
    q = target_frequency.astype(np.float64)
    if q.sum() <= 0:
        raise AssertionError("No mapped V3_TRAIN purchase frequency for logQ")
    q /= q.sum()
    stats = {
        "raw_v3_train_purchase_events": raw_purchases,
        "mapped_target_purchase_events": int(mapped_purchases),
        "non_empty_strict_history_examples": int(len(arrays.target_items)),
        "unique_users": int(len(users_with_examples)),
        "unique_target_items": int(len(unique_targets)),
        "dropped_empty_history": int(mapped_purchases - len(arrays.target_items)),
        "history_length": int(history_length),
        "mean_history_length": float(arrays.history_lengths.mean()),
        "median_history_length": float(np.median(arrays.history_lengths)),
        "max_history_timestamp_strictly_before_target": strict,
        "same_examples_for_v3a_and_v3b": True,
    }
    return arrays, stats, q.astype(np.float32)


@dataclass
class QueryPopulation:
    name: str
    user_ids: np.ndarray
    user_indices: np.ndarray
    history_items: np.ndarray
    history_behaviors: np.ndarray
    history_lengths: np.ndarray
    ground_truth: dict[int, set[int]]
    stats: dict[str, Any]


def build_query_population(
    name: str,
    target_day: pd.DataFrame,
    history: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    history_length: int,
) -> QueryPopulation:
    pad_item_idx = len(item_to_idx)
    raw_purchase_events = int(target_day.behavior_type.eq("buy").sum())
    purchase_pairs = target_day.loc[target_day.behavior_type.eq("buy"), ["user_id", "item_id"]].drop_duplicates()
    total_interactions = int(len(purchase_pairs))
    cold_target_interactions = int((~purchase_pairs.item_id.isin(item_to_idx)).sum())
    candidate_pairs = purchase_pairs.loc[
        purchase_pairs.user_id.isin(user_to_idx) & purchase_pairs.item_id.isin(item_to_idx)
    ].copy()

    requested_users = set(map(int, candidate_pairs.user_id.unique()))
    by_user_items: dict[int, list[int]] = {user: [] for user in requested_users}
    by_user_behaviors: dict[int, list[int]] = {user: [] for user in requested_users}
    ordered = history.loc[history.user_id.isin(requested_users)].assign(
        _source_row=history.loc[history.user_id.isin(requested_users)].index.to_numpy()
    ).sort_values(["user_id", "timestamp", "_source_row"], kind="mergesort")
    if len(ordered) and int(ordered.timestamp.max()) >= int(target_day.timestamp.min()):
        raise AssertionError(f"{name} query history reaches target day")
    for row in ordered.itertuples(index=False):
        item_idx = item_to_idx.get(int(row.item_id))
        behavior_idx = BEHAVIOR_TO_INDEX.get(str(row.behavior_type))
        if item_idx is not None and behavior_idx is not None:
            user = int(row.user_id)
            by_user_items[user].append(item_idx)
            by_user_behaviors[user].append(behavior_idx)
    eligible_users = sorted(user for user in requested_users if by_user_items[user])
    eligible_set = set(eligible_users)
    eligible_pairs = candidate_pairs.loc[candidate_pairs.user_id.isin(eligible_set)].copy()
    ground_truth = {
        int(user): set(map(int, group.item_id))
        for user, group in eligible_pairs.groupby("user_id", sort=False)
    }
    history_items: list[np.ndarray] = []
    history_behaviors: list[np.ndarray] = []
    history_lengths: list[int] = []
    for user in eligible_users:
        items = by_user_items[user][-history_length:]
        behavior_values = by_user_behaviors[user][-history_length:]
        item_array = np.full(history_length, pad_item_idx, dtype=np.int64)
        behavior_array = np.zeros(history_length, dtype=np.int64)
        item_array[: len(items)] = items
        behavior_array[: len(behavior_values)] = behavior_values
        history_items.append(item_array)
        history_behaviors.append(behavior_array)
        history_lengths.append(len(items))
    stats = {
        "protocol": "COMMON WARM POPULATION; purchase user/item pairs; no historical-item filtering",
        "raw_purchase_events": raw_purchase_events,
        "total_purchase_interactions": total_interactions,
        "eligible_interactions": int(len(eligible_pairs)),
        "eligible_users": int(len(eligible_users)),
        "cold_target_interactions": cold_target_interactions,
        "eligible_coverage": float(len(eligible_pairs) / total_interactions) if total_interactions else 0.0,
        "query_history_source_rows": int(len(history)),
        "query_history_strictly_before_target_day": True,
    }
    return QueryPopulation(
        name=name,
        user_ids=np.asarray(eligible_users, dtype=np.int64),
        user_indices=np.asarray([user_to_idx[user] for user in eligible_users], dtype=np.int64),
        history_items=np.stack(history_items) if history_items else np.empty((0, history_length), dtype=np.int64),
        history_behaviors=np.stack(history_behaviors) if history_behaviors else np.empty((0, history_length), dtype=np.int64),
        history_lengths=np.asarray(history_lengths, dtype=np.int64),
        ground_truth=ground_truth,
        stats=stats,
    )


def masked_mean(event_embeddings: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if event_embeddings.ndim != 3 or valid_mask.shape != event_embeddings.shape[:2]:
        raise ValueError("masked_mean shape mismatch")
    counts = valid_mask.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("empty history is not allowed")
    return (event_embeddings * valid_mask.unsqueeze(-1)).sum(dim=1) / counts


class ItemTowerMixin:
    item_embedding: nn.Embedding
    item_mlp: nn.Module

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_mlp(self.item_embedding(item_indices)), p=2, dim=-1)


class IDOnlyDualEncoder(nn.Module, ItemTowerMixin):
    def __init__(self, num_users: int, num_items: int) -> None:
        super().__init__()
        # Item components are initialized first in both models so their initial
        # candidate representation is identical under the shared seed.
        self.item_embedding = nn.Embedding(num_items + 1, EMBEDDING_DIM, padding_idx=num_items)
        self.item_mlp = nn.Sequential(nn.Linear(EMBEDDING_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, OUTPUT_DIM))
        self.user_embedding = nn.Embedding(num_users, EMBEDDING_DIM)
        self.user_mlp = nn.Sequential(nn.Linear(EMBEDDING_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, OUTPUT_DIM))

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_mlp(self.user_embedding(user_indices)), p=2, dim=-1)


class SequenceDualEncoder(nn.Module, ItemTowerMixin):
    def __init__(self, num_items: int) -> None:
        super().__init__()
        self.pad_item_idx = num_items
        self.item_embedding = nn.Embedding(num_items + 1, EMBEDDING_DIM, padding_idx=num_items)
        self.item_mlp = nn.Sequential(nn.Linear(EMBEDDING_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, OUTPUT_DIM))
        self.behavior_embedding = nn.Embedding(len(BEHAVIOR_TO_INDEX) + 1, BEHAVIOR_EMBEDDING_DIM, padding_idx=0)
        self.user_mlp = nn.Sequential(nn.Linear(EMBEDDING_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, OUTPUT_DIM))

    def encode_users(self, history_items: torch.Tensor, history_behaviors: torch.Tensor) -> torch.Tensor:
        valid = history_items.ne(self.pad_item_idx)
        events = self.item_embedding(history_items) + self.behavior_embedding(history_behaviors)
        pooled = masked_mean(events, valid)
        return F.normalize(self.user_mlp(pooled), p=2, dim=-1)


def false_negative_mask(user_indices: torch.Tensor, target_items: torch.Tensor) -> torch.Tensor:
    if user_indices.ndim != 1 or target_items.ndim != 1 or len(user_indices) != len(target_items):
        raise ValueError("false_negative_mask expects aligned one-dimensional inputs")
    same_user = user_indices[:, None].eq(user_indices[None, :])
    same_target = target_items[:, None].eq(target_items[None, :])
    diagonal = torch.eye(len(user_indices), dtype=torch.bool, device=user_indices.device)
    return (same_user | same_target) & ~diagonal


def inbatch_logits_and_labels(
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    user_indices: torch.Tensor,
    target_items: torch.Tensor,
    q_values: torch.Tensor,
    temperature: float,
    eps: float = LOGQ_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = user_vectors @ item_vectors.T / temperature
    corrected = raw - torch.log(q_values.clamp_min(eps))[None, :]
    false_negatives = false_negative_mask(user_indices, target_items)
    corrected = corrected.masked_fill(false_negatives, -torch.inf)
    labels = torch.arange(len(user_indices), device=user_indices.device)
    return corrected, labels, false_negatives


@torch.inference_mode()
def encode_item_catalog(model: ItemTowerMixin, num_items: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, num_items, ENCODE_BATCH_SIZE):
        indices = torch.arange(start, min(start + ENCODE_BATCH_SIZE, num_items))
        parts.append(model.encode_items(indices).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(parts)


@torch.inference_mode()
def encode_query_population(model: nn.Module, kind: str, population: QueryPopulation) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(population.user_ids), ENCODE_BATCH_SIZE):
        end = min(start + ENCODE_BATCH_SIZE, len(population.user_ids))
        if kind == "id":
            values = model.encode_users(torch.from_numpy(population.user_indices[start:end]))
        else:
            values = model.encode_users(
                torch.from_numpy(population.history_items[start:end]),
                torch.from_numpy(population.history_behaviors[start:end]),
            )
        parts.append(values.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(parts) if parts else np.empty((0, OUTPUT_DIM), dtype=np.float32)


@torch.inference_mode()
def exact_chunked_topk(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    top_k: int,
    user_batch_size: int = USER_BATCH_SIZE,
    item_chunk_size: int = ITEM_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    if user_embeddings.ndim != 2 or item_embeddings.ndim != 2:
        raise ValueError("exact retrieval expects 2D embeddings")
    if user_embeddings.shape[1] != item_embeddings.shape[1]:
        raise ValueError("embedding dimensions differ")
    effective_k = min(int(top_k), len(item_embeddings))
    item_tensor = torch.from_numpy(item_embeddings)
    all_scores: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for user_start in range(0, len(user_embeddings), user_batch_size):
        users = torch.from_numpy(user_embeddings[user_start : user_start + user_batch_size])
        best_scores = torch.full((len(users), effective_k), -torch.inf)
        best_indices = torch.full((len(users), effective_k), -1, dtype=torch.long)
        for item_start in range(0, len(item_embeddings), item_chunk_size):
            item_end = min(item_start + item_chunk_size, len(item_embeddings))
            scores = users @ item_tensor[item_start:item_end].T
            local_scores, local_indices = torch.topk(scores, min(effective_k, item_end - item_start), dim=1)
            local_indices += item_start
            merged_scores = torch.cat([best_scores, local_scores], dim=1)
            merged_indices = torch.cat([best_indices, local_indices], dim=1)
            best_scores, positions = torch.topk(merged_scores, effective_k, dim=1)
            best_indices = torch.gather(merged_indices, 1, positions)
        all_scores.append(best_scores.cpu().numpy())
        all_indices.append(best_indices.cpu().numpy())
    return np.vstack(all_scores), np.vstack(all_indices)


def recommendation_dict(population: QueryPopulation, top_indices: np.ndarray, item_ids: np.ndarray) -> dict[int, list[int]]:
    return {
        int(user): [int(item_ids[index]) for index in row if index >= 0]
        for user, row in zip(population.user_ids, top_indices)
    }


def evaluate_recommendations(
    recommendations: dict[int, list[int]], ground_truth: dict[int, set[int]], ks: Iterable[int] = (20, 50)
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for k in ks:
        recalls: list[float] = []
        hitrates: list[float] = []
        ndcgs: list[float] = []
        interaction_hits = 0
        total = 0
        for user, targets in ground_truth.items():
            recs = recommendations.get(int(user), [])[: int(k)]
            hits = set(recs) & targets
            recalls.append(len(hits) / len(targets))
            hitrates.append(float(bool(hits)))
            dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(recs, 1) if item in targets)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(targets), int(k)) + 1))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            interaction_hits += len(hits)
            total += len(targets)
        rows.append(
            {
                "K": int(k),
                "Recall": float(np.mean(recalls)) if recalls else 0.0,
                "HitRate": float(np.mean(hitrates)) if hitrates else 0.0,
                "NDCG": float(np.mean(ndcgs)) if ndcgs else 0.0,
                "hit_interactions": int(interaction_hits),
                "total_interactions": int(total),
                "interaction_recall": float(interaction_hits / total) if total else 0.0,
            }
        )
    return {"users": int(len(ground_truth)), "interactions": int(sum(map(len, ground_truth.values()))), "metrics": rows}


def metric(metrics: dict[str, Any], k: int, name: str) -> float:
    return float(next(row[name] for row in metrics["metrics"] if int(row["K"]) == int(k)))


def evaluate_model(
    model: nn.Module,
    kind: str,
    population: QueryPopulation,
    item_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[int, list[int]], np.ndarray, np.ndarray]:
    model.eval()
    item_vectors = encode_item_catalog(model, len(item_ids))
    user_vectors = encode_query_population(model, kind, population)
    scores, indices = exact_chunked_topk(user_vectors, item_vectors, TOP_K)
    recs = recommendation_dict(population, indices, item_ids)
    metrics = evaluate_recommendations(recs, population.ground_truth)
    return metrics, recs, user_vectors, item_vectors


def base_config(name: str, kind: str, num_users: int, num_items: int, mapping_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "experiment": name,
        "evaluation_scope": "INTERNAL TEMPORAL EVALUATION inside historical train.csv; not protected-test",
        "user_representation": "raw user ID embedding" if kind == "id" else "last-50 behavior-aware shared-item embeddings with masked mean pooling",
        "shared_history_candidate_item_embedding": bool(kind == "sequence"),
        "num_users": int(num_users),
        "num_items": int(num_items),
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "behavior_embedding_dim": BEHAVIOR_EMBEDDING_DIM if kind == "sequence" else None,
        "temperature": TEMPERATURE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "history_length": HISTORY_LENGTH,
        "objective": "in-batch softmax / InfoNCE with duplicate-target and same-user false-negative masking",
        "logq_correction": True,
        "logq_source": "V3_TRAIN purchase target frequency",
        "selection_rule": "maximum internal V3_SELECT Recall@50; NDCG@50 tie-break",
        "retrieval": "chunked exact inner product over full V3_TRAIN item catalog",
        "historical_item_filtering": False,
        "mapping_hashes": mapping_hashes,
    }


def train_model(
    name: str,
    kind: str,
    examples: TrainingExamples,
    q_probabilities: np.ndarray,
    select_population: QueryPopulation,
    item_ids: np.ndarray,
    num_users: int,
    mapping_hashes: dict[str, str],
) -> tuple[nn.Module, dict[str, Any], dict[int, list[int]], list[dict[str, Any]]]:
    output_dir = A_DIR if kind == "id" else B_DIR
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(SEED)
    model: nn.Module = IDOnlyDualEncoder(num_users, len(item_ids)) if kind == "id" else SequenceDualEncoder(len(item_ids))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    config = base_config(name, kind, num_users, len(item_ids), mapping_hashes)
    json_write(output_dir / "config.json", config)
    history_rows: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf)
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    best_recs: dict[int, list[int]] | None = None
    q_tensor = torch.from_numpy(q_probabilities)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        generator = torch.Generator().manual_seed(SEED + epoch)
        permutation = torch.randperm(len(examples.target_items), generator=generator).numpy()
        total_loss = 0.0
        seen = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            batch_indices = permutation[start : start + BATCH_SIZE]
            users = torch.from_numpy(examples.user_indices[batch_indices])
            targets = torch.from_numpy(examples.target_items[batch_indices])
            if kind == "id":
                user_vectors = model.encode_users(users)
            else:
                user_vectors = model.encode_users(
                    torch.from_numpy(examples.history_items[batch_indices]),
                    torch.from_numpy(examples.history_behaviors[batch_indices]),
                )
            item_vectors = model.encode_items(targets)
            logits, labels, _ = inbatch_logits_and_labels(
                user_vectors, item_vectors, users, targets, q_tensor[targets], TEMPERATURE
            )
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} produced non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)

        select_metrics, select_recs, _, _ = evaluate_model(model, kind, select_population, item_ids)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "select_recall_at_20": metric(select_metrics, 20, "Recall"),
            "select_recall_at_50": metric(select_metrics, 50, "Recall"),
            "select_hitrate_at_50": metric(select_metrics, 50, "HitRate"),
            "select_ndcg_at_50": metric(select_metrics, 50, "NDCG"),
        }
        history_rows.append(row)
        key = (row["select_recall_at_50"], row["select_ndcg_at_50"])
        print(
            f"{name} epoch={epoch:02d} loss={row['train_loss']:.6f} "
            f"SELECT R20={row['select_recall_at_20']:.6f} R50={row['select_recall_at_50']:.6f} "
            f"HR50={row['select_hitrate_at_50']:.6f} N50={row['select_ndcg_at_50']:.6f}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = select_metrics
            best_recs = select_recs
            temporary = output_dir / "best.pt.tmp"
            torch.save(
                {"state_dict": model.state_dict(), "config": config, "best_epoch": best_epoch, "selection_metrics": best_metrics},
                temporary,
            )
            temporary.replace(output_dir / "best.pt")

    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)
    if best_metrics is None or best_recs is None:
        raise AssertionError(f"{name} did not select a checkpoint")
    # Strict reload proves the persisted checkpoint is self-consistent.
    reload_model: nn.Module = IDOnlyDualEncoder(num_users, len(item_ids)) if kind == "id" else SequenceDualEncoder(len(item_ids))
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    reload_model.load_state_dict(checkpoint["state_dict"], strict=True)
    best_metrics, best_recs, _, _ = evaluate_model(reload_model, kind, select_population, item_ids)
    select_payload = {
        "experiment": name,
        "split": "V3_SELECT",
        "role": "checkpoint selection only",
        "population": select_population.stats,
        "best_epoch": int(best_epoch),
        "selection_rule": config["selection_rule"],
        "metrics": best_metrics,
        "strict_checkpoint_reload": True,
    }
    json_write(output_dir / "select_metrics.json", select_payload)
    return reload_model, select_payload, best_recs, history_rows


def popularity_recommendations(train: pd.DataFrame, users: np.ndarray, item_to_idx: dict[int, int], top_k: int) -> dict[int, list[int]]:
    counts = train.loc[train.item_id.isin(item_to_idx)].groupby("item_id", sort=False).size()
    ranked = sorted(((int(item), int(count)) for item, count in counts.items()), key=lambda pair: (-pair[1], item_to_idx[pair[0]]))
    template = [item for item, _ in ranked[:top_k]]
    return {int(user): template.copy() for user in users}


def recent_unique_history_indices(history: pd.DataFrame, users: Iterable[int], item_to_idx: dict[int, int]) -> dict[int, list[int]]:
    requested = set(map(int, users))
    subset = history.loc[history.user_id.isin(requested) & history.item_id.isin(item_to_idx)].copy()
    subset = (
        subset.sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset=["user_id", "item_id"], keep="last")
        .sort_values(["user_id", "timestamp"], kind="mergesort")
        .groupby("user_id", group_keys=False)
        .tail(HISTORY_LENGTH)
    )
    result = {int(user): [] for user in requested}
    for user, group in subset.groupby("user_id", sort=False):
        result[int(user)] = [item_to_idx[int(item)] for item in group.item_id]
    return result


def build_itemcf_matrix(train: pd.DataFrame, user_to_idx: dict[int, int], item_to_idx: dict[int, int]) -> tuple[csr_matrix, np.ndarray]:
    itemcf_train = (
        train.sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset=["user_id", "item_id"], keep="last")
        .sort_values(["user_id", "timestamp"], kind="mergesort")
        .groupby("user_id", group_keys=False)
        .tail(HISTORY_LENGTH)
    )
    rows = itemcf_train.user_id.map(user_to_idx).to_numpy(dtype=np.int64)
    cols = itemcf_train.item_id.map(item_to_idx).to_numpy(dtype=np.int64)
    matrix = csr_matrix(
        (np.ones(len(itemcf_train), dtype=np.float32), (rows, cols)),
        shape=(len(user_to_idx), len(item_to_idx)),
    )
    counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
    return matrix, counts


def itemcf_recommendations(
    matrix: csr_matrix,
    item_counts: np.ndarray,
    query_histories: dict[int, list[int]],
    item_ids: np.ndarray,
    top_k: int,
) -> dict[int, list[int]]:
    seed_items = sorted({item for history in query_histories.values() for item in history})
    neighbors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if seed_items:
        seed_array = np.asarray(seed_items, dtype=np.int64)
        co = (matrix[:, seed_array].T @ matrix).tocsr()
        for position, seed in enumerate(seed_items):
            start, end = co.indptr[position], co.indptr[position + 1]
            candidates = co.indices[start:end]
            values = co.data[start:end]
            keep = candidates != seed
            candidates = candidates[keep]
            values = values[keep]
            if len(candidates):
                denom = np.sqrt(item_counts[seed] * item_counts[candidates])
                valid = denom > 0
                candidates = candidates[valid]
                scores = values[valid] / denom[valid]
                if len(scores) > ITEMCF_NEIGHBORS_PER_SEED:
                    positions = np.argpartition(scores, -ITEMCF_NEIGHBORS_PER_SEED)[-ITEMCF_NEIGHBORS_PER_SEED:]
                    positions = positions[np.argsort(scores[positions], kind="mergesort")[::-1]]
                else:
                    positions = np.argsort(scores, kind="mergesort")[::-1]
                neighbors[seed] = (candidates[positions], scores[positions])
            else:
                neighbors[seed] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
    recommendations: dict[int, list[int]] = {}
    for user, history in query_histories.items():
        accumulated: dict[int, float] = defaultdict(float)
        for seed in history:
            candidates, scores = neighbors.get(seed, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
            for candidate, score in zip(candidates, scores):
                accumulated[int(candidate)] += float(score)
        ranked = sorted(accumulated.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
        recommendations[int(user)] = [int(item_ids[index]) for index, _ in ranked]
    return recommendations


def internal_segments(train: pd.DataFrame, item_ids: np.ndarray) -> dict[int, str]:
    counts = train.groupby("item_id", sort=False).size().to_dict()
    ranked = sorted((int(item) for item in item_ids), key=lambda item: (-int(counts.get(item, 0)), item))
    head_end = math.ceil(len(ranked) * 0.20)
    torso_end = math.ceil(len(ranked) * 0.50)
    result: dict[int, str] = {}
    for position, item in enumerate(ranked):
        result[item] = "Head" if position < head_end else "Torso" if position < torso_end else "Tail"
    return result


def concentration(recommendations: dict[int, list[int]], catalog_size: int, segments: dict[int, str]) -> dict[str, Any]:
    top1 = [items[0] for items in recommendations.values() if items]
    top50 = [item for items in recommendations.values() for item in items[:50]]
    counts = pd.Series(top1).value_counts() if top1 else pd.Series(dtype=np.int64)
    union = set(top50)
    exposure = {segment: 0 for segment in ("Head", "Torso", "Tail")}
    for item in top50:
        exposure[segments[int(item)]] += 1
    total = len(top50)
    return {
        "unique_top1_items": int(len(set(top1))),
        "dominant_top1_item": int(counts.index[0]) if len(counts) else None,
        "dominant_top1_share": float(counts.iloc[0] / len(top1)) if top1 else 0.0,
        "top50_union_item_count": int(len(union)),
        "top50_catalog_coverage": float(len(union) / catalog_size) if catalog_size else 0.0,
        "head_exposure_share": float(exposure["Head"] / total) if total else 0.0,
        "torso_exposure_share": float(exposure["Torso"] / total) if total else 0.0,
        "tail_exposure_share": float(exposure["Tail"] / total) if total else 0.0,
    }


def result_row(model: str, metrics: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    return {
        "Model": model,
        "R20": metric(metrics, 20, "Recall"),
        "R50": metric(metrics, 50, "Recall"),
        "HR50": metric(metrics, 50, "HitRate"),
        "NDCG50": metric(metrics, 50, "NDCG"),
        "DominantTop1": diagnostic["dominant_top1_share"],
        "Top50Coverage": diagnostic["top50_catalog_coverage"],
    }


def is_evaluation_complete(output_dir: Path = OUTPUT_DIR) -> bool:
    path = output_dir / "final_summary.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("evaluation_completed"))


def protected_artifact_hashes() -> dict[str, str]:
    paths = [
        ROOT / "artifacts" / "two_tower" / "two_tower_best.pt",
        ROOT / "artifacts" / "two_tower" / "experiments" / "TT_V1_BUY_UNIFORM" / "checkpoint_best.pt",
        ROOT / "artifacts" / "two_tower" / "experiments" / "TT_V1_BUY_UNIFORM_SEED2027" / "best.pt",
        ROOT / "artifacts" / "two_tower" / "experiments" / "TT_V2_MULTI_UNIFORM" / "checkpoint_best.pt",
        ROOT / "artifacts" / "ranking" / "ranker.pkl",
        ROOT / "artifacts" / "generative" / "final_summary.json",
    ]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in paths if path.exists()}


def main() -> None:
    if is_evaluation_complete():
        print("V3 internal evaluation is already completed; refusing to retrain or repeat final eval.")
        return
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Partial V3 output exists; inspect it before any retry: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    protected_before = protected_artifact_hashes()
    set_seed(SEED)

    print("Reading only data/processed/dev/train.csv", flush=True)
    frame = pd.read_csv(TRAIN_PATH)
    train, select, evaluation_frame, split_manifest = build_temporal_split(frame)
    json_write(OUTPUT_DIR / "temporal_split.json", split_manifest)
    print(pd.DataFrame(split_manifest["date_statistics"]).to_string(index=False), flush=True)
    print(
        f"V3_TRAIN={split_manifest['v3_train_dates']} V3_SELECT={split_manifest['v3_select_date']} "
        f"V3_EVAL={split_manifest['v3_eval_date']}",
        flush=True,
    )

    user_map, item_map, user_to_idx, item_to_idx = stable_mappings(train)
    user_map.to_csv(OUTPUT_DIR / "v3_user_mapping.csv", index=False)
    item_map.to_csv(OUTPUT_DIR / "v3_item_mapping.csv", index=False)
    mapping_hashes = {
        "user_mapping_sha256": sha256_file(OUTPUT_DIR / "v3_user_mapping.csv"),
        "item_mapping_sha256": sha256_file(OUTPUT_DIR / "v3_item_mapping.csv"),
    }
    mapping_manifest = {
        "construction_source": "V3_TRAIN only",
        "construction_rule": "pandas stable unique() in V3_TRAIN first-occurrence order; all behavior types",
        "num_users": int(len(user_map)),
        "num_items": int(len(item_map)),
        **mapping_hashes,
        "selection_or_eval_rows_used": False,
    }
    json_write(OUTPUT_DIR / "mapping_manifest.json", mapping_manifest)

    examples, example_stats, q_probabilities = build_training_examples(
        train, user_to_idx, item_to_idx, HISTORY_LENGTH
    )
    json_write(OUTPUT_DIR / "train_example_stats.json", example_stats)
    item_ids = item_map.sort_values("item_idx").item_id.to_numpy(dtype=np.int64)

    # V3_SELECT is the only model-selection population.  Its query history is
    # V3_TRAIN only and target-day events are never added to history.
    select_population = build_query_population(
        "V3_SELECT", select, train, user_to_idx, item_to_idx, HISTORY_LENGTH
    )

    model_a, a_select_payload, a_select_recs, _ = train_model(
        "V3A_ID_INBATCH_LOGQ", "id", examples, q_probabilities,
        select_population, item_ids, len(user_map), mapping_hashes,
    )
    model_b, b_select_payload, b_select_recs, _ = train_model(
        "V3B_SEQUENCE_INBATCH_LOGQ", "sequence", examples, q_probabilities,
        select_population, item_ids, len(user_map), mapping_hashes,
    )

    segments = internal_segments(train, item_ids)
    popularity_select_recs = popularity_recommendations(train, select_population.user_ids, item_to_idx, TOP_K)
    popularity_select_metrics = evaluate_recommendations(popularity_select_recs, select_population.ground_truth)
    itemcf_matrix, itemcf_counts = build_itemcf_matrix(train, user_to_idx, item_to_idx)
    itemcf_select_histories = recent_unique_history_indices(train, select_population.user_ids, item_to_idx)
    itemcf_select_recs = itemcf_recommendations(
        itemcf_matrix, itemcf_counts, itemcf_select_histories, item_ids, TOP_K
    )
    itemcf_select_metrics = evaluate_recommendations(itemcf_select_recs, select_population.ground_truth)

    select_concentration = {
        "Popularity": concentration(popularity_select_recs, len(item_ids), segments),
        "Internal ItemCF": concentration(itemcf_select_recs, len(item_ids), segments),
        "V3A ID + InBatch + logQ": concentration(a_select_recs, len(item_ids), segments),
        "V3B Sequence + InBatch + logQ": concentration(b_select_recs, len(item_ids), segments),
    }
    json_write(
        OUTPUT_DIR / "itemcf_select_metrics.json",
        {"split": "V3_SELECT", "population": select_population.stats, "metrics": itemcf_select_metrics, "concentration": select_concentration["Internal ItemCF"]},
    )

    interpretation_rules = {
        "positive": "If V3B Eval Recall@50 > V3A and concentration is healthier, explicit behavioral history helps under the controlled internal protocol.",
        "partial": "If V3B > V3A but remains below ItemCF, sequence representation fixes part of the weakness but neural retrieval still trails local item-item retrieval.",
        "neutral": "If V3B approximately equals V3A, user representation alone does not explain the weak Two-Tower.",
        "negative": "If V3B < V3A, simple mean pooling does not improve retrieval under this setting.",
        "no_post_eval_tuning": True,
    }
    pre_eval = {
        "created_at_utc": utc_now(),
        "evaluation_not_yet_read": True,
        "evaluation_completed": False,
        "scope": "post-freeze internal temporal evaluation; original validation/test prohibited",
        "split_hashes": {key: value["sha256"] for key, value in split_manifest["splits"].items()},
        "mapping_hashes": mapping_hashes,
        "training_config": base_config("V3_SHARED_CONFIG", "sequence", len(user_map), len(item_map), mapping_hashes),
        "selected_checkpoints": {
            "V3A": {"path": "artifacts/v3_sequence_retrieval/V3A/best.pt", "sha256": sha256_file(A_DIR / "best.pt"), "best_epoch": a_select_payload["best_epoch"]},
            "V3B": {"path": "artifacts/v3_sequence_retrieval/V3B/best.pt", "sha256": sha256_file(B_DIR / "best.pt"), "best_epoch": b_select_payload["best_epoch"]},
        },
        "selection_metrics": {"V3A": a_select_payload, "V3B": b_select_payload},
        "source_code_sha256": sha256_file(Path(__file__)),
        "interpretation_rules": interpretation_rules,
        "protected_artifact_hashes_before": protected_before,
    }
    json_write(OUTPUT_DIR / "pre_eval_freeze.json", pre_eval)
    print("PRE-EVAL FREEZE WRITTEN; internal V3_EVAL has not been evaluated", flush=True)

    # One-time internal evaluation starts only after the freeze above.  Model
    # parameters, mappings, item catalog and ItemCF similarity remain fixed.
    eval_history = pd.concat([train, select], ignore_index=False)
    eval_population = build_query_population(
        "V3_EVAL", evaluation_frame, eval_history, user_to_idx, item_to_idx, HISTORY_LENGTH
    )
    a_eval_metrics, a_eval_recs, _, _ = evaluate_model(model_a, "id", eval_population, item_ids)
    b_eval_metrics, b_eval_recs, _, _ = evaluate_model(model_b, "sequence", eval_population, item_ids)
    a_eval_payload = {
        "experiment": "V3A_ID_INBATCH_LOGQ", "split": "V3_EVAL", "role": "one-time held-forward internal evaluation",
        "population": eval_population.stats, "best_epoch": a_select_payload["best_epoch"], "metrics": a_eval_metrics,
    }
    b_eval_payload = {
        "experiment": "V3B_SEQUENCE_INBATCH_LOGQ", "split": "V3_EVAL", "role": "one-time held-forward internal evaluation",
        "population": eval_population.stats, "best_epoch": b_select_payload["best_epoch"], "metrics": b_eval_metrics,
    }
    json_write(A_DIR / "eval_metrics.json", a_eval_payload)
    json_write(B_DIR / "eval_metrics.json", b_eval_payload)

    popularity_eval_recs = popularity_recommendations(train, eval_population.user_ids, item_to_idx, TOP_K)
    popularity_eval_metrics = evaluate_recommendations(popularity_eval_recs, eval_population.ground_truth)
    itemcf_eval_histories = recent_unique_history_indices(eval_history, eval_population.user_ids, item_to_idx)
    itemcf_eval_recs = itemcf_recommendations(
        itemcf_matrix, itemcf_counts, itemcf_eval_histories, item_ids, TOP_K
    )
    itemcf_eval_metrics = evaluate_recommendations(itemcf_eval_recs, eval_population.ground_truth)
    eval_concentration = {
        "Popularity": concentration(popularity_eval_recs, len(item_ids), segments),
        "Internal ItemCF": concentration(itemcf_eval_recs, len(item_ids), segments),
        "V3A ID + InBatch + logQ": concentration(a_eval_recs, len(item_ids), segments),
        "V3B Sequence + InBatch + logQ": concentration(b_eval_recs, len(item_ids), segments),
    }
    json_write(
        OUTPUT_DIR / "itemcf_eval_metrics.json",
        {"split": "V3_EVAL", "population": eval_population.stats, "metrics": itemcf_eval_metrics, "concentration": eval_concentration["Internal ItemCF"]},
    )
    json_write(
        OUTPUT_DIR / "popularity_metrics.json",
        {
            "selection": {"metrics": popularity_select_metrics, "concentration": select_concentration["Popularity"]},
            "evaluation": {"metrics": popularity_eval_metrics, "concentration": eval_concentration["Popularity"]},
        },
    )
    json_write(A_DIR / "concentration.json", {"selection": select_concentration["V3A ID + InBatch + logQ"], "evaluation": eval_concentration["V3A ID + InBatch + logQ"]})
    json_write(B_DIR / "concentration.json", {"selection": select_concentration["V3B Sequence + InBatch + logQ"], "evaluation": eval_concentration["V3B Sequence + InBatch + logQ"]})

    select_models = {
        "Popularity": popularity_select_metrics,
        "Internal ItemCF": itemcf_select_metrics,
        "V3A ID + InBatch + logQ": a_select_payload["metrics"],
        "V3B Sequence + InBatch + logQ": b_select_payload["metrics"],
    }
    eval_models = {
        "Popularity": popularity_eval_metrics,
        "Internal ItemCF": itemcf_eval_metrics,
        "V3A ID + InBatch + logQ": a_eval_metrics,
        "V3B Sequence + InBatch + logQ": b_eval_metrics,
    }
    select_table = pd.DataFrame([result_row(name, select_models[name], select_concentration[name]) for name in select_models])
    eval_table = pd.DataFrame([result_row(name, eval_models[name], eval_concentration[name]) for name in eval_models])
    select_table.to_csv(OUTPUT_DIR / "comparison_select.csv", index=False)
    eval_table.to_csv(OUTPUT_DIR / "comparison_eval.csv", index=False)

    a_r50 = metric(a_eval_metrics, 50, "Recall")
    b_r50 = metric(b_eval_metrics, 50, "Recall")
    itemcf_r50 = metric(itemcf_eval_metrics, 50, "Recall")
    deltas = {
        "recall_at_50": b_r50 - a_r50,
        "hitrate_at_50": metric(b_eval_metrics, 50, "HitRate") - metric(a_eval_metrics, 50, "HitRate"),
        "ndcg_at_50": metric(b_eval_metrics, 50, "NDCG") - metric(a_eval_metrics, 50, "NDCG"),
        "dominant_top1_share": eval_concentration["V3B Sequence + InBatch + logQ"]["dominant_top1_share"] - eval_concentration["V3A ID + InBatch + logQ"]["dominant_top1_share"],
        "top50_catalog_coverage": eval_concentration["V3B Sequence + InBatch + logQ"]["top50_catalog_coverage"] - eval_concentration["V3A ID + InBatch + logQ"]["top50_catalog_coverage"],
        "sequence_vs_itemcf_recall_at_50_gap": b_r50 - itemcf_r50,
    }
    history_helped = b_r50 > a_r50
    concentration_healthier = (
        eval_concentration["V3B Sequence + InBatch + logQ"]["dominant_top1_share"]
        < eval_concentration["V3A ID + InBatch + logQ"]["dominant_top1_share"]
        and eval_concentration["V3B Sequence + InBatch + logQ"]["top50_catalog_coverage"]
        > eval_concentration["V3A ID + InBatch + logQ"]["top50_catalog_coverage"]
    )
    if history_helped and concentration_healthier:
        result_class = "POSITIVE"
        interpretation = "Explicit behavioral history improves neural retrieval and concentration under the controlled internal protocol."
    elif history_helped:
        result_class = "MIXED"
        interpretation = "Sequence representation improves Recall@50, but concentration does not improve on both preregistered diagnostics."
    else:
        result_class = "NEGATIVE"
        interpretation = "Simple behavior-aware mean pooling does not improve Recall@50 over the controlled ID-only baseline."
    if b_r50 < itemcf_r50:
        interpretation += " Neural retrieval still trails internal ItemCF."

    protected_after = protected_artifact_hashes()
    if protected_before != protected_after:
        raise AssertionError("A protected historical artifact changed during V3")
    pre_eval["evaluation_completed"] = True
    pre_eval["evaluation_completed_at_utc"] = utc_now()
    pre_eval["protected_artifacts_unchanged"] = True
    json_write(OUTPUT_DIR / "pre_eval_freeze.json", pre_eval)
    final_summary = {
        "status": "completed",
        "evaluation_completed": True,
        "result_class": result_class,
        "scope": "POST-FREEZE CONTROLLED RETRIEVAL EXTENSION; INTERNAL TEMPORAL EVALUATION",
        "protected_original_validation_test_accessed": False,
        "old_artifacts_overwritten": False,
        "temporal_protocol": {
            "v3_train": split_manifest["v3_train_dates"],
            "v3_select": split_manifest["v3_select_date"],
            "v3_eval": split_manifest["v3_eval_date"],
        },
        "training_examples": example_stats,
        "catalog": mapping_manifest,
        "populations": {"select": select_population.stats, "eval": eval_population.stats},
        "select": {name: result_row(name, select_models[name], select_concentration[name]) for name in select_models},
        "eval": {name: result_row(name, eval_models[name], eval_concentration[name]) for name in eval_models},
        "sequence_vs_id_deltas": deltas,
        "interpretation": {
            "did_explicit_history_help": history_helped,
            "did_concentration_improve": concentration_healthier,
            "did_neural_retrieval_close_itemcf_gap": bool(b_r50 >= itemcf_r50),
            "conclusion": interpretation,
            "causal_boundary": "V3A and V3B share data, batch order, objective, logQ, optimizer, epochs and evaluation; only user representation differs.",
            "historical_boundary": "Do not compare V3 as a replacement protected-test result; historical V0/V1/V2, ranker and GenRec conclusions remain unchanged.",
        },
        "protected_artifact_hashes_unchanged": True,
        "completed_at_utc": utc_now(),
    }
    json_write(OUTPUT_DIR / "final_summary.json", final_summary)
    print(select_table.to_string(index=False), flush=True)
    print(eval_table.to_string(index=False), flush=True)
    print(f"V3 RESULT {result_class} — PROJECT FINAL FREEZE", flush=True)


if __name__ == "__main__":
    main()
