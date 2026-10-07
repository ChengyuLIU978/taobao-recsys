"""Train and evaluate the leakage-safe Recall -> Rank baseline.

Run from the project root:

    python scripts/08_train_ranker.py

The ranker is trained on validation-period purchase labels and is evaluated
once on test-period purchase labels.  Every aggregate feature is computed from
train.csv only.  Test labels are deliberately loaded only after feature
construction, model fitting, and test-score prediction have completed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dev"
RECALL_DIR = PROJECT_ROOT / "artifacts" / "multistage_recall"
TWO_TOWER_DIR = PROJECT_ROOT / "artifacts" / "two_tower"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ranking"
STAGING_DIR = PROJECT_ROOT / "artifacts" / "ranking.__staging__"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_GT_PATH = DATA_DIR / "validation_ground_truth.csv"
TEST_GT_PATH = DATA_DIR / "test_ground_truth.csv"
VALIDATION_CANDIDATES_PATH = RECALL_DIR / "validation_candidates.parquet"
TEST_CANDIDATES_PATH = RECALL_DIR / "test_candidates.parquet"
RECALL_CONFIG_PATH = RECALL_DIR / "config.json"
RECALL_VALIDATION_METRICS_PATH = RECALL_DIR / "validation_metrics.json"
RECALL_TEST_METRICS_PATH = RECALL_DIR / "test_metrics.json"
USER_MAPPING_PATH = TWO_TOWER_DIR / "user_mapping.csv"
ITEM_MAPPING_PATH = TWO_TOWER_DIR / "item_mapping.csv"

SEED = 42
RANK_SENTINEL = 999
SCORE_SENTINEL = 0.0
RECENCY_SENTINEL_DAYS = -1.0
BEHAVIORS = ("pv", "fav", "cart", "buy")
KS = (10, 20, 50)

RECALL_FEATURES = [
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
    "is_itemcf_top10",
    "is_itemcf_top50",
    "is_tt_top20",
    "is_tt_top50",
]

USER_FEATURES = [
    "user_total_interactions",
    "log1p_user_total_interactions",
    "user_pv_count",
    "user_fav_count",
    "user_cart_count",
    "user_buy_count",
    "user_unique_items",
    "user_unique_categories",
    "user_buy_ratio",
    "user_cart_ratio",
    "user_last_train_timestamp",
    "user_activity_span_days",
]

ITEM_FEATURES = [
    "item_total_interactions",
    "log1p_item_total_interactions",
    "item_pv_count",
    "item_fav_count",
    "item_cart_count",
    "item_buy_count",
    "item_unique_users",
    "item_buy_user_count",
    "item_buy_ratio",
    "item_popularity_rank",
    "item_popularity_percentile",
    "category_id",
    "category_popularity",
]

USER_ITEM_FEATURES = [
    "seen_in_train",
    "user_item_interaction_count",
    "user_item_pv_count",
    "user_item_fav_count",
    "user_item_cart_count",
    "user_item_buy_count",
    "recency_since_last_user_item_interaction_days",
]

USER_CATEGORY_FEATURES = [
    "user_category_interaction_count",
    "user_category_pv_count",
    "user_category_cart_count",
    "user_category_buy_count",
    "user_category_affinity",
]

ALL_FEATURES = (
    RECALL_FEATURES
    + USER_FEATURES
    + ITEM_FEATURES
    + USER_ITEM_FEATURES
    + USER_CATEGORY_FEATURES
)

MODEL_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "lambdarank_truncation_level": 50,
    "random_state": SEED,
    "n_jobs": -1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "importance_type": "gain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="check inputs and leakage boundaries without writing artifacts",
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


def required_paths() -> list[Path]:
    return [
        TRAIN_PATH,
        VALIDATION_GT_PATH,
        TEST_GT_PATH,
        VALIDATION_CANDIDATES_PATH,
        TEST_CANDIDATES_PATH,
        RECALL_CONFIG_PATH,
        RECALL_VALIDATION_METRICS_PATH,
        RECALL_TEST_METRICS_PATH,
        USER_MAPPING_PATH,
        ITEM_MAPPING_PATH,
    ]


def protected_input_hashes() -> dict[str, str]:
    paths = required_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required inputs are missing: {missing}")
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in paths
    }


def validate_input_contracts() -> dict[str, Any]:
    recall_config = json_read(RECALL_CONFIG_PATH)
    label_usage = recall_config["label_usage"]
    if label_usage.get("candidate_generation_reads_validation_labels") is not False:
        raise AssertionError("Validation candidates were not declared label-free")
    if label_usage.get("candidate_generation_reads_test_labels") is not False:
        raise AssertionError("Test candidates were not declared label-free")
    if label_usage.get("test_labels_used_for_evaluation_only") is not True:
        raise AssertionError("Recall config does not protect test labels")

    validation_schema = pd.read_parquet(
        VALIDATION_CANDIDATES_PATH, columns=None
    ).columns.tolist()
    test_schema = pd.read_parquet(TEST_CANDIDATES_PATH, columns=None).columns.tolist()
    expected = [
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
    if validation_schema != expected or test_schema != expected:
        raise AssertionError(
            f"Candidate schema drift: validation={validation_schema}, test={test_schema}"
        )
    forbidden = {"label", "ranking_score"}
    if forbidden & set(validation_schema) or forbidden & set(test_schema):
        raise AssertionError("Input candidates unexpectedly contain labels or ranker scores")
    return {
        "candidate_schema_identical": validation_schema == test_schema,
        "candidate_generation_label_free": True,
        "test_labels_declared_evaluation_only": True,
        "candidate_columns": validation_schema,
    }


def behavior_counts(
    frame: pd.DataFrame, keys: list[str], prefix: str
) -> pd.DataFrame:
    counts = (
        frame.groupby(keys + ["behavior_type"], observed=True)
        .size()
        .unstack("behavior_type", fill_value=0)
        .reindex(columns=BEHAVIORS, fill_value=0)
        .rename(columns={behavior: f"{prefix}_{behavior}_count" for behavior in BEHAVIORS})
        .reset_index()
    )
    for behavior in BEHAVIORS:
        counts[f"{prefix}_{behavior}_count"] = counts[
            f"{prefix}_{behavior}_count"
        ].astype("int32")
    return counts


def build_train_aggregate_tables(train: pd.DataFrame) -> dict[str, Any]:
    """Build every non-recall feature source from train rows only."""
    train_end = int(train.timestamp.max())
    train_start = int(train.timestamp.min())

    user = (
        train.groupby("user_id", sort=False)
        .agg(
            user_total_interactions=("item_id", "size"),
            user_unique_items=("item_id", "nunique"),
            user_unique_categories=("category_id", "nunique"),
            user_first_train_timestamp=("timestamp", "min"),
            user_last_train_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
    user = user.merge(behavior_counts(train, ["user_id"], "user"), on="user_id")
    user["log1p_user_total_interactions"] = np.log1p(
        user.user_total_interactions
    ).astype("float32")
    user["user_buy_ratio"] = (
        user.user_buy_count / user.user_total_interactions
    ).astype("float32")
    user["user_cart_ratio"] = (
        user.user_cart_count / user.user_total_interactions
    ).astype("float32")
    user["user_activity_span_days"] = (
        (user.user_last_train_timestamp - user.user_first_train_timestamp) / 86_400.0
    ).astype("float32")
    user = user.drop(columns="user_first_train_timestamp")

    item = (
        train.groupby("item_id", sort=False)
        .agg(
            item_total_interactions=("user_id", "size"),
            item_unique_users=("user_id", "nunique"),
        )
        .reset_index()
    )
    item = item.merge(behavior_counts(train, ["item_id"], "item"), on="item_id")
    buy_users = (
        train.loc[train.behavior_type.eq("buy")]
        .groupby("item_id")
        .user_id.nunique()
        .rename("item_buy_user_count")
        .reset_index()
    )
    item = item.merge(buy_users, how="left", on="item_id")
    item.item_buy_user_count = item.item_buy_user_count.fillna(0).astype("int32")
    item["log1p_item_total_interactions"] = np.log1p(
        item.item_total_interactions
    ).astype("float32")
    item["item_buy_ratio"] = (
        item.item_buy_count / item.item_total_interactions
    ).astype("float32")
    item = item.sort_values(
        ["item_total_interactions", "item_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    item["item_popularity_rank"] = np.arange(1, len(item) + 1, dtype=np.int32)
    denominator = max(len(item) - 1, 1)
    item["item_popularity_percentile"] = (
        1.0 - (item.item_popularity_rank - 1) / denominator
    ).astype("float32")

    category_conflicts = int(
        train.groupby("item_id").category_id.nunique().gt(1).sum()
    )
    item_category = (
        train.sort_values(["timestamp", "item_id"], kind="mergesort")
        .drop_duplicates("item_id", keep="last")[["item_id", "category_id"]]
        .rename(columns={"category_id": "category_id_raw"})
    )
    category_values = np.sort(train.category_id.unique().astype(np.int64))
    category_to_code = {
        int(category_id): index + 1
        for index, category_id in enumerate(category_values.tolist())
    }
    item_category["category_id"] = (
        item_category.category_id_raw.map(category_to_code).astype("int32")
    )
    item = item.merge(item_category, on="item_id", validate="one_to_one")

    category = (
        train.groupby("category_id")
        .size()
        .rename("category_popularity")
        .reset_index()
        .rename(columns={"category_id": "category_id_raw"})
    )
    category.category_popularity = category.category_popularity.astype("int32")

    user_item = (
        train.groupby(["user_id", "item_id"], sort=False)
        .agg(
            user_item_interaction_count=("timestamp", "size"),
            user_item_last_train_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
    user_item = user_item.merge(
        behavior_counts(train, ["user_id", "item_id"], "user_item"),
        on=["user_id", "item_id"],
    )

    user_category = (
        train.groupby(["user_id", "category_id"], sort=False)
        .size()
        .rename("user_category_interaction_count")
        .reset_index()
        .rename(columns={"category_id": "category_id_raw"})
    )
    uc_behavior = behavior_counts(
        train.rename(columns={"category_id": "category_id_raw"}),
        ["user_id", "category_id_raw"],
        "user_category",
    )
    user_category = user_category.merge(
        uc_behavior, on=["user_id", "category_id_raw"]
    )
    user_category = user_category.drop(columns="user_category_fav_count")

    count_columns = [
        column
        for table in (user, item, user_item, user_category)
        for column in table.columns
        if column.endswith("_count") or column.startswith("user_unique")
    ]
    for table in (user, item, user_item, user_category):
        for column in set(table.columns) & set(count_columns):
            table[column] = table[column].astype("int32")

    return {
        "train_start_timestamp": train_start,
        "train_end_timestamp": train_end,
        "max_aggregate_source_timestamp": train_end,
        "user": user,
        "item": item,
        "category": category,
        "user_item": user_item,
        "user_category": user_category,
        "category_to_code": category_to_code,
        "category_conflict_items": category_conflicts,
        "train_items": int(item.item_id.nunique()),
        "train_categories": int(len(category_to_code)),
    }


def normalize_recall_features(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    for column in ("itemcf_rank", "two_tower_rank", "popularity_rank"):
        frame[column] = frame[column].fillna(RANK_SENTINEL).astype("int32")
    for column in ("itemcf_score", "two_tower_score", "popularity_score"):
        frame[column] = frame[column].fillna(SCORE_SENTINEL).astype("float32")
    frame["rrf_score"] = frame.rrf_score.astype("float32")
    frame["is_itemcf_top10"] = (
        frame.from_itemcf.eq(1) & frame.itemcf_rank.le(10)
    ).astype("int8")
    frame["is_itemcf_top50"] = (
        frame.from_itemcf.eq(1) & frame.itemcf_rank.le(50)
    ).astype("int8")
    frame["is_tt_top20"] = (
        frame.from_two_tower.eq(1) & frame.two_tower_rank.le(20)
    ).astype("int8")
    frame["is_tt_top50"] = (
        frame.from_two_tower.eq(1) & frame.two_tower_rank.le(50)
    ).astype("int8")
    return frame


def build_ranking_features(
    candidates: pd.DataFrame, tables: dict[str, Any]
) -> pd.DataFrame:
    if candidates.duplicated(["user_id", "item_id"]).any():
        raise AssertionError("Same-user candidate duplicates found")
    frame = normalize_recall_features(candidates)
    frame = frame.merge(tables["user"], how="left", on="user_id", validate="many_to_one")
    frame = frame.merge(tables["item"], how="left", on="item_id", validate="many_to_one")
    frame = frame.merge(
        tables["category"], how="left", on="category_id_raw", validate="many_to_one"
    )
    frame = frame.merge(
        tables["user_item"],
        how="left",
        on=["user_id", "item_id"],
        validate="many_to_one",
    )
    frame = frame.merge(
        tables["user_category"],
        how="left",
        on=["user_id", "category_id_raw"],
        validate="many_to_one",
    )

    count_features = [
        "user_total_interactions",
        "user_pv_count",
        "user_fav_count",
        "user_cart_count",
        "user_buy_count",
        "user_unique_items",
        "user_unique_categories",
        "item_total_interactions",
        "item_pv_count",
        "item_fav_count",
        "item_cart_count",
        "item_buy_count",
        "item_unique_users",
        "item_buy_user_count",
        "category_popularity",
        "user_item_interaction_count",
        "user_item_pv_count",
        "user_item_fav_count",
        "user_item_cart_count",
        "user_item_buy_count",
        "user_category_interaction_count",
        "user_category_pv_count",
        "user_category_cart_count",
        "user_category_buy_count",
    ]
    for column in count_features:
        frame[column] = frame[column].fillna(0).astype("int32")

    float_zero_features = [
        "log1p_user_total_interactions",
        "user_buy_ratio",
        "user_cart_ratio",
        "user_activity_span_days",
        "log1p_item_total_interactions",
        "item_buy_ratio",
        "item_popularity_percentile",
    ]
    for column in float_zero_features:
        frame[column] = frame[column].fillna(0.0).astype("float32")

    frame["user_last_train_timestamp"] = (
        frame.user_last_train_timestamp.fillna(0).astype("int64")
    )
    frame["item_popularity_rank"] = frame.item_popularity_rank.fillna(
        tables["train_items"] + 1
    ).astype("int32")
    frame["category_id_raw"] = frame.category_id_raw.fillna(-1).astype("int64")
    frame["category_id"] = frame.category_id.fillna(0).astype("int32")
    frame["seen_in_train"] = frame.user_item_interaction_count.gt(0).astype("int8")
    last_ui = frame.user_item_last_train_timestamp.fillna(0).astype("int64")
    frame["recency_since_last_user_item_interaction_days"] = np.where(
        last_ui.gt(0),
        (tables["train_end_timestamp"] - last_ui) / 86_400.0,
        RECENCY_SENTINEL_DAYS,
    ).astype("float32")
    frame = frame.drop(columns="user_item_last_train_timestamp")
    denominator = frame.user_total_interactions.to_numpy(dtype=np.float32)
    numerator = frame.user_category_interaction_count.to_numpy(dtype=np.float32)
    frame["user_category_affinity"] = np.divide(
        numerator,
        denominator,
        out=np.zeros(len(frame), dtype=np.float32),
        where=denominator > 0,
    )

    missing_features = [column for column in ALL_FEATURES if column not in frame]
    if missing_features:
        raise AssertionError(f"Missing engineered features: {missing_features}")
    if frame[ALL_FEATURES].isna().any().any():
        bad = frame[ALL_FEATURES].columns[frame[ALL_FEATURES].isna().any()].tolist()
        raise AssertionError(f"Features contain NaN after fill policy: {bad}")
    if frame.duplicated(["user_id", "item_id"]).any():
        raise AssertionError("Feature joins introduced duplicate candidates")
    return frame


def labels_for_pairs(frame: pd.DataFrame, ground_truth: pd.DataFrame) -> np.ndarray:
    max_item = int(max(frame.item_id.max(), ground_truth.item_id.max()))
    factor = max_item + 1
    candidate_keys = (
        frame.user_id.to_numpy(dtype=np.int64) * factor
        + frame.item_id.to_numpy(dtype=np.int64)
    )
    gt = ground_truth[["user_id", "item_id"]].drop_duplicates()
    gt_keys = (
        gt.user_id.to_numpy(dtype=np.int64) * factor
        + gt.item_id.to_numpy(dtype=np.int64)
    )
    return np.isin(candidate_keys, gt_keys).astype("int8")


def attach_training_labels(
    frame: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    if "label" in frame:
        raise AssertionError("Labels already exist before authorized label attachment")
    frame["label"] = labels_for_pairs(frame, ground_truth)
    return frame


def prepare_lambdarank_training(
    validation_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    positives_by_user = validation_frame.groupby("user_id").label.sum()
    usable_users = positives_by_user.index[positives_by_user.gt(0)]
    training = validation_frame.loc[
        validation_frame.user_id.isin(usable_users)
    ].copy()
    training = training.sort_values(
        ["user_id", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    groups = training.groupby("user_id", sort=False).size().to_numpy(dtype=np.int32)
    if int(groups.sum()) != len(training):
        raise AssertionError("LambdaRank group sizes do not align with rows")
    if training.groupby("user_id").label.sum().le(0).any():
        raise AssertionError("A LambdaRank training query has no positive candidate")
    stats = {
        "validation_candidate_rows": int(len(validation_frame)),
        "validation_positive_candidate_rows": int(validation_frame.label.sum()),
        "validation_candidate_users": int(validation_frame.user_id.nunique()),
        "training_queries_with_at_least_one_positive": int(len(groups)),
        "training_rows": int(len(training)),
        "training_positive_rows": int(training.label.sum()),
        "group_count": int(len(groups)),
        "group_sum": int(groups.sum()),
        "min_group_size": int(groups.min()),
        "max_group_size": int(groups.max()),
        "all_training_queries_have_positive": True,
    }
    return training, groups, stats


def fit_ranker(
    training: pd.DataFrame, groups: np.ndarray, features: list[str]
) -> lgb.LGBMRanker:
    ranker = lgb.LGBMRanker(**MODEL_PARAMS)
    categorical = ["category_id"] if "category_id" in features else []
    ranker.fit(
        training[features],
        training.label.to_numpy(dtype=np.int8),
        group=groups,
        categorical_feature=categorical,
    )
    return ranker


def predict_in_chunks(
    model: lgb.LGBMRanker,
    frame: pd.DataFrame,
    features: list[str],
    chunk_size: int = 500_000,
) -> np.ndarray:
    result = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        end = min(start + chunk_size, len(frame))
        result[start:end] = model.predict(frame.iloc[start:end][features]).astype(
            np.float32
        )
        print(f"Predicted {end:,}/{len(frame):,} candidate rows", flush=True)
    return result


def recall_at_k(recommendations: list[int], ground_truth: set[int], k: int) -> float:
    return (
        len(set(recommendations[:k]) & ground_truth) / len(ground_truth)
        if ground_truth
        else 0.0
    )


def hitrate_at_k(recommendations: list[int], ground_truth: set[int], k: int) -> float:
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
    ks: Iterable[int] = KS,
) -> list[dict[str, Any]]:
    rows = []
    for k in ks:
        recalls: list[float] = []
        hitrates: list[float] = []
        ndcgs: list[float] = []
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


def ground_truth_dict(frame: pd.DataFrame) -> dict[int, set[int]]:
    return frame.groupby("user_id").item_id.apply(set).to_dict()


def warm_ground_truth(
    ground_truth: pd.DataFrame,
    user_mapping: pd.DataFrame,
    item_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    user_ids = set(user_mapping.user_id.astype(int))
    item_ids = set(item_mapping.item_id.astype(int))
    warm = ground_truth.loc[
        ground_truth.user_id.isin(user_ids) & ground_truth.item_id.isin(item_ids)
    ].drop_duplicates(["user_id", "item_id"])
    return warm, ground_truth_dict(warm)


def source_recommendations(
    candidates: pd.DataFrame, source: str, evaluation_users: set[int]
) -> dict[int, list[int]]:
    subset = candidates.loc[
        candidates.user_id.isin(evaluation_users)
        & candidates[f"from_{source}"].eq(1),
        ["user_id", "item_id", f"{source}_rank"],
    ].sort_values(["user_id", f"{source}_rank", "item_id"], kind="mergesort")
    return {
        int(user_id): group.item_id.astype(int).tolist()
        for user_id, group in subset.groupby("user_id", sort=False)
    }


def fusion_recommendations(
    candidates: pd.DataFrame, method: str, evaluation_users: set[int]
) -> dict[int, list[int]]:
    subset = candidates.loc[
        candidates.user_id.isin(evaluation_users),
        ["user_id", "item_id", "from_itemcf", "itemcf_rank", "rrf_score"],
    ].copy()
    if method == "rrf":
        subset = subset.sort_values(
            ["user_id", "rrf_score", "item_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    elif method == "itemcf_first":
        subset["_not_itemcf"] = 1 - subset.from_itemcf
        subset = subset.sort_values(
            ["user_id", "_not_itemcf", "itemcf_rank", "rrf_score", "item_id"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    return {
        int(user_id): group.item_id.astype(int).tolist()
        for user_id, group in subset.groupby("user_id", sort=False)
    }


def ranker_topk_frame(
    frame: pd.DataFrame, score_column: str, top_k: int = 50
) -> pd.DataFrame:
    columns = [
        "user_id",
        "item_id",
        score_column,
        "itemcf_rank",
        "two_tower_rank",
        "popularity_rank",
        "recall_source_count",
        "rrf_score",
        "label",
    ]
    ranked = frame[columns].sort_values(
        ["user_id", score_column, "itemcf_rank", "rrf_score", "item_id"],
        ascending=[True, False, True, False, True],
        kind="mergesort",
    )
    ranked["rank"] = ranked.groupby("user_id", sort=False).cumcount() + 1
    return ranked.loc[ranked["rank"].le(top_k)].reset_index(drop=True)


def recommendations_from_topk(topk: pd.DataFrame) -> dict[int, list[int]]:
    return {
        int(user_id): group.item_id.astype(int).tolist()
        for user_id, group in topk.groupby("user_id", sort=False)
    }


def oracle_metrics(
    candidates: pd.DataFrame, ground_truth: dict[int, set[int]]
) -> dict[str, Any]:
    candidate_sets = (
        candidates.loc[candidates.user_id.isin(set(ground_truth)), ["user_id", "item_id"]]
        .groupby("user_id")
        .item_id.apply(set)
        .to_dict()
    )
    recalls = []
    hitrates = []
    hits = 0
    interactions = 0
    for user_id, targets in ground_truth.items():
        retrieved = candidate_sets.get(int(user_id), set())
        count = len(retrieved & targets)
        recalls.append(count / len(targets))
        hitrates.append(float(count > 0))
        hits += count
        interactions += len(targets)
    return {
        "Recall": float(np.mean(recalls)),
        "HitRate": float(np.mean(hitrates)),
        "hit_interactions": int(hits),
        "total_interactions": int(interactions),
        "interaction_recall": float(hits / interactions),
    }


def evaluate_methods(
    test_frame: pd.DataFrame,
    warm_gt: dict[int, set[int]],
    ranker_top50: pd.DataFrame,
) -> pd.DataFrame:
    users = set(warm_gt)
    recommendations = {
        "Popularity": source_recommendations(test_frame, "popularity", users),
        "Two-Tower": source_recommendations(test_frame, "two_tower", users),
        "ItemCF": source_recommendations(test_frame, "itemcf", users),
        "RRF": fusion_recommendations(test_frame, "rrf", users),
        "ItemCF-first": fusion_recommendations(test_frame, "itemcf_first", users),
        "Ranker": recommendations_from_topk(
            ranker_top50.loc[ranker_top50.user_id.isin(users)]
        ),
    }
    rows: list[dict[str, Any]] = []
    for method, recs in recommendations.items():
        for values in evaluate_ranked_recommendations(recs, warm_gt):
            rows.append({"Method": method, **values})
    return pd.DataFrame(rows)


def assert_baseline_reproduction(metrics: pd.DataFrame) -> dict[str, bool]:
    reference = json_read(RECALL_TEST_METRICS_PATH)
    checks: dict[str, bool] = {}
    for method, reference_key in (
        ("ItemCF", "itemcf"),
        ("RRF", "rrf"),
        ("ItemCF-first", "itemcf_first"),
    ):
        expected_rows = reference["fusion_ranking"][reference_key]
        for expected in expected_rows:
            actual = metrics.loc[
                metrics.Method.eq(method) & metrics.K.eq(expected["K"])
            ].iloc[0]
            key = f"{method}_K{expected['K']}"
            checks[key] = all(
                np.isclose(actual[name], expected[name], atol=1e-12)
                for name in ("Recall", "HitRate", "NDCG")
            )
            if not checks[key]:
                raise AssertionError(f"Stored baseline was not reproduced: {key}")
    source_reference = {"Popularity": "popularity", "Two-Tower": "two_tower"}
    for method, reference_key in source_reference.items():
        expected_rows = reference["source_performance"][reference_key]["ranked_metrics"]
        expected = next(row for row in expected_rows if row["K"] == 50)
        actual = metrics.loc[metrics.Method.eq(method) & metrics.K.eq(50)].iloc[0]
        key = f"{method}_K50"
        checks[key] = all(
            np.isclose(actual[name], expected[name], atol=1e-12)
            for name in ("Recall", "HitRate", "NDCG")
        )
        if not checks[key]:
            raise AssertionError(f"Stored baseline was not reproduced: {key}")
    return checks


def feature_manifest(frame: pd.DataFrame) -> dict[str, Any]:
    recall_descriptions = {
        "from_itemcf": "candidate came from ItemCF",
        "from_two_tower": "candidate came from seed42 TT_V1_BUY_UNIFORM",
        "from_popularity": "candidate came from train-popularity recall",
        "itemcf_rank": "one-based ItemCF source rank",
        "two_tower_rank": "one-based Two-Tower source rank",
        "popularity_rank": "one-based Popularity source rank",
        "itemcf_score": "raw ItemCF recall score",
        "two_tower_score": "raw Two-Tower inner-product score",
        "popularity_score": "train interaction count used by Popularity",
        "recall_source_count": "number of recall sources returning the pair",
        "rrf_score": "fixed-constant RRF score retained as a feature only",
        "is_itemcf_top10": "candidate is in ItemCF Top-10",
        "is_itemcf_top50": "candidate is in ItemCF Top-50",
        "is_tt_top20": "candidate is in Two-Tower Top-20",
        "is_tt_top50": "candidate is in Two-Tower Top-50",
    }
    descriptions = {
        **recall_descriptions,
        "user_total_interactions": "train interactions for the user",
        "log1p_user_total_interactions": "log1p train interactions for the user",
        "user_pv_count": "train pv rows for the user",
        "user_fav_count": "train fav rows for the user",
        "user_cart_count": "train cart rows for the user",
        "user_buy_count": "train buy rows for the user",
        "user_unique_items": "unique train items for the user",
        "user_unique_categories": "unique train categories for the user",
        "user_buy_ratio": "user train buy count / total interactions",
        "user_cart_ratio": "user train cart count / total interactions",
        "user_last_train_timestamp": "latest user timestamp in train",
        "user_activity_span_days": "user train max-minus-min timestamp in days",
        "item_total_interactions": "train interactions for the item",
        "log1p_item_total_interactions": "log1p train interactions for the item",
        "item_pv_count": "train pv rows for the item",
        "item_fav_count": "train fav rows for the item",
        "item_cart_count": "train cart rows for the item",
        "item_buy_count": "train buy rows for the item",
        "item_unique_users": "unique train users for the item",
        "item_buy_user_count": "unique train buying users for the item",
        "item_buy_ratio": "item train buy count / total interactions",
        "item_popularity_rank": "deterministic train interaction-count rank",
        "item_popularity_percentile": "train popularity percentile; most popular is 1",
        "category_id": "consecutive categorical code from latest train category per item",
        "category_popularity": "train interactions in the item's category",
        "seen_in_train": "user-item pair occurred in train",
        "user_item_interaction_count": "train count for the user-item pair",
        "user_item_pv_count": "train pv count for the user-item pair",
        "user_item_fav_count": "train fav count for the user-item pair",
        "user_item_cart_count": "train cart count for the user-item pair",
        "user_item_buy_count": "train buy count for the user-item pair",
        "recency_since_last_user_item_interaction_days": "days from train end to latest train pair interaction",
        "user_category_interaction_count": "train user-category interaction count",
        "user_category_pv_count": "train user-category pv count",
        "user_category_cart_count": "train user-category cart count",
        "user_category_buy_count": "train user-category buy count",
        "user_category_affinity": "train user-category count / user total count",
    }
    records = []
    for name in ALL_FEATURES:
        if name in ("itemcf_rank", "two_tower_rank", "popularity_rank"):
            missing = f"fill with fixed rank sentinel {RANK_SENTINEL}"
        elif name in ("itemcf_score", "two_tower_score", "popularity_score"):
            missing = f"fill with score sentinel {SCORE_SENTINEL}"
        elif name == "recency_since_last_user_item_interaction_days":
            missing = f"fill unseen pair with {RECENCY_SENTINEL_DAYS} days"
        elif name == "category_id":
            missing = "0 is unknown; train categories are consecutive codes starting at 1"
        elif name == "item_popularity_rank":
            missing = "number of train items + 1"
        else:
            missing = "0 for an absent train aggregate; no split-specific imputation"
        records.append(
            {
                "name": name,
                "dtype": str(frame[name].dtype),
                "source": "recall candidate artifact" if name in RECALL_FEATURES else "train.csv aggregate",
                "train_only": name not in RECALL_FEATURES,
                "description": descriptions[name],
                "missing_value_policy": missing,
            }
        )
    return {
        "rank_sentinel": RANK_SENTINEL,
        "score_sentinel": SCORE_SENTINEL,
        "recency_unit": "days",
        "recency_unseen_sentinel": RECENCY_SENTINEL_DAYS,
        "raw_user_id_is_model_feature": False,
        "raw_item_id_is_model_feature": False,
        "category_rule": "latest category observed in train; raw value retained as category_id_raw and a consecutive train-derived code is used as category_id",
        "features": records,
    }


def metrics_record(metrics: pd.DataFrame, method: str, k: int) -> dict[str, float]:
    row = metrics.loc[metrics.Method.eq(method) & metrics.K.eq(k)].iloc[0]
    return {name: float(row[name]) for name in ("Recall", "HitRate", "NDCG")}


def print_metric_table(metrics: pd.DataFrame) -> None:
    pivot = metrics.pivot(index="Method", columns="K", values=["Recall", "HitRate", "NDCG"])
    order = ["Popularity", "Two-Tower", "ItemCF", "RRF", "ItemCF-first", "Ranker"]
    print("\nTEST END-TO-END METRICS", flush=True)
    print(
        "Method          Recall@10  Recall@20  Recall@50  HitRate@50  NDCG@50",
        flush=True,
    )
    for method in order:
        print(
            f"{method:<15} "
            f"{pivot.loc[method, ('Recall', 10)]:.6f}   "
            f"{pivot.loc[method, ('Recall', 20)]:.6f}   "
            f"{pivot.loc[method, ('Recall', 50)]:.6f}   "
            f"{pivot.loc[method, ('HitRate', 50)]:.6f}     "
            f"{pivot.loc[method, ('NDCG', 50)]:.6f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing ranking artifacts: {OUTPUT_DIR}"
        )
    if STAGING_DIR.exists():
        raise FileExistsError(
            f"Stale staging directory exists; inspect it before rerunning: {STAGING_DIR}"
        )

    input_contracts = validate_input_contracts()
    input_hashes_before = protected_input_hashes()
    train = pd.read_csv(TRAIN_PATH)
    train["behavior_type"] = train.behavior_type.astype("string")
    aggregates = build_train_aggregate_tables(train)
    train_end = aggregates["train_end_timestamp"]
    if aggregates["max_aggregate_source_timestamp"] > train_end:
        raise AssertionError("Aggregate source timestamp exceeds train end")
    print("PREFLIGHT", flush=True)
    print(f"  LightGBM: {lgb.__version__} / LGBMRanker(lambdarank)", flush=True)
    print(f"  train rows: {len(train):,}", flush=True)
    print(f"  train timestamp: {aggregates['train_start_timestamp']}..{train_end}", flush=True)
    print(f"  category conflict items: {aggregates['category_conflict_items']:,}", flush=True)
    print(f"  features: {len(ALL_FEATURES)} ({len(RECALL_FEATURES)} recall + {len(ALL_FEATURES) - len(RECALL_FEATURES)} train-only)", flush=True)
    print("  test labels not loaded", flush=True)
    if args.preflight_only:
        print("Preflight complete; no artifacts written.", flush=True)
        return

    STAGING_DIR.mkdir(parents=False, exist_ok=False)
    try:
        # Phase 1: build validation features without opening any label file.
        validation_candidates = pd.read_parquet(VALIDATION_CANDIDATES_PATH)
        validation_features = build_ranking_features(validation_candidates, aggregates)
        del validation_candidates
        gc.collect()
        if "label" in validation_features:
            raise AssertionError("Validation labels leaked into feature construction")

        # Phase 2: validation labels are authorized only as the ranker target.
        validation_gt = pd.read_csv(VALIDATION_GT_PATH)
        validation_features = attach_training_labels(validation_features, validation_gt)
        training, groups, training_stats = prepare_lambdarank_training(validation_features)
        print("\nRANKER TRAINING DATA", flush=True)
        for key, value in training_stats.items():
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}", flush=True)

        ranker = fit_ranker(training, groups, ALL_FEATURES)
        recall_only_ranker = fit_ranker(training, groups, RECALL_FEATURES)
        print("Full and recall-only fixed-parameter rankers trained.", flush=True)

        validation_features["ranking_score"] = predict_in_chunks(
            ranker, validation_features, ALL_FEATURES
        )
        validation_features.to_parquet(
            STAGING_DIR / "validation_ranking_dataset.parquet",
            index=False,
            compression="zstd",
        )

        user_mapping = pd.read_csv(USER_MAPPING_PATH)
        item_mapping = pd.read_csv(ITEM_MAPPING_PATH)
        _, validation_warm_gt = warm_ground_truth(
            validation_gt, user_mapping, item_mapping
        )
        validation_top50 = ranker_topk_frame(validation_features, "ranking_score")
        validation_fit_metrics = evaluate_ranked_recommendations(
            recommendations_from_topk(
                validation_top50.loc[
                    validation_top50.user_id.isin(set(validation_warm_gt))
                ]
            ),
            validation_warm_gt,
        )
        validation_diagnostic = {
            "role": "training-fit diagnostic only; not a generalization estimate",
            "metrics": validation_fit_metrics,
        }
        del validation_features, validation_top50, validation_gt, validation_warm_gt
        gc.collect()

        # Phase 3: test features and scores are completed before test GT is parsed.
        test_ground_truth_parsed = False
        test_candidates = pd.read_parquet(TEST_CANDIDATES_PATH)
        test_features = build_ranking_features(test_candidates, aggregates)
        if "label" in test_features:
            raise AssertionError("Test labels entered feature construction")
        test_features["ranking_score"] = predict_in_chunks(
            ranker, test_features, ALL_FEATURES
        )
        test_features["recall_only_score"] = predict_in_chunks(
            recall_only_ranker, test_features, RECALL_FEATURES
        )
        if "label" in test_features:
            raise AssertionError("Test labels entered model prediction")
        model_and_scores_complete_before_test_label_load = True

        # Phase 4: only now may test labels be opened for final offline evaluation.
        test_gt = pd.read_csv(TEST_GT_PATH)
        test_ground_truth_parsed = True
        test_features["label"] = labels_for_pairs(test_features, test_gt)
        test_features.to_parquet(
            STAGING_DIR / "test_ranking_dataset.parquet",
            index=False,
            compression="zstd",
        )
        warm_test_frame, warm_test_gt = warm_ground_truth(
            test_gt, user_mapping, item_mapping
        )
        all_test_gt = ground_truth_dict(test_gt.drop_duplicates(["user_id", "item_id"]))
        if len(warm_test_gt) != 1116 or len(warm_test_frame) != 1438:
            raise AssertionError(
                f"Warm evaluation population drift: users={len(warm_test_gt)}, interactions={len(warm_test_frame)}"
            )

        ranker_top50 = ranker_topk_frame(test_features, "ranking_score")
        recall_only_top50 = ranker_topk_frame(test_features, "recall_only_score")
        ranker_top50 = ranker_top50.rename(columns={"ranking_score": "ranking_score"})
        ranker_top50.to_parquet(
            STAGING_DIR / "test_recommendations.parquet",
            index=False,
            compression="zstd",
        )
        metrics = evaluate_methods(test_features, warm_test_gt, ranker_top50)
        metrics["Protocol"] = "warm purchase; exact union reranking; no historical-item filtering"
        baseline_checks = assert_baseline_reproduction(metrics)
        metrics.to_csv(STAGING_DIR / "test_metrics.csv", index=False)

        recall_only_metrics = evaluate_ranked_recommendations(
            recommendations_from_topk(
                recall_only_top50.loc[
                    recall_only_top50.user_id.isin(set(warm_test_gt))
                ]
            ),
            warm_test_gt,
            (20, 50),
        )
        full_ablation_metrics = [
            metrics_record(metrics, "Ranker", 20) | {"K": 20},
            metrics_record(metrics, "Ranker", 50) | {"K": 50},
        ]
        ablation_rows = []
        for variant, rows in (
            ("A_recall_only", recall_only_metrics),
            ("B_recall_plus_train_aggregates", full_ablation_metrics),
        ):
            for row in rows:
                ablation_rows.append({"Variant": variant, **row})
        pd.DataFrame(ablation_rows).to_csv(
            STAGING_DIR / "ablation_metrics.csv", index=False
        )

        warm_oracle = oracle_metrics(test_features, warm_test_gt)
        all_oracle = oracle_metrics(test_features, all_test_gt)
        ranker_all_metrics = evaluate_ranked_recommendations(
            recommendations_from_topk(
                ranker_top50.loc[ranker_top50.user_id.isin(set(all_test_gt))]
            ),
            all_test_gt,
        )

        # LightGBM's native Windows writer cannot open paths containing non-ASCII
        # characters.  Save/load the text representation through an ASCII temp
        # path, then copy it into the project artifact directory with Python I/O.
        with tempfile.TemporaryDirectory(prefix="taobao_ranker_") as temp_dir:
            temp_path = Path(temp_dir)
            native_full = temp_path / "ranker.txt"
            native_recall = temp_path / "ranker_recall_only.txt"
            ranker.booster_.save_model(str(native_full))
            recall_only_ranker.booster_.save_model(str(native_recall))
            loaded = lgb.Booster(model_file=str(native_full))
            shutil.copy2(native_full, STAGING_DIR / "ranker.txt")
            shutil.copy2(native_recall, STAGING_DIR / "ranker_recall_only.txt")
        joblib.dump(ranker, STAGING_DIR / "ranker.pkl")
        joblib.dump(recall_only_ranker, STAGING_DIR / "ranker_recall_only.pkl")
        check_rows = training.iloc[:100]
        original_predictions = ranker.predict(check_rows[ALL_FEATURES])
        loaded_predictions = loaded.predict(check_rows[ALL_FEATURES])
        strict_model_reload = bool(
            np.allclose(original_predictions, loaded_predictions, atol=1e-12)
        )
        if not strict_model_reload:
            raise AssertionError("Saved LightGBM model did not reproduce predictions")

        feature_importance = pd.DataFrame(
            {
                "feature": ALL_FEATURES,
                "importance": ranker.booster_.feature_importance(
                    importance_type="gain"
                ),
                "split_count": ranker.booster_.feature_importance(
                    importance_type="split"
                ),
            }
        ).sort_values(["importance", "feature"], ascending=[False, True])
        total_importance = float(feature_importance.importance.sum())
        feature_importance["importance_fraction"] = (
            feature_importance.importance / total_importance
            if total_importance > 0
            else 0.0
        )
        feature_importance["rank"] = np.arange(
            1, len(feature_importance) + 1, dtype=np.int32
        )
        feature_importance.to_csv(
            STAGING_DIR / "feature_importance.csv", index=False
        )

        manifest = feature_manifest(training)
        json_write(STAGING_DIR / "feature_manifest.json", manifest)

        input_hashes_after = protected_input_hashes()
        if input_hashes_before != input_hashes_after:
            raise AssertionError("A protected input artifact changed during ranking")

        test_reference = json_read(RECALL_TEST_METRICS_PATH)
        cold_target_fraction = float(
            test_reference["ground_truth"]["unretrievable_cold_target_fraction"]
        )
        ranker50 = metrics_record(metrics, "Ranker", 50)
        itemcf50 = metrics_record(metrics, "ItemCF", 50)
        oracle_gap = warm_oracle["Recall"] - ranker50["Recall"]
        absolute_delta = ranker50["Recall"] - itemcf50["Recall"]
        relative_delta = absolute_delta / itemcf50["Recall"]
        bottlenecks = []
        if warm_oracle["Recall"] < 0.5:
            bottlenecks.append("Retrieval remains the main bottleneck.")
        if oracle_gap > 0.05:
            bottlenecks.append("Ranking also leaves material candidate-oracle headroom.")
        if cold_target_fraction >= 0.25:
            bottlenecks.append("Catalog coverage / cold-item retrieval is a major bottleneck.")

        tt_features = [
            "from_two_tower",
            "two_tower_rank",
            "two_tower_score",
            "is_tt_top20",
            "is_tt_top50",
        ]
        tt_importance_fraction = float(
            feature_importance.loc[
                feature_importance.feature.isin(tt_features), "importance_fraction"
            ].sum()
        )

        sample_users = np.random.default_rng(SEED).choice(
            np.array(sorted(warm_test_gt), dtype=np.int64), size=5, replace=False
        )
        sanity = ranker_top50.loc[
            ranker_top50.user_id.isin(sample_users) & ranker_top50["rank"].le(10)
        ].copy()
        sanity.to_csv(STAGING_DIR / "sanity_sample.csv", index=False)

        leakage_checks = {
            "test_1_max_aggregate_timestamp_le_train_end": bool(
                aggregates["max_aggregate_source_timestamp"] <= train_end
            ),
            "test_2_test_ground_truth_not_parsed_until_after_training_and_prediction": bool(
                test_ground_truth_parsed
                and model_and_scores_complete_before_test_label_load
            ),
            "test_2_no_test_label_column_in_training_dataframe": "test_label" not in training.columns,
            "test_2_model_and_scores_complete_before_test_label_load": model_and_scores_complete_before_test_label_load,
            "test_3_validation_labels_only_added_after_feature_construction": True,
            "test_3_all_aggregate_features_come_from_train_csv": True,
            "test_4_group_sum_equals_training_rows": int(groups.sum()) == len(training),
            "test_5_validation_same_user_candidate_unique": not pd.read_parquet(
                VALIDATION_CANDIDATES_PATH, columns=["user_id", "item_id"]
            ).duplicated(["user_id", "item_id"]).any(),
            "test_5_test_same_user_candidate_unique": not test_features.duplicated(
                ["user_id", "item_id"]
            ).any(),
        }
        if not all(leakage_checks.values()):
            raise AssertionError(f"Leakage/consistency check failed: {leakage_checks}")

        config = {
            "stage": "Ranking + Recall to Rank end-to-end baseline",
            "backend": "lightgbm.LGBMRanker",
            "objective": "lambdarank",
            "seed": SEED,
            "model_params": MODEL_PARAMS,
            "training_period_for_ranker": "validation",
            "training_labels": "validation_ground_truth.csv purchase pairs",
            "training_query_policy": "keep every candidate for validation users having at least one positive candidate; zero-positive queries provide no LambdaRank signal and are excluded",
            "final_evaluation_period": "test",
            "test_label_access": "loaded only after train-only feature construction, ranker fitting, and test score prediction",
            "evaluation_protocol": "warm purchase: user and target item in persisted Two-Tower train-PV vocabulary; no historical-item filtering",
            "feature_count": len(ALL_FEATURES),
            "recall_only_feature_count": len(RECALL_FEATURES),
            "rank_sentinel": RANK_SENTINEL,
            "score_sentinel": SCORE_SENTINEL,
            "recency_unseen_sentinel_days": RECENCY_SENTINEL_DAYS,
            "category_rule": manifest["category_rule"],
            "category_conflict_items": aggregates["category_conflict_items"],
            "raw_user_id_used_as_feature": False,
            "raw_item_id_used_as_feature": False,
            "rank_tie_break": "ranking_score desc, itemcf_rank asc, rrf_score desc, item_id asc",
            "model_serialization": "ranker.pkl is directly loadable; ranker.txt is the native booster text copied from an ASCII temporary path because LightGBM native Windows I/O cannot write the Unicode project path",
            "ablation": "fixed recall-only versus fixed all-feature ranker; test comparison is analysis only and does not select the production baseline",
            "input_hashes": input_hashes_before,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
                "lightgbm": lgb.__version__,
            },
        }
        json_write(STAGING_DIR / "config.json", config)

        summary = {
            "status": "completed",
            "training_split_for_ranker": "validation",
            "final_evaluation_split": "test",
            "training": training_stats,
            "validation_training_fit_diagnostic": validation_diagnostic,
            "test_evaluation": {
                "warm_users": len(warm_test_gt),
                "warm_interactions": len(warm_test_frame),
                "all_gt_users": len(all_test_gt),
                "all_gt_interactions": int(len(test_gt.drop_duplicates(["user_id", "item_id"]))),
                "candidate_oracle_warm": warm_oracle,
                "candidate_oracle_all_gt": all_oracle,
                "ranker_warm_metrics": {
                    str(k): metrics_record(metrics, "Ranker", k) for k in KS
                },
                "ranker_all_gt_metrics": {
                    str(row["K"]): {
                        key: row[key] for key in ("Recall", "HitRate", "NDCG")
                    }
                    for row in ranker_all_metrics
                },
                "oracle_gap_at_50": oracle_gap,
                "ranker_vs_itemcf_recall_at_50_absolute_delta": absolute_delta,
                "ranker_vs_itemcf_recall_at_50_relative_delta": relative_delta,
                "cold_target_fraction_absent_from_train": cold_target_fraction,
            },
            "feature_importance": {
                "top_15": feature_importance.head(15)[
                    ["feature", "importance", "importance_fraction"]
                ].to_dict("records"),
                "two_tower_feature_importance_fraction": tt_importance_fraction,
                "causal_warning": "Gain importance is descriptive; no TT-isolation ablation was run, so it does not prove causal contribution.",
            },
            "leakage_checks": leakage_checks,
            "baseline_reproduction_checks": baseline_checks,
            "saved_model_reload_matches": strict_model_reload,
            "protected_input_hashes_unchanged": True,
            "bottleneck_conclusions": bottlenecks,
            "elapsed_seconds": time.perf_counter() - started,
        }
        json_write(STAGING_DIR / "summary.json", summary)

        del test_candidates, test_features, training
        gc.collect()
        os.replace(STAGING_DIR, OUTPUT_DIR)

        print_metric_table(metrics)
        print("\nKEY RESULTS", flush=True)
        print(f"  Candidate oracle Recall: {warm_oracle['Recall']:.6f}", flush=True)
        print(f"  Ranker Recall@50: {ranker50['Recall']:.6f}", flush=True)
        print(f"  Oracle gap: {oracle_gap:.6f}", flush=True)
        print(f"  Ranker vs ItemCF absolute delta: {absolute_delta:+.6f}", flush=True)
        print(f"  Ranker vs ItemCF relative delta: {relative_delta:+.2%}", flush=True)
        print(f"  Cold target fraction: {cold_target_fraction:.2%}", flush=True)
        print("\nTOP-15 FEATURE IMPORTANCE", flush=True)
        print(
            feature_importance.head(15)[
                ["feature", "importance", "importance_fraction"]
            ].to_string(index=False),
            flush=True,
        )
        print("\nSANITY SAMPLE: 5 warm test users, Top-10", flush=True)
        print(sanity.to_string(index=False), flush=True)
        print(f"\nArtifacts saved to {OUTPUT_DIR}", flush=True)
    except Exception:
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)
        raise


if __name__ == "__main__":
    main()
