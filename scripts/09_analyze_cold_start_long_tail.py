"""Cold-start, catalog-coverage, long-tail, and popularity-bias diagnostics.

Run from the project root:

    python scripts/09_analyze_cold_start_long_tail.py

This script is analysis-only.  It does not train or modify any retriever or
ranker.  Popularity statistics and item segments are derived exclusively from
train.csv.  Test labels are used only to segment and evaluate already-saved
recommendations and candidate sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dev"
TWO_TOWER_DIR = PROJECT_ROOT / "artifacts" / "two_tower"
RECALL_DIR = PROJECT_ROOT / "artifacts" / "multistage_recall"
RANKING_DIR = PROJECT_ROOT / "artifacts" / "ranking"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "analysis"
STAGING_DIR = PROJECT_ROOT / "artifacts" / "analysis.__staging__"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TEST_GT_PATH = DATA_DIR / "test_ground_truth.csv"
USER_MAPPING_PATH = TWO_TOWER_DIR / "user_mapping.csv"
ITEM_MAPPING_PATH = TWO_TOWER_DIR / "item_mapping.csv"
TEST_CANDIDATES_PATH = RECALL_DIR / "test_candidates.parquet"
RECALL_TEST_METRICS_PATH = RECALL_DIR / "test_metrics.json"
RANKING_SUMMARY_PATH = RANKING_DIR / "summary.json"
RANKING_METRICS_PATH = RANKING_DIR / "test_metrics.csv"
RANKER_RECOMMENDATIONS_PATH = RANKING_DIR / "test_recommendations.parquet"

BEHAVIORS = ("pv", "fav", "cart", "buy")
METHODS = ("Popularity", "ItemCF", "Two-Tower", "Ranker")
EXPOSURE_METHODS = ("ItemCF", "Two-Tower", "Ranker")
KS = (10, 20, 50)
ITEMCF_MAX_HISTORY = 50
BUCKET_LABELS = {
    "A": "Warm user + Warm item",
    "B": "Cold user + Warm item",
    "C": "Warm user + Cold item",
    "D": "Cold user + Cold item",
}
SEGMENTS = ("Head", "Torso", "Tail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate source artifacts and print locked analysis rules without writing",
    )
    return parser.parse_args()


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=convert),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_paths() -> list[Path]:
    paths = [TRAIN_PATH, TEST_PATH, TEST_GT_PATH, USER_MAPPING_PATH, ITEM_MAPPING_PATH]
    paths.extend(sorted(path for path in RECALL_DIR.iterdir() if path.is_file()))
    paths.extend(sorted(path for path in RANKING_DIR.iterdir() if path.is_file()))
    return paths


def protected_hashes() -> dict[str, str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in protected_paths()
    }


def behavior_counts(train: pd.DataFrame) -> pd.DataFrame:
    counts = (
        train.groupby(["item_id", "behavior_type"], observed=True)
        .size()
        .unstack("behavior_type", fill_value=0)
        .reindex(columns=BEHAVIORS, fill_value=0)
        .rename(columns={name: f"train_{name}_count" for name in BEHAVIORS})
        .reset_index()
    )
    for name in BEHAVIORS:
        counts[f"train_{name}_count"] = counts[f"train_{name}_count"].astype(
            "int32"
        )
    return counts


def build_item_popularity_segments(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    item = (
        train.groupby("item_id")
        .agg(
            train_interaction_count=("user_id", "size"),
            train_unique_users=("user_id", "nunique"),
        )
        .reset_index()
        .merge(behavior_counts(train), on="item_id", validate="one_to_one")
    )
    item = item.sort_values(
        ["train_interaction_count", "item_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    item["popularity_rank"] = np.arange(1, len(item) + 1, dtype=np.int32)
    denominator = max(len(item) - 1, 1)
    item["popularity_percentile"] = (
        (item.popularity_rank - 1) / denominator
    ).astype("float32")

    head_end = int(math.floor(0.20 * len(item)))
    torso_end = int(math.floor(0.50 * len(item)))
    segment = np.full(len(item), "Tail", dtype=object)
    segment[:head_end] = "Head"
    segment[head_end:torso_end] = "Torso"
    item["segment"] = pd.Categorical(
        segment, categories=list(SEGMENTS), ordered=True
    )
    for column in ("train_interaction_count", "train_unique_users"):
        item[column] = item[column].astype("int32")

    segment_statistics: dict[str, Any] = {}
    for name in SEGMENTS:
        values = item.loc[item.segment.eq(name), "train_interaction_count"]
        segment_statistics[name] = {
            "items": int(len(values)),
            "minimum_interaction_count": int(values.min()),
            "median_interaction_count": float(values.median()),
            "maximum_interaction_count": int(values.max()),
        }
    metadata = {
        "total_train_known_items": int(len(item)),
        "head_items": head_end,
        "torso_items": torso_end - head_end,
        "tail_items": len(item) - torso_end,
        "sort_rule": "train_interaction_count DESC, item_id ASC",
        "boundary_rule": "direct row-index split after deterministic sort; Head rows [0,floor(20%)), Torso [floor(20%),floor(50%)), Tail remaining",
        "popularity_percentile_direction": "0 = most popular, 1 = least popular",
        "segment_statistics": segment_statistics,
    }
    return item, metadata


def build_itemcf_catalog(train: pd.DataFrame) -> set[int]:
    itemcf_train = (
        train.sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "item_id"], keep="last")
        .sort_values(["user_id", "timestamp"])
        .groupby("user_id", group_keys=False)
        .tail(ITEMCF_MAX_HISTORY)
    )
    return set(itemcf_train.item_id.astype(int))


def annotate_test_ground_truth(
    test_gt: pd.DataFrame,
    train_users: set[int],
    train_items: set[int],
    tt_items: set[int],
    itemcf_items: set[int],
    segments: pd.DataFrame,
    candidates: pd.DataFrame,
    ranker_recommendations: pd.DataFrame,
) -> pd.DataFrame:
    if test_gt.duplicated(["user_id", "item_id"]).any():
        raise AssertionError("Test ground truth contains duplicate user-item pairs")
    result = test_gt[["user_id", "item_id"]].copy()
    result["user_seen_in_train"] = result.user_id.isin(train_users)
    result["target_item_seen_in_train_catalog"] = result.item_id.isin(train_items)
    result["target_item_seen_in_two_tower_vocab"] = result.item_id.isin(tt_items)
    result["target_item_seen_in_itemcf_catalog"] = result.item_id.isin(itemcf_items)

    result["bucket"] = np.select(
        [
            result.user_seen_in_train & result.target_item_seen_in_train_catalog,
            ~result.user_seen_in_train & result.target_item_seen_in_train_catalog,
            result.user_seen_in_train & ~result.target_item_seen_in_train_catalog,
            ~result.user_seen_in_train & ~result.target_item_seen_in_train_catalog,
        ],
        ["A", "B", "C", "D"],
        default="ERROR",
    )
    item_segments = segments[["item_id", "segment"]].copy()
    item_segments["segment"] = item_segments.segment.astype("string")
    result = result.merge(item_segments, how="left", on="item_id", validate="many_to_one")
    result["segment"] = result.segment.fillna("Cold")

    source_hits = candidates[
        ["user_id", "item_id", "from_itemcf", "from_two_tower", "from_popularity"]
    ]
    result = result.merge(
        source_hits, how="left", on=["user_id", "item_id"], validate="one_to_one"
    )
    for column in ("from_itemcf", "from_two_tower", "from_popularity"):
        result[column] = result[column].fillna(0).astype("int8")
    result["candidate_union_hit"] = (
        result[["from_itemcf", "from_two_tower", "from_popularity"]]
        .max(axis=1)
        .astype("int8")
    )

    ranker_hits = ranker_recommendations.loc[
        ranker_recommendations["rank"].le(50), ["user_id", "item_id"]
    ].drop_duplicates()
    ranker_hits["ranker_top50_hit"] = np.int8(1)
    result = result.merge(
        ranker_hits, how="left", on=["user_id", "item_id"], validate="one_to_one"
    )
    result["ranker_top50_hit"] = result.ranker_top50_hit.fillna(0).astype("int8")
    return result


def source_topk(
    candidates: pd.DataFrame, source: str, k: int = 50
) -> pd.DataFrame:
    rank_column = f"{source}_rank"
    frame = candidates.loc[
        candidates[f"from_{source}"].eq(1) & candidates[rank_column].le(k),
        ["user_id", "item_id", rank_column],
    ].copy()
    frame = frame.sort_values(
        ["user_id", rank_column, "item_id"], kind="mergesort"
    ).rename(columns={rank_column: "rank"})
    if frame.duplicated(["user_id", "item_id"]).any():
        raise AssertionError(f"Duplicate {source} recommendations")
    return frame


def recommendations_dict(frame: pd.DataFrame) -> dict[int, list[int]]:
    ordered = frame.sort_values(["user_id", "rank", "item_id"], kind="mergesort")
    return {
        int(user_id): group.item_id.astype(int).tolist()
        for user_id, group in ordered.groupby("user_id", sort=False)
    }


def ndcg_at_k(recommendations: list[int], targets: set[int], k: int) -> float:
    if not targets:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(recommendations[:k], 1)
        if item_id in targets
    )
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(targets), k) + 1)
    )
    return dcg / idcg if idcg else 0.0


def evaluate_population(
    ground_truth: pd.DataFrame,
    recommendations: dict[int, list[int]],
    ks: Iterable[int] = KS,
) -> dict[str, Any]:
    if ground_truth.empty:
        result: dict[str, Any] = {
            "interactions": 0,
            "users": 0,
            "unique_target_items": 0,
            "hitrate_at_50": None,
        }
        for k in ks:
            result[f"hits_at_{k}"] = 0
            result[f"recall_at_{k}"] = None
            result[f"macro_user_recall_at_{k}"] = None
            result[f"ndcg_at_{k}"] = None
        return result

    gt_by_user = ground_truth.groupby("user_id").item_id.apply(set).to_dict()
    result = {
        "interactions": int(len(ground_truth)),
        "users": int(len(gt_by_user)),
        "unique_target_items": int(ground_truth.item_id.nunique()),
    }
    for k in ks:
        hits = 0
        user_recalls = []
        ndcgs = []
        users_with_hit = 0
        for user_id, targets in gt_by_user.items():
            recs = recommendations.get(int(user_id), [])[: int(k)]
            hit_count = len(set(recs) & targets)
            hits += hit_count
            user_recalls.append(hit_count / len(targets))
            ndcgs.append(ndcg_at_k(recs, targets, int(k)))
            if k == 50 and hit_count > 0:
                users_with_hit += 1
        result[f"hits_at_{k}"] = int(hits)
        result[f"recall_at_{k}"] = float(hits / len(ground_truth))
        result[f"macro_user_recall_at_{k}"] = float(np.mean(user_recalls))
        result[f"ndcg_at_{k}"] = float(np.mean(ndcgs))
        if k == 50:
            result["hitrate_at_50"] = float(users_with_hit / len(gt_by_user))
    return result


def evaluate_hit_indicator(
    ground_truth: pd.DataFrame, hit_column: str
) -> dict[str, Any]:
    if ground_truth.empty:
        return {
            "target_interactions": 0,
            "hit_interactions": 0,
            "oracle_recall": None,
            "oracle_hitrate": None,
            "macro_user_oracle_recall": None,
        }
    hits = int(ground_truth[hit_column].sum())
    user_stats = ground_truth.groupby("user_id").agg(
        targets=("item_id", "size"), hits=(hit_column, "sum")
    )
    return {
        "target_interactions": int(len(ground_truth)),
        "hit_interactions": hits,
        "oracle_recall": float(hits / len(ground_truth)),
        "oracle_hitrate": float(user_stats.hits.gt(0).mean()),
        "macro_user_oracle_recall": float((user_stats.hits / user_stats.targets).mean()),
    }


def build_cold_start_outputs(
    annotated: pd.DataFrame,
    recommendations: dict[str, dict[int, list[int]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    total_interactions = len(annotated)
    total_users = annotated.user_id.nunique()
    rows = []
    buckets: dict[str, Any] = {}
    for bucket in ("A", "B", "C", "D"):
        subset = annotated.loc[annotated.bucket.eq(bucket)]
        retrievable = bucket in ("A", "B")
        bucket_summary = {
            "label": BUCKET_LABELS[bucket],
            "interactions": int(len(subset)),
            "unique_users": int(subset.user_id.nunique()),
            "unique_target_items": int(subset.item_id.nunique()),
            "fraction_of_all_test_interactions": float(
                len(subset) / total_interactions
            ),
            "fraction_of_all_test_purchase_users": float(
                subset.user_id.nunique() / total_users
            ),
            "structurally_retrievable": retrievable,
            "interpretation": (
                "train-known target under the closed catalog"
                if retrievable
                else "structurally unretrievable under the current closed-catalog system"
            ),
        }
        buckets[bucket] = bucket_summary
        for method, recs in recommendations.items():
            metrics = evaluate_population(subset, recs)
            rows.append(
                {
                    "bucket": bucket,
                    "bucket_label": BUCKET_LABELS[bucket],
                    "interactions": int(len(subset)),
                    "users": int(subset.user_id.nunique()),
                    "unique_target_items": int(subset.item_id.nunique()),
                    "fraction": float(len(subset) / total_interactions),
                    "fraction_of_all_test_purchase_users": float(
                        subset.user_id.nunique() / total_users
                    ),
                    "method": method,
                    "recall_at_10": metrics["recall_at_10"],
                    "recall_at_20": metrics["recall_at_20"],
                    "recall_at_50": metrics["recall_at_50"],
                    "hitrate_at_50": metrics["hitrate_at_50"],
                    "ndcg_at_10": metrics["ndcg_at_10"],
                    "ndcg_at_20": metrics["ndcg_at_20"],
                    "ndcg_at_50": metrics["ndcg_at_50"],
                    "structurally_retrievable": retrievable,
                    "interpretation": bucket_summary["interpretation"],
                }
            )
    return pd.DataFrame(rows), buckets


def build_long_tail_outputs(
    annotated: pd.DataFrame,
    recommendations: dict[str, dict[int, list[int]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distribution_rows = []
    for segment in (*SEGMENTS, "Cold"):
        subset = annotated.loc[annotated.segment.eq(segment)]
        distribution_rows.append(
            {
                "segment": segment,
                "target_interactions": int(len(subset)),
                "unique_target_items": int(subset.item_id.nunique()),
                "unique_users": int(subset.user_id.nunique()),
                "fraction_of_all_test_purchase_interactions": float(
                    len(subset) / len(annotated)
                ),
            }
        )
    distribution = pd.DataFrame(distribution_rows)

    metric_rows = []
    oracle_rows = []
    for segment in SEGMENTS:
        subset = annotated.loc[annotated.segment.eq(segment)]
        oracle = evaluate_hit_indicator(subset, "candidate_union_hit")
        for method in METHODS:
            metrics = evaluate_population(subset, recommendations[method])
            metric_rows.append(
                {
                    "segment": segment,
                    "method": method,
                    "target_interactions": metrics["interactions"],
                    "target_users": metrics["users"],
                    "hits_at_10": metrics["hits_at_10"],
                    "hits_at_20": metrics["hits_at_20"],
                    "hits_at_50": metrics["hits_at_50"],
                    "recall_at_10": metrics["recall_at_10"],
                    "recall_at_20": metrics["recall_at_20"],
                    "recall_at_50": metrics["recall_at_50"],
                    "hitrate_at_50": metrics["hitrate_at_50"],
                    "ndcg_at_10": metrics["ndcg_at_10"],
                    "ndcg_at_20": metrics["ndcg_at_20"],
                    "ndcg_at_50": metrics["ndcg_at_50"],
                    "metric_definition": "micro interaction Recall; per-segment target interactions are the denominator",
                }
            )
        ranker_metrics = next(
            row
            for row in metric_rows
            if row["segment"] == segment and row["method"] == "Ranker"
        )
        oracle_rows.append(
            {
                "segment": segment,
                **oracle,
                "ranker_recall_at_50": ranker_metrics["recall_at_50"],
                "segment_oracle_gap": (
                    oracle["oracle_recall"] - ranker_metrics["recall_at_50"]
                ),
            }
        )
    return distribution, pd.DataFrame(metric_rows), pd.DataFrame(oracle_rows)


def gini_coefficient(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or np.any(values < 0) or values.sum() == 0:
        return None
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * values) / (n * values.sum())) - (n + 1) / n)


def build_exposure_outputs(
    recommendation_frames: dict[str, pd.DataFrame],
    item_segments: pd.DataFrame,
    expected_users: np.ndarray,
    total_train_interactions: int,
    demand_distribution: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    item_lookup = item_segments[
        [
            "item_id",
            "segment",
            "train_interaction_count",
            "popularity_percentile",
        ]
    ].copy()
    item_lookup["segment"] = item_lookup.segment.astype("string")
    segment_item_counts = item_segments.segment.value_counts().to_dict()
    full_catalog = item_segments.item_id.to_numpy(dtype=np.int64)
    demand_known = demand_distribution.loc[
        demand_distribution.segment.isin(SEGMENTS)
    ].copy()
    known_demand_total = int(demand_known.target_interactions.sum())
    demand_shares = {
        str(row.segment): float(row.target_interactions / known_demand_total)
        for row in demand_known.itertuples()
    }

    exposure_rows = []
    novelty_rows = []
    user_frames = []
    amplification_rows = []
    user_distribution_summary: dict[str, Any] = {}
    for method in EXPOSURE_METHODS:
        recs = recommendation_frames[method].merge(
            item_lookup, how="left", on="item_id", validate="many_to_one"
        )
        recs["exposure_segment"] = recs.segment.fillna("Unmapped")
        recs["novelty"] = np.where(
            recs.train_interaction_count.notna(),
            -np.log2(
                recs.train_interaction_count.astype(float) / total_train_interactions
            ),
            np.nan,
        )
        counts = recs.exposure_segment.value_counts()
        head_count = int(counts.get("Head", 0))
        torso_count = int(counts.get("Torso", 0))
        tail_count = int(counts.get("Tail", 0))
        unmapped_count = int(counts.get("Unmapped", 0))
        total = int(len(recs))
        mapped_total = head_count + torso_count + tail_count
        unique = recs.item_id.nunique()
        unique_by_segment = recs.groupby("exposure_segment").item_id.nunique()
        exposure_counts = (
            recs.groupby("item_id").size().reindex(full_catalog, fill_value=0).to_numpy()
        )

        per_user_novelty = recs.groupby("user_id").novelty.mean()
        novelty_rows.append(
            {
                "method": method,
                "users_with_recommendations": int(recs.user_id.nunique()),
                "mean_user_top50_novelty": float(per_user_novelty.mean()),
                "median_user_top50_novelty": float(per_user_novelty.median()),
                "novelty_definition": "per-user mean of -log2(train_interaction_count / total_train_interactions)",
            }
        )
        exposure_rows.append(
            {
                "method": method,
                "recommendation_count": total,
                "mapped_recommendation_count": mapped_total,
                "unmapped_recommendation_count": unmapped_count,
                "head_count": head_count,
                "torso_count": torso_count,
                "tail_count": tail_count,
                "head_share": float(head_count / mapped_total),
                "torso_share": float(torso_count / mapped_total),
                "tail_share": float(tail_count / mapped_total),
                "unique_items": int(unique),
                "catalog_coverage": float(unique / len(item_segments)),
                "head_coverage": float(
                    unique_by_segment.get("Head", 0) / segment_item_counts["Head"]
                ),
                "torso_coverage": float(
                    unique_by_segment.get("Torso", 0) / segment_item_counts["Torso"]
                ),
                "tail_coverage": float(
                    unique_by_segment.get("Tail", 0) / segment_item_counts["Tail"]
                ),
                "mean_popularity_percentile": float(
                    recs.popularity_percentile.mean()
                ),
                "median_popularity_percentile": float(
                    recs.popularity_percentile.median()
                ),
                "gini_coefficient_full_train_catalog": gini_coefficient(exposure_counts),
                "popularity_percentile_direction": "0 = most popular, 1 = least popular",
            }
        )

        user_counts = (
            recs.groupby(["user_id", "exposure_segment"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["Head", "Torso", "Tail", "Unmapped"], fill_value=0)
            .reindex(expected_users, fill_value=0)
        )
        user_counts.columns = [column.lower() + "_count" for column in user_counts]
        user_counts["recommendation_count"] = user_counts.sum(axis=1)
        for segment in ("head", "torso", "tail"):
            user_counts[f"{segment}_fraction"] = np.divide(
                user_counts[f"{segment}_count"],
                user_counts.recommendation_count,
                out=np.full(len(user_counts), np.nan, dtype=float),
                where=user_counts.recommendation_count.gt(0),
            )
        user_counts = user_counts.reset_index(names="user_id")
        user_counts.insert(0, "method", method)
        user_frames.append(user_counts)
        valid = user_counts.loc[user_counts.recommendation_count.gt(0)]
        user_distribution_summary[method] = {
            segment: {
                "mean": float(valid[f"{segment}_fraction"].mean()),
                "median": float(valid[f"{segment}_fraction"].median()),
                "p10": float(valid[f"{segment}_fraction"].quantile(0.10)),
                "p90": float(valid[f"{segment}_fraction"].quantile(0.90)),
            }
            for segment in ("head", "torso", "tail")
        }

        shares = {
            "Head": head_count / mapped_total,
            "Torso": torso_count / mapped_total,
            "Tail": tail_count / mapped_total,
        }
        for segment in SEGMENTS:
            amplification_rows.append(
                {
                    "method": method,
                    "segment": segment,
                    "train_known_test_target_demand_share": demand_shares[segment],
                    "recommendation_exposure_share": float(shares[segment]),
                    "exposure_minus_demand": float(
                        shares[segment] - demand_shares[segment]
                    ),
                }
            )

    return (
        pd.DataFrame(exposure_rows),
        pd.DataFrame(novelty_rows),
        pd.concat(user_frames, ignore_index=True),
        pd.DataFrame(amplification_rows),
        user_distribution_summary,
    )


def build_two_tower_contribution(annotated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment in (*SEGMENTS, "Cold", "Overall"):
        subset = (
            annotated
            if segment == "Overall"
            else annotated.loc[annotated.segment.eq(segment)]
        )
        tt_hit = subset.from_two_tower.eq(1)
        itemcf_hit = subset.from_itemcf.eq(1)
        exclusive = tt_hit & ~itemcf_hit
        rows.append(
            {
                "segment": segment,
                "target_interactions": int(len(subset)),
                "two_tower_full_quota_hits": int(tt_hit.sum()),
                "itemcf_full_quota_hits": int(itemcf_hit.sum()),
                "two_tower_exclusive_over_itemcf_hits": int(exclusive.sum()),
                "two_tower_exclusive_rate_over_segment_targets": float(
                    exclusive.mean() if len(subset) else 0.0
                ),
                "source_quotas": "Two-Tower Top-100 versus ItemCF Top-200",
            }
        )
    return pd.DataFrame(rows)


def print_tables(
    cold_metrics: pd.DataFrame,
    long_tail_metrics: pd.DataFrame,
    oracle_metrics: pd.DataFrame,
    exposure_metrics: pd.DataFrame,
    novelty_metrics: pd.DataFrame,
    ceiling: dict[str, Any],
) -> None:
    print("\nCOLD START", flush=True)
    print("Bucket  Interactions  Fraction  Ranker R@50  Interpretation", flush=True)
    for bucket in ("A", "B", "C", "D"):
        row = cold_metrics.loc[
            cold_metrics.bucket.eq(bucket) & cold_metrics.method.eq("Ranker")
        ].iloc[0]
        recall = "N/A" if pd.isna(row.recall_at_50) else f"{row.recall_at_50:.6f}"
        print(
            f"{bucket:<7} {int(row.interactions):>12,}  {row.fraction:>8.2%}  "
            f"{recall:>11}  {row.interpretation}",
            flush=True,
        )

    print("\nLONG-TAIL PERFORMANCE", flush=True)
    print("Segment  Targets  ItemCF R50  TT R50  Ranker R50  Oracle R  Ranker NDCG50", flush=True)
    for segment in SEGMENTS:
        rows = long_tail_metrics.loc[long_tail_metrics.segment.eq(segment)]
        get = lambda method, metric: float(rows.loc[rows.method.eq(method), metric].iloc[0])
        oracle = oracle_metrics.loc[oracle_metrics.segment.eq(segment)].iloc[0]
        print(
            f"{segment:<7} {int(oracle.target_interactions):>7,}  "
            f"{get('ItemCF', 'recall_at_50'):.6f}   "
            f"{get('Two-Tower', 'recall_at_50'):.6f}  "
            f"{get('Ranker', 'recall_at_50'):.6f}    "
            f"{oracle.oracle_recall:.6f}   "
            f"{get('Ranker', 'ndcg_at_50'):.6f}",
            flush=True,
        )

    print("\nRECOMMENDATION EXPOSURE", flush=True)
    print("Model       Head %    Torso %   Tail %   Coverage  Tail cov.  Novelty", flush=True)
    for method in EXPOSURE_METHODS:
        exposure = exposure_metrics.loc[exposure_metrics.method.eq(method)].iloc[0]
        novelty = novelty_metrics.loc[novelty_metrics.method.eq(method)].iloc[0]
        print(
            f"{method:<11} {exposure.head_share:>7.2%}  {exposure.torso_share:>7.2%}  "
            f"{exposure.tail_share:>7.2%}  {exposure.catalog_coverage:>8.2%}  "
            f"{exposure.tail_coverage:>8.2%}  {novelty.mean_user_top50_novelty:.4f}",
            flush=True,
        )

    print("\nCLOSED-CATALOG CEILING", flush=True)
    print(f"All test purchase interactions      = {ceiling['all_test_purchase_interactions']:,}")
    print(f"Train-known target interactions     = {ceiling['train_known_target_interactions']:,}")
    print(f"Train-cold target interactions      = {ceiling['train_cold_target_interactions']:,}")
    print(f"Closed-catalog Recall upper bound   = {ceiling['closed_catalog_upper_bound']:.6f}")
    print(f"Ranker all-GT Recall@50             = {ceiling['ranker_all_gt_interaction_recall_at_50']:.6f}")
    print(f"Candidate oracle all-GT Recall      = {ceiling['candidate_oracle_all_gt_interaction_recall']:.6f}")
    print(f"Ranker / closed-catalog ceiling     = {ceiling['fraction_of_closed_catalog_ceiling_reached_by_ranker']:.6f}")
    print(f"Oracle / closed-catalog ceiling     = {ceiling['fraction_of_closed_catalog_ceiling_reached_by_oracle']:.6f}")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {OUTPUT_DIR}")
    if STAGING_DIR.exists():
        raise FileExistsError(f"Stale staging directory exists: {STAGING_DIR}")

    hashes_before = protected_hashes()
    train = pd.read_csv(TRAIN_PATH)
    test_users = np.sort(pd.read_csv(TEST_PATH, usecols=["user_id"]).user_id.unique())
    test_gt = pd.read_csv(TEST_GT_PATH).drop_duplicates(["user_id", "item_id"])
    user_mapping = pd.read_csv(USER_MAPPING_PATH)
    item_mapping = pd.read_csv(ITEM_MAPPING_PATH)
    if int(train.timestamp.max()) != 1512143999:
        raise AssertionError("Train boundary drifted")

    segments, segment_metadata = build_item_popularity_segments(train)
    train_users = set(train.user_id.astype(int))
    train_items = set(segments.item_id.astype(int))
    tt_users = set(user_mapping.user_id.astype(int))
    tt_items = set(item_mapping.item_id.astype(int))
    itemcf_items = build_itemcf_catalog(train)
    if len(itemcf_items) != 222_183:
        raise AssertionError(f"ItemCF catalog drift: {len(itemcf_items):,}")

    print("PREFLIGHT", flush=True)
    print(f"  train users/items: {len(train_users):,}/{len(train_items):,}", flush=True)
    print(f"  Two-Tower users/items: {len(tt_users):,}/{len(tt_items):,}", flush=True)
    print(f"  ItemCF last-50 matrix catalog: {len(itemcf_items):,}", flush=True)
    print(f"  test GT interactions/users: {len(test_gt):,}/{test_gt.user_id.nunique():,}", flush=True)
    print("  mode: descriptive diagnostic only; no model training or selection", flush=True)
    if args.preflight_only:
        print("Preflight complete; no artifacts written.", flush=True)
        return

    STAGING_DIR.mkdir(parents=False, exist_ok=False)
    try:
        candidates = pd.read_parquet(TEST_CANDIDATES_PATH)
        ranker_recommendations = pd.read_parquet(RANKER_RECOMMENDATIONS_PATH)
        recommendation_frames = {
            "Popularity": source_topk(candidates, "popularity", 50),
            "ItemCF": source_topk(candidates, "itemcf", 50),
            "Two-Tower": source_topk(candidates, "two_tower", 50),
            "Ranker": ranker_recommendations[
                ["user_id", "item_id", "rank"]
            ].copy(),
        }
        recommendations = {
            method: recommendations_dict(frame)
            for method, frame in recommendation_frames.items()
        }

        annotated = annotate_test_ground_truth(
            test_gt,
            train_users,
            train_items,
            tt_items,
            itemcf_items,
            segments,
            candidates,
            ranker_recommendations,
        )
        cold_metrics, bucket_summary = build_cold_start_outputs(
            annotated, recommendations
        )
        target_distribution, long_tail_metrics, oracle_metrics = build_long_tail_outputs(
            annotated, recommendations
        )
        exposure_metrics, novelty_metrics, user_exposure, amplification, user_exposure_summary = build_exposure_outputs(
            {method: recommendation_frames[method] for method in EXPOSURE_METHODS},
            segments,
            test_users,
            len(train),
            target_distribution,
        )
        tt_contribution = build_two_tower_contribution(annotated)

        train_known = int(annotated.target_item_seen_in_train_catalog.sum())
        train_cold = int((~annotated.target_item_seen_in_train_catalog).sum())
        tt_cold = int((~annotated.target_item_seen_in_two_tower_vocab).sum())
        itemcf_cold = int((~annotated.target_item_seen_in_itemcf_catalog).sum())
        ranker_hits = int(annotated.ranker_top50_hit.sum())
        oracle_hits = int(annotated.candidate_union_hit.sum())
        closed_ceiling = train_known / len(annotated)
        ranker_micro_recall = ranker_hits / len(annotated)
        oracle_micro_recall = oracle_hits / len(annotated)
        ranker_all = evaluate_population(annotated, recommendations["Ranker"])
        oracle_all = evaluate_hit_indicator(annotated, "candidate_union_hit")
        ceiling = {
            "all_test_purchase_interactions": int(len(annotated)),
            "train_known_target_interactions": train_known,
            "train_cold_target_interactions": train_cold,
            "train_cold_target_unique_items": int(
                annotated.loc[
                    ~annotated.target_item_seen_in_train_catalog, "item_id"
                ].nunique()
            ),
            "closed_catalog_upper_bound": float(closed_ceiling),
            "ranker_all_gt_interaction_hits_at_50": ranker_hits,
            "ranker_all_gt_interaction_recall_at_50": float(ranker_micro_recall),
            "ranker_all_gt_macro_user_recall_at_50": ranker_all[
                "macro_user_recall_at_50"
            ],
            "candidate_oracle_all_gt_interaction_hits": oracle_hits,
            "candidate_oracle_all_gt_interaction_recall": float(oracle_micro_recall),
            "candidate_oracle_all_gt_macro_user_recall": oracle_all[
                "macro_user_oracle_recall"
            ],
            "fraction_of_closed_catalog_ceiling_reached_by_ranker": float(
                ranker_micro_recall / closed_ceiling
            ),
            "fraction_of_closed_catalog_ceiling_reached_by_oracle": float(
                oracle_micro_recall / closed_ceiling
            ),
            "metric_note": "Primary ceiling ratios use interaction-level Recall so numerator and denominator share the same interaction population. Macro per-user Recall is also preserved for compatibility with prior project metrics.",
        }

        cold_users = set(annotated.loc[~annotated.user_seen_in_train, "user_id"].astype(int))
        warm_users = set(annotated.loc[annotated.user_seen_in_train, "user_id"].astype(int))
        cold_user_gt = annotated.loc[annotated.user_id.isin(cold_users)]
        warm_user_gt = annotated.loc[annotated.user_id.isin(warm_users)]
        cold_user_popularity = evaluate_population(
            cold_user_gt, recommendations["Popularity"]
        )
        warm_user_popularity = evaluate_population(
            warm_user_gt, recommendations["Popularity"]
        )

        strict_warm_mask = annotated.user_id.isin(tt_users) & annotated.item_id.isin(
            tt_items
        )
        cold_summary = {
            "definitions": {
                "train_catalog": "every item appearing at least once in train.csv",
                "two_tower_vocabulary": "persisted train-PV item_mapping.csv; narrower than the full train catalog",
                "itemcf_catalog": "items retained by the exact current ItemCF matrix construction: last 50 unique train items per user",
                "cold_item": "target absent from the full train catalog",
                "tail_item": "train-known item in the bottom 50% by deterministic train interaction-count item rank",
            },
            "all_gt": {
                "interactions": int(len(annotated)),
                "users": int(annotated.user_id.nunique()),
            },
            "strict_existing_warm_gt": {
                "definition": "user and item both in persisted Two-Tower train-PV mappings",
                "interactions": int(strict_warm_mask.sum()),
                "users": int(annotated.loc[strict_warm_mask, "user_id"].nunique()),
            },
            "catalogs": {
                "train_users": len(train_users),
                "train_items": len(train_items),
                "two_tower_users": len(tt_users),
                "two_tower_items": len(tt_items),
                "itemcf_matrix_items": len(itemcf_items),
            },
            "cold_target_statistics": {
                "train_cold_interactions": train_cold,
                "train_cold_unique_items": int(
                    annotated.loc[
                        ~annotated.target_item_seen_in_train_catalog, "item_id"
                    ].nunique()
                ),
                "two_tower_vocab_cold_interactions": tt_cold,
                "two_tower_vocab_cold_unique_items": int(
                    annotated.loc[
                        ~annotated.target_item_seen_in_two_tower_vocab, "item_id"
                    ].nunique()
                ),
                "itemcf_catalog_cold_interactions": itemcf_cold,
                "itemcf_catalog_cold_unique_items": int(
                    annotated.loc[
                        ~annotated.target_item_seen_in_itemcf_catalog, "item_id"
                    ].nunique()
                ),
            },
            "buckets": bucket_summary,
            "cold_user_fallback": {
                "cold_user_count": len(cold_users),
                "cold_user_purchase_interactions": int(len(cold_user_gt)),
                "cold_user_warm_item_interactions": int(
                    cold_user_gt.target_item_seen_in_train_catalog.sum()
                ),
                "popularity_metrics_cold_users": cold_user_popularity,
                "popularity_metrics_warm_users": warm_user_popularity,
                "interpretation": (
                    "No train-cold purchase users exist in this test GT; cold-user fallback effectiveness cannot be estimated. The three popularity-only candidate users previously reported are not train-cold users."
                    if not cold_users
                    else "Cold-user sample is small; interpret fallback metrics cautiously."
                ),
            },
        }

        checks = {
            "cold_buckets_mutually_exclusive": bool(
                annotated.bucket.isin(BUCKET_LABELS).all()
            ),
            "cold_buckets_collectively_exhaustive": int(
                annotated.bucket.value_counts().sum()
            )
            == len(annotated),
            "popularity_segments_non_overlapping": not segments.duplicated(
                "item_id"
            ).any(),
            "all_train_items_have_exactly_one_segment": len(segments)
            == len(train_items)
            and segments.segment.notna().all(),
            "cold_items_are_not_tail": bool(
                annotated.loc[
                    ~annotated.target_item_seen_in_train_catalog, "segment"
                ].eq("Cold").all()
            ),
            "target_distribution_sums_to_test_gt": int(
                target_distribution.target_interactions.sum()
            )
            == len(annotated),
            "exposure_partitions_valid": bool(
                (
                    exposure_metrics.head_count
                    + exposure_metrics.torso_count
                    + exposure_metrics.tail_count
                    + exposure_metrics.unmapped_recommendation_count
                ).eq(exposure_metrics.recommendation_count).all()
            ),
            "all_popularity_statistics_are_train_only": True,
            "no_test_driven_training_or_model_selection": True,
        }
        if not all(checks.values()):
            raise AssertionError(f"Analysis validation failed: {checks}")

        segments.to_parquet(
            STAGING_DIR / "item_popularity_segments.parquet",
            index=False,
            compression="zstd",
        )
        annotated.to_parquet(
            STAGING_DIR / "test_purchase_segments.parquet",
            index=False,
            compression="zstd",
        )
        cold_metrics.to_csv(STAGING_DIR / "cold_start_metrics.csv", index=False)
        json_write(STAGING_DIR / "cold_start_summary.json", cold_summary)
        target_distribution.to_csv(
            STAGING_DIR / "long_tail_target_distribution.csv", index=False
        )
        long_tail_metrics.to_csv(STAGING_DIR / "long_tail_metrics.csv", index=False)
        oracle_metrics.to_csv(
            STAGING_DIR / "segment_oracle_metrics.csv", index=False
        )
        exposure_metrics.to_csv(STAGING_DIR / "exposure_metrics.csv", index=False)
        novelty_metrics.to_csv(STAGING_DIR / "novelty_metrics.csv", index=False)
        user_exposure.to_csv(
            STAGING_DIR / "user_exposure_distribution.csv", index=False
        )
        amplification.to_csv(
            STAGING_DIR / "popularity_bias_amplification.csv", index=False
        )
        json_write(STAGING_DIR / "closed_catalog_ceiling.json", ceiling)
        tt_contribution.to_csv(
            STAGING_DIR / "two_tower_segment_contribution.csv", index=False
        )

        hashes_after = protected_hashes()
        if hashes_before != hashes_after:
            raise AssertionError("An existing retriever/recall/ranking artifact changed")
        checks["existing_model_and_pipeline_artifacts_unchanged"] = True

        config = {
            "stage": "Cold Start + Long Tail Segmentation Analysis",
            "analysis_only": True,
            "model_training_performed": False,
            "test_driven_model_selection_performed": False,
            "sources": {
                "train_only_popularity": "data/processed/dev/train.csv",
                "test_ground_truth": "data/processed/dev/test_ground_truth.csv",
                "fixed_candidates": "artifacts/multistage_recall/test_candidates.parquet",
                "fixed_ranker_top50": "artifacts/ranking/test_recommendations.parquet",
            },
            "itemcf_catalog_definition": cold_summary["definitions"]["itemcf_catalog"],
            "popularity_segment_metadata": segment_metadata,
            "metric_definitions": {
                "recall": "micro interaction recall: target hits divided by target interactions in that exact bucket/segment",
                "hitrate": "fraction of bucket/segment users with at least one hit",
                "ndcg": "mean per-user binary-relevance NDCG using only targets in the exact bucket/segment",
                "candidate_oracle": "exact user-item membership anywhere in the fixed union candidate set",
                "exposure_population": f"all {len(test_users):,} users present in test.csv",
                "popularity_percentile": "0 = most popular, 1 = least popular",
            },
            "protected_input_hashes": hashes_before,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
            },
        }
        json_write(STAGING_DIR / "config.json", config)

        summary = {
            "status": "completed",
            "analysis_scope": "descriptive and diagnostic only",
            "cold_start_status": "segmentation and evaluation completed; not solved",
            "long_tail_status": "segmentation, metrics, oracle, exposure, novelty and bias analysis completed",
            "segment_metadata": segment_metadata,
            "cold_start": cold_summary,
            "closed_catalog_ceiling": ceiling,
            "target_distribution": target_distribution.to_dict("records"),
            "long_tail_metrics": long_tail_metrics.to_dict("records"),
            "segment_oracle_metrics": oracle_metrics.to_dict("records"),
            "exposure_metrics": exposure_metrics.to_dict("records"),
            "novelty_metrics": novelty_metrics.to_dict("records"),
            "popularity_bias_amplification": amplification.to_dict("records"),
            "two_tower_segment_contribution": tt_contribution.to_dict("records"),
            "user_exposure_distribution_summary": user_exposure_summary,
            "gini_status": "completed without third-party dependencies",
            "validation_checks": checks,
            "protected_artifacts_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
        json_write(STAGING_DIR / "analysis_summary.json", summary)
        os.replace(STAGING_DIR, OUTPUT_DIR)

        print_tables(
            cold_metrics,
            long_tail_metrics,
            oracle_metrics,
            exposure_metrics,
            novelty_metrics,
            ceiling,
        )
        print(f"\nArtifacts saved to {OUTPUT_DIR}", flush=True)
    except Exception:
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)
        raise


if __name__ == "__main__":
    main()
