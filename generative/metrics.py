from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def ground_truth_dict(frame: pd.DataFrame) -> dict[int, set[int]]:
    return {int(user): set(map(int, group.item_id)) for user, group in frame.groupby("user_id", sort=False)}


def evaluate_recommendations(recommendations: dict[int, list[int]], ground_truth: dict[int, set[int]], ks: Iterable[int]) -> dict:
    rows = []
    for k in ks:
        recalls = []; hitrates = []; ndcgs = []; interaction_hits = 0; total = 0
        for user, targets in ground_truth.items():
            recs = recommendations.get(int(user), [])[:int(k)]
            hits = set(recs) & targets
            recalls.append(len(hits) / len(targets) if targets else 0.0)
            hitrates.append(float(bool(hits)))
            dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(recs, 1) if item in targets)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(targets), int(k)) + 1))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            interaction_hits += len(hits); total += len(targets)
        rows.append({
            "K": int(k), "Recall": float(np.mean(recalls)) if recalls else 0.0,
            "HitRate": float(np.mean(hitrates)) if hitrates else 0.0,
            "NDCG": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "hit_interactions": int(interaction_hits), "total_interactions": int(total),
            "interaction_recall": float(interaction_hits / total) if total else 0.0,
        })
    return {"users": len(ground_truth), "interactions": int(sum(map(len, ground_truth.values()))), "metrics": rows}


def oracle_metrics(recommendations: dict[int, list[int]], ground_truth: dict[int, set[int]]) -> dict:
    recalls = []; hitrates = []; hits = 0; total = 0
    for user, targets in ground_truth.items():
        recalled = set(recommendations.get(int(user), []))
        count = len(recalled & targets)
        recalls.append(count / len(targets)); hitrates.append(float(count > 0)); hits += count; total += len(targets)
    return {
        "Recall": float(np.mean(recalls)) if recalls else 0.0,
        "HitRate": float(np.mean(hitrates)) if hitrates else 0.0,
        "hit_interactions": int(hits), "total_interactions": int(total),
        "interaction_recall": float(hits / total) if total else 0.0,
    }

