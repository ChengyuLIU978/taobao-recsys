from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.common import ANALYSIS_DIR, ARTIFACT_DIR, DATA_DIR, RECALL_DIR, TWO_TOWER_DIR, json_write, load_config, select_device, set_seed, sha256
from generative.evaluation import build_trie, generate_dataframe, incremental_analysis, make_generator, recommendations_dict, segment_metrics
from generative.metrics import evaluate_recommendations, ground_truth_dict, oracle_metrics
from generative.sequence_data import build_user_histories, load_sid_catalog


PROTECTED = [
    TWO_TOWER_DIR / "experiments" / "TT_V1_BUY_UNIFORM" / "checkpoint_best.pt",
    RECALL_DIR / "validation_candidates.parquet", RECALL_DIR / "test_candidates.parquet",
    ROOT / "artifacts" / "ranking" / "ranker.pkl",
    ANALYSIS_DIR / "analysis_summary.json",
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_recommendations(candidates: pd.DataFrame, source: str | None, users: set[int]) -> dict[int, list[int]]:
    subset = candidates.loc[candidates.user_id.isin(users)].copy()
    if source is not None:
        subset = subset.loc[subset[f"from_{source}"].eq(1)].sort_values(["user_id", f"{source}_rank"], kind="mergesort")
    return {int(user): group.item_id.astype(int).tolist() for user, group in subset.groupby("user_id", sort=False)}


def histories_for_users(history_arrays: dict[str, np.ndarray], users: list[int]) -> tuple[np.ndarray, np.ndarray]:
    positions = {int(user): i for i, user in enumerate(history_arrays["user_ids"])}
    available = [user for user in users if user in positions]
    histories = np.stack([history_arrays["histories"][positions[user]] for user in available])
    return np.asarray(available, dtype=np.int64), histories


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not np.any(values):
        return 0.0
    ordered = np.sort(values)
    n = len(ordered)
    return float((2.0 * np.dot(np.arange(1, n + 1), ordered) / (n * ordered.sum())) - (n + 1) / n)


def exposure_metrics(recommendations: pd.DataFrame, segments: pd.DataFrame) -> dict[str, Any]:
    top50 = recommendations.loc[recommendations["rank"].le(50)].copy()
    top50 = top50.drop(columns=[column for column in ("segment", "train_interaction_count") if column in top50.columns])
    top50 = top50.merge(
        segments[["item_id", "segment", "train_interaction_count"]], on="item_id", how="left", validate="many_to_one"
    )
    if top50.segment.isna().any():
        raise AssertionError("Generative recommendation outside the train catalog")
    segment_counts = top50.segment.value_counts()
    catalog_counts = segments.segment.value_counts()
    unique_by_segment = top50.groupby("segment").item_id.nunique()
    exposure = top50.item_id.value_counts().reindex(segments.item_id, fill_value=0).to_numpy(dtype=np.float64)
    total_train_interactions = float(segments.train_interaction_count.sum())
    popularity_probability = top50.train_interaction_count.to_numpy(dtype=np.float64) / total_train_interactions
    top50 = top50.assign(novelty=-np.log2(popularity_probability))
    user_novelty = top50.groupby("user_id").novelty.mean()
    return {
        "recommendation_count": int(len(top50)), "users_with_recommendations": int(top50.user_id.nunique()),
        "head_count": int(segment_counts.get("Head", 0)), "torso_count": int(segment_counts.get("Torso", 0)),
        "tail_count": int(segment_counts.get("Tail", 0)),
        "head_share": float(segment_counts.get("Head", 0) / len(top50)),
        "torso_share": float(segment_counts.get("Torso", 0) / len(top50)),
        "tail_share": float(segment_counts.get("Tail", 0) / len(top50)),
        "unique_items": int(top50.item_id.nunique()),
        "catalog_coverage": float(top50.item_id.nunique() / len(segments)),
        "head_coverage": float(unique_by_segment.get("Head", 0) / catalog_counts["Head"]),
        "torso_coverage": float(unique_by_segment.get("Torso", 0) / catalog_counts["Torso"]),
        "tail_coverage": float(unique_by_segment.get("Tail", 0) / catalog_counts["Tail"]),
        "gini_coefficient_full_train_catalog": gini(exposure),
        "mean_user_top50_novelty": float(user_novelty.mean()),
        "median_user_top50_novelty": float(user_novelty.median()),
    }


def segment_increment_rows(split: str, gt: pd.DataFrame, segments: pd.DataFrame, generated: dict[int, list[int]], union: dict[int, list[int]], itemcf: dict[int, list[int]]) -> list[dict]:
    joined = gt.merge(segments[["item_id", "segment"]], on="item_id", how="left", validate="many_to_one")
    rows = []
    for segment in ("Head", "Torso", "Tail"):
        part = joined.loc[joined.segment.eq(segment), ["user_id", "item_id"]]
        truth = ground_truth_dict(part)
        combined = {user: list(dict.fromkeys(union.get(user, []) + generated.get(user, []))) for user in truth}
        gen_eval = evaluate_recommendations(generated, truth, (20, 50))["metrics"]
        gen_by_k = {row["K"]: row for row in gen_eval}
        itemcf_r50 = evaluate_recommendations(itemcf, truth, (50,))["metrics"][0]
        current = oracle_metrics(union, truth); after = oracle_metrics(combined, truth)
        exclusive = 0
        for user, targets in truth.items():
            exclusive += len((targets & set(generated.get(user, []))) - set(union.get(user, [])))
        rows.append({
            "split": split, "segment": segment, "target_interactions": int(len(part)),
            "generative_recall_at_20": gen_by_k[20]["Recall"], "generative_recall_at_50": gen_by_k[50]["Recall"],
            "itemcf_recall_at_50": itemcf_r50["Recall"], "current_union_oracle_recall": current["Recall"],
            "union_plus_generative_oracle_recall": after["Recall"],
            "macro_oracle_delta": after["Recall"] - current["Recall"], "generative_exclusive_hits": int(exclusive),
        })
    return rows


def shared_prefix(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    count = 0
    for a, b in zip(left, right, strict=True):
        if a != b:
            break
        count += 1
    return count


def failure_analysis(
    validation_gt: pd.DataFrame, itemcf: dict[int, list[int]], generated_frame: pd.DataFrame,
    validation_histories: dict[str, np.ndarray], semantic_frame: pd.DataFrame, token_manifest: dict, seed: int,
) -> pd.DataFrame:
    generated = recommendations_dict(generated_frame)
    rows = []
    segment_map = semantic_frame.set_index("item_id").segment.to_dict()
    raw_sid = {int(r.item_id): (int(r.c0), int(r.c1), int(r.c2), int(r.suffix)) for r in semantic_frame.itertuples()}
    token_to_item = {}
    ranges = token_manifest["ranges"]
    for item, sid in raw_sid.items():
        token_to_item[(sid[0] + ranges["level0"]["start"], sid[1] + ranges["level1"]["start"], sid[2] + ranges["level2"]["start"], sid[3] + ranges["suffix"]["start"])] = item
    behavior_reverse = {int(value): key for key, value in token_manifest["behavior_tokens"].items()}
    history_position = {int(user): i for i, user in enumerate(validation_histories["user_ids"])}
    misses = []
    for pair in validation_gt[["user_id", "item_id"]].drop_duplicates().itertuples(index=False):
        user = int(pair.user_id); target = int(pair.item_id)
        if target in raw_sid and target not in set(itemcf.get(user, [])):
            misses.append((user, target, target in set(generated.get(user, []))))
    hits = [row for row in misses if row[2]]
    non_hits = [row for row in misses if not row[2]]
    rng = np.random.default_rng(seed)
    rng.shuffle(hits); rng.shuffle(non_hits)
    chosen = hits[:10] + non_hits[:max(20 - min(10, len(hits)), 20)]
    for user, target, gen_hit in chosen:
        history_text = []; behavior_text = []
        if user in history_position:
            for event in validation_histories["histories"][history_position[user]]:
                if int(event[0]) == 0:
                    continue
                behavior_text.append(behavior_reverse[int(event[0])])
                history_text.append(str(token_to_item[tuple(map(int, event[1:5]))]))
        top10 = generated.get(user, [])[:10]
        target_sid = raw_sid[target]
        prefix = max((shared_prefix(target_sid, raw_sid[item]) for item in top10), default=0)
        rows.append({
            "user_id": user, "target_item": target, "target_segment": segment_map.get(target),
            "generative_hit": bool(gen_hit), "history_item_ids": " ".join(history_text),
            "history_behaviors": " ".join(behavior_text), "generated_top10": " ".join(map(str, top10)),
            "target_sid": "-".join(map(str, target_sid)), "max_generated_sid_prefix_match": int(prefix),
            "failure_mode": "generative recovered ItemCF miss" if gen_hit else "both ItemCF and Generative missed",
        })
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config(); seed = int(config["seed"]); set_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads())); device = select_device()
    protected_before = {str(path.relative_to(ROOT)): sha256(path) for path in PROTECTED if path.exists()}
    freeze_path = ARTIFACT_DIR / "pre_test_freeze.json"; freeze = read_json(freeze_path)
    if freeze.get("test_evaluated"):
        raise RuntimeError("Final test evaluation is already marked complete; refusing to rerun")
    if freeze["config_sha256"] != sha256(ROOT / "configs" / "generative_taobao.json"):
        raise AssertionError("Frozen config changed before test")
    if freeze["generator_checkpoint_sha256"] != sha256(ARTIFACT_DIR / "generator_best.pt"):
        raise AssertionError("Frozen generator checkpoint changed before test")

    token_manifest = read_json(ARTIFACT_DIR / "token_manifest.json")
    semantic_frame = pd.read_parquet(ARTIFACT_DIR / "semantic_ids.parquet")
    catalog = load_sid_catalog(semantic_frame, token_manifest)
    trie = build_trie(semantic_frame, token_manifest)
    model = make_generator(config, token_manifest).to(device)
    checkpoint = torch.load(ARTIFACT_DIR / "generator_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval()

    # Candidate generation uses split user IDs and train-only history. Test GT is
    # deliberately not loaded until recommendations have been generated/saved.
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_histories, test_history_stats = build_user_histories(
        train_df, test_df.user_id.unique().astype(np.int64), catalog, token_manifest,
        int(config["sequence"]["max_history_items"]),
    )
    np.savez_compressed(ARTIFACT_DIR / "test_histories.npz", **test_histories)
    generated_path = ARTIFACT_DIR / "test_generated_recommendations.parquet"
    base_columns = ["user_id", "rank", "item_id", "beam_score", "c0", "c1", "c2", "suffix"]
    if generated_path.exists():
        persisted = pd.read_parquet(generated_path)
        if not set(base_columns).issubset(persisted.columns):
            raise AssertionError("Persisted test generation artifact has an invalid schema")
        generated_frame = persisted[base_columns].copy()
        attempted = int(generated_frame.user_id.nunique() * int(config["generator"]["beam_size"]))
        validity = {
            "attempted_beams": attempted, "valid_final_sids": int(len(generated_frame)),
            "invalid_or_incomplete_paths": 0, "duplicate_items_removed": 0,
            "valid_generation_rate": float(len(generated_frame) / attempted) if attempted else 0.0,
            "resumed_from_frozen_generation_artifact": True,
        }
        print(f"Resuming exact frozen test generations: users={generated_frame.user_id.nunique():,}, rows={len(generated_frame):,}", flush=True)
    else:
        generated_frame, validity = generate_dataframe(
            model, test_histories["histories"], test_histories["user_ids"], trie, token_manifest, device,
            int(config["generator"]["beam_size"]), int(config["generator"]["evaluation_user_batch_size"]),
        )
        generated_frame.to_parquet(generated_path, index=False, engine="pyarrow")
        print(f"Saved label-free test generations: users={generated_frame.user_id.nunique():,}, rows={len(generated_frame):,}", flush=True)

    # One-time diagnostic label read begins here.
    test_gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
    user_mapping = pd.read_csv(TWO_TOWER_DIR / "user_mapping.csv")
    warm_test = test_gt.loc[
        test_gt.user_id.isin(set(user_mapping.user_id.astype(int))) & test_gt.item_id.isin(catalog.active_items),
        ["user_id", "item_id"],
    ].drop_duplicates()
    generated = recommendations_dict(generated_frame)
    standalone = evaluate_recommendations(generated, ground_truth_dict(warm_test), config["evaluation"]["ks"])
    segments = pd.read_parquet(ANALYSIS_DIR / "item_popularity_segments.parquet")
    segment_results = segment_metrics(generated, warm_test, segments)
    candidates = pd.read_parquet(RECALL_DIR / "test_candidates.parquet")
    users = set(map(int, warm_test.user_id.unique()))
    current_union = source_recommendations(candidates, None, users)
    itemcf = source_recommendations(candidates, "itemcf", users)
    incremental = incremental_analysis(generated, current_union, itemcf, warm_test, segments)
    segment_rows_test = segment_increment_rows("test", warm_test, segments, generated, current_union, itemcf)

    validation_metrics = read_json(ARTIFACT_DIR / "validation_generative_metrics.json")
    validation_gt = pd.read_csv(DATA_DIR / "validation_ground_truth.csv")
    warm_validation = validation_gt.loc[
        validation_gt.user_id.isin(set(user_mapping.user_id.astype(int))) & validation_gt.item_id.isin(catalog.active_items),
        ["user_id", "item_id"],
    ].drop_duplicates()
    validation_generated_frame = pd.read_parquet(ARTIFACT_DIR / "validation_generated_recommendations.parquet")
    validation_generated = recommendations_dict(validation_generated_frame)
    validation_candidates = pd.read_parquet(RECALL_DIR / "validation_candidates.parquet")
    validation_users = set(map(int, warm_validation.user_id.unique()))
    validation_union = source_recommendations(validation_candidates, None, validation_users)
    validation_itemcf = source_recommendations(validation_candidates, "itemcf", validation_users)
    segment_rows_validation = segment_increment_rows(
        "validation", warm_validation, segments, validation_generated, validation_union, validation_itemcf
    )

    generated_frame = generated_frame.merge(segments[["item_id", "segment"]], on="item_id", how="left", validate="many_to_one")
    gt_pairs = set(map(tuple, test_gt[["user_id", "item_id"]].itertuples(index=False, name=None)))
    generated_frame["is_gt"] = [tuple(pair) in gt_pairs for pair in generated_frame[["user_id", "item_id"]].itertuples(index=False, name=None)]
    generated_frame.to_parquet(generated_path, index=False, engine="pyarrow")
    exposure = exposure_metrics(generated_frame, segments)

    incremental_rows = []
    for split, data in (("validation", validation_metrics["incremental"]), ("test", incremental)):
        incremental_rows.append({
            "split": split,
            "current_union_hits": data["current_union_oracle"]["hit_interactions"],
            "generative_hits": data["generative_oracle"]["hit_interactions"],
            "gen_exclusive_vs_itemcf": data["generative_exclusive_vs_itemcf_total"],
            "gen_exclusive_vs_current_union": data["generative_exclusive_vs_current_union_total"],
            "current_union_oracle_recall": data["current_union_oracle"]["Recall"],
            "union_plus_gen_oracle_recall": data["current_union_plus_generative_oracle"]["Recall"],
            "absolute_oracle_delta": data["absolute_macro_oracle_delta"],
            "exclusive_head": data["generative_exclusive_vs_current_union_by_segment"]["Head"],
            "exclusive_torso": data["generative_exclusive_vs_current_union_by_segment"]["Torso"],
            "exclusive_tail": data["generative_exclusive_vs_current_union_by_segment"]["Tail"],
        })
    pd.DataFrame(incremental_rows).to_csv(ARTIFACT_DIR / "incremental_recall_analysis.csv", index=False)
    pd.DataFrame(segment_rows_validation + segment_rows_test).to_csv(ARTIFACT_DIR / "segment_incremental_recall.csv", index=False)

    validation_histories_npz = np.load(ARTIFACT_DIR / "validation_histories.npz")
    validation_histories = {key: validation_histories_npz[key] for key in validation_histories_npz.files}
    failures = failure_analysis(
        warm_validation, validation_itemcf, validation_generated_frame, validation_histories,
        semantic_frame, token_manifest, seed,
    )
    failures.to_csv(ARTIFACT_DIR / "failure_analysis.csv", index=False)

    dataset_stats = read_json(ARTIFACT_DIR / "generative_dataset_stats.json")
    test_purchases = test_df.loc[test_df.behavior_type.eq("buy")]
    covered = test_purchases.loc[test_purchases.item_id.isin(catalog.active_items)]
    history_users = set(map(int, test_histories["user_ids"]))
    final_events = covered.loc[covered.user_id.isin(history_users)]
    dataset_stats["status"] = "train_validation_test_completed"
    dataset_stats["test"] = {
        "purchase_events": int(len(test_purchases)), "eligible_targets": int(len(covered)),
        "eligible_users": int(covered.user_id.nunique()), "history_covered_examples": int(len(final_events)),
        "final_examples": int(len(final_events)), "dropped_target_not_in_active_catalog": int(len(test_purchases) - len(covered)),
        "dropped_empty_active_history": int(len(covered) - len(final_events)), **test_history_stats,
        "history_timestamp_strictly_before_target": bool(train_df.timestamp.max() < test_purchases.timestamp.min()),
    }
    dataset_stats["leakage_checks"]["test_rows_or_labels_read"] = "only after config/checkpoint freeze and label-free generations saved"
    json_write(ARTIFACT_DIR / "generative_dataset_stats.json", dataset_stats)

    test_metrics = {
        "status": "completed_once_after_validation_freeze", "population": {
            "strict_warm_users": int(warm_test.user_id.nunique()), "strict_warm_interactions": int(len(warm_test)),
            "all_gt_users": int(test_gt.user_id.nunique()), "all_gt_interactions": int(len(test_gt)),
            "active_target_coverage": float(len(warm_test) / len(test_gt)),
            "targets_outside_active_v1_catalog": int((~test_gt.item_id.isin(catalog.active_items)).sum()),
            "train_cold_targets": int((~test_gt.item_id.isin(set(train_df.item_id.astype(int)))).sum()),
        },
        "standalone": standalone, "validity": validity, "segment_metrics": segment_results,
        "incremental": incremental, "exposure": exposure, "best_epoch": int(checkpoint["best_epoch"]),
    }
    json_write(ARTIFACT_DIR / "test_generative_metrics.json", test_metrics)

    protected_after = {str(path.relative_to(ROOT)): sha256(path) for path in PROTECTED if path.exists()}
    if protected_before != protected_after:
        raise AssertionError("A protected traditional-system artifact changed")
    summary = {
        "status": "completed",
        "project_boundary": "TIGER-style collaborative Semantic-ID experimental complementary recall source; not exact TIGER and not a cold-start solution",
        "embedding_source": read_json(ARTIFACT_DIR / "embedding_manifest.json"),
        "catalog_mode": config["catalog"]["mode"], "catalog_size": int(len(semantic_frame)),
        "rqvae": read_json(ARTIFACT_DIR / "rqvae_metrics.json"),
        "semantic_ids": read_json(ARTIFACT_DIR / "semantic_id_stats.json"),
        "generator_config": read_json(ARTIFACT_DIR / "generator_config.json"),
        "validation_population": validation_metrics["population"], "validation_metrics": validation_metrics,
        "test_population": test_metrics["population"], "test_metrics": test_metrics,
        "segment_metrics": {"validation": segment_rows_validation, "test": segment_rows_test},
        "incremental_itemcf_hits": {"validation": validation_metrics["incremental"]["generative_exclusive_vs_itemcf_total"], "test": incremental["generative_exclusive_vs_itemcf_total"]},
        "incremental_current_union_hits": {"validation": validation_metrics["incremental"]["generative_exclusive_vs_current_union_total"], "test": incremental["generative_exclusive_vs_current_union_total"]},
        "union_oracle_before": {"validation": validation_metrics["incremental"]["current_union_oracle"]["Recall"], "test": incremental["current_union_oracle"]["Recall"]},
        "union_oracle_after": {"validation": validation_metrics["incremental"]["current_union_plus_generative_oracle"]["Recall"], "test": incremental["current_union_plus_generative_oracle"]["Recall"]},
        "oracle_delta": {"validation": validation_metrics["incremental"]["absolute_macro_oracle_delta"], "test": incremental["absolute_macro_oracle_delta"]},
        "valid_generation_rate": {"validation": validation_metrics["validity"]["valid_generation_rate"], "test": validity["valid_generation_rate"]},
        "tests": {"status": "pending final regression run"},
        "protected_traditional_artifacts_unchanged": True,
        "artifact_paths": sorted(str(path.relative_to(ROOT)) for path in ARTIFACT_DIR.iterdir() if path.is_file()),
    }
    json_write(ARTIFACT_DIR / "generative_summary.json", summary)
    json_write(ARTIFACT_DIR / "final_summary.json", summary)
    freeze["test_evaluated"] = True
    freeze["test_evaluation_status"] = "completed_once"
    freeze["test_metrics_sha256"] = sha256(ARTIFACT_DIR / "test_generative_metrics.json")
    json_write(freeze_path, freeze)
    print("FINAL TEST EVALUATION COMPLETED ONCE")
    print(f"standalone={standalone}")
    print(f"incremental={incremental}")
    print(f"exposure={exposure}")


if __name__ == "__main__":
    main()
