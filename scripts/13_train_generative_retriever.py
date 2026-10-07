from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.common import ANALYSIS_DIR, ARTIFACT_DIR, DATA_DIR, RECALL_DIR, TWO_TOWER_DIR, json_write, load_config, select_device, set_seed, sha256
from generative.evaluation import build_trie, generate_dataframe, incremental_analysis, make_generator, recommendations_dict, segment_metrics
from generative.metrics import evaluate_recommendations, ground_truth_dict
from generative.sequence_data import build_train_purchase_examples, build_user_histories, load_sid_catalog


def source_recommendations(candidates: pd.DataFrame, source: str | None, users: set[int]) -> dict[int, list[int]]:
    subset = candidates.loc[candidates.user_id.isin(users)].copy()
    if source is not None:
        subset = subset.loc[subset[f"from_{source}"].eq(1)].sort_values(["user_id", f"{source}_rank"], kind="mergesort")
    return {int(user): group.item_id.astype(int).tolist() for user, group in subset.groupby("user_id", sort=False)}


def validation_population(validation_gt: pd.DataFrame, user_mapping: pd.DataFrame, active_items: set[int]) -> pd.DataFrame:
    users = set(user_mapping.user_id.astype(int))
    return validation_gt.loc[
        validation_gt.user_id.isin(users) & validation_gt.item_id.isin(active_items), ["user_id", "item_id"]
    ].drop_duplicates().copy()


def histories_for_users(history_arrays: dict[str, np.ndarray], users: list[int]) -> tuple[np.ndarray, np.ndarray]:
    position = {int(user): idx for idx, user in enumerate(history_arrays["user_ids"])}
    available = [user for user in users if int(user) in position]
    histories = np.stack([history_arrays["histories"][position[int(user)]] for user in available])
    return np.asarray(available, dtype=np.int64), histories


def main() -> None:
    config = load_config(); seed = int(config["seed"]); set_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = select_device(); cfg = config["generator"]
    with (ARTIFACT_DIR / "token_manifest.json").open("r", encoding="utf-8") as handle:
        token_manifest = json.load(handle)
    semantic_frame = pd.read_parquet(ARTIFACT_DIR / "semantic_ids.parquet")
    catalog = load_sid_catalog(semantic_frame, token_manifest)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")
    train_arrays, train_stats = build_train_purchase_examples(
        train_df, catalog, token_manifest, int(config["sequence"]["max_history_items"])
    )
    np.savez_compressed(ARTIFACT_DIR / "train_sequences.npz", **train_arrays)
    validation_histories, validation_history_stats = build_user_histories(
        train_df, validation_df.user_id.unique().astype(np.int64), catalog, token_manifest,
        int(config["sequence"]["max_history_items"]),
    )
    np.savez_compressed(ARTIFACT_DIR / "validation_histories.npz", **validation_histories)
    validation_purchases = validation_df.loc[validation_df.behavior_type.eq("buy")]
    covered_purchases = validation_purchases.loc[validation_purchases.item_id.isin(catalog.active_items)]
    history_users = set(map(int, validation_histories["user_ids"]))
    final_validation_events = covered_purchases.loc[covered_purchases.user_id.isin(history_users)]
    validation_stats = {
        "purchase_events": int(len(validation_purchases)), "eligible_targets": int(len(covered_purchases)),
        "eligible_users": int(covered_purchases.user_id.nunique()),
        "history_covered_examples": int(len(final_validation_events)), "final_examples": int(len(final_validation_events)),
        "dropped_target_not_in_active_catalog": int(len(validation_purchases) - len(covered_purchases)),
        "dropped_empty_active_history": int(len(covered_purchases) - len(final_validation_events)),
        **validation_history_stats,
        "history_timestamp_strictly_before_target": bool(train_df.timestamp.max() < validation_purchases.timestamp.min()),
    }
    json_write(ARTIFACT_DIR / "generative_dataset_stats.json", {
        "status": "train_and_validation_completed; test_not_read",
        "history_source": config["sequence"]["history_source"],
        "history_cutoff_protocol": config["sequence"]["history_cutoff_protocol"],
        "train": train_stats, "validation": validation_stats,
        "test": {"status": "not_read_before_checkpoint_selection"},
        "leakage_checks": {
            "generator_training_rows_source": "train.csv only",
            "validation_rows_used_as_training_labels": False,
            "test_rows_or_labels_read": False,
            "active_catalog_selected_without_validation_or_test_targets": True,
        },
    })

    validation_gt = pd.read_csv(DATA_DIR / "validation_ground_truth.csv")
    user_mapping = pd.read_csv(TWO_TOWER_DIR / "user_mapping.csv")
    warm_validation = validation_population(validation_gt, user_mapping, catalog.active_items)
    evaluation_users = sorted(map(int, warm_validation.user_id.unique()))
    eval_user_ids, eval_histories = histories_for_users(validation_histories, evaluation_users)
    if set(eval_user_ids) != set(evaluation_users):
        raise AssertionError("A strict-warm validation user has no active train history")

    trie = build_trie(semantic_frame, token_manifest)
    model = make_generator(config, token_manifest).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    batch_size = int(cfg["batch_size"]); rng = np.random.default_rng(seed)
    checkpoint_path = ARTIFACT_DIR / "generator_best.pt"
    history = []; best_recall = -1.0; best_ndcg = -1.0; best_epoch = 0; stale = 0
    print("GENERATOR TRAINING")
    print(f"device={device}, examples={len(train_arrays['histories']):,}, validation_users={len(eval_user_ids):,}, trie_items={len(trie):,}")
    print(f"config={cfg}")
    for epoch in range(1, int(cfg["epochs"]) + 1):
        started = time.perf_counter(); model.train(); total_loss = 0.0; seen = 0
        order = rng.permutation(len(train_arrays["histories"]))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            histories_tensor = torch.from_numpy(train_arrays["histories"][idx].astype(np.int64)).to(device)
            targets = torch.from_numpy(train_arrays["targets"][idx].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(histories_tensor, targets[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[:, 1:].reshape(-1), ignore_index=0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite generator loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += float(loss.item()) * len(idx); seen += len(idx)
        generated_frame, validity = generate_dataframe(
            model, eval_histories, eval_user_ids, trie, token_manifest, device,
            int(cfg["beam_size"]), int(cfg["evaluation_user_batch_size"]),
        )
        recommendations = recommendations_dict(generated_frame)
        metrics = evaluate_recommendations(recommendations, ground_truth_dict(warm_validation), (10, 20))
        by_k = {row["K"]: row for row in metrics["metrics"]}
        recall20 = float(by_k[20]["Recall"]); ndcg20 = float(by_k[20]["NDCG"])
        row = {
            "epoch": epoch, "train_ce": total_loss / seen,
            "validation_recall_at_10": by_k[10]["Recall"], "validation_recall_at_20": recall20,
            "validation_ndcg_at_20": ndcg20, "valid_generation_rate": validity["valid_generation_rate"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        improved = recall20 > best_recall + 1e-12 or (abs(recall20 - best_recall) <= 1e-12 and ndcg20 > best_ndcg + 1e-12)
        if improved:
            best_recall = recall20; best_ndcg = ndcg20; best_epoch = epoch; stale = 0
            torch.save({
                "model_state_dict": model.state_dict(), "config": config, "best_epoch": epoch,
                "validation_recall_at_20": recall20, "validation_ndcg_at_20": ndcg20,
                "semantic_ids_sha256": sha256(ARTIFACT_DIR / "semantic_ids.parquet"),
                "token_manifest_sha256": sha256(ARTIFACT_DIR / "token_manifest.json"),
            }, checkpoint_path)
        else:
            stale += 1
        print(f"epoch={epoch:02d} train_ce={row['train_ce']:.6f} val_R20={recall20:.6f} val_N20={ndcg20:.6f} valid={validity['valid_generation_rate']:.6f} seconds={row['epoch_seconds']:.1f} best={best_epoch}", flush=True)
        if stale >= int(cfg["early_stopping_patience"]):
            print(f"early_stopping epoch={epoch}")
            break
    with (ARTIFACT_DIR / "generator_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)

    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    generated_frame, validity = generate_dataframe(
        model, eval_histories, eval_user_ids, trie, token_manifest, device,
        int(cfg["beam_size"]), int(cfg["evaluation_user_batch_size"]),
    )
    recommendations = recommendations_dict(generated_frame)
    standalone = evaluate_recommendations(recommendations, ground_truth_dict(warm_validation), config["evaluation"]["ks"])
    segments = pd.read_parquet(ANALYSIS_DIR / "item_popularity_segments.parquet")
    segment_results = segment_metrics(recommendations, warm_validation, segments)
    candidates = pd.read_parquet(RECALL_DIR / "validation_candidates.parquet")
    users = set(map(int, warm_validation.user_id.unique()))
    current_union = source_recommendations(candidates, None, users)
    itemcf = source_recommendations(candidates, "itemcf", users)
    incremental = incremental_analysis(recommendations, current_union, itemcf, warm_validation, segments)
    generated_frame = generated_frame.merge(segments[["item_id", "segment"]], on="item_id", how="left", validate="many_to_one")
    gt_pairs = set(map(tuple, warm_validation[["user_id", "item_id"]].itertuples(index=False, name=None)))
    generated_frame["is_gt"] = [tuple(pair) in gt_pairs for pair in generated_frame[["user_id", "item_id"]].itertuples(index=False, name=None)]
    generated_frame.to_parquet(ARTIFACT_DIR / "validation_generated_recommendations.parquet", index=False, engine="pyarrow")
    validation_metrics = {
        "status": "completed_before_test_evaluation", "selection_population": "strict existing warm validation GT",
        "population": {"users": int(warm_validation.user_id.nunique()), "interactions": int(len(warm_validation)),
                       "all_gt_interactions": int(len(validation_gt)), "active_target_coverage": float(len(warm_validation) / len(validation_gt))},
        "standalone": standalone, "validity": validity, "segment_metrics": segment_results,
        "incremental": incremental, "best_epoch": int(saved["best_epoch"]),
    }
    json_write(ARTIFACT_DIR / "validation_generative_metrics.json", validation_metrics)
    json_write(ARTIFACT_DIR / "generator_config.json", {
        **cfg, "seed": seed, "history_source": config["sequence"]["history_source"],
        "history_cutoff_protocol": config["sequence"]["history_cutoff_protocol"],
        "event_pooling": config["sequence"]["resource_implementation"],
        "selection_rule": "validation Recall@20; NDCG@20 tie-break", "best_epoch": int(saved["best_epoch"]),
        "test_read_during_training_or_selection": False,
    })
    freeze = {
        "status": "configuration_and_validation_frozen_before_test",
        "config_sha256": sha256(ROOT / "configs" / "generative_taobao.json"),
        "generator_checkpoint_sha256": sha256(checkpoint_path),
        "semantic_ids_sha256": sha256(ARTIFACT_DIR / "semantic_ids.parquet"),
        "validation_metrics_sha256": sha256(ARTIFACT_DIR / "validation_generative_metrics.json"),
        "beam_size": int(cfg["beam_size"]), "active_catalog_size": int(len(semantic_frame)),
        "best_epoch": int(saved["best_epoch"]), "test_evaluated": False,
    }
    json_write(ARTIFACT_DIR / "pre_test_freeze.json", freeze)
    print(f"best_epoch={saved['best_epoch']} validation_R20={best_recall:.6f} validation_N20={best_ndcg:.6f}")
    print(f"validation_incremental={incremental}")
    print("TEST HAS NOT BEEN READ OR EVALUATED")


if __name__ == "__main__":
    main()
