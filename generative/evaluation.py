from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from .beam_search import constrained_beam_search_batch
from .metrics import evaluate_recommendations, ground_truth_dict, oracle_metrics
from .semantic_ids import sid_to_tokens, tokens_to_sid
from .transformer import GenerativeRetriever
from .trie import SemanticIDTrie


def make_generator(config: dict[str, Any], token_manifest: dict[str, Any]) -> GenerativeRetriever:
    cfg = config["generator"]
    return GenerativeRetriever(
        vocab_size=int(token_manifest["vocab_size"]), d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]), encoder_layers=int(cfg["encoder_layers"]),
        decoder_layers=int(cfg["decoder_layers"]), dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]), max_history_items=int(config["sequence"]["max_history_items"]),
        pad_token_id=int(token_manifest["special_tokens"]["PAD"]),
    )


def build_trie(semantic_frame: pd.DataFrame, token_manifest: dict[str, Any]) -> SemanticIDTrie:
    codes = semantic_frame[["c0", "c1", "c2"]].to_numpy(dtype=np.int64)
    suffix = semantic_frame.suffix.to_numpy(dtype=np.int64)
    paths = sid_to_tokens(codes, suffix, token_manifest)
    trie = SemanticIDTrie(int(token_manifest["special_tokens"]["EOS"]))
    for path, item_id in zip(paths, semantic_frame.item_id.to_numpy(dtype=np.int64), strict=True):
        trie.insert(tuple(map(int, path)), int(item_id))
    return trie


def generate_dataframe(
    model: GenerativeRetriever,
    histories: np.ndarray,
    user_ids: np.ndarray,
    trie: SemanticIDTrie,
    token_manifest: dict[str, Any],
    device: torch.device,
    beam_size: int,
    user_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_rows: list[dict] = []
    totals = {"attempted_beams": 0, "valid_final_sids": 0, "invalid_or_incomplete_paths": 0, "duplicate_items_removed": 0}
    for start in range(0, len(user_ids), user_batch_size):
        end = min(start + user_batch_size, len(user_ids))
        tensor = torch.from_numpy(histories[start:end].astype(np.int64, copy=False)).to(device)
        rows, stats = constrained_beam_search_batch(
            model, tensor, user_ids[start:end], trie,
            int(token_manifest["special_tokens"]["BOS"]), beam_size,
        )
        all_rows.extend(rows)
        for key in totals:
            totals[key] += int(stats[key])
    output = []
    for row in all_rows:
        c0, c1, c2, suffix = tokens_to_sid(row.pop("token_path"), token_manifest)
        output.append({**row, "c0": c0, "c1": c1, "c2": c2, "suffix": suffix})
    totals["valid_generation_rate"] = (
        totals["valid_final_sids"] / totals["attempted_beams"] if totals["attempted_beams"] else 0.0
    )
    frame = pd.DataFrame(output, columns=["user_id", "rank", "item_id", "beam_score", "c0", "c1", "c2", "suffix"])
    return frame, totals


def recommendations_dict(frame: pd.DataFrame) -> dict[int, list[int]]:
    ordered = frame.sort_values(["user_id", "rank"], kind="mergesort")
    return {int(user): group.item_id.astype(int).tolist() for user, group in ordered.groupby("user_id", sort=False)}


def segment_metrics(recommendations: dict[int, list[int]], gt_frame: pd.DataFrame, segments: pd.DataFrame) -> list[dict]:
    joined = gt_frame.merge(segments[["item_id", "segment"]], on="item_id", how="left", validate="many_to_one")
    rows = []
    for segment in ("Head", "Torso", "Tail"):
        part = joined.loc[joined.segment.eq(segment), ["user_id", "item_id"]]
        result = evaluate_recommendations(recommendations, ground_truth_dict(part), (20, 50))
        by_k = {row["K"]: row for row in result["metrics"]}
        rows.append({
            "segment": segment, "target_interactions": int(len(part)), "target_users": int(part.user_id.nunique()),
            "recall_at_20": by_k[20]["Recall"], "recall_at_50": by_k[50]["Recall"],
            "interaction_recall_at_20": by_k[20]["interaction_recall"],
            "interaction_recall_at_50": by_k[50]["interaction_recall"],
            "ndcg_at_20": by_k[20]["NDCG"], "ndcg_at_50": by_k[50]["NDCG"],
        })
    return rows


def incremental_analysis(
    generated: dict[int, list[int]], current_union: dict[int, list[int]], itemcf: dict[int, list[int]],
    gt_frame: pd.DataFrame, segments: pd.DataFrame,
) -> dict[str, Any]:
    gt = ground_truth_dict(gt_frame)
    combined = {user: list(dict.fromkeys(current_union.get(user, []) + generated.get(user, []))) for user in gt}
    current_metrics = oracle_metrics(current_union, gt)
    combined_metrics = oracle_metrics(combined, gt)
    gen_metrics = oracle_metrics(generated, gt)
    itemcf_metrics = oracle_metrics(itemcf, gt)
    segment_map = segments.set_index("item_id").segment.to_dict()
    exclusive_itemcf = {"Head": 0, "Torso": 0, "Tail": 0, "Cold": 0}
    exclusive_union = {"Head": 0, "Torso": 0, "Tail": 0, "Cold": 0}
    gen_hits = {"Head": 0, "Torso": 0, "Tail": 0, "Cold": 0}
    for user, targets in gt.items():
        gen_set = set(generated.get(user, [])); union_set = set(current_union.get(user, [])); itemcf_set = set(itemcf.get(user, []))
        for target in targets:
            segment = str(segment_map.get(target, "Cold"))
            if target in gen_set:
                gen_hits[segment] += 1
                if target not in itemcf_set:
                    exclusive_itemcf[segment] += 1
                if target not in union_set:
                    exclusive_union[segment] += 1
    return {
        "population_users": len(gt), "population_interactions": int(sum(map(len, gt.values()))),
        "itemcf_oracle": itemcf_metrics, "current_union_oracle": current_metrics,
        "generative_oracle": gen_metrics, "current_union_plus_generative_oracle": combined_metrics,
        "absolute_macro_oracle_delta": combined_metrics["Recall"] - current_metrics["Recall"],
        "absolute_interaction_oracle_delta": combined_metrics["interaction_recall"] - current_metrics["interaction_recall"],
        "generative_hits_by_segment": gen_hits,
        "generative_exclusive_vs_itemcf_by_segment": exclusive_itemcf,
        "generative_exclusive_vs_current_union_by_segment": exclusive_union,
        "generative_exclusive_vs_itemcf_total": int(sum(exclusive_itemcf.values())),
        "generative_exclusive_vs_current_union_total": int(sum(exclusive_union.values())),
    }

