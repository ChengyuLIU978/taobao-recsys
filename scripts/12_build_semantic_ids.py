from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.common import ANALYSIS_DIR, ARTIFACT_DIR, json_write, load_config, select_device, set_seed, sha256
from generative.rqvae import RQVAE, utilization_metrics
from generative.semantic_ids import assign_collision_suffix, build_token_manifest, sid_to_tokens


def make_model(cfg: dict) -> RQVAE:
    return RQVAE(
        input_dim=int(cfg["input_dim"]), hidden_dim=int(cfg["hidden_dim"]),
        latent_dim=int(cfg["latent_dim"]), num_codebooks=int(cfg["num_codebooks"]),
        codebook_size=int(cfg["codebook_size"]), beta=float(cfg["beta"]),
    )


@torch.no_grad()
def encode_all(model: RQVAE, embeddings: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    result = np.empty((len(embeddings), 3), dtype=np.int16)
    model.eval()
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        x = torch.from_numpy(np.array(embeddings[start:end], dtype=np.float32, copy=True)).to(device)
        result[start:end] = model.encode_codes(x).cpu().numpy().astype(np.int16)
    return result


def collision_statistics(codes: np.ndarray) -> dict:
    _, counts = np.unique(codes, axis=0, return_counts=True)
    colliding = counts[counts > 1]
    unique_tuples = int(len(counts))
    collision_excess = int(len(codes) - unique_tuples)
    return {
        "items": int(len(codes)),
        "unique_raw_tuples": unique_tuples,
        "collision_excess_items": collision_excess,
        "collision_rate": collision_excess / len(codes),
        "collision_groups": int(len(colliding)),
        "items_in_collision_groups": int(colliding.sum()) if len(colliding) else 0,
        "max_collision_group_size": int(colliding.max()) if len(colliding) else 1,
        "mean_collision_group_size": float(colliding.mean()) if len(colliding) else 1.0,
    }


def exact_prefix_pairs(codes: np.ndarray, prefix: int, limit: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    n = len(codes)
    if prefix == 0:
        pairs = []
        while len(pairs) < limit:
            left = int(rng.integers(n)); right = int(rng.integers(n))
            if left != right and codes[left, 0] != codes[right, 0]:
                pairs.append((left, right))
        return pairs
    keys = codes[:, :prefix]
    order = np.lexsort(tuple(keys[:, i] for i in range(prefix - 1, -1, -1)))
    ordered = keys[order]
    changes = np.ones(n, dtype=bool)
    changes[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    starts = np.flatnonzero(changes)
    ends = np.r_[starts[1:], n]
    candidate_groups = np.flatnonzero((ends - starts) > 1)
    rng.shuffle(candidate_groups)
    pairs: list[tuple[int, int]] = []
    for group in candidate_groups:
        members = order[starts[group]:ends[group]]
        base = int(members[0])
        for other_value in members[1:]:
            other = int(other_value)
            if prefix == 3 or codes[base, prefix] != codes[other, prefix]:
                pairs.append((base, other))
                break
        if len(pairs) >= limit:
            break
    return pairs


def prefix_similarity(embeddings: np.ndarray, codes: np.ndarray, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for prefix in range(4):
        pairs = exact_prefix_pairs(codes, prefix, 5000, rng)
        if pairs:
            left = np.fromiter((p[0] for p in pairs), dtype=np.int64)
            right = np.fromiter((p[1] for p in pairs), dtype=np.int64)
            cosine = np.sum(np.asarray(embeddings[left]) * np.asarray(embeddings[right]), axis=1)
            mean = float(cosine.mean())
            median = float(np.median(cosine))
        else:
            mean = None; median = None
        rows.append({"exact_shared_prefix_length": prefix, "pairs": len(pairs), "mean_cosine": mean, "median_cosine": median})
    return rows


def segment_code_metrics(frame: pd.DataFrame, codebook_size: int) -> list[dict]:
    rows = []
    for segment, group in frame.groupby("segment", sort=False):
        ids = torch.from_numpy(group[["c0", "c1", "c2"]].to_numpy(dtype=np.int64))
        for level in utilization_metrics(ids, codebook_size):
            rows.append({"segment": str(segment), **level})
    return rows


def main() -> None:
    config = load_config(); cfg = config["rqvae"]; seed = int(config["seed"])
    set_seed(seed); device = select_device()
    embeddings = np.load(ARTIFACT_DIR / "v1_item_embeddings.npy", mmap_mode="r")
    item_ids = np.load(ARTIFACT_DIR / "item_ids.npy")
    checkpoint_path = ARTIFACT_DIR / "rqvae_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = make_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    codes = encode_all(model, embeddings, device, int(cfg["batch_size"]))
    suffix = assign_collision_suffix(item_ids, codes)
    collision = collision_statistics(codes)

    segment_frame = pd.read_parquet(ANALYSIS_DIR / "item_popularity_segments.parquet")
    segment_frame = segment_frame[["item_id", "segment", "train_interaction_count"]]
    frame = pd.DataFrame({
        "item_id": item_ids.astype(np.int64),
        "c0": codes[:, 0], "c1": codes[:, 1], "c2": codes[:, 2], "suffix": suffix,
    }).merge(segment_frame, on="item_id", how="left", validate="one_to_one")
    if frame.segment.isna().any():
        raise AssertionError("A V1 item is missing from train-only popularity segments")
    if frame.duplicated(["c0", "c1", "c2", "suffix"]).any():
        raise AssertionError("Final four-part Semantic IDs are not unique")
    if frame.item_id.nunique() != len(frame):
        raise AssertionError("Active item IDs are not unique")
    semantic_path = ARTIFACT_DIR / "semantic_ids.parquet"
    frame.to_parquet(semantic_path, index=False, engine="pyarrow")
    np.savez_compressed(
        ARTIFACT_DIR / "semantic_id_lookup.npz",
        item_ids=item_ids.astype(np.int64), codes=codes.astype(np.int16), suffix=suffix.astype(np.int32),
    )
    manifest = build_token_manifest(int(cfg["codebook_size"]), int(suffix.max()), int(config["sequence"]["max_history_items"]))
    json_write(ARTIFACT_DIR / "token_manifest.json", manifest)
    sid_tokens = sid_to_tokens(codes, suffix, manifest)
    if np.unique(sid_tokens, axis=0).shape[0] != len(frame):
        raise AssertionError("Tokenized final Semantic IDs are not unique")

    stats = {
        "status": "completed",
        "active_catalog_items": int(len(frame)),
        "catalog_mode": config["catalog"]["mode"],
        "raw_tuple": collision,
        "suffix_max": int(suffix.max()),
        "final_sid_unique_count": int(len(frame)),
        "final_sid_collision_count": 0,
        "item_to_sid_to_item_roundtrip": True,
        "segment_items": {str(k): int(v) for k, v in frame.segment.value_counts().items()},
        "segment_codebook_metrics": segment_code_metrics(frame, int(cfg["codebook_size"])),
        "prefix_similarity_diagnostic": prefix_similarity(embeddings, codes, seed),
        "semantic_ids_sha256": sha256(semantic_path),
        "lookup_sha256": sha256(ARTIFACT_DIR / "semantic_id_lookup.npz"),
        "cold_items_injected": False,
        "interpretation": "Collaborative-prefix diagnostic only; this does not establish content semantics.",
    }
    json_write(ARTIFACT_DIR / "semantic_id_stats.json", stats)
    rqvae_metrics_path = ARTIFACT_DIR / "rqvae_metrics.json"
    import json
    with rqvae_metrics_path.open("r", encoding="utf-8") as handle:
        rqvae_metrics = json.load(handle)
    rqvae_metrics["raw_tuple_collision"] = collision
    rqvae_metrics["suffix_max"] = int(suffix.max())
    rqvae_metrics["final_sid_collision_count"] = 0
    json_write(rqvae_metrics_path, rqvae_metrics)
    print("SEMANTIC IDS")
    print(f"items={len(frame):,} unique_raw_tuples={collision['unique_raw_tuples']:,}")
    print(f"collision_rate={collision['collision_rate']:.6f} max_group={collision['max_collision_group_size']} suffix_max={suffix.max()}")
    print("final_sid_collision_count=0 roundtrip=True")
    print(f"prefix_similarity={stats['prefix_similarity_diagnostic']}")


if __name__ == "__main__":
    main()
