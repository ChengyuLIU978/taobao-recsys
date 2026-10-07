from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.common import ARTIFACT_DIR, TWO_TOWER_DIR, V1_DIR, ensure_new_artifact_dir, json_write, load_config, runtime_manifest, sha256
from generative.embeddings import TwoTowerModel


def main() -> None:
    config = load_config()
    ensure_new_artifact_dir()
    checkpoint_path = V1_DIR / "checkpoint_best.pt"
    mapping_path = TWO_TOWER_DIR / "item_mapping.csv"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    item_mapping = pd.read_csv(mapping_path).sort_values("item_idx", kind="mergesort")
    if not np.array_equal(item_mapping.item_idx.to_numpy(), np.arange(len(item_mapping))):
        raise AssertionError("Persisted item mapping is not a contiguous item_idx ordering")
    if len(item_mapping) != int(checkpoint["num_items"]):
        raise AssertionError("Checkpoint and persisted item mapping sizes differ")

    model = TwoTowerModel(
        num_users=int(checkpoint["num_users"]),
        num_items=int(checkpoint["num_items"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        temperature=float(checkpoint["temperature"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    item_count = len(item_mapping)
    output_dim = int(checkpoint["output_dim"])
    embeddings = np.empty((item_count, output_dim), dtype=np.float32)
    batch_size = int(config["embedding"]["batch_size"])
    with torch.inference_mode():
        for start in range(0, item_count, batch_size):
            end = min(start + batch_size, item_count)
            ids = torch.arange(start, end, dtype=torch.long)
            embeddings[start:end] = model.encode_items(ids).cpu().numpy().astype(np.float32)
    item_ids = item_mapping.item_id.to_numpy(dtype=np.int64)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all():
        raise AssertionError("Non-finite V1 item embedding detected")
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise AssertionError("V1 item tower output is not L2 normalized")

    rng = np.random.default_rng(int(config["seed"]))
    check_indices = np.sort(rng.choice(item_count, size=20, replace=False))
    roundtrip = item_mapping.set_index("item_idx").loc[check_indices, "item_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(roundtrip, item_ids[check_indices]):
        raise AssertionError("item_ids[i] does not match the persisted mapping")

    embedding_path = ARTIFACT_DIR / "v1_item_embeddings.npy"
    item_ids_path = ARTIFACT_DIR / "item_ids.npy"
    np.save(embedding_path, embeddings, allow_pickle=False)
    np.save(item_ids_path, item_ids, allow_pickle=False)
    manifest = {
        "status": "completed",
        "source_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_item_mapping": str(mapping_path.relative_to(ROOT)),
        "source_item_mapping_sha256": sha256(mapping_path),
        "strict_state_dict_load": True,
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_seed": int(checkpoint["config"]["seed"]),
        "item_count": int(item_count),
        "embedding_dimension": int(output_dim),
        "dtype": str(embeddings.dtype),
        "normalization": "L2 normalized final V1 item-tower representation",
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "mapping_checks": [
            {"item_idx": int(i), "item_id": int(item_ids[i])} for i in check_indices
        ],
        "embedding_sha256": sha256(embedding_path),
        "item_ids_sha256": sha256(item_ids_path),
        "runtime": runtime_manifest(),
    }
    json_write(ARTIFACT_DIR / "embedding_manifest.json", manifest)
    print("V1 EMBEDDING EXPORT")
    print(f"checkpoint={checkpoint_path}")
    print(f"checkpoint_sha256={manifest['source_checkpoint_sha256']}")
    print(f"mapping_sha256={manifest['source_item_mapping_sha256']}")
    print(f"shape={embeddings.shape}, dtype={embeddings.dtype}, strict_load=True")
    print(f"norm_range=[{norms.min():.7f}, {norms.max():.7f}]")


if __name__ == "__main__":
    main()

