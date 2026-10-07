from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative.common import ARTIFACT_DIR, ensure_new_artifact_dir, json_write, load_config, select_device, set_seed, sha256, stable_train_mask
from generative.rqvae import RQVAE, utilization_metrics


def make_model(cfg: dict) -> RQVAE:
    return RQVAE(
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        latent_dim=int(cfg["latent_dim"]),
        num_codebooks=int(cfg["num_codebooks"]),
        codebook_size=int(cfg["codebook_size"]),
        beta=float(cfg["beta"]),
    )


@torch.no_grad()
def initialize_codebooks(model: RQVAE, embeddings: np.ndarray, train_indices: np.ndarray, cfg: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    max_samples = min(int(cfg["kmeans_max_samples"]), len(train_indices))
    sample_indices = np.sort(rng.choice(train_indices, size=max_samples, replace=False))
    parts = []
    for start in range(0, max_samples, 8192):
        batch = torch.from_numpy(np.asarray(embeddings[sample_indices[start:start + 8192]], dtype=np.float32))
        parts.append(model.encoder(batch).cpu().numpy())
    residual = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    diagnostics = []
    for level in range(model.quantizer.num_codebooks):
        kmeans = MiniBatchKMeans(
            n_clusters=model.quantizer.codebook_size,
            random_state=seed + level,
            batch_size=int(cfg["kmeans_batch_size"]),
            max_iter=int(cfg["kmeans_max_iter"]),
            n_init=1,
            reassignment_ratio=0.01,
        )
        labels = kmeans.fit_predict(residual)
        centers = kmeans.cluster_centers_.astype(np.float32)
        model.quantizer.codebooks[level].copy_(torch.from_numpy(centers))
        counts = np.bincount(labels, minlength=model.quantizer.codebook_size)
        diagnostics.append({
            "level": level,
            "active_codes": int((counts > 0).sum()),
            "most_used_share": float(counts.max() / counts.sum()),
            "inertia": float(kmeans.inertia_),
        })
        residual = residual - centers[labels]
    return {"sample_count": max_samples, "levels": diagnostics}


@torch.no_grad()
def evaluate(model: RQVAE, embeddings: np.ndarray, indices: np.ndarray, batch_size: int, device: torch.device, codebook_size: int) -> dict:
    model.eval()
    sums = {"total_loss": 0.0, "reconstruction_loss": 0.0, "codebook_loss": 0.0, "commitment_loss": 0.0}
    all_codes = []
    count = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        x = torch.from_numpy(np.asarray(embeddings[batch_idx], dtype=np.float32)).to(device)
        out = model(x)
        n = len(batch_idx)
        count += n
        for key in sums:
            sums[key] += float(getattr(out, key).item()) * n
        all_codes.append(out.code_ids.cpu())
    metrics = {key: value / count for key, value in sums.items()}
    metrics["codebooks"] = utilization_metrics(torch.cat(all_codes), codebook_size)
    metrics["mean_utilization"] = float(np.mean([row["utilization"] for row in metrics["codebooks"]]))
    return metrics


def main() -> None:
    config = load_config()
    cfg = config["rqvae"]
    seed = int(config["seed"])
    set_seed(seed)
    ensure_new_artifact_dir()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = select_device()
    embeddings_path = ARTIFACT_DIR / "v1_item_embeddings.npy"
    item_ids_path = ARTIFACT_DIR / "item_ids.npy"
    if not embeddings_path.exists() or not item_ids_path.exists():
        raise FileNotFoundError("Run scripts/10_export_v1_item_embeddings.py first")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    item_ids = np.load(item_ids_path, mmap_mode="r")
    train_mask = stable_train_mask(item_ids, float(cfg["validation_fraction"]), seed)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(~train_mask)
    if not len(validation_indices):
        raise AssertionError("Empty deterministic RQ-VAE validation split")

    model = make_model(cfg).cpu()
    init_metrics = initialize_codebooks(model, embeddings, train_indices, cfg, seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    batch_size = int(cfg["batch_size"])
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    checkpoint_path = ARTIFACT_DIR / "rqvae_best.pt"
    best_reconstruction = float("inf")
    best_utilization = -1.0
    best_epoch = 0
    stale = 0

    print("RQ-VAE TRAINING")
    print(f"device={device}, catalog={len(item_ids):,}, train={len(train_indices):,}, validation={len(validation_indices):,}")
    print(f"config={cfg}")
    for epoch in range(1, int(cfg["epochs"]) + 1):
        started = time.perf_counter()
        order = rng.permutation(train_indices)
        model.train()
        train_sums = {"total_loss": 0.0, "reconstruction_loss": 0.0, "codebook_loss": 0.0, "commitment_loss": 0.0}
        seen = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            x = torch.from_numpy(np.asarray(embeddings[batch_idx], dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            out.total_loss.backward()
            optimizer.step()
            n = len(batch_idx)
            seen += n
            for key in train_sums:
                train_sums[key] += float(getattr(out, key).item()) * n
        validation = evaluate(model, embeddings, validation_indices, batch_size, device, int(cfg["codebook_size"]))
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / seen for key, value in train_sums.items()},
            **{f"validation_{key}": validation[key] for key in train_sums},
            "utilization_level0": validation["codebooks"][0]["utilization"],
            "utilization_level1": validation["codebooks"][1]["utilization"],
            "utilization_level2": validation["codebooks"][2]["utilization"],
            "perplexity_level0": validation["codebooks"][0]["perplexity"],
            "perplexity_level1": validation["codebooks"][1]["perplexity"],
            "perplexity_level2": validation["codebooks"][2]["perplexity"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        reconstruction = float(validation["reconstruction_loss"])
        utilization = float(validation["mean_utilization"])
        improved = reconstruction < best_reconstruction - 1e-12 or (
            abs(reconstruction - best_reconstruction) <= 1e-12 and utilization > best_utilization
        )
        if improved:
            best_reconstruction = reconstruction
            best_utilization = utilization
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "seed": seed,
                "best_epoch": best_epoch,
                "validation_metrics": validation,
                "embedding_manifest_sha256": sha256(ARTIFACT_DIR / "embedding_manifest.json"),
            }, checkpoint_path)
        else:
            stale += 1
        print(
            f"epoch={epoch:02d} train_total={row['train_total_loss']:.6f} "
            f"val_recon={reconstruction:.6f} util={utilization:.3f} "
            f"seconds={row['epoch_seconds']:.1f} best={best_epoch}",
            flush=True,
        )
        if stale >= int(cfg["early_stopping_patience"]):
            print(f"early_stopping epoch={epoch}")
            break

    with (ARTIFACT_DIR / "rqvae_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    final_validation = evaluate(model, embeddings, validation_indices, batch_size, device, int(cfg["codebook_size"]))
    rqvae_config = {
        **cfg,
        "seed": seed,
        "catalog_mode": config["catalog"]["mode"],
        "catalog_size": int(len(item_ids)),
        "train_items": int(len(train_indices)),
        "validation_items": int(len(validation_indices)),
        "split_source": "stable uint64 item_id hash; test behavior/labels unused",
        "device": str(device),
        "initialization_diagnostics": init_metrics,
    }
    json_write(ARTIFACT_DIR / "rqvae_config.json", rqvae_config)
    json_write(ARTIFACT_DIR / "rqvae_metrics.json", {
        "status": "completed",
        "best_epoch": int(saved["best_epoch"]),
        "validation": final_validation,
        "checkpoint_selection": "minimum validation reconstruction loss; mean utilization tie-break",
        "codebook_collapse": any(row["utilization"] < 0.1 for row in final_validation["codebooks"]),
        "checkpoint_sha256": sha256(checkpoint_path),
    })
    print(f"best_epoch={saved['best_epoch']} validation_reconstruction={final_validation['reconstruction_loss']:.8f}")
    print(f"codebooks={final_validation['codebooks']}")


if __name__ == "__main__":
    main()
