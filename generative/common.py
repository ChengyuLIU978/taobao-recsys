from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "generative"
DATA_DIR = ROOT / "data" / "processed" / "dev"
CONFIG_PATH = ROOT / "configs" / "generative_taobao.json"
V1_DIR = ROOT / "artifacts" / "two_tower" / "experiments" / "TT_V1_BUY_UNIFORM"
TWO_TOWER_DIR = ROOT / "artifacts" / "two_tower"
RECALL_DIR = ROOT / "artifacts" / "multistage_recall"
ANALYSIS_DIR = ROOT / "artifacts" / "analysis"

BEHAVIOR_TO_NAME = {"pv": "PV", "fav": "FAV", "cart": "CART", "buy": "BUY"}


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value)!r}")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=default)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stable_train_mask(item_ids: np.ndarray, validation_fraction: float, seed: int) -> np.ndarray:
    """Stable multiplicative uint64 hash; True means RQ-VAE train."""
    values = item_ids.astype(np.uint64, copy=False)
    mixed = values ^ np.uint64(seed)
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    threshold = int(round(validation_fraction * 10_000))
    return (mixed % np.uint64(10_000)) >= np.uint64(threshold)


def ensure_new_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def runtime_manifest() -> dict[str, Any]:
    import platform

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(select_device()),
        "cpu_count": os.cpu_count(),
    }

