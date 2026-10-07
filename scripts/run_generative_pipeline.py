from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "generative"
STAGES = [
    "10_export_v1_item_embeddings.py",
    "11_train_rqvae.py",
    "12_build_semantic_ids.py",
    "13_train_generative_retriever.py",
    "14_evaluate_generative_recall.py",
]


def main() -> None:
    final_summary = ARTIFACT_DIR / "final_summary.json"
    if final_summary.exists():
        with final_summary.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("status") == "completed":
            print("Generative pipeline is already complete; refusing to retrain or repeat final test evaluation.")
            print(f"catalog_mode={summary['catalog_mode']} catalog_size={summary['catalog_size']}")
            print(f"final_summary={final_summary}")
            return
    for script in STAGES:
        path = ROOT / "scripts" / script
        print(f"RUNNING {path.name}", flush=True)
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
