from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "EOS": 2, "SEP": 3}
BEHAVIOR_TOKENS = {"PV": 4, "FAV": 5, "CART": 6, "BUY": 7}


def assign_collision_suffix(item_ids: np.ndarray, codes: np.ndarray) -> np.ndarray:
    if codes.ndim != 2 or codes.shape[1] != 3:
        raise ValueError("Expected [items, 3] raw semantic codes")
    frame = pd.DataFrame({
        "item_id": item_ids.astype(np.int64),
        "c0": codes[:, 0].astype(np.int16),
        "c1": codes[:, 1].astype(np.int16),
        "c2": codes[:, 2].astype(np.int16),
    })
    ordered = frame.sort_values(["c0", "c1", "c2", "item_id"], kind="mergesort")
    ordered["suffix"] = ordered.groupby(["c0", "c1", "c2"], sort=False).cumcount()
    return ordered.sort_index().suffix.to_numpy(dtype=np.int32)


def build_token_manifest(codebook_size: int, max_suffix: int, max_history_items: int) -> dict[str, Any]:
    l0_start = max(BEHAVIOR_TOKENS.values()) + 1
    l1_start = l0_start + codebook_size
    l2_start = l1_start + codebook_size
    suffix_start = l2_start + codebook_size
    return {
        "special_tokens": SPECIAL_TOKENS,
        "behavior_tokens": BEHAVIOR_TOKENS,
        "ranges": {
            "level0": {"start": l0_start, "end_inclusive": l0_start + codebook_size - 1},
            "level1": {"start": l1_start, "end_inclusive": l1_start + codebook_size - 1},
            "level2": {"start": l2_start, "end_inclusive": l2_start + codebook_size - 1},
            "suffix": {"start": suffix_start, "end_inclusive": suffix_start + max_suffix},
        },
        "codebook_size": int(codebook_size),
        "max_suffix": int(max_suffix),
        "vocab_size": int(suffix_start + max_suffix + 1),
        "event_width": 6,
        "event_layout": ["behavior", "level0", "level1", "level2", "suffix", "SEP"],
        "max_history_items": int(max_history_items),
        "max_flat_history_tokens": int(max_history_items * 6),
        "decoder_target_layout": ["BOS", "level0", "level1", "level2", "suffix", "EOS"],
        "ranges_are_non_overlapping": True,
    }


def sid_to_tokens(codes: np.ndarray, suffixes: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    ranges = manifest["ranges"]
    tokens = np.empty((len(codes), 4), dtype=np.int64)
    tokens[:, 0] = codes[:, 0] + int(ranges["level0"]["start"])
    tokens[:, 1] = codes[:, 1] + int(ranges["level1"]["start"])
    tokens[:, 2] = codes[:, 2] + int(ranges["level2"]["start"])
    tokens[:, 3] = suffixes + int(ranges["suffix"]["start"])
    return tokens


def tokens_to_sid(tokens: tuple[int, int, int, int], manifest: dict[str, Any]) -> tuple[int, int, int, int]:
    ranges = manifest["ranges"]
    return (
        int(tokens[0] - ranges["level0"]["start"]),
        int(tokens[1] - ranges["level1"]["start"]),
        int(tokens[2] - ranges["level2"]["start"]),
        int(tokens[3] - ranges["suffix"]["start"]),
    )

