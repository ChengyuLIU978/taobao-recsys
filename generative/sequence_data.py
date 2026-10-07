from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


BEHAVIOR_COLUMN_TO_TOKEN = {"pv": "PV", "fav": "FAV", "cart": "CART", "buy": "BUY"}


@dataclass
class SIDCatalog:
    item_to_sid_tokens: dict[int, np.ndarray]
    item_to_raw_sid: dict[int, tuple[int, int, int, int]]
    active_items: set[int]


def load_sid_catalog(semantic_frame: pd.DataFrame, token_manifest: dict[str, Any]) -> SIDCatalog:
    ranges = token_manifest["ranges"]
    item_to_sid_tokens: dict[int, np.ndarray] = {}
    item_to_raw_sid: dict[int, tuple[int, int, int, int]] = {}
    for row in semantic_frame.itertuples(index=False):
        raw = (int(row.c0), int(row.c1), int(row.c2), int(row.suffix))
        tokens = np.asarray([
            raw[0] + int(ranges["level0"]["start"]),
            raw[1] + int(ranges["level1"]["start"]),
            raw[2] + int(ranges["level2"]["start"]),
            raw[3] + int(ranges["suffix"]["start"]),
        ], dtype=np.int16)
        item_id = int(row.item_id)
        item_to_sid_tokens[item_id] = tokens
        item_to_raw_sid[item_id] = raw
    return SIDCatalog(item_to_sid_tokens, item_to_raw_sid, set(item_to_sid_tokens))


def event_tokens(behavior: str, sid_tokens: np.ndarray, token_manifest: dict[str, Any]) -> np.ndarray:
    behavior_name = BEHAVIOR_COLUMN_TO_TOKEN[str(behavior)]
    return np.asarray([
        int(token_manifest["behavior_tokens"][behavior_name]),
        int(sid_tokens[0]), int(sid_tokens[1]), int(sid_tokens[2]), int(sid_tokens[3]),
        int(token_manifest["special_tokens"]["SEP"]),
    ], dtype=np.int16)


def pack_history(events: list[np.ndarray], max_history_items: int) -> tuple[np.ndarray, int]:
    selected = events[-max_history_items:]
    result = np.zeros((max_history_items, 6), dtype=np.int16)
    if selected:
        result[:len(selected)] = np.stack(selected)
    return result, len(selected)


def build_train_purchase_examples(
    train_df: pd.DataFrame,
    catalog: SIDCatalog,
    token_manifest: dict[str, Any],
    max_history_items: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    histories: list[np.ndarray] = []
    lengths: list[int] = []
    targets: list[np.ndarray] = []
    users: list[int] = []
    target_items: list[int] = []
    target_timestamps: list[int] = []
    max_history_timestamps: list[int] = []
    raw_purchase_events = int(train_df.behavior_type.eq("buy").sum())
    target_covered = 0
    history_covered = 0
    eligible_users: set[int] = set()
    sorted_frame = train_df.sort_values(["user_id", "timestamp"], kind="mergesort")
    for user_id, group in sorted_frame.groupby("user_id", sort=False):
        history_events: list[np.ndarray] = []
        history_times: list[int] = []
        # All events sharing a timestamp are simultaneous at this dataset's
        # resolution.  Construct purchase examples before adding any event
        # from that timestamp so every history timestamp is strictly smaller.
        current_timestamp: int | None = None
        pending_events: list[np.ndarray] = []
        for row in group.itertuples(index=False):
            timestamp = int(row.timestamp)
            if current_timestamp is None:
                current_timestamp = timestamp
            elif timestamp != current_timestamp:
                history_events.extend(pending_events)
                history_times.extend([current_timestamp] * len(pending_events))
                pending_events = []
                current_timestamp = timestamp
            item_id = int(row.item_id)
            sid = catalog.item_to_sid_tokens.get(item_id)
            is_purchase = str(row.behavior_type) == "buy"
            if is_purchase and sid is not None:
                target_covered += 1
                eligible_users.add(int(user_id))
                if history_events:
                    history_covered += 1
                    packed, length = pack_history(history_events, max_history_items)
                    histories.append(packed); lengths.append(length)
                    target = np.asarray([
                        token_manifest["special_tokens"]["BOS"], *sid.tolist(), token_manifest["special_tokens"]["EOS"]
                    ], dtype=np.int16)
                    targets.append(target); users.append(int(user_id)); target_items.append(item_id)
                    target_timestamps.append(timestamp); max_history_timestamps.append(int(max(history_times[-max_history_items:])))
            if sid is not None:
                pending_events.append(event_tokens(str(row.behavior_type), sid, token_manifest))
    if not histories:
        raise AssertionError("No generative train examples were constructed")
    arrays = {
        "histories": np.stack(histories), "history_lengths": np.asarray(lengths, dtype=np.int16),
        "targets": np.stack(targets), "user_ids": np.asarray(users, dtype=np.int64),
        "target_item_ids": np.asarray(target_items, dtype=np.int64),
        "target_timestamps": np.asarray(target_timestamps, dtype=np.int64),
        "max_history_timestamps": np.asarray(max_history_timestamps, dtype=np.int64),
    }
    causal = bool(np.all(arrays["max_history_timestamps"] < arrays["target_timestamps"]))
    if not causal:
        raise AssertionError("A train history event is not strictly earlier than its purchase target")
    stats = {
        "purchase_events": raw_purchase_events,
        "eligible_targets": int(target_covered),
        "eligible_users": int(len(eligible_users)),
        "history_covered_examples": int(history_covered),
        "final_examples": int(len(histories)),
        "dropped_target_not_in_active_catalog": int(raw_purchase_events - target_covered),
        "dropped_empty_active_history": int(target_covered - history_covered),
        "average_history_length": float(np.mean(lengths)),
        "median_history_length": float(np.median(lengths)),
        "p90_history_length": float(np.quantile(lengths, 0.9)),
        "history_timestamp_strictly_before_target": causal,
    }
    return arrays, stats


def build_user_histories(
    train_df: pd.DataFrame,
    user_ids: np.ndarray,
    catalog: SIDCatalog,
    token_manifest: dict[str, Any],
    max_history_items: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    requested = set(map(int, user_ids))
    by_user: dict[int, list[np.ndarray]] = {user: [] for user in requested}
    by_user_times: dict[int, list[int]] = {user: [] for user in requested}
    subset = train_df.loc[train_df.user_id.isin(requested)].sort_values(["user_id", "timestamp"], kind="mergesort")
    for row in subset.itertuples(index=False):
        sid = catalog.item_to_sid_tokens.get(int(row.item_id))
        if sid is not None:
            user_id = int(row.user_id)
            by_user[user_id].append(event_tokens(str(row.behavior_type), sid, token_manifest))
            by_user_times[user_id].append(int(row.timestamp))
    kept_users = sorted(user for user, events in by_user.items() if events)
    histories = []; lengths = []; max_timestamps = []
    for user in kept_users:
        packed, length = pack_history(by_user[user], max_history_items)
        histories.append(packed); lengths.append(length); max_timestamps.append(max(by_user_times[user][-max_history_items:]))
    arrays = {
        "user_ids": np.asarray(kept_users, dtype=np.int64),
        "histories": np.stack(histories) if histories else np.empty((0, max_history_items, 6), dtype=np.int16),
        "history_lengths": np.asarray(lengths, dtype=np.int16),
        "max_history_timestamps": np.asarray(max_timestamps, dtype=np.int64),
    }
    stats = {
        "requested_users": int(len(requested)), "users_with_active_history": int(len(kept_users)),
        "users_without_active_history": int(len(requested) - len(kept_users)),
        "average_history_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_history_length": float(np.median(lengths)) if lengths else 0.0,
        "p90_history_length": float(np.quantile(lengths, 0.9)) if lengths else 0.0,
        "history_source": "train.csv only",
    }
    return arrays, stats

