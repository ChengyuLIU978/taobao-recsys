from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .trie import SemanticIDTrie


@torch.no_grad()
def constrained_beam_search_batch(
    model: Any,
    histories: torch.Tensor,
    user_ids: np.ndarray,
    trie: SemanticIDTrie,
    bos_token_id: int,
    beam_size: int,
) -> tuple[list[dict[str, int | float | tuple[int, ...]]], dict[str, int | float]]:
    model.eval()
    memory, memory_padding = model.encode(histories)
    batch_size = histories.shape[0]
    beams: list[list[tuple[tuple[int, ...], float]]] = [[((), 0.0)] for _ in range(batch_size)]
    for _step in range(5):
        flat: list[tuple[int, tuple[int, ...], float]] = []
        for user_pos, user_beams in enumerate(beams):
            flat.extend((user_pos, prefix, score) for prefix, score in user_beams)
        decoder = torch.tensor(
            [[bos_token_id, *prefix] for _, prefix, _ in flat], dtype=torch.long, device=histories.device
        )
        memory_indices = torch.tensor([user_pos for user_pos, _, _ in flat], dtype=torch.long, device=histories.device)
        logits = model.decode(memory[memory_indices], memory_padding[memory_indices], decoder)[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        expansions: list[list[tuple[tuple[int, ...], float]]] = [[] for _ in range(batch_size)]
        for row, (user_pos, prefix, score) in enumerate(flat):
            children = trie.valid_children(prefix)
            for token in children:
                expansions[user_pos].append((prefix + (int(token),), score + float(log_probs[row, token])))
        beams = []
        for user_expansions in expansions:
            user_expansions.sort(key=lambda value: (-value[1], value[0]))
            beams.append(user_expansions[:beam_size])
    rows: list[dict[str, int | float | tuple[int, ...]]] = []
    invalid = 0
    duplicates = 0
    for user_pos, user_beams in enumerate(beams):
        seen: set[int] = set()
        rank = 0
        for path, score in user_beams:
            if not trie.accepts(path):
                invalid += 1
                continue
            item_id = trie.item_for(path)
            if item_id in seen:
                duplicates += 1
                continue
            seen.add(item_id); rank += 1
            rows.append({
                "user_id": int(user_ids[user_pos]), "rank": rank, "item_id": int(item_id),
                "beam_score": float(score), "token_path": path[:4],
            })
    attempted = batch_size * beam_size
    return rows, {
        "attempted_beams": attempted,
        "valid_final_sids": int(len(rows)),
        "invalid_or_incomplete_paths": int(invalid),
        "duplicate_items_removed": int(duplicates),
        "valid_generation_rate": float(len(rows) / attempted) if attempted else 0.0,
    }

