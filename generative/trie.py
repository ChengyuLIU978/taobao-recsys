from __future__ import annotations

from collections import defaultdict


class SemanticIDTrie:
    """Compact four-level prefix constraint with implicit EOS transition."""

    def __init__(self, eos_token_id: int):
        self.eos_token_id = int(eos_token_id)
        self._children: list[dict[tuple[int, ...], set[int]]] = [defaultdict(set) for _ in range(4)]
        self._items: dict[tuple[int, int, int, int], int] = {}

    def insert(self, path: tuple[int, int, int, int], item_id: int) -> None:
        if len(path) != 4:
            raise ValueError("Semantic token path must contain four tokens")
        for depth, token in enumerate(path):
            self._children[depth][path[:depth]].add(int(token))
        if path in self._items:
            raise ValueError("Duplicate final Semantic ID path")
        self._items[path] = int(item_id)

    def valid_children(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        if len(prefix) < 4:
            return tuple(sorted(self._children[len(prefix)].get(prefix, ())))
        if len(prefix) == 4 and prefix in self._items:
            return (self.eos_token_id,)
        return ()

    def accepts(self, path: tuple[int, ...]) -> bool:
        return len(path) == 5 and path[-1] == self.eos_token_id and path[:4] in self._items

    def is_valid_prefix(self, prefix: tuple[int, ...]) -> bool:
        if len(prefix) <= 4:
            return bool(self.valid_children(prefix)) or (len(prefix) == 4 and prefix in self._items)
        return self.accepts(prefix)

    def item_for(self, full_path: tuple[int, ...]) -> int:
        if not self.accepts(full_path):
            raise KeyError("Invalid or incomplete Semantic ID path")
        return self._items[full_path[:4]]

    def __len__(self) -> int:
        return len(self._items)

