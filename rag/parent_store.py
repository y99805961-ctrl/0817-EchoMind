"""Small in-memory Parent Store backed by stable JSONL records."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

from .corpus import read_jsonl
from .models import ParentChunk


class ParentStore:
    def __init__(self, parents: Iterable[ParentChunk] = ()):
        self._parents: Dict[str, ParentChunk] = {parent.parent_id: parent for parent in parents}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ParentStore":
        return cls(ParentChunk.from_dict(row) for row in read_jsonl(path))

    def get(self, parent_id: str) -> Optional[ParentChunk]:
        return self._parents.get(parent_id)

    def require(self, parent_id: str) -> ParentChunk:
        parent = self.get(parent_id)
        if parent is None:
            raise KeyError(f"Parent not found: {parent_id}")
        return parent

    def __len__(self) -> int:
        return len(self._parents)

    def values(self):
        return self._parents.values()
