"""Recursive child splitting constrained to one Parent section at a time."""
from __future__ import annotations

import re
from typing import Iterable, List

from .models import ChildChunk, ParentChunk


class ParentChildChunker:
    def __init__(self, child_size: int = 400, overlap: int = 60):
        if child_size <= 0:
            raise ValueError("child_size must be positive")
        if overlap < 0 or overlap >= child_size:
            raise ValueError("overlap must be non-negative and smaller than child_size")
        self.child_size = child_size
        self.overlap = overlap

    def split_parent(self, parent: ParentChunk) -> List[ChildChunk]:
        pieces = self._atomic_pieces(parent.content)
        if not pieces:
            return []
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= self.child_size:
                current = f"{current}\n{piece}"
            else:
                chunks.append(current.strip())
                overlap_text = self._tail(current)
                current = f"{overlap_text}\n{piece}".strip() if overlap_text else piece
                # If the overlap makes the chunk too large, preserve the
                # current semantic unit and let the next pass trim only by
                # character fallback.
                if len(current) > self.child_size and len(piece) <= self.child_size:
                    current = current[-self.child_size:]
        if current.strip():
            chunks.append(current.strip())

        return [
            ChildChunk(
                doc_id=parent.doc_id,
                parent_id=parent.parent_id,
                child_id=f"{parent.parent_id}#c{index + 1:02d}",
                title=parent.title,
                section=parent.section,
                section_path=list(parent.section_path),
                source=parent.source,
                child_index=index,
                content=chunk,
                metadata={
                    "doc_id": parent.doc_id,
                    "parent_id": parent.parent_id,
                    "child_id": f"{parent.parent_id}#c{index + 1:02d}",
                    "title": parent.title,
                    "section": parent.section,
                    "section_path": list(parent.section_path),
                    "source": parent.source,
                    "child_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
            if chunk
        ]

    def split_parents(self, parents: Iterable[ParentChunk]) -> List[ChildChunk]:
        result: List[ChildChunk] = []
        for parent in parents:
            result.extend(self.split_parent(parent))
        return result

    def _atomic_pieces(self, text: str) -> List[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        pieces: List[str] = []
        for paragraph in paragraphs or [text.strip()]:
            if len(paragraph) <= self.child_size:
                pieces.append(paragraph)
                continue
            sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*", paragraph) if part.strip()]
            for sentence in sentences or [paragraph]:
                if len(sentence) <= self.child_size:
                    pieces.append(sentence)
                else:
                    pieces.extend(
                        sentence[index:index + self.child_size]
                        for index in range(0, len(sentence), self.child_size)
                    )
        return pieces

    def _tail(self, text: str) -> str:
        if self.overlap <= 0:
            return ""
        tail = text[-self.overlap:]
        # Prefer starting at a word/sentence boundary when possible.
        boundary = max(tail.rfind("。"), tail.rfind("！"), tail.rfind("？"), tail.rfind("\n"))
        return tail[boundary + 1:] if boundary >= 0 and boundary + 1 < len(tail) else tail
