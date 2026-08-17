"""Clean Parent context formatting; retrieval evidence stays in debug only."""
from __future__ import annotations

from typing import Iterable, List

from .models import ParentSelection


def pack_context(selections: Iterable[ParentSelection], max_chars: int = 5000) -> str:
    prefix = (
        "[Knowledge Context]\n"
        "Grounding: 对政策、时效、费用、权限和业务规则，只依据以下知识内容；信息不足时明确说明当前知识库无法确认。"
    )
    parts: List[str] = []
    used = len(prefix)
    for index, selection in enumerate(selections, start=1):
        parent = selection.parent
        block = (
            f"[Knowledge {index}]\n"
            f"Source: {parent.source}\n"
            f"Section: {parent.section}\n"
            f"Content:\n{parent.content.strip()}"
        )
        separator = "\n\n"
        remaining = max_chars - used - len(separator)
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        if block:
            parts.append(block)
            used += len(separator) + len(block)
        if used >= max_chars:
            break
    return prefix + ("\n\n" + "\n\n".join(parts) if parts else "")
