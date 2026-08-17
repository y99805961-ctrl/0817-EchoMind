"""Build and persist the deterministic Parent and Child JSONL corpus."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .chunking import ParentChildChunker
from .config import RAGConfig
from .markdown_parser import ParsedDocument, parse_directory
from .models import ChildChunk, ParentChunk


def source_hash(source_dir: str | Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(source_dir).glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_corpus(config: RAGConfig) -> Tuple[List[ParentChunk], List[ChildChunk], Dict[str, object]]:
    documents: List[ParsedDocument] = parse_directory(
        config.corpus.source_dir,
        max_parent_chars=config.chunking.max_parent_chars,
        min_parent_chars=config.chunking.min_parent_chars,
    )
    parents = [parent for document in documents for parent in document.parents]
    children = ParentChildChunker(
        child_size=config.chunking.target_child_chars,
        overlap=config.chunking.child_overlap_chars,
    ).split_parents(parents)
    manifest: Dict[str, object] = {
        "index_version": "rag-v2-parent-child-1",
        "source_hash": source_hash(config.corpus.source_dir),
        "source_documents": len(documents),
        "parents": len(parents),
        "children": len(children),
        "chunk_config": {
            "strategy": config.chunking.strategy,
            "target_child_chars": config.chunking.target_child_chars,
            "child_overlap_chars": config.chunking.child_overlap_chars,
            "max_parent_chars": config.chunking.max_parent_chars,
        },
        "embedding_model": config.dense.model,
        "embedding_dim": None,
        "collection_name": config.dense.collection,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return parents, children, manifest


def write_jsonl(path: str | Path, rows: Sequence[object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if hasattr(row, "to_dict"):
                value = row.to_dict()
            else:
                value = row
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, object]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
