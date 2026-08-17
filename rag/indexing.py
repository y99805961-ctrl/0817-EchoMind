"""Index rebuild and stats CLI for the v2 Parent–Child corpus."""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from core.embedding_service import EmbeddingServiceError, get_embedding_service

from .config import RAGConfig, load_rag_config
from .corpus import build_corpus, write_jsonl

logger = logging.getLogger(__name__)


def _connect_chroma(config: RAGConfig):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required to write the dense v2 collection") from exc
    host = __import__("os").getenv("CHROMA_HOST", "").strip()
    port = int(__import__("os").getenv("CHROMA_PORT", "8000"))
    if host:
        client = chromadb.HttpClient(host=host, port=port, settings=chromadb.Settings(anonymized_telemetry=False))
    else:
        client = chromadb.PersistentClient(path=str(Path(config.corpus.index_dir) / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False))
    return client


def rebuild_index(config: Optional[RAGConfig] = None, *, embedding_service=None, chroma_client=None) -> Dict[str, object]:
    config = config or load_rag_config()
    parents, children, manifest = build_corpus(config)
    write_jsonl(config.parents_path, parents)
    write_jsonl(config.children_path, children)

    if not children:
        raise RuntimeError("The configured source directory produced no child chunks")

    service = embedding_service or get_embedding_service(
        model_name=config.dense.model,
        device=config.dense.device,
        batch_size=config.dense.batch_size,
    )
    try:
        embeddings = service.encode_batch([child.content for child in children])
    except EmbeddingServiceError:
        raise
    if len(embeddings) != len(children):
        raise RuntimeError("Embedding service returned an unexpected number of vectors")
    manifest["embedding_dim"] = len(embeddings[0]) if embeddings else None

    client = chroma_client or _connect_chroma(config)
    collection = client.get_or_create_collection(
        name=config.dense.collection,
        metadata={"description": "EchoMind Parent-Child RAG v2 children", "index_version": "rag-v2"},
    )
    # Rebuild is explicit and deterministic. Remove only the v2 collection;
    # the legacy knowledge_base collection is intentionally untouched.
    existing = collection.get()
    existing_ids = list(existing.get("ids", [])) if isinstance(existing, dict) else []
    if existing_ids:
        collection.delete(ids=existing_ids)
    collection.add(
        ids=[child.child_id for child in children],
        embeddings=embeddings,
        documents=[child.content for child in children],
        metadatas=[_chroma_metadata(child.metadata) for child in children],
    )
    manifest["dense_indexed"] = True
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _chroma_metadata(metadata: Dict[str, object]) -> Dict[str, object]:
    """Chroma metadata is scalar-only; JSONL retains the richer array form."""
    result: Dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value if value is not None else ""
        elif isinstance(value, list):
            result[key] = json.dumps(value, ensure_ascii=False)
        else:
            result[key] = str(value)
    return result


def index_stats(config: Optional[RAGConfig] = None) -> Dict[str, object]:
    config = config or load_rag_config()
    parents, children = [], []
    from .corpus import read_jsonl
    from .models import ChildChunk, ParentChunk
    parents = [ParentChunk.from_dict(row) for row in read_jsonl(config.parents_path)]
    children = [ChildChunk.from_dict(row) for row in read_jsonl(config.children_path)]
    if not parents and not children and Path(config.corpus.source_dir).exists():
        # Stats may inspect the source corpus without rebuilding or embedding it.
        from .corpus import build_corpus
        source_parents, source_children, _ = build_corpus(config)
        parents, children = source_parents, source_children
    parent_lengths = [len(item.content) for item in parents]
    child_lengths = [len(item.content) for item in children]
    manifest = {}
    if config.manifest_path.exists():
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    result = {
        "source_documents": len({item.doc_id for item in parents}),
        "parents": len(parents),
        "children": len(children),
        "avg_parent_chars": _mean(parent_lengths),
        "p50_parent_chars": _percentile(parent_lengths, 50),
        "p95_parent_chars": _percentile(parent_lengths, 95),
        "max_parent_chars": max(parent_lengths, default=0),
        "avg_child_chars": _mean(child_lengths),
        "p50_child_chars": _percentile(child_lengths, 50),
        "p95_child_chars": _percentile(child_lengths, 95),
        "embedding_model": manifest.get("embedding_model", config.dense.model),
        "embedding_dim": manifest.get("embedding_dim"),
        "collection_name": manifest.get("collection_name", config.dense.collection),
        "index_status": "ready" if parents and children and manifest else "missing",
    }
    return result


def _mean(values: List[int]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def _percentile(values: List[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EchoMind RAG v2 index utility")
    parser.add_argument("command", choices=["rebuild", "stats"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    config = load_rag_config(args.config)
    if args.command == "rebuild":
        print(json.dumps(rebuild_index(config), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(index_stats(config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
