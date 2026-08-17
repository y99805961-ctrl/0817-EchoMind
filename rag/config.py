"""Configuration for the RAG v2 baseline, with environment overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - requirements include PyYAML in normal installs
    yaml = None


@dataclass
class CorpusConfig:
    source_dir: str = "data/demo_docs/customer_kb_v2"
    index_dir: str = "data/rag_index"
    children_file: str = "children.jsonl"
    parents_file: str = "parents.jsonl"
    manifest_file: str = "manifest.json"


@dataclass
class ChunkingConfig:
    strategy: str = "parent_child"
    target_child_chars: int = 400
    child_overlap_chars: int = 60
    max_parent_chars: int = 1500
    min_parent_chars: int = 80


@dataclass
class DenseConfig:
    model: str = "BAAI/bge-m3"
    top_k_per_query: int = 10
    collection: str = "knowledge_base_v2_children"
    device: str = "auto"
    batch_size: int = 4


@dataclass
class BM25Config:
    top_k_per_query: int = 10
    tokenizer: str = "jieba+protected_tokens"
    k1: float = 1.5
    b: float = 0.75


@dataclass
class RewriteConfig:
    enabled: bool = True
    count: int = 3
    max_concurrency: int = 10


@dataclass
class RRFConfig:
    k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    original_query_weight: float = 1.0
    rewrite_query_weight: float = 1.0


@dataclass
class RerankerConfig:
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    candidate_k: int = 25
    final_child_k: int = 8
    device: str = "auto"


@dataclass
class ContextConfig:
    max_parents: int = 3
    max_context_chars: int = 5000


@dataclass
class RAGConfig:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    rrf: RRFConfig = field(default_factory=RRFConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    def resolve_paths(self, root: Optional[Path] = None) -> "RAGConfig":
        root = root or project_root()
        self.corpus.source_dir = str(_resolve(root, self.corpus.source_dir))
        self.corpus.index_dir = str(_resolve(root, self.corpus.index_dir))
        return self

    @property
    def index_path(self) -> Path:
        return Path(self.corpus.index_dir)

    @property
    def parents_path(self) -> Path:
        return self.index_path / self.corpus.parents_file

    @property
    def children_path(self) -> Path:
        return self.index_path / self.corpus.children_file

    @property
    def manifest_path(self) -> Path:
        return self.index_path / self.corpus.manifest_file

    def apply_environment(self) -> "RAGConfig":
        mapping = {
            "RAG_SOURCE_DIR": (self.corpus, "source_dir", str),
            "RAG_INDEX_DIR": (self.corpus, "index_dir", str),
            "RAG_CHILD_SIZE": (self.chunking, "target_child_chars", int),
            "RAG_CHILD_OVERLAP": (self.chunking, "child_overlap_chars", int),
            "RAG_MAX_PARENT_CHARS": (self.chunking, "max_parent_chars", int),
            "RAG_DENSE_TOP_K": (self.dense, "top_k_per_query", int),
            "RAG_BM25_TOP_K": (self.bm25, "top_k_per_query", int),
            "RAG_REWRITE_MAX_CONCURRENCY": (self.rewrite, "max_concurrency", int),
            "RAG_COLLECTION": (self.dense, "collection", str),
            "RAG_RERANKER_ENABLED": (self.reranker, "enabled", _as_bool),
            "RAG_RERANKER_DEVICE": (self.reranker, "device", str),
            "RAG_MAX_PARENTS": (self.context, "max_parents", int),
            "RAG_MAX_CONTEXT_CHARS": (self.context, "max_context_chars", int),
        }
        for env_name, (target, attr, converter) in mapping.items():
            value = os.getenv(env_name)
            if value is not None and value != "":
                setattr(target, attr, converter(value))
        return self


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def load_rag_config(path: Optional[str | Path] = None, root: Optional[Path] = None) -> RAGConfig:
    root = root or project_root()
    path = Path(path) if path else root / "config" / "rag.yaml"
    raw: Dict[str, Any] = {}
    if path.exists():
        if yaml is None:
            raise RuntimeError("PyYAML is required to load config/rag.yaml")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("RAG config must be a mapping")
        raw = loaded

    def section(name: str, cls):
        values = raw.get(name, {})
        return cls(**values) if isinstance(values, dict) else cls()

    config = RAGConfig(
        corpus=section("corpus", CorpusConfig),
        chunking=section("chunking", ChunkingConfig),
        dense=section("dense", DenseConfig),
        bm25=section("bm25", BM25Config),
        rewrite=section("rewrite", RewriteConfig),
        rrf=section("rrf", RRFConfig),
        reranker=section("reranker", RerankerConfig),
        context=section("context", ContextConfig),
    )
    return config.apply_environment().resolve_paths(root)
