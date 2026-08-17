"""Typed internal models shared by the RAG pipeline and evaluation code."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParentChunk:
    doc_id: str
    parent_id: str
    title: str
    section: str
    section_path: List[str]
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParentChunk":
        return cls(
            doc_id=str(data.get("doc_id", "")),
            parent_id=str(data["parent_id"]),
            title=str(data.get("title", "")),
            section=str(data.get("section", "")),
            section_path=list(data.get("section_path", [])),
            source=str(data.get("source", "")),
            content=str(data.get("content", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ChildChunk:
    doc_id: str
    parent_id: str
    child_id: str
    title: str
    section: str
    section_path: List[str]
    source: str
    child_index: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChildChunk":
        return cls(
            doc_id=str(data.get("doc_id", "")),
            parent_id=str(data["parent_id"]),
            child_id=str(data["child_id"]),
            title=str(data.get("title", "")),
            section=str(data.get("section", "")),
            section_path=list(data.get("section_path", [])),
            source=str(data.get("source", "")),
            child_index=int(data.get("child_index", 0)),
            content=str(data.get("content", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RetrievalHit:
    child_id: str
    parent_id: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retriever: str = ""
    rank: int = 0
    score: float = 0.0
    query_source: str = "original"

    @property
    def dense_score(self) -> Optional[float]:
        value = self.metadata.get("dense_score")
        return float(value) if value is not None else None

    @property
    def bm25_score(self) -> Optional[float]:
        value = self.metadata.get("bm25_score")
        return float(value) if value is not None else None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data


@dataclass
class FusedHit:
    child_id: str
    parent_id: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    rrf_score: float = 0.0
    matched_queries: List[str] = field(default_factory=list)
    dense_ranks: Dict[str, int] = field(default_factory=dict)
    bm25_ranks: Dict[str, int] = field(default_factory=dict)
    original_query_hits: List[str] = field(default_factory=list)
    rewrite_query_hits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data


@dataclass
class RerankedHit:
    child_id: str
    parent_id: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    reranker_score: float = 0.0
    reranker_rank: int = 0
    rrf_score: float = 0.0
    matched_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data


@dataclass
class ParentSelection:
    parent: ParentChunk
    parent_score: float
    child_hit_count: int
    child_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent.parent_id,
            "doc_id": self.parent.doc_id,
            "title": self.parent.title,
            "section": self.parent.section,
            "section_path": list(self.parent.section_path),
            "source": self.parent.source,
            "content": self.parent.content,
            "parent_score": self.parent_score,
            "child_hit_count": self.child_hit_count,
            "child_ids": list(self.child_ids),
        }


@dataclass
class RAGTiming:
    rewrite_ms: float = 0.0
    dense_ms: float = 0.0
    bm25_ms: float = 0.0
    rrf_ms: float = 0.0
    rerank_ms: float = 0.0
    parent_expand_ms: float = 0.0
    context_pack_ms: float = 0.0
    total_ms: float = 0.0
    cold_start_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {key: round(float(value), 3) for key, value in asdict(self).items()}


@dataclass
class RAGResult:
    query: str
    rewritten_queries: List[str] = field(default_factory=list)
    selected_parents: List[ParentSelection] = field(default_factory=list)
    context_text: str = ""
    retrieved_children: List[RetrievalHit] = field(default_factory=list)
    dense_results: List[RetrievalHit] = field(default_factory=list)
    bm25_results: List[RetrievalHit] = field(default_factory=list)
    rrf_hits: List[FusedHit] = field(default_factory=list)
    reranked_hits: List[RerankedHit] = field(default_factory=list)
    timing: RAGTiming = field(default_factory=RAGTiming)
    fallbacks: List[str] = field(default_factory=list)
    reranker_status: str = "disabled"
    index_status: str = "ready"

    @property
    def has_context(self) -> bool:
        return bool(self.context_text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "rewritten_queries": list(self.rewritten_queries),
            "selected_parents": [item.to_dict() for item in self.selected_parents],
            "context_text": self.context_text,
            "retrieved_children": [item.to_dict() for item in self.retrieved_children],
            "dense_results": [item.to_dict() for item in self.dense_results],
            "bm25_results": [item.to_dict() for item in self.bm25_results],
            "rrf_hits": [item.to_dict() for item in self.rrf_hits],
            "reranked_hits": [item.to_dict() for item in self.reranked_hits],
            "timing": self.timing.to_dict(),
            "fallbacks": list(self.fallbacks),
            "reranker_status": self.reranker_status,
            "index_status": self.index_status,
        }
