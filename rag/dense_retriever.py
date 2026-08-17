"""Explicit-embedding BGE dense retrieval over the v2 child corpus."""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence

from core.embedding_service import EmbeddingServiceError, get_embedding_service

from .models import ChildChunk, RetrievalHit


class DenseRetriever:
    def __init__(
        self,
        children: Sequence[ChildChunk],
        embedding_service=None,
        collection=None,
        top_k: int = 10,
    ):
        self.children = list(children)
        self.by_id: Dict[str, ChildChunk] = {child.child_id: child for child in self.children}
        self.embedding_service = embedding_service or get_embedding_service()
        self.collection = collection
        self.top_k = top_k
        self._embeddings: Optional[List[List[float]]] = None
        self._query_embeddings: Dict[str, List[float]] = {}

    def set_embeddings(self, embeddings: Sequence[Sequence[float]]) -> None:
        if len(embeddings) != len(self.children):
            raise ValueError("embeddings and children must have the same length")
        self._embeddings = [[float(value) for value in row] for row in embeddings]

    def precompute_query_embeddings(self, queries: Sequence[str]) -> None:
        pending = [str(query) for query in queries if str(query) not in self._query_embeddings]
        if not pending:
            return
        vectors = self.embedding_service.encode_batch(pending)
        self._query_embeddings.update({query: vector for query, vector in zip(pending, vectors)})

    def search(self, query: str, top_k: Optional[int] = None, query_source: str = "original") -> List[RetrievalHit]:
        return self.search_many([query], top_k=top_k, query_sources=[query_source])[0]

    def search_many(
        self,
        queries: Sequence[str],
        top_k: Optional[int] = None,
        query_sources: Optional[Sequence[str]] = None,
    ) -> List[List[RetrievalHit]]:
        queries = list(queries)
        if not queries:
            return []
        top_k = top_k or self.top_k
        query_sources = list(query_sources or ["original"] * len(queries))
        if self.collection is not None:
            self.precompute_query_embeddings(queries)
            vectors = [self._query_embeddings[query] for query in queries]
            result = self.collection.query(
                query_embeddings=vectors,
                n_results=min(top_k, max(1, len(self.children))),
                include=["documents", "metadatas", "distances"],
            )
            return [self._from_chroma(result, index, query_sources[index]) for index in range(len(queries))]

        if self._embeddings is None:
            # Direct/in-memory mode is useful for local evaluation and small
            # corpora. Encode the document batch once and reuse it; production
            # rebuilds persist the same vectors in the explicit Chroma v2
            # collection instead.
            self._embeddings = self.embedding_service.encode_batch([child.content for child in self.children])
        self.precompute_query_embeddings(queries)
        query_vectors = [self._query_embeddings[query] for query in queries]
        return [
            self._search_vectors(vector, top_k, query_sources[index])
            for index, vector in enumerate(query_vectors)
        ]

    def _search_vectors(self, query_vector: Sequence[float], top_k: int, query_source: str) -> List[RetrievalHit]:
        scored = []
        for child, vector in zip(self.children, self._embeddings or []):
            score = _dot(query_vector, vector)
            scored.append((score, child))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalHit(
                child_id=child.child_id,
                parent_id=child.parent_id,
                content=child.content,
                source=child.source,
                metadata={**child.metadata, "dense_score": float(score)},
                retriever="dense",
                rank=index + 1,
                score=float(score),
                query_source=query_source,
            )
            for index, (score, child) in enumerate(scored[:top_k])
        ]

    def _from_chroma(self, result, query_index: int, query_source: str) -> List[RetrievalHit]:
        documents = (result.get("documents") or [[]])[query_index] or []
        metadatas = (result.get("metadatas") or [[]])[query_index] or []
        distances = (result.get("distances") or [[]])[query_index] or []
        hits: List[RetrievalHit] = []
        for index, content in enumerate(documents):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            if isinstance(metadata.get("section_path"), str):
                try:
                    metadata["section_path"] = json.loads(metadata["section_path"])
                except json.JSONDecodeError:
                    pass
            child_id = str(metadata.get("child_id", ""))
            child = self.by_id.get(child_id)
            if child is None:
                continue
            distance = float(distances[index]) if index < len(distances) else 0.0
            score = 1.0 - distance
            hits.append(RetrievalHit(
                child_id=child_id,
                parent_id=str(metadata.get("parent_id", child.parent_id)),
                content=str(content),
                source=str(metadata.get("source", child.source)),
                metadata={**child.metadata, **metadata, "dense_score": score},
                retriever="dense",
                rank=len(hits) + 1,
                score=score,
                query_source=query_source,
            ))
        return hits


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(float(a) * float(b) for a, b in zip(left, right))
