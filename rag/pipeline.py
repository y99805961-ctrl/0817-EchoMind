"""Production RAG v2 orchestration: rewrite → hybrid → RRF → rerank → Parent context."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.embedding_service import EmbeddingServiceError, get_embedding_service

from .bm25_retriever import BM25Retriever
from .config import RAGConfig, load_rag_config
from .context_builder import pack_context
from .corpus import read_jsonl, source_hash
from .dense_retriever import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .models import (
    ChildChunk,
    FusedHit,
    ParentSelection,
    RAGResult,
    RAGTiming,
    RerankedHit,
    RetrievalHit,
)
from .parent_store import ParentStore
from .query_rewriter import QueryRewriter
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class IndexMissingError(RuntimeError):
    pass


class RAGPipeline:
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        *,
        rewriter: Optional[QueryRewriter] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        parent_store: Optional[ParentStore] = None,
        startup_fallbacks: Optional[Sequence[str]] = None,
    ):
        self.config = config or load_rag_config()
        self.rewriter = rewriter or QueryRewriter(count=self.config.rewrite.count)
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.reranker = reranker or CrossEncoderReranker(
            model_name=self.config.reranker.model,
            device=self.config.reranker.device,
        )
        self.parent_store = parent_store or ParentStore()
        self.startup_fallbacks = list(startup_fallbacks or [])
        if self.dense is not None:
            self.index_status = "ready"
        elif self.bm25 is not None:
            self.index_status = "ready_bm25_only"
        else:
            self.index_status = "missing"

    @classmethod
    def from_index(
        cls,
        config: Optional[RAGConfig] = None,
        *,
        embedding_service=None,
        chroma_client=None,
        rewrite_client=None,
        rewrite_model: str = "",
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> "RAGPipeline":
        config = config or load_rag_config()
        children_rows = read_jsonl(config.children_path)
        parents_rows = read_jsonl(config.parents_path)
        if not children_rows or not parents_rows or not config.manifest_path.exists():
            return cls(config, rewriter=QueryRewriter(rewrite_client, rewrite_model, config.rewrite.count), reranker=reranker)

        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        expected_hash = source_hash(config.corpus.source_dir)
        if manifest.get("source_hash") != expected_hash:
            logger.warning("RAG index source hash mismatch; rebuild is required")
            return cls(config, rewriter=QueryRewriter(rewrite_client, rewrite_model, config.rewrite.count), reranker=reranker)
        if manifest.get("chunk_config", {}).get("target_child_chars") != config.chunking.target_child_chars:
            logger.warning("RAG index chunk config mismatch; rebuild is required")
            return cls(config, rewriter=QueryRewriter(rewrite_client, rewrite_model, config.rewrite.count), reranker=reranker)

        children = [ChildChunk.from_dict(row) for row in children_rows]
        from .models import ParentChunk
        parents = [ParentChunk.from_dict(row) for row in parents_rows]
        parent_store = ParentStore(parents)
        collection = None
        startup_fallbacks: List[str] = []
        try:
            if chroma_client is not None:
                collection = chroma_client.get_collection(config.dense.collection)
            else:
                import chromadb
                import os
                host = os.getenv("CHROMA_HOST", "").strip()
                if host:
                    client = chromadb.HttpClient(host=host, port=int(os.getenv("CHROMA_PORT", "8000")), settings=chromadb.Settings(anonymized_telemetry=False))
                    client.heartbeat()
                else:
                    client = chromadb.PersistentClient(path=str(config.index_path / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False))
                collection = client.get_collection(config.dense.collection)
        except Exception as exc:
            startup_fallbacks.append("chroma_endpoint_unavailable_local_fallback")
            logger.warning("Dense v2 collection unavailable at configured Chroma endpoint; trying local index: %s", type(exc).__name__)
            try:
                import chromadb
                local_client = chromadb.PersistentClient(path=str(config.index_path / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False))
                collection = local_client.get_collection(config.dense.collection)
            except Exception as local_exc:
                startup_fallbacks.append("dense_collection_unavailable_bm25_only")
                logger.warning("Local dense v2 collection unavailable: %s", type(local_exc).__name__)

        dense = None
        if collection is not None:
            dense = DenseRetriever(
                children,
                embedding_service=embedding_service or get_embedding_service(
                    model_name=config.dense.model,
                    device=config.dense.device,
                    batch_size=config.dense.batch_size,
                ),
                collection=collection,
                top_k=config.dense.top_k_per_query,
            )
        bm25 = BM25Retriever(
            children,
            top_k=config.bm25.top_k_per_query,
            k1=config.bm25.k1,
            b=config.bm25.b,
        )
        return cls(
            config,
            rewriter=QueryRewriter(rewrite_client, rewrite_model, config.rewrite.count),
            dense_retriever=dense,
            bm25_retriever=bm25,
            reranker=reranker,
            parent_store=parent_store,
            startup_fallbacks=startup_fallbacks,
        )

    async def retrieve(self, query: str, *, top_k: Optional[int] = None, rewrite: Optional[bool] = None) -> RAGResult:
        started = time.perf_counter()
        result = RAGResult(query=query, index_status=self.index_status, fallbacks=list(self.startup_fallbacks))
        if self.index_status == "missing":
            result.fallbacks.append("index_missing")
            result.timing.total_ms = _elapsed(started)
            return result

        rewrite_started = time.perf_counter()
        if rewrite is False or not self.config.rewrite.enabled:
            queries = [query]
        else:
            queries = await self.rewriter.rewrite(query, self.config.rewrite.count)
        result.rewritten_queries = queries
        rewrite_status = getattr(self.rewriter, "last_status", "")
        if rewrite_status == "failed":
            result.fallbacks.append("rewrite_failed_original_only")
        elif self.config.rewrite.enabled and self.config.rewrite.count > 0 and len(queries) == 1:
            result.fallbacks.append("rewrite_original_only")
        result.timing.rewrite_ms = _elapsed(rewrite_started)
        query_sources = ["original" if index == 0 else "rewrite" for index in range(len(queries))]

        dense_lists: List[List[RetrievalHit]] = []
        bm25_lists: List[List[RetrievalHit]] = []

        async def run_dense():
            stage_started = time.perf_counter()
            if self.dense is None:
                return [], _elapsed(stage_started), None
            try:
                rows = await asyncio.to_thread(
                    self.dense.search_many,
                    queries,
                    top_k or self.config.dense.top_k_per_query,
                    query_sources,
                )
                return rows, _elapsed(stage_started), None
            except Exception as exc:
                return [], _elapsed(stage_started), exc

        async def run_bm25():
            stage_started = time.perf_counter()
            if self.bm25 is None:
                return [], _elapsed(stage_started), None
            try:
                rows = await asyncio.to_thread(
                    self.bm25.search_many,
                    queries,
                    top_k or self.config.bm25.top_k_per_query,
                    query_sources,
                )
                return rows, _elapsed(stage_started), None
            except Exception as exc:
                return [], _elapsed(stage_started), exc

        dense_task = asyncio.create_task(run_dense())
        bm25_task = asyncio.create_task(run_bm25())
        dense_out, bm25_out = await asyncio.gather(dense_task, bm25_task)
        dense_lists, result.timing.dense_ms, dense_error = dense_out
        bm25_lists, result.timing.bm25_ms, bm25_error = bm25_out
        if dense_error is not None:
            result.fallbacks.append("dense_failed_bm25_only")
            logger.warning("Dense retrieval failed: %s", dense_error)
        if bm25_error is not None:
            result.fallbacks.append("bm25_failed_dense_only")
            logger.warning("BM25 retrieval failed: %s", bm25_error)

        result.dense_results = [hit for rows in dense_lists for hit in rows]
        result.bm25_results = [hit for rows in bm25_lists for hit in rows]
        if not result.dense_results and not result.bm25_results:
            result.fallbacks.append("all_retrievers_failed")
            result.timing.total_ms = _elapsed(started)
            return result

        rrf_started = time.perf_counter()
        result.rrf_hits = reciprocal_rank_fusion(
            [*dense_lists, *bm25_lists],
            k=self.config.rrf.k,
            dense_weight=self.config.rrf.dense_weight,
            bm25_weight=self.config.rrf.bm25_weight,
            original_query_weight=self.config.rrf.original_query_weight,
            rewrite_query_weight=self.config.rrf.rewrite_query_weight,
        )[: self.config.reranker.candidate_k]
        result.timing.rrf_ms = _elapsed(rrf_started)

        rerank_started = time.perf_counter()
        if self.config.reranker.enabled and self.reranker is not None:
            reranked, status = await asyncio.to_thread(
                self.reranker.rerank,
                query,
                result.rrf_hits,
                self.config.reranker.final_child_k,
            )
            result.reranked_hits = reranked
            result.reranker_status = status
            if status == "failed":
                result.fallbacks.append("reranker_failed_rrf_order")
        else:
            result.reranked_hits = [_rrf_as_reranked(hit, index) for index, hit in enumerate(result.rrf_hits[: self.config.reranker.final_child_k])]
            result.reranker_status = "disabled"
        result.timing.rerank_ms = _elapsed(rerank_started)

        expand_started = time.perf_counter()
        result.selected_parents = self._expand_parents(result.reranked_hits)
        result.timing.parent_expand_ms = _elapsed(expand_started)
        pack_started = time.perf_counter()
        result.context_text = pack_context(
            result.selected_parents,
            max_chars=self.config.context.max_context_chars,
        )
        result.timing.context_pack_ms = _elapsed(pack_started)
        result.retrieved_children = [
            RetrievalHit(
                child_id=hit.child_id,
                parent_id=hit.parent_id,
                content=hit.content,
                source=hit.source,
                metadata=dict(hit.metadata),
                retriever="reranker",
                rank=hit.reranker_rank,
                score=hit.reranker_score,
                query_source="original",
            )
            for hit in result.reranked_hits
        ]
        result.timing.cold_start_ms = float(getattr(self.reranker, "cold_start_ms", 0.0))
        result.timing.total_ms = _elapsed(started)
        return result

    def _expand_parents(self, hits: Sequence[RerankedHit]) -> List[ParentSelection]:
        grouped: Dict[str, ParentSelection] = {}
        for hit in hits:
            parent = self.parent_store.get(hit.parent_id)
            if parent is None:
                continue
            score = float(hit.reranker_score)
            # RerankedHit carries RRF score as a fallback score. The first
            # available child is the max score for its Parent.
            if hit.parent_id not in grouped:
                grouped[hit.parent_id] = ParentSelection(parent, score, 1, [hit.child_id])
            else:
                selection = grouped[hit.parent_id]
                selection.parent_score = max(selection.parent_score, score)
                selection.child_hit_count += 1
                selection.child_ids.append(hit.child_id)
        return sorted(grouped.values(), key=lambda item: item.parent_score, reverse=True)[: self.config.context.max_parents]

    async def search_handler(self, params: Dict[str, Any], context: Any = None) -> List[Dict[str, Any]]:
        result = await self.retrieve(str(params.get("query", "")), top_k=int(params.get("top_k", self.config.reranker.final_child_k)))
        return result.to_dict().get("retrieved_children", [])


def _rrf_as_reranked(hit: FusedHit, index: int) -> RerankedHit:
    return RerankedHit(
        child_id=hit.child_id,
        parent_id=hit.parent_id,
        content=hit.content,
        source=hit.source,
        metadata=dict(hit.metadata),
        reranker_score=hit.rrf_score,
        reranker_rank=index + 1,
        rrf_score=hit.rrf_score,
        matched_queries=list(hit.matched_queries),
    )


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
