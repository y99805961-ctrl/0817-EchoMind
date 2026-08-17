import asyncio

from rag.config import RAGConfig
from rag.models import ChildChunk, ParentChunk
from rag.parent_store import ParentStore
from rag.pipeline import RAGPipeline
from rag.reranker import CrossEncoderReranker
from rag.bm25_retriever import BM25Retriever


def test_pipeline_can_fallback_to_bm25_when_dense_is_unavailable():
    parent = ParentChunk("doc", "doc#p", "Title", "Error", ["Title", "Error"], "doc.md", "AUTH_4011 requires verification.")
    child = ChildChunk("doc", "doc#p", "doc#p#c01", "Title", "Error", ["Title", "Error"], "doc.md", 0, "AUTH_4011 requires verification.")
    config = RAGConfig()
    config.rewrite.enabled = False
    config.reranker.enabled = False
    pipeline = RAGPipeline(config, dense_retriever=None, bm25_retriever=BM25Retriever([child]), parent_store=ParentStore([parent]), reranker=CrossEncoderReranker())
    result = asyncio.run(pipeline.retrieve("AUTH_4011"))
    assert result.context_text
    assert result.reranker_status == "disabled"
    assert not result.fallbacks or "dense_failed_bm25_only" in result.fallbacks
