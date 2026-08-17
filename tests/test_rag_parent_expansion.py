import asyncio

from rag.config import RAGConfig
from rag.models import ChildChunk, ParentChunk
from rag.parent_store import ParentStore
from rag.pipeline import RAGPipeline
from rag.reranker import CrossEncoderReranker
from rag.dense_retriever import DenseRetriever
from rag.bm25_retriever import BM25Retriever


class FakeEmbedding:
    def encode_batch(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_child_hits_expand_and_deduplicate_parents():
    parents = [ParentChunk("doc", "doc#p1", "Title", "One", ["Title", "One"], "doc.md", "parent one"), ParentChunk("doc", "doc#p2", "Title", "Two", ["Title", "Two"], "doc.md", "parent two")]
    children = [ChildChunk("doc", "doc#p1", "doc#p1#c01", "Title", "One", ["Title", "One"], "doc.md", 0, "refund"), ChildChunk("doc", "doc#p1", "doc#p1#c02", "Title", "One", ["Title", "One"], "doc.md", 1, "refund timing"), ChildChunk("doc", "doc#p2", "doc#p2#c01", "Title", "Two", ["Title", "Two"], "doc.md", 0, "account")]
    dense = DenseRetriever(children, embedding_service=FakeEmbedding())
    dense.set_embeddings([[1, 0], [1, 0], [1, 0]])
    config = RAGConfig()
    config.rewrite.enabled = False
    config.reranker.enabled = True
    config.reranker.final_child_k = 3
    config.context.max_parents = 3
    pipeline = RAGPipeline(config, dense_retriever=dense, bm25_retriever=BM25Retriever(children), parent_store=ParentStore(parents), reranker=CrossEncoderReranker(scorer=lambda q, hits: [3.0, 2.0, 1.0]))
    result = asyncio.run(pipeline.retrieve("refund"))
    assert len(result.selected_parents) == 2
    assert result.selected_parents[0].parent.parent_id == "doc#p1"
    assert result.selected_parents[0].child_hit_count == 2
