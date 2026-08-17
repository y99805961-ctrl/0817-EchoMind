import pytest

from rag.bm25_retriever import BM25Retriever
from rag.config import load_rag_config
from rag.corpus import build_corpus
from rag.dense_retriever import DenseRetriever
from rag.parent_store import ParentStore
from rag.reranker import CrossEncoderReranker


pytestmark = pytest.mark.integration


def test_real_bge_dense_hits_refund_child():
    config = load_rag_config()
    parents, children, _ = build_corpus(config)
    service = pytest.importorskip("sentence_transformers")
    import core.embedding_service as embedding_module
    encoder = embedding_module.EmbeddingService(device="cpu")
    retriever = DenseRetriever(children, embedding_service=encoder, top_k=10)
    hits = retriever.search("退款到账")
    assert hits
    assert any("refund_return#" in hit.parent_id for hit in hits)


def test_real_bm25_protected_codes():
    config = load_rag_config()
    _, children, _ = build_corpus(config)
    hits = BM25Retriever(children).search("AUTH_4011")
    assert hits
    assert any("account_security#" in hit.parent_id for hit in hits[:3])


def test_real_cross_encoder_prefers_relevant_child():
    pytest.importorskip("sentence_transformers")
    reranker = CrossEncoderReranker(device="cpu")
    from rag.models import FusedHit
    hits = [FusedHit("irrelevant", "p1", "account settings", "a.md"), FusedHit("relevant", "p2", "refund timing for bank cards", "b.md")]
    ranked, status = reranker.rerank("refund successful but bank card has not received it", hits, 2)
    assert status == "ok"
    assert ranked[0].child_id == "relevant"
