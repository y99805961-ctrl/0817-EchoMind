from rag.bm25_retriever import BM25Retriever, tokenize
from rag.models import ChildChunk


def child(child_id, content):
    return ChildChunk("doc", "doc#p", child_id, "Title", "Section", ["Title", "Section"], "doc.md", 0, content)


def test_protected_tokens_are_not_split():
    tokens = tokenize("支付失败 AUTH_4011，订单 ORD1234 request_id=abc")
    assert "auth_4011" in tokens
    assert "ord1234" in tokens
    assert "request_id" in tokens


def test_exact_error_code_wins_sparse_retrieval():
    retriever = BM25Retriever([
        child("c1", "登录失败时查看 AUTH_4011。"),
        child("c2", "普通登录问题排查。"),
    ])
    hits = retriever.search("AUTH_4011")
    assert hits and hits[0].child_id == "c1"
