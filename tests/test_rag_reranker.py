from rag.models import FusedHit
from rag.reranker import CrossEncoderReranker


def test_reranker_uses_original_query_and_returns_scores():
    def scorer(query, hits):
        assert query == "refund"
        return [1.0 if "refund" in hit.content else 0.1 for hit in hits]
    hits = [FusedHit("c1", "p1", "refund timing", "doc.md"), FusedHit("c2", "p2", "account", "doc.md")]
    result, status = CrossEncoderReranker(scorer=scorer).rerank("refund", hits, 2)
    assert status == "ok"
    assert result[0].child_id == "c1"
