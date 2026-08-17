from rag.fusion import reciprocal_rank_fusion
from rag.models import RetrievalHit


def hit(child_id, retriever, rank, query_source="original"):
    return RetrievalHit(child_id, "p", child_id, "doc.md", retriever=retriever, rank=rank, score=1.0, query_source=query_source)


def test_rrf_aggregates_same_child_across_queries_and_retrievers():
    result = reciprocal_rank_fusion([[hit("c1", "dense", 2), hit("c2", "dense", 1)], [hit("c1", "bm25", 1, "rewrite")]])
    assert [item.child_id for item in result].count("c1") == 1
    assert result[0].child_id == "c1"
    assert result[0].dense_ranks["original"] == 2
    assert result[0].bm25_ranks["rewrite"] == 1
