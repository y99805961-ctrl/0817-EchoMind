import asyncio

from rag.query_rewriter import QueryRewriter


def test_rewrite_always_keeps_original_and_cleans_duplicates():
    rewriter = QueryRewriter(rewrite_fn=lambda query, count: [query, "银行卡退款到账需要多久", "银行卡退款到账需要多久", ""])
    queries = asyncio.run(rewriter.rewrite("退款成功为什么银行卡还没到账"))
    assert queries[0] == "退款成功为什么银行卡还没到账"
    assert queries.count("银行卡退款到账需要多久") == 1


def test_rewrite_failure_falls_back_to_original():
    def broken(query, count):
        raise RuntimeError("offline")
    assert asyncio.run(QueryRewriter(rewrite_fn=broken).rewrite("PAY_5001")) == ["PAY_5001"]
