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


def test_rewrite_many_uses_bounded_concurrency():
    state = {"active": 0, "max_active": 0}

    async def rewrite_fn(query, count):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return [f"{query} 改写"]

    rewriter = QueryRewriter(rewrite_fn=rewrite_fn, max_concurrency=3)
    rows = asyncio.run(rewriter.rewrite_many([f"q{i}" for i in range(8)]))
    assert len(rows) == 8
    assert state["max_active"] == 3
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["latency_ms"] >= 0 for row in rows)
