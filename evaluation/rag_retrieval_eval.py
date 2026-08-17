"""Gold retrieval evaluation and ablation runner for the production RAG pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

from rag.config import load_rag_config
from rag.metrics import latency_summary, summarize_cases
from rag.pipeline import RAGPipeline


DEFAULT_BENCHMARK = Path(__file__).parents[1] / "data" / "eval" / "rag" / "rag_benchmark_60.json"


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK) -> List[Dict[str, Any]]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(cases) != 60:
        raise ValueError(f"RAG benchmark must contain 60 cases, got {len(cases)}")
    return cases


async def evaluate_retrieval(
    pipeline: RAGPipeline,
    cases: Sequence[Dict[str, Any]],
    *,
    method: str = "production",
) -> Dict[str, Any]:
    """Run all methods against the same benchmark without changing its gold labels."""
    original_dense, original_bm25 = pipeline.dense, pipeline.bm25
    original_rewrite, original_rerank = pipeline.config.rewrite.enabled, pipeline.config.reranker.enabled
    if method == "dense":
        pipeline.bm25 = None
        pipeline.config.rewrite.enabled = False
        pipeline.config.reranker.enabled = False
    elif method == "bm25":
        pipeline.dense = None
        pipeline.config.rewrite.enabled = False
        pipeline.config.reranker.enabled = False
    elif method == "hybrid":
        pipeline.config.rewrite.enabled = False
        pipeline.config.reranker.enabled = False
    elif method == "hybrid_rewrite":
        pipeline.config.reranker.enabled = False

    result_rows: List[Dict[str, Any]] = []
    try:
        if pipeline.dense is not None:
            # BGE-M3 query vectors are batch encoded once per benchmark and
            # reused across ablations; this is the same batch path used for
            # multi-query production retrieval.
            await asyncio.to_thread(
                pipeline.dense.precompute_query_embeddings,
                [case["query"] for case in cases],
            )
        for case in cases:
            result = await pipeline.retrieve(case["query"], rewrite=method not in {"dense", "bm25", "hybrid"})
            result_rows.append({
                "id": case["id"],
                "child_ids": [hit.child_id for hit in result.retrieved_children] or [hit.child_id for hit in result.reranked_hits],
                "parent_ids": [selection.parent.parent_id for selection in result.selected_parents],
                "rewritten_queries": result.rewritten_queries,
                "rewrite_status": result.rewrite_status,
                "rewrite_error": result.rewrite_error,
                "rewrite_latency_ms": result.timing.rewrite_ms,
                "rewrite_fallback": any("rewrite" in item for item in result.fallbacks),
                "timing": result.timing.to_dict(),
                "fallbacks": result.fallbacks,
            })
    finally:
        pipeline.dense, pipeline.bm25 = original_dense, original_bm25
        pipeline.config.rewrite.enabled, pipeline.config.reranker.enabled = original_rewrite, original_rerank

    metrics = summarize_cases(cases, result_rows)
    metrics["latency"] = {
        key: latency_summary([row["timing"].get(key, 0.0) for row in result_rows])
        for key in ("rewrite_ms", "dense_ms", "bm25_ms", "rrf_ms", "rerank_ms", "total_ms")
    }
    rewrite_rows = [row for row in result_rows if row.get("rewrite_status") in {"ok", "failed", "empty"}]
    metrics["rewrite"] = {
        "cases": len(rewrite_rows),
        "successes": sum(row.get("rewrite_status") == "ok" for row in rewrite_rows),
        "multi_query_cases": sum(len(row.get("rewritten_queries", [])) > 1 for row in rewrite_rows),
        "fallback_cases": sum(bool(row.get("rewrite_fallback")) for row in rewrite_rows),
        "success_rate": round(sum(row.get("rewrite_status") == "ok" for row in rewrite_rows) / len(rewrite_rows), 6) if rewrite_rows else 0.0,
    }
    return {"method": method, "metrics": metrics, "cases": result_rows}


async def run_retrieval_ablation(pipeline: RAGPipeline, cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rewrite_started = asyncio.get_running_loop().time()
    if pipeline.config.rewrite.enabled:
        await pipeline.prefetch_rewrites([case["query"] for case in cases])
    rewrite_prefetch_ms = (asyncio.get_running_loop().time() - rewrite_started) * 1000
    results = {}
    for method in ("dense", "bm25", "hybrid", "hybrid_rewrite", "production"):
        results[method] = await evaluate_retrieval(pipeline, cases, method=method)
    return {"benchmark": "rag_benchmark_60.json", "methods": results, "rewrite_prefetch_ms": round(rewrite_prefetch_ms, 3), "rewrite_max_concurrency": pipeline.config.rewrite.max_concurrency}


def write_ablation_report(result: Dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


async def _main(args) -> int:
    cases = load_benchmark(args.benchmark)
    from dotenv import load_dotenv
    load_dotenv()
    rewrite_client = None
    rewrite_model = ""
    api_key = __import__("os").getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        from anthropic import AsyncAnthropic
        client_kwargs = {"api_key": api_key}
        if __import__("os").getenv("ANTHROPIC_BASE_URL"):
            client_kwargs["base_url"] = __import__("os").getenv("ANTHROPIC_BASE_URL")
        rewrite_client = AsyncAnthropic(**client_kwargs)
        rewrite_model = __import__("os").getenv("ANTHROPIC_MODEL", "")
    pipeline = RAGPipeline.from_index(load_rag_config(args.config), rewrite_client=rewrite_client, rewrite_model=rewrite_model)
    result = await run_retrieval_ablation(pipeline, cases)
    import torch
    result["runtime"] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dense_service_device": getattr(getattr(pipeline.dense, "embedding_service", None), "device", None),
        "dense_model_device": str(getattr(getattr(pipeline.dense, "embedding_service", None), "_model", None).device) if getattr(getattr(pipeline.dense, "embedding_service", None), "_model", None) is not None else "",
        "reranker_service_device": getattr(pipeline.reranker, "device", None),
        "reranker_model_device": str(getattr(getattr(pipeline.reranker, "_model", None), "device", "")),
        "collection": load_rag_config(args.config).dense.collection,
        "index_status": pipeline.index_status,
        "startup_fallbacks": pipeline.startup_fallbacks,
    }
    output = args.output or Path(__file__).parents[1] / "data" / "eval" / "results" / "rag" / "retrieval_ablation.json"
    write_ablation_report(result, output)
    print(json.dumps({name: value["metrics"] for name, value in result["methods"].items()}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate EchoMind RAG retrieval")
    parser.add_argument("--config")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
