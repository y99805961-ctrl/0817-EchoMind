"""Compare only Parent–Child child-size/overlap settings with dense-only retrieval."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
from pathlib import Path

from core.embedding_service import get_embedding_service
from rag.config import load_rag_config
from rag.corpus import build_corpus
from rag.dense_retriever import DenseRetriever
from rag.metrics import latency_summary, summarize_cases
from rag.parent_store import ParentStore
from rag.pipeline import RAGPipeline
from rag.reranker import CrossEncoderReranker
from .rag_retrieval_eval import DEFAULT_BENCHMARK, load_benchmark


async def evaluate_chunk_parameters(cases, embedding_service=None, config=None):
    base = config or load_rag_config()
    service = embedding_service or get_embedding_service(model_name=base.dense.model, device=base.dense.device)
    output = {}
    for child_size, overlap in ((300, 50), (400, 60), (550, 80)):
        candidate = copy.deepcopy(base)
        candidate.chunking.target_child_chars = child_size
        candidate.chunking.child_overlap_chars = overlap
        parents, children, _ = build_corpus(candidate)
        vectors = await asyncio.to_thread(service.encode_batch, [child.content for child in children])
        dense = DenseRetriever(children, embedding_service=service, top_k=10)
        dense.set_embeddings(vectors)
        candidate.rewrite.enabled = False
        candidate.reranker.enabled = False
        pipeline = RAGPipeline(
            candidate,
            dense_retriever=dense,
            bm25_retriever=None,
            parent_store=ParentStore(parents),
            reranker=CrossEncoderReranker(),
        )
        await asyncio.to_thread(
            dense.precompute_query_embeddings,
            [case["query"] for case in cases],
        )
        rows = []
        for case in cases:
            result = await pipeline.retrieve(case["query"], rewrite=False)
            rows.append({
                "child_ids": [hit.child_id for hit in result.retrieved_children],
                "parent_ids": [item.parent.parent_id for item in result.selected_parents],
                "timing": result.timing.to_dict(),
            })
        metrics = summarize_cases(cases, rows)
        metrics.update({
            "child_size": child_size,
            "overlap": overlap,
            "children_count": len(children),
            "avg_child_chars": round(statistics.mean(len(child.content) for child in children), 2) if children else 0.0,
            "latency": latency_summary([row["timing"].get("total_ms", 0.0) for row in rows]),
        })
        output[f"{child_size}/{overlap}"] = metrics
    return output


async def _main(args):
    result = await evaluate_chunk_parameters(load_benchmark(args.benchmark), config=load_rag_config(args.config))
    output = Path(args.output or Path(__file__).parents[1] / "data" / "eval" / "results" / "rag" / "chunk_parameter_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare RAG v2 child parameters")
    parser.add_argument("--config")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
