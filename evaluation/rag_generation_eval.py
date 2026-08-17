"""Generation evaluation grounded in the same production RAGResult context."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from core.llm_utils import extract_text_content
from rag.config import load_rag_config
from rag.pipeline import RAGPipeline
from .rag_retrieval_eval import DEFAULT_BENCHMARK, load_benchmark


logger = logging.getLogger(__name__)


class RAGGenerationEvaluator:
    def __init__(self, pipeline: RAGPipeline, generator: Callable[[str, str], Any], judge: Optional[Callable[..., Any]] = None):
        self.pipeline = pipeline
        self.generator = generator
        self.judge = judge

    async def evaluate(self, cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for case in cases:
            retrieval = await self.pipeline.retrieve(case["query"])
            generated = self.generator(case["query"], retrieval.context_text)
            if hasattr(generated, "__await__"):
                generated = await generated
            scores = await self._score(case, retrieval.context_text, str(generated))
            rows.append({
                "id": case["id"],
                "answer": str(generated),
                "scores": scores,
                "timing": retrieval.timing.to_dict(),
                "rewritten_queries": retrieval.rewritten_queries,
                "fallbacks": retrieval.fallbacks,
            })
        keys = ("correctness", "completeness", "relevance", "faithfulness", "abstention_accuracy")
        averages = {key: round(statistics.mean([row["scores"][key] for row in rows]), 4) if rows else 0.0 for key in keys}
        result = {"benchmark": "rag_benchmark_60.json", "metrics": averages, "cases": rows}
        if self.judge is not None and hasattr(self.judge, "stats"):
            result["judge"] = self.judge.stats
        return result

    async def _score(self, case: Dict[str, Any], context: str, answer: str) -> Dict[str, float]:
        if self.judge is not None:
            result = self.judge(case, context, answer)
            if hasattr(result, "__await__"):
                result = await result
            return {key: float(result.get(key, 0.0)) for key in ("correctness", "completeness", "relevance", "faithfulness", "abstention_accuracy")}
        if not case.get("answerable", True):
            abstained = any(token in answer for token in ("无法确认", "没有足够", "知识库", "无法提供"))
            return {"correctness": float(abstained), "completeness": float(abstained), "relevance": 1.0 if abstained else 0.0, "faithfulness": float(abstained), "abstention_accuracy": float(abstained)}
        key_points = [str(point) for point in case.get("key_points", [])]
        matched = sum(1 for point in key_points if point and point in answer)
        completeness = matched / len(key_points) if key_points else 0.0
        grounded = bool(context) and completeness > 0
        return {"correctness": float(grounded), "completeness": completeness, "relevance": float(bool(answer.strip())), "faithfulness": float(grounded), "abstention_accuracy": 1.0}


class AnthropicGenerator:
    """Production-style grounded answer generator for the generation benchmark."""
    SYSTEM = (
        "你是 EchoMind 智能客服。优先依据提供的 Knowledge Context 回答。"
        "对于政策、时效、费用、权限和业务规则，不得根据常识补造事实。"
        "如果上下文不足以确认，请明确说当前知识库没有足够信息确认，并给出核实或升级路径。"
    )

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    async def __call__(self, question: str, context: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0.0,
            system=self.SYSTEM,
            messages=[
                {"role": "user", "content": f"Knowledge Context:\n{context or '(empty)'}"},
                {"role": "assistant", "content": "我会只依据已提供的知识上下文回答。"},
                {"role": "user", "content": question},
            ],
        )
        return extract_text_content(response.content)


class AnthropicRAGJudge:
    """Optional LLM-as-Judge adapter; deterministic evaluation remains the fallback."""
    PROMPT = """你是 RAG 质量评测专家。只返回 JSON 数字字段。
问题：{question}
Gold Answer：{gold_answer}
Gold Key Points：{key_points}
Retrieved Context：{context}
Generated Answer：{answer}

评分字段均为0到1：correctness、completeness、relevance、faithfulness、abstention_accuracy。
faithfulness 只判断回答事实是否被 Retrieved Context 支撑；answerable=false 时，只有明确承认知识库不足才算 abstention_accuracy=1。"""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.successes = 0
        self.failures = 0
        self.failure_types: Dict[str, int] = {}

    @property
    def stats(self) -> Dict[str, Any]:
        total = self.successes + self.failures
        return {
            "successes": self.successes,
            "failures": self.failures,
            "failure_rate": round(self.failures / total, 6) if total else 0.0,
            "failure_types": dict(sorted(self.failure_types.items())),
        }

    async def __call__(self, case: Dict[str, Any], context: str, answer: str) -> Dict[str, float]:
        prompt = self.PROMPT.format(
            question=case["query"],
            gold_answer=case.get("gold_answer", ""),
            key_points=json.dumps(case.get("key_points", []), ensure_ascii=False),
            context=context,
            answer=answer,
        )
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(response.content)
            start, end = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[start:end + 1])
            self.successes += 1
            return {key: max(0.0, min(1.0, float(data.get(key, 0.0)))) for key in ("correctness", "completeness", "relevance", "faithfulness", "abstention_accuracy")}
        except Exception as exc:
            self.failures += 1
            failure_type = type(exc).__name__
            self.failure_types[failure_type] = self.failure_types.get(failure_type, 0) + 1
            logger.warning("RAG judge failed; returning zero scores: %s", failure_type)
            return {"correctness": 0.0, "completeness": 0.0, "relevance": 0.0, "faithfulness": 0.0, "abstention_accuracy": 0.0}


async def _main(args) -> int:
    cases = load_benchmark(args.benchmark)
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    rewrite_client = None
    rewrite_model = ""
    client = None
    model = ""
    if api_key:
        from anthropic import AsyncAnthropic
        kwargs = {"api_key": api_key}
        if os.getenv("ANTHROPIC_BASE_URL"):
            kwargs["base_url"] = os.getenv("ANTHROPIC_BASE_URL")
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        client = AsyncAnthropic(**kwargs)
        rewrite_client = client
        rewrite_model = model
    pipeline = RAGPipeline.from_index(
        load_rag_config(args.config),
        rewrite_client=rewrite_client,
        rewrite_model=rewrite_model,
    )
    if pipeline.dense is not None:
        await asyncio.to_thread(
            pipeline.dense.precompute_query_embeddings,
            [case["query"] for case in cases],
        )
    if client is not None:
        evaluator = RAGGenerationEvaluator(
            pipeline,
            AnthropicGenerator(client, model),
            AnthropicRAGJudge(client, model),
        )
    else:
        async def unavailable_generator(question: str, context: str) -> str:
            return "当前知识库没有足够信息确认。" if not context else "当前运行未配置生成模型。"
        evaluator = RAGGenerationEvaluator(pipeline, unavailable_generator)
    result = await evaluator.evaluate(cases)
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
        "judge": "anthropic" if client is not None else "deterministic_fallback",
    }
    output = Path(args.output or Path(__file__).parents[1] / "data" / "eval" / "results" / "rag" / "generation_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate grounded RAG generation")
    parser.add_argument("--config")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
