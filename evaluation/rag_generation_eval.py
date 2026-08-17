"""Generation evaluation grounded in the same production RAGResult context."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from core.llm_utils import extract_text_content
from rag.config import load_rag_config
from rag.pipeline import RAGPipeline
from .rag_retrieval_eval import DEFAULT_BENCHMARK, load_benchmark


logger = logging.getLogger(__name__)
SCORE_KEYS = ("correctness", "completeness", "relevance", "faithfulness", "abstention_accuracy")


class RAGGenerationEvaluator:
    def __init__(self, pipeline: RAGPipeline, generator: Callable[[str, str], Any], judge: Optional[Callable[..., Any]] = None):
        self.pipeline = pipeline
        self.generator = generator
        self.judge = judge

    async def evaluate(self, cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        wall_started = time.perf_counter()
        rewrite_started = time.perf_counter()
        await self.pipeline.prefetch_rewrites([case["query"] for case in cases])
        rewrite_prefetch_ms = (time.perf_counter() - rewrite_started) * 1000
        rows: List[Dict[str, Any]] = []
        for case in cases:
            retrieval = await self.pipeline.retrieve(case["query"])
            generation_started = time.perf_counter()
            generation_error = None
            try:
                generated = self.generator(case["query"], retrieval.context_text)
                if hasattr(generated, "__await__"):
                    generated = await generated
                generated = str(generated)
                generation_status = "ok"
            except Exception as exc:
                generated = ""
                generation_status = "generation_failed"
                generation_error = {
                    "type": type(exc).__name__,
                    "message": _safe_exception_message(exc),
                }
                logger.warning("Generation failed; continuing case as unavailable: %s", generation_error["type"])
            generation_latency_ms = round((time.perf_counter() - generation_started) * 1000, 3)
            if generation_status == "ok":
                judged = await self._score(case, retrieval.context_text, generated)
            else:
                judged = {
                    "scores": {key: None for key in SCORE_KEYS},
                    "judge_status": "not_run_generation_failed",
                    "judge_error_type": "GenerationNotRun",
                    "judge_error_message": generation_error["message"],
                    "judge_raw_output": "",
                    "judge_latency_ms": 0.0,
                    "judge_retried": False,
                    "judge_retry_count": 0,
                }
            rows.append({
                "id": case["id"],
                "answer": generated,
                "generation_status": generation_status,
                "generation_error_type": generation_error["type"] if generation_error else None,
                "generation_error_message": generation_error["message"] if generation_error else None,
                "generation_latency_ms": generation_latency_ms,
                "scores": judged["scores"],
                "timing": retrieval.timing.to_dict(),
                "rewritten_queries": retrieval.rewritten_queries,
                "rewrite_status": retrieval.rewrite_status,
                "rewrite_error": retrieval.rewrite_error,
                "rewrite_latency_ms": retrieval.timing.rewrite_ms,
                "rewrite_fallback": any("rewrite" in item for item in retrieval.fallbacks),
                "fallbacks": retrieval.fallbacks,
                **{key: judged.get(key) for key in (
                    "judge_status",
                    "judge_error_type",
                    "judge_error_message",
                    "judge_raw_output",
                    "judge_latency_ms",
                    "judge_retried",
                    "judge_retry_count",
                )},
            })
        valid_rows = [
            row for row in rows
            if row.get("judge_status") in {"ok", "deterministic"}
            and all(row["scores"].get(key) is not None for key in SCORE_KEYS)
        ]
        averages = {
            key: round(statistics.mean([float(row["scores"][key]) for row in valid_rows]), 4)
            if valid_rows else None
            for key in SCORE_KEYS
        }
        judge_successes = sum(row.get("judge_status") == "ok" for row in rows)
        judge_failures = sum(row.get("judge_status") == "judge_failed" for row in rows)
        judge_not_run = sum(row.get("judge_status") == "not_run_generation_failed" for row in rows)
        generation_failures = sum(row.get("generation_status") == "generation_failed" for row in rows)
        rewrite_successes = sum(row.get("rewrite_status") == "ok" for row in rows)
        rewrite_failures = sum(row.get("rewrite_fallback") for row in rows)
        result = {
            "benchmark": "rag_benchmark_60.json",
            "metrics": averages,
            "quality_eval": {
                "valid_judged_cases": len(valid_rows),
                "unavailable_cases": len(rows) - len(valid_rows),
                "judge_successes": judge_successes,
                "judge_failures": judge_failures,
                "judge_not_run": judge_not_run,
                "generation_failures": generation_failures,
                "judge_success_rate": round(judge_successes / len(rows), 6) if rows else 0.0,
                "judge_failure_rate": round(judge_failures / len(rows), 6) if rows else 0.0,
            },
            "rewrite_eval": {
                "rewrite_successes": rewrite_successes,
                "rewrite_fallbacks": rewrite_failures,
                "rewrite_success_rate": round(rewrite_successes / len(rows), 6) if rows else 0.0,
                "max_concurrency": self.pipeline.config.rewrite.max_concurrency,
            },
            "timing": {
                "wall_clock_ms": round((time.perf_counter() - wall_started) * 1000, 3),
                "rewrite_prefetch_ms": round(rewrite_prefetch_ms, 3),
                "case_count": len(rows),
            },
            "cases": rows,
        }
        if self.judge is not None and hasattr(self.judge, "stats"):
            result["judge"] = self.judge.stats
        return result

    async def _score(self, case: Dict[str, Any], context: str, answer: str) -> Dict[str, Any]:
        if self.judge is not None:
            result = self.judge(case, context, answer)
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, dict) and "scores" in result:
                return result
            scores = {key: float(result[key]) if result.get(key) is not None else None for key in SCORE_KEYS}
            return {"scores": scores, "judge_status": "ok"}
        if not case.get("answerable", True):
            abstained = any(token in answer for token in ("无法确认", "没有足够", "知识库", "无法提供"))
            scores = {"correctness": float(abstained), "completeness": float(abstained), "relevance": 1.0 if abstained else 0.0, "faithfulness": float(abstained), "abstention_accuracy": float(abstained)}
            return {"scores": scores, "judge_status": "deterministic"}
        key_points = [str(point) for point in case.get("key_points", [])]
        matched = sum(1 for point in key_points if point and point in answer)
        completeness = matched / len(key_points) if key_points else 0.0
        grounded = bool(context) and completeness > 0
        scores = {"correctness": float(grounded), "completeness": completeness, "relevance": float(bool(answer.strip())), "faithfulness": float(grounded), "abstention_accuracy": 1.0}
        return {"scores": scores, "judge_status": "deterministic"}


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

    def __init__(self, client, model: str, max_retries: Optional[int] = None, retry_base_seconds: Optional[float] = None, max_tokens: Optional[int] = None, max_tokens_cap: Optional[int] = None, disable_thinking: Optional[bool] = None):
        self.client = client
        self.model = model
        self.max_tokens = max(512, int(os.getenv("RAG_JUDGE_MAX_TOKENS", "2048") if max_tokens is None else max_tokens))
        self.max_tokens_cap = max(self.max_tokens, int(os.getenv("RAG_JUDGE_MAX_TOKENS_CAP", "4096") if max_tokens_cap is None else max_tokens_cap))
        env_thinking = os.getenv("RAG_JUDGE_DISABLE_THINKING")
        self.disable_thinking = _as_bool(env_thinking) if env_thinking is not None else _is_deepseek_provider()
        self.max_retries = min(3, max(0, int(os.getenv("RAG_JUDGE_MAX_RETRIES", "2") if max_retries is None else max_retries)))
        self.retry_base_seconds = max(0.0, float(os.getenv("RAG_JUDGE_RETRY_BASE_SECONDS", "0.5") if retry_base_seconds is None else retry_base_seconds))
        self.successes = 0
        self.failures = 0
        self.retries = 0
        self.failure_types: Dict[str, int] = {}

    @property
    def stats(self) -> Dict[str, Any]:
        total = self.successes + self.failures
        return {
            "successes": self.successes,
            "failures": self.failures,
            "retries": self.retries,
            "max_tokens": self.max_tokens,
            "max_tokens_cap": self.max_tokens_cap,
            "disable_thinking": self.disable_thinking,
            "failure_rate": round(self.failures / total, 6) if total else 0.0,
            "failure_types": dict(sorted(self.failure_types.items())),
        }

    async def __call__(self, case: Dict[str, Any], context: str, answer: str) -> Dict[str, Any]:
        prompt = self.PROMPT.format(
            question=case["query"],
            gold_answer=case.get("gold_answer", ""),
            key_points=json.dumps(case.get("key_points", []), ensure_ascii=False),
            context=context,
            answer=answer,
        )
        started = time.perf_counter()
        retry_count = 0
        request_tokens = self.max_tokens
        last_stop_reason: Optional[str] = None
        last_raw = ""
        last_error_type: Optional[str] = None
        last_error_message: Optional[str] = None
        while True:
            try:
                request = {
                    "model": self.model,
                    "max_tokens": request_tokens,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self.disable_thinking:
                    request["extra_body"] = {"thinking": {"type": "disabled"}}
                response = await self.client.messages.create(**request)
                last_stop_reason = getattr(response, "stop_reason", None)
                last_raw = _redact_sensitive(extract_text_content(response.content))
                scores = self._parse_scores(last_raw)
                self.successes += 1
                self.retries += retry_count
                return {
                    "scores": scores,
                    "judge_status": "ok",
                    "judge_error_type": None,
                    "judge_error_message": None,
                    "judge_raw_output": last_raw,
                    "judge_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "judge_retried": retry_count > 0,
                    "judge_retry_count": retry_count,
                }
            except Exception as exc:
                last_error_type = type(exc).__name__
                last_error_message = _safe_exception_message(exc)
                retryable = _is_retryable_judge_error(exc)
                if isinstance(exc, JudgeFormatError) and last_stop_reason == "max_tokens" and request_tokens < self.max_tokens_cap:
                    request_tokens = min(self.max_tokens_cap, request_tokens * 2)
                    retryable = True
                    last_error_message = f"{last_error_message}; increased_max_tokens={request_tokens}"
                if retryable and retry_count < self.max_retries:
                    delay = self.retry_base_seconds * (2 ** retry_count)
                    retry_count += 1
                    logger.warning("RAG judge transient failure (%s); retry %d/%d", last_error_type, retry_count, self.max_retries)
                    if delay:
                        await asyncio.sleep(delay)
                    continue
                self.failures += 1
                self.retries += retry_count
                self.failure_types[last_error_type] = self.failure_types.get(last_error_type, 0) + 1
                logger.warning("RAG judge failed; score unavailable: %s", last_error_type)
                return {
                    "scores": {key: None for key in SCORE_KEYS},
                    "judge_status": "judge_failed",
                    "judge_error_type": last_error_type,
                    "judge_error_message": last_error_message,
                    "judge_raw_output": last_raw,
                    "judge_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "judge_retried": retry_count > 0,
                    "judge_retry_count": retry_count,
                }

    @staticmethod
    def _parse_scores(raw: str) -> Dict[str, float]:
        text = str(raw or "").strip()
        if not text:
            raise JudgeFormatError("empty_response")
        decoder = json.JSONDecoder()
        objects: List[dict[str, Any]] = []
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                objects.append(value)
        for data in objects:
            missing = [key for key in SCORE_KEYS if key not in data]
            if missing:
                continue
            scores: Dict[str, float] = {}
            for key in SCORE_KEYS:
                value = data[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise JudgeFormatError(f"non_numeric_field:{key}")
                scores[key] = max(0.0, min(1.0, float(value)))
            return scores
        if objects:
            raise JudgeFormatError("missing_required_fields")
        raise JudgeFormatError("invalid_json")


class JudgeFormatError(ValueError):
    """The provider responded, but the response was not usable judge JSON."""


def _safe_exception_message(exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    for token in (os.getenv("ANTHROPIC_API_KEY", ""), os.getenv("OPENAI_API_KEY", "")):
        if token:
            message = message.replace(token, "[REDACTED]")
    message = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", message)
    return message[:1000]


def _redact_sensitive(value: Any) -> str:
    text = str(value or "")
    for token in (os.getenv("ANTHROPIC_API_KEY", ""), os.getenv("OPENAI_API_KEY", "")):
        if token:
            text = text.replace(token, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", text)
    return text[:6000]


def _redact_url(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"(?i)([?&](?:api[_-]?key|token|key)=)[^&]+", r"\1[REDACTED]", value)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_deepseek_provider() -> bool:
    return "deepseek" in f"{os.getenv('ANTHROPIC_BASE_URL', '')} {os.getenv('ANTHROPIC_MODEL', '')}".lower()


def _status_code(exc: BaseException) -> Optional[int]:
    value = getattr(exc, "status_code", None)
    if value is None:
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable_judge_error(exc: BaseException) -> bool:
    code = _status_code(exc)
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if code in {400, 401, 403, 404, 422}:
        return False
    if any(token in name or token in message for token in ("authentication", "unauthorized", "invalid api key", "model not found", "invalid_request")):
        return False
    if isinstance(exc, JudgeFormatError):
        return True
    if code == 429 or (code is not None and 500 <= code <= 599):
        return True
    return any(token in name or token in message for token in ("timeout", "timed out", "rate limit", "ratelimit", "connection", "temporar", "overloaded", "service unavailable"))


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
    judge = None
    if client is not None:
        judge = AnthropicRAGJudge(client, model)
        evaluator = RAGGenerationEvaluator(
            pipeline,
            AnthropicGenerator(client, model),
            judge,
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
        "judge": {
            "provider": "anthropic_compatible" if client is not None else "deterministic_fallback",
            "model": model or None,
            "base_url": _redact_url(os.getenv("ANTHROPIC_BASE_URL")) if client is not None else None,
            "max_retries": judge.max_retries if judge is not None else 0,
            "max_tokens": judge.max_tokens if judge is not None else None,
            "disable_thinking": judge.disable_thinking if judge is not None else None,
        },
    }
    output = Path(args.output or Path(__file__).parents[1] / "data" / "eval" / "results" / "rag" / "generation_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


def _judge_latency_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None}
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return round(ordered[index], 3)

    return {
        "mean_ms": round(statistics.mean(ordered), 3),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
    }


def _is_provider_level_judge_failure(outcome: Dict[str, Any]) -> bool:
    error_type = str(outcome.get("judge_error_type") or "").lower()
    message = str(outcome.get("judge_error_message") or "").lower()
    return error_type in {"apistatuserror", "apiconnectionerror", "apitimeouterror", "timeout", "connectionerror"} or any(
        token in message for token in ("401", "402", "403", "429", "5xx", "insufficient balance", "provider unavailable")
    )


async def _judge_only_main(args) -> int:
    """Judge historical answers only; never invokes Rewrite or Generation."""
    from dotenv import load_dotenv

    load_dotenv()
    if not args.input:
        raise ValueError("--input is required with --judge-only")
    input_path = Path(args.input)
    history = json.loads(input_path.read_text(encoding="utf-8"))
    historical_rows = history.get("cases", history) if isinstance(history, dict) else history
    if not isinstance(historical_rows, list):
        raise ValueError("historical generation input must contain a cases list")
    historical_by_id = {str(row.get("id")): row for row in historical_rows if row.get("id") is not None and "answer" in row}

    benchmark = load_benchmark(args.benchmark)
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(benchmark):
            raise ValueError(f"--limit must be between 1 and {len(benchmark)}")
        benchmark = benchmark[:args.limit]
    selected = []
    for case in benchmark:
        historical = historical_by_id.get(str(case["id"]))
        if historical is None:
            raise ValueError(f"historical answer missing for benchmark case {case['id']}")
        selected.append((case, str(historical.get("answer", ""))))

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for --judge-only")
    from anthropic import AsyncAnthropic

    kwargs = {"api_key": api_key}
    if os.getenv("ANTHROPIC_BASE_URL"):
        kwargs["base_url"] = os.getenv("ANTHROPIC_BASE_URL")
    model = os.getenv("ANTHROPIC_MODEL", "qwen3.7-plus")
    client = AsyncAnthropic(**kwargs)

    # Judge-only defaults from the Phase 2.2 runbook. Existing explicit env
    # values still win, and no credential is written into the result.
    os.environ.setdefault("RAG_JUDGE_DISABLE_THINKING", "true")
    os.environ.setdefault("RAG_JUDGE_MAX_TOKENS", "512")
    os.environ.setdefault("RAG_JUDGE_MAX_TOKENS_CAP", "1024")
    os.environ.setdefault("RAG_JUDGE_MAX_RETRIES", "1")
    os.environ.setdefault("RAG_JUDGE_RETRY_BASE_SECONDS", "0.5")
    judge = AnthropicRAGJudge(client, model)

    # Explicitly force local-only model loading. Retrieval is allowed solely
    # to reconstruct missing historical context, with Rewrite disabled.
    os.environ.setdefault("RAG_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pipeline = RAGPipeline.from_index(load_rag_config(args.config), rewrite_client=None, rewrite_model="")
    queries = [case["query"] for case, _ in selected]
    if pipeline.dense is not None:
        await asyncio.to_thread(pipeline.dense.precompute_query_embeddings, queries)

    rows: List[Dict[str, Any]] = []
    judge_latencies: List[float] = []
    provider_error_streak = 0
    stopped_early = False
    stop_reason = None
    for case, answer in selected:
        retrieval = await pipeline.retrieve(case["query"], rewrite=False)
        outcome = await judge(case, retrieval.context_text, answer)
        judge_latencies.append(float(outcome.get("judge_latency_ms") or 0.0))
        if outcome.get("judge_status") == "ok":
            provider_error_streak = 0
        elif _is_provider_level_judge_failure(outcome):
            provider_error_streak += 1
        else:
            provider_error_streak = 0
        rows.append({
            "id": case["id"],
            "answer": answer,
            "scores": outcome["scores"],
            "judge_status": outcome.get("judge_status"),
            "judge_error_type": outcome.get("judge_error_type"),
            "judge_error_message": outcome.get("judge_error_message"),
            "judge_raw_output": outcome.get("judge_raw_output", ""),
            "judge_latency_ms": outcome.get("judge_latency_ms"),
            "judge_retried": outcome.get("judge_retried", False),
            "judge_retry_count": outcome.get("judge_retry_count", 0),
            "context_source": "local_retrieval_replay",
            "retrieval_fallbacks": retrieval.fallbacks,
            "retrieval_timing": retrieval.timing.to_dict(),
        })
        if provider_error_streak >= 3:
            stopped_early = True
            stop_reason = "three_consecutive_provider_errors"
            break

    valid_rows = [
        row for row in rows
        if row.get("judge_status") == "ok" and all(row["scores"].get(key) is not None for key in SCORE_KEYS)
    ]
    metrics = {
        key: round(statistics.mean([float(row["scores"][key]) for row in valid_rows]), 4) if valid_rows else None
        for key in SCORE_KEYS
    }
    successes = sum(row.get("judge_status") == "ok" for row in rows)
    failures = sum(row.get("judge_status") == "judge_failed" for row in rows)
    result = {
        "evaluation_mode": "judge_only",
        "answer_source": str(input_path),
        "answer_generator": "historical_deepseek",
        "judge_provider": "qwen_anthropic_compatible",
        "judge_model": model,
        "benchmark": "rag_benchmark_60.json",
        "benchmark_total_cases": 60,
        "cases_evaluated": len(rows),
        "historical_answers_loaded": len(rows),
        "generation_calls": 0,
        "rewrite_llm_calls": 0,
        "local_retrieval_calls": len(rows),
        "judge_attempted": len(rows),
        "judge_successes": successes,
        "judge_failures": failures,
        "valid_judged_cases": len(valid_rows),
        "judge_success_rate": round(successes / len(rows), 6) if rows else 0.0,
        "judge_latency": _judge_latency_summary(judge_latencies),
        "metrics": metrics,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "judge_stats": judge.stats,
        "runtime": {
            "model": model,
            "base_url": _redact_url(os.getenv("ANTHROPIC_BASE_URL")),
            "judge_max_retries": judge.max_retries,
            "judge_max_tokens": judge.max_tokens,
            "judge_disable_thinking": judge.disable_thinking,
            "dense_model_device": str(getattr(getattr(getattr(pipeline.dense, "embedding_service", None), "_model", None), "device", "")),
            "reranker_model_device": str(getattr(getattr(pipeline.reranker, "_model", None), "device", "")),
            "collection": load_rag_config(args.config).dense.collection,
            "index_status": pipeline.index_status,
        },
        "cases": rows,
    }
    output = Path(args.output or Path(__file__).parents[1] / "data" / "eval" / "results" / "rag" / "generation_judge_qwen.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("evaluation_mode", "cases_evaluated", "historical_answers_loaded", "generation_calls", "rewrite_llm_calls", "judge_attempted", "judge_successes", "judge_failures", "valid_judged_cases", "judge_success_rate", "judge_latency", "stopped_early", "stop_reason", "metrics")}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate grounded RAG generation")
    parser.add_argument("--config")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output")
    parser.add_argument("--judge-only", action="store_true", help="Judge historical answers without Rewrite or Generation")
    parser.add_argument("--input", help="Historical generation_eval.json for --judge-only")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N benchmark cases")
    args = parser.parse_args(argv)
    return asyncio.run(_judge_only_main(args) if args.judge_only else _main(args))


if __name__ == "__main__":
    raise SystemExit(main())
