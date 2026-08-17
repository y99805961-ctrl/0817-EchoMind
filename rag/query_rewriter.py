"""LLM multi-query rewriting with strict fallback and entity preservation."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Awaitable, Callable, List, Optional

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(
        self,
        client=None,
        model: str = "",
        count: int = 3,
        rewrite_fn: Optional[Callable[..., Any]] = None,
        max_concurrency: int = 10,
    ):
        self.client = client
        self.model = model
        self.count = max(0, int(count))
        self.rewrite_fn = rewrite_fn
        self.max_concurrency = max(1, int(max_concurrency or os.getenv("RAG_REWRITE_MAX_CONCURRENCY", "10")))
        self.last_status = "idle"
        self.last_error: Optional[str] = None
        self._cache: dict[tuple[str, int], List[str]] = {}
        self._metadata: dict[tuple[str, int], dict[str, Any]] = {}

    def status_for(self, query: str, count: Optional[int] = None) -> dict[str, Any]:
        key = (str(query or "").strip(), self.count if count is None else max(0, int(count)))
        return dict(self._metadata.get(key, {
            "status": "idle",
            "error": None,
            "latency_ms": 0.0,
            "retry_count": 0,
        }))

    async def rewrite_many(
        self,
        queries: List[str],
        count: Optional[int] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[dict[str, Any]]:
        """Rewrite remotely with bounded I/O concurrency; never parallelizes GPU retrieval."""
        limit = max(1, int(max_concurrency or self.max_concurrency))
        semaphore = asyncio.Semaphore(limit)

        async def one(query: str) -> dict[str, Any]:
            async with semaphore:
                rewritten = await self.rewrite(query, count)
                metadata = self.status_for(query, count)
                return {
                    "query": str(query),
                    "rewritten_queries": rewritten,
                    **metadata,
                }

        return list(await asyncio.gather(*(one(query) for query in queries)))

    def _finish(
        self,
        query: str,
        count: int,
        values: List[str],
        status: str,
        error: Optional[str],
        started: float,
    ) -> List[str]:
        self._cache[(query, count)] = list(values)
        self._metadata[(query, count)] = {
            "status": status,
            "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "retry_count": 0,
        }
        self.last_status = status
        self.last_error = error
        return list(values)

    async def rewrite(self, query: str, count: Optional[int] = None) -> List[str]:
        started = time.perf_counter()
        query = str(query or "").strip()
        count = self.count if count is None else max(0, int(count))
        if not query:
            return self._finish(query, count, [], "empty", "empty_query", started)
        cache_key = (query, count)
        if cache_key in self._cache:
            metadata = self._metadata.get(cache_key, {})
            self.last_status = metadata.get("status", "cached")
            self.last_error = metadata.get("error")
            return list(self._cache[cache_key])
        if count == 0 or (self.client is None and self.rewrite_fn is None):
            return self._finish(query, count, [query], "disabled", None, started)
        try:
            if self.rewrite_fn is not None:
                result = self.rewrite_fn(query, count)
                if hasattr(result, "__await__"):
                    result = await result
                candidates = result
            else:
                prompt = self._prompt(query, count)
                request = {
                    "model": self.model,
                    "max_tokens": int(os.getenv("RAG_REWRITE_MAX_TOKENS", "2048")),
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if _is_deepseek_provider():
                    request["extra_body"] = {"thinking": {"type": "disabled"}}
                response = await self.client.messages.create(**request)
                raw = extract_text_content(response.content)
                candidates = self._parse_json_array(raw)
            cleaned = self.clean(query, candidates, count)
            status = "ok" if len(cleaned) > 1 else "empty"
            return self._finish(query, count, cleaned, status, None, started)
        except Exception as exc:
            error = _safe_error_message(exc)
            logger.warning("Query rewrite failed; using original query: %s", error)
            return self._finish(query, count, [query], "failed", error, started)

    @staticmethod
    def clean(original: str, candidates: Any, count: int = 3) -> List[str]:
        values = candidates if isinstance(candidates, list) else []
        cleaned: List[str] = [original]
        seen = {original.casefold()}
        for candidate in values:
            if not isinstance(candidate, str):
                continue
            value = re.sub(r"\s+", " ", candidate).strip().strip('"\'')
            if not value or len(value) > 500:
                continue
            if value.casefold() in seen:
                continue
            # Mechanical repeats are not useful retrieval angles.
            if value.casefold().replace(" ", "") == original.casefold().replace(" ", ""):
                continue
            seen.add(value.casefold())
            cleaned.append(value)
            if len(cleaned) >= count + 1:
                break
        return cleaned

    @staticmethod
    def _parse_json_array(raw: str) -> List[str]:
        text = str(raw or "").strip()
        candidates = []
        for value in (text, text.strip("` ")):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    for key in ("queries", "rewrites", "search_queries"):
                        if isinstance(parsed.get(key), list):
                            return parsed[key]
            except json.JSONDecodeError:
                pass
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start:end + 1])
                if isinstance(value, list):
                    return value
            except json.JSONDecodeError:
                pass
        # Some compatible providers wrap the requested JSON in a short
        # numbered/bulleted explanation. Keep only those lines as a safe
        # last parser; unrelated prose still falls back to the original.
        for line in text.splitlines():
            match = re.match(r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$", line)
            if match and match.group(1).strip():
                candidates.append(match.group(1).strip().strip('"\''))
        if candidates:
            return candidates
        raise ValueError("rewrite response did not contain parseable queries")

    @staticmethod
    def _prompt(query: str, count: int) -> str:
        return f"""你是知识库检索查询改写器。请把原始问题改写成 {count} 个不同检索角度的短查询。

严格要求：
1. 保持原始意图，不添加用户未说过的事实；
2. 原样保留订单号、错误码、金额、时间约束、支付渠道、ID 和专有词；
3. 每个查询体现不同检索角度，例如流程、时效、异常原因或处理条件；
4. 只返回 JSON 字符串数组，不要解释。

原始问题：{query}"""


def _safe_error_message(exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    for token in (os.getenv("ANTHROPIC_API_KEY", ""), os.getenv("OPENAI_API_KEY", "")):
        if token:
            message = message.replace(token, "[REDACTED]")
    message = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", message)
    return message[:1000]


def _is_deepseek_provider() -> bool:
    return "deepseek" in f"{os.getenv('ANTHROPIC_BASE_URL', '')} {os.getenv('ANTHROPIC_MODEL', '')}".lower()
