"""Intent recognition v2: LLM + BGE-M3 templates + high-precision rules.

The public ``IntentRecognizer.recognize`` API remains asynchronous and keeps the
existing IntentResult fields.  New diagnostics are additive so the existing
orchestrator and /chat response continue to work with only a small type widening
for nested source-score details.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from anthropic import AsyncAnthropic

from core.embedding_service import EmbeddingServiceError, get_embedding_service
from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY = "query"
    COMPLAINT = "complaint"
    REQUEST = "request"
    GREETING = "greeting"
    ESCALATION = "escalation"
    TECHNICAL = "technical"
    BILLING = "billing"
    ACCOUNT = "account"
    FEEDBACK = "feedback"
    ORDER_STATUS = "order_status"
    LOGISTICS = "logistics"
    REFUND = "refund"
    INVOICE = "invoice"
    PAYMENT_ISSUE = "payment_issue"
    ACCOUNT_SECURITY = "account_security"
    TECHNICAL_LOGIN = "technical_login"
    TECHNICAL_CRASH = "technical_crash"
    HUMAN_HANDOFF = "human_handoff"
    OTHER = "other"


class UrgencyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    urgency: UrgencyLevel
    intent_group: str
    entities: Dict[str, List[str]]
    reasoning: str
    latency_ms: float
    source_scores: Dict[str, Any] = field(default_factory=dict)
    top_candidates: List[Dict[str, Any]] = field(default_factory=list)
    top1_score: float = 0.0
    top2_score: float = 0.0
    margin: float = 0.0


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TEMPLATE_PATH = _ROOT / "data" / "intent" / "templates.json"

_SPECIFIC_INTENTS = {
    IntentCategory.ORDER_STATUS,
    IntentCategory.LOGISTICS,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.ACCOUNT_SECURITY,
    IntentCategory.TECHNICAL_LOGIN,
    IntentCategory.TECHNICAL_CRASH,
    IntentCategory.HUMAN_HANDOFF,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.ORDER_STATUS: IntentCategory.QUERY,
    IntentCategory.LOGISTICS: IntentCategory.QUERY,
    IntentCategory.REFUND: IntentCategory.BILLING,
    IntentCategory.INVOICE: IntentCategory.BILLING,
    IntentCategory.PAYMENT_ISSUE: IntentCategory.BILLING,
    IntentCategory.ACCOUNT_SECURITY: IntentCategory.ACCOUNT,
    IntentCategory.TECHNICAL_LOGIN: IntentCategory.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: IntentCategory.TECHNICAL,
    IntentCategory.HUMAN_HANDOFF: IntentCategory.ESCALATION,
}

_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH: ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM: ["这周", "本周", "soon", "快点"],
}

# Rules intentionally target high-precision signals instead of attempting to
# cover all natural language.  A query may receive multiple domain scores for
# routing evaluation, while final intent still uses the same fusion gate.
_RULES: Sequence[Tuple[IntentCategory, str, str]] = (
    (IntentCategory.HUMAN_HANDOFF, r"转人工|人工客服|找人工|真人客服|人工介入", "human_handoff"),
    (IntentCategory.ACCOUNT_SECURITY, r"账号被盗|账户被盗|异常登录|陌生设备|未经授权|盗用", "account_security"),
    (IntentCategory.TECHNICAL_LOGIN, r"(?<!\d)401(?!\d)|无法登录|登录失败|验证码收不到|认证失败|未授权", "technical_login"),
    (IntentCategory.TECHNICAL_CRASH, r"(?<!\d)500(?!\d)|闪退|崩溃|崩掉|crash|服务器错误", "technical_crash"),
    (IntentCategory.PAYMENT_ISSUE, r"重复扣款|扣了两次|多扣|支付失败|扣费异常|未经授权支付", "payment_issue"),
    (IntentCategory.INVOICE, r"发票|开票|抬头|税号|电子票据", "invoice"),
    (IntentCategory.REFUND, r"退款|返款|退货退款|退回款|款项退回", "refund"),
    (IntentCategory.LOGISTICS, r"物流|快递|配送|运单|包裹|派送|签收", "logistics"),
    (IntentCategory.ORDER_STATUS, r"订单状态|订单进度|订单处理中|订单.*发货|订单.*取消", "order_status"),
    (IntentCategory.ESCALATION, r"投诉|经理|主管|负责人|升级处理", "escalation"),
    (IntentCategory.COMPLAINT, r"太差|失望|不满意|糟糕|没人处理|拖了太久", "complaint"),
    (IntentCategory.GREETING, r"你好|嗨|hello|hi|早上好|晚上好", "greeting"),
    (IntentCategory.FEEDBACK, r"满意|感谢|好评|很棒|不错|专业", "feedback"),
    (IntentCategory.ACCOUNT, r"邮箱|账户资料|个人信息|账号设置|注销账户", "account"),
    (IntentCategory.BILLING, r"账单|费用明细|收费|结算", "billing"),
    (IntentCategory.TECHNICAL, r"系统异常|页面错误|程序故障|接口错误", "technical"),
    (IntentCategory.REQUEST, r"帮我|请协助|麻烦处理|我要办理|申请", "request"),
    (IntentCategory.QUERY, r"怎么|如何|哪里|查询|查看|什么情况", "query"),
)


def clamp01(value: Any) -> float:
    """Normalize any numeric source score to the closed interval [0, 1]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))


def normalize_llm_score(value: Any) -> float:
    return clamp01(value)


def normalize_rule_score(value: Any) -> float:
    return clamp01(value)


def normalize_embedding_score(cosine: Any) -> float:
    """Map cosine similarity from [-1, 1] into a comparable [0, 1] score."""
    try:
        value = float(cosine)
    except (TypeError, ValueError):
        return 0.0
    return clamp01((value + 1.0) / 2.0)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """Async recognizer with injectable sources for deterministic offline tests."""

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        llm_weight: Optional[float] = None,
        embedding_weight: Optional[float] = None,
        rule_weight: Optional[float] = None,
        embedding_top_n: Optional[int] = None,
        templates_path: Optional[str] = None,
        embedding_service: Optional[Any] = None,
        llm_classifier: Optional[Any] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.model = model
        self.threshold = float(
            confidence_threshold
            if confidence_threshold is not None
            else os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.50")
        )
        self.margin_threshold = float(
            margin_threshold
            if margin_threshold is not None
            else os.getenv("INTENT_MARGIN_THRESHOLD", "0.05")
        )
        self.weights = {
            "llm": _configured_weight(llm_weight, "INTENT_LLM_WEIGHT", 0.45),
            "embedding": _configured_weight(embedding_weight, "INTENT_EMBED_WEIGHT", 0.35),
            "rules": _configured_weight(rule_weight, "INTENT_RULE_WEIGHT", 0.20),
        }
        self.embedding_top_n = max(
            1,
            int(embedding_top_n or os.getenv("INTENT_EMBEDDING_TOP_N", "3")),
        )
        self.mode = (mode or os.getenv("INTENT_MODE", "llm_embedding")).lower()
        self._active_sources = _sources_for_mode(self.mode)
        self._templates_path = Path(templates_path) if templates_path else _DEFAULT_TEMPLATE_PATH
        self._templates = self._load_templates()
        self._tpl_embeddings: Dict[str, List[List[float]]] = {}
        self._embedding_service = embedding_service or (
            get_embedding_service() if "embedding" in self._active_sources else None
        )
        self._llm_classifier = llm_classifier
        self._client = None
        if "llm" in self._active_sources and llm_classifier is None and api_key:
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = AsyncAnthropic(**kwargs)
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1
        started = time.monotonic()
        clean_message = self._clean_text(message)

        llm_result: Dict[str, Any] = {"status": "disabled", "scores": {}}
        embedding_result: Dict[str, Any] = {"status": "disabled", "scores": {}}
        if self._active_sources == {"llm", "embedding", "rules"}:
            llm_result, embedding_result = await asyncio.gather(
                self._llm_recognize(clean_message, history),
                self._embedding_recognize(clean_message),
            )
        else:
            if "llm" in self._active_sources:
                llm_result = await self._llm_recognize(clean_message, history)
            if "embedding" in self._active_sources:
                embedding_result = await self._embedding_recognize(clean_message)
        # Rules are always collected as high-precision/routing diagnostics.
        # They participate in fusion only when the explicit rules ablation
        # mode is selected; production defaults to LLM+BGE semantic fusion.
        rules_result = self._rule_recognize(clean_message)

        fusion = self._fuse(llm_result, embedding_result, rules_result, message=clean_message)
        final_intent = fusion["intent"]
        result = IntentResult(
            intent=final_intent,
            confidence=fusion["confidence"],
            urgency=self._urgency(clean_message, final_intent),
            intent_group=self._intent_group(final_intent),
            entities=self._extract_entities(clean_message),
            reasoning=str(llm_result.get("reasoning", "")),
            latency_ms=(time.monotonic() - started) * 1000,
            source_scores=fusion["source_scores"],
            top_candidates=fusion["top_candidates"],
            top1_score=fusion["top1_score"],
            top2_score=fusion["top2_score"],
            margin=fusion["margin"],
        )
        self._cache[key] = result
        if len(self._cache) > 1000:
            for old_key in list(self._cache)[:500]:
                del self._cache[old_key]
        return result

    def reload_templates(self, templates_path: Optional[str] = None) -> None:
        if templates_path:
            self._templates_path = Path(templates_path)
        self._templates = self._load_templates()
        self._tpl_embeddings.clear()

    def learn(self, message: str, correct: IntentCategory) -> None:
        """Add an in-memory correction without mutating the benchmark or template file."""
        values = self._templates.setdefault(correct.value, [])
        if message not in values:
            values.append(message)
            self._tpl_embeddings.pop(correct.value, None)
            logger.info("Added runtime intent correction for %s", correct.value)

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        if self._llm_classifier is not None:
            try:
                data = self._llm_classifier(message, history)
                if inspect.isawaitable(data):
                    data = await data
                return self._coerce_llm_result(data)
            except Exception as exc:
                logger.warning("Injected LLM classifier failed: %s", exc)
                return {"status": "failed", "scores": {}, "reasoning": "LLM 失败", "failed": True}
        if self._client is None:
            return {"status": "failed", "scores": {}, "reasoning": "LLM unavailable", "failed": True}

        examples = "\n".join(
            f'  示例: "{templates[0]}" -> {label}'
            for label, templates in self._templates.items()
            if templates
        )
        context = ""
        if history:
            context = "\n最近上下文:\n" + "\n".join(
                f"{self._clean_text(item.get('role', 'user'))}: "
                f"{self._clean_text(item.get('content', ''))}"
                for item in history[-3:]
            )
        prompt = self._clean_text(
            f"""你是 EchoMind 客服意图分类器。请优先选择细粒度业务意图，只有无法判断时才使用宽泛类别。
返回严格 JSON，字段为 intent、confidence、reasoning。只返回一个 JSON 对象，
不要 Markdown 或额外文字。
可选意图: {', '.join(item.value for item in IntentCategory)}

参考示例:
{examples}
{context}
用户消息: {message}
"""
        )
        try:
            response = await self._client.messages.create(
                model=self.model,
                # DeepSeek-compatible reasoning responses can consume the
                # smaller budget before emitting the final JSON object.
                max_tokens=4096,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            try:
                return self._parse_llm_response(response)
            except ValueError:
                # Some reasoning responses occasionally end without exposing
                # the final JSON object. Retry once with a compact correction;
                # never infer an intent from non-JSON prose.
                retry_prompt = (
                    "只输出一个完整 JSON 对象，不要解释、不要 Markdown、不要额外文字。"
                    "字段必须是 intent、confidence、reasoning。"
                    f"intent 必须是以下之一：{', '.join(item.value for item in IntentCategory)}。"
                    f"用户消息：{message}"
                )
                retry_response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.0,
                    messages=[{"role": "user", "content": retry_prompt}],
                )
                return self._parse_llm_response(retry_response)
        except Exception as exc:
            logger.warning("LLM recognition failed: %s", exc)
            return {"status": "failed", "scores": {}, "reasoning": "LLM 失败", "failed": True}

    def _parse_llm_response(self, response: Any) -> Dict[str, Any]:
        raw = extract_text_content(response.content)
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON object")
        return self._coerce_llm_result(json.loads(raw[start:end]))

    def _coerce_llm_result(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("LLM result must be an object")
        raw_intent = data.get("intent", "other")
        try:
            intent = IntentCategory(str(raw_intent))
        except ValueError:
            intent = IntentCategory.OTHER
        confidence = normalize_llm_score(data.get("confidence", 0.0))
        scores = data.get("scores")
        if not isinstance(scores, dict):
            scores = {intent.value: confidence}
        scores = {
            str(label): normalize_llm_score(value)
            for label, value in scores.items()
            if str(label) in {item.value for item in IntentCategory}
        }
        if not scores and intent != IntentCategory.OTHER:
            scores[intent.value] = confidence
        return {
            "status": "ok",
            "intent": intent,
            "confidence": confidence,
            "scores": scores,
            "reasoning": str(data.get("reasoning", "")),
        }

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        if self._embedding_service is None:
            return {
                "status": "failed",
                "scores": {},
                "raw_cosine_scores": {},
                "top_templates": [],
                "raw_top_templates": [],
                "failed": True,
            }
        try:
            await self._load_template_embeddings()
            query_vector = await self._encode_query(message)
            scores: Dict[str, float] = {}
            raw_cosine_scores: Dict[str, float] = {}
            top_templates: List[Dict[str, Any]] = []
            raw_top_templates: List[Dict[str, Any]] = []
            for label, vectors in self._tpl_embeddings.items():
                ranked = sorted(
                    (
                        _cosine(query_vector, vector),
                        template,
                    )
                    for template, vector in zip(self._templates[label], vectors)
                )
                selected = ranked[-self.embedding_top_n :]
                raw_cosine_scores[label] = sum(score for score, _ in selected) / len(selected)
                scores[label] = normalize_embedding_score(raw_cosine_scores[label])
                top_templates.extend(
                    {
                        "intent": label,
                        "template": template,
                        "score": round(normalize_embedding_score(score), 6),
                        "raw_cosine": round(score, 6),
                    }
                    for score, template in selected
                )
                raw_top_templates.extend(
                    {
                        "intent": label,
                        "template": template,
                        "raw_cosine": round(score, 6),
                    }
                    for score, template in selected
                )
            top_templates.sort(key=lambda item: item["score"], reverse=True)
            raw_top_templates.sort(key=lambda item: item["raw_cosine"], reverse=True)
            return {
                "status": "ok",
                "scores": scores,
                "raw_cosine_scores": {
                    label: round(score, 6) for label, score in raw_cosine_scores.items()
                },
                "top_templates": top_templates[:10],
                "raw_top_templates": raw_top_templates[:10],
            }
        except EmbeddingServiceError as exc:
            logger.warning("Embedding recognition unavailable: %s", exc)
            return {
                "status": "failed",
                "scores": {},
                "raw_cosine_scores": {},
                "top_templates": [],
                "raw_top_templates": [],
                "failed": True,
            }
        except Exception as exc:
            logger.warning("Embedding recognition failed: %s", exc)
            return {
                "status": "failed",
                "scores": {},
                "raw_cosine_scores": {},
                "top_templates": [],
                "raw_top_templates": [],
                "failed": True,
            }

    async def _load_template_embeddings(self) -> None:
        missing = [label for label in self._templates if label not in self._tpl_embeddings]
        if not missing:
            return
        labels: List[str] = []
        texts: List[str] = []
        for label in missing:
            for template in self._templates[label]:
                labels.append(label)
                texts.append(template)
        vectors = await self._encode_batch(texts)
        if len(vectors) != len(texts):
            raise EmbeddingServiceError("Embedding service returned an unexpected batch length")
        index = 0
        for label in missing:
            count = len(self._templates[label])
            self._tpl_embeddings[label] = vectors[index : index + count]
            index += count

    async def _encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if hasattr(self._embedding_service, "aencode_batch"):
            return await self._embedding_service.aencode_batch(texts)
        return await asyncio.to_thread(self._embedding_service.encode_batch, texts)

    async def _encode_query(self, text: str) -> List[float]:
        if hasattr(self._embedding_service, "aencode_query"):
            return await self._embedding_service.aencode_query(text)
        return await asyncio.to_thread(self._embedding_service.encode_query, text)

    def _rule_recognize(self, message: str) -> Dict[str, Any]:
        scores: Dict[str, float] = {}
        hits: List[Dict[str, str]] = []
        normalized = message.lower()
        for intent, pattern, label in _RULES:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                current = scores.get(intent.value, 0.0)
                increment = 0.80 if intent in {
                    IntentCategory.HUMAN_HANDOFF,
                    IntentCategory.TECHNICAL_LOGIN,
                    IntentCategory.TECHNICAL_CRASH,
                } else 0.75 if intent in {
                    IntentCategory.REFUND,
                    IntentCategory.INVOICE,
                    IntentCategory.PAYMENT_ISSUE,
                    IntentCategory.ACCOUNT_SECURITY,
                } else 0.65
                scores[intent.value] = min(1.0, current + increment)
                hits.append({"intent": intent.value, "rule": label})
        scores = {label: normalize_rule_score(score) for label, score in scores.items()}
        return {
            "status": "ok",
            "scores": scores,
            "hits": hits,
            "high_precision_signals": hits,
        }

    def _fuse(
        self,
        llm: Dict[str, Any],
        embedding: Dict[str, Any],
        rules: Dict[str, Any],
        message: str = "",
    ) -> Dict[str, Any]:
        results = {"llm": llm, "embedding": embedding, "rules": rules}
        active_weights = {
            name: self.weights[name]
            for name in self._active_sources
            if results[name].get("status") == "ok" and self.weights[name] > 0
        }
        weight_total = sum(active_weights.values())
        labels = [item.value for item in IntentCategory]
        fused = {label: 0.0 for label in labels}
        if weight_total:
            for name, weight in active_weights.items():
                factor = weight / weight_total
                for label, value in results[name].get("scores", {}).items():
                    if label in fused:
                        fused[label] += factor * clamp01(value)
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        top1_label, top1 = ordered[0]
        _, top2 = ordered[1]
        margin = max(0.0, top1 - top2)
        gated = (
            IntentCategory(top1_label)
            if top1_label != IntentCategory.OTHER.value
            and top1 >= self.threshold
            and margin >= self.margin_threshold
            else IntentCategory.OTHER
        )
        exceptional_override = self._exceptional_override(message, rules)
        if exceptional_override:
            gated = IntentCategory(exceptional_override["intent"])
        confidence = 1.0 if exceptional_override else top1
        source_scores = {
            "llm": {
                "intent": getattr(llm.get("intent"), "value", llm.get("intent")),
                "score": normalize_llm_score(llm.get("confidence", 0.0)),
                "status": llm.get("status", "disabled"),
            },
            "embedding": {
                "intent_scores": embedding.get("scores", {}),
                "raw_cosine_scores": embedding.get("raw_cosine_scores", {}),
                "top_templates": embedding.get("top_templates", []),
                "raw_top_templates": embedding.get("raw_top_templates", []),
                "status": embedding.get("status", "disabled"),
            },
            "rules": {
                "intent_scores": rules.get("scores", {}),
                "hits": rules.get("hits", []),
                "high_precision_signals": rules.get("high_precision_signals", rules.get("hits", [])),
                "exceptional_overrides": [exceptional_override] if exceptional_override else [],
                "status": rules.get("status", "disabled"),
            },
            "fusion": {
                "weights": active_weights,
                "scores": {label: round(score, 6) for label, score in ordered[:10]},
            },
        }
        return {
            "intent": gated,
            "confidence": round(confidence, 6),
            "top1_score": round(top1, 6),
            "top2_score": round(top2, 6),
            "margin": round(margin, 6),
            "top_candidates": [
                {"intent": label, "score": round(score, 6)}
                for label, score in ordered[:5]
            ],
            "source_scores": source_scores,
        }

    @staticmethod
    def _exceptional_override(
        message: str,
        rules: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """Return only explicit safety/handoff overrides.

        Ordinary refund, invoice, payment, 401, and 500 signals remain
        routing evidence; they never override the LLM+BGE semantic result.
        """
        labels = {str(hit.get("intent")) for hit in rules.get("hits", [])}
        if "human_handoff" in labels:
            return {"intent": IntentCategory.HUMAN_HANDOFF.value, "reason": "explicit_human_handoff"}
        lower = (message or "").lower()
        security_signal = any(
            keyword in lower
            for keyword in ("账号被盗", "账户被盗", "账号遭盗", "未经授权", "盗用")
        ) or "account_security" in labels
        urgent = any(keyword in lower for keyword in _URGENCY_KEYWORDS[UrgencyLevel.CRITICAL])
        if urgent and security_signal:
            return {"intent": IntentCategory.ESCALATION.value, "reason": "critical_security_event"}
        return None

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        order_ids = re.findall(
            r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})",
            message,
            flags=re.IGNORECASE,
        )
        dates = re.findall(
            r"(今天|明天|昨天|前天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)",
            message,
        )
        amounts = re.findall(
            r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))",
            message,
            flags=re.IGNORECASE,
        )
        error_codes = re.findall(
            r"(?:错误码|error\s*code|状态码|报错)\s*[:：#]?\s*([45]\d{2}|[A-Z]{1,5}-?\d{2,5})"
            r"|(?<!\d)([45]\d{2})(?!\d)",
            message,
            flags=re.IGNORECASE,
        )
        flattened_codes = [first or second for first, second in error_codes]
        return {
            "order_id": self._unique(order_ids),
            "product": [],
            "date": self._unique(dates),
            "amount": self._unique(amounts),
            "error_code": self._unique(flattened_codes),
        }

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        lower = message.lower()
        for level, keywords in _URGENCY_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                return level
        if intent in {IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF}:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _load_templates(self) -> Dict[str, List[str]]:
        try:
            payload = json.loads(self._templates_path.read_text(encoding="utf-8"))
            raw = payload.get("templates", payload)
            if not isinstance(raw, dict):
                raise ValueError("templates must be a JSON object")
            allowed = {item.value for item in IntentCategory}
            templates = {
                str(label): [str(text) for text in values if str(text).strip()]
                for label, values in raw.items()
                if str(label) in allowed and isinstance(values, list)
            }
            missing = allowed - set(templates)
            if missing:
                raise ValueError(f"template file missing labels: {sorted(missing)}")
            return templates
        except Exception as exc:
            raise RuntimeError(f"Unable to load intent templates from {self._templates_path}: {exc}") from exc

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]]) -> str:
        payload: Dict[str, Any] = {
            "message": self._clean_text(message)[:500],
            "mode": self.mode,
            "threshold": self.threshold,
            "margin": self.margin_threshold,
        }
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", ""))[:300],
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: Sequence[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }

    @property
    def template_stats(self) -> Dict[str, Any]:
        counts = {label: len(values) for label, values in self._templates.items()}
        return {
            "intent_count": len(counts),
            "template_count": sum(counts.values()),
            "counts": counts,
            "embedding_cache_loaded": bool(self._tpl_embeddings),
        }


def _configured_weight(value: Optional[float], env_name: str, default: float) -> float:
    raw = value if value is not None else os.getenv(env_name, str(default))
    return max(0.0, float(raw))


def _sources_for_mode(mode: str) -> set[str]:
    mapping = {
        "rules_only": {"rules"},
        "embedding_only": {"embedding"},
        "llm_only": {"llm"},
        "llm_embedding": {"llm", "embedding"},
        "llm_embedding_rules": {"llm", "embedding", "rules"},
        "full": {"llm", "embedding", "rules"},
    }
    if mode not in mapping:
        raise ValueError(f"Unsupported intent recognition mode: {mode}")
    return mapping[mode]
