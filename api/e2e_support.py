"""Deterministic provider support used only when E2E_TEST_MODE=1."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Iterable


def _last_user_text(messages: Iterable[dict[str, Any]]) -> str:
    values = [str(item.get("content", "")) for item in messages if item.get("role") == "user"]
    return values[-1] if values else ""


def _intent_for(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("转人工", "人工客服", "真人客服")):
        return "human_handoff"
    if "401" in lowered or "无法登录" in lowered or "登录不上" in lowered:
        return "technical_login"
    if any(token in lowered for token in ("重复扣款", "扣了两次", "支付失败")):
        return "payment_issue"
    if "退款" in lowered or "退货" in lowered:
        return "refund"
    if "发票" in lowered or "开票" in lowered:
        return "invoice"
    if any(token in lowered for token in ("发货", "物流", "快递")):
        return "logistics"
    if any(token in lowered for token in ("你好", "嗨", "hello", "hi")):
        return "greeting"
    return "other"


class FakeAnthropicClient:
    """Small async client matching the ``messages.create`` surface used here."""

    def __init__(self) -> None:
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        prompt = "\n".join(str(item.get("content", "")) for item in kwargs.get("messages", []))
        system = str(kwargs.get("system", ""))
        if "intent、confidence、reasoning" in prompt or "客服意图分类器" in prompt:
            user_text = _last_user_text(kwargs.get("messages", []))
            if "用户消息:" in user_text:
                user_text = user_text.rsplit("用户消息:", 1)[-1]
            intent = _intent_for(user_text)
            return _text_response(json.dumps({"intent": intent, "confidence": 0.98, "reasoning": "deterministic E2E provider"}, ensure_ascii=False))
        if "查询改写" in prompt or ("改写" in prompt and "数组" in prompt):
            original = prompt.rsplit("原始问题:", 1)[-1].strip().splitlines()[0]
            return _text_response(json.dumps([original], ensure_ascii=False))
        if "提炼用户偏好" in prompt:
            return _text_response('{"preferences": [], "entities": {}}')
        if "候选答案" in prompt or ("相关性" in prompt and "数组" in prompt):
            return _text_response("[]")
        if "账单" in system or "退款" in system or "发票" in system:
            return _text_response("这是一个可预测的 Billing 测试回复，真实 Agent 链路已执行。")
        if "技术支持" in system or "故障" in system:
            return _text_response("这是一个可预测的 Technical 测试回复，真实 Agent 链路已执行。")
        return _text_response("这是一个可预测的 General 测试回复，真实 Agent 链路已执行。")


def _text_response(text: str) -> Any:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")


def is_e2e_test_mode() -> bool:
    return os.getenv("E2E_TEST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
