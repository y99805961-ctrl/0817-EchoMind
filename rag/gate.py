"""Stable Intent-v2-compatible gate for deciding whether RAG should run."""
from __future__ import annotations

from typing import Any


SKIP_INTENTS = {"greeting", "feedback", "human_handoff", "other"}
KNOWLEDGE_INTENTS = {
    "order_status", "logistics", "refund", "invoice", "payment_issue",
    "account_security", "technical_login", "technical_crash", "billing",
    "account", "technical", "query",
}


def should_use_knowledge(message: str, intent: Any = None) -> bool:
    value = getattr(intent, "value", intent)
    if value in SKIP_INTENTS:
        return False
    if value in KNOWLEDGE_INTENTS:
        return True
    message = (message or "").strip().lower()
    if not message:
        return False
    if message in {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}:
        return False
    keywords = [
        "退款", "退货", "换货", "订单", "物流", "配送", "发票", "扣款", "支付", "账单", "订阅",
        "登录", "报错", "错误", "崩溃", "会员", "积分", "账户", "密码", "地址", "人工", "升级",
        "refund", "order", "invoice", "payment", "error", "login", "401", "403", "500",
    ]
    if value in {"complaint", "request", "escalation"}:
        return any(keyword in message for keyword in keywords)
    return len(message) >= 4 or any(keyword in message for keyword in keywords)
