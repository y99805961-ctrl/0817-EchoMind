import asyncio

from core.intent_recognizer import IntentRecognizer


def recognize(recognizer, message):
    return asyncio.run(recognizer.recognize(message))


def test_high_precision_rules_prefer_fine_grained_intents():
    recognizer = IntentRecognizer(mode="rules_only")
    assert recognize(recognizer, "退款一直不到账").intent.value == "refund"
    assert recognize(recognizer, "页面返回401").intent.value == "technical_login"
    assert recognize(recognizer, "App突然闪退").intent.value == "technical_crash"
    assert recognize(recognizer, "同一笔重复扣款").intent.value == "payment_issue"
    assert recognize(recognizer, "请转人工客服").intent.value == "human_handoff"
