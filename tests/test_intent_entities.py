import asyncio

from core.intent_recognizer import IntentRecognizer


def test_entity_extraction_preserves_existing_shape():
    recognizer = IntentRecognizer(mode="rules_only")
    result = asyncio.run(
        recognizer.recognize("订单号 ORD7788 扣了我 299 元，昨天开始一直报 401")
    )
    assert result.entities["order_id"] == ["ORD7788"]
    assert result.entities["amount"] == ["299 元"]
    assert result.entities["date"] == ["昨天"]
    assert result.entities["error_code"] == ["401"]
    assert set(result.entities) == {"order_id", "product", "date", "amount", "error_code"}
