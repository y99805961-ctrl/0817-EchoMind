from core.intent_recognizer import IntentCategory, IntentRecognizer


def test_low_top1_is_gated_to_other():
    recognizer = IntentRecognizer(mode="llm_only", confidence_threshold=0.8, margin_threshold=0.05)
    fused = recognizer._fuse(
        {"status": "ok", "scores": {"refund": 0.6}, "confidence": 0.6},
        {"status": "disabled", "scores": {}},
        {"status": "disabled", "scores": {}},
    )
    assert fused["intent"] == IntentCategory.OTHER
    assert fused["top1_score"] == 0.6


def test_small_top1_top2_margin_is_gated_to_other():
    recognizer = IntentRecognizer(mode="llm_only", confidence_threshold=0.5, margin_threshold=0.1)
    fused = recognizer._fuse(
        {"status": "ok", "scores": {"refund": 0.7, "invoice": 0.66}, "confidence": 0.7},
        {"status": "disabled", "scores": {}},
        {"status": "disabled", "scores": {}},
    )
    assert fused["intent"] == IntentCategory.OTHER
    assert fused["margin"] == 0.04
