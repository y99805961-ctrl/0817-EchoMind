from core.intent_recognizer import (
    IntentCategory,
    IntentRecognizer,
    normalize_embedding_score,
    normalize_llm_score,
    normalize_rule_score,
)


def test_source_normalization_is_bounded():
    assert normalize_llm_score(1.4) == 1.0
    assert normalize_rule_score(-0.2) == 0.0
    assert normalize_embedding_score(-1.0) == 0.0
    assert normalize_embedding_score(0.0) == 0.5
    assert normalize_embedding_score(1.0) == 1.0


def test_fusion_renormalizes_available_sources_and_keeps_diagnostics():
    recognizer = IntentRecognizer(mode="llm_embedding_rules")
    fused = recognizer._fuse(
        {"status": "failed", "scores": {}},
        {"status": "disabled", "scores": {}},
        {"status": "ok", "scores": {"refund": 0.8}, "hits": [{"rule": "refund"}]},
    )
    assert fused["intent"] == IntentCategory.REFUND
    assert fused["source_scores"]["rules"]["hits"]
    assert fused["source_scores"]["fusion"]["weights"] == {"rules": 0.2}


def test_production_mode_uses_llm_and_embedding_without_rule_weight():
    recognizer = IntentRecognizer(mode="llm_embedding")
    fused = recognizer._fuse(
        {"status": "ok", "scores": {"refund": 0.9}, "confidence": 0.9},
        {"status": "ok", "scores": {"refund": 0.8}},
        {"status": "ok", "scores": {"technical_crash": 1.0}, "hits": [{"intent": "technical_crash"}]},
        message="页面报500但我想退款",
    )
    assert fused["intent"] == IntentCategory.REFUND
    assert fused["source_scores"]["fusion"]["weights"] == {"llm": 0.45, "embedding": 0.35}


def test_only_explicit_handoff_is_an_intent_override():
    recognizer = IntentRecognizer(mode="llm_embedding")
    handoff = recognizer._fuse(
        {"status": "ok", "scores": {"refund": 0.9}, "confidence": 0.9},
        {"status": "ok", "scores": {"refund": 0.8}},
        {"status": "ok", "scores": {"human_handoff": 1.0}, "hits": [{"intent": "human_handoff"}]},
        message="请转人工处理退款",
    )
    ordinary_signal = recognizer._fuse(
        {"status": "ok", "scores": {"refund": 0.9}, "confidence": 0.9},
        {"status": "ok", "scores": {"refund": 0.8}},
        {"status": "ok", "scores": {"technical_crash": 1.0}, "hits": [{"intent": "technical_crash"}]},
        message="页面报500但我想退款",
    )
    assert handoff["intent"] == IntentCategory.HUMAN_HANDOFF
    assert ordinary_signal["intent"] == IntentCategory.REFUND
