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
