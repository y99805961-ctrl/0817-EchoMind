import asyncio

import pytest

from core.intent_recognizer import IntentRecognizer
from evaluation.intent_calibration import _calibrated_prediction, _softmax


def test_softmax_temperature_preserves_ranking_and_normalizes():
    scores = _softmax({"refund": 0.8, "technical_crash": 0.5}, temperature=0.05)
    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores["refund"] > scores["technical_crash"]


def test_calibration_keeps_ungated_top1_when_margin_gate_rejects():
    result = _calibrated_prediction(
        {"refund": 0.70, "invoice": 0.66, "other": 0.20},
        confidence_threshold=0.50,
        margin_threshold=0.10,
    )
    assert result["ungated_top1"] == "refund"
    assert result["predicted_intent"] == "other"
    assert result["gate_rejected"] is True


def test_embedding_diagnostics_include_raw_cosine_scores():
    class FakeEmbeddingService:
        async def aencode_batch(self, texts):
            return [[1.0, 0.0] for _ in texts]

        async def aencode_query(self, text):
            return [1.0, 0.0]

    recognizer = IntentRecognizer(
        mode="embedding_only",
        embedding_service=FakeEmbeddingService(),
    )
    result = asyncio.run(recognizer._embedding_recognize("测试"))
    assert result["raw_cosine_scores"]["refund"] == pytest.approx(1.0)
    assert result["raw_top_templates"][0]["raw_cosine"] == pytest.approx(1.0)
