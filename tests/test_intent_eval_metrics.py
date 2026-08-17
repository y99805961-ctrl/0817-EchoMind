import pytest

from evaluation.intent_eval import _classification_metrics, _entity_metrics


def test_classification_metrics_and_confusion_are_machine_readable():
    result = _classification_metrics(["refund", "refund", "invoice"], ["refund", "invoice", "invoice"])
    assert result["accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert result["per_class"]["refund"]["support"] == 2
    assert result["confusion_matrix"]["refund"]["invoice"] == 1


def test_entity_metrics_are_micro_averaged():
    result = _entity_metrics([
        {"gold": {"order_id": ["A1"], "amount": ["9 元"]}, "pred": {"order_id": ["A1"], "amount": []}},
        {"gold": {"order_id": []}, "pred": {"order_id": ["B2"]}},
    ])
    assert result["entity_precision"] == 0.5
    assert result["entity_recall"] == 0.5
    assert result["entity_f1"] == 0.5
