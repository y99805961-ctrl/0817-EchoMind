import json
from pathlib import Path

from core.intent_recognizer import IntentRecognizer
from evaluation.intent_eval import load_json, template_leakage_check


ROOT = Path(__file__).resolve().parents[1]


def test_template_dataset_has_all_labels_and_expected_counts():
    recognizer = IntentRecognizer(mode="rules_only")
    stats = recognizer.template_stats
    assert stats["intent_count"] == 19
    assert stats["template_count"] == 132
    assert stats["counts"]["refund"] == 8
    assert stats["counts"]["order_status"] == 8


def test_templates_do_not_copy_frozen_benchmark_queries():
    benchmark = load_json(ROOT / "data/eval/intent/intent_benchmark_76.json")
    result = template_leakage_check(benchmark)
    assert result["passed"] is True
    assert result["exact_match_count"] == 0
