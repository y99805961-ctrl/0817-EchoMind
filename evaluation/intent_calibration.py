"""Intent semantic-score calibration experiments.

This module is deliberately separate from the production fusion path.  It
reuses the frozen templates and 76-case benchmark, measures the same BGE-M3
template scores in three score spaces, and never changes the configured
three-way fusion weights.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from evaluation.intent_eval import (
    INTENT_BENCHMARK,
    RESULT_DIR,
    TEMPLATE_PATH,
    _classification_metrics,
    load_json,
    percentile,
    template_leakage_check,
    _make_recognizer,
)
from core.intent_recognizer import IntentRecognizer, _cosine, normalize_embedding_score


DEFAULT_OUTPUT = RESULT_DIR / "semantic_calibration.json"
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_MARGIN_THRESHOLD = 0.05
DEFAULT_RAW_MARGIN_THRESHOLD = 0.05
DEFAULT_TEMPERATURE = 0.05
DEFAULT_TEMPERATURE_MARGIN_THRESHOLD = 0.05


def _top_n_mean(values: Sequence[float], top_n: int) -> float:
    selected = sorted(values)[-max(1, top_n) :]
    return statistics.mean(selected) if selected else 0.0


def _raw_label_scores(
    recognizer: IntentRecognizer,
    query_vector: Sequence[float],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Return raw cosine label scores and the selected template diagnostics."""
    scores: Dict[str, float] = {}
    templates: List[Dict[str, Any]] = []
    for label, vectors in recognizer._tpl_embeddings.items():
        ranked = sorted(
            (
                _cosine(query_vector, vector),
                template,
            )
            for template, vector in zip(recognizer._templates[label], vectors)
        )
        selected = ranked[-recognizer.embedding_top_n :]
        scores[label] = _top_n_mean([score for score, _ in selected], recognizer.embedding_top_n)
        templates.extend(
            {
                "intent": label,
                "template": template,
                "raw_cosine": round(score, 6),
            }
            for score, template in selected
        )
    templates.sort(key=lambda item: (-item["raw_cosine"], item["intent"], item["template"]))
    return scores, templates[:10]


def _softmax(scores: Mapping[str, float], temperature: float) -> Dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if not scores:
        return {}
    scaled = {label: float(score) / temperature for label, score in scores.items()}
    maximum = max(scaled.values())
    exponentials = {label: math.exp(value - maximum) for label, value in scaled.items()}
    total = sum(exponentials.values())
    return {label: value / total for label, value in exponentials.items()}


def _rank(scores: Mapping[str, float]) -> Tuple[str, float, float, float, List[Dict[str, Any]]]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return "other", 0.0, 0.0, 0.0, []
    top1_label, top1 = ordered[0]
    top2 = ordered[1][1] if len(ordered) > 1 else 0.0
    return (
        top1_label,
        top1,
        top2,
        max(0.0, top1 - top2),
        [{"intent": label, "score": round(score, 6)} for label, score in ordered[:5]],
    )


def _calibrated_prediction(
    scores: Mapping[str, float],
    confidence_threshold: float,
    margin_threshold: float,
) -> Dict[str, Any]:
    top1_label, top1, top2, margin, candidates = _rank(scores)
    accepted = (
        top1_label != "other"
        and top1 >= confidence_threshold
        and margin >= margin_threshold
    )
    return {
        "ungated_top1": top1_label,
        "ungated_top1_score": round(top1, 6),
        "ungated_top2_score": round(top2, 6),
        "ungated_margin": round(margin, 6),
        "predicted_intent": top1_label if accepted else "other",
        "gate_rejected": bool(top1_label != "other" and not accepted),
        "top_candidates": candidates,
    }


def _metrics_for_calibration(
    rows: Sequence[Dict[str, Any]],
    calibration_name: str,
) -> Dict[str, Any]:
    expected = [row["expected_intent"] for row in rows]
    predictions = [row[calibration_name]["predicted_intent"] for row in rows]
    ungated = [row[calibration_name]["ungated_top1"] for row in rows]
    classification = _classification_metrics(expected, predictions)
    return {
        "accuracy": classification["accuracy"],
        "macro_f1": classification["macro_f1"],
        "other_rejection_count": sum(prediction == "other" for prediction in predictions),
        "gate_rejection_count": sum(row[calibration_name]["gate_rejected"] for row in rows),
        "ungated_top1_accuracy": round(
            sum(gold == prediction for gold, prediction in zip(expected, ungated)) / len(rows),
            6,
        ) if rows else 0.0,
        # Kept as an explicit alias because this is the ranking metric requested
        # by the experiment brief.
        "top1_ranking_accuracy": round(
            sum(gold == prediction for gold, prediction in zip(expected, ungated)) / len(rows),
            6,
        ) if rows else 0.0,
        "confusion_matrix": classification["confusion_matrix"],
    }


async def run_calibration(
    output_path: Path = DEFAULT_OUTPUT,
    raw_margin_threshold: float = DEFAULT_RAW_MARGIN_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
    temperature_margin_threshold: float = DEFAULT_TEMPERATURE_MARGIN_THRESHOLD,
) -> Dict[str, Any]:
    """Run all calibration variants on the unchanged 76-case benchmark."""
    benchmark = load_json(INTENT_BENCHMARK)
    recognizer = _make_recognizer("embedding_only")

    # This intentionally measures first-use model loading + template encoding
    # + one query.  All online latency samples below happen after this warm-up.
    cold_started = time.perf_counter()
    await recognizer._load_template_embeddings()
    await recognizer._encode_query(benchmark[0]["query"])
    cold_start_ms = (time.perf_counter() - cold_started) * 1000

    rows: List[Dict[str, Any]] = []
    warm_latencies: List[float] = []
    for case in benchmark:
        started = time.perf_counter()
        query_vector = await recognizer._encode_query(case["query"])
        raw_scores, raw_templates = _raw_label_scores(recognizer, query_vector)
        warm_latencies.append((time.perf_counter() - started) * 1000)

        current_scores = {
            label: normalize_embedding_score(score)
            for label, score in raw_scores.items()
        }
        raw_predictions = _calibrated_prediction(
            raw_scores,
            confidence_threshold=0.0,
            margin_threshold=raw_margin_threshold,
        )
        current_predictions = _calibrated_prediction(
            current_scores,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            margin_threshold=DEFAULT_MARGIN_THRESHOLD,
        )
        temperature_scores = _softmax(raw_scores, temperature)
        temperature_predictions = _calibrated_prediction(
            temperature_scores,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            margin_threshold=temperature_margin_threshold,
        )
        rows.append({
            "id": case.get("id"),
            "query": case["query"],
            "expected_intent": case["expected_intent"],
            "raw_cosine_scores": {label: round(score, 6) for label, score in raw_scores.items()},
            "raw_top_templates": raw_templates,
            "current": current_predictions,
            "raw_cosine_margin": raw_predictions,
            "temperature_softmax": temperature_predictions,
        })

    metrics = {
        name: _metrics_for_calibration(rows, name)
        for name in ("current", "raw_cosine_margin", "temperature_softmax")
    }
    payload = {
        "status": "OK",
        "benchmark": str(INTENT_BENCHMARK),
        "template_path": str(TEMPLATE_PATH),
        "total": len(rows),
        "exploratory_only": True,
        "note": (
            "Calibration thresholds and temperature are exploratory on the frozen 76-case "
            "benchmark; no template or production three-way weight was changed."
        ),
        "configuration": {
            "current": {
                "score_space": "(raw_cosine + 1) / 2",
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "margin_threshold": DEFAULT_MARGIN_THRESHOLD,
                "exploratory": False,
            },
            "raw_cosine_margin": {
                "score_space": "raw cosine",
                "confidence_threshold": 0.0,
                "margin_threshold": raw_margin_threshold,
                "exploratory": True,
                "note": (
                    "0.0 confidence is affine-equivalent to current normalized 0.50; "
                    "the raw-space margin is intentionally exploratory."
                ),
            },
            "temperature_softmax": {
                "score_space": "softmax(raw cosine / temperature)",
                "temperature": temperature,
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "margin_threshold": temperature_margin_threshold,
                "exploratory": True,
            },
            "production_fusion_weights_unchanged": {
                "llm": 0.45,
                "embedding": 0.35,
                "rules": 0.20,
            },
        },
        "metrics": metrics,
        "latency": {
            "cold_start_ms": round(cold_start_ms, 6),
            "warm_mean_ms": round(statistics.mean(warm_latencies), 6),
            "warm_p50_ms": round(percentile(warm_latencies, 0.50), 6),
            "warm_p95_ms": round(percentile(warm_latencies, 0.95), 6),
            "sample_count": len(warm_latencies),
            "definition": "BGE query encode + template cosine scoring after model/template warm-up",
        },
        "template_leakage": template_leakage_check(benchmark, TEMPLATE_PATH),
        "template_stats": recognizer.template_stats,
        "cases": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Intent semantic score calibration")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--raw-margin-threshold", type=float, default=DEFAULT_RAW_MARGIN_THRESHOLD)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--temperature-margin-threshold",
        type=float,
        default=DEFAULT_TEMPERATURE_MARGIN_THRESHOLD,
    )
    return parser


async def main_async(args: argparse.Namespace) -> None:
    payload = await run_calibration(
        Path(args.output),
        raw_margin_threshold=args.raw_margin_threshold,
        temperature=args.temperature,
        temperature_margin_threshold=args.temperature_margin_threshold,
    )
    print(json.dumps({
        "status": payload["status"],
        "total": payload["total"],
        "exploratory_only": payload["exploratory_only"],
        "metrics": payload["metrics"],
        "latency": payload["latency"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main_async(build_parser().parse_args()))
