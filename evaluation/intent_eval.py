"""Intent and routing benchmark runner.

This module reads the frozen benchmark files but never writes to them.  When a
required external dependency is unavailable, the result is explicitly marked
``NOT_RUN`` and no metric is manufactured from fallback failures.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from agents.agent_orchestrator import AgentOrchestrator, Request
from core.intent_recognizer import IntentCategory, IntentRecognizer

ROOT = Path(__file__).resolve().parents[1]
INTENT_BENCHMARK = ROOT / "data" / "eval" / "intent" / "intent_benchmark_76.json"
ROUTING_BENCHMARK = ROOT / "data" / "eval" / "intent" / "routing_benchmark_12.json"
TEMPLATE_PATH = ROOT / "data" / "intent" / "templates.json"
RESULT_DIR = ROOT / "data" / "eval" / "results" / "intent"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _classification_metrics(expected: Sequence[str], predicted: Sequence[str]) -> Dict[str, Any]:
    labels = sorted(set(expected) | set(predicted))
    per_class: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        tp = sum(p == label and g == label for g, p in zip(expected, predicted))
        fp = sum(p == label and g != label for g, p in zip(expected, predicted))
        fn = sum(p != label and g == label for g, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(_f1(precision, recall), 6),
            "support": tp + fn,
        }
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gold, pred in zip(expected, predicted):
        confusion[gold][pred] += 1
    confusion_matrix = {gold: dict(sorted(row.items())) for gold, row in sorted(confusion.items())}
    confusion_pairs = [
        {"gold": gold, "predicted": pred, "count": count}
        for gold, row in confusion_matrix.items()
        for pred, count in row.items()
        if gold != pred and count > 0
    ]
    confusion_pairs.sort(key=lambda item: (-item["count"], item["gold"], item["predicted"]))
    macro_precision = statistics.mean(v["precision"] for v in per_class.values()) if per_class else 0.0
    macro_recall = statistics.mean(v["recall"] for v in per_class.values()) if per_class else 0.0
    macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0
    accuracy = sum(g == p for g, p in zip(expected, predicted)) / len(expected) if expected else 0.0
    return {
        "accuracy": round(accuracy, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "confusion_summary": confusion_pairs,
    }


def _entity_metrics(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fields = sorted({field for case in cases for field in set(case["gold"]) | set(case["pred"])})
    totals = {field: {"tp": 0, "fp": 0, "fn": 0} for field in fields}
    for case in cases:
        for field in fields:
            gold = set(case["gold"].get(field, []))
            pred = set(case["pred"].get(field, []))
            totals[field]["tp"] += len(gold & pred)
            totals[field]["fp"] += len(pred - gold)
            totals[field]["fn"] += len(gold - pred)

    def summarize(total: Dict[str, int]) -> Dict[str, float]:
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
        recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        return {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(_f1(precision, recall), 6),
        }

    aggregate = {key: sum(value[key] for value in totals.values()) for key in ("tp", "fp", "fn")}
    return {
        "entity_precision": summarize(aggregate)["precision"],
        "entity_recall": summarize(aggregate)["recall"],
        "entity_f1": summarize(aggregate)["f1"],
        "per_field": {field: summarize(total) for field, total in totals.items()},
        "counts": aggregate,
    }


def template_leakage_check(benchmark: Sequence[Dict[str, Any]], template_path: Path = TEMPLATE_PATH) -> Dict[str, Any]:
    payload = load_json(template_path)
    templates = payload.get("templates", payload)
    benchmark_queries = {str(case.get("query", "")).strip() for case in benchmark}
    template_queries = {
        str(text).strip()
        for values in templates.values()
        if isinstance(values, list)
        for text in values
    }
    exact = sorted(query for query in benchmark_queries & template_queries if query)
    return {"exact_match_count": len(exact), "exact_matches": exact, "passed": not exact}


async def evaluate_intent(
    recognizer: IntentRecognizer,
    benchmark_path: Path = INTENT_BENCHMARK,
) -> Dict[str, Any]:
    benchmark = load_json(benchmark_path)
    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    required_sources = recognizer._active_sources
    unavailable: Counter[str] = Counter()
    for case in benchmark:
        started = time.perf_counter()
        result = await recognizer.recognize(case["query"], history=case.get("history"))
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        for source in required_sources:
            status = result.source_scores.get(source, {}).get("status", "disabled")
            if status != "ok":
                unavailable[f"{source}:{status}"] += 1
        rows.append({
            "id": case.get("id"),
            "query": case["query"],
            "expected_intent": case["expected_intent"],
            "predicted_intent": result.intent.value,
            "expected_group": case.get("expected_group"),
            "predicted_group": result.intent_group,
            "expected_entities": case.get("entities", {}),
            "predicted_entities": result.entities,
            "confidence": result.confidence,
            "top_candidates": result.top_candidates,
            "margin": result.margin,
            "latency_ms": round(latency_ms, 6),
            "source_scores": result.source_scores,
        })

    if unavailable:
        return {
            "status": "NOT_RUN",
            "reason": "required source unavailable or failed",
            "unavailable_sources": dict(unavailable),
            "total": len(benchmark),
            "completed": len(rows),
            "template_leakage": template_leakage_check(benchmark),
            "template_stats": recognizer.template_stats,
            "cases": rows,
        }

    classification = _classification_metrics(
        [row["expected_intent"] for row in rows],
        [row["predicted_intent"] for row in rows],
    )
    entity = _entity_metrics([
        {"gold": row["expected_entities"], "pred": row["predicted_entities"]}
        for row in rows
    ])
    return {
        "status": "OK",
        "total": len(rows),
        **classification,
        **entity,
        "latency": {
            "mean_ms": round(statistics.mean(latencies), 6) if latencies else None,
            "p50_ms": round(percentile(latencies, 0.50), 6) if latencies else None,
            "p95_ms": round(percentile(latencies, 0.95), 6) if latencies else None,
        },
        "template_leakage": template_leakage_check(benchmark),
        "template_stats": recognizer.template_stats,
        "cases": rows,
    }


def _make_recognizer(
    mode: str,
    confidence_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
) -> IntentRecognizer:
    load_dotenv(ROOT / ".env")
    return IntentRecognizer(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        mode=mode,
    )


async def run_ablation(output_path: Path, skip_llm: bool = False) -> Dict[str, Any]:
    methods = [
        ("rules_only", "Rules Only"),
        ("embedding_only", "BGE-M3 Only"),
        ("llm_only", "LLM Only"),
        ("llm_embedding", "LLM + BGE-M3"),
        ("llm_embedding_rules", "LLM + BGE-M3 + Rules"),
    ]
    results = []
    for mode, label in methods:
        if skip_llm and "llm" in mode:
            metrics = {
                "status": "NOT_RUN",
                "reason": "LLM smoke test returned HTTP 401; skipped to avoid repeated unauthenticated requests.",
            }
        else:
            metrics = await evaluate_intent(_make_recognizer(mode))
        row = {
            "method": mode,
            "label": label,
            "status": metrics.get("status"),
            "accuracy": metrics.get("accuracy", "NOT_RUN"),
            "macro_f1": metrics.get("macro_f1", "NOT_RUN"),
            "entity_f1": metrics.get("entity_f1", "NOT_RUN"),
            "mean_latency_ms": metrics.get("latency", {}).get("mean_ms", "NOT_RUN"),
            "p95_latency_ms": metrics.get("latency", {}).get("p95_ms", "NOT_RUN"),
            "reason": metrics.get("reason"),
            "unavailable_sources": metrics.get("unavailable_sources", {}),
        }
        results.append(row)
    payload = {"status": "OK", "benchmark": str(INTENT_BENCHMARK), "results": results}
    _write_json(output_path, payload)
    return payload


async def run_threshold_search(
    output_path: Path,
    mode: str = "embedding_only",
) -> Dict[str, Any]:
    """Run the requested confidence/margin grid without claiming test optimality."""
    rows = []
    for confidence in (0.40, 0.50, 0.60):
        for margin in (0.03, 0.05, 0.10):
            metrics = await evaluate_intent(
                _make_recognizer(
                    mode,
                    confidence_threshold=confidence,
                    margin_threshold=margin,
                )
            )
            rows.append({
                "confidence_threshold": confidence,
                "margin_threshold": margin,
                "status": metrics.get("status"),
                "accuracy": metrics.get("accuracy", "NOT_RUN"),
                "macro_f1": metrics.get("macro_f1", "NOT_RUN"),
                "entity_f1": metrics.get("entity_f1", "NOT_RUN"),
                "p95_latency_ms": metrics.get("latency", {}).get("p95_ms", "NOT_RUN"),
                "reason": metrics.get("reason"),
            })
    payload = {
        "status": "OK",
        "mode": mode,
        "exploratory_only": True,
        "note": "No development split was provided; do not treat this grid as test-optimal tuning.",
        "results": rows,
    }
    _write_json(output_path, payload)
    return payload


async def evaluate_routing(output_path: Path, intent_mode: str = "rules_only") -> Dict[str, Any]:
    benchmark = load_json(ROUTING_BENCHMARK)
    recognizer = _make_recognizer(intent_mode)
    orchestrator = AgentOrchestrator(api_key="routing-benchmark-placeholder")
    rows = []
    primary_correct = 0
    support_true = support_total = 0
    exact = 0
    for case in benchmark:
        intent_result = await recognizer.recognize(case["query"])
        req = Request(
            message=case["query"],
            user_id="routing-eval",
            conv_id=str(case.get("id", "routing")),
            intent=intent_result.intent,
            intent_group=intent_result.intent_group,
            urgency=intent_result.urgency,
            entities=intent_result.entities,
            intent_confidence=intent_result.confidence,
        )
        decision = orchestrator._route_decision(req)
        predicted_primary = decision.primary_agent.value
        predicted_support = sorted(agent.value for agent in decision.supporting_agents)
        expected_primary = case["expected_primary_domain"]
        expected_support = sorted(case.get("expected_supporting_domains", []))
        primary_correct += predicted_primary == expected_primary
        support_true += len(set(predicted_support) & set(expected_support))
        support_total += len(expected_support)
        exact += predicted_primary == expected_primary and predicted_support == expected_support
        rows.append({
            "id": case.get("id"),
            "expected_primary": expected_primary,
            "predicted_primary": predicted_primary,
            "expected_supporting": expected_support,
            "predicted_supporting": predicted_support,
            "intent": intent_result.intent.value,
            "routing_reason": decision.reason,
        })
    payload = {
        "status": "OK",
        "intent_mode": intent_mode,
        "total": len(rows),
        "primary_routing_accuracy": round(primary_correct / len(rows), 6) if rows else 0.0,
        "supporting_agent_recall": round(support_true / support_total, 6) if support_total else 1.0,
        "exact_match": round(exact / len(rows), 6) if rows else 0.0,
        "cases": rows,
    }
    _write_json(output_path, payload)
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    if args.ablation:
        payload = await run_ablation(
            Path(args.output or RESULT_DIR / "ablation.json"),
            skip_llm=args.skip_llm,
        )
    elif args.threshold_search:
        payload = await run_threshold_search(
            Path(args.output or RESULT_DIR / "threshold_search.json"),
            mode=args.threshold_mode,
        )
    elif args.routing:
        payload = await evaluate_routing(
            Path(args.output or RESULT_DIR / "routing_benchmark.json"),
            intent_mode=args.intent_mode,
        )
    else:
        metrics = await evaluate_intent(_make_recognizer(args.mode))
        _write_json(Path(args.output or RESULT_DIR / f"intent_{args.mode}.json"), metrics)
        payload = metrics
    print(json.dumps({key: value for key, value in payload.items() if key not in {"cases", "results"}}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen Intent/Routing benchmarks")
    parser.add_argument(
        "--mode",
        choices=["rules_only", "embedding_only", "llm_only", "llm_embedding", "llm_embedding_rules", "full"],
        default="llm_embedding_rules",
    )
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM modes after a failed smoke test")
    parser.add_argument("--threshold-search", action="store_true")
    parser.add_argument("--threshold-mode", default="embedding_only", choices=["rules_only", "embedding_only"])
    parser.add_argument("--routing", action="store_true")
    parser.add_argument("--intent-mode", default="rules_only")
    return parser


if __name__ == "__main__":
    asyncio.run(main_async(build_parser().parse_args()))
