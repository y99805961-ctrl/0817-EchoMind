"""Gold retrieval metrics for Child and final Parent rankings."""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Sequence, Set


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return 1.0 if gold.intersection(ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int = 10) -> float:
    gold = set(gold_ids)
    for index, value in enumerate(ranked_ids[:k], start=1):
        if value in gold:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int = 10) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(index + 2) for index, value in enumerate(ranked_ids[:k]) if value in gold)
    ideal_len = min(len(gold), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_len))
    return dcg / idcg if idcg else 0.0


def no_answer_false_positive(answerable: bool, selected_parent_ids: Sequence[str]) -> float:
    return 1.0 if not answerable and bool(selected_parent_ids) else 0.0


def summarize_cases(cases: Sequence[Dict[str, Any]], results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, List[float]] = {}
    no_answer_flags: List[float] = []
    for case, result in zip(cases, results):
        child_ids = list(result.get("child_ids", []))
        parent_ids = list(result.get("parent_ids", []))
        gold_child_ids = case.get("gold_child_ids", [])
        gold_parent_ids = case.get("gold_parent_ids", [])
        for k in (3, 5, 10):
            metrics.setdefault(f"child_recall@{k}", []).append(recall_at_k(child_ids, gold_child_ids, k))
            metrics.setdefault(f"parent_recall@{k}", []).append(recall_at_k(parent_ids, gold_parent_ids, k))
        metrics.setdefault("mrr@10", []).append(reciprocal_rank(child_ids, gold_child_ids, 10))
        metrics.setdefault("ndcg@10", []).append(ndcg_at_k(child_ids, gold_child_ids, 10))
        no_answer_flags.append(no_answer_false_positive(bool(case.get("answerable", True)), parent_ids))
    output = {key: round(statistics.mean(values), 4) if values else 0.0 for key, values in metrics.items()}
    output["no_answer_false_positive_rate"] = round(statistics.mean(no_answer_flags), 4) if no_answer_flags else 0.0
    output["cases"] = len(cases)
    return output


def latency_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "mean": round(statistics.mean(ordered), 3),
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
    }


def _percentile(values: Sequence[float], percentile: int) -> float:
    position = (len(values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight
