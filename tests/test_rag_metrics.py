from rag.metrics import ndcg_at_k, recall_at_k, reciprocal_rank, summarize_cases


def test_rank_metrics_use_gold_ids():
    ranked = ["c2", "c1", "c3"]
    assert recall_at_k(ranked, ["c1"], 2) == 1.0
    assert reciprocal_rank(ranked, ["c1"], 10) == 0.5
    assert ndcg_at_k(ranked, ["c1"], 10) > 0


def test_no_answer_is_reported_separately():
    report = summarize_cases([{"answerable": False, "gold_child_ids": [], "gold_parent_ids": []}], [{"child_ids": ["c1"], "parent_ids": ["p1"]}])
    assert report["no_answer_false_positive_rate"] == 1.0
