"""Reciprocal Rank Fusion over stable child IDs."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Sequence

from .models import FusedHit, RetrievalHit


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievalHit]],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    original_query_weight: float = 1.0,
    rewrite_query_weight: float = 1.0,
) -> List[FusedHit]:
    aggregate: "OrderedDict[str, FusedHit]" = OrderedDict()
    for results in result_lists:
        for hit in results:
            if not hit.child_id:
                continue
            if hit.child_id not in aggregate:
                aggregate[hit.child_id] = FusedHit(
                    child_id=hit.child_id,
                    parent_id=hit.parent_id,
                    content=hit.content,
                    source=hit.source,
                    metadata=dict(hit.metadata),
                )
            fused = aggregate[hit.child_id]
            retriever_weight = dense_weight if hit.retriever == "dense" else bm25_weight
            query_weight = original_query_weight if hit.query_source == "original" else rewrite_query_weight
            fused.rrf_score += retriever_weight * query_weight / (k + max(1, hit.rank))
            if hit.query_source not in fused.matched_queries:
                fused.matched_queries.append(hit.query_source)
            if hit.retriever == "dense":
                fused.dense_ranks[hit.query_source] = min(hit.rank, fused.dense_ranks.get(hit.query_source, hit.rank))
            elif hit.retriever == "bm25":
                fused.bm25_ranks[hit.query_source] = min(hit.rank, fused.bm25_ranks.get(hit.query_source, hit.rank))
            if hit.query_source == "original":
                if hit.retriever not in fused.original_query_hits:
                    fused.original_query_hits.append(hit.retriever)
            elif hit.retriever not in fused.rewrite_query_hits:
                fused.rewrite_query_hits.append(hit.retriever)
    return sorted(aggregate.values(), key=lambda hit: hit.rrf_score, reverse=True)
