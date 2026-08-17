# RAG Retrieval Ablation

The ablation runner is `python -m evaluation.rag_retrieval_eval`. Every method uses the same frozen 60-case benchmark:

1. Dense only
2. BM25 only
3. Dense + BM25 + RRF
4. Rewrite + Dense + BM25 + RRF
5. Rewrite + Dense + BM25 + RRF + Cross-Encoder

The JSON output is written to `data/eval/results/rag/retrieval_ablation.json`. It includes Child Recall@3/5/10, Parent Recall@3/5/10, MRR@10, NDCG@10, no-answer false-positive rate, and rewrite/dense/BM25/RRF/rerank/total latency distributions.

The report must answer whether Hybrid beats Dense, whether Rewrite adds recall or noise, whether reranking improves MRR/NDCG, the latency cost of reranking, and whether Parent expansion improves the final answer context. “Better” is not declared without those measurements.

## Final measured run

| Method | Parent Recall@5 | Child Recall@10 | MRR@10 | NDCG@10 | P95 total |
|---|---:|---:|---:|---:|---:|
| Dense | 0.7500 | 0.8167 | 0.7093 | 0.7202 | 2.649 ms |
| BM25 | 0.6833 | 0.7667 | 0.5697 | 0.6097 | 2.212 ms |
| Hybrid | 0.7667 | 0.8167 | 0.6613 | 0.6872 | 4.792 ms |
| Hybrid + Rewrite | 0.7667 | 0.8000 | 0.7083 | 0.7196 | 13,750.743 ms |
| Production + Rewrite + Cross-Encoder | 0.7833 | 0.8000 | 0.7125 | 0.7229 | 13,306.393 ms |

This final run used the configured rewrite provider and GPU model environment. No-answer false-positive rate was 0.1667 for every method. Hybrid improves Parent Recall@5 over Dense (0.7667 vs 0.7500); the production Cross-Encoder path improves MRR/NDCG over plain Hybrid, while rewrite latency dominates the tail. Full raw output: `data/eval/results/rag/retrieval_ablation.json`.

The earlier CPU-only run is retained as historical diagnostic data, not as the final claim: production Parent Recall@5 0.8000, MRR@10 0.7139, NDCG@10 0.7236, P95 3,547.969 ms. The final environment check confirmed BGE and Cross-Encoder model devices as `cuda:0` on an RTX 5060 Laptop GPU.

## Phase 2.1 rerun

The rerun used the same frozen 60 cases, local cached Cross-Encoder, and
`max_concurrency=10` Rewrite prefetch. The provider returned HTTP 402 for all
Rewrite requests, so every Rewrite case explicitly fell back to the original
query. Production retrieval remained stable: Parent Recall@5 `0.7833`, Child
Recall@10 `0.8000`, MRR@10 `0.7125`, NDCG@10 `0.7229`, and P95 total `408.376
ms`. BGE and reranker model devices were both `cuda:0`.

The Phase 2.1 artifact is `data/eval/results/rag/retrieval_ablation_phase21.json`.
Its Rewrite prefetch wall time was 2,844 ms; Rewrite latency recorded per case
was mean 440.635 ms, p50 415.713 ms, and p95 526.898 ms. These are fallback
latencies under HTTP 402 and must not be described as successful provider
Rewrite performance.
