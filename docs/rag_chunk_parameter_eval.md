# Parent–Child Chunk Parameter Evaluation

The Parent strategy is fixed. Only Child parameters are compared:

| Candidate | Child size | Overlap | Retrieval mode |
|---|---:|---:|---|
| A | 300 | 50 | BGE Dense only, no rewrite, no reranker |
| B | 400 | 60 | BGE Dense only, no rewrite, no reranker |
| C | 550 | 80 | BGE Dense only, no rewrite, no reranker |

The runner is `python -m evaluation.rag_chunk_parameter_eval`. It evaluates Child Recall@5/10, Parent Recall@3/5, MRR@10, NDCG@10, average Child length, Child count, and latency against the frozen `rag_benchmark_60.json`. The final configuration must be selected from these measurements rather than assumed in advance.

No benchmark query is added to the source corpus during this experiment; that would create retrieval leakage.

## Measured run

The 2026-08-17 CPU run used the frozen 60 cases and BGE-M3 Dense only:

| Candidate | Children | Child Recall@10 | Parent Recall@5 | MRR@10 | NDCG@10 | Avg Child chars | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300/50 | 167 | 0.8167 | 0.7500 | 0.7071 | 0.7183 | 149.60 | 7.301 ms |
| 400/60 | 165 | 0.8167 | 0.7500 | 0.7093 | 0.7202 | 151.22 | 8.397 ms |
| 550/80 | 164 | 0.8167 | 0.7500 | 0.7121 | 0.7224 | 151.76 | 7.456 ms |

The production configuration remains **400/60** for this frozen benchmark because its gold Child IDs are tied to the rebuilt 400/60 corpus and 550/80's MRR/NDCG gain is marginal while recall is identical. The 550/80 result is recorded rather than silently treated as a universal improvement. Full raw output: `data/eval/results/rag/chunk_parameter_eval.json`.
