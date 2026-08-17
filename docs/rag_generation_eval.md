# RAG Generation Evaluation

`evaluation/rag_generation_eval.py` runs generation from the same production
`RAGPipeline.retrieve()` used by the API. The judge receives the question,
gold answer, gold key points, retrieved Parent context, and generated answer.
The ten `answerable=false` cases are included and scored for explicit
abstention.

## Final run

The frozen 60-case benchmark completed with 50 answerable and 10 no-answer
cases. It used the same production pipeline, `knowledge_base_v2_children`,
BGE-M3, and Cross-Encoder on `cuda:0` (RTX 5060 Laptop GPU). The run produced
60 generated answers; 51 cases returned multi-query rewrites and 9 cases
recorded the explicit original-query fallback.

The raw judge metrics are diagnostic only: every aggregate field is `0.0167`,
and 59 of 60 rows are all-zero. The current judge adapter collapses provider
or parse exceptions into zero-valued scores, so this result cannot be presented
as a valid answer-quality score. A compact standalone judge smoke returned
valid JSON; the long benchmark judge requests need better error/status
instrumentation. This limitation is recorded rather than hidden.

The run's retrieval timing was mean 9,518.267 ms, p50 8,789.809 ms, and p95
14,446.299 ms; rewrite provider latency is the dominant component. RAGAS is
optional and does not block the primary retrieval plus judge evaluation.
Raw output: `data/eval/results/rag/generation_eval.json`.
