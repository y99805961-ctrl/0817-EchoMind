# EchoMind RAG v2 Final Report

## Git and scope

```text
base tag: intent-v2-final
branch: feat/rag-v2
scope: Parent–Child Hybrid RAG 2.0 only; no Memory/MCP/frontend redesign
```

The frozen Intent v2 benchmark files were not modified. The RAG benchmark is
`data/eval/rag/rag_benchmark_60.json` with SHA-256
`1e6fcbdff41267fd6bea016bab5cf99bb89962a14fc21a4f5ddcecd5e18150fd`.

## Corpus and index

| Item | Final value |
|---|---:|
| Source documents | 10 |
| Parents | 164 |
| Children | 165 |
| Average Parent / Child chars | 154.97 / 151.22 |
| Dense model / dimension | BAAI/bge-m3 / 1024 |
| Chroma collection | `knowledge_base_v2_children` |
| Index status | ready |

The index uses stable business-derived IDs, explicit Chroma `embeddings=` and
`query_embeddings=`, and the same child corpus for Dense and BM25. The final
chunk configuration is Parent by H2/H3/paragraph semantic boundaries, Child
400 characters with 60-character overlap. The frozen chunk experiment showed
identical Recall@10 for 300/50, 400/60, and 550/80; 400/60 remains the
production-compatible choice because the benchmark gold child IDs are frozen
to that corpus.

## Retrieval results

| Method | Parent R@5 | Child R@10 | MRR@10 | NDCG@10 | P95 total |
|---|---:|---:|---:|---:|---:|
| Dense | 0.7500 | 0.8167 | 0.7093 | 0.7202 | 2.649 ms |
| BM25 | 0.6833 | 0.7667 | 0.5697 | 0.6097 | 2.212 ms |
| Hybrid + RRF | 0.7667 | 0.8167 | 0.6613 | 0.6872 | 4.792 ms |
| Rewrite + Hybrid | 0.7667 | 0.8000 | 0.7083 | 0.7196 | 13,750.743 ms |
| Production + Cross-Encoder | 0.7833 | 0.8000 | 0.7125 | 0.7229 | 13,306.393 ms |

No-answer false-positive rate was 0.1667 for every retrieval method. The
production `/chat`, `/search`, and evaluation paths all call the same pipeline
and collection. Local Windows verification used the explicit
`chroma_endpoint_unavailable_local_fallback`; Dense remained ready rather than
silently degrading to BM25-only.

## GPU and latency verification

The `.venv` check passed with no broken requirements:
`torch=2.13.0+cu130`, CUDA available, RTX 5060 Laptop GPU. Direct model smoke
and production-pipeline smoke both reported BGE and Cross-Encoder model
devices as `cuda:0`.

For `rewrite=False` production retrieval, the first cold query measured
23,482.597 ms total, including a 6,745.080 ms model cold start. Two warm
queries measured 284.976 ms and 442.528 ms total. Generation evaluation with
real rewrite had mean/p50/p95 total retrieval latency of
9,518.267 / 8,789.809 / 14,446.299 ms.

## Generation evaluation

All 60 generation cases completed. There were 51 multi-query rewrite results
and 9 explicit rewrite fallbacks. The LLM judge output is not accepted as a
quality conclusion: 59 of 60 rows were all-zero and the adapter currently
collapses judge exceptions into zero-valued scores. The raw answers and scores
remain in `data/eval/results/rag/generation_eval.json`; this provider/judge
instrumentation issue is a known limitation, not a manufactured metric.

## Regression and tests

```text
pytest: 44 passed, 1 skipped
Intent 76-case llm_embedding: OK; accuracy 0.934211; macro-F1 0.925230; entity-F1 0.941176
Routing 12-case llm_embedding: OK; primary accuracy 1.0; supporting recall 1.0; exact match 1.0
```

Regression artifacts are `data/eval/results/intent/rag_v2_intent_regression.json`
and `data/eval/results/intent/rag_v2_routing_regression.json`.

## Modified files

- `rag/`: corpus parsing, Parent–Child chunking, index build, Dense/BM25,
  rewrite, RRF, reranker, parent expansion, context packing, pipeline, and gate.
- `api/main.py`, `mcp/tool_manager.py`, `requirements.txt`, `config/rag.yaml`,
  `.gitignore`.
- `evaluation/`, `data/eval/rag/`, `data/eval/results/rag/`, and RAG tests.
- `docs/rag_*.md` architecture, evaluation, and handoff reports.

## Known issues and safe claims

- The configured Docker Chroma hostname is unavailable in the Windows run;
  local persistent Chroma fallback is explicit and tested.
- Chroma emits a non-blocking PostHog telemetry compatibility warning.
- Long-form external judge calls need better error/status instrumentation before
  generation quality scores can be used.

Safe résumé claims are limited to the implemented and measured features:
stable Parent–Child chunking, BGE-M3 Dense plus BM25 hybrid retrieval, RRF,
Cross-Encoder reranking, parent expansion, explicit fallback behavior, frozen
60-case retrieval evaluation, GPU execution verification, and Intent/Routing
regression results above.
