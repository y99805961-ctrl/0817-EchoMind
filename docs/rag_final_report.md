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

## Phase 2.1 Generation and Rewrite

The Judge repair is implemented and verified against the actual provider:
DeepSeek thinking is disabled for structured JSON judging, valid JSON can be
embedded in Markdown/prose, retries are bounded, and every case records
`judge_status`, `judge_error_type`, `judge_error_message`, redacted
`judge_raw_output`, latency, and retry count. Failed scores are `null` and are
excluded from averages.

The 60-case Phase 2.1 audit completed all rows but the provider returned HTTP
402 `Insufficient Balance` for generation. Therefore: generation failures
60/60, valid judged cases 0/60, judge-not-run 60/60, and all quality metrics
are `null`. Rewrite also recorded 60/60 original-query fallbacks. The result is
explicitly unavailable, not a quality score. See
`data/eval/results/rag/generation_eval_phase21.json`.

Rewrite prefetch used bounded concurrency 10 and took 2,776.585 ms. The full
provider-unavailable audit took 47,139.952 ms. The prior serial run was
observed at approximately 1,468 seconds, but no speedup claim is made because
the provider did not execute the current generation workload. The prior real
answers remain in `data/eval/results/rag/generation_eval.json`.

## Phase 2.2 Judge-only validation

Phase 2.2 reused the 60 historical answers in
`data/eval/results/rag/generation_eval.json` by frozen benchmark ID. The
answers came from the prior real DeepSeek generation run. No answers were
regenerated, and the old generation scores were ignored. Qwen3.7-Plus was
used only as an independent LLM-as-Judge.

The judge-only run completed 60/60 cases successfully. It made zero
Generation calls and zero remote Rewrite calls. To supply context without
rerunning Rewrite, each case used a local production retrieval replay with
`rewrite=False` (60 local retrieval calls). The resulting quality metrics are:

| Metric | Phase 2.2 value |
|---|---:|
| Correctness | 0.9317 |
| Completeness | 0.8917 |
| Relevance | 0.9667 |
| Faithfulness | 0.9583 |
| Abstention accuracy | 0.7500 |
| Judge success / valid | 60 / 60 |
| Judge latency mean / P95 | 1,558.100 / 2,113.217 ms |

The complete artifact is `data/eval/results/rag/generation_judge_qwen.json`;
the smoke artifact is `data/eval/results/rag/generation_judge_qwen_smoke.json`.

## Regression and tests

```text
pytest: 49 passed, 1 skipped
Intent 76-case llm_embedding: prior valid run OK; current Phase 2.1 rerun NOT_RUN (76/76 provider 402)
Routing 12-case llm_embedding: prior valid run OK; current Phase 2.1 rerun NOT_RUN (12/12 provider 402)
```

Regression artifacts are `data/eval/results/intent/rag_v2_intent_regression.json`
and `data/eval/results/intent/rag_v2_routing_regression.json`. Phase 2.1
unavailable reruns are `rag_v2_intent_phase21.json` and
`rag_v2_routing_phase21.json`.

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
- Phase 2.1 generation quality remains unavailable because the configured
  DeepSeek account returned insufficient balance. Phase 2.2 provides an
  independent Qwen Judge score for the historical answers, but it is not a
  new generation-quality run and should not be described as Qwen generation.
- The Phase 2.1 provider-unavailable rerun cannot establish a valid generation
  quality score or successful Rewrite latency improvement.

Safe résumé claims are limited to the implemented and measured features:
stable Parent–Child chunking, BGE-M3 Dense plus BM25 hybrid retrieval, RRF,
Cross-Encoder reranking, parent expansion, explicit fallback behavior, frozen
60-case retrieval evaluation, GPU execution verification, and Intent/Routing
regression results above.
