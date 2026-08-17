# RAG Generation Evaluation

`evaluation/rag_generation_eval.py` runs generation from the same production
`RAGPipeline.retrieve()` used by the API. The judge receives the question,
gold answer, gold key points, retrieved Parent context, and generated answer.
The ten `answerable=false` cases are included and scored for explicit
abstention.

## Phase 2.1 run

The frozen 60-case audit completed all 60 rows using the same production
pipeline and local GPU models. The Judge now records status, error type and
message, redacted raw output, latency, retry state, and nullable scores per
case. It no longer converts a failed Judge call into zero.

The configured provider was `deepseek-v4-pro` through the Anthropic-compatible
endpoint. DeepSeek documents that thinking is enabled by default and can be
disabled with the Anthropic-compatible `thinking` control; a real smoke also
showed that thinking could consume the whole JSON budget. The account then
returned HTTP 402 `Insufficient Balance`. The Judge now disables DeepSeek
thinking for structured scoring, supports code-fence/embedded JSON parsing,
retries transient/incomplete responses with bounded exponential backoff, and
records unavailable scores as `null`. See the [DeepSeek thinking-mode
documentation](https://api-docs.deepseek.com/guides/thinking_mode).

The Phase 2.1 run recorded 60/60 generation failures with HTTP 402, 0 valid
judged cases, and all five quality aggregates as `null`. This is a provider
availability result, not a generation-quality score. Query Rewrite likewise
recorded 60/60 explicit original-query fallbacks. Raw output:
`data/eval/results/rag/generation_eval_phase21.json`.

The bounded Rewrite prefetch used `max_concurrency=10`, completed in 2,776.585
ms, and the full unavailable-provider audit took 47,139.952 ms. The previous
serial run was observed at approximately 1,468 seconds, but it is not a valid
speedup comparison because the current run stopped receiving paid generation
responses. Historical serial rewrite latency remains in
`data/eval/results/rag/generation_eval.json`.
