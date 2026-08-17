# Intent v2 ablation and routing report

## Benchmark integrity

The evaluation runner reads the frozen 76-case intent benchmark and 12-case
routing benchmark without modifying either file. The exact template leakage
check found `0` copied benchmark queries. The template set contains 19 intents
and 132 templates: six for each broad category and eight for each fine-grained
category.

## Ablation results

All values below come from `data/eval/results/intent/ablation.json`. `NOT_RUN`
means the required source was unavailable or failed; it is not a zero score.

| Method | Accuracy | Macro-F1 | Entity F1 | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rules Only | 0.486842 | 0.506536 | 0.941176 | 0.064459 ms | 0.058000 ms |
| BGE-M3 Only | 0.328947 | 0.371850 | 0.941176 | 210.832989 ms | 59.282500 ms |
| LLM Only | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| LLM + BGE-M3 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| LLM + BGE-M3 + Rules | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

Both the rules-only and BGE-M3-only runs are genuine executions of all 76
cases. Entity micro scores were precision `1.000000`, recall `0.888889`, and
F1 `0.941176`; field-level results are retained in the detailed JSON files.
BGE-M3 loaded locally on CPU with 1024-dimensional vectors. For the required
sanity query, `钱什么时候退回来` had cosine `0.779918` to a refund template
versus `0.513805` to a technical-crash template.

The configured DeepSeek-compatible LLM endpoint failed the single allowed
smoke test with HTTP 401, so all LLM-containing rows remain `NOT_RUN` and no
additional unauthenticated benchmark requests were sent.

## What the results show

- Rules are strong on exact signals: invoice, logistics, greeting, refund, and
  explicit error-code cases were handled well, and extracted entities had no
  false positives in this run.
- Rules are weak on colloquial and broad semantic categories. The largest
  observed confusions were `order_status -> other` (4),
  `payment_issue -> other` (4), `query -> other` (4), and
  `technical_login -> other` (4). These are the cases where BGE-M3 and an
  available LLM are intended to add coverage.
- No claim is made that the three-way pipeline improves over either component
  until a working LLM endpoint is supplied.

## Threshold search

The initial configuration is confidence `0.50`, margin `0.05`, and embedding
`top_n=3`. The requested 3x3 grid (`0.40/0.50/0.60` x `0.03/0.05/0.10`)
was executed against the real BGE-M3-only path and is saved in
`threshold_search.json`. Because this benchmark has only 76 cases and no
separate development split was provided, threshold search is explicitly
`exploratory only`; no test-set-tuned final claim is made.

## Routing benchmark

The final three-way routing benchmark is `NOT_RUN` because the LLM smoke test
failed with HTTP 401. The old rules-only 1.0/1.0/1.0 result is not used as a
final-system score. A separate real BGE-M3-only component run through the
existing `AgentOrchestrator` produced:

```text
Primary Routing Accuracy: 0.166667
Supporting Recall:        0.625000
Exact Match:              0.166667
```

Final three-way routing: `NOT_RUN` pending valid LLM credentials.

The changes were limited to compound-case scoring: technical login evidence
can take primary position over the existing account-security billing mapping,
and supporting general/billing domains are retained where the benchmark
expects them. No router framework or new agent family was introduced.

## Final configuration

```text
LLM weight:       0.45
Embedding weight: 0.35
Rule weight:      0.20

confidence threshold: 0.50
margin threshold:     0.05
embedding top_n:      3
```

These are configurable starting values, not empirically final values. BGE-M3
is installed and validated; the LLM credential still needs correction before
the complete three-way system can be selected and compared.

## FastAPI smoke

The full FastAPI lifespan passed with ChromaDB installed and returned `/health`
200. Five `/chat` compatibility cases also returned 200 with expected routing
for greeting, refund, payment issue, technical login, and human handoff. Those
endpoint requests used deterministic in-process LLM/agent fakes specifically to
avoid retrying the known invalid external credential; they validate the main
chain contract, not external LLM quality.
