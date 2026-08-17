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
| Rules Only | 0.486842 | 0.506536 | 0.941176 | 0.068734 ms | 0.054000 ms |
| BGE-M3 Only | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| LLM Only | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| LLM + BGE-M3 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| LLM + BGE-M3 + Rules | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

The rules-only run is a genuine execution of all 76 cases. Entity micro scores
were precision `1.000000`, recall `0.888889`, and F1 `0.941176`; field-level
results are retained in `rules_only.json`. The local environment did not have
`sentence-transformers` or `torch`, so all BGE-containing rows are `NOT_RUN`.
The configured external LLM returned HTTP 401 for all 76 requests, so all
LLM-containing rows are also `NOT_RUN`.

## What the results show

- Rules are strong on exact signals: invoice, logistics, greeting, refund, and
  explicit error-code cases were handled well, and extracted entities had no
  false positives in this run.
- Rules are weak on colloquial and broad semantic categories. The largest
  observed confusions were `order_status -> other` (4),
  `payment_issue -> other` (4), `query -> other` (4), and
  `technical_login -> other` (4). These are the cases where BGE-M3 and an
  available LLM are intended to add coverage.
- No claim is made that the three-way pipeline improves over rules-only until
  valid BGE-M3 weights and a working LLM endpoint are available.

## Threshold search

The initial configuration is confidence `0.50`, margin `0.05`, and embedding
`top_n=3`. Candidate exploratory values are confidence `0.40/0.50/0.60` and
margin `0.03/0.05/0.10`. Because this benchmark has only 76 cases and no
separate development split was provided, threshold search is explicitly
`exploratory only`; no test-set-tuned final claim is made.

## Routing benchmark

The 12 routing cases were run through the existing `AgentOrchestrator` decision
logic using the deterministic rules-only intent source because LLM/BGE were
unavailable:

```text
Primary Routing Accuracy: 1.000000
Supporting Recall:        1.000000
Exact Match:              1.000000
```

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

These are configurable starting values, not empirically final values. They
should be re-evaluated after installing BGE-M3 and restoring a valid LLM
credential, preferably with a development split.
