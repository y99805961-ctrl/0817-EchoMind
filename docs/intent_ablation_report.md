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

## Semantic score calibration

The separate calibration experiment is saved in
`data/eval/results/intent/semantic_calibration.json`. It reads the same 76
cases and the same 132 templates; no benchmark query was added to the
templates. The production three-way weights remain `0.45 / 0.35 / 0.20`.

`other_rejection_count` is the number of final predictions returned as
`other`; `gate_rejection_count` is the subset whose ungated Top1 was a
non-`other` intent but failed the confidence/margin gate. `ungated_top1_accuracy`
and `top1_ranking_accuracy` intentionally ignore that gate.

| Calibration | Accuracy | Macro-F1 | Other rejection | Gate rejection | Ungated Top1 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current `(cosine + 1) / 2` | 0.328947 | 0.371850 | 53 | 53 | 0.855263 |
| Raw cosine margin | 0.631579 | 0.676504 | 30 | 30 | 0.855263 |
| Temperature softmax | 0.500000 | 0.541581 | 40 | 40 | 0.855263 |

The raw cosine run uses confidence `0.0` (the affine-equivalent boundary to
normalized confidence `0.50`) and raw margin `0.05`. The temperature run uses
`softmax(raw cosine / 0.05)`, confidence `0.50`, and probability margin `0.05`.
These raw-margin and temperature values are explicitly exploratory; they are
not selected as production defaults. The identical `0.855263` ungated Top1
score across all three rows shows that the ranking itself is unchanged. The
current default loses accuracy mainly when the normalized margin gate rejects
an otherwise correct Top1 candidate.

The evaluator warms BGE-M3 and all template embeddings before collecting the
online samples. The measured latency is query encoding plus template cosine
scoring, excluding model/template initialization:

```text
cold_start_ms: 12843.1512
warm_mean_ms:  52.634287
warm_p50_ms:   50.475900
warm_p95_ms:   60.173800
samples:       76
```

Every calibration case also stores `raw_cosine_scores` and the selected raw
template matches for diagnosis. No final three-way weight was changed pending
the LLM credential fix.

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
