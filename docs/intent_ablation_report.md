# Intent v2 ablation and routing report

## Benchmark integrity

The evaluation runner reads the frozen 76-case intent benchmark and 12-case
routing benchmark without modifying either file. The exact template leakage
check found `0` copied benchmark queries. The template set contains 19 intents
and 132 templates: six for each broad category and eight for each fine-grained
category.

## Runtime and LLM smoke

The evaluation used the project `.venv` with Python 3.11.9. `.env` is ignored by
Git (`git check-ignore .env` passed) and the full API key is never written to
logs, results, or source. The safe configuration check reported:

```text
base_url: https://api.deepseek.com/anthropic
model: deepseek-v4-pro
api_key_loaded: true
LLM smoke: PASS
```

The smoke input was `你好`. The DeepSeek-compatible endpoint returned a valid
response. The classifier uses `temperature=0.0`, `max_tokens=4096`, and one
compact JSON-only retry only when a response ends without a parseable JSON
object; no intent is inferred from non-JSON prose.

## Ablation results

All values below come from `data/eval/results/intent/ablation.json`. Every row
is `OK` over all 76 frozen cases. `Accuracy` is the gated accuracy; the BGE row
also reports the gate-free ranking diagnostic requested by the calibration
experiment.

| Method | Gated Accuracy | Macro-F1 | Ungated Top1 | Other rejection | Entity F1 | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules Only | 0.486842 | 0.506536 | 0.500000 | 32 | 0.941176 | 0.073317 ms | 0.058800 ms |
| BGE-M3 Only | 0.328947 | 0.371850 | 0.855263 | 53 | 0.941176 | 3533.617949 ms | 4265.229200 ms |
| LLM Only | 0.921053 | 0.896157 | 0.921053 | 1 | 0.941176 | 5982.555441 ms | 13197.182200 ms |
| LLM + BGE-M3 | 0.934211 | 0.916040 | 0.934211 | 1 | 0.941176 | 4589.739399 ms | 8364.675400 ms |
| LLM + BGE-M3 + Rules | 0.907895 | 0.890097 | 0.907895 | 1 | 0.941176 | 4994.558234 ms | 10385.292000 ms |

All five rows are genuine executions of all 76 cases. Entity micro scores were
precision `1.000000`, recall `0.888889`, and F1 `0.941176`; field-level results
are retained in the detailed JSON files. The LLM-containing rows are real
endpoint results, not deterministic fakes.
BGE-M3 loaded locally on CPU with 1024-dimensional vectors. For the required
sanity query, `钱什么时候退回来` had cosine `0.779918` to a refund template
versus `0.513805` to a technical-crash template.

The `0.45 / 0.35 / 0.20` LLM/embedding/rules weights were used only as the
baseline requested for this run. They are not declared the final best
parameters, and no weight or production calibration change was made.

## Production Intent finalization

The production recognizer now uses `llm_embedding`: LLM and BGE-M3 are the
semantic intent sources, with the same 76-case benchmark remaining frozen. The
latest production rerun is saved in `intent_llm_embedding.json`:

```text
Production Intent Mode: llm_embedding
Intent Accuracy:        0.921053
Intent Macro-F1:         0.902673
Ungated Top1 Accuracy:   0.934211
Other rejection count:   1
```

Rules are still collected on every production request as high-precision
signals, routing/domain evidence, urgency inputs, entity support, and the two
exceptional overrides (`explicit_human_handoff` and `critical_security_event`).
Refund, invoice, payment, 401, and 500 rules do not compete as a fixed
third-source weight in production. The `llm_embedding_rules` path remains
available for ablation only.

## What the results show

- Rules are strong on exact signals: invoice, logistics, greeting, refund, and
  explicit error-code cases were handled well, and extracted entities had no
  false positives in this run.
- Rules are weak on colloquial and broad semantic categories. The largest
  observed confusions were `order_status -> other` (4),
  `payment_issue -> other` (4), `query -> other` (4), and
  `technical_login -> other` (4). These are the cases where BGE-M3 and an
  available LLM are intended to add coverage.
- On this baseline run, LLM-only and LLM+BGE outperform the three-way result;
  this is an observation for later weight analysis, not a final parameter
  selection.

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
templates. This experiment does not change production `llm_embedding`; the
three-way rules-weighted path remains an ablation configuration.

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
template matches for diagnosis. The calibration variants remain exploratory;
no final three-way weight or production calibration was changed.

## Routing benchmark

The final routing run used the production `llm_embedding` Intent output for all
12 cases and completed with status `OK`. It did not use rules-only or BGE-only
substitution. Every case stores semantic intent, top candidates, rule signals,
domain scores, primary agent, supporting agents, and routing reason in
`routing_benchmark.json`:

```text
Primary Routing Accuracy: 1.000000
Supporting Recall:        1.000000
Exact Match:              1.000000
```

There are no remaining routing failures in this run. The result comes from
general domain evidence scoring: explicit technical failure signals can make
Technical primary while Billing remains supporting; ordinary billing signals
do not override the semantic intent.

The changes use generalized semantic candidates, rule signals, entities,
urgency, and domain evidence for compound routing. Technical login evidence
can take primary position over the existing account-security billing mapping,
and supporting general/billing domains are retained where the benchmark
expects them. No router framework or new agent family was introduced.

## Final configuration

```text
Production mode:  llm_embedding
LLM weight:       0.45
Embedding weight: 0.35
Rule weight:      0.00 (routing/override evidence only)

confidence threshold: 0.50
margin threshold:     0.05
embedding top_n:      3
```

The LLM and embedding weights are retained as the current production baseline;
the rules weight is removed from production semantic competition but remains in
the ablation path. Calibration search remains exploratory, and no test-set
hardcoding or production weight tuning was added.

## FastAPI smoke

The full FastAPI lifespan passed with ChromaDB installed and returned `/health`
200. Five `/chat` compatibility cases also returned 200 with expected routing
for greeting, refund, payment issue, technical login, and human handoff. Those
endpoint requests used deterministic in-process LLM/agent fakes to validate the
main chain contract; the ablation and routing numbers above are the separate
real external-LLM results.
