# EchoMind Intent v2 architecture

```text
User Query + recent history
              |
      +-------+--------+
      |       |        |
     LLM    BGE-M3   Rules
      |       |        |
  0..1 scores  template cosine/top-N  0..1 rule scores
      +-------+--------+
              |
      source normalization
              |
      configurable weighted fusion
              |
     top1 confidence + top2 margin
              |
      confidence/margin gate
              |
     fine-grained IntentResult
       + entities + diagnostics
              |
        AgentOrchestrator routing
```

## Shared BGE-M3 service

`core/embedding_service.py` provides a process-wide `EmbeddingService` using
`BAAI/bge-m3` by default. Model imports and weights are lazy: importing the
application does not download or initialize a model. `BGE_DEVICE=auto` selects
CUDA when available and CPU otherwise. Optional FP16 is used only on CUDA.
GPU out-of-memory during encoding logs a warning and retries with a CPU model;
missing runtime packages or model-load failures raise a clear
`EmbeddingServiceError`.

The service exposes query and batch encoding and is intentionally independent of
Intent so future retrieval code can reuse it. Intent template vectors are batch
encoded once and kept in memory. `reload_templates()` explicitly clears that
cache after a template file change.

## Template and benchmark isolation

`data/intent/templates.json` is the only source for semantic examples. Major
fine-grained intents have eight templates and broad intents have six. The
evaluation runner reads, but never modifies, the frozen benchmark files under
`data/eval/intent/`. The runner also performs an exact query leakage check and
reports the result in machine-readable output.

For each query, the embedding path computes cosine similarity to every template,
normalizes it to `[0, 1]`, and averages the best configurable `top_n` templates
per intent. It does not collapse the category to one center vector.

## Three-way score fusion

LLM output is converted to a per-intent score map. The rule layer emits per-intent
high-precision scores and hit diagnostics for signals such as refund, invoice,
duplicate payment, 401/500, crash, suspicious login, and human handoff. All
active source values are clamped or transformed into `[0, 1]` before fusion.

The initial configurable weights are:

```text
LLM       0.45
Embedding 0.35
Rules     0.20
```

If a source is disabled or unavailable, active weights are renormalized across
the remaining sources. The source statuses and detailed candidate diagnostics
are retained in `IntentResult.source_scores`.

## Confidence and margin gating

The default confidence threshold is `0.50`; the default top1-top2 margin
threshold is `0.05`. If either condition fails, the final intent is `other`.
The result still retains `top_candidates`, `top1_score`, `top2_score`, and
`margin`, making low-confidence cases inspectable without routing them as if
they were certain.

## Why the combination is deliberate

LLM classification handles context and ambiguous business language, but is
external, slower, and sensitive to availability. BGE-M3 adds local semantic
matching for colloquial or synonymous expressions without replacing the LLM.
Rules remain valuable because exact signals such as `401`, duplicate charges,
or “转人工” are high precision and should not be lost to semantic variation.

## Compatibility and routing

The async recognizer signature, enum values, intent groups, urgency values,
entity fields, and existing orchestrator route table remain compatible. The
orchestrator received only small routing adjustments for compound benchmark
cases: strong technical evidence can make a security case primary while its
existing billing ownership remains supporting, and general-domain supporting
agents are retained for billing/order compounds. No new router framework or
agent family was introduced.

## Configuration

```text
BGE_MODEL_NAME=BAAI/bge-m3
BGE_DEVICE=auto
BGE_USE_FP16=true
BGE_BATCH_SIZE=16
INTENT_LLM_WEIGHT=0.45
INTENT_EMBED_WEIGHT=0.35
INTENT_RULE_WEIGHT=0.20
INTENT_CONFIDENCE_THRESHOLD=0.50
INTENT_MARGIN_THRESHOLD=0.05
INTENT_EMBEDDING_TOP_N=3
```

The final weights and thresholds should be selected from a development split
when data volume permits. With this 76-case benchmark, threshold search is
reported as exploratory rather than treated as a test-set optimization claim.
