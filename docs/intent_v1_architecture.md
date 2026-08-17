# EchoMind Intent v1 architecture audit

## Scope

This document records the implementation that existed on `main` before the
Intent v2 changes. It is based on `core/intent_recognizer.py`,
`agents/agent_orchestrator.py`, `api/main.py`, and `evaluation/evaluator.py`.

## Intent taxonomy

The v1 enum contained 19 values:

`query`, `complaint`, `request`, `greeting`, `escalation`, `technical`,
`billing`, `account`, `feedback`, `order_status`, `logistics`, `refund`,
`invoice`, `payment_issue`, `account_security`, `technical_login`,
`technical_crash`, `human_handoff`, and `other`.

The broad intents were `query`, `complaint`, `request`, `greeting`,
`escalation`, `technical`, `billing`, `account`, and `feedback`. The fine-grained
intents were the nine order, logistics, refund, invoice, payment, security,
login, crash, and human-handoff categories. Fine-grained results were mapped to
their broad group for downstream routing.

## Three source layers

### LLM

The recognizer built a few-shot prompt from the in-code `_TEMPLATES` dictionary,
included up to three recent history messages, and requested JSON containing
`intent`, `confidence`, and `reasoning`. Parsing took the substring between the
first `{` and last `}`. Parse/API errors returned `other` with zero confidence.

### Vector / embedding

The v1 implementation did not use BGE-M3. If an Anthropic-compatible client
exposed an `embeddings.create` method it attempted a remote Voyage model;
otherwise it used a deterministic character 1/2/3-gram MD5 hash vector. Template
vectors were lazily created and held in process memory. Each category used the
single best template cosine similarity, not a configurable top-N aggregate.

When `ANTHROPIC_BASE_URL` was set, the vector path was disabled entirely because
the code assumed third-party clients did not support embeddings.

### Rules / patterns

The recognizer used ordered keyword lists for specific categories first, then
generic categories. A matching category received `0.5 + 0.25 * (hit_count - 1)`
up to 1.0. Rules included refund, invoice, payment, security, login, crash,
logistics, order status, escalation, complaint, greeting, request, billing,
technical, and account signals.

## Fusion and confidence

With the default client configuration the v1 weights were LLM `0.70`, vector
`0.20`, and rules `0.10`. With a configured base URL it used LLM `0.85` and
rules `0.15`, with no vector contribution. The implementation summed the
confidence attached to each source's top intent; scores were not normalized to
a common semantic scale first. A pattern-based special case could replace a
generic winner with a specific rule result. There was a single confidence
threshold, but no top1-top2 margin calculation and no candidate list.

## Entities

Rules extracted `order_id`, `date`, `amount`, and `error_code`, plus an always
empty `product` field. Extraction happened after classification and did not call
the LLM.

## Consumers and compatibility boundaries

`IntentResult` was consumed by `AgentOrchestrator`, which copied intent, group,
urgency, and confidence into `Request`; `api/main.py` used the same result to
decide whether to search the knowledge base and to populate `/chat` output.
The orchestrator mapped technical categories to `TechnicalAgent`, billing and
account categories to `BillingAgent`, escalation/handoff to escalation, and
everything else to `GeneralAgent`.

Important boundaries were the public async `recognize(message, history)` method,
the `IntentResult` fields, enum values, entity dictionary shape, and the
orchestrator's category-to-agent table. Intent v2 keeps those fields and values
and adds diagnostics rather than replacing the call chain.

## v1 baseline execution

The frozen `data/eval/intent/intent_benchmark_76.json` was executed against the
pre-v2 code before source changes. All 76 external LLM calls returned HTTP 401
authentication failures, so the baseline is recorded as `NOT_RUN`; no accuracy,
F1, entity F1, or latency score is claimed from those failed calls.
