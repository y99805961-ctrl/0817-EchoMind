# NexusOps Frontend / E2E / Deployment Report

Date: 2026-08-17  
Branch: `feat/frontend-e2e`

## Local Entry Points

- `http://localhost:5175`: frontend development / Vite hot reload. Start it with `VITE_PORT=5175 npm run dev -- --host 0.0.0.0`; its `/api/*` requests proxy to the Docker Compose entry at `http://localhost`.
- `http://localhost`: final Docker Compose demo entry, served by the production Nginx gateway.

## Frontend Architecture

- React 18 + TypeScript + Vite, with Ant Design for primitives, Zustand for persisted client state, React Router for the three product routes, `fetch` for API calls, and `react-markdown` for assistant output.
- `AppShell` owns the global header, NexusOps navigation, health indicator and mobile drawers; the duplicate route rail has been removed.
- `chatStore` persists conversations, stable `userId`, active `conv_id`, messages and diagnostics in `localStorage`; the backend remains the source of truth for answers and routing traces.
- `client.ts` centralizes `/api` requests and normalized error handling. Chat, knowledge search/stats/upload/rebuild/trace, health and monitor data are all loaded from real backend endpoints.
- Chat supports Enter-to-send, Shift+Enter newline, loading state, Markdown rendering, backend failure copy and retry. Diagnostics show intent/confidence, primary/supporting agents, knowledge usage, latency, entities, source scores and routing reason.

## Product Presentation

NexusOps 是现有 Agent Runtime 在企业运营协同场景下的产品化表达。底层 Intent、RAG、Memory、Routing 与 Evaluation 实现保持不变，前端通过 Presentation Mapping 将内部角色映射为运营协调、技术可靠性、收入与合规和运营升级通道。

## Visual Design

The UI uses a Fresh Botanical palette: warm white canvas, clear green/blue/apricot/coral accents, restrained borders, rounded surfaces and generous whitespace. The Chat page is a request-history / workspace / collaboration-trace layout on desktop and collapses to drawers on narrow screens. Knowledge and Monitor use the same product language. The 1440x900 QA screenshots are:

- `data/eval/results/e2e/screenshots/nexusops-workspace-empty.png`
- `data/eval/results/e2e/screenshots/nexusops-workspace-result.png`
- `data/eval/results/e2e/screenshots/nexusops-knowledge.png`
- `data/eval/results/e2e/screenshots/nexusops-monitor.png`

## API Contract

The browser contract is `/api/*`; Nginx strips that prefix before forwarding to FastAPI. Direct product endpoints remain available for health and operations:

| UI capability | Browser request | Backend target |
|---|---|---|
| Chat | `POST /api/chat` | `POST /chat` |
| Health | `GET /api/health` | `GET /health` |
| Knowledge stats | `GET /api/knowledge/stats` | `GET /knowledge/stats` |
| Knowledge search | `POST /api/search` | `POST /search` |
| Markdown upload | `POST /api/knowledge/upload` | `POST /knowledge/upload` |
| Rebuild | `POST /api/knowledge/rebuild` | `POST /knowledge/rebuild` |
| RAG trace | `POST /api/debug/rag` | `POST /debug/rag` |
| Monitor | `GET /api/monitor` | `GET /monitor` |

Nginx keeps `/health`, `/docs`, `/redoc`, `/openapi.json` and `/metrics`; only Nginx publishes a host port in the production Compose file.

## Docker Topology

```text
Browser :80
    |
    v
Nginx (echomind-nginx)
    |-- /             -> frontend:80 (React SPA + history fallback)
    |-- /api/*        -> echomind:8000 (prefix stripped)
    |-- /health/docs/... -> echomind:8000
    |
    +-- echomind:8000 -> redis:6379 + chromadb:8000
    +-- prometheus:9090 (internal monitoring)
```

`frontend/Dockerfile` is a Node build stage followed by an Nginx static stage. `docker-compose.yml` adds the frontend service, waits for frontend health before starting the root Nginx, and does not publish backend, Redis, ChromaDB or Prometheus host ports.

Validation used an isolated Compose project because this workstation already had same-name EchoMind containers and host port allocations. The isolated run completed with `frontend`, `echomind`, `redis`, `chromadb`, `nginx` and `prometheus` all healthy. Root SPA, `/knowledge`, `/api/health`, `/api/docs`, `/api/redoc`, `/api/openapi.json`, `/api/metrics`, `/docs`, `/redoc` and `/openapi.json` returned HTTP 200; `/metrics` also returned successfully from inside Nginx where its existing access restriction permits it.

## Core E2E Cases

The frozen case list is [`data/eval/e2e/e2e_cases.json`](../data/eval/e2e/e2e_cases.json), version `phase3-v1`, with 14 deterministic Core cases:

1. App boot and no console errors
2. New conversation and persistence
3. Greeting
4. Refund
5. Technical login / 401
6. Human handoff
7. RAG knowledge usage
8. No-RAG greeting
9. Compound routing
10. Multi-turn memory
11. Conversation switch
12. Knowledge search playground
13. Knowledge stats
14. Backend failure UX and retry affordance

Core result: **14/14 passed (100%)**. The run used `E2E_TEST_MODE=1` from the backend startup script. The Fake Provider is startup-only and passes through the real FastAPI, memory, intent, routing, agent and RAG code paths. There is no Playwright `/chat` route mock.

## Live Smoke Cases

The live suite has three structural smoke cases against the configured real provider:

- `LIVE-001` Greeting
- `LIVE-002` Refund
- `LIVE-003` RAG knowledge

Live result: **3/3 passed (100%)**. These checks only assert response presence and expected UI diagnostics/routing structure; there is no LLM judge and no semantic generation of additional cases.

## Final Test Results

| Check | Result |
|---|---:|
| `frontend npm run build` | PASS |
| `frontend npm run lint` | PASS |
| `pytest -q` | 49 passed, 1 skipped |
| `pip check` | PASS |
| Core E2E | 14/14, 100% |
| Live smoke | 3/3, 100% |
| Core case duration mean / nearest-rank P95 | 4,054 ms / 38,779 ms |
| Live case duration mean / nearest-rank P95 | 36,701 ms / 45,920 ms |
| Visual QA | 1 Playwright test passed; 4 screenshots emitted |

The Core P95 is a small-sample nearest-rank statistic and includes the first real local embedding/reranker initialization. Subsequent deterministic cases are materially shorter.

## Scope Exclusions

- No Intent benchmark or RAG benchmark/result files were changed.
- No 60-case generation was performed.
- No existing Intent/RAG algorithm rewrite was introduced.
- No fake production metrics or fabricated benchmark outcomes were added.
- No LLM judge was used for Core or Live smoke.

## Known Limitations

- The frontend production bundle emits Vite's existing single-chunk warning (>500 kB); build remains successful.
- `npm install` reports two moderate dependency-audit findings; no automatic audit rewrite was applied.
- Live smoke depends on the configured provider credentials, network availability and provider latency.
- The Knowledge page deliberately displays `N/A` when optional index metadata is unavailable rather than inventing values.
- Local deterministic runs may log non-blocking Chroma/PostHog telemetry warnings when the configured external Chroma endpoint is unavailable; the tested RAG path falls back to its local index as designed.
