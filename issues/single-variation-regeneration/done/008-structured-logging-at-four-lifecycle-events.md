# Structured logging at four lifecycle events

## Parent PRD

`prds/2026-04-29-single-variation-regeneration-prd.md`

## What to build

Emit operator-facing structured log lines at the four key lifecycle events of a single-variation regen, so log analytics can answer questions about regen usage rates, success rates, fallback frequency, and elapsed time without spelunking through unstructured logs.

The four log lines live in the regen endpoint (or a small helper called from the endpoint and pipeline as appropriate), each emitted at a distinct point in the flow:

- `staging.variation_regen.started` — emitted when the endpoint accepts the request, after concurrency / 404 / 400 / 409 checks pass
- `staging.variation_regen.completed` — emitted on terminal success
- `staging.variation_regen.failed` — emitted on terminal failure (image-gen error, LLM error, etc.)
- `staging.variation_regen.fallback_to_fresh` — emitted alongside the `variation_fallback` SSE event when retry has no prior prompt

Each line includes structured fields: `project_id`, `room_id`, `variation_id`, `strategy` (the requested strategy), `effective_strategy` (the actual strategy used after any fallback), `elapsed_ms` (where applicable), `tokens_used` (where applicable). No PII or secrets in the payload.

Promotion to a metrics sink (Application Insights, OpenTelemetry, etc.) is explicitly **out of scope for this PRD** — see the PRD's Out of Scope section.

See PRD sections **Implementation Decisions → Backend** (the structured-logging bullet) and **User Stories → 22** for full context.

## Acceptance criteria

- [ ] Backend: `staging.variation_regen.started` log line fires after the endpoint's concurrency/404/400/409 checks pass and before the pipeline call
- [ ] Backend: `staging.variation_regen.completed` log line fires on terminal success
- [ ] Backend: `staging.variation_regen.failed` log line fires on terminal failure
- [ ] Backend: `staging.variation_regen.fallback_to_fresh` log line fires alongside the `variation_fallback` SSE event
- [ ] Backend: each log line includes structured fields `project_id`, `room_id`, `variation_id`, `strategy`, `effective_strategy`
- [ ] Backend: the `completed` and `failed` lines also include `elapsed_ms` and `tokens_used` (the latter where available — e.g., 0 or null for retry-no-LLM-call flows)
- [ ] Backend: no PII or secrets are logged
- [ ] Backend test: a happy-path retry emits exactly `started` and `completed` log lines with the expected fields
- [ ] Backend test: a happy-path fresh emits exactly `started` and `completed`
- [ ] Backend test: a retry-no-prior-prompt path emits `started`, `fallback_to_fresh`, `completed` in order
- [ ] Backend test: a failure path emits `started` and `failed`

## Blocked by

- Blocked by `004-retry-to-fresh-fallback-signaling.md` (the `fallback_to_fresh` log line is paired with the `variation_fallback` SSE event, which doesn't exist until that slice lands)

## User stories addressed

Reference by number from the parent PRD:

- User story 22 (structured log lines at 4 events)
