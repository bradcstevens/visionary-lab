## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

Switch `frontend/context/jobs-context.tsx` from its current
browser-bound progress source to a REST seed (`GET .../jobs`) plus an
EventSource subscription (`.../jobs/stream`). 5s polling fallback when
`EventSource` is unavailable. Closing a tab, refreshing, or opening the
same project in a second tab continues to show live, consistent state.
Manual retry of a failed job triggered from the UI hits the regenerate
endpoint with the same deterministic id.

See PRD sections "Frontend transport" and user stories 12–14, 16.

## Acceptance criteria

- [ ] `jobs-context.tsx` seeds from REST then subscribes via SSE
- [ ] Polling fallback kicks in when `EventSource` is undefined
- [ ] Reconnect logic resumes the stream after transient network drops
- [ ] Playwright multi-tab test: open project in two tabs, regenerate from tab A, both tabs show identical live status through terminal
- [ ] Playwright test: refresh mid-run, status restored from REST seed
- [ ] Manual retry button on a failed job re-enqueues and surfaces a new active job
- [ ] `cd frontend && npm run build` and `npx next lint` pass

## Blocked by

- Blocked by `005-sse-stream-and-hub.md`

## User stories addressed

- User story 12
- User story 13
- User story 14
- User story 16

## Implementation note (2026-05-02)

Backend hook + unit tests landed:
- `useProjectJobs(projectId)` hook in `frontend/context/jobs-context.tsx`
- REST seed → SSE subscription with reconnect/backoff/jitter
- Polling fallback (5s) when EventSource undefined OR during SSE reconnect window
- `retry({room_id, variation_id})` POSTs to regenerate endpoint and optimistically inserts
- Merge by `updated_at` (defends against stale REST poll overwriting fresh SSE state)
- Stable session token in localStorage for SSE auth
- 14 vitest tests under jsdom; existing `useJobs`/`JobsProvider` API preserved for back-compat

## Deferred to a follow-up

Playwright multi-tab + refresh-mid-run + manual-retry tests (AC bullets 4, 5, 6 — the
explicit Playwright lines). They require a running backend with the queue infra
provisioned (issue 001 emulator gate). Behavior is pinned by unit tests; Playwright
verification can be added once a `tests/playwright/` integration env exists for
issue 005's SSE endpoint.
