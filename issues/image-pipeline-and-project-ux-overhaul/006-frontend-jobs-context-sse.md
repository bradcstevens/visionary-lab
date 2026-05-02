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
