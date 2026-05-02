## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`SSEHub` — a per-replica in-memory pub/sub that subscribes to the Cosmos
change-feed via `JobStore` and fans events out to connected EventSource
clients filtered by `project_id`. New endpoint
`GET /api/v1/staging/projects/{id}/jobs/stream` exposes the stream over
SSE. Auth via session cookie or `?access_token=`. Soft cap of 10 streams
per session; 429 beyond. Response headers set `Cache-Control: no-cache,
no-transform` and `X-Accel-Buffering: no` so Front Door does not buffer.
15s heartbeat comment line keeps the connection warm.

See PRD sections "SSEHub", "API contracts", and "Frontend transport".

## Acceptance criteria

- [ ] `SSEHub` subscribes once per replica to the Cosmos change-feed and routes events by `project_id`
- [ ] `/jobs/stream` endpoint streams `text/event-stream` with documented headers and 15s heartbeats
- [ ] Auth accepts session cookie OR `?access_token=`
- [ ] More than 10 concurrent streams per session returns 429
- [ ] Integration test: open stream → enqueue job via REST → observe ordered state events through to terminal
- [ ] Disconnect cleanly drops the subscription

## Blocked by

- Blocked by `004-rest-enqueue-list-cancel.md`

## User stories addressed

- User story 13
- User story 14
