## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`ProgressEstimator` — synthetic 3-phase progress calibrated against a
rolling p50 per `(model, kind)` cached in a Cosmos `stats` doc. Phases:
queued 0–10%, generating 10–90%, finalizing 90–100%. Sane default on
cold start. Output is monotonic non-decreasing within a single job. The
`JobWorker` calls the estimator on a timer during the generating phase
and writes `phase` + `progress` to `JobStore`; SSE delivers the updates
to clients.

See PRD section "ProgressEstimator" and user stories 25–26.

## Acceptance criteria

- [ ] `ProgressEstimator` reads/writes the `stats` doc and updates p50 on each completed job
- [ ] Cold-start fallback returns sensible defaults
- [ ] Output never decreases within a single job
- [ ] `JobWorker` writes `phase` and `progress` updates while running; SSE clients receive them in order
- [ ] Unit tests cover phase boundaries, p50 update, cold-start fallback, monotonicity
- [ ] `stats` doc is seeded only by new jobs (no historical replay)

## Blocked by

- Blocked by `005-sse-stream-and-hub.md`

## User stories addressed

- User story 25
- User story 26
