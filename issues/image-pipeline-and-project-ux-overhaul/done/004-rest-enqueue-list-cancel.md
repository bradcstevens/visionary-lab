## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

REST surface that lets the frontend enqueue, observe, and cancel jobs,
plus pipeline integration that switches `staging_pipeline.py` from
inline-await to enqueue when `FEATURE_ASYNC_QUEUE` is on (in-process
generation kept as fallback for one release).

Endpoints:

- `POST   /api/v1/staging/projects/{id}/jobs/regenerate` — enqueues one job per variation (or per supplied filter), returns `{job_ids}`
- `GET    /api/v1/staging/projects/{id}/jobs` — lists jobs with status + progress
- `DELETE /api/v1/staging/jobs/{job_id}` — sets `cancel_requested`

See PRD sections "API contracts (additions)" and "Feature flags".

## Acceptance criteria

- [ ] All three endpoints implemented in `backend/api/endpoints/staging.py`
- [ ] `staging_pipeline.py` enqueues via `JobQueue` instead of inline awaiting when `FEATURE_ASYNC_QUEUE=true`
- [ ] Regenerate endpoint produces deterministic job ids and is idempotent on retry
- [ ] Cancel endpoint flips `cancel_requested` and returns 202 even if the job has already completed
- [ ] `FEATURE_ASYNC_QUEUE` defaults to true in dev/staging; flag gating documented
- [ ] API tests cover happy path + idempotent regenerate + cancel-after-terminal

## Blocked by

- Blocked by `003-jobworker-consumer.md`

## User stories addressed

- User story 12
- User story 16
- User story 17
