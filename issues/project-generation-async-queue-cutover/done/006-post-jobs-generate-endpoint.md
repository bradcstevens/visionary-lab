## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

A new `POST /api/v1/staging/projects/{id}/jobs/generate` endpoint that
enqueues a `kind="generate_project"` job. This is the user-facing
"Generate becomes a fire-and-forget action" surface from the PRD.
Brief composition runs inline in the handler (~30–90s blocking) and
the composed prompts are stashed in the job payload so a worker retry
doesn't recompute them.

See PRD section "`POST /api/v1/staging/projects/{id}/jobs/generate`".

End-to-end behaviour:

- Endpoint is gated by the existing async-queue feature flag; with
  the flag off, return `503` (consistent with other queue-backed
  endpoints).
- Body shape: `{"regenerate_all": bool}` (defaults to false when
  body is absent).
- 404 if the project doesn't exist.
- Inline brief composition (`brief_to_prompts(...)`); failures bubble
  out as a 5xx without leaving a half-created job behind in
  `JobStore`. (Test pin: brief failure path asserts no job doc was
  written.)
- When `regenerate_all=true`: pre-cancel any in-flight
  `regenerate_variation` jobs for the project via the cascade-cancel
  helper from issue 002. This prevents a late-arriving variation
  success from writing state into a project that has just been
  cleared (rubber-duck blocking #4).
- Create the job via
  `JobStore.create_job(project_id, room_id="__project__",
  variation_id="__project__", revision=uuid.uuid4().hex,
  kind="generate_project", payload={"regenerate_all": ...,
  "brief_prompts": ...})`.
- **UUID4 (not an integer revision counter) is intentional.** Two
  concurrent POSTs each compute a distinct UUID4, both produce
  distinct job documents, both get enqueued. An integer-counter
  read-then-create would race and silently collapse the second click
  via JobStore's idempotent insert (rubber-duck blocking #2).
- Enqueue via `JobQueue.enqueue(job_id=doc["id"],
  project_id=project_id)` and return `{"job_id": ...}` with status
  202.
- Endpoint MUST NOT stream — the response is the job id and HTTP
  returns immediately after enqueue.

## Acceptance criteria

- [ ] `POST /api/v1/staging/projects/{id}/jobs/generate` exists and
      returns 202 with `{"job_id": ...}` on the happy path.
- [ ] Endpoint is gated by the async-queue feature flag (503 when
      off).
- [ ] Missing project returns 404 with no side effects.
- [ ] Brief composition runs inline; on failure, no job doc is
      created in `JobStore`.
- [ ] Job is created with `kind="generate_project"`,
      `room_id="__project__"`, `variation_id="__project__"`,
      `revision=uuid.uuid4().hex`, and a payload that includes both
      `regenerate_all` and the precomputed `brief_prompts`.
- [ ] Job is enqueued via `JobQueue.enqueue(...)`.
- [ ] When `regenerate_all=true`, in-flight non-terminal
      `regenerate_variation` jobs for the project are cascade-
      cancelled before the new job is created (rubber-duck blocking
      #4 regression).
- [ ] **Two concurrent POSTs produce two distinct job documents.**
      Test simulates the race and asserts both `job_id`s differ and
      both docs exist (rubber-duck blocking #2 regression).
- [ ] New tests in `tests/test_staging_endpoints_generate_jobs.py`
      cover: happy path; concurrent POST race; `regenerate_all`
      payload propagation; cascade-cancel on regenerate_all; 404 on
      missing project; 503 with feature flag off; brief failure
      doesn't leave a half-created job.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` is green.

## Blocked by

- Blocked by `002-project-lease-and-cascade-cancel.md`

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 19
- User story 21
- User story 22
- User story 23
- User story 26
- User story 27
- User story 31 (regression coverage for double-click + regen-all
  pre-empts variation jobs)
