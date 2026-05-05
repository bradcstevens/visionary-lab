# Producer dedupe + error classification + frontend idempotency-key

## Parent PRD

`prd.md`

## What to build

A vertical slice that prevents duplicate jobs on rapid clicks and
classifies producer-side errors into a stable `ErrorKind` enum that the
front-end can map to friendly copy.

End-to-end behavior: the front-end mints `crypto.randomUUID()` per
`enqueueProjectGeneration` call inside the service layer and sends it as
`Idempotency-Key`. The producer uses this key as the deterministic job
id revision so a retried request collapses into the existing job. A
lease precheck against `current_project_job_id` catches the case where a
second click arrives during the in-flight window of a different
idempotency-key. The producer responds with `200 { already_in_flight:
true, job_id }` for both dedupe paths and `202 { job_id,
already_in_flight: false }` for new work. Errors return a structured
`{ error_kind, user_message, detail }` body.

See "Slice 2 — Producer dedupe, error classification, frontend
idempotency-key" in `prd.md` for the full design, including the
`project_generation_producer` and `job_errors` deep modules, the
discriminated-union return type, and the `ErrorKind` enum values.

## Acceptance criteria

- [ ] New deep module `project_generation_producer` returns a
  discriminated union: `AlreadyInFlight(job_id)`, `NewlyEnqueued(job_id)`,
  or `EnqueueFailed(error_kind, http_status, user_message)`.
- [ ] HTTP endpoint becomes a thin wrapper that translates the union
  into 200 / 202 / 4xx-5xx responses with the contract shapes defined
  in `prd.md` "Cross-slice contracts".
- [ ] Idempotency-Key header is extracted, validated, and used as the
  revision component of the deterministic job id
  (`{project_id}:project:project:{idempotency_key}`).
- [ ] `create_job` 409 collision (Cosmos `If-None-Match: *`) is
  interpreted as an idempotent retry → `200 { already_in_flight: true,
  job_id: existing }`.
- [ ] Lease precheck reads `current_project_job_id`; if set and
  referenced job is non-terminal, returns
  `200 { already_in_flight: true, job_id: holder }` without composing
  the brief or creating a new job.
- [ ] Brief composition is gated behind both dedupe checks (only runs
  on the new-work path).
- [ ] CAS lease acquire on the success path uses the existing
  idempotent `acquire_project_lease` ("holder is me" re-acquire).
- [ ] New deep module `job_errors` exposes a single classifier function:
  `(exception) -> (error_kind, user_message, http_status)`.
- [ ] `ErrorKind` enum has exactly five values: `QUEUE_PERMISSION`,
  `BRIEF_FAILED`, `STORE_FAILED`, `UNAVAILABLE`, `UNKNOWN`.
- [ ] Azure Core `ClientAuthenticationError` and `HttpResponseError`
  carrying `AuthorizationPermissionMismatch` map to `QUEUE_PERMISSION`
  with a message naming the missing role.
- [ ] LLM brief-composition errors → `BRIEF_FAILED`; Cosmos write
  errors → `STORE_FAILED`; everything else → `UNKNOWN`.
- [ ] Job document schema gains optional `error_kind: Optional[ErrorKind]`
  alongside the existing `error: { type, message }` substructure.
- [ ] The worker writes `error_kind` on terminal-failure transitions
  (dispatch failure, poison-queue exhaustion).
- [ ] The producer writes `error_kind` on enqueue-failure terminal
  states.
- [ ] Front-end `enqueueProjectGeneration` service helper mints
  `crypto.randomUUID()` per call inside the service layer (callers do
  not pass it in), sets the `Idempotency-Key` header, and parses the
  `{ job_id, already_in_flight }` response shape.
- [ ] Front-end error parsing extracts `error_kind` from the response
  body so downstream UX (in `004-…`) can branch on it.
- [ ] Unit tests: table-driven classifier tests covering all five
  `ErrorKind` mappings (including both `HttpResponseError` and
  `ClientAuthenticationError` shapes for `QUEUE_PERMISSION`); fake-driven
  producer tests covering first-time enqueue, same-idempotency-key
  retry, lease-held-different-key, queue-enqueue exception, brief
  exception, and CAS conflict during lease acquire.
- [ ] Playwright test: rapid double-click results in exactly one job
  being created.
- [ ] All checks pass locally: `uv run pytest tests/
  --ignore=tests/integration -v`, `cd frontend && npm run build`, `cd
  frontend && npx next lint`, `cd frontend && npx playwright test`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 3 (second click silently absorbed, no duplicate job)
- User story 23 (worker writes structured `error_kind` on terminal
  failure)
- User story 28 (producer dedupe is a single function with
  discriminated-union return)
- User story 29 (error classifier is a single pure function with enum
  return)

User stories 4 (toast for second click) and 21 (specific
`QUEUE_PERMISSION` user copy) are landed by issue `004-…`, which
consumes the contract this issue establishes.
