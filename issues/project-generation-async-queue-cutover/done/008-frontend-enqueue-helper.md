## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

Add a frontend service helper for the new POST endpoint from issue
006. This slice gives the page a typed call site with a client-side
abort timeout sized to fail loud rather than spin forever on inline
brief-composition stalls.

See PRD sections "Frontend changes → `frontend/services/stagingApi.ts`"
and "Operational risks → frontend abort timeout".

End-to-end behaviour:

- New `enqueueProjectGeneration(projectId: string, options?:
  { regenerateAll?: boolean }): Promise<{ job_id: string }>` in
  `frontend/services/stagingApi.ts`.
- POSTs to `/api/v1/staging/projects/{projectId}/jobs/generate` with
  body `{ regenerate_all: !!options?.regenerateAll }`.
- Response is parsed JSON `{ job_id }`; non-2xx responses raise with
  the body included in the error message (matching how other
  helpers in `stagingApi.ts` surface errors).
- **180-second client-side abort timeout** via `AbortController`.
  This is comfortably above expected P99 inline-brief latency and
  below typical Azure Front Door defaults (~240s) so the UI fails
  loud rather than spinning indefinitely. Abort surfaces as a
  recognizable error (e.g. `EnqueueGenerationTimeoutError`) the page
  can render a "couldn't reach generation; try again" message
  against in issue 011.
- No streaming, no SSE — this is a plain JSON POST that returns the
  job id.
- The helper is callable from any consumer; this slice does NOT wire
  it into the page (issue 011 does that).

## Acceptance criteria

- [ ] `enqueueProjectGeneration` is exported from
      `frontend/services/stagingApi.ts`.
- [ ] It POSTs to the correct endpoint path with body
      `{ regenerate_all: ... }`.
- [ ] It returns `{ job_id }` on 2xx and rejects with a descriptive
      error on non-2xx.
- [ ] A client-side `AbortController` aborts the fetch at 180s and
      the rejection has a recognizable shape (e.g. a typed error or
      a `name === "EnqueueGenerationTimeoutError"` field).
- [ ] New vitest unit tests cover: happy path; non-2xx error
      surfaced; abort at 180s produces the timeout error.
- [ ] No call sites are wired yet; `npm run build` and `next lint`
      stay green.
- [ ] `cd frontend && npx vitest run` and `cd frontend && npx next
      lint` are green.

## Blocked by

- Blocked by `006-post-jobs-generate-endpoint.md`

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 26
