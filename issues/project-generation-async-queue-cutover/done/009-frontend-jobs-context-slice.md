## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

Extend `frontend/context/jobs-context.tsx` so it recognises the new
`kind="generate_project"` and surfaces an `inFlightProjectGeneration`
slice to consumers. The context already polls `/jobs?project_id=X` and
subscribes to `/jobs/stream`; this slice teaches the kind-switch in
that context what to do with a project-level job.

See PRD section "Frontend changes → `frontend/context/jobs-context.tsx`".

End-to-end behaviour:

- Context exposes a new `inFlightProjectGeneration` value of shape
  `{ jobId: string, progress: number, phase: string, status: string }
  | null`.
- The slice is non-null when there is exactly one non-terminal
  `kind="generate_project"` job for the current project; null
  otherwise.
- When more than one such job exists (a queued follow-up while one
  is running — supported by the PRD's "multiple project jobs may be
  enqueued concurrently for the same project"), the slice reflects
  the **currently running** job. The queued job is a UI concern for
  a future slice (out of scope here; `inFlightProjectGeneration`
  surfaces the active one and the page renders it).
- A cancel handler exposed on the context issues `DELETE /jobs/{id}`
  for the active project job; the SSE stream picks up the resulting
  status change and the slice flips to null on terminal status.
- Existing handling of `kind="regenerate_variation"` is unchanged.

This slice does NOT render anything — issue 010 builds the banner and
issue 011 wires it into the page.

## Acceptance criteria

- [ ] `inFlightProjectGeneration` is exposed by `jobs-context` with
      the documented shape, non-null only when an in-flight
      `generate_project` job for the project exists.
- [ ] When a `generate_project` job reaches a terminal status
      (succeeded / failed / cancelled), the slice flips to null on
      the next change-feed event.
- [ ] When two `generate_project` jobs exist (one running, one
      queued), the slice reflects the running one.
- [ ] A `cancelProjectGeneration` handler issues `DELETE /jobs/{id}`
      for the slice's `jobId`; status is observed via SSE, not
      optimistically mutated.
- [ ] Existing variation-regen consumers of the context see no
      behavioural change.
- [ ] New vitest unit tests in
      `frontend/context/__tests__/jobs-context.test.tsx` cover:
      `kind=generate_project` job surfaces in
      `inFlightProjectGeneration`; cancel handler issues the DELETE;
      slice flips to null on terminal status.
- [ ] `cd frontend && npx vitest run` is green; `cd frontend && npx
      next lint` is green.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 4
- User story 6
- User story 7
- User story 8
- User story 9
