## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

Wire the new helper, the new context slice, and the new banner into
the project page so clicking Generate enqueues a job (instead of
opening a streaming POST), and the page renders progress from
`/jobs/stream` instead of from a per-request stream. In the same
change, **migrate the existing `frontend/tests/e2e/project-
generation.spec.ts`** so it asserts on the new flow rather than the
dead legacy POST stream — leaving it asserting on the legacy path is
explicitly called out in the PRD as rubber-duck blocking finding #6.

See PRD sections "Frontend changes → `frontend/app/projects/[id]/
page.tsx`" and "Existing `project-generation.spec.ts` migration".

End-to-end behaviour:

- On mount, the project page (already) fetches the project document
  and (already) consumes `jobs-context`. If
  `inFlightProjectGeneration` is non-null, render
  `ProjectGenerationBanner` (from issue 010) bound to the slice.
- The Generate button becomes a small state machine driven off the
  same slice:
  - Idle (slice null): "Generate" — clicking calls
    `enqueueProjectGeneration` (from issue 008). While the POST is
    in flight (~30–90s of inline brief composition), the button
    shows a "Composing brief…" loading state.
  - In-flight (slice non-null): the Generate action is replaced by
    the banner, which carries the Cancel control. The Generate
    button is either hidden or disabled with a "Generating…" label.
  - Cancel (slice non-null and user clicks Cancel in banner): the
    banner's `onCancel` calls
    `jobs-context.cancelProjectGeneration` (from issue 009). UI
    flips to terminal state when the change feed delivers the
    cancelled status.
- The page **stops calling `streamGeneration`** for initial
  generation; the legacy `useGenerationFleet` per-stream silence
  watchdog is no longer registered for the initial-generation path.
  The legacy hook is **kept in place** for the variation-stream
  regen code path — its full removal is out of scope.
- A 180s client-side abort (from issue 008) failing surfaces as a
  user-visible error message ("Couldn't start generation, please
  try again") rather than a silent spinner.
- **Migrate `frontend/tests/e2e/project-generation.spec.ts`**: the
  existing spec mocks `POST /staging/projects/{id}/generate` and
  waits for that call. After the cutover the page never makes that
  call; the spec is rewritten to:
  - Mock / observe `POST /jobs/generate` (the new endpoint).
  - Drive `/jobs/stream` events to simulate progress.
  - Assert the banner renders, the run reaches `succeeded`, and the
    legacy `streamGeneration` POST is never made.

## Acceptance criteria

- [ ] Clicking Generate calls `enqueueProjectGeneration` (issue 008)
      and never calls `streamGeneration` for the initial-generation
      path.
- [ ] When `inFlightProjectGeneration` is non-null, the banner
      renders bound to the slice.
- [ ] The Generate button reflects the slice's state machine: idle
      → composing → in-flight → idle.
- [ ] Cancel in the banner cancels the running job via
      `jobs-context`.
- [ ] No per-stream silence watchdog is registered for initial
      generation (the legacy `useGenerationFleet` integration for
      this path is removed; it stays in place for variation regen).
- [ ] A 180s abort surfaces as a user-visible error, not a frozen
      spinner.
- [ ] `frontend/tests/e2e/project-generation.spec.ts` is migrated to
      assert on `/jobs/generate` + `/jobs/stream`; it no longer
      mocks or waits for `POST /staging/projects/{id}/generate`.
- [ ] `cd frontend && npx vitest run`, `cd frontend && npx next
      lint`, and `cd frontend && npm run build` are green.
- [ ] The migrated `project-generation.spec.ts` passes locally
      against the dev backend.

## Blocked by

- Blocked by `008-frontend-enqueue-helper.md`
- Blocked by `009-frontend-jobs-context-slice.md`
- Blocked by `010-frontend-project-generation-banner.md`

## User stories addressed

Reference by number from the parent PRD:

- User story 5
- User story 6
- User story 7
- User story 8
- User story 10
- User story 11
- User story 12
- User story 13
