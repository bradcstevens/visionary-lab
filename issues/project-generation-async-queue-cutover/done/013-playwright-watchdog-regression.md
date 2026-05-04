## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

The end-to-end test that pins the watchdog regression closed forever.
This spec proves three things at once — and per rubber-duck blocking
finding #5, all three assertions are required to claim the symptom is
unreachable in the new architecture.

See PRD section "End-to-end coverage (Playwright) → Watchdog
regression".

End-to-end behaviour the test exercises:

- Open a project page in the test environment.
- Click Generate.
- **Assertion 1**: Network observer confirms the click triggers a
  POST to `/jobs/generate` and **does NOT** trigger a POST to the
  legacy `/staging/projects/{id}/generate`.
- **Assertion 2**: The page does **NOT** call `streamGeneration` for
  the initial-generation path. (Confirmed via either: a network-
  level assertion on the absence of the legacy stream POST, or a
  test-instrumentation assertion that the legacy `useGenerationFleet`
  per-stream watchdog is not registered for the project-generation
  job.)
- **Assertion 3**: A silent `/jobs/stream` (no SSE events for at
  least 130 seconds — comfortably above the legacy 120s watchdog
  threshold) does **NOT** surface a "Generation stalled" or
  "stream lost" recovery banner over the in-flight project.
- Without all three assertions, the test claims something it
  doesn't prove. Each assertion is explicit and named.

Test conventions match issue 012. The 130s silent window can be
mocked at the SSE level so the spec runs in deterministic CI time
(no real wall-clock 130s wait). The intent is to pin behaviour, not
to actually idle the run.

## Acceptance criteria

- [ ] New `frontend/tests/e2e/project-generation-watchdog-
      regression.spec.ts` exercises the flow above.
- [ ] Assertion 1 (POST goes to `/jobs/generate`, not the legacy
      `/staging/projects/{id}/generate`) is explicit and named.
- [ ] Assertion 2 (page does not call `streamGeneration` / no
      per-stream watchdog registered for initial generation) is
      explicit and named.
- [ ] Assertion 3 (130s of silent `/jobs/stream` produces no
      stalled / stream-lost recovery banner) is explicit and
      named, and uses mocked / fast-forwarded time so CI runs
      quickly.
- [ ] The spec runs across the same browsers configured for the
      existing E2E suite.
- [ ] `cd frontend && npx playwright test
      tests/e2e/project-generation-watchdog-regression.spec.ts`
      passes locally.
- [ ] CI Playwright report is captured under `tests/playwright/
      <YYYY-MM-DD-HHMMSS>/` per the repo's local-testing
      convention.

## Blocked by

- Blocked by `011-frontend-page-wiring-and-spec-migration.md`

## User stories addressed

Reference by number from the parent PRD:

- User story 11
- User story 12
- User story 13
- User story 30
