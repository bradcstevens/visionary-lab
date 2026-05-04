## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

The mandatory end-to-end test that pins the headline behaviour of this
PRD: a page refresh in the middle of a project-generation run does not
kill the run, and the banner reattaches to the in-progress job after
reload.

See PRD section "End-to-end coverage (Playwright)" — refresh-resume is
called out as **mandatory**.

End-to-end behaviour the test exercises:

- Open a freshly created project in the test environment.
- Click Generate; assert the POST hits `/jobs/generate` (NOT the
  legacy `/staging/projects/{id}/generate`) and returns 202 with a
  `job_id`.
- Wait for the first progress event from `/jobs/stream` to land in
  the UI (banner appears with non-zero progress or a phase label).
- `page.reload()`.
- After reload, assert the banner reappears mid-run with current
  progress (i.e., the page recovered the in-flight job state from
  the existing `jobs-context` fetch + SSE reconnect, not by
  re-issuing Generate).
- Assert the run eventually reaches `succeeded` and the banner
  disappears.

Test conventions follow `frontend/tests/e2e/regenerate-variation.spec.ts`
(the prior art for queue-backed E2E tests). The spec should be
deterministic enough to run in CI: use the existing test-data
projects (`tests/projects/...`) and short, mocked image-pipeline
delays where possible.

## Acceptance criteria

- [ ] New `frontend/tests/e2e/project-generation-resume.spec.ts`
      exercises the flow above end-to-end.
- [ ] The spec asserts the Generate click hits `/jobs/generate` and
      not the legacy stream POST.
- [ ] The spec asserts the banner is visible and shows current
      progress after `page.reload()` mid-run.
- [ ] The spec asserts the run reaches `succeeded` after reload.
- [ ] The spec runs across the same browsers configured for the
      existing E2E suite (Chromium and any others currently in
      `playwright.config.ts`).
- [ ] `cd frontend && npx playwright test
      tests/e2e/project-generation-resume.spec.ts` passes locally
      against a backend running the new endpoints, dispatcher, and
      worker.
- [ ] CI Playwright report is captured under `tests/playwright/
      <YYYY-MM-DD-HHMMSS>/` per the repo's local-testing convention.

## Blocked by

- Blocked by `005-generate-project-dispatcher.md`
- Blocked by `006-post-jobs-generate-endpoint.md`
- Blocked by `007-worker-production-entrypoint.md`
- Blocked by `011-frontend-page-wiring-and-spec-migration.md`

## User stories addressed

Reference by number from the parent PRD:

- User story 6
- User story 7
- User story 9
- User story 29
