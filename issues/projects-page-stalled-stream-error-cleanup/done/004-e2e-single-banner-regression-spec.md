## Parent PRD

`prds/2026-05-02-projects-page-stalled-stream-error-cleanup-prd.md`

## What to build

Add a Playwright e2e spec that lands on a stalled-project fixture and
asserts exactly one `[data-testid="recovery-banner"]` is present in the
DOM, so any future change reintroducing the three-banner stack fails CI
loudly and independently of banner copy. The stalled condition is seeded
by intercepting the project fetch with a Playwright route handler that
returns a fixture project with `status: 'processing'`, zero in-flight
ops, and zero progress — no two-minute watchdog wait, no new
`tests/projects/` fixture-loader plumbing.

See PRD sections "Testing Decisions" and "Regression coverage".

## Acceptance criteria

- [ ] New Playwright spec under the existing e2e suite covers the
      stalled-project landing path
- [ ] Spec uses a `page.route` interceptor for the project fetch; no new
      files under `tests/projects/`
- [ ] Spec asserts `getByTestId('recovery-banner').count() === 1` on the
      stalled-project landing page
- [ ] Spec asserts the visible banner's `data-recovery-kind` matches the
      seeded condition (`interrupted` for the zero-in-flight stalled
      fixture)
- [ ] Spec passes locally via `cd frontend && npx playwright test` and
      against the deployed environment in CI

## Blocked by

- Blocked by `002-recovery-state-classifier-unified-banner.md`
- Blocked by `003-room-status-pill-component-with-stalled-state.md`

## User stories addressed

- User story 15
