# Drop queued retries when the global generation stream errors

## Parent PRD

`prds/2026-04-30-failed-variation-retry-queue-prd.md`

## What to build

When the global staged-generation SSE stream itself terminates with
an `'error'` event (the page already shows an error banner in this
case), drop all queued retries so the system does not immediately
fire N more requests against the same broken upstream. The Retry
buttons on the failed variations are restored to clickable state so
the user can re-trigger manually after acknowledging the error.

End-to-end behavior to demo: the user has at least one queued
retry. The global generation stream emits an `error` event. The
existing error banner appears. The Queued indicator on the failed
variation disappears and the Retry button is restored. No
per-variation regen POST fires.

This slice adds a single `clear()` call inside `handleStreamEvent`'s
`'error'` case, AFTER the existing error handling so the banner
still renders normally. See PRD section "Page integration" (the
last sentence about the `'error'` case) and "Testing Decisions" →
scenario 4 ("Drop on global error"). User story 9 motivates the
change.

## Acceptance criteria

- [ ] `handleStreamEvent`'s `'error'` case calls `clear()` from
      the `useRetryQueue` hook AFTER the existing error-handling
      logic (the error banner and any existing toast/log entries
      still render exactly as today).
- [ ] No new toast is fired by the drop path itself; the existing
      error banner already communicates the failure to the user.
- [ ] An info-level activity-log entry is added when the queue is
      non-empty at the moment of the drop, following the existing
      copy/icon conventions in `app/projects/[id]/page.tsx`. (No log
      entry when the queue is empty, to avoid noise on every error.)
- [ ] After the drop, the Retry button is restored on the failed
      variation thumbnail (the Queued indicator was removed because
      `queuedVariationIds.has(variation.id)` is now false).
- [ ] The Playwright spec
      `frontend/tests/e2e/retry-queue-during-generation.spec.ts`
      gains a fourth scenario: **Drop on global error** — queue a
      Retry → emit `error` event on the global stream → assert the
      error banner appears → assert the Queued indicator clears and
      the Retry button is restored → assert no variation regen POST
      fires.
- [ ] No backend changes. No changes to the
      `streamVariationRegeneration` API service contract.
- [ ] Local checks pass before commit:
      `uv run pytest tests/ --ignore=tests/integration -v`,
      `cd frontend && npx playwright test` (full E2E suite),
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

- Blocked by `002-retry-queue-core-with-queued-indicator.md`
  (the queue and `clear()` method must exist).

## User stories addressed

Reference by number from the parent PRD:

- User story 9
