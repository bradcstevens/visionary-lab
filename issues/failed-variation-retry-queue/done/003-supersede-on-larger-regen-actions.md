# Supersede queued retries when a larger regen action fires

## Parent PRD

`prds/2026-04-30-failed-variation-retry-queue-prd.md`

## What to build

When the user triggers a regen action that subsumes individual
failed-variation retries — Regenerate Room on a room header,
Generate Remaining on the project header, or Regenerate All — the
queued per-variation retries are silently cleared so they do not
fire redundantly after the bigger action completes.

End-to-end behavior to demo: the user has one or more queued
retries on a room (queued indicator visible on the failed
thumbnail). The user clicks Regenerate Room. The queued indicator
disappears, the failed thumbnail enters the processing state from
the room-level regen, and after the room regen completes no
per-variation regen POST fires.

This slice adds three `clear()` call sites at the top of
`handleRegenerateRoom`, `startGeneration`, and `handleRegenerateAll`,
placed BEFORE the existing `isGenerating` guards so the clear
happens even on accidental double-clicks. See PRD sections "Page
integration" (the `clear()` paragraph) and "Testing Decisions" →
scenario 2 ("Supersede on Regenerate Room"). User stories 7 and 8
motivate the change.

## Acceptance criteria

- [ ] `handleRegenerateRoom` calls `clear()` from the
      `useRetryQueue` hook BEFORE its existing `isGenerating` guard.
- [ ] `startGeneration` (the entry point for "Generate Remaining"
      on the project header) calls `clear()` BEFORE its existing
      `isGenerating` guard.
- [ ] `handleRegenerateAll` calls `clear()` BEFORE its existing
      `isGenerating` guard.
- [ ] No new toast or activity-log entry is fired by the supersede
      path itself; the cleared entries simply disappear from the
      UI. (The PRD describes this as "silent.")
- [ ] The Playwright spec
      `frontend/tests/e2e/retry-queue-during-generation.spec.ts`
      gains a third scenario: **Supersede on Regenerate Room** —
      queue a Retry → click Regenerate on the room header → assert
      no per-variation regen POST fires → assert the Queued
      indicator clears and the thumbnail enters the processing
      state from the room regen. Pattern follows the SSE-mocking
      pattern from
      `frontend/tests/e2e/regen-failure-preserves-prior-image.spec.ts`.
- [ ] The hook's existing unit tests (from issue 002) already
      cover `clear()` semantics, so no new hook unit tests are
      required by this slice unless additional edge cases surface
      during implementation.
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

- User story 7
- User story 8
