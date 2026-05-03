## Parent PRD

`prds/2026-05-02-projects-page-stalled-stream-error-cleanup-prd.md`

## What to build

Annotate the watchdog-synthesized SSE `error` event in `useGenerationFleet`
with a top-level `synthetic: true` discriminator, and guard the
project-detail page's `setGenerationError` call so it only fires for real
server-sent error events. After this slice, a watchdog fire still records
the lost op, logs to the activity log, and toasts as today, but it no
longer flips `generationError` and therefore no longer paints the red
destructive banner. The amber lost-op / stale-processing banners that
exist today still render — they are rationalized in slice `002`.

See PRD sections "Watchdog-synthesized event annotation" and the
`useGenerationFleet hook` bullet under "Modules touched".

## Acceptance criteria

- [ ] `SSEEvent` type gains an optional top-level `synthetic?: boolean` field
- [ ] `useGenerationFleet` `finalize` for `reason === 'watchdog'` emits the
      synthesized `error` event with `synthetic: true`
- [ ] Project detail page's SSE `error` handler calls `setGenerationError`
      only when `!event.synthetic`
- [ ] Activity-log entry and toast for the watchdog fire are unchanged
- [ ] Real server-sent `error` events (no `synthetic` field) still set
      `generationError` exactly as today
- [ ] Existing watchdog tests in the fleet hook's test file are extended
      to assert the synthesized event carries `synthetic: true`
- [ ] A page-level component test renders with a fake fleet that fires a
      synthetic error and asserts the destructive red banner is NOT in
      the document

## Blocked by

None - can start immediately.

## User stories addressed

- User story 9
- User story 13

---

## Completion note

Implemented in commit (this slice).

- `StagingStreamEvent` (the actual type name; the PRD/issue called it `SSEEvent`)
  gained an optional top-level `synthetic?: boolean` field, intentionally NOT
  plumbed through the wire serializer.
- `useGenerationFleet` watchdog finalize stamps `synthetic: true` on the
  synthesized error event.
- Project detail page's `'error'` case guards `setGenerationError(...)` with
  `if (!event.synthetic)`. The activity-log entry, toast, and `loadProject()`
  side effects still fire on every error event (real or synthetic), preserving
  the lost-op + activity-log paths exactly as today.
- Existing fleet watchdog test extended to assert `synthetic: true` on the
  synthesized event payload.
- New page-level test
  (`frontend/app/projects/[id]/__tests__/page-synthetic-error.test.tsx`)
  renders `ProjectDetailPage` with mocked dependencies, captures the
  `handleStreamEvent` callback the page hands to `startProject`, drives it
  with a synthetic and a real error, and asserts the destructive
  "Generation encountered an error" banner is absent for synthetic and
  present for real. The mock factory uses `globalThis` stash to side-step
  Vitest's hoisted-mock TDZ rule.

Verification

- `cd frontend && npx vitest run` -> 252 passed (250 baseline + 2 new).
- `cd frontend && npx eslint <changed files>` -> clean on changed files.
  Pre-existing baseline lint findings on `page.tsx` (set-state-in-effect,
  exhaustive-deps, unused-vars) and `services/stagingApi.ts:137` (`data?: any`)
  are unchanged by this slice.
- `cd frontend && npm run build` -> clean, all routes generated.
- Backend pytest skipped per AGENTS.md "run only the loops relevant to the
  files you changed" — this is a frontend-only slice.
- Playwright e2e is the responsibility of issue 004.
