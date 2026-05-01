# Live "In Flight (N)" panel inside the activity log

## Parent PRD

`prds/2026-04-30-projects-page-improvements-prd.md`

## What to build

Add an "In Flight (N)" section pinned to the top of the activity
log panel, above the chronological log. It lists every active
operation (full-project generation, per-room generation, per-
variation regenerate or edit-prompt) with its label and a live-
updating elapsed timer. State is derived entirely from the
`useGenerationFleet` hook (slice 007) — no new backend endpoint,
no server-side queue persistence.

End-to-end behavior:

- Frontend: a new store `ActivityFeed` consolidates today's
  `activity-log-context` and the new In Flight section. Exposes:
  - `entries`: chronological log (existing behavior, minus the
    auto-open removed in slice 006)
  - `inFlight`: derived list of active operations with `id`,
    `label`, `kind` (`project | room | variation | edit-prompt`),
    `startedAt`
  - `log(entry)`, `startOp(op)`, `endOp(opId)` mutators
  - The `useGenerationFleet` hook from slice 007 calls
    `startOp`/`endOp` as it manages SSE streams.
- The `ActivityLogPanel` renders an "In Flight (N)" section above
  the chronological log when `inFlight.length > 0`, with one row
  per operation showing the operation's label and a live elapsed
  timer.
- Operations appear in the In Flight section the moment the user
  clicks Generate (before the first SSE event lands). Operations
  waiting on a backend semaphore slot show with an honest "queued"
  or "starting" label so the user does not double-click. Completed
  operations drop out of In Flight and remain in the chronological
  log below.
- Backend: no change. State is purely client-side derived. If the
  page refreshes mid-generation, the In Flight section starts
  empty — the underlying generations on the server side are
  unaffected and continue via the existing stale-processing
  reconcile path.
- Tests: a Playwright scenario kicks off multiple generations and
  asserts the In Flight (N) section renders with the correct count
  and per-operation rows; opens the panel manually (auto-open is
  gone from slice 006); confirms completed operations leave the
  In Flight section and remain in the chronological log.

See PRD sections **"Solution → 8. Live 'In Flight' panel inside
the activity log"**, **"Implementation Decisions → Frontend
modules"** (`ActivityFeed` store and panel-rendering bullets),
**"Cross-cutting decisions"** (purely client-side derived,
no server-side queue persistence), and **"Testing Decisions →
Frontend tests"** (concurrent-generation scenario assertion that
the panel shows 3 entries).

## Acceptance criteria

- [ ] A new `ActivityFeed` store consolidates today's activity-
      log-context with the new In Flight surface. Exposes
      `entries`, `inFlight` (derived list with `id`, `label`,
      `kind`, `startedAt`), and `log` / `startOp` / `endOp`
      mutators.
- [ ] The `useGenerationFleet` hook from slice 007 calls
      `startOp` when an operation begins and `endOp` when it
      terminates (success, failure, abort, or watchdog fire) — the
      In Flight panel and the per-operation flags driving buttons
      cannot drift apart.
- [ ] The activity log panel renders an "In Flight (N)" section
      pinned above the chronological log when
      `inFlight.length > 0`. Each row shows the operation's label
      and a live-updating elapsed time. The chronological log
      below is unchanged in rendering.
- [ ] Operations appear in In Flight on click (before the first
      SSE event), so the panel acknowledges intent immediately.
- [ ] Operations waiting on a backend semaphore slot render with
      an honest "queued" or "starting" label.
- [ ] Completed operations drop out of the In Flight section and
      remain in the chronological log below.
- [ ] No backend changes. No server-side queue persistence — on
      page refresh the In Flight section starts empty (the under-
      lying server-side generations are unaffected).
- [ ] A new Playwright spec covers: kick off three rooms
      concurrently; manually open the activity log (per slice 006,
      auto-open is gone); assert the In Flight (N) section shows
      3 entries with the correct labels and live elapsed timers
      ticking up; let one complete; assert it drops out of In
      Flight and appears in the chronological log below.
- [ ] Local checks pass before commit:
      `cd frontend && npx playwright test`,
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

- Blocked by `007-generation-fleet-hook-and-watchdog.md` (the
  In Flight section derives its state entirely from
  `useGenerationFleet`).

## User stories addressed

Reference by number from the parent PRD:

- User story 30
- User story 31
- User story 32
- User story 33
