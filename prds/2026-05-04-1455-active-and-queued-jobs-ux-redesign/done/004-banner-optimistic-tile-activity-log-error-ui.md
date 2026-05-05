# Banner, optimistic tile, activity log, error UI

## Parent PRD

`prd.md`

## What to build

A vertical slice that gives the user continuous, friendly, accurate
visibility from the moment they click Generate, including a synchronous
banner with phased copy, an optimistic project tile on enqueue, a toast
on dedupe, an activity log driven by SSE phase transitions, and
human-readable error copy mapped from the `ErrorKind` contract
established in `002-…`.

End-to-end behavior: clicking Generate immediately renders a banner
("Composing design brief…" → "Submitting to queue…" at 14s). On a 202
response, an optimistic project tile renders before the SSE seed
catches up. On a 200 (`already_in_flight: true`) response, the banner
is suppressed in favor of a "Generation already in progress" toast plus
a scroll-to-existing-tile. The activity log gains an entry on every
meaningful phase transition (no progress-percentage spam). Errors render
a banner with friendly copy mapped from `error_kind`, with an
expandable "Show technical details" section for raw exception
diagnostics.

See "Slice 5 — Banner, optimistic tile, activity log, error UI" in
`prd.md` for the full design, including the
`activity-log-derivation` and `error-kind-copy` deep modules and the
copy table.

## Acceptance criteria

- [ ] `startGeneration` callback in the project page renders a banner
  synchronously on click (replacing today's silent `isEnqueueing`
  state).
- [ ] Banner copy is phased: 0–14 seconds reads "Composing design
  brief…"; from 15 seconds onward reads "Submitting to queue…". No
  fake progress percentage is shown.
- [ ] On a 202 response, an optimistic project tile renders immediately
  in the room grid so the user sees their work before the SSE seed
  catches up.
- [ ] On a 200 (`already_in_flight: true`) response, the banner is
  suppressed in favor of a small toast "Generation already in
  progress", and the page scrolls to the existing tile.
- [ ] New front-end deep module `activity-log-derivation` exposes a
  pure function `deriveLogEntries(prev, current) -> LogEntry[]` that:
  - emits an entry on phase changes (extracted from each job's `phase`
    field),
  - drops progress-percentage tick events,
  - includes heartbeat-stale warnings sourced from the staleness
    detector contract that issue `005-…` will consume.
- [ ] Activity log entries are in-memory only and reset on page reload
  (job tile state is preserved by the existing SSE seed restoration —
  no new restore work in this slice).
- [ ] New front-end deep module `error-kind-copy` exposes a mapping
  `ErrorKind -> { userMessage, retryable, showAdminContact }` for all
  five enum values plus the unknown-kind fallback ("Couldn't start
  generation, try again").
- [ ] The `QUEUE_PERMISSION` user message is developer-targeted and
  names the specific Azure role needed.
- [ ] Recovery banner gains an expandable "Show technical details"
  section, collapsed by default, that displays the raw `error.message`
  and `error.type` fields.
- [ ] Unit tests for `activity-log-derivation` (pure function): no
  change → no entries; phase change → one entry;
  progress-percentage-only change → no entries (suppression); terminal
  failure with `error_kind` → one entry with mapped copy; multiple jobs
  simultaneously → multiple entries.
- [ ] Unit tests for `error-kind-copy`: table-driven mapping for all
  five `ErrorKind` values plus the unknown-kind fallback.
- [ ] Playwright tests: happy path (click → banner → optimistic tile →
  activity log entries → completion); error path (forced RBAC error →
  banner with `QUEUE_PERMISSION` copy plus expandable raw details);
  dedupe path (200 `already_in_flight` → toast and no banner).
- [ ] All checks pass locally: `uv run pytest tests/
  --ignore=tests/integration -v`, `cd frontend && npm run build`, `cd
  frontend && npx next lint`, `cd frontend && npx playwright test`.

## Blocked by

- Blocked by `002-producer-dedupe-error-classification.md` — depends on
  the producer 200/202/4xx response shape, the `Idempotency-Key`
  service-layer wiring, and the `ErrorKind` enum contract.

## User stories addressed

Reference by number from the parent PRD:

- User story 1 (immediate visual feedback on click)
- User story 2 (banner tells the user what the system is doing —
  phased copy)
- User story 4 (second click produces a "generation already in
  progress" toast)
- User story 11 (activity log updates as worker progresses through
  phases)
- User story 12 (activity log omits progress-percentage ticks)
- User story 20 (user-friendly error explanation)
- User story 21 (specific recognized-error copy that names the cause,
  e.g. Azure permissions)
- User story 22 (expandable "Show technical details" section in error
  banner)
- User story 25 (job tile state preserved across reloads via existing
  SSE seed)
- User story 27 (activity-log derivation is a pure function for
  testing)
