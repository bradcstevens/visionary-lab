# Cancel-all endpoint, staleness detector, header subline

## Parent PRD

`prd.md`

## What to build

A vertical slice that lets the user notice when generation has stalled
and recover with a single click. Adds a cancel-all-jobs HTTP endpoint, a
front-end staleness detector with two thresholds, a dynamic explanatory
subline beneath the existing header counters, and a one-tap "Cancel
queued jobs" recovery button that cascades cancellation across every
non-terminal job for the project.

End-to-end behavior: while a project is generating, the front-end
tracks `lastEventByJobId` and runs a 5-second `setInterval` that
invokes `computeStaleness(jobs, lastEventByJobId, now)`. At 45 seconds
of pickup or stalled-running silence, the header subline switches to a
soft warning. At 120 seconds it escalates to a hard warning and a
one-tap "Cancel queued jobs" button appears (no confirmation dialog).
Click cascades cancellation through `_cascade_cancel_project_jobs`,
disables the button with a "Cancelling…" subline, and either dismisses
on SSE confirmation or falls back to a 10-second toast explaining the
worker will pick up the cancellation when it comes online.

See "Slice 4 — Cancel-all endpoint, staleness detector, header
subline" in `prd.md` for the full design, including the copy table,
the `job-staleness` deep module signature, and the cancel response
shape.

## Acceptance criteria

- [ ] New HTTP endpoint `DELETE /staging/projects/{project_id}/jobs`
  returns `202 { status: "accepted", cancelled_count, project_id }`
  and reuses the existing `_cascade_cancel_project_jobs` helper.
- [ ] Endpoint is idempotent: already-terminal projects return
  `cancelled_count: 0` (no error).
- [ ] New front-end deep module `job-staleness` exposes a pure function
  `computeStaleness(jobs, lastEventByJobId, now) -> StalenessState[]`.
- [ ] Detector A (pickup): uses `now - job.created_at` for jobs in
  `pending` status. Returns `fresh | soft | hard` at thresholds 45s and
  120s.
- [ ] Detector B (stalled): uses `now - lastEventByJobId[id]` for jobs
  in `running` status. Returns `fresh | soft | hard` at thresholds 45s
  and 120s.
- [ ] Jobs context tracks `lastEventByJobId`, updated on every
  `event: job` SSE delivery.
- [ ] Jobs context runs a 5-second `setInterval` (no pause on hidden
  tab) that invokes `computeStaleness` and exposes the result as a
  hook value.
- [ ] Project page header keeps its existing active/queued/running
  counters and gains a dynamic subline driven by the staleness state,
  using the copy table from `prd.md` Slice 4.
- [ ] "Cancel queued jobs" button appears only at the 120s hard
  threshold, is one-tap (no confirmation dialog), and always cascades
  across every non-terminal job for the project.
- [ ] Cancel click flow: button disabled + spinner + "Cancelling…"
  subline; SSE confirmation dismisses the banner and shows a success
  toast; if no SSE confirmation arrives within 10 seconds the banner
  dismisses with a fallback toast explaining the cancellation was
  queued.
- [ ] The same banner UX is reused during the embedded-worker startup
  race (the brief window after lifespan startup before the worker has
  actually polled the queue).
- [ ] Unit tests for `computeStaleness` (pure function, table driven):
  fresh state; soft threshold A (pending 45s+); soft threshold B
  (running silent 45s+); hard threshold A (pending 120s+); hard
  threshold B (running silent 120s+); fresh transition (event arrives,
  state resets); empty jobs array.
- [ ] Playwright test: the 120s staleness state shows the cancel-queued
  button; clicking it cascades cancellation and the banner dismisses
  on SSE confirmation.
- [ ] All checks pass locally: `uv run pytest tests/
  --ignore=tests/integration -v`, `cd frontend && npm run build`, `cd
  frontend && npx next lint`, `cd frontend && npx playwright test`.

## Blocked by

- Blocked by `002-producer-dedupe-error-classification.md` — depends
  on the producer/error contract for cancel responses and shared
  error handling.
- Blocked by `004-banner-optimistic-tile-activity-log-error-ui.md` —
  reuses the banner UX and toast affordances introduced there
  ("Cancelling…" subline, success/fallback toasts, embedded-worker
  startup-race banner reuse).

## User stories addressed

Reference by number from the parent PRD:

- User story 13 (visible warning in header within 45 seconds when
  worker stops responding)
- User story 14 (clear "Cancel queued jobs" button at 120s of
  staleness)
- User story 15 (cancel cascades across every non-terminal job in a
  single click)
- User story 16 (immediate visual feedback when cancel is clicked)
- User story 17 (cancel while worker offline — banner dismisses within
  10 seconds with fallback toast)
- User story 18 (header keeps active/queued counters)
- User story 19 (header gains a dynamic explanatory subline)
- User story 26 (staleness detector is a pure function for testing)
