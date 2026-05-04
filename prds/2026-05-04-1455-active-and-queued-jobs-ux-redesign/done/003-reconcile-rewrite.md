# Reconcile rewrite — derive project status from jobs container

## Parent PRD

`prd.md`

## What to build

A vertical slice that makes the project status truthful: a project that
is genuinely queued and waiting stays `processing`, and only flips to
`failed` when work has truly failed.

End-to-end behavior: `reconcile_project` is split into two functions.
Variation cleanup (with its existing staleness gate) stays in the
original function but no longer mutates project status. A new function
`compute_project_status_from_jobs(project, store)` derives the canonical
status by reading the active job from the jobs container, falling back
to a pure room-derived helper when no active job is present. The buggy
"mixed room statuses ⇒ failed" branch is removed entirely. After this
slice the bug-report scenario is fully resolved.

See "Slice 3 — Reconcile rewrite" in `prd.md` for the full design.

## Acceptance criteria

- [ ] `reconcile_project` no longer mutates `project_data["status"]` on
  any code path. Variation cleanup behavior (with the existing staleness
  gate) is preserved unchanged.
- [ ] New function `compute_project_status_from_jobs(project, store)`:
  - short-circuits when project status is not `processing`,
  - short-circuits when `current_project_job_id` is missing,
  - fetches the active job from the jobs container only when both
    conditions are met,
  - returns `None` (no change) when an active non-terminal job is found,
  - falls back to `_derive_status_from_rooms` otherwise.
- [ ] New pure helper `_derive_status_from_rooms` is a separate testable
  function. The "mixed room statuses ⇒ failed" branch is removed; mixed
  states fall through to `pending`.
- [ ] All four existing callsites — `list_projects`, `get_project`,
  `reset_project`, and the additional callsite at line 963 — gain a
  `Depends(get_job_store)` dependency, call both helpers, and perform a
  single document writeback if either mutated the project.
- [ ] Project failure is now produced **only** by: worker dispatch
  failure, worker poison-queue exhaustion, the cancellation cascade, or
  a producer-side hard error. No reconcile path produces a `failed`
  status.
- [ ] Unit tests for `compute_project_status_from_jobs` covering:
  status≠processing short-circuit, missing `current_project_job_id`
  short-circuit, active non-terminal job → no change, terminal job
  present → derived from rooms, missing job document → derived from
  rooms, mixed-room states → `pending` (not `failed`).
- [ ] Unit tests for `_derive_status_from_rooms` (pure function, table
  driven).
- [ ] Negative-property test on `reconcile_project`: given any input it
  must not mutate `project_data["status"]`.
- [ ] Integration test reproducing the bug-report scenario end-to-end:
  project=`processing`, no `current_project_job_id`, rooms in `pending`
  → status remains `processing`, does not flip to `failed`.
- [ ] Legacy projects without `current_project_job_id` are handled
  implicitly via the room-derived fallback (no migration script needed).
- [ ] Playwright test: a project with stuck pending jobs stays in
  `processing` and does not flip to `failed` after the prior staleness
  window elapses.
- [ ] All checks pass locally: `uv run pytest tests/
  --ignore=tests/integration -v`, `cd frontend && npm run build`, `cd
  frontend && npx next lint`, `cd frontend && npx playwright test`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 9 (queued/waiting projects stay `processing`)
- User story 10 (project flips to `failed` only on genuine failure)
