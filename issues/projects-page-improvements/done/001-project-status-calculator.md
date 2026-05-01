# Project status calculator + truthful project status badge

## Parent PRD

`prds/2026-04-30-projects-page-improvements-prd.md`

## What to build

Replace the duplicated, drift-prone inline project-status recalculations
that live in three regen endpoints with a single pure helper that all
regen paths call. The user-facing fix: when one of N rooms finishes its
variations and the rest are still pending, the project header badge
must keep reading `pending`, not flip to `completed`.

End-to-end behavior:

- Backend: a new pure helper
  `ProjectStatusCalculator.compute_status(rooms) -> ProjectStatus`
  becomes the single source of truth. Returns `pending` if any room is
  `pending` or `processing`, `failed` if every room reached a terminal
  state but none completed, otherwise `completed`. Called from every
  existing regen path (`generate_project`, `regenerate_room`,
  `regenerate_variation`). The status field continues to persist on
  every regen finish so the badge stays correct after refresh.
- Frontend: no change. The badge already reads `project.status`.
- Tests: exhaustive table-driven unit tests on the calculator (incl.
  the explicit Issue 1 bug case: 1 completed + 4 pending must return
  `pending`) plus a Playwright multi-room scenario asserting the
  badge transitions correctly.

See PRD sections **"Solution → 1. Truthful project status"**,
**"Implementation Decisions → Backend modules"** (first bullet), and
**"Testing Decisions → Backend unit tests"**
(`tests/test_project_status_calculator.py`).

This slice is a foundational piece for slice 004 (which uses the
calculator from the new edit-prompt path).

## Acceptance criteria

- [ ] A new module exposes
      `ProjectStatusCalculator.compute_status(rooms) -> ProjectStatus`.
      The function is pure (no I/O, no mutation).
- [ ] Status rules per PRD: `pending` if any room is `pending` or
      `processing`; `failed` if every room is in a terminal state and
      none completed; `completed` otherwise.
- [ ] Edge cases handled explicitly: zero rooms; single-room project;
      mixed completed + failed terminal (returns `completed` since at
      least one completed).
- [ ] All three existing regen paths (`generate_project`,
      `regenerate_room`, `regenerate_variation`) invoke
      `ProjectStatusCalculator.compute_status` to produce the project
      status. The duplicated inline branches are deleted, not left
      dead.
- [ ] Project status continues to be persisted on every regen finish.
- [ ] `tests/test_project_status_calculator.py` is added with table-
      driven coverage: all-pending, all-completed, all-failed,
      mixed pending+completed (must return `pending`), mixed
      completed+failed terminal, zero-rooms, single-room cases. The
      Issue 1 bug case (1 completed + 4 pending → `pending`) is
      asserted explicitly.
- [ ] `tests/test_staging_pipeline.py` is extended with a regression
      test for the Issue 1 bug case to prove all regen paths now
      delegate to the calculator.
- [ ] A new Playwright spec covers the multi-room scenario: project
      with 5 rooms; generate one room; assert badge stays `pending`;
      generate the remaining four; assert badge transitions to
      `completed`.
- [ ] No frontend code change beyond the new Playwright spec — the
      badge already binds to `project.status`.
- [ ] Local checks pass before commit:
      `uv run pytest tests/ --ignore=tests/integration -v`,
      `cd frontend && npx playwright test`,
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 3
- User story 4
