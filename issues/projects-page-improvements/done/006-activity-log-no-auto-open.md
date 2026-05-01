# Activity log opens on demand only

## Parent PRD

`prds/2026-04-30-projects-page-improvements-prd.md`

## What to build

Stop the right-side activity log panel from auto-opening when the
first log entry lands. The panel only opens when the user clicks
the toggle. The existing notification dot on the toggle (red for
errors, blue for any other activity) provides an unobtrusive cue
that there is something to see.

End-to-end behavior:

- Frontend: the auto-open block in `activity-log-context.tsx` (the
  three-line branch that opens the panel on first log entry) is
  deleted. The existing notification dot behavior on the toggle is
  preserved unchanged. No other changes to the panel itself.
- Tests: a Playwright scenario triggers a generation and asserts
  the panel stays closed; clicking the toggle opens it.

See PRD sections **"Solution → 6. Activity log opens on demand
only"**, **"Implementation Decisions → Frontend modules"** (auto-
open deletion bullet), and **"Testing Decisions → Frontend
tests"** (no-auto-open scenario).

## Acceptance criteria

- [ ] The auto-open trigger in `activity-log-context.tsx` (the
      branch that opens the panel on first log entry) is removed.
- [ ] The panel only opens in response to a user click on the
      toggle.
- [ ] The existing notification-dot behavior on the toggle is
      preserved unchanged (red for errors, blue for any other
      activity).
- [ ] A new Playwright spec covers: trigger a generation; assert
      the panel stays closed; click the toggle; assert the panel
      opens (and, once slice 008 lands, that the In Flight section
      is visible — but this slice does not depend on 008 being
      complete, the assertion can be limited to "panel opens").
- [ ] No backend changes.
- [ ] Local checks pass before commit:
      `cd frontend && npx playwright test`,
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 23
- User story 24
