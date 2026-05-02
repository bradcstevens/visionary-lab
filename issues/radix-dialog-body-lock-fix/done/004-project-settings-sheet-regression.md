## Parent PRD

`prds/2026-05-01-radix-dialog-body-lock-fix-prd.md`

## What to build

Add a close-then-interactive regression assertion to
`project-settings-sheet.spec.ts` and remove the existing
`force: true` Playwright workarounds — and the comments that
explain them — that were put in place to coexist with the broken
Radix lock.

`ProjectSettingsSheet` already follows the always-mounted +
controlled-`open` shape, so this slice has **no component
implementation change** — it is purely an e2e regression. The
assertion verifies three observable behaviors after the sheet
closes via each of Cancel, ✕, Esc, and click-outside:

1. `<body>` does not have inline `pointer-events: none`
2. `<body>` does not have a `data-scroll-locked` attribute
3. A non-`force` Playwright click on a known page element below
   the closed sheet succeeds

The third assertion is the ground-truth user-facing check — it
catches any *other* mechanism that could leave the page
non-interactive (stuck focus-trap, orphan portal, scheduler bug)
beyond the two known stuck-attribute signatures. The assertion
runs in two flavors: once with no inner dropdown ever opened, and
once after the user has opened and dismissed each of the
Model / Quality / Size dropdowns (the path the PRD specifically
calls out as a regression vector).

The existing `force: true` workarounds and their lock-related
comments are deleted in the same change — the new assertion takes
their place as the regression guard, and leaving the workarounds
in would let the spec coexist with broken behavior again.

See PRD sections "Test contract surface" and "Prior art" for the
exact assertion shape and the comment/workaround removal context.

## Acceptance criteria

- [ ] No implementation changes to `ProjectSettingsSheet` or its
      consumers
- [ ] `project-settings-sheet.spec.ts` gains a close-then-interactive
      regression assertion that runs after each of: Cancel button,
      ✕ button, Escape key, click-outside; verifying inline
      `pointer-events: none` is absent on `<body>`, `data-scroll-locked`
      is absent on `<body>`, and a non-`force` Playwright click on
      a normal page element below the sheet succeeds
- [ ] The regression assertion is exercised in two paths: (a) sheet
      opened and immediately closed, (b) sheet opened, each of
      Model / Quality / Size dropdowns opened and dismissed, then
      sheet closed
- [ ] Existing `force: true` Playwright clicks in
      `project-settings-sheet.spec.ts` are removed
- [ ] Comments in `project-settings-sheet.spec.ts` that explain
      the Radix `pointer-events` lock and justify the
      `force: true` workarounds are removed
- [ ] Failure messages on the new assertion are at least as
      informative as the removed comments — a future contributor
      reading a CI failure understands the lock-leak family being
      caught
- [ ] Existing happy-path assertions in
      `project-settings-sheet.spec.ts` continue to pass

## Blocked by

- Blocked by `002-body-lock-guard.md` (the close-then-interactive
  assertion needs the layout-level guard in place to pass reliably,
  since `ProjectSettingsSheet` itself receives no implementation
  change in this slice)

## User stories addressed

Reference by number from the parent PRD:

- User story 4
- User story 5
- User story 9 (Project Settings close paths)
- User story 13
- User story 14 (Project Settings regression assertion)
- User story 15 (non-force click in Project Settings assertion)
- User story 24 (Project Settings overlay's PR carries its own regression spec)
