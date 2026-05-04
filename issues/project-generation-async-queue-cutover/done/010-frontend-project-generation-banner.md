## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

A new `ProjectGenerationBanner` React component that renders the
project-level progress experience. The banner displays a phase label
and progress percentage, and exposes a Cancel control that calls a
provided handler. It is a pure presentational component bound to the
`inFlightProjectGeneration` slice — issue 011 plugs the slice into the
banner on the project page.

See PRD section "Frontend changes → `ProjectGenerationBanner`
component".

End-to-end behaviour:

- New component (e.g. `frontend/components/staging/
  ProjectGenerationBanner.tsx`).
- Props: `{ progress: number; phase: string; status: string;
  onCancel: () => void; cancelling?: boolean }`.
- Renders a banner with:
  - A meaningful phase label such as "Composing brief", "Generating
    Living Room", "Finalizing" — derived from the `phase` prop. The
    label-derivation rules belong here so the page wiring can stay
    thin.
  - A progress indicator (numeric percentage and / or progress bar)
    driven by the `progress` prop.
  - A Cancel button that invokes `onCancel` and disables itself
    while `cancelling` is true.
- Renders nothing (returns `null`) when `status` is terminal
  (`succeeded`, `failed`, `cancelled`). The page wiring already
  hides the banner when the slice is null, but this is a
  belt-and-suspenders for the in-between change-feed event where
  status flips terminal but the slice has not yet been recomputed.
- A11y: the cancel button is a real `<button>` with an accessible
  name; the banner is announced via `role="status"` (or equivalent)
  so screen readers see the progress change without having to
  inspect the DOM.

This slice ships only the component and its vitest unit tests.

## Acceptance criteria

- [ ] `ProjectGenerationBanner` component exists with the documented
      props.
- [ ] Banner renders a phase label, a progress percentage / bar,
      and a Cancel button.
- [ ] Cancel button invokes `onCancel`; when `cancelling` is true,
      the button is disabled.
- [ ] Banner returns `null` when `status` is terminal.
- [ ] Cancel button is a real `<button>` with an accessible name;
      banner has an appropriate live-region role.
- [ ] New vitest unit tests in
      `frontend/components/staging/__tests__/
      ProjectGenerationBanner.test.tsx` cover: renders progress and
      phase; clicking Cancel calls the handler; the banner
      disappears (returns null) on terminal status; cancel button
      disables when `cancelling=true`.
- [ ] `cd frontend && npx vitest run` is green; `cd frontend && npx
      next lint` is green; `cd frontend && npm run build` is green.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 2
- User story 3
- User story 14
- User story 18
