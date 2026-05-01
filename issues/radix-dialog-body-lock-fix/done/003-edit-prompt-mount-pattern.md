## Parent PRD

`prds/2026-05-01-radix-dialog-body-lock-fix-prd.md`

## What to build

Standardize the Edit Prompt dialog onto the same shape the project
detail page's other two overlay-using surfaces already share:
**always-mounted overlay component + controlled `open` prop + state
reset triggered by the `open` prop's rising edge**. The dialog is
currently the one violating that shape — the parent project detail
page conditionally mounts it via `{editPromptTarget && (…)}` — and
that violation is the structural cause of the unmount-while-open
landmine for this overlay specifically.

The dialog component is updated so its draft state and submitting
flag are reset inside the dialog itself via an effect keyed on `open`
going from false to true (the same snapshot-on-open pattern
`ProjectSettingsSheet` already uses). The parent project detail page
drops its `{editPromptTarget && (…)}` wrap and instead mounts the
dialog unconditionally, feeding its `open` prop from the truthiness
of the target. The dialog's existing doc-comment is rewritten — it
currently instructs future contributors to "conditionally mount this
dialog so each open is a fresh mount," which is exactly the pattern
that caused the bug; the new comment records the corrected pattern
and explicitly warns future contributors away from conditional
mount.

The slice ships with an extension to `edit-prompt-dialog.spec.ts`
that adds the close-then-interactive regression assertion (no
`pointer-events:none` on `<body>`, no `data-scroll-locked` on
`<body>`, and a non-`force` Playwright click on a normal page
element below succeeds) for each close path — Cancel button, ✕
button, Escape key, outside click — plus the cross-variation /
same-variation prefill cases that prove draft state does not bleed
between opens. Existing happy-path assertions in that spec are not
expected to require rewriting; if any depend on the old
remount-on-open lifecycle they are updated to the new model.

See PRD section "Mount-pattern standardization" for the rationale
and "Test contract surface" for the assertion shape.

## Acceptance criteria

- [ ] `EditPromptDialog` is rendered with a controlled `open` prop and
      is no longer conditionally mounted by its parent
- [ ] The project detail page (`frontend/app/projects/[id]/page.tsx`
      or its equivalent) drops the `{editPromptTarget && (…)}` wrap
      around the dialog and feeds `open` from the truthiness of the
      target instead
- [ ] Draft prompt text and submitting flag are reset inside
      `EditPromptDialog` via an effect keyed on `open` transitioning
      from `false` to `true`
- [ ] The dialog's source-file doc-comment no longer instructs
      contributors to conditionally mount the dialog; the rewritten
      comment explains the always-mounted + open-edge-reset pattern
      and explicitly warns against the prior conditional-mount
      approach
- [ ] Re-opening the dialog on a *different* variation shows that
      variation's prior adapted prompt (not the previous variation's
      and not an abandoned draft)
- [ ] Re-opening the dialog on the *same* variation after closing
      without saving shows the prior adapted prompt fresh (not the
      abandoned draft)
- [ ] `edit-prompt-dialog.spec.ts` gains a close-then-interactive
      regression assertion that runs after each of: Cancel button,
      ✕ button, Escape key, click-outside; the assertion verifies
      `<body>` has no inline `pointer-events: none`, no
      `data-scroll-locked` attribute, and a non-`force` Playwright
      click on a normal page element below succeeds
- [ ] `edit-prompt-dialog.spec.ts` gains assertions for the
      cross-variation prefill case (story 2) and the same-variation
      prefill case (story 3)
- [ ] Existing happy-path assertions in `edit-prompt-dialog.spec.ts`
      continue to pass; any that relied on remount-on-open lifecycle
      are updated rather than removed

## Blocked by

- Blocked by `002-body-lock-guard.md` (the close-then-interactive
  e2e assertion needs the layout-level guard in place to backstop
  any leak this overlay's specific fix does not cover)

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 3
- User story 9 (Edit Prompt close paths)
- User story 10
- User story 11
- User story 12
- User story 14 (Edit Prompt regression assertion)
- User story 15 (non-force click in Edit Prompt assertion)
- User story 24 (Edit Prompt overlay's PR carries its own regression spec)
