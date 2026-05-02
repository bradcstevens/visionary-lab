## Parent PRD

`prds/2026-05-01-radix-dialog-body-lock-fix-prd.md`

## What to build

Add a close-then-interactive regression assertion to
`image-lightbox.spec.ts`. The lightbox itself receives no
implementation change in this slice (the accessibility-only fix
ships in `001-lightbox-accessibility-fix.md`); this is purely an
e2e regression that asserts the lightbox-close path no longer
leaves the page non-interactive.

The assertion verifies three observable behaviors after the
lightbox closes via each of ✕ button and Escape key:

1. `<body>` does not have inline `pointer-events: none`
2. `<body>` does not have a `data-scroll-locked` attribute
3. A non-`force` Playwright click on a known page element below
   the closed lightbox succeeds

The third assertion is the ground-truth user-facing check — it
catches any *other* mechanism that could leave the page
non-interactive beyond the two known stuck-attribute signatures.

The assertion runs after the user-natural review flow the PRD
calls out: open the lightbox on one variation, navigate through
completed variations with the arrow keys, then close. This
exercises the full review-to-action path and confirms that
moving from review back to action does not require a refresh.

See PRD sections "Test contract surface" and "Test philosophy"
for the assertion shape.

## Acceptance criteria

- [ ] No implementation changes to `ImageLightbox` or its consumers
      in this slice
- [ ] `image-lightbox.spec.ts` gains a close-then-interactive
      regression assertion that runs after each of: ✕ button,
      Escape key; verifying inline `pointer-events: none` is absent
      on `<body>`, `data-scroll-locked` is absent on `<body>`, and
      a non-`force` Playwright click on a normal page element
      below the lightbox succeeds
- [ ] The regression assertion is exercised after navigating
      through at least two completed variations with the arrow
      keys before close, matching the user-natural review flow
      called out in the PRD
- [ ] Failure messages on the new assertion are informative — a
      future contributor reading a CI failure understands the
      lock-leak family being caught
- [ ] Existing happy-path assertions in `image-lightbox.spec.ts`
      continue to pass

## Blocked by

- Blocked by `002-body-lock-guard.md` (the close-then-interactive
  assertion needs the layout-level guard in place to pass
  reliably, since `ImageLightbox` itself receives no
  implementation change in this slice)

## User stories addressed

Reference by number from the parent PRD:

- User story 6
- User story 9 (lightbox close paths)
- User story 14 (lightbox regression assertion)
- User story 15 (non-force click in lightbox assertion)
- User story 24 (lightbox overlay's PR carries its own regression spec)
