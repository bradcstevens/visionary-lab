## Parent PRD

`prds/2026-05-01-radix-dialog-body-lock-fix-prd.md`

## What to build

Add a visually-hidden `DialogDescription` to the image lightbox so the Radix
dialog primitive has the description it requires for assistive technology.
The lightbox currently ships only with a visually-hidden `DialogTitle`,
which causes Radix to emit a "Missing Description for DialogContent"
warning on every mount. The component change is small — add the
description element immediately after the existing title — but its
visible effect is twofold: screen readers announce a coherent context
for the dialog, and the browser console no longer floods with the same
warning every time the lightbox opens.

This slice does not modify the lightbox's close behavior or its
underlying Radix usage; it is purely an accessibility / console-hygiene
improvement that can ship independently of the rest of the body-lock
work.

See PRD section "Lightbox accessibility" for the rationale and exact
shape of the addition.

## Acceptance criteria

- [ ] `ImageLightbox` renders a visually-hidden `DialogDescription`
      immediately after its existing visually-hidden `DialogTitle`
- [ ] Opening the lightbox no longer produces the "Missing Description
      for DialogContent" warning in the browser console (verifiable by
      manual inspection in dev mode)
- [ ] The lightbox's existing visual layout, animation, and keyboard
      navigation are unchanged
- [ ] Existing happy-path assertions in `image-lightbox.spec.ts`
      continue to pass without modification

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 7
- User story 8
