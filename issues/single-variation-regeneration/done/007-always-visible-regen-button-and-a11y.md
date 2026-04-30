# Always-visible regen button and a11y

## Parent PRD

`prds/2026-04-29-single-variation-regeneration-prd.md`

## What to build

Make the per-variation regenerate affordance discoverable on **all input modalities** (mouse, touch, keyboard, screen reader) by replacing the current hover-gated icon with an always-visible quiet icon button on completed-state thumbnails.

- **Always-visible quiet icon** in the bottom-right corner of completed-state variation thumbnails (the variation badge stays in the top-right). Visible on desktop, mobile, and keyboard-focused states.
- **Keyboard reachability:** the button has explicit focus styles and is reachable via tab navigation.
- **Screen reader support:** the dropdown trigger has an explicit `aria-label="Regenerate variation N"`. The thumbnail container gets `aria-busy={isRegenerating}` so screen readers announce state changes when a regen is in flight.
- **Existing behavior preserved:** the dropdown menu content (Retry Same Prompt / Try Something New) keeps its current wiring; only the trigger affordance changes. The lightbox toolbar's regen dropdown is left as-is — its `aria-label` is already sufficient.

See PRD sections **Implementation Decisions → Frontend — `VariationThumbnail`**, **Implementation Decisions → Frontend — `ImageLightbox`**, and **User Stories → 5, 7, 8, 9** for full context.

## Acceptance criteria

- [ ] Frontend: completed-state thumbnails show the regen icon button without requiring hover
- [ ] Frontend: the regen icon is visually quiet (low-emphasis style) so it doesn't compete with the image content
- [ ] Frontend: the icon button is keyboard-reachable via tab and triggers the dropdown via Enter/Space
- [ ] Frontend: the dropdown trigger has `aria-label="Regenerate variation N"` (with N substituted)
- [ ] Frontend: the thumbnail container has `aria-busy={isRegenerating}`
- [ ] Frontend: the dropdown opens on touch (tap) without requiring hover
- [ ] Frontend: the icon button has visible focus styles when keyboard-focused
- [ ] Frontend: lightbox regen dropdown remains unchanged
- [ ] Playwright test: a thumbnail's regen button is reachable via tab navigation and triggers the dropdown via keyboard
- [ ] Playwright test: on a touch viewport (e.g., mobile preset), the regen button is visible without hover and tapping opens the dropdown

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 5 (discoverable from thumbnail)
- User story 7 (touch-reachable)
- User story 8 (keyboard-reachable)
- User story 9 (screen reader announceable)
