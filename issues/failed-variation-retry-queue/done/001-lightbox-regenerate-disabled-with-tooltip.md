# Lightbox Regenerate disabled-with-tooltip during in-flight generation

## Parent PRD

`prds/2026-04-30-failed-variation-retry-queue-prd.md`

## What to build

Replace the `ImageLightbox`'s silent no-op behavior on its Regenerate
action with an honest disabled-with-tooltip state while staged
generation is in flight. The thumbnail-level dropdown is already
correctly hidden during generation (see `RoomGroup.tsx`'s
`!isGenerating` guard) and is out of scope; only the lightbox path
changes.

End-to-end behavior: while the project's global generation stream is
running OR a single-variation regen is in flight, the Regenerate
menu/button inside the lightbox renders disabled with a tooltip
explaining "Generating other variations… regenerate available when
complete". When both are idle, the existing behavior returns
unchanged.

This slice is independent of the retry-queue work and can ship on
its own. See PRD sections "Page → ImageLightbox wiring" and
"Component prop flow" (the `ImageLightbox` paragraph) for the
contract details. User stories 11 and 12 motivate the change.

## Acceptance criteria

- [ ] `ImageLightbox` accepts a new optional `isBlocked?: boolean` prop.
- [ ] When `isBlocked` is true, the lightbox's Regenerate
      menu/button is rendered visibly disabled (not hidden) with a
      tooltip explaining why.
- [ ] When `isBlocked` is false, the lightbox's Regenerate behavior
      is unchanged from today.
- [ ] The project detail page computes
      `isBlocked = isGenerating || regeneratingVariationId !== null`
      and passes it to `ImageLightbox`. The existing `onRegenerate`
      prop continues to be passed; the lightbox simply does not
      invoke it while disabled.
- [ ] Per-PRD note: the thumbnail-level dropdown
      (`RoomGroup.tsx`'s `!isGenerating` guard) is left alone — this
      slice does NOT change the thumbnail dropdown.
- [ ] A Playwright E2E spec covers the disabled-with-tooltip
      behavior: open the lightbox during in-flight generation,
      assert the Regenerate control is visibly disabled, assert the
      tooltip text appears on hover/focus, and assert that clicking
      it does not fire a regen request.
- [ ] The new spec follows the established pattern from
      `frontend/tests/e2e/lightbox-regen-state-sync.spec.ts` and the
      SSE-mocking pattern from
      `frontend/tests/e2e/regen-failure-preserves-prior-image.spec.ts`.
- [ ] No backend changes. No changes to
      `streamVariationRegeneration` or any API service contract.
- [ ] Local checks pass before commit:
      `cd frontend && npx playwright test` (full E2E suite),
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 11
- User story 12
