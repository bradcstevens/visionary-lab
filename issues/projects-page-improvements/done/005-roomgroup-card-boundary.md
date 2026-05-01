# RoomGroup Card boundary + "Image N of M" label

## Parent PRD

`prds/2026-04-30-projects-page-improvements-prd.md`

## What to build

Wrap each `RoomGroup` in a subtle Card primitive so the per-room
Generate button is visibly enclosed in the same container as the
images it acts on. Add a small "Image N of M" label to the room
header for additional context. No layout restructuring, no
behavioral change — just a visible boundary and a positional label.

End-to-end behavior:

- Frontend: `RoomGroup` is wrapped in
  `<Card className="p-4 space-y-3">{...}</Card>`. The existing
  internal layout (title row with Generate button, status message,
  image grid) is preserved inside. The Card uses subtle border and
  padding only — not heavy chrome — so the page stays scannable.
  A small "Image N of M" label is added next to the room title.
  Existing room title, status badge, and "N/M variations" counter
  remain in the same row inside the Card.
- Tests: a Playwright DOM/screenshot assertion confirms each room
  is wrapped in a single visible container and that the Generate
  button sits within the same boundary as the images.

See PRD sections **"Solution → 5. RoomGroup card boundary"**,
**"Implementation Decisions → Frontend modules"** (RoomGroup Card
bullet), and **"Testing Decisions → What is NOT tested"** (visual
regression note).

## Acceptance criteria

- [ ] `RoomGroup` is wrapped in the existing `Card` primitive with
      subtle border and small padding (per PRD: not heavy chrome).
- [ ] The existing internal layout — title row with Generate
      button, status message, image grid — is preserved unchanged
      inside the Card.
- [ ] The room title, status badge, and existing "N/M variations"
      counter remain in the same row inside the Card.
- [ ] A small "Image N of M" label appears next to the room title,
      reflecting this room's position in the project's room list.
- [ ] No behavioral change — Generate button click handlers, the
      image grid, and any existing per-room state are untouched.
- [ ] A new Playwright spec asserts that each rendered room is
      enclosed in a single visible container element and that the
      Generate button is a descendant of the same container as the
      image grid (no shared parent across rooms).
- [ ] Local checks pass before commit:
      `cd frontend && npx playwright test`,
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 20
- User story 21
- User story 22
