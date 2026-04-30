# 002 — Frontend rename to ObjectPaletteTable with category column

## Parent PRD

`issues/2026-04-29-per-image-object-quantities-prd.md`

## What to build

The frontend half of the generic-object-model migration. Together with issue `001`, this delivers the first user-visible slice: opening a legacy project shows the brief editor with generic field labels and a category column instead of plant-specific ones, and new projects use the generic object model from end to end.

End-to-end behaviour delivered by this slice:

- Frontend types mirror the backend: `PlantEntry` is replaced by `ObjectEntry`, an `ObjectCategory` string-literal type is added, and `DesignBrief.plant_palette` is replaced by `object_palette`. No type referencing `plant_palette` remains in the frontend.
- `PlantPaletteTable.tsx` is renamed to `ObjectPaletteTable.tsx`. Field labels are generic — Name, Description, Category, Default Qty, Size, Placement, Visual Notes — instead of Species / Botanical Name. A new editable Category dropdown column sits between Name and Default Qty, populated from the `ObjectCategory` enum.
- `DesignBriefEditor` is updated to consume `object_palette` and to render `ObjectPaletteTable` (the broader tabbed layout lands in issue `003`).
- The wizard, design chat, generate-brief flow, and PUT-brief flow all round-trip the new shape unchanged. Saving and reloading a project preserves entries.
- `per_image_objects` is **not** wired up in this slice. Per-image overrides land in issue `003`.

See PRD section "Frontend" (the `ObjectPaletteTable` and types subset) and "Backwards-compat migration" (the user-visible result).

## Acceptance criteria

- [ ] `frontend/types` exports `ObjectCategory` (`'plant' | 'tree' | 'rock' | 'furniture' | 'lighting' | 'hardscape' | 'decor' | 'other'`) and `ObjectEntry` matching the backend shape.
- [ ] `PlantEntry` is removed from frontend types; nothing in the frontend imports it.
- [ ] `DesignBrief` frontend type uses `object_palette: ObjectEntry[]`; `plant_palette` is removed.
- [ ] `frontend/components/staging/PlantPaletteTable.tsx` is renamed to `ObjectPaletteTable.tsx`. Imports and consumers are updated.
- [ ] `ObjectPaletteTable` renders generic field labels (Name, Description, Category, Default Qty, Size, Placement, Visual Notes). Species / Botanical Name labels are gone.
- [ ] `ObjectPaletteTable` adds an editable Category dropdown column between Name and Default Qty, populated from `ObjectCategory`. Editing it updates the entry in local state.
- [ ] `DesignBriefEditor` renders `ObjectPaletteTable` and consumes `object_palette`.
- [ ] `cd frontend && npx next lint` passes.
- [ ] `cd frontend && npm run build` passes.
- [ ] `cd frontend && npx playwright test` passes against the existing E2E scenarios (no regressions in the existing wizard and ai-design-session flows). Save reports under `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.
- [ ] Manually verified: opening a project whose persisted `design_brief` contains legacy `plant_palette` shows the brief editor populated correctly with generic labels and a category column.

## Blocked by

- Blocked by `issues/001-backend-generic-object-model-and-migration.md`

## User stories addressed

- User story 1
- User story 2
- User story 13
- User story 14
- User story 18
- User story 19
