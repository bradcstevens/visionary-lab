# Per-Image Object Quantities & Generic Object Model

**Status:** Design
**Date:** 2026-04-29

## Problem

The current staging feature uses a project-wide `plant_palette` on the `DesignBrief` to define what to add to generated images. This has two limitations:

1. **Plant-specific schema.** Fields like `species` and `botanical_name` only fit plants. Adding rocks, furniture, lighting, or hardscape feels awkward and forces users to misuse plant fields.
2. **Quantities are project-wide.** If a user specifies "8 of plant X" and the project has three input images, the generation pipeline tries to fit 8 of that plant into every image. Some images can comfortably hold more objects than others, so a single project-wide quantity produces visually unbalanced results.

Users need to specify object quantities **per individual original image** that was uploaded as part of the project's collection, and the underlying model needs to support arbitrary object types — not just plants.

## Goals

1. Replace the plant-specific data model with a generic object model that supports plants, trees, rocks, furniture, lighting, hardscape, decor, and future categories.
2. Allow per-image quantity (and placement) overrides on top of project-wide defaults defined in the object palette.
3. Preserve the existing AI-driven design chat workflow so the palette is still auto-suggested from conversation.
4. Provide a clear UI where users adjust per-image quantities without losing sight of project defaults.
5. Migrate existing projects without data loss.

## Non-Goals

- Adding object-type-specific generation logic (e.g., physics-aware furniture placement). The pipeline still treats objects as text in prompts.
- Changing the image generation model or pipeline architecture.
- Building a reusable "object library" across projects. Object palettes remain scoped to a single project.

## Approach

**Approach selected: Per-image overrides on `DesignBrief` (Approach 1).**

The object palette on `DesignBrief` defines the catalog of available objects with project-wide defaults. A new field, `per_image_objects`, holds a sparse map of per-room overrides. At generation time the pipeline merges defaults with per-image overrides to produce an effective object list for each image.

This was chosen over the alternative — moving full object lists onto each `Room` — because it keeps a single source of truth for object definitions, avoids duplicating data across rooms, and aligns with the existing `per_image_notes` pattern already on `DesignBrief`. Stable per-object UUIDs make the override-by-id pattern robust against palette reordering or deletion.

## Data Model Changes

### New: `ObjectCategory` enum

```python
class ObjectCategory(str, Enum):
    PLANT = "plant"
    TREE = "tree"
    ROCK = "rock"
    FURNITURE = "furniture"
    LIGHTING = "lighting"
    HARDSCAPE = "hardscape"   # patio, walkway, retaining wall
    DECOR = "decor"
    OTHER = "other"
```

### New: `ObjectEntry` (replaces `PlantEntry`)

```python
class ObjectEntry(BaseModel):
    id: str                                # UUID, stable identifier
    name: str                              # "Vanderwolf's Pyramid Pine", "Adirondack Chair"
    description: Optional[str] = None      # Botanical name OR free-form description
    category: ObjectCategory
    default_quantity: int = 1              # Project-level default
    size: str = ""                         # "8-10 ft tall", "36in wide"
    placement: str = ""                    # Default placement guidance
    visual_notes: Optional[str] = None     # Visual characteristics for image generation
```

### New: `ImageObjectOverride`

```python
class ImageObjectOverride(BaseModel):
    object_id: str                         # References ObjectEntry.id
    quantity: int                          # Per-image quantity
    placement: Optional[str] = None        # None = inherit palette placement
    enabled: bool = True                   # False = skip this object for this image
```

### Updated: `DesignBrief`

```python
class DesignBrief(BaseModel):
    global_instructions: str
    object_palette: List[ObjectEntry] = Field(default_factory=list)         # was plant_palette
    placement_guide: PlacementGuide = Field(default_factory=PlacementGuide)
    per_image_objects: Dict[str, List[ImageObjectOverride]] = Field(default_factory=dict)
    per_image_notes: Dict[str, str] = Field(default_factory=dict)           # unchanged
    preserve_elements: List[str] = Field(default_factory=list)
    settings: StagingSettings = Field(default_factory=StagingSettings)
```

`per_image_objects` is keyed by `room_id`. A missing key means "no overrides — use palette defaults for every object." A key present with an empty list also means no overrides.

## Resolution Logic

For each image (room) at generation time, the pipeline computes an effective object list:

1. Start with the project's `object_palette`.
2. Look up `per_image_objects[room_id]`. If absent, return the palette as-is using each object's `default_quantity` and `placement`.
3. For each override in the list:
   - Match the override to an `ObjectEntry` by `object_id`.
   - If `enabled=False`, exclude that object from this image entirely.
   - Otherwise, replace the object's quantity with `override.quantity` and use `override.placement` if set (else fall back to the palette default).
4. Emit only objects with `quantity > 0` (zero-quantity entries are suppressed silently).

Pseudocode lives in `BriefGeneratorService._resolve_objects_for_image(brief, room_id)` and returns a list of `ResolvedObject` dataclass instances used purely for prompt construction. `ResolvedObject` is not persisted — it is a transient projection.

## Backend Changes

### `backend/models/design_brief.py`

- Replace `PlantEntry` with `ObjectEntry`.
- Add `ImageObjectOverride`, `ObjectCategory`.
- Update `DesignBrief.plant_palette` → `object_palette`.
- Add `DesignBrief.per_image_objects`.

### `backend/core/brief_generator.py`

- Update `BRIEF_GENERATION_PROMPT` so the LLM extracts a generic `object_palette` (not `plant_palette`). The prompt instructs the LLM to choose a category for every extracted object and to omit per-image quantities (the palette only carries defaults — overrides are user-driven afterward).
- Add `_assign_object_ids(palette)` to assign fresh UUIDs to AI-generated objects.
- Replace `plant_summary` construction with `_resolve_objects_for_image(brief, room_id)` and per-image `object_summary` strings inside `brief_to_prompts`.
- Update `BRIEF_TO_PROMPTS_TEMPLATE`: rename "Plants:" to "Objects:" and update the wording to be category-agnostic.

### `backend/core/staging_pipeline.py`

- Update `INDOOR_PROMPT_TEMPLATE` and `OUTDOOR_PROMPT_TEMPLATE` to refer to "objects" rather than "plants/items" specifically. The OUTDOOR template still emphasizes landscape terminology (back row, border, flanking) since that placement language is universal.

### Backwards-compat migration

- Add `_migrate_legacy_plant_palette(raw_brief: Dict[str, Any]) -> Dict[str, Any]` in `backend/models/design_brief.py` (or a new `backend/core/brief_migration.py`).
- Migration runs whenever a brief is loaded from Cosmos DB:
  - If `plant_palette` exists and `object_palette` does not, transform each entry: assign a UUID, map `species`→`name`, `botanical_name`→`description`, set `category=PLANT`, copy `quantity`→`default_quantity`, copy `size`/`placement`/`visual_notes`.
  - Drop the old `plant_palette` key after migration.
  - Initialize `per_image_objects` as empty if missing.
- On the next project save, the migrated brief is persisted.

### Storage

No Cosmos DB schema migration required. The `StagingProject.design_brief` field is `Dict[str, Any]`, so migrated documents are written through the normal save flow.

## Frontend Changes

### Type definitions (`frontend/services/stagingApi.ts`)

- Replace `PlantEntry` interface with `ObjectEntry`, add `ImageObjectOverride`, add `ObjectCategory` string-literal type.
- Update `DesignBrief` interface: `plant_palette` → `object_palette`, add `per_image_objects: Record<string, ImageObjectOverride[]>`.

### Component renames and additions

- `PlantPaletteTable.tsx` → `ObjectPaletteTable.tsx`. Adds a Category dropdown and uses generic field names (Name, Description). The Category column appears between Name and Default Qty.
- New `PerImageObjectTable.tsx`: renders the resolved object list for a single room with quantity/placement editors, "Use Default" buttons, and a Skip toggle.
- New `BriefEditorTabs.tsx`: tab strip rendering one "Default Palette" tab + one tab per uploaded room. Each tab thumbnail shows the room label and a small image preview.

### `DesignBriefEditor.tsx`

- Wraps `BriefEditorTabs.tsx`. The "Default Palette" tab embeds `ObjectPaletteTable` plus the existing `PlacementGuide`, `preserve_elements`, and global instructions controls. Each image tab embeds `PerImageObjectTable` plus the existing per-image notes textarea.
- State management is unchanged in shape: the parent owns `designBrief` and passes setters down. Edits to per-image tabs update `designBrief.per_image_objects[roomId]`.

### Override behavior in the UI

- Each row in `PerImageObjectTable` starts pre-filled with the palette default (`default_quantity`, `placement`).
- Editing the quantity field creates an override entry for that object on that room (if one doesn't exist) and writes the new quantity.
- Clicking "Use Default" removes the override entry for that row.
- Toggling "Skip" sets `enabled: false`. Skipped rows render dimmed.
- A small visual indicator (purple dot + "(overridden)" tag) appears on rows that have any non-default value.

### Palette deletion cleanup

When a user deletes an `ObjectEntry` from the palette, the parent setter prunes any entries in `per_image_objects[*]` whose `object_id` matches the deleted object's id, keeping the override map consistent.

### Project detail page

The same `DesignBriefEditor` is used in the project detail page for editing briefs after generation. No additional work is required there beyond importing the new component.

## Validation & Edge Cases

- **Empty palette.** If the palette is empty, every per-image tab still renders but shows an empty state ("Add objects to the palette to set per-image quantities").
- **Zero-quantity overrides.** Allowed and treated identically to `enabled: false` (object is skipped for that image). UI permits both forms; the resolver normalizes by skipping any object with `quantity == 0` or `enabled == false`.
- **Negative quantities.** Validators on `ImageObjectOverride.quantity` enforce `>= 0`. Frontend Inputs use `min={0}`.
- **Unknown `object_id` in overrides.** The resolver silently ignores override entries whose `object_id` does not match any current palette entry. Cleanup of stale overrides happens on object deletion in the UI; resolver tolerance covers any drift.
- **Old projects with legacy `plant_palette`.** Loaded through the migration helper. AI-generated briefs from older sessions get UUIDs on first read.
- **Object reordering in the palette.** Override resolution uses `object_id`, not array index, so reordering is safe.

## Testing

### Backend (pytest)

- `tests/unit/test_design_brief_migration.py` — Migration of legacy `plant_palette` to `object_palette` with UUID assignment.
- `tests/unit/test_brief_generator_resolution.py` — `_resolve_objects_for_image` correctness:
  - No overrides → defaults used.
  - Quantity override.
  - Placement override (None inherits, set replaces).
  - `enabled: false` excludes object.
  - Unknown `object_id` ignored.
  - Zero-quantity entries omitted.
- `tests/unit/test_brief_to_prompts_per_image.py` — `brief_to_prompts` produces a different `object_summary` for two rooms with different overrides.

### Frontend (Playwright E2E)

- New scenario in `tests/projects/`: "Per-image object quantities" — uploads three images, AI proposes a palette, user adjusts quantities differently for each image, generates, and asserts that prompt records on each room's variations reflect the per-image quantities. Reuses the activity-log SSE event capture pattern.
- Component-level: typing in a per-image quantity input creates an override; clicking "Use Default" clears it; deleting a palette object removes its overrides.

### Build/lint

- `cd frontend && npx next lint`
- `cd frontend && npm run build`

## Implementation Order

1. Backend models (`design_brief.py`, migration helper) + unit tests.
2. Backend generator (`brief_generator.py`) + unit tests for resolution and per-image prompt construction.
3. Backend prompt templates and brief-generation prompt update.
4. Frontend type updates (`stagingApi.ts`).
5. `ObjectPaletteTable.tsx` (rename + category column).
6. `PerImageObjectTable.tsx` + `BriefEditorTabs.tsx`.
7. `DesignBriefEditor.tsx` rework to use tabs.
8. Palette deletion cascade cleanup in the wizard state.
9. E2E test for per-image quantities.
10. Backwards-compat verification on an existing legacy project.

## Open Questions

None. All design decisions confirmed during brainstorming.
