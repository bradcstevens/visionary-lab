# Per-Image Object Quantities — Scenario

Reference scenario for issue 003 of `prds/2026-04-29-per-image-object-quantities-design-prd.md`. The scenario covers per-image quantity / placement / skip overrides on a 3-image project where each image legitimately wants different quantities of the same palette objects.

## Why this scenario

The most concrete real-world example is the backyard scenario in `../backyard-landscaping/`:

- **East fence wide-angle** → wants 5 × Vanderwolf's Pyramid Limber Pine along the back row.
- **Pergola close-up** → wants 0 pines (skip — they would overlap with the pergola structure) and 4 × dwarf grasses.
- **Firepit angle** → wants 2 pines as accent, 8 × ornamental grasses around the firepit ring.

A single uniform palette would either over-plant the pergola shot or under-plant the wide-angle shot. Per-image overrides let one design brief drive all three.

## Test coverage that uses this scenario

### Backend

- `tests/test_brief_resolver.py::TestResolveObjectsForImageWithOverrides` — covers every rule of the resolver: quantity override, placement-None inherits, placement-non-None replaces, `enabled=False` excludes, `quantity=0` excludes, unknown-object-id silent drop, palette-order preservation, duplicate last-write-wins, empty-palette + overrides → empty, etc.
- `tests/test_brief_generator.py::TestBriefToPromptsPerImageObjectSummary` — proves `brief_to_prompts` calls `resolve_objects_for_image` per image, so the prompts the LLM sees diverge across rooms when overrides are set.

### Frontend

- `frontend/tests/e2e/per-image-quantities.spec.ts` — drives the wizard happy path through to the brief editor (step 4), exercises tab rendering, override indicator behavior, "Use Default", "Skip", placement inheritance, palette-deletion pruning, and per-image notes binding.

## Image fixtures

This scenario reuses the photos in `../backyard-landscaping/` rather than duplicating them. The Playwright spec uses synthetic mocked images (the existing `frontend/tests/e2e/fixtures/test-room-{1,2}.png` PNGs) because Playwright drives the wizard through mocked staging API responses; no real upload of the backyard PNGs is required for the spec to be deterministic.

For manual / interactive verification with real images, point the wizard at any three images from `../backyard-landscaping/` (e.g. `backyard-from-east-fence-straight-on-to-west-fence.png`, `pergola-from-middle-patio-straight-on.png`, `firepit-from-patio-right-of-staircase.png`).

## Acceptance criteria check-list

The scenario satisfies the issue's acceptance criteria:

- [x] `ImageObjectOverride` model with required `quantity` (≥ 0), `placement` (`Optional[str]` with mode='before' validator that strips/normalises empty → None), `enabled` defaulting to True.
- [x] `DesignBrief.per_image_objects: Dict[str, List[ImageObjectOverride]]`.
- [x] Resolver `resolve_objects_for_image(brief, room_id)` applies overrides on top of the palette with last-write-wins, palette-order preservation, and the skip semantics above.
- [x] `brief_to_prompts` resolves objects per image so each room can have its own `object_summary`.
- [x] Frontend `PerImageObjectTable` with canonical state model, default-equivalent pruning, and explicit Skip / Use Default actions.
- [x] Frontend `BriefEditorTabs` wrapper with Default Palette + per-image tabs.
- [x] Palette deletion in `DesignBriefEditor` prunes orphan overrides AND drops empty-array room keys.
- [x] Per-image notes textarea: clearing → empty after `trim()` → key removed from the dict.
- [x] Empty-palette → empty-state message on per-image tabs.
