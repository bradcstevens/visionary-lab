# 003 — Per-image quantity and placement overrides

## Parent PRD

`issues/2026-04-29-per-image-object-quantities-prd.md`

## What to build

The core feature of the PRD: a designer can tune object quantities and placement per uploaded image without touching the project palette, and each generated image renders against its own effective object list.

End-to-end behaviour delivered by this slice:

- The brief editor in step 4 of the new-project wizard shows a tab strip: a "Default Palette" tab (existing palette editor from issue `002`) plus one tab per uploaded image, each labelled with the room name and a small image preview.
- Each per-image tab renders a table pre-filled with the project palette defaults. Editing a quantity, editing placement, clicking "Skip", or clicking "Use Default" writes a sparse override entry into `per_image_objects[room_id]` (or removes it). An override indicator marks any row that has any entry in `per_image_objects[room_id]`, regardless of whether the values currently coincide with palette defaults.
- Clearing the placement input back to empty reverts to inherit; an explicit empty-string override is never persisted (the model boundary coerces it to `None`).
- Each per-image tab also exposes a textarea bound to `brief.per_image_notes[room_id]`. Clearing the textarea deletes the key.
- Deleting an `ObjectEntry` from the palette prunes any matching entries from every `per_image_objects[*]` so the override map stays consistent.
- At render time the pipeline produces a different `object_summary` per image, reflecting the merged result of palette defaults plus that image's overrides. Two rooms with different overrides yield different prompts.

The chat-driven LLM-pre-fill of overrides and the regenerate-preserves-overrides flow are out of scope for this slice — they land in issue `004`. This slice only covers the user editing overrides directly in the editor.

See PRD sections "Data model" (`ImageObjectOverride`, `per_image_objects`), "Resolution logic" (full rules), "Prompt-to-image rendering" (per-room `object_summary`), and "Frontend" (`PerImageObjectTable`, `BriefEditorTabs`, palette-delete pruning).

## Acceptance criteria

- [ ] `ImageObjectOverride` model exists with fields `object_id`, `quantity` (>= 0), `placement` (optional, `None` means inherit, anything else replaces), `enabled` (default true; false means skip). A field-level validator on `placement` normalises empty/whitespace strings to `None`. A validator rejects negative `quantity`.
- [ ] `DesignBrief.per_image_objects: Dict[str, List[ImageObjectOverride]]` exists with default `{}`. The legacy migration helper initialises it to `{}` when missing.
- [ ] `brief_resolver.resolve_objects_for_image(brief, room_id)` implements the full ruleset: starts from palette; applies overrides keyed by `object_id`; silently skips overrides whose `object_id` is not in the current palette; treats `enabled=False` and `quantity=0` as equivalent skip signals; falls back to palette placement when override `placement is None`; returns `ResolvedObject` projections.
- [ ] `BriefGeneratorService.brief_to_prompts` constructs a separate `object_summary` per image using the resolver. Two rooms with different overrides produce different `object_summary` strings.
- [ ] Frontend types add `ImageObjectOverride` (with `placement: string | null`, never `undefined`) and `DesignBrief.per_image_objects`.
- [ ] `frontend/components/staging/PerImageObjectTable.tsx` exists. Rows are pre-filled with palette defaults. Editing quantity creates an override; editing placement creates a placement override; clearing the placement input back to empty removes the placement override (and removes the override entry entirely if no other fields differ from default); a "Use Default" action removes the override entry; a "Skip" action sets `enabled=false`. The override indicator shows whenever the row has any entry in `per_image_objects[room_id]`.
- [ ] `frontend/components/staging/BriefEditorTabs.tsx` exists and renders one Default Palette tab plus one tab per uploaded image. Each image tab thumbnail shows the room label and a small image preview.
- [ ] `DesignBriefEditor` is reworked to wrap `BriefEditorTabs`. The Default Palette tab embeds `ObjectPaletteTable` plus the existing placement guide / preserve / global-instructions controls. Each image tab embeds `PerImageObjectTable` plus a textarea bound to `brief.per_image_notes[room_id]` (clearing the textarea deletes the key). The previously-reserved `imageLabels` prop is now consumed.
- [ ] When an `ObjectEntry` is deleted from the palette, the parent setter prunes any matching entries from `per_image_objects[*]`.
- [ ] An empty palette renders an empty-state message on the per-image tab.
- [ ] Persistence model is unchanged: edits live in local React state during step 4 and persist via the existing PUT-brief endpoint when the user clicks "Save & Continue". No autosave, no per-tab save.
- [ ] Unit tests for `brief_resolver`: per-image resolution rules (no overrides; quantity override; placement override with `None`-inherits and non-`None`-replaces; `enabled=False` excludes; unknown `object_id` ignored; zero-quantity entries omitted). Tests for `brief_to_prompts` asserting two rooms with different overrides yield different `object_summary` strings.
- [ ] New Playwright E2E scenario folder under `tests/projects/per-image-quantities/`. Uploads three images, lets the AI propose a palette, the test adjusts quantities differently per image, generates, and asserts that the prompt records on each room's variations reflect the per-image quantities. Reuses the existing activity-log SSE event-capture pattern.
- [ ] In-scenario component-level frontend assertions: typing in a per-image quantity input creates an override entry; clicking "Use Default" clears it; deleting a palette object removes its overrides from every per-image room; clearing the placement input reverts to inherit (no `""` override is persisted).
- [ ] `uv run pytest tests/ --ignore=tests/integration -v`, `cd frontend && npx next lint`, `cd frontend && npm run build`, and `cd frontend && npx playwright test` all pass. Save Playwright reports under `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.

## Blocked by

- Blocked by `issues/002-frontend-object-palette-table.md`

## User stories addressed

- User story 3
- User story 4
- User story 5
- User story 6
- User story 7
- User story 8
- User story 9
- User story 10
- User story 11
- User story 20
- User story 21
- User story 22
- User story 23
- User story 26
- User story 28
- User story 30 (per-image quantities half)
