# 001 — Backend generic object model, migration, and resolver foundation

## Parent PRD

`issues/2026-04-29-per-image-object-quantities-prd.md`

## What to build

The backend half of replacing `plant_palette` with a generic `object_palette`. This is the foundation every later slice depends on, so it must land first. It is not user-visible until issue `002` updates the frontend to consume the new shape.

End-to-end behaviour delivered by this slice:

- Every brief — newly generated, persisted, or returned by the API — exposes `object_palette` (never `plant_palette`) and entries carry a stable UUID `id`, `name`, optional `description`, `category`, `default_quantity`, `size`, `placement`, optional `visual_notes`.
- Legacy persisted briefs are migrated transparently. Opening a project that still has `plant_palette` in Cosmos returns `object_palette` to the caller and writes the migrated dict back to Cosmos opportunistically on first read. No code path can surface legacy keys.
- A new `brief_resolver` module owns the pure operations (no LLM dependency) so they can be unit-tested in isolation: `migrate_legacy_plant_palette`, a basic `resolve_objects_for_image` that returns palette entries projected into `ResolvedObject` (override handling lands in issue `003`), and the `ResolvedObject` dataclass.
- `BRIEF_TO_PROMPTS_TEMPLATE` uses the generic placeholder `{object_summary}` and generic surrounding copy ("Objects:" rather than "Plants:"). `INDOOR_PROMPT_TEMPLATE` and `OUTDOOR_PROMPT_TEMPLATE` are explicitly **not** changed.
- Category mis-classifications from the LLM (`"Plant"`, `"plants"`, `"shrub"`, `"bush"`, `"light"`, hallucinated values) are silently coerced rather than raising a validation error.

See PRD sections "Data model", "Resolution logic" (the migration + basic-resolve subset), "Backwards-compat migration", "Prompt-to-image rendering", and "API" (GET project endpoint).

## Acceptance criteria

- [ ] `ObjectCategory` string enum exists with values `plant`, `tree`, `rock`, `furniture`, `lighting`, `hardscape`, `decor`, `other`.
- [ ] `ObjectEntry` model exists with the fields listed in the PRD and a field-level coercion validator on `category` that lowercases, strips a trailing `s`, maps known synonyms (`shrub`/`bush` → `plant`, `light` → `lighting`), and falls back to `OTHER` for unknown values without raising.
- [ ] `PlantEntry` is removed; nothing in the backend imports it.
- [ ] `DesignBrief.plant_palette` is replaced with `object_palette: List[ObjectEntry]`. A `model_validator(mode='before')` invokes `migrate_legacy_plant_palette` so any raw legacy dict is migrated on construction.
- [ ] `brief_resolver` module exists (separate from `BriefGeneratorService`, no LLM dependency) and exports: `migrate_legacy_plant_palette`, `resolve_objects_for_image`, `ResolvedObject`. (Override-aware behaviour and `reconcile_overrides_by_name` are deferred to later issues.)
- [ ] `migrate_legacy_plant_palette` is pure and idempotent: returns input unchanged if `object_palette` already present; otherwise converts each `PlantEntry` into an `ObjectEntry` (fresh UUID; `species` → `name`, `botanical_name` → `description`, `quantity` → `default_quantity`; copies `size`, `placement`, `visual_notes`); chooses `category` heuristically (numeric ≥ 6 with `ft`/`feet`/`tall` in `size` OR a tree-name token in `species` → `TREE`, else `PLANT`); drops the legacy `plant_palette` key.
- [ ] GET project endpoint runs `migrate_legacy_plant_palette` on the persisted `design_brief` dict before serialising the response and writes the migrated dict back to Cosmos opportunistically on first read; subsequent reads are no-ops.
- [ ] `BRIEF_TO_PROMPTS_TEMPLATE` uses `{object_summary}` and generic "Objects:" copy. `BriefGeneratorService.brief_to_prompts` builds an `object_summary` from the resolved object list (a single project-wide summary in this slice; per-room summaries land in issue `003`).
- [ ] `INDOOR_PROMPT_TEMPLATE` and `OUTDOOR_PROMPT_TEMPLATE` are unchanged.
- [ ] `BRIEF_GENERATION_PROMPT` is updated to extract a generic `object_palette` (with category per entry); `per_image_objects` extraction is deferred to issue `004`. `BriefGeneratorService.generate_brief` assigns a UUID to each new palette entry.
- [ ] Unit tests: parameterised category-coercion table covering `"Plant"`, `"plants"`, `"shrub"`, `"bush"`, `"light"`, `"plant_tree_hybrid"`; legacy migration tests covering UUID assignment, heuristic tree categorisation, and idempotency; `DesignBrief(**raw)` test asserting legacy dicts are migrated automatically; GET project endpoint test asserting the response body contains `object_palette` (not `plant_palette`) for a project whose persisted dict still has legacy keys, and that the migrated form is written back to Cosmos.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` passes.

## Blocked by

None — can start immediately.

## User stories addressed

- User story 1
- User story 2
- User story 13
- User story 14
- User story 15
- User story 18
- User story 19
- User story 24
- User story 25
- User story 27
- User story 29
