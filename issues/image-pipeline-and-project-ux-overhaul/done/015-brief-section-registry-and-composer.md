## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`BriefSectionRegistry` — single source of truth for the eight canonical
sections (Edit Task, Edit Zone, Do Not Alter, Object Palette,
Arrangement, Regional Constraints, Aesthetic Goal, Scale & Fidelity) —
drives both the wizard step config and the settings panel tab config.
`DesignBrief` model extended with the eight sections and `raw_override`.
`PromptComposer` extended to render the eight-section brief
deterministically into the top-level prompt markdown, honoring
`raw_override` when set.

See PRD sections "BriefSectionRegistry", "PromptComposer (extended)",
"Modified — `models/design_brief.py`", and user stories 32, 35.

## Acceptance criteria

- [ ] `BriefSectionRegistry` exports the eight canonical sections with stable ids and ordering
- [ ] `DesignBrief` schema includes the eight sections and `raw_override`
- [ ] `PromptComposer` renders sections deterministically; same input → identical markdown output
- [ ] When `raw_override` is set, composer returns it unchanged
- [ ] Unit test: round-trip sections → rendered markdown → re-extracted sections
- [ ] Unit test: `raw_override` wins when set
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` passes

## Blocked by

None - can start immediately.

## User stories addressed

- User story 32
- User story 35
