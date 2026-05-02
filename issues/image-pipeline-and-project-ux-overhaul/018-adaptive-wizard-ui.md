## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`NewProjectWizard.tsx` rewritten as an 8-step skeleton driven by
`BriefSectionRegistry`. Step 1 asks what the user is visualizing; the
domain classifier and `RegionalPackLoader` then tailor subsequent steps
with AI-generated questions and quick-reply chips. Partial wizard
progress is autosaved to the in-progress project record so the user can
leave and return. On completion, the structured `DesignBrief` (eight
sections) is composed deterministically by `PromptComposer` into the
top-level prompt. Behind `FEATURE_ADAPTIVE_WIZARD`.

This slice is HITL — the wizard's question phrasing, chip layout, and
step transitions are user-facing UX decisions worth a design review
before merge.

See PRD sections "Modified — `NewProjectWizard.tsx`", "Feature flags",
and user stories 27–32.

## Acceptance criteria

- [ ] Wizard renders 8 steps from `BriefSectionRegistry`
- [ ] Step 1 captures the visualization domain; classifier loads the matching regional pack
- [ ] Each step shows quick-reply chips sourced from the pack and AI-generated follow-up questions
- [ ] Partial progress autosaves; reloading the wizard restores state
- [ ] Final submit composes the brief via `PromptComposer` and creates a project whose top-level prompt matches the canonical sections
- [ ] Playwright test: full wizard run against `tests/projects/backyard-landscaping` asserts the rendered top-level prompt matches the canonical section structure
- [ ] `FEATURE_ADAPTIVE_WIZARD` defaults to true in dev/staging
- [ ] `cd frontend && npm run build` and `npx next lint` pass
- [ ] Design review sign-off recorded in PR

## Blocked by

- Blocked by `015-brief-section-registry-and-composer.md`
- Blocked by `017-regional-pack-loader.md`

## User stories addressed

- User story 27
- User story 28
- User story 29
- User story 30
- User story 31
- User story 32
