## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`ProjectSettingsSheet.tsx` rewritten to be driven by
`BriefSectionRegistry` so the wizard and the settings panel always
expose the same eight sections. Adds a read-only rendered-prompt
preview tab so users can see exactly what will be sent to the model.
`raw_override` controls allow a power user to write the prompt by hand
with a visible banner and a one-click revert. Whenever a section is
changed, a Regenerate affected images button appears so re-runs are
explicit and never accidental.

See PRD sections "Modified — `ProjectSettingsSheet.tsx`" and user
stories 33–36.

## Acceptance criteria

- [ ] Settings panel renders one tab per section from `BriefSectionRegistry`; ordering matches the wizard
- [ ] Editing a field PATCHes the project; no jobs created on save
- [ ] Read-only rendered-prompt preview tab shows composer output that updates after each save
- [ ] `raw_override` toggle activates a banner; one click reverts to structured composition
- [ ] Regenerate affected images button appears only after a section change and triggers the regenerate endpoint
- [ ] Playwright test: edit a section → save → preview updates → no jobs created; click Regenerate → jobs appear
- [ ] Playwright test: enable `raw_override` → banner visible → revert restores composed prompt
- [ ] `cd frontend && npm run build` and `npx next lint` pass

## Blocked by

- Blocked by `004-rest-enqueue-list-cancel.md`
- Blocked by `015-brief-section-registry-and-composer.md`

## User stories addressed

- User story 33
- User story 34
- User story 35
- User story 36

## Implementation note

Implemented as a registry-driven `BriefSectionsEditor` sub-component
mounted inside the existing `ProjectSettingsSheet`. The wholesale
"rewrite" framing in the issue body is honored structurally — the
sections / preview / raw_override / regenerate surface is fully
driven by `BriefSectionRegistry` and the saved-prompt symmetry with
the wizard is now testable from a single source of truth — without
churning the unrelated name / variations / quality / size editing
surface that already shipped from earlier slices and that has its
own test coverage.

Playwright tests deferred (consistent with the pattern in commits
21deb8e, e4779c4, 1747681): both Playwright lines need a backend
fixture project with structured `sections` + non-null
`raw_override`, plus a running queue (issue 001 emulator gate).
Behavior is pinned by 38 new vitest specs (8 registry, 15
composer, 15 editor) under jsdom.
