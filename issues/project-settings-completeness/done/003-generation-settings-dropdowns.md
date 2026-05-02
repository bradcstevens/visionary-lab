# Generation settings: dropdowns + size + read-only model

## Parent PRD

`prds/2026-05-01-project-settings-completeness-prd.md`

## What to build

Replace the current Settings-tab Generation Settings UI so it mirrors
every wizard-exposed generation parameter and uses constrained inputs
instead of a free-text field.

Frontend-only changes:

- `GenerationSettingsForm`
  (`frontend/components/staging/GenerationSettingsForm.tsx`):
  - Replace the free-text `quality` input with a dropdown of `low`,
    `medium`, `high`, `auto`.
  - Add a `size` dropdown of `auto`, `1024x1024`, `1024x1536`,
    `1536x1024`. Thread `size` through `onChange`.
  - Add a read-only label that displays the project's current `model`
    value (`project.settings.model`). Make clear in the UI that it is
    read-only.
  - Keep the existing prop shape: `{ settings, onChange, disabled }`.
  - Keep the existing `variations_per_room` integer 1–10 input.
- `useProjectSettings` payload diff
  (`frontend/hooks/staging/useProjectSettings.ts`):
  - Include `size` in the outgoing `settings` payload when it changed
    (currently silently dropped).
  - **Never** include `model` in the outgoing payload, even if
    `draft.settings.model` differs from `project.settings.model`
    (defensive — model is read-only).
  - The existing settings-merge semantics on the backend mean we still
    only send the keys that actually changed.
- `ProjectSettingsTab` does not need new wiring beyond what
  `GenerationSettingsForm` already encapsulates — but verify it renders
  the new fields and that `Save settings` / `Discard changes` continue
  to work with the new fields included in the dirty check.
- `RegeneratePrompt` banner still appears after a save that included a
  generation-settings change.

The backend already supports `quality`, `size`, and `model` in
`StagingSettings` and `UpdateProjectRequest` — no backend work in this
slice. The PRD calls out that backend validation of dropdown values is
out of scope; invalid values surface as generate-time errors via the
existing toast path.

The PRD's out-of-scope `style` and `room_count` fields are NOT added to
the form (vestigial — wizard never sets them).

## Acceptance criteria

- [ ] The "Quality" control is a dropdown with exactly these options:
      `low`, `medium`, `high`, `auto`. The dropdown shows the project's
      current value as the initial selection.
- [ ] The "Size" control is a dropdown with exactly these options:
      `auto`, `1024x1024`, `1024x1536`, `1536x1024`. The dropdown shows
      the project's current value as the initial selection.
- [ ] The "Model" control is rendered as a read-only label showing
      `project.settings.model`; there is no input affordance.
- [ ] Changing `quality` and clicking "Save settings" persists the new
      value across a hard reload.
- [ ] Changing `size` and clicking "Save settings" persists the new
      value across a hard reload.
- [ ] After saving a generation-settings change, the existing
      `RegeneratePrompt` banner appears.
- [ ] `useProjectSettings.save` never includes `model` in the
      `settings` portion of the payload (asserted via
      network-intercept in the Playwright test, or via a small frontend
      hook test if a hook test runner already exists per the PRD).
- [ ] "Discard changes" reverts the dropdowns to the project's
      currently-persisted values for both `quality` and `size`.
- [ ] `frontend/tests/e2e/edit-project.spec.ts` is extended with the
      above assertions; reports saved to
      `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.
- [ ] `cd frontend && npx playwright test`, `cd frontend && npm run build`,
      and `cd frontend && npx next lint` all pass locally.

## Blocked by

None — can start immediately. Independent of the prompt-mirror work.

## User stories addressed

Reference by number from the parent PRD:

- User story 9
- User story 10
- User story 11
- User story 12
- User story 13
- User story 14
