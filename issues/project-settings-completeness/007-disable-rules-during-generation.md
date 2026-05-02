# Disable rules during in-flight generation (with Danger Zone regression guard)

## Parent PRD

`prds/2026-05-01-project-settings-completeness-prd.md`

## What to build

Wire the per-control disable rules across the Project Settings tab so
that destructive or input-changing actions cannot race the in-flight
generation pipeline, while small housekeeping renames remain available.
Also serves as the regression guard for the existing Danger Zone
project-delete flow: this PRD must not change it.

Frontend-only changes — composes prior slices, no new components, no
backend work:

- `ProjectSettingsTab`
  (`frontend/components/staging/edit/ProjectSettingsTab.tsx`) computes
  a single `isGenerating = project.status === 'processing'` flag and
  applies the rules below.
- Disabled when `isGenerating`:
  - "Add photos" in `ProjectRoomsManager` (from 006).
  - "Delete" / "Yes, delete this room" in `ProjectRoomsManager` (from
    005). The inline confirm row is unreachable when `isGenerating` —
    either the row's "Delete" button is disabled, or the entire
    delete affordance is hidden. Either is acceptable as long as
    confirmation cannot be triggered.
  - "Save settings" button in `ProjectSettingsTab` (covers both prompt
    and generation-settings saves, since the page uses one save
    button).
- Allowed when `isGenerating`:
  - Per-room rename input in `ProjectRoomsManager` (from 004).
  - Project-name rename via `EditableProjectName` (existing
    component — verify it remains enabled).
  - Local prompt and generation-settings editing (the user can type
    into the textarea and dropdowns, but the "Save settings" button is
    disabled — combined with the existing `isDirty` check, this means
    typing produces a draft they can keep but not yet persist).
- Wire `disabled` from `ProjectSettingsTab` into `ProjectRoomsManager`
  using the rule "disabled for everything in the manager *except*
  rename" — concretely: `ProjectRoomsManager` accepts the existing
  `disabled` prop and treats it as "disable add and delete"; the
  rename input is never disabled by this prop. (If keeping a single
  boolean prop is too coarse, split into `{ disableAdd, disableDelete }`
  but keep the prop interface intentionally narrow per the PRD.)
- Danger Zone (existing "Delete project" flow) is NOT in scope to
  change. Add an E2E regression assertion that it still opens, still
  prompts for confirm, and still deletes — unchanged from current
  behavior.

The PRD's "Out of Scope" item "Allowing the user to change the model"
is implicitly enforced by 003 (model is read-only). No additional
disable rules needed here.

## Acceptance criteria

- [ ] When `project.status === 'processing'`:
  - "Add photos" affordance is disabled (no file picker opens on
    click).
  - Per-row "Delete" affordance cannot trigger a delete (button
    disabled or affordance hidden — confirm row is unreachable).
  - "Save settings" button is disabled regardless of `isDirty`.
  - Per-room rename input remains enabled and a successful rename
    still persists via `updateRoom`.
  - Project-name rename via `EditableProjectName` remains enabled.
  - The user can still type in the prompt textarea and change
    dropdowns locally; only the Save action is blocked.
- [ ] When `project.status` is anything other than `'processing'`,
      none of the controls above are disabled by this rule (other
      pre-existing disable rules continue to apply unchanged).
- [ ] The Danger Zone "Delete project" flow opens, prompts for
      confirmation, and deletes the project — unchanged from current
      behavior, and this remains true regardless of `project.status`
      (the PRD does not say to disable Danger Zone during generation,
      and we do not add that here).
- [ ] `frontend/tests/e2e/edit-project.spec.ts` is extended with an
      "active generation" scenario asserting each disable / allow rule
      above, plus a Danger Zone regression case; reports saved to
      `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.
- [ ] `cd frontend && npx playwright test`, `cd frontend && npm run build`,
      and `cd frontend && npx next lint` all pass locally.

## Blocked by

- Blocked by `002-settings-canonical-prompt.md` (Save-settings disable
  applies to the prompt save path).
- Blocked by `003-generation-settings-dropdowns.md` (Save-settings
  disable applies to the generation-settings save path).
- Blocked by `004-rooms-manager-scaffold-and-rename.md` (rename
  remains enabled — needs to exist).
- Blocked by `005-rooms-manager-delete-with-confirm.md` (delete is
  what we disable).
- Blocked by `006-rooms-manager-add-photos-with-analysis.md`
  (add-photos is what we disable).

## User stories addressed

Reference by number from the parent PRD:

- User story 18
- User story 19
- User story 22 (Danger Zone delete flow regression guard)
