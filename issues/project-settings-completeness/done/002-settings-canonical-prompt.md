# Settings tab shows the canonical project prompt

## Parent PRD

`prds/2026-05-01-project-settings-completeness-prd.md`

## What to build

Make the Project Settings tab show the *real* prompt the user authored
in the AI Design Session, not the wizard's `"Draft — pending AI Design
Session"` placeholder. Saving the prompt from Settings persists it in
the right place and propagates everywhere the prompt is rendered.

Frontend-only changes (the backend mirror that keeps `project.prompt`
and `design_brief.global_instructions` in sync lives in 001):

- `useProjectSettings` hook
  (`frontend/hooks/staging/useProjectSettings.ts`) derives the displayed
  prompt as `design_brief.global_instructions ?? project.prompt`. The
  draft baseline used for `isDirty` and `reset` follows the same rule.
- The save payload routes the prompt edit:
  - When a `design_brief` exists, the edit is sent as a partial
    `design_brief` update (only `global_instructions` changes; other
    brief fields are preserved).
  - When no `design_brief` exists, the edit is sent as a plain `prompt`
    field on the `UpdateProjectRequest`.
- `ProjectSettingsTab`
  (`frontend/components/staging/edit/ProjectSettingsTab.tsx`) renders a
  small dismissible hint *only* when `design_brief` is null:
  "Once a design brief exists, your prompt is stored as part of it."
- Save flow continues to surface the existing `RegeneratePrompt` banner
  when the saved payload included a prompt change.
- The Projects-list project card preview reflects the new prompt after
  save (no extra work expected — this is a propagation regression
  guard, since the card already reads `project.prompt`, which the
  backend mirror keeps current).

No new tabs. No moves of existing functionality.

## Acceptance criteria

- [ ] Opening Project Settings on a project that has a `design_brief`
      shows the design brief's `global_instructions` in the
      "Project prompt" textarea — not the
      `"Draft — pending AI Design Session"` placeholder.
- [ ] Editing the prompt and clicking "Save settings" persists the new
      value across a hard reload.
- [ ] After save, the new prompt is visible:
  - in the Brief tab as `global_instructions`,
  - on the Projects list card preview for that project,
  - inside the per-image edit dialog in the gallery (existing surface,
    unchanged code path — verify no regression).
- [ ] When `design_brief` is null, the Settings tab still shows and
      allows editing the prompt via the legacy `project.prompt` field,
      and renders the "Once a design brief exists, your prompt is
      stored as part of it." hint.
- [ ] After saving a prompt edit, the existing `RegeneratePrompt`
      banner appears.
- [ ] The "Discard changes" button still resets the draft to the
      currently-displayed canonical prompt (not to a stale
      `project.prompt` value).
- [ ] Existing project-name editing in Settings still works unchanged.
- [ ] `frontend/tests/e2e/edit-project.spec.ts` is extended with the
      above assertions; reports saved to
      `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.
- [ ] `cd frontend && npx playwright test`, `cd frontend && npm run build`,
      and `cd frontend && npx next lint` all pass locally.

## Blocked by

- Blocked by `001-backend-prompt-brief-mirror.md` (the frontend's
  "send only the prompt and trust the server to mirror it" save model
  depends on the backend mirror landing first).

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 3
- User story 4
- User story 21
