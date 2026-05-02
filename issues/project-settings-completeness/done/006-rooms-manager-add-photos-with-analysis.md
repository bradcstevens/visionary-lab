# Project rooms manager: add photos with analysis + retry

## Parent PRD

`prds/2026-05-01-project-settings-completeness-prd.md`

## What to build

Add the "Add photos" affordance to `ProjectRoomsManager` so a returning
user can expand the scope of an existing project from Settings without
recreating it. New rooms are first-class — they go through the same
backend analysis the wizard runs, and they appear in the rooms list
with the same metadata as the originals.

Frontend-only changes (uses existing `uploadRooms`, `analyzeImages`,
and `getProject` API client functions in
`frontend/services/stagingApi.ts`):

- Add an "Add photos" button to `ProjectRoomsManager`. Clicking it
  opens a hidden file input that accepts multiple image files.
- On file selection, run the upload→analyze→refresh sequence:
  1. `uploadRooms(project.id, [{ file, name }, ...])` — backend
     creates the new room records.
  2. `analyzeImages(project.id)` — backend runs the same image
     analysis pipeline used by the wizard.
  3. `getProject(project.id)` — refetch the canonical project state,
     run the existing SAS-token resolution, then call
     `onProjectUpdate(updatedProject)`. No full page reload.
- If step 2 (analysis) fails but step 1 (upload) succeeded, the upload
  is preserved (the new rooms exist on the project). Surface a
  non-blocking toast: "Couldn't analyze the new photos." with a
  "Retry analysis" action that re-runs `analyzeImages` followed by
  the same project refresh.
- If step 1 (upload) fails, surface an error toast and do not mutate
  project state.
- The design brief is **not** automatically regenerated when rooms are
  added — the user keeps their edits, and the existing Brief tab +
  Regenerate banner remain the path for incorporating new rooms into
  the brief. Do not call any brief-generation endpoint from this
  slice.

The disable-while-processing rule for "Add photos" lands in 007 — this
slice continues to forward `disabled` from `ProjectSettingsTab`
unchanged.

Note that `ProjectRoomsManager` MUST still respect its narrow prop
interface — the upload→analyze→refresh sequence is implemented
entirely within the component (or a private hook in the same file),
not by leaking new props upward.

## Acceptance criteria

- [ ] `ProjectRoomsManager` renders an "Add photos" affordance.
- [ ] Selecting one or more image files triggers the
      upload→analyze→refresh sequence in order, awaits each step, and
      then calls `onProjectUpdate` with the refetched project.
- [ ] After a successful add, each new room appears in the rooms list
      with its analyzed metadata (label, thumbnail) populated, just
      like wizard-uploaded rooms.
- [ ] If `analyzeImages` fails, the new room rows still appear (the
      upload is preserved), and a non-blocking toast offers "Retry
      analysis" that re-runs `analyzeImages` + project refresh.
- [ ] If `uploadRooms` fails, no new rows appear, no project state
      mutation occurs, and an error toast is shown.
- [ ] No call is made to any brief-generation or brief-regenerate
      endpoint as part of this flow (verified via network-intercept in
      the Playwright test).
- [ ] The disable-while-processing rule for "Add photos" is NOT
      implemented here — it lands in 007. (`disabled` prop is
      forwarded; this slice does not change the value being passed.)
- [ ] `frontend/tests/e2e/edit-project.spec.ts` is extended with both
      an add-photos-success case and an analysis-failure-then-retry
      case; reports saved to
      `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.
- [ ] `cd frontend && npx playwright test`, `cd frontend && npm run build`,
      and `cd frontend && npx next lint` all pass locally.

## Blocked by

- Blocked by `004-rooms-manager-scaffold-and-rename.md`
  (`ProjectRoomsManager` and its rooms-list rendering must exist
  first).

## User stories addressed

Reference by number from the parent PRD:

- User story 7
- User story 8
- User story 17
