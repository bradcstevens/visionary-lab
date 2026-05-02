# Project rooms manager: delete with confirm

## Parent PRD

`prds/2026-05-01-project-settings-completeness-prd.md`

## What to build

Add inline delete-with-confirm to `ProjectRoomsManager`. Deleting a
room cascades its variations and storage cleanup (existing backend
behavior of `removeRoom`) and the project card on the Projects list
updates its room count after the delete returns.

Frontend-only changes (existing backend `removeRoom` endpoint already
cascades variations and blob cleanup):

- Each room row gains a "Delete" action that opens an inline confirm
  row in the same component (no modal popover from a different mount
  point — keep the deep-module rule from the PRD): "Delete this room
  and all variations?" with explicit "Yes, delete" and "Cancel"
  buttons.
- On "Yes, delete", call `removeRoom(project.id, room.id)`, then call
  `onProjectUpdate(updatedProject)` with the response payload (which
  is the new project state). No full page reload; no
  optimistic-then-rollback dance — wait for the server to confirm.
- On "Cancel", simply collapse the confirm row.
- If `removeRoom` fails, the confirm row stays visible with an inline
  error and the room row is preserved. Match the existing failure UX
  used elsewhere in the staging flow.
- The Projects-list project card already reads the project's room count
  from the project document — verify the count updates after the delete
  is reflected upstream (this is a propagation regression guard, not
  new card code).

## Acceptance criteria

- [ ] Each room row shows a "Delete" action in addition to the rename
      affordance from 004.
- [ ] Clicking "Delete" reveals an inline confirm row with explicit
      "Yes, delete" and "Cancel" buttons in the same component (not a
      modal mounted elsewhere).
- [ ] Confirming the delete calls `removeRoom`, awaits the response,
      and only then removes the row from the rendered list via
      `onProjectUpdate`.
- [ ] After a successful delete:
  - the row is gone from the Settings rooms list,
  - the room's variations are gone from the Gallery tab,
  - the project card on the Projects list page shows the updated room
    count after a hard reload.
- [ ] Cancelling the confirm row leaves the room row intact and does
      not call any API.
- [ ] If `removeRoom` returns an error, the row remains and an inline
      error is shown; no project state mutation occurs.
- [ ] The disable-while-processing rule for delete is NOT implemented
      here — it lands in 007. (`disabled` prop is forwarded; this slice
      does not change the value being passed.)
- [ ] `frontend/tests/e2e/edit-project.spec.ts` is extended with a
      delete-with-confirm E2E case; reports saved to
      `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.
- [ ] `cd frontend && npx playwright test`, `cd frontend && npm run build`,
      and `cd frontend && npx next lint` all pass locally.

## Blocked by

- Blocked by `004-rooms-manager-scaffold-and-rename.md`
  (`ProjectRoomsManager` and its row rendering must exist first).

## User stories addressed

Reference by number from the parent PRD:

- User story 6
- User story 15 (delete half)
- User story 16
