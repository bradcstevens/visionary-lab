# Project rooms manager: scaffold + inline rename

## Parent PRD

`prds/2026-05-01-project-settings-completeness-prd.md`

## What to build

Introduce the new `ProjectRoomsManager` component as a deep module
mounted on the Project Settings tab between "Project details" and
"Generation settings", and ship the first room operation it owns:
inline rename.

Frontend-only changes (uses existing backend `updateRoom` endpoint and
`updateRoom` API client function in `frontend/services/stagingApi.ts` —
no new network surface in this slice):

- New `ProjectRoomsManager` component
  (e.g. `frontend/components/staging/edit/ProjectRoomsManager.tsx`)
  with the exact prop interface called out in the PRD:
  `{ project, onProjectUpdate, disabled }`. The component MUST NOT
  reference routing, lightboxes, SSE streams, or generation state
  beyond its `disabled` prop.
- Renders the project's rooms as a list, each row showing the room
  thumbnail (or label-only fallback if not yet available), the editable
  room label, and per-row action affordances.
- Inline rename behavior follows the existing rename-project pattern in
  `frontend/components/staging/edit/EditableProjectName.tsx`: the user
  enters edit mode, types the new label, and the component awaits
  `updateRoom(project.id, room.id, { label })`; only after the API
  resolves does it call `onProjectUpdate(updatedProject)` so local
  state and server state never diverge.
- Persists per-action immediately — there is no Save/Discard for room
  edits (matches the PRD's "room operations persist immediately per
  action" rule).
- Mounted in `ProjectSettingsTab` between the "Project details"
  section and the "Generation settings" section. The section heading
  is "Rooms".
- The `disabled` prop is wired through from `ProjectSettingsTab` but
  this slice does NOT yet implement the per-action disable rules from
  the PRD (those land in 007). For now, simply accept and forward
  `disabled` to the rename input as a baseline so 007 only has to
  thread the right value down.

This slice intentionally ships rename only. Delete and add are 005 and
006 respectively, both blocked by this scaffold.

## Acceptance criteria

- [ ] `ProjectRoomsManager` exists as a new component file with the
      exact prop interface `{ project, onProjectUpdate, disabled }` and
      no other props.
- [ ] The component does not import routing primitives, lightbox
      components, SSE clients, or any module that exposes generation
      state — only the staging API client functions it needs and shared
      UI primitives.
- [ ] The Project Settings tab renders the new "Rooms" section between
      "Project details" and "Generation settings".
- [ ] Each room row displays the room's current label and offers an
      edit affordance.
- [ ] Renaming a room calls `updateRoom` and only updates the local
      project state via `onProjectUpdate` after the API call resolves
      successfully (server-confirmed pattern).
- [ ] After a successful rename, the new label persists across a hard
      reload of the Project Settings tab.
- [ ] If `updateRoom` fails, the row reverts to the previous label and
      the error is surfaced (toast or inline text — match the existing
      project-rename error UX).
- [ ] When `disabled` is true, the rename input is also disabled
      (forward-only — the project-status-driven value is set in 007).
- [ ] `frontend/tests/e2e/edit-project.spec.ts` is extended with a
      rename E2E case; reports saved to
      `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.
- [ ] `cd frontend && npx playwright test`, `cd frontend && npm run build`,
      and `cd frontend && npx next lint` all pass locally.

## Blocked by

None — can start immediately. Independent of the prompt and
generation-settings slices.

## User stories addressed

Reference by number from the parent PRD:

- User story 5
- User story 15 (rename half — delete and add land in 005 and 006)
