# Backend prompt ↔ design_brief.global_instructions mirror

## Parent PRD

`prds/2026-05-01-project-settings-completeness-prd.md`

## What to build

Add a small bidirectional mirror between `project.prompt` and
`project.design_brief.global_instructions` inside the existing staging
endpoints. This is the only backend change in the PRD and it makes
"the prompt" a single coherent value across Settings, Brief, gallery
dialogs, project cards, regenerate flows, and version snapshots —
without forcing the frontend to issue two writes per save.

The mirror lives in the endpoint layer (`backend/api/endpoints/staging.py`),
not in the storage layer. `staging_storage.py` stays a dumb persistence
shim.

Apply the mirror in two existing handlers (see the PRD's
"Backend mirror behavior" section):

- `update_project` (PATCH `/projects/{id}`):
  - If both `prompt` and `design_brief` are present in the request, the
    brief's `global_instructions` wins. Then `project.prompt` is set
    from that `global_instructions`.
  - If only `prompt` is present and the project has a `design_brief`,
    copy the new prompt into `design_brief.global_instructions`.
  - If only `design_brief` is present, set `project.prompt` from the new
    `global_instructions` — but only if `global_instructions` is a
    non-empty string; otherwise leave `project.prompt` untouched.
- `update_brief` (PUT `/projects/{id}/brief`): always set
  `project.prompt` from the brief's `global_instructions` when
  non-empty.

The mirror only operates on these inbound update endpoints. Version
snapshot restore already writes both `prompt` and `design_brief`
atomically, so revert remains coherent without changes to that path —
this issue only needs a regression-guarding test for the revert flow,
not new revert code.

No frontend changes in this slice. The slice is end-to-end verifiable
at the HTTP boundary via pytest.

## Acceptance criteria

- [ ] `update_project` with `prompt` only and a brief present mirrors
      the new prompt into `design_brief.global_instructions`.
- [ ] `update_project` with `prompt` only and `design_brief is None`
      changes only `prompt` (no spurious brief creation).
- [ ] `update_project` with `design_brief` only and a non-empty
      `global_instructions` mirrors that value into `project.prompt`.
- [ ] `update_project` with `design_brief` only and missing/empty
      `global_instructions` leaves `project.prompt` untouched.
- [ ] `update_project` with both `prompt` and `design_brief` present
      uses the brief's `global_instructions` for both fields (brief
      wins).
- [ ] `update_brief` (PUT) with non-empty `global_instructions` mirrors
      that value into `project.prompt`.
- [ ] The existing `update_project` per-project asyncio lock and the
      "settings merge" semantics from issue 002 of
      projects-page-improvements remain intact (no regression).
- [ ] A pytest test demonstrates that restoring a version snapshot
      (existing flow) still ends with `project.prompt` and
      `design_brief.global_instructions` equal to each other (regression
      guard for user story 20).
- [ ] All 6 mirror cases from the PRD's "Backend (pytest)" section are
      covered by black-box tests at the HTTP boundary, asserting
      observable behavior of the persisted project document.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` passes.

## Blocked by

None — can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 3 (backend half: keeps prompt and brief in sync server-side)
- User story 20 (revert restores both fields consistently — regression
  guard via snapshot-restore test)
