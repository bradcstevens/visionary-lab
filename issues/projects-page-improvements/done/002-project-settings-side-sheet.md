# Project Settings side sheet (PATCH endpoint + editable form)

## Parent PRD

`prds/2026-04-30-projects-page-improvements-prd.md`

## What to build

Make project settings editable post-creation. The project header
overflow menu gains a "Project settings" entry that opens a side
sheet mirroring the wizard's editable surfaces (name, top-level
prompt, `StagingSettings`, design brief). Saving applies changes
to **future** generations only — every existing variation and its
prompt are untouched. Editing photos (add/remove/replace) is
explicitly out of scope.

End-to-end behavior:

- Backend: a new endpoint `PATCH /projects/{id}` accepts a partial
  update body containing any of `name`, `prompt`, `settings` (full
  `StagingSettings`), `design_brief`. Returns the updated project.
  Never modifies `rooms`, `analyses`, or `status`. Never triggers
  regeneration.
- Frontend: a new "Project settings" entry is added to the header
  overflow menu. Selecting it opens a side `Sheet` primitive
  containing a form matching the wizard's surfaces. Submit calls the
  new PATCH endpoint and reloads the project. The form makes it
  visually clear that changes apply to future generations only.
- Tests: endpoint unit test asserting partial update semantics, plus
  a Playwright scenario covering open → change → save → assert
  existing variations untouched → regenerate → assert new variations
  use the new settings.

See PRD sections **"Solution → 2. Project Settings side sheet"**,
**"Implementation Decisions → Backend modules"** (PATCH endpoint
bullet), **"Implementation Decisions → Frontend modules"** (the
"Project settings" entry bullet), and **"Testing Decisions →
Backend unit tests"** (`tests/test_staging_endpoints_patch_project
.py`).

## Acceptance criteria

- [ ] A new `PATCH /projects/{id}` endpoint accepts a partial body
      with any of `name`, `prompt`, `settings`, `design_brief`.
      Validates `settings` as a full `StagingSettings`. Returns the
      updated project.
- [ ] The endpoint never modifies `rooms`, `analyses`, or `status`,
      and never enqueues a regeneration.
- [ ] The project header overflow menu gains a "Project settings"
      entry (alongside existing entries).
- [ ] Selecting the entry opens a side `Sheet` primitive with a form
      mirroring the wizard's editable surfaces (name, top-level
      prompt, `StagingSettings` controls — variations-per-room,
      model, quality, size — and design brief).
- [ ] The form copy makes clear that saved changes apply to future
      generations only and do not rewrite completed variations.
- [ ] Submit calls `PATCH /projects/{id}` with only the changed
      fields and reloads the project on success. Errors surface
      via the existing toast pattern.
- [ ] Photo editing (add/remove/replace) is explicitly NOT added to
      the sheet — this slice does not change the photo set.
- [ ] `tests/test_staging_endpoints_patch_project.py` asserts:
      partial update only touches fields in the body; rooms/
      analyses/status are never modified; invalid `StagingSettings`
      returns a 4xx; a successful update returns the updated project.
- [ ] A new Playwright scenario covers: open Settings sheet from
      overflow menu; change variations-per-room from 5 to 3; save;
      assert existing variations are untouched; regenerate one room;
      assert the new variations match the new count.
- [ ] Local checks pass before commit:
      `uv run pytest tests/ --ignore=tests/integration -v`,
      `cd frontend && npx playwright test`,
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 5
- User story 6
- User story 7
- User story 8
- User story 9
