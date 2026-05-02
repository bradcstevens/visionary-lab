## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

When a project is deleted, all its in-flight jobs (`pending` or
`running`) transition to `cancelled` and the worker stops dispatching
their messages. No leaked Azure compute or storage cost. The cascade is
driven by the project-delete endpoint, which uses `JobStore` to mark
`cancel_requested` on every non-terminal job for that `project_id`
before returning.

See PRD section "Integration tests — Project-deletion cascade" and user
story 18.

## Acceptance criteria

- [ ] Project-delete endpoint marks all non-terminal jobs for the project as `cancel_requested`
- [ ] Worker observes the flag and transitions each job to `cancelled` at the next safe point
- [ ] Integration test: create project → enqueue several jobs → delete project → assert every job reaches `cancelled` and no further blob writes occur
- [ ] Deleted project no longer surfaces in `GET /jobs` for that id

## Blocked by

- Blocked by `004-rest-enqueue-list-cancel.md`

## User stories addressed

- User story 18
