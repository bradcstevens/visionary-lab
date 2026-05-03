## Parent PRD

`prds/2026-05-02-projects-page-stalled-stream-error-cleanup-prd.md`

## What to build

Extract a single `RoomStatusPill` component that owns all five visual
room states — `pending`, `processing`, `completed`, `failed`, and the
new amber `stalled` — and migrate every inline pill render on the
project detail page and `ProgressTracker` to use it. The component takes
`(status, projectRecoveryState)` and renders the amber stalled treatment
when `status === 'processing'` and
`recovery.kind === 'interrupted' || recovery.kind === 'stream-lost'`.
The four pre-existing visuals stay pixel-identical to today; only the
new `stalled` (amber) treatment is added.

See PRD section "`RoomStatusPill` visuals" and the `RoomStatusPill`
bullet under "Modules touched".

## Acceptance criteria

- [ ] New component `RoomStatusPill({ status, projectRecoveryState })`
      renders the five visual states
- [ ] Component exposes `data-status={status}` and (when stalled
      treatment fires) `data-stalled="true"` for behavioral testing
- [ ] Stalled treatment fires iff `status === 'processing' &&
      (recovery.kind === 'interrupted' || recovery.kind === 'stream-lost')`
- [ ] All inline pill renders in the project detail page room list and
      in `ProgressTracker` are migrated to the component
- [ ] The four pre-existing pill visuals
      (`pending`/`processing`/`completed`/`failed`) are pixel-identical
      to the pre-migration rendering
- [ ] One unit test per `(status, projectRecoveryState)` combination that
      produces a visually distinct treatment, asserting `data-status`
      and (where applicable) `data-stalled` rather than CSS class names
      or snapshots

## Blocked by

- Blocked by `002-recovery-state-classifier-unified-banner.md`

## User stories addressed

- User story 5
- User story 6
- User story 14
