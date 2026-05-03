## Parent PRD

`prds/2026-05-02-projects-page-stalled-stream-error-cleanup-prd.md`

## What to build

Introduce the `getRecoveryState` classifier (pure tagged-union function,
co-located with `staging-header.ts`) and use it to collapse the three
existing in-line banner blocks on the project detail page into a single
banner driven by `recoveryState.kind`. The header status badge derives
from the same classifier (so `stream-lost`/`interrupted` render an
`interrupted` badge instead of `processing`), and the header CTA is
hidden whenever `recoveryState.kind !== 'none'`. Each banner arm renders
its primary/secondary buttons with the side-effect chains specified in
the PRD and exposes `data-testid="recovery-banner"`,
`data-recovery-kind={kind}`, and per-button test ids. The
`isStaleProcessing` derived boolean is deleted.

See PRD sections "`getRecoveryState` contract", "Header status badge",
and "Banner block".

## Acceptance criteria

- [ ] New module exports `getRecoveryState(input): RecoveryState` with
      the discriminated union and precedence specified in the PRD
      (`error` > `stream-lost` > `interrupted` > `none`)
- [ ] Unit test table covers every precedence edge (error+lost-op,
      lost-op+interrupted, error+interrupted+lost-op, all-empty)
- [ ] Project detail page calls `getRecoveryState` once per render with
      `projectStatus`, `isAnyInFlight`, project-scoped `fleet.lostOps`,
      and `parseApiError`-parsed `generationError`
- [ ] The three existing banner blocks (`generationError`,
      `isStaleProcessing`, project-scope `fleet.lostOps`) are deleted and
      replaced with one block switching on `recoveryState.kind`
- [ ] Banner exposes `data-testid="recovery-banner"`,
      `data-recovery-kind`, and `recovery-banner-primary` /
      `recovery-banner-secondary` / `recovery-banner-detail` test ids
- [ ] `error` arm: red destructive variant; primary `Retry` clears
      `generationError`, dismisses every project-scope lost op, then
      calls `handleRegenerateAll`
- [ ] `stream-lost` arm: amber; primary "Retry generation" calls
      `dismissLostOp(lostOpId)` then `startGeneration()` (no
      `loadProject` interleave); secondary "Dismiss" calls
      `dismissLostOp(lostOpId)` then `loadProject()`
- [ ] `interrupted` arm: amber; primary "Reset & Retry" calls
      `handleResetProject`; secondary "Refresh" calls `loadProject`; no
      Dismiss button
- [ ] Each primary button renders a one-line side-effect helper sentence
      beneath it
- [ ] Header `<Badge>` maps `recoveryState.kind` to badge label/variant
      before falling through to `project.status`; `error` →
      destructive, `stream-lost`/`interrupted` → secondary amber
      `interrupted`, `none` → existing behavior
- [ ] Header CTA is hidden (not disabled) at the page call site whenever
      `recoveryState.kind !== 'none'`; `getHeaderAction(rooms)` signature
      is unchanged
- [ ] `isStaleProcessing` derived boolean is removed
- [ ] Existing specs that copy-match "Generation encountered an error"
      for a watchdog fire are migrated from copy-matchers to the new
      `data-testid` selectors

## Blocked by

- Blocked by `001-synthetic-watchdog-event-flag.md`

## User stories addressed

- User story 1
- User story 2
- User story 3
- User story 4
- User story 7
- User story 8
- User story 10
- User story 11
- User story 12
