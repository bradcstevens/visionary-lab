## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

Frontend progress visualization. `ProgressTracker.tsx` gains
`kind: "per-image" | "per-project"`. `VariationThumbnail.tsx` overlays
the per-image bar driven by the SSE-fed jobs context. The project page
header renders a per-project aggregate bar. Both bars clearly
distinguish queued vs running (e.g. striped/indeterminate vs
determinate) and disappear once every contributing job reaches a
terminal state.

See PRD sections "Modified — `ProgressTracker.tsx`,
`VariationThumbnail.tsx`, `app/projects/[id]/page.tsx`" and user stories
22–24, 26.

## Acceptance criteria

- [ ] `ProgressTracker` accepts `kind` prop and renders per-image vs per-project styling
- [ ] Per-image bar overlays each generating tile with phase-aware visual state
- [ ] Per-project bar in header reflects aggregate of all active jobs for the project
- [ ] Both bars hide once all relevant jobs reach a terminal state
- [ ] Playwright test: queued regeneration shows queued state, transitions to running, then both bars disappear at completion
- [ ] Playwright multi-tab test: bars are consistent across tabs

## Blocked by

- Blocked by `006-frontend-jobs-context-sse.md`
- Blocked by `008-progress-estimator-and-emission.md`

## User stories addressed

- User story 22
- User story 23
- User story 24
- User story 26
