# Retry queue core: `useRetryQueue` hook + thumbnail Queued indicator + toast + activity log

## Parent PRD

`prds/2026-04-30-failed-variation-retry-queue-prd.md`

## What to build

The core tracer bullet for the failed-variation retry queue. Build
the `useRetryQueue` hook (the single deep module introduced by this
PRD), wire it through the project detail page and the existing
`RoomGroup` → `VariationThumbnail` chain, and surface the queued
state via a thumbnail indicator, a toast, and an activity-log entry.

End-to-end behavior to demo: the user is on the project detail page
during in-flight staged generation. One variation fails mid-stream
and renders the Retry button. The user clicks Retry. Immediately,
the thumbnail's Retry button is replaced with a "Queued" indicator
(spinner + badge + caption) visually distinct from both the
processing-state spinner and the failed-state error tile, a
`toast.info("Retry queued — will run when generation completes")`
appears, and an info-level activity-log entry is added. Subsequent
clicks on the same failed variation while it is queued are
deduplicated silently (no extra toast, no extra log). Once the
global generation stream terminates and `isGenerating` returns to
false, the queued retry fires serially via the existing
`handleRegenerateVariation` callback with `strategy: 'fresh'`. If
the queue contains multiple entries, they fire one at a time, each
waiting for `regeneratingVariationId` to clear before the next
fires.

The hook also implements the drain-time drop rule: when popping a
queued entry, if the variation no longer exists in the current
`project` snapshot OR its status is no longer `'failed'`, the
entry is dropped without dispatching and the next entry is
considered. The activity-log hook for that drop event lives on the
consumer side per the PRD.

`clear()` is exposed for slices 003 and 004 but is NOT yet wired
into the supersede or drop-on-error code paths in this slice; those
two slices add the call sites.

See PRD sections "Deep module: `useRetryQueue` hook", "Page
integration" (excluding the `clear()` call sites), "Component prop
flow" (the `RoomGroup` and `VariationThumbnail` paragraphs),
"Queue persistence and identity", "Dispatch ordering", and
"Strategy used for queued retries" for the full contract.

## Acceptance criteria

- [ ] New hook `frontend/hooks/useRetryQueue.ts` exposes:
      - `enqueue(variationId)` returning `'dispatched' | 'queued' | 'deduped'`.
      - `clear()` to empty the queue and prevent pending dispatches.
      - `queuedIds: ReadonlySet<string>` for read-only UI consumption.
- [ ] Hook is agnostic about WHAT it dispatches — it accepts an
      `onDispatch(room, variationIndex, strategy)` callback from its
      consumer; future call sites do not require hook changes.
- [ ] Hook accepts `project`, `isGenerating`, and
      `regeneratingVariationId` as inputs, and uses an internal
      effect to detect the idle transition that triggers draining.
- [ ] Drain order is strictly serial: pop one, wait for
      `regeneratingVariationId` to clear, then pop the next.
- [ ] Drop rule on pop: if the variation no longer exists in the
      current project OR its status is no longer `'failed'`, drop
      silently from the hook's perspective (consumer's `onDispatch`
      is NOT called) and consider the next entry. The hook gives the
      consumer enough information (e.g., return value, callback, or
      separate `onDrop` hook) to write the activity-log entry.
- [ ] Queue lives entirely in React state on the project detail
      page. No `localStorage`, no `sessionStorage`, no server
      persistence. Page unmount discards the queue (this is implicit
      from React state but should be verified by test).
- [ ] Variation identity is by `variation.id`, robust to project
      reloads triggered by `debouncedReload` between SSE events.
- [ ] Project detail page wiring:
      - `handleRetryVariation` is routed through `enqueue` instead
        of calling `handleRegenerateVariation` directly.
      - On `'dispatched'` outcome, behavior matches today (no extra
        toast — the existing flow takes over).
      - On `'queued'` outcome, fire
        `toast.info("Retry queued — will run when generation completes")`
        and add an info-level activity-log entry following the
        existing copy/icon conventions in `app/projects/[id]/page.tsx`.
      - On `'deduped'` outcome, no toast and no log — silent.
      - On a drain-time drop (drop rule), add an info-level
        activity-log entry; no toast.
- [ ] `RoomGroup` accepts a new optional
      `queuedVariationIds?: Set<string>` prop and passes a derived
      `isQueued?: boolean` flag down to each `VariationThumbnail`
      computed per-variation as `queuedVariationIds.has(variation.id)`.
- [ ] `VariationThumbnail` accepts a new optional
      `isQueued?: boolean` prop. In the failed branch, when
      `isQueued` is true, the Retry button is replaced by a Queued
      indicator: a `Loader2` spinner + "Queued" badge + a small
      descriptive caption ("Will retry when generation finishes"
      or similar). The Queued visual is deliberately distinct from
      both the processing-state spinner and the failed-state error
      tile.
- [ ] Page computes `queuedVariationIds: Set<string>` from the
      hook's `queuedIds` and passes it into `RoomGroup`.
- [ ] Queued retries always dispatch with `strategy: 'fresh'`,
      matching the current behavior of `handleRetryVariation`.
- [ ] Hook unit tests at
      `frontend/hooks/__tests__/useRetryQueue.test.ts` (or
      co-located) using React Testing Library's `renderHook`
      against jsdom — establishing the pattern since the codebase
      does not currently have a hooks test directory. Tests cover:
      - Enqueue when idle dispatches immediately (returns
        `'dispatched'`, invokes `onDispatch` synchronously).
      - Enqueue while busy queues and returns `'queued'`;
        subsequent enqueues for the same id return `'deduped'` and
        do not change the queue.
      - When `regeneratingVariationId` and `isGenerating` both go
        idle and the queue is non-empty, the next entry is
        dispatched.
      - `clear()` empties the queue and prevents pending dispatches.
      - Drop rule: if the project state at drain time has no
        failed variation matching the queued id, that entry is
        dropped (no `onDispatch` call) and the next entry is
        considered.
- [ ] New Playwright E2E spec
      `frontend/tests/e2e/retry-queue-during-generation.spec.ts`
      covers two of the four PRD scenarios in this slice:
      - **Queue happy path**: mock the global generation SSE to
        emit four `variation_completed` and one `variation_failed`
        mid-stream → click Retry on the failed variation → assert
        the Queued indicator is visible and the Retry button is no
        longer there → emit `project_completed` → assert the
        variation regen POST fires once → assert
        `variation_completed` updates the thumbnail.
      - **Dedup on multi-click**: click Retry three times in rapid
        succession on the same failed variation while in-flight →
        assert only one `toast.info` lands and only one variation
        regen POST fires after the global stream completes.
- [ ] The new spec follows the SSE-mocking pattern from
      `frontend/tests/e2e/regen-failure-preserves-prior-image.spec.ts`
      and the toast/activity-log assertion pattern from
      `frontend/tests/e2e/retry-fallback-toast.spec.ts` and
      `frontend/tests/e2e/activity-log-copy.spec.ts`.
- [ ] No backend changes. No changes to the
      `streamVariationRegeneration` API service contract or to
      `handleRegenerateVariation` itself beyond the route-through.
- [ ] `clear()` is exposed but NOT yet called from
      `handleRegenerateRoom`, `startGeneration`,
      `handleRegenerateAll`, or the stream `'error'` handler — those
      call sites are added by issues 003 and 004.
- [ ] Local checks pass before commit:
      `uv run pytest tests/ --ignore=tests/integration -v`,
      `cd frontend && npx playwright test` (full E2E suite),
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 3
- User story 4
- User story 5
- User story 6
- User story 10
- User story 13
- User story 14
- User story 15
