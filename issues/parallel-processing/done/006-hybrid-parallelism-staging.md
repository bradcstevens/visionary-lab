## Parent PRD

`prds/2026-04-29-parallel-processing-prd.md`

## What to build

Make the staging pipeline genuinely parallel: rooms run concurrently *and*
variations within each room run concurrently, all sharing the global
image-call cap from slice 4 and the per-project lock from slice 5. This is
the user-visible payoff of the PRD — a 5×5 project completes in minutes
instead of tens of minutes, and a 1×5 project gets the same parallel
speedup that today only multi-room projects can get (and only barely).

End-to-end:

- In `StagingPipeline.process_room`, fan variations out concurrently using
  `asyncio.gather` (or equivalent) instead of awaiting them in sequence.
  Each variation call still flows through `ImagePipelineService` and the
  global image-call semaphore (slice 4), so the rate-limit cap is enforced
  uniformly. Each variation's status write still goes through the
  per-project lock (slice 5), so interleaved completions can't clobber
  project state.
- `process_single_variation` no longer acquires the room-level
  `STAGING_CONCURRENT_ROOMS` semaphore. It is a single image call; only
  the global image-call cap should apply. (The room-level semaphore
  continues to gate full *room* workers, which hold base64 originals and
  SSE generators in memory.)
- Bump the `STAGING_CONCURRENT_ROOMS` default from 3 to 10 in
  `backend/core/config.py`. This cap is now purely a memory bound, not a
  rate-limit bound, so the higher default is appropriate.
- Within-room SSE event ordering is no longer guaranteed. The SSE event
  *types* and *payloads* are unchanged; the existing `debouncedReload()`
  on the frontend already refetches the source-of-truth project document,
  so interleaved arrival is fine. Confirm explicitly that no backend code
  asserts intra-room ordering.
- Cancellation: when the SSE client disconnects, in-flight variation tasks
  must be cancelled cleanly with no zombie writes to Cosmos (no completion
  event for a cancelled variation, no half-written status). Ensure the
  fan-out uses cancellation-aware patterns (e.g. `asyncio.gather` with
  `return_exceptions=False` plus a guard, or a `TaskGroup`-equivalent).

Tests verify externally-observable behavior at the public seams
(`image_pipeline.process_pipeline` and `staging_storage.update_project`
mocked at their boundaries, per the PRD's testing decisions):

- A 1×5 project (one room, five variations) runs all five image calls
  concurrently subject to the global cap. Use `asyncio.Event`-gated mocks
  so the concurrency observation is deterministic.
- A 25-room project bounds room workers to `STAGING_CONCURRENT_ROOMS` (the
  new default of 10), and within each room, variations still fan out
  subject to the global cap.
- An SSE client disconnect mid-job cancels all in-flight variation tasks
  cleanly, with no further `update_project` calls for cancelled variations
  after the cancellation point.
- The frontend Playwright E2E suite
  (`frontend/tests/e2e/project-generation.spec.ts`) continues to pass
  unchanged — interleaved event arrival is already exercised there.

See the parent PRD's *Hybrid parallelism* and *Room-level cap* sections,
and *Testing Decisions → Staging pipeline hybrid parallelism*.

## Acceptance criteria

- [ ] `StagingPipeline.process_room` runs variations concurrently; each
      variation call still flows through `ImagePipelineService` and the
      global image-call semaphore.
- [ ] Variation status writes go through the per-project lock from slice
      5 — no last-writer-wins loss when two variations of the same room
      complete near-simultaneously.
- [ ] `process_single_variation` does not acquire the room-level
      `STAGING_CONCURRENT_ROOMS` semaphore.
- [ ] `STAGING_CONCURRENT_ROOMS` default is `10` in `backend/core/config.py`.
- [ ] No backend code asserts intra-room SSE event ordering. SSE event
      types and payloads are unchanged.
- [ ] Client disconnect cancels in-flight variation tasks cleanly with no
      zombie writes to Cosmos and no further completion events for the
      cancelled variations.
- [ ] New tests cover: 1×5 runs all five image calls concurrently subject
      to the global cap; 25-room project bounds room workers to
      `STAGING_CONCURRENT_ROOMS`; client disconnect cancels in-flight
      tasks cleanly with no zombie writes.
- [ ] `tests/test_parallel_rooms.py` and `tests/test_staging_pipeline.py`
      are updated for the new hybrid behavior and continue to pass.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` passes.
- [ ] `cd frontend && npx playwright test` passes — including
      `frontend/tests/e2e/project-generation.spec.ts`, unchanged.
- [ ] `cd frontend && npm run build` and `npx next lint` both pass.

## Blocked by

- Blocked by `004-global-image-semaphore.md` — variation fan-out relies
  on the global image-call cap to keep rate-limit exposure bounded.
- Blocked by `005-per-project-cosmos-lock.md` — variation fan-out relies
  on the per-project lock to keep concurrent project-state writes safe.

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 3
- User story 4
- User story 13
- User story 15
- User story 22 (the hybrid-parallelism tests above are the second cluster
  of deterministic-without-Azure tests this story calls for)
