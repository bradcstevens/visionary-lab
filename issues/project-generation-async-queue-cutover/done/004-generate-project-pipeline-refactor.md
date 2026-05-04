## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

Refactor `staging_pipeline.generate_project` from an async-generator
that yields events to a streaming consumer into a function suitable
for queue execution. The queue-friendly form takes a
`progress_callback` (invoked where the generator used to `yield`) and
an `is_cancelled()` poll, and reuses precomputed brief prompts when
the job payload supplies them.

See PRD sections "Worker dispatcher" and "Brief reuse on retry". This
slice does NOT remove the legacy async-generator surface — it stays
in the codebase so existing tests and the legacy `POST /generate`
endpoint keep working until follow-up cleanup.

End-to-end behaviour:

- New `generate_project_for_job(project, brief_prompts,
  progress_callback, is_cancelled)` lives in
  `backend/core/staging_pipeline.py` (or a sibling module; same
  package).
- Same room-iteration and variation-iteration logic as today's
  `generate_project()`, but every place the async-generator currently
  `yield`s an event becomes a synchronous call to
  `progress_callback({...})` with the same event payload shape.
- When `brief_prompts` is non-None (i.e., the job payload carried a
  precomputed compose-result from the POST handler), the function
  uses it directly. `brief_to_prompts()` is NOT invoked. This is the
  "brief reuse on retry" contract — pinned by an explicit unit test.
- `is_cancelled()` is polled between rooms and between variations
  within a room; on True, the function raises a project-scoped
  cancel signal (e.g. `JobCancelled`). Variation-state mutation on
  cancel is the dispatcher's job (issue 005) — this function only
  signals.
- The legacy `generate_project()` async-generator surface continues
  to exist and behave as it does today; refactor it to delegate to
  `generate_project_for_job` if that's clean, otherwise keep the two
  side-by-side. Either choice is acceptable so long as both surfaces
  remain green against existing tests.

This slice produces no user-visible delivery on its own — it is the
function shape issue 005 will dispatch into.

## Acceptance criteria

- [ ] `generate_project_for_job(project, brief_prompts,
      progress_callback, is_cancelled)` exists with the signature
      above.
- [ ] `progress_callback` is invoked at the same logical points the
      async-generator yields today (per-room start/end,
      per-variation start/end, terminal events) with equivalent
      payloads.
- [ ] When `brief_prompts` is provided, `brief_to_prompts()` is not
      called; the function uses the prompts directly.
- [ ] `is_cancelled()` is polled between rooms and between variations;
      the function raises a cancel signal (e.g. `JobCancelled`) when
      it returns True.
- [ ] The legacy `generate_project()` async-generator surface
      continues to pass its existing tests.
- [ ] New unit tests in `tests/test_staging_pipeline_for_job.py`
      cover: progress callback cadence; smart-skip filters
      PENDING/FAILED rooms by default; brief reuse on retry (the
      regression pin for rubber-duck non-blocking #2);
      `is_cancelled()` polled at the documented points.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` is green.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 28 (enables the worker dispatcher in 005)

(No direct user-visible delivery on its own; this is scaffolding for
issue 005.)
