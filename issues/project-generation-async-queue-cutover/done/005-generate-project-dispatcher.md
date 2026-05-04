## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

Wire the `kind="generate_project"` branch of `staging_dispatcher` to a
new `generate_project_dispatcher` that owns the cross-replica safe,
cancel-aware, regenerate-all-aware execution of a project-level job.
This is the slice that makes the async-queue path actually deliver a
generated project from a queued job.

See PRD sections "Worker dispatcher (project-kind branch)" and
"`regenerate_all=true` semantics".

End-to-end behaviour:

- `generate_project_dispatcher(job, is_cancelled)` lives in the
  staging dispatcher module from issue 003. The kind-switch's
  placeholder for `generate_project` is replaced with a real call to
  this function.
- **Acquire distributed per-project lease** via the helper from issue
  002. If a different non-terminal job already holds the lease, the
  dispatcher abandons the message (returns without `complete()`) so
  Storage Queue redelivers it after the visibility timeout. If
  unset / our id / terminal owner, claim it and proceed.
- If `payload.regenerate_all` is true: reset every variation to
  PENDING, **clear `image_url`, `thumb_url`, `md_url` on each
  variation**, and **schedule deletion of those derivative blobs**
  (mirrors `_schedule_blob_cleanup` from `staging_pipeline.py`,
  applied in bulk). Plain in-place reset would leak old blobs.
- **Brief reuse on retry**: if `payload.brief_prompts` is present,
  pass it into `generate_project_for_job` from issue 004. Don't
  recompose.
- Run `generate_project_for_job(project, brief_prompts,
  progress_callback, is_cancelled)`. The progress callback writes to
  `JobStore` via `update_job(progress=..., phase=...)` so events
  propagate through the change feed and reach all subscribed pages
  via `/jobs/stream`.
- **Cancel mid-flight**: when `is_cancelled()` returns True (or the
  pipeline raises `JobCancelled`), revert variations in PROCESSING
  to PENDING, **preserve COMPLETED variations**, persist the
  project, and surface the cancel via the job's terminal status. Do
  NOT mark anything as FAILED.
- **Cancel-during-image-edit edge**: if cancel arrives mid-
  `image_pipeline.run()`, the variation may transition to COMPLETED
  before the next `is_cancelled()` poll. That variation is
  preserved as completed (matches the
  "preserve_completed_revert_in_flight" decision in the PRD).
- **Smart-skip**: with `regenerate_all=false`, rooms whose
  variations are all already COMPLETED are skipped without
  re-rendering. PENDING/FAILED rooms are processed.
- **Release the project lease in `finally`** using the helper from
  issue 002.

## Acceptance criteria

- [ ] The `generate_project` branch of `staging_dispatcher` calls
      `generate_project_dispatcher`; the placeholder error from issue
      003 is gone.
- [ ] On lease busy (a different non-terminal job already holds
      `current_project_job_id`), the dispatcher returns without
      calling `complete()` so Storage Queue redelivers.
- [ ] `regenerate_all=true` resets variations to PENDING, clears
      `image_url`/`thumb_url`/`md_url`, and schedules deletion of the
      previously-pointed blobs.
- [ ] `payload.brief_prompts` is passed through to
      `generate_project_for_job` and `brief_to_prompts()` is NOT
      called inside the dispatcher.
- [ ] Progress callback writes to `JobStore.update_job(progress=...,
      phase=...)`.
- [ ] Cancel mid-flight reverts PROCESSING variations to PENDING,
      preserves COMPLETED variations, never marks anything FAILED.
- [ ] A variation that completes between `is_cancelled()` polls is
      preserved as COMPLETED (cancel-during-image-edit edge).
- [ ] `regenerate_all=false` smart-skips rooms whose variations are
      all COMPLETED; PENDING/FAILED rooms are processed.
- [ ] Lease is released in `finally`.
- [ ] New tests in `tests/test_staging_dispatcher.py` cover: kind-
      switch routing to the project dispatcher; lease-busy
      abandonment (rubber-duck blocking #3); cancel preserves
      completed and reverts processing to PENDING; cancel-during-
      image-edit edge (rubber-duck non-blocking #4); regenerate_all
      clears blob URLs and schedules blob deletion.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` is green.

## Blocked by

- Blocked by `002-project-lease-and-cascade-cancel.md`
- Blocked by `003-staging-dispatcher-module.md`
- Blocked by `004-generate-project-pipeline-refactor.md`

## User stories addressed

Reference by number from the parent PRD:

- User story 14
- User story 15
- User story 16
- User story 17
- User story 20
- User story 22
- User story 23
- User story 31 (regression coverage for cancel + regenerate-all)
