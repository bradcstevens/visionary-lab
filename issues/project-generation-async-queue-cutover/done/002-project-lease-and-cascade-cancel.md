## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

Two pure helper utilities that the project dispatcher and the new POST
endpoint will consume in later slices: a distributed per-project lease
implemented via Cosmos ETag CAS on the project document, and a
cascade-cancel helper that aborts in-flight `regenerate_variation` jobs
for a given project before a `regenerate_all=true` reset.

The pre-existing in-process `_PROJECT_LOCKS` dictionary in
`backend/core/staging_pipeline.py` is **not sufficient** because KEDA
can scale workers to 2+ replicas; the distributed lease replaces it
for project-scoped serialization. See PRD sections "Cross-replica
safety" and "`regenerate_all=true` semantics".

End-to-end behaviour:

- Project document gains a `current_project_job_id` field used as the
  lease holder.
- A lease-acquire helper attempts an ETag-conditional update of the
  project doc setting `current_project_job_id = self.job_id`. If a
  different non-terminal job already holds the lease, the helper
  returns a "lease busy" signal so the caller (issue 005) can abandon
  the message for redelivery. If unset / our id / terminal owner, the
  CAS update succeeds.
- A lease-release helper clears `current_project_job_id` (also via
  ETag-conditional update) and tolerates the project doc having been
  mutated underneath us during the run.
- A cascade-cancel helper enumerates non-terminal
  `regenerate_variation` jobs for a project and marks them cancelled
  via `JobStore`. Cancellations propagate through the existing
  `is_cancelled()` polling path; this helper does NOT need to wait for
  the worker side to finish reverting state.
- Helpers live in a place both the dispatcher (issue 005) and the
  endpoint (issue 006) can import (e.g.
  `backend/core/project_lease.py`).

This slice ships only the helpers and their unit tests. No worker
wiring, no endpoint, no UI. The helpers are the foundation that issues
005 and 006 build on.

## Acceptance criteria

- [ ] Lease-acquire helper succeeds when `current_project_job_id` is
      unset, equal to the caller's job id, or owned by a terminal
      job; it returns a "lease busy" signal when a different
      non-terminal job holds it.
- [ ] Lease-acquire uses Cosmos ETag CAS on the project document; a
      stale ETag (someone else just won the race) results in "lease
      busy", not an exception bubbling out.
- [ ] Lease-release clears `current_project_job_id` only if it still
      points at the caller's job id; a foreign owner is left
      untouched.
- [ ] Cascade-cancel helper finds all non-terminal
      `regenerate_variation` jobs for a project and marks them
      cancelled via `JobStore`; jobs in terminal status are skipped.
- [ ] Cascade-cancel is safe to call when there are zero in-flight
      variation jobs (no error, no network calls beyond the query).
- [ ] New unit tests cover: lease busy on foreign non-terminal owner;
      lease re-acquired when previous owner is terminal; concurrent
      acquire — only one of two callers wins the CAS and the other
      gets "lease busy"; release no-ops on foreign owner; cascade-
      cancel touches only `regenerate_variation` kinds, only
      non-terminal statuses.
- [ ] No call to the helpers from production code yet (issue 005/006
      consume them).
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` is green.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 22
- User story 23
- User story 24
- User story 25
