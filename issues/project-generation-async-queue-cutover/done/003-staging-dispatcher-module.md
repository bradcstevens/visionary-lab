## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

Introduce a new `staging_dispatcher` module that owns the kind-switch
between job kinds, and port the existing `regenerate_variation`
dispatcher into it so both kinds share the same `JobWorker` glue. This
is a pure refactor with no behavior change — the existing
single-variation regen path must continue to work identically — but it
is the structural prerequisite for adding `kind="generate_project"` in
issue 005 and for wiring the production worker entrypoint in issue 007.

See PRD section "Worker dispatcher" for the kind-switch shape:

```python
async def staging_dispatcher(job: dict, is_cancelled) -> dict:
    kind = job.get("kind")
    if kind == "regenerate_variation":
        return await regenerate_variation_dispatcher(job, is_cancelled)
    if kind == "generate_project":
        return await generate_project_dispatcher(job, is_cancelled)
    raise ValueError(f"Unknown kind: {kind}")
```

In this slice, `generate_project` raises `ValueError("Unknown kind")`
because the project dispatcher does not yet exist; issue 005 fills in
that branch.

End-to-end behaviour:

- New module (e.g. `backend/core/dispatchers.py` or
  `backend/core/staging_dispatcher.py`) exports `staging_dispatcher`.
- The existing `regenerate_variation` dispatcher logic is moved into
  this module under a new `regenerate_variation_dispatcher` function,
  with its previous home reduced to a thin re-export shim (or removed
  entirely if no external callers reference it).
- Existing tests for `regenerate_variation` pass without changes — if
  any test imports the dispatcher by symbol, the import path is
  updated in the same change.
- `JobWorker` consumers in tests and any worker-bootstrap site can
  pass `staging_dispatcher` in place of `regenerate_variation_*`.

## Acceptance criteria

- [ ] New module exposes a `staging_dispatcher(job, is_cancelled)`
      callable with the kind-switch shape above.
- [ ] `regenerate_variation` logic is fully moved (not duplicated)
      into the new module; the previous location is either a
      re-export or removed.
- [ ] `kind == "generate_project"` raises `ValueError("Unknown
      kind: generate_project")` (or routes through the same unknown-
      kind branch); a unit test pins this so issue 005 visibly
      replaces the placeholder.
- [ ] All existing `tests/test_job_worker.py`,
      `tests/test_job_worker_progress.py`, and any
      `regenerate_variation` dispatcher tests pass with no logic
      changes — only import-path updates if needed.
- [ ] New unit test in `tests/test_staging_dispatcher.py` covers the
      kind-switch routing for `regenerate_variation`, the unknown-
      kind error, and the placeholder error for `generate_project`.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` is green.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 28 (production worker exists — partial; this enables 007)

(No direct user-visible delivery on its own; this is scaffolding for
issues 005 and 007.)
