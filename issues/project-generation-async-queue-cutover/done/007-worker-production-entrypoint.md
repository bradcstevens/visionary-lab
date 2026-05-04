## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

Give the worker a real production home so enqueued jobs are actually
processed in production, not parked forever on the queue. This slice
replaces import-time env-var branching with an explicit pure helper, a
dedicated bootstrap module, and a containerApp entrypoint pointed at
that bootstrap.

See PRD section "Production worker entrypoint".

End-to-end behaviour:

- New `backend/runtime.py` exposes a pure helper:
  ```python
  def choose_runtime(env: Mapping[str, str]) -> Literal["api", "worker"]:
      return "worker" if env.get("ROLE") == "worker" else "api"
  ```
  No import-time side effects; trivially testable.
- New `backend/worker_main.py` is the worker bootstrap. It constructs
  `JobStore`, `JobQueue`, `ProgressEstimator`, the `staging_dispatcher`
  from issue 003, and wires them into a `JobWorker` (which already
  carries the heartbeat from issue 001). It then `await worker.run()`.
- The worker container runs `python -m backend.worker_main` directly;
  the API container keeps `uvicorn backend.main:app`. Both run from the
  same image. **No more import-time `if env=="worker"` branching** —
  different processes for different roles.
- `infra/modules/containerAppWorker.bicep` (or whichever existing
  module already provisions the worker container with KEDA scaling)
  is updated to invoke the new entrypoint with `ROLE=worker`.
- Existing API behaviour is unchanged; the API container does not
  import the worker bootstrap.

## Acceptance criteria

- [ ] `backend/runtime.py` exposes `choose_runtime(env)` that returns
      `"worker"` when `env["ROLE"] == "worker"` and `"api"`
      otherwise; the helper has no import-time side effects.
- [ ] `backend/worker_main.py` constructs `JobStore`, `JobQueue`,
      `ProgressEstimator`, the staging dispatcher, and `JobWorker`,
      then runs the worker.
- [ ] The bicep module that provisions the worker container points
      at `python -m backend.worker_main` and sets `ROLE=worker`.
- [ ] The API container's entrypoint is unchanged (`uvicorn
      backend.main:app`).
- [ ] Both containers run from the same image; no environment-driven
      branching at import time.
- [ ] New `tests/test_runtime.py` covers `choose_runtime`: worker
      role, default api role, missing env var, empty env var.
- [ ] New `tests/test_worker_main.py` mocks the JobWorker / queue /
      store constructors and asserts the bootstrap wires the worker
      with the staging dispatcher (kind-switch + heartbeat already
      live there from issues 001 and 003).
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` is green.
- [ ] (Manual / CI verification on deploy) An enqueued
      `kind="generate_project"` job is picked up and processed by the
      worker container, not parked on the queue.

## Blocked by

- Blocked by `001-jobqueue-heartbeat-and-pop-receipt.md`
- Blocked by `003-staging-dispatcher-module.md`

## User stories addressed

Reference by number from the parent PRD:

- User story 6 (refresh-resume requires the worker to keep running
  after the page disconnects)
- User story 7 (close-tab-and-come-back requires a durable worker)
- User story 24 (cross-replica safety presumes a real worker
  fleet)
- User story 25 (at-most-one project job physically running with
  multiple workers — paired with the lease from 002)
- User story 28 (the production worker process actually exists)
