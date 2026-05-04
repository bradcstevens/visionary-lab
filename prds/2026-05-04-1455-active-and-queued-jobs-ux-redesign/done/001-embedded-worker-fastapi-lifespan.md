# Embedded worker via FastAPI lifespan

## Parent PRD

`prd.md`

## What to build

A vertical slice that makes a worker always present in development without
requiring the user to start a second process, while keeping production
unaffected.

End-to-end behavior: when a developer runs `uv run fastapi dev`, the API
process auto-spawns an embedded worker that drains the queue. When the
production API container starts (`ROLE=worker` is unset and
`AUTO_START_WORKER=False`), no embedded worker is spawned. Stale jobs from
prior failed runs drain naturally on the next embedded-worker boot.

See "Slice 1 — Embedded worker via FastAPI lifespan" in `prd.md` for the
full design, including the new `embedded_worker` deep module, the
extracted shared `build_worker()` factory, the `[embedded-worker]` log
prefix contract, and the Bicep production override.

## Acceptance criteria

- [ ] New deep module `embedded_worker` owns the auto-start policy
  decision, asyncio task handle, log prefix, and shutdown sequence.
- [ ] FastAPI lifespan handler delegates to `embedded_worker` on startup
  and shutdown.
- [ ] `should_auto_start(role, env_flag)` policy: starts only when
  `AUTO_START_WORKER` is true and `ROLE != "worker"`.
- [ ] `AUTO_START_WORKER` defaults to **on in development**, **off in
  production**.
- [ ] Existing `build_worker()` factory in the standalone worker entry
  point is extracted into a shared module consumed by both code paths.
- [ ] Bicep API container module sets `AUTO_START_WORKER=False`
  explicitly so a deployment misconfiguration cannot accidentally start
  a second worker instance.
- [ ] Embedded worker log lines are prefixed with `[embedded-worker]`.
- [ ] Pre-existing stuck jobs from before this change drain naturally on
  the first embedded-worker boot (no manual cleanup script required).
- [ ] Unit tests: lifespan integration test with a mocked worker
  (start/stop ordering); table-driven `should_auto_start` policy test
  over `(role, env_flag)` pairs; `[embedded-worker]` log-prefix
  assertion on emitted records.
- [ ] Playwright test: full project generation completes end-to-end in
  dev with only the API process running.
- [ ] All checks pass locally: `uv run pytest tests/
  --ignore=tests/integration -v`, `cd frontend && npm run build`, `cd
  frontend && npx next lint`, `cd frontend && npx playwright test`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 5 (project generation runs end-to-end in `uv run fastapi
  dev` without a second process)
- User story 6 (Playwright tests don't need a coordinator script)
- User story 7 (production auto-spawn off by default)
- User story 8 (`[embedded-worker]` log prefix for triage)
- User story 24 (stale jobs drain naturally on next worker boot)
