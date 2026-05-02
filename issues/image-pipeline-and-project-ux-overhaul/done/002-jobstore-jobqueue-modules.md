## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

The two deep persistence modules backing the queue. `JobStore` wraps the
Cosmos `jobs` container (partition key `/project_id`, deterministic id
`{project_id}:{room_id}:{variation_id}:{revision}`, idempotent insert via
`If-None-Match: *`, status/progress updates, project-scoped queries,
change-feed subscription helper). `JobQueue` wraps Azure Storage Queue
(`imagejobs` + `imagejobs-poison`, visibility timeout 90s, message TTL
7 days, max dequeue 3, managed-identity auth).

Both modules are testable in isolation against the Cosmos emulator and
Azurite. No HTTP, no pipeline coupling.

See PRD sections "JobStore", "JobQueue", "Schema (Cosmos `jobs`
container)", and "Modules to test (unit)".

## Acceptance criteria

- [ ] `JobStore` exposes create/update/get/list-by-project/subscribe-change-feed; deterministic-id insert is idempotent
- [ ] `JobQueue` exposes enqueue/dequeue/complete/abandon and routes a 3rd-failure message to `imagejobs-poison`
- [ ] Cosmos doc shape matches PRD schema (id, project_id, room_id, variation_id, revision, kind, status, progress, phase, attempts, payload, result, error, created_at, updated_at)
- [ ] Unit tests pass against the Cosmos emulator and Azurite covering: idempotent insert, state transitions, change-feed delivery, partition-scoped query, enqueue/dequeue round-trip, TTL, poison-on-3rd-failure
- [ ] No connection strings in code or config — managed identity only
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` passes

## Blocked by

- Blocked by `001-queue-infra-bicep.md`

## User stories addressed

- User story 19
- User story 21
- User story 40
