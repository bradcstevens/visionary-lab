## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`JobWorker` — the long-running consumer that runs in the worker Container
App replicas. Pulls messages from `JobQueue`, fetches the job from
`JobStore`, dispatches to the existing `ImagePipelineService`, and writes
status, phase, and progress back. Honors `cancel_requested` between
external calls. Caps per-replica concurrency at the existing
`IMAGE_GEN_MAX_CONCURRENT` semaphore (acquired via `call_with_retry`).
On exception, abandons the message so the queue's max-dequeue policy
escalates to poison after 3 attempts. Emits structured log events
`job.enqueued`, `job.started`, `job.progress`, `job.succeeded`,
`job.failed`.

See PRD sections "JobWorker" and "Further Notes".

## Acceptance criteria

- [ ] `JobWorker` consumes from `JobQueue`, executes via `ImagePipelineService`, and persists state via `JobStore`
- [ ] Worker honors `cancel_requested` before each external call and transitions the job to `cancelled`
- [ ] Worker respects `IMAGE_GEN_MAX_CONCURRENT` per replica
- [ ] Failed runs increment `attempts`; the 3rd failure leaves the message in `imagejobs-poison` and the job in `failed` with structured `error`
- [ ] Structured log events emitted at the documented points; example KQL queries added to `DEPLOYMENT.md`
- [ ] Integration test: enqueue → kill worker → restart → job resumes via visibility-timeout re-delivery
- [ ] Unit tests cover happy path, retry-then-poison, cancel-honored

## Blocked by

- Blocked by `002-jobstore-jobqueue-modules.md`

## User stories addressed

- User story 15
- User story 19
- User story 20
