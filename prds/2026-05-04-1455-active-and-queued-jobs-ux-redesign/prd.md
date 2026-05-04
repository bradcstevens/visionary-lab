# Active and Queued Jobs UX Redesign

## Problem Statement

A user opens an existing project and clicks **Generate** in the header. For
30 to 90 seconds nothing visible happens. Reasonably concluding that the
click was missed, the user clicks again. The front-end finally catches up
and shows two active jobs with two queued, none of them running. A few
minutes later the project flips to `failed`. The activity log says only
"Starting generation". There is no further information about what is
happening, what went wrong, or what to do next.

The system has produced six distinct rough edges in this single sequence:

1. **The Generate button gives no feedback during the producer-side
   blocking window.** The producer endpoint composes the design brief
   inline (a 30 to 90 second LLM call) before returning the new job id.
   While that runs, the front-end shows nothing — no spinner attached to
   the button, no banner, no optimistic tile in the room grid. The user
   cannot tell whether the click registered.

2. **Repeated clicks create duplicate jobs.** The producer mints a fresh
   `revision = uuid.uuid4().hex` on every request, so a click during the
   blocking window produces a second `generate_project` job for the same
   project. Two intents become two jobs, both waiting in the queue.

3. **Nothing consumes the queue in local development.** `uv run fastapi
   dev` only starts the API container; the worker is a separate
   `python -m backend.worker_main` process the user has to remember to
   start. When the worker is not running, queued jobs sit forever — the
   counter "0 running" is mathematically correct but practically silent.

4. **The project status flips to `failed` based on wall-clock staleness.**
   `staging_reconcile.reconcile_project` runs on every project read. After
   the configured staleness window, if room statuses are mixed in any way
   that is not strictly "all pending" or "all completed", the project is
   marked `failed`. This decision is made without consulting the jobs
   container, so a project whose work is genuinely queued and waiting is
   indistinguishable from a project that actually failed.

5. **The activity log does not bridge job state.** SSE `event: job` deltas
   are consumed and update the in-memory `jobs` slice, but they never
   become activity-log entries. The user sees one log entry from the
   initial click ("Starting generation") and nothing else, even when the
   worker is making progress and emitting phase transitions.

6. **The header counters do not explain themselves.** "2 active, 2 queued,
   0 running" is shown without a subline that says what the user should
   do, why the count is what it is, or whether the system is healthy.

7. **Errors surface as raw exception strings.** When the producer fails
   to enqueue (the bug-report case was an Azure Storage
   `AuthorizationPermissionMismatch` 502), the front-end renders the
   verbatim exception message in a banner. The user has no actionable
   guidance: should they retry, fix something, contact an administrator?

The throughline: the user has no continuous visibility from click to
completion, the system creates duplicate work when the user reasonably
retries, the project status lies, and when something does go wrong the
error message is for developers, not for the person trying to design a
room.

## Solution

A redesign of the project-generation request lifecycle that gives the
user continuous, accurate visibility from the moment they click Generate
through completion or recovery, and that prevents the system from
producing duplicates, lying about status, or hiding what went wrong.

The new behavior is built on five vertical slices:

1. **A worker is always present in development.** The FastAPI lifespan
   handler auto-spawns an embedded worker as an `asyncio` task when the
   `AUTO_START_WORKER` environment variable is set, defaulting on in
   development and off in production. The user never has to remember to
   start a separate worker process. Production is unaffected because the
   worker container is identified by `ROLE=worker` and explicitly skips
   the auto-spawn path.

2. **A second click never creates a duplicate job.** The producer is
   single-flight at two layers. An `Idempotency-Key` HTTP header drives
   deterministic job-id construction so a retried request collapses into
   the existing job. A pre-acquire of `current_project_job_id` (the
   project-scope lease primitive that already exists) ensures a click
   that arrives while another generation is in flight returns the
   existing job's id rather than starting a new one. The producer
   responds with `200 { already_in_flight: true }` for both dedupe paths
   and `202 { job_id }` for new work, so the front-end can branch on the
   distinction.

3. **The project status is derived from the jobs container, not from a
   wall clock.** `reconcile_project` is split into two functions: one
   that performs orphan variation cleanup (preserving today's behavior
   with its existing staleness gate), and one that derives the canonical
   project status by reading the active job (if any) from the jobs
   container, falling back to a room-derived status when no active job
   exists. The buggy "rooms are mixed, therefore project failed" branch
   is removed entirely — project failure is a hard claim made only by
   the worker (on dispatch failure or poison-queue exhaustion), the
   cancellation cascade, or a producer-side hard error.

4. **The user can see what is happening, and recover when it gets
   stuck.** A staleness detector in the front-end watches every job in
   the project. A "pickup detector" tracks how long a `pending` job has
   been waiting for a worker; a "stalled detector" tracks how long a
   `running` job has been silent. At 45 seconds either detector adds a
   soft warning subline; at 120 seconds it escalates to a hard warning
   with a one-tap "Cancel queued jobs" button that cascades cancel
   across every non-terminal job for the project. The header counters
   stay; a dynamic explanatory subline appears beneath them.

5. **The user gets continuous visibility and friendly errors.** The
   moment the Generate click registers, a banner appears with phased
   copy ("Composing design brief…" for the first 14 seconds, then
   "Submitting to queue…"). When the producer returns 202, an optimistic
   tile renders immediately so the user sees their work in the grid
   before the SSE seed catches up. When the producer returns 200
   (already in flight), the banner is suppressed in favor of a small
   toast plus a scroll-to-existing-tile. The activity log is derived
   from SSE phase transitions, gaining a line for every meaningful
   phase change without spamming on progress-percentage ticks. Errors
   carry a small `error_kind` enum that the front-end maps to friendly
   user messages, with an expandable "Show technical details" section
   below for diagnostics.

The bug-report scenario, after this redesign: the click immediately
shows a banner with phased copy. The embedded worker (running in
process) picks up the job within seconds and starts emitting phase
events, which the activity log narrates in real time. A second click
during the blocking window is suppressed by either the
idempotency-key dedupe or the lease pre-acquire, depending on timing,
and surfaces as a toast rather than creating a second job. The project
status reflects the actual state of the queued work, not a wall-clock
heuristic. If the worker is genuinely down, the staleness detector
escalates within 45 seconds and offers a recovery action.

## User Stories

1. As a designer working on an existing project, I want immediate visual
   feedback when I click Generate, so that I know my click registered
   and I do not click again expecting nothing happened.

2. As a designer, I want the Generate banner to tell me what the system
   is currently doing ("Composing design brief", "Submitting to queue"),
   so that I have an honest sense of progress through the
   producer-side blocking window.

3. As a designer who clicks Generate twice during the blocking window,
   I want the second click to be silently absorbed rather than create
   a second job, so that I am not stuck explaining to myself why my
   project shows two queued jobs.

4. As a designer who clicks Generate twice, I want the second click to
   produce a small toast that explains "generation already in
   progress", so that I have feedback that the click was received but
   gracefully suppressed.

5. As a designer running locally with `uv run fastapi dev`, I want
   project generation to actually run end-to-end without me starting a
   second process, so that I can iterate on the front-end without
   tripping over a missing worker.

6. As a developer running local Playwright tests, I want the worker to
   start automatically with the API, so that the test suite does not
   need a coordinator script to launch the worker container.

7. As an operator deploying to production, I want the API container's
   embedded-worker auto-spawn to be off by default, so that the API and
   worker remain independently scalable container apps.

8. As an operator inspecting production logs, I want any embedded
   worker (in dev or accidentally-enabled paths) to prefix its log
   lines with `[embedded-worker]`, so that I can distinguish embedded
   from standalone worker output without grepping by container name.

9. As a designer whose project is queued and waiting for a worker, I
   want my project status to remain `processing`, so that I can trust
   the badge to mean my work is still in progress rather than failed.

10. As a designer, I want the project to flip to `failed` only when the
    work has genuinely failed (worker exhausted retries, hard
    producer-side error, explicit cancellation), so that the failure
    status carries real information.

11. As a designer, I want my activity log to update as the worker
    progresses through phases ("Generating room 1", "Composing
    prompts", etc.), so that I can follow what the system is doing
    without staring at a frozen UI.

12. As a designer, I want the activity log to omit fine-grained
    progress-percentage ticks, so that the log is a high-signal record
    of state changes rather than a wall of timestamps.

13. As a designer whose worker has stopped responding, I want a
    visible warning in the header within 45 seconds, so that I know
    something is wrong before I have spent five minutes assuming it is
    still working.

14. As a designer whose worker is genuinely down, I want a clear "Cancel
    queued jobs" button after 120 seconds of staleness, so that I can
    recover without refreshing the page or asking a developer for help.

15. As a designer, I want the cancel-queued-jobs button to cancel
    everything non-terminal for the project in a single click, so that
    I can recover from a stuck state without canceling each job
    individually.

16. As a designer who clicks Cancel, I want immediate visual feedback
    (button disabled, "Cancelling…" copy), so that I know my click was
    received and the system is acting on it.

17. As a designer who clicks Cancel while the worker is offline, I want
    the banner to dismiss within 10 seconds with a toast that explains
    "the worker will pick up the cancellation when it comes online", so
    that I am not stuck waiting on an SSE confirmation that will never
    arrive.

18. As a designer, I want the header to keep showing counters of active
    and queued jobs, so that I retain the at-a-glance summary I rely on
    today.

19. As a designer, I want the header to add a dynamic subline beneath
    the counters, so that I always have a one-line answer to "what is
    happening right now?".

20. As a designer who hits an error during enqueue, I want a
    user-friendly explanation of what went wrong, so that I know
    whether to retry, fix something, or contact an administrator.

21. As a designer who hits a recognized error like an Azure
    permissions issue, I want a specific message that names the cause
    ("Backend isn't authorized to write to the job queue…"), so that I
    or my administrator can act on the right thing.

22. As a developer triaging a production error, I want the error
    banner to include an expandable "Show technical details" section
    with the raw exception, so that I can copy-paste it into a bug
    report or stack trace search.

23. As an operator, I want the worker to write a structured
    `error_kind` field on terminal-failure job documents, so that
    log-aggregation queries can categorize failure modes without
    parsing exception messages.

24. As a designer with two stale jobs from a previous failed click, I
    want the worker to drain those jobs naturally on its next boot,
    so that I do not have to manually clean up stale queue messages.

25. As a designer who refreshes the page mid-generation, I want the
    SSE seed to restore my job tile state, so that my work-in-progress
    is preserved across reloads. (The activity log itself resets — its
    state is in-memory by design.)

26. As a developer writing front-end tests, I want the staleness
    detector to be a pure function of (jobs, lastEventByJobId, now),
    so that I can test it with mocked clocks rather than real timers.

27. As a developer writing front-end tests, I want the activity-log
    derivation to be a pure function of (previous jobs, current jobs),
    so that I can test the diff logic with table-driven inputs.

28. As a developer writing back-end tests, I want the producer's
    dedupe pipeline to be a single function whose return value is a
    discriminated union (`AlreadyInFlight | NewlyEnqueued |
    EnqueueFailed`), so that the HTTP endpoint becomes a thin wrapper
    and the business logic is unit-testable without `httpx`.

29. As a developer adding a new producer-side error category, I want
    the error classifier to be a single pure function with an enum
    return type, so that I add the new category in one place and the
    UI mapping picks it up via the enum.

## Implementation Decisions

Five vertical slices, each end-to-end shippable to production. Each
slice closes one ring of the problem. The bug-report scenario is fully
addressed by the end of Slice 3; Slices 4 and 5 add recovery
affordances and UX polish.

### Slice 1 — Embedded worker via FastAPI lifespan

A new deep module **`embedded_worker`** encapsulates the lifecycle of a
worker that lives inside the API process for development. It owns the
policy decision (auto-start when `AUTO_START_WORKER` is true and the
runtime is not already a dedicated worker container), the asyncio task
handle, the `[embedded-worker]` log prefix, and the clean shutdown
sequence. The FastAPI lifespan handler delegates to this module on
startup and shutdown.

The existing `build_worker()` factory in the standalone worker entry
point is extracted into a shared module so both code paths consume the
same construction logic.

The Bicep API container module receives an explicit
`AUTO_START_WORKER=False` environment setting in production, so a
misconfigured deployment cannot accidentally start two worker
instances.

### Slice 2 — Producer dedupe, error classification, frontend idempotency-key

A new deep module **`project_generation_producer`** encapsulates the
new request flow: extract and validate the `Idempotency-Key` header,
construct the deterministic job id, perform the lease precheck against
`current_project_job_id`, compose the design brief only past both
dedupe gates, perform the CAS lease acquire, enqueue the queue
message, and classify any exception. Its return type is a
discriminated union: `AlreadyInFlight(job_id)`,
`NewlyEnqueued(job_id)`, or `EnqueueFailed(error_kind, http_status,
user_message)`. The HTTP endpoint becomes a thin wrapper that
translates this into 200 (already in flight), 202 (newly enqueued),
or a 4xx/5xx with the structured error body.

A new deep module **`job_errors`** encapsulates the
exception-to-`ErrorKind` mapping. The enum has five values:
`QUEUE_PERMISSION`, `BRIEF_FAILED`, `STORE_FAILED`, `UNAVAILABLE`,
`UNKNOWN`. The classifier function takes an exception, returns
`(error_kind, user_message, http_status)`. Azure Core
`ClientAuthenticationError` and `HttpResponseError` carrying
`AuthorizationPermissionMismatch` map to `QUEUE_PERMISSION` with a
developer-targeted message that names the missing role. LLM-side
errors during inline brief composition map to `BRIEF_FAILED`. Cosmos
write errors map to `STORE_FAILED`. Anything else maps to `UNKNOWN`.

The job document schema gains an optional `error_kind` field
alongside the existing `error: { type, message }` substructure. The
worker writes `error_kind` on terminal-failure transitions.

The deterministic job id format is extended to use the
idempotency-key as the revision component:
`{project_id}:project:project:{idempotency_key}`. A `create_job`
collision (Cosmos 409 from `If-None-Match: *`) is interpreted as
proof that an idempotent retry has arrived, and the response is
`200 { already_in_flight: true, job_id: existing }`.

The lease precheck reads the project's `current_project_job_id`. If
it is set and the referenced job is non-terminal, the producer
responds with `200 { already_in_flight: true, job_id: holder }`
without composing the brief or creating a new job. The producer then
performs its own CAS lease acquire as part of the success path; the
existing `acquire_project_lease` primitive already supports the
"holder is me" idempotent re-acquire.

On the front-end, the `enqueueProjectGeneration` service helper mints
`crypto.randomUUID()` per call inside the service layer (callers do
not pass it in), sets the `Idempotency-Key` header, and parses the
response shape `{ job_id, already_in_flight }`. Errors are parsed for
`error_kind` to enable the Slice 5 UI mapping.

### Slice 3 — Reconcile rewrite

The existing `reconcile_project` function is split. Variation cleanup
behavior (with its existing staleness gate) is preserved unchanged in
the original function, but the project status mutation is removed
entirely. A new function `compute_project_status_from_jobs(project,
store)` derives the canonical project status by short-circuiting on
non-`processing` projects and on missing `current_project_job_id`,
fetching the active job from the jobs container only when both
conditions are met, returning `None` (no change) when an active
non-terminal job is found, and falling back to a pure
`_derive_status_from_rooms` helper otherwise. The buggy "mixed room
statuses imply failed" branch is removed; mixed states fall through
to "pending".

All four callsites (`list_projects`, `get_project`, `reset_project`,
and the additional location at line 963) gain a `Depends(get_job_store)`
dependency and call both helpers, performing a single writeback if
either mutated the document.

### Slice 4 — Cancel-all endpoint, staleness detector, header subline

A new HTTP endpoint `DELETE /staging/projects/{project_id}/jobs`
returns `202 { status, cancelled_count, project_id }` and reuses the
existing `_cascade_cancel_project_jobs` helper. The endpoint is
idempotent (already-terminal projects return `cancelled_count: 0`).

A new front-end deep module **`job-staleness`** encapsulates the
detector logic as a pure function: `computeStaleness(jobs,
lastEventByJobId, now) -> StalenessState[]`. Detector A (pickup) uses
`now - job.created_at` for jobs in `pending` status. Detector B
(stalled) uses `now - lastEventByJobId[id]` for jobs in `running`
status. Both detectors return `fresh | soft | hard` with thresholds
at 45s and 120s.

The jobs context tracks `lastEventByJobId` updated on every
`event: job` SSE delivery, and runs a 5-second `setInterval` (no
pause on hidden tab) that invokes `computeStaleness` and exposes the
result as a hook value.

The project page header keeps its existing counters and gains a
dynamic subline driven by the staleness state. Copy table:

|             | 45s soft                                      | 120s hard                                                                       |
|-------------|----------------------------------------------|--------------------------------------------------------------------------------|
| pending (A) | "Waiting for worker to pick up your job…"     | "Worker may be unavailable. Try cancelling and starting again."                |
| running (B) | "Generation paused — last update 45s ago"     | "Worker stopped responding. Cancel to free the queue and retry."               |

The "Cancel queued jobs" button appears at the 120s hard threshold,
is one-tap (no confirmation dialog), and always cascades. Click flow:
disable button + spinner + "Cancelling…" subline; SSE confirmation
dismisses the banner and shows a success toast; if no confirmation
arrives within 10 seconds the banner dismisses with a fallback toast
explaining the cancellation was queued.

The same banner UX is reused during the embedded-worker startup race
(the brief window after lifespan startup before the worker has
actually polled the queue).

### Slice 5 — Banner, optimistic tile, activity log, error UI

The `startGeneration` callback in the project page renders a banner
synchronously on click (replacing today's silent `isEnqueueing`
state). The banner copy is phased: from 0–14 seconds it reads
"Composing design brief…"; at 15 seconds it switches to "Submitting
to queue…". No fake progress percentage is shown.

On a 202 response, an optimistic project tile is rendered immediately
in the room grid so the user sees their work before the SSE seed
catches up. On a 200 response (`already_in_flight: true`), the banner
is suppressed in favor of a small toast ("Generation already in
progress") and the page scrolls to the existing tile.

A new front-end deep module **`activity-log-derivation`** encapsulates
the SSE-to-log-entry diff as a pure function: `deriveLogEntries(prev,
current) -> LogEntry[]`. The function emits an entry on phase changes
(extracted from the `phase` field on each job), drops
progress-percentage tick events, and includes heartbeat-stale
warnings sourced from the Slice 4 detector. Activity log entries are
in-memory only; they reset on page reload (the user's job tile state
is preserved via SSE seed restoration, which is sufficient).

A new front-end deep module **`error-kind-copy`** encapsulates the
`ErrorKind -> { userMessage, retryable, showAdminContact }` mapping.
The `QUEUE_PERMISSION` message is developer-targeted and names the
specific Azure role needed. The recovery banner gains an expandable
"Show technical details" section (collapsed by default) that displays
the raw `error.message` and `error.type` fields for diagnostics.

### Cross-slice contracts

The new producer response shape is a stable contract consumed by
Slices 4 and 5:

- 200 `{ job_id, already_in_flight: true }` — dedupe hit, no new work.
- 202 `{ job_id, already_in_flight: false }` — newly enqueued.
- 4xx/5xx `{ error_kind, user_message, detail }` — classified failure.

The new cancel-all endpoint response shape:

- 202 `{ status: "accepted", cancelled_count: number, project_id }`.

The job document schema additions:

- `error_kind: Optional[ErrorKind]` — written by the worker on
  terminal failure and by the producer on enqueue failure.

## Testing Decisions

A good test for this redesign exercises external behavior — the
inputs and outputs of the deep modules and the user-visible behavior
in Playwright — and avoids asserting on private implementation
details. Each deep module has a small surface that can be tested with
table-driven inputs, fakes, or mocked clocks; tests should not reach
into `__private` attributes or assert on log lines that are not part
of the contract.

All seven deep modules receive unit tests:

1. **`embedded_worker`** — lifespan integration test with a mocked
   worker verifies start/stop ordering, the `should_auto_start`
   policy is table-driven over `(role, env_flag)` pairs, and the
   `[embedded-worker]` log prefix appears on emitted records.
   Prior art: existing FastAPI lifespan tests in the repo.

2. **`job_errors`** — table-driven classifier tests over
   `(exception_class, exception_args) -> (error_kind, http_status)`.
   The `AuthorizationPermissionMismatch` case includes both raw
   `HttpResponseError` and `ClientAuthenticationError` inputs.
   Prior art: existing classifier tests in
   `backend/tests/core/test_*.py`.

3. **`project_generation_producer`** — the discriminated-union return
   type makes this straightforward: a fake `JobStore` plus fake
   `lease_storage` plus fake `queue` plus fake `brief_service` lets
   each branch be exercised. Cases include first-time enqueue, same
   idempotency-key retry, lease-held different idempotency-key,
   queue-enqueue exception, brief composition exception, and CAS
   conflict during lease acquire. Prior art:
   `backend/tests/core/test_staging_dispatcher.py` for fake-driven
   scenario coverage.

4. **`staging_reconcile` split** — `compute_project_status_from_jobs`
   gets a table of `(project state, job state) -> expected status`
   covering: status≠processing short-circuit, no current_project_job_id
   short-circuit, active non-terminal job → no change, terminal job
   present → derived from rooms, missing job document → derived from
   rooms, mixed-room states → "pending" (not "failed"). The
   `_derive_status_from_rooms` pure function is tested separately.
   `reconcile_project` itself is tested for the negative property:
   given any input it must not mutate `project_data["status"]`. An
   integration test reproduces the bug-report scenario end-to-end
   (project=processing, no current_project_job_id, rooms in pending)
   and asserts the project status remains processing rather than
   flipping to failed.

5. **`job-staleness`** — pure function tests with table inputs
   `(jobs, lastEventByJobId, now)` covering: fresh state, soft
   threshold A (pending 45s+), soft threshold B (running silent
   45s+), hard threshold A (pending 120s+), hard threshold B
   (running silent 120s+), fresh transition (event arrives, state
   resets), no jobs (empty array). Prior art:
   `frontend/lib/__tests__/*.test.ts`.

6. **`activity-log-derivation`** — pure function tests with diff
   sequences: no change → no entries, phase change → one entry,
   progress-percentage-only change → no entries (suppression),
   terminal failure with error_kind → one entry with mapped copy,
   multiple jobs simultaneously → multiple entries.

7. **`error-kind-copy`** — table-driven mapping tests for all five
   `ErrorKind` values, plus the unknown-kind fallback to the default
   "Couldn't start generation, try again" copy.

In addition to unit tests, each slice includes Playwright additions:

- **Slice 1**: full project generation completes end-to-end in dev
  (proof that the embedded worker runs).
- **Slice 2**: rapid double-click results in exactly one job created;
  the second click surfaces a toast (not a banner).
- **Slice 3**: a project with stuck pending jobs stays in
  `processing` status and does not flip to `failed` after the
  staleness window elapses.
- **Slice 4**: the 120s staleness state shows the cancel-queued
  button; clicking it cascades cancellation and the banner dismisses
  on SSE confirmation.
- **Slice 5**: the happy path (click → banner → optimistic tile →
  activity log entries → completion); the error path (forced RBAC
  error → banner with `QUEUE_PERMISSION` copy plus expandable raw
  details); the dedupe path (200 already_in_flight → toast and no
  banner).

Playwright reports are saved to `tests/playwright/<YYYY-MM-DD-HHMMSS>/`
per the AGENTS.md convention. Each slice merges only after backend
pytest, frontend `npm run build`, `npx next lint`, and `npx
playwright test` all pass locally.

## Out of Scope

- Cleanup of pre-existing stuck jobs from before the fix lands. The
  worker drains them naturally on its first boot after Slice 1.
- Localization of error copy. Strings are English-only.
- Percentage rollouts or A-B testing of the new UX. Each slice ships
  to all users at once, behind the existing `FEATURE_ASYNC_QUEUE`
  flag.
- Migration of legacy projects without `current_project_job_id`. The
  `compute_project_status_from_jobs` fallback to room-derived status
  handles them implicitly.
- Rewriting the `JobWorker._emit_progress` synthetic-progress
  estimator. The worker continues to emit progress; the front-end
  simply does not bridge progress-percentage ticks into the activity
  log.
- A backend confirmation dialog for cancel-all. The cancel action is
  one-tap by design (Q12 decision).
- Auto-cancellation of long-running jobs based on staleness. The
  user is the one with context to decide when to give up; the
  backend's existing visibility-timeout-and-poison-queue mechanism
  handles silent failures within roughly 5 minutes of redelivery
  attempts.
- A new feature flag for the redesign itself. The existing
  `FEATURE_ASYNC_QUEUE` flag already gates the entire producer
  surface.

## Further Notes

The lease primitive (`current_project_job_id` on the project
document, with etag-protected CAS via `MatchConditions.IfNotModified`)
already exists in `backend/core/project_lease.py`. The
`acquire_project_lease` function is already designed to be idempotent
when the holder is the same job id, which lets the producer
pre-acquire a lease that the worker dispatcher then re-acquires on
pickup without contract changes.

The deterministic job id format (`{pid}:{rid}:{vid}:{revision}`)
already accepts an arbitrary revision string and the `create_job`
flow already swallows 409 collisions. Plumbing the
`Idempotency-Key` header through as the revision is a one-line
change to the job-id factory.

The 90-second visibility timeout, 30-second heartbeat, and 3-attempt
poison-queue threshold in the existing `JobWorker` infrastructure
provide the silent-failure recovery for Detector B (the user-visible
warning at 45s/120s is purely informational; the actual recovery is
queue-level redelivery).

The `[embedded-worker]` log prefix is for human triage and is not
parsed by any automated system; it can be added or removed without
breaking contracts.
