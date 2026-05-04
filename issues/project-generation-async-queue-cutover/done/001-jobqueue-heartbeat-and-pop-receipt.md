## Parent PRD

`prds/2026-05-03-project-generation-async-queue-cutover-prd.md`

## What to build

Make the `JobWorker` safe to run multi-minute jobs by adding a Storage
Queue visibility-timeout heartbeat that **rolls forward the pop receipt
on every extension**. Without this, every successful long-running
project-generation job will be redelivered after the original visibility
window expires, and the project pipeline will run twice.

End-to-end behaviour, per PRD section "Heartbeat for long-running
messages":

- New `JobQueue.extend_visibility(message, timeout_seconds)` wrapper that
  calls Storage Queue's `update_message`, captures the SDK's refreshed
  pop receipt, and **mutates `message.raw` in place** so a subsequent
  `complete()` (which delegates to `delete_message` using `message.raw`)
  uses the latest receipt rather than a stale one.
- `JobWorker.process_one()` spawns an asyncio heartbeat task that fires
  every 30 seconds while the dispatcher runs. The heartbeat is
  unconditional (not gated on job kind) — fast variation jobs (~20s)
  cancel the task before its first wake; the design must remain safe
  for borderline ~31s jobs.
- Heartbeat extension and message completion are serialized via an
  `asyncio.Lock` attached to the message so an in-flight extend cannot
  race with a `complete()` call.
- `ResourceNotFoundError` and `HttpResponseError` raised by the
  heartbeat after the message has already been deleted are swallowed
  (logged at debug, loop exits cleanly).
- The heartbeat task is cancelled when the dispatcher returns or
  raises; cancellation is awaited and `asyncio.CancelledError` is
  suppressed.

This slice ships only the queue/worker plumbing. No new dispatcher
kinds, no endpoints, no frontend. It is verified end-to-end through the
unit tests below.

## Acceptance criteria

- [ ] `JobQueue.extend_visibility(message, timeout_seconds)` exists and
      writes both `pop_receipt` and `next_visible_on` from the SDK
      response back onto `message.raw` in place.
- [ ] `JobWorker.process_one()` spawns and cancels a heartbeat task per
      message; cancellation is awaited and the suppressed
      `CancelledError` doesn't leak.
- [ ] An `asyncio.Lock` per message serializes `extend_visibility` and
      `complete()`; concurrent extend-then-complete uses the freshest
      pop receipt.
- [ ] After the dispatcher returns, `complete()` succeeds even when at
      least one heartbeat extension fired during the run (regression
      pin for the rubber-duck blocking #1 finding — without the
      write-back, this would 404 and trigger redelivery).
- [ ] `ResourceNotFoundError` / `HttpResponseError` raised inside the
      heartbeat after `complete()` ran is swallowed; the test asserts
      no propagation and the heartbeat task ends cleanly.
- [ ] New `tests/test_job_worker_heartbeat.py` exercises: heartbeat
      fires every 30s; extend-then-complete uses the new receipt;
      concurrent extend + complete serializes via the lock;
      ResourceNotFoundError on extend after complete is swallowed;
      heartbeat is cancelled when the dispatcher returns or raises.
- [ ] Existing `tests/test_job_worker.py` and
      `tests/test_job_worker_progress.py` still pass.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` is green.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 24
- User story 25
- User story 28
