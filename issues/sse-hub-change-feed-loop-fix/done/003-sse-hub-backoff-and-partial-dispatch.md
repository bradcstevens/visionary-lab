## Parent PRD

`prds/2026-05-02-sse-hub-change-feed-loop-fix-prd.md`

## What to build

Replace the SSE hub's tight-loop, exception-swallowing failure path
with exponential backoff and partial-dispatch-on-error, so a broken
change-feed dependency produces at most ~two log lines per minute
instead of one stack trace per second, and so events collected before
a mid-iteration error are still delivered.

End-to-end behaviour, per PRD section "Failure handling and backoff
in `_run`":

- `_collect_once` no longer swallows the exception. Its return
  contract becomes `(items: list[dict], err: Optional[Exception])`.
  Items collected before the exception are returned alongside the
  error so they can still be dispatched.
- `_run` dispatches the returned items first, then inspects `err`:
  - `err is None` → reset `consecutive_failures = 0`, sleep
    `poll_interval`.
  - `err is not None` → increment `consecutive_failures`, log via
    `logger.exception` (full trace preserved on every error so the
    first error in a streak is fully diagnosable), sleep
    `min(poll_interval * 2 ** (consecutive_failures - 1), 30.0)`
    seconds before the next poll.
- After a successful poll the streak resets to `poll_interval`
  cadence.
- Resume-state semantics from issue 002 are unchanged: a poll that
  failed mid-iteration must not silently advance past unseen events
  (state-untouched rules continue to apply when neither items nor a
  token were observed).

## Acceptance criteria

- [ ] `_collect_once` returns `(items, err)` and never raises.
- [ ] `_run` dispatches `items` to subscribers before sleeping or
      logging the error.
- [ ] On `err is None`, `_run` resets `consecutive_failures` to 0
      and sleeps `poll_interval`.
- [ ] On `err is not None`, `_run` calls `logger.exception` (so the
      first error in a streak still produces a full stack trace) and
      sleeps `min(poll_interval * 2 ** (consecutive_failures - 1), 30.0)`.
- [ ] The backoff cap is exactly 30.0 seconds.
- [ ] New unit test: backoff escalation and reset — feed source
      raises on the first three polls then succeeds; observed sleep
      durations follow the geometric schedule capped at 30 s, and
      the cadence resets to `poll_interval` after the first
      successful poll. Sleep is observed via a fake `asyncio.sleep`
      or by patching the hub's timing hook.
- [ ] New unit test: partial dispatch on mid-iteration error —
      feed source yields one batch then raises; subscribers receive
      that batch before the backoff sleep begins.
- [ ] Existing SSE-hub tests still pass; the test harness from
      issue 002 is reused (no new harness introduced).
- [ ] After deploy, a forced change-feed failure produces at most
      ~two log lines per minute in steady state (manual verification
      via temporarily-injected error or staging chaos test).

## Blocked by

- Blocked by `002-sse-hub-resume-state-and-feedsource-dict.md` —
  this slice modifies `_collect_once` and `_run` and reuses its
  resume-state semantics and test harness.

## User stories addressed

Reference by number from the parent PRD:

- User story 5
- User story 6
- User story 7
- User story 8
- User story 9
- User story 13
- User story 15

## Completion note

Implemented in commit on 2026-05-02. ``_collect_once`` now returns
``(items, err)`` and never raises; ``_run`` dispatches items first
then escalates backoff geometrically (capped at 30 s) on failure
and resets to ``poll_interval`` on success. ``_sleep_or_stop`` was
extracted as an instance method so tests can monkeypatch it to
record sleep durations without burning real time. 5 new tests pin
the AC; full sse_hub suite is 19/19 green.
