## Parent PRD

`prds/2026-05-02-sse-change-feed-start-time-type-fix-prd.md`

## What to build

Fix the `Invalid start_time` traceback that silently breaks the SSE change-feed stream by threading timezone-aware `datetime` end-to-end where ISO 8601 strings are currently used as the Cosmos change-feed resume marker.

End-to-end behavior after this slice:

- A user who refreshes a project page mid-analysis sees job status updates resume in near real-time.
- The backend log no longer floods with `ValueError: Invalid start_time '...'` on every SSE poll cycle.
- The `_feed` closure in `get_or_create_hub` forwards a `datetime` (not a string) to `JobStore.subscribe_change_feed(start_time=...)` on cold start and on every post-event poll that resumes via timestamp.

Implementation per the PRD's Implementation Decisions section:

- `backend/core/job_store.py`: change `subscribe_change_feed` parameter from `start_time: Optional[str] = None` to `start_time: Optional[datetime] = None`; update docstring to call out the SDK contract (`datetime` only; `"Now"`/`"Beginning"` literals not exposed) and the mutual exclusivity with `continuation` and the cold-start branch. Add `from datetime import datetime` (and `timezone` if needed).
- `backend/core/sse_hub.py`: change internal `_since` from `Optional[str]` to `Optional[datetime]` (timezone-aware UTC); replace the poll-loop `time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())` with `poll_start_dt = datetime.now(timezone.utc)`; replace the cold-start construction in `get_or_create_hub` with `boot_dt = datetime.now(timezone.utc)`; update the `"since"` state-dict comment to reflect `Optional[datetime]` (key name unchanged); have the `_feed` closure forward `since` / `boot_dt` directly without conversion. Add `from datetime import datetime, timezone`. Drop now-unused `time.strftime` / `time.gmtime` resume-marker calls (leave `time` import alone if used elsewhere).

Behavioral contract preserved exactly (per PRD): `continuation` → `start_time` → `is_start_from_beginning=True` priority order; `_collect_once` state-update rules (token trumps timestamp; items-without-token sets `_since` to pre-call moment; empty-and-no-token leaves both untouched); continuation tokens stay SDK-opaque strings; `_hub_instance` stays a process-singleton.

Regression tests per the PRD's Testing Decisions section, co-located with existing `SSEHub` / `JobStore` tests under `tests/`, using the existing fake/recording-container pattern:

- `JobStore.subscribe_change_feed`: when called with `start_time=<datetime>`, the recording container's `query_items_change_feed` receives the same `datetime` object — assert `isinstance(captured_kwarg, datetime)` (load-bearing regression assertion) and identity/equality with the input. Also verify the `continuation` and cold-start (`is_start_from_beginning=True`) branches still forward the right kwargs.
- `SSEHub._collect_once`: with a feed source that yields items but no continuation token, `_since` becomes a timezone-aware UTC `datetime` and `_continuation` is cleared; token-wins-over-timestamp and empty-and-no-token-leaves-state-untouched branches still hold under the new type.
- `get_or_create_hub`'s `_feed` closure (or equivalent extraction point): on cold start, the value forwarded as `start_time` is a `datetime`, not a string.
- Update any existing tests that pass ISO strings as `start_time` to pass `datetime` instead — those tests were encoding the bug.

## Acceptance criteria

- [ ] `JobStore.subscribe_change_feed` signature is `start_time: Optional[datetime] = None`; docstring updated per PRD.
- [ ] `SSEHub._since` holds `Optional[datetime]` (timezone-aware UTC); state-dict `"since"` comment updated; key name unchanged.
- [ ] Poll-loop pre-call timestamp is `datetime.now(timezone.utc)` (no `time.strftime` / `time.gmtime` for resume-marker construction).
- [ ] `get_or_create_hub` cold-start `boot_dt` is `datetime.now(timezone.utc)`, captured (not the `"Now"` sentinel) so events between hub boot and first poll completion are included.
- [ ] `_feed` closure forwards `since` / `boot_dt` directly to `subscribe_change_feed(start_time=...)` with no conversion.
- [ ] Continuation-token path is byte-for-byte unchanged (SDK-opaque string; no shape change).
- [ ] Mutual-exclusivity priority order (`continuation` → `start_time` → `is_start_from_beginning=True`) is preserved.
- [ ] `_collect_once` state-update rules are preserved under the new type (token trumps timestamp; items-without-token sets `_since` to pre-call moment; empty-and-no-token untouched).
- [ ] Regression test asserts `isinstance(captured_kwarg, datetime)` on the value forwarded to `query_items_change_feed` — fails loudly if anyone reintroduces an ISO string into `start_time`.
- [ ] Tests cover cold-start, items-without-token timestamp resume, continuation-token resume, and empty-and-no-token branches.
- [ ] Any pre-existing test that passed an ISO string as `start_time` is updated to pass a `datetime`.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` passes locally.
- [ ] After the fix, exercising `/jobs/stream` on a cold backend produces no `Invalid start_time` traceback, and a browser refresh during an in-flight analysis shows job status updates resuming in the UI.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 3
- User story 4
- User story 5
- User story 6
- User story 7
- User story 8
- User story 9
- User story 10
- User story 11
- User story 12

## Completion note

Done. `subscribe_change_feed` now takes `Optional[datetime]`; `SSEHub._since` is `Optional[datetime]`; both ISO-string `time.strftime` sites replaced with `datetime.now(timezone.utc)`. Tests updated + 2 new regression pins (`isinstance(captured, datetime)` on `subscribe_change_feed` start_time + cold-start `get_sse_hub` closure). Full sse_hub + job_store suites green (35/35).
