## Parent PRD

`prds/2026-05-02-sse-hub-change-feed-loop-fix-prd.md`

## What to build

Fix `JobStore.subscribe_change_feed` in `backend/core/job_store.py` so it
passes exactly one of the mutually-exclusive resume kwargs
(`continuation`, `start_time`, or `is_start_from_beginning`) to the Cosmos
SDK on every call. This is the direct cause of the once-per-second
`ValueError: is_start_from_beginning and start_time are exclusive` crash
that is currently pinning the projects page in a retry loop.

End-to-end behaviour:

- Signature becomes
  `subscribe_change_feed(start_time: Optional[str] = None, *, continuation: Optional[str] = None)`.
- Resume-priority order, evaluated once per call (per PRD
  "`JobStore.subscribe_change_feed` interface"):
  1. `continuation` truthy → only `continuation=...` reaches the SDK.
  2. else `start_time` truthy → only `start_time=...` reaches the SDK.
  3. else → only `is_start_from_beginning=True` reaches the SDK.
- Continuation-token extraction prefers
  `iterator.response_headers["etag"]` after `by_page()` iteration, with a
  fallback to `getattr(iterator, "continuation_token", None)`, then
  `None`.
- Yield contract is unchanged: `(items, continuation_token_or_None)` per
  page.

No callers of `subscribe_change_feed` are modified in this slice — the
new keyword-only `continuation` parameter is additive, and existing
positional `start_time` callers keep working. The SSE hub continues to
pass `start_time=boot_iso` until issue 002 lands.

## Acceptance criteria

- [ ] `subscribe_change_feed` signature matches the PRD exactly
      (`start_time` positional-or-keyword, `continuation` keyword-only).
- [ ] When called with `continuation="abc"`, the underlying
      `container.query_items_change_feed` mock receives `continuation`
      and neither `start_time` nor `is_start_from_beginning` in its
      kwargs.
- [ ] When called with `start_time="2026-..."` and no `continuation`,
      the mock receives `start_time` and neither `continuation` nor
      `is_start_from_beginning`.
- [ ] When called with neither argument, the mock receives
      `is_start_from_beginning=True` and neither `continuation` nor
      `start_time`.
- [ ] Continuation-token extraction returns the value of
      `response_headers["etag"]` when present, falls back to
      `iterator.continuation_token` when only that is set, and returns
      `None` when neither is available.
- [ ] Four new unit tests under `tests/` exercise the three resume-
      priority branches and the extraction-precedence path, using
      in-memory mocks (no live Cosmos).
- [ ] Existing `JobStore` and SSE-hub unit tests still pass
      (`uv run pytest tests/ --ignore=tests/integration -v`).
- [ ] The projects page no longer produces the
      `is_start_from_beginning and start_time are exclusive` error in
      backend logs after deploy (manual verification).

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 11
- User story 12

---

## Completion note

Implemented in `backend/core/job_store.py`:

- `subscribe_change_feed(start_time=None, *, continuation=None)` now
  forwards exactly one of `continuation`, `start_time`, or
  `is_start_from_beginning=True` (in that priority order). Eliminates
  the `is_start_from_beginning and start_time are exclusive` SDK error.
- New helper `_extract_continuation_token(iterator)` prefers
  `iterator.response_headers["etag"]`, falls back to
  `iterator.continuation_token`, then `None`.

Tests added in `tests/test_job_store.py` (4 new + reused existing one):

- `test_subscribe_change_feed_continuation_takes_priority`
- `test_subscribe_change_feed_start_time_when_no_continuation`
- `test_subscribe_change_feed_cold_start_uses_is_start_from_beginning`
- `test_subscribe_change_feed_continuation_extraction_precedence`
- Existing `test_subscribe_change_feed_yields_items_and_continuation`
  retargeted to the new etag-first extraction path via a shared
  `_make_change_feed_iterator` helper.

Verification: `uv run pytest tests/test_job_store.py tests/test_sse_hub.py -v`
→ 25 passed. Full backend suite: 714 passed, 1 pre-existing unrelated
failure (`test_backyard_scenario.py::test_backyard_test_data_exists`,
missing BACKYARD.md fixture). Manual log verification on deploy is the
last AC bullet — the unit-test resume-priority pins guarantee the SDK
will never see two exclusive kwargs together.

Issue 002 (SSE hub resume state) can now consume the new keyword-only
`continuation` parameter.
