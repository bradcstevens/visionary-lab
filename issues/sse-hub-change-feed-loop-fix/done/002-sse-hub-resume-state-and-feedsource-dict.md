## Parent PRD

`prds/2026-05-02-sse-hub-change-feed-loop-fix-prd.md`

## What to build

Make the SSE hub remember a real resume marker between polls so events
are delivered exactly once per subscriber, rather than re-broadcasting
every event since boot or silently advancing past unseen events.

End-to-end behaviour, per PRD sections "`SSEHub` resume state" and
"`get_sse_hub` singleton closure":

- `SSEHub` holds two opaque resume-state fields, `_continuation`
  (Cosmos token) and `_since` (ISO-8601 UTC timestamp with `Z`
  suffix, matching the existing `boot_iso` format).
- The `FeedSource` type alias becomes
  `Callable[[dict], Iterable[tuple]]` where the dict is
  `{"continuation": Optional[str], "since": Optional[str]}`.
- `_collect_once` updates state after a successful poll:
  - If any token was returned during iteration →
    `_continuation = token`, `_since = None`.
  - Else if items were produced but no token was returned →
    `_continuation = None`, `_since = poll_start_iso` captured
    *before* the call to `feed_source`.
  - Else (no items, no token) → state untouched, so the prior
    resume marker is preserved.
- `_collect_once` continues to run on a worker thread via
  `asyncio.to_thread`; resume-state fields are read and written
  exclusively from that worker thread, so no lock is introduced.
- The `get_sse_hub` singleton-wiring closure interprets the state
  dict:
  - `state["continuation"]` truthy →
    `store.subscribe_change_feed(continuation=...)`.
  - else `state["since"]` truthy →
    `store.subscribe_change_feed(start_time=...)`.
  - else cold-start →
    `store.subscribe_change_feed(start_time=boot_iso)`.

Failure handling and backoff are still out of scope for this slice —
`_collect_once` may continue to swallow exceptions as it does today
until issue 003 lands. This slice is demoable as: a healthy backend
with the change feed correctly resuming via continuation token (no
duplicate dispatches across polls).

## Acceptance criteria

- [ ] `SSEHub` exposes `_continuation` and `_since` fields managed
      per the rules above.
- [ ] The `FeedSource` type alias and the singleton closure in
      `get_sse_hub` accept the resume-state dict shape
      `{"continuation": ..., "since": ...}`.
- [ ] Cold-start (both fields `None`) calls
      `subscribe_change_feed(start_time=boot_iso)` exactly as today.
- [ ] After a poll that yields a token, the next poll's dict has
      `continuation` set to that token and `since = None`.
- [ ] After a poll that yields items but no token, the next poll's
      dict has `since` set to the ISO timestamp captured immediately
      before the prior poll, and `continuation = None`.
- [ ] After a poll that yields no items and no token, the next
      poll's dict matches the previous poll's dict (state untouched).
- [ ] New unit test: resume round-trip — feed source records the
      dicts it was called with; a token returned on poll N appears as
      `state["continuation"]` on poll N+1.
- [ ] New unit test: `since` fallback — feed source returns items
      with `token=None`; next poll's `state["since"]` matches the ISO
      timestamp captured immediately before the failing poll, not
      `boot_iso`.
- [ ] Existing SSE-hub tests still pass; tests that construct their
      own feed source are updated to accept the new dict argument.

## Blocked by

- Blocked by `001-jobstore-subscribe-change-feed-kwargs.md` — this
  slice depends on `JobStore.subscribe_change_feed` accepting
  `continuation=` as a keyword-only argument.

## User stories addressed

Reference by number from the parent PRD:

- User story 3
- User story 4
- User story 10
- User story 14

---

## Completion note

Implemented in commit on 2026-05-02. SSEHub now holds `_continuation` +
`_since`; `FeedSource` accepts the `{"continuation", "since"}` dict;
`_collect_once` applies the three-rule state update (token wins,
items-without-token sets `since` to pre-call ISO, idle poll leaves state
untouched). `get_sse_hub` closure routes the dict to the matching
`subscribe_change_feed` resume kwarg. 4 new SSE-hub unit tests + 1
retargeted existing test all pass; full backend suite 718 passed
(same pre-existing `test_backyard_scenario` failure, unrelated).
