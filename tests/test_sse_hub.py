"""Tests for ``SSEHub`` — per-replica in-memory pub/sub for job state
events (issue 005, image-pipeline-and-project-ux-overhaul PRD).

Pinned behavior:

- One feed_source pump per hub instance.
- Subscribers register by ``project_id``; events with mismatching
  ``project_id`` never land in their queue.
- Multiple subscribers on the same project all receive every matching
  item (broadcast).
- Polling cycles are driven by an injected feed_source so tests don't
  need Cosmos.
- Subscription handles drop themselves from the hub on close.
- The pump survives a single failed iteration (logged, not crashed).
"""
from __future__ import annotations

import asyncio
from typing import Any, Iterator, Optional

import pytest

from backend.core.sse_hub import SSEHub


# ---------------------------------------------------------------------------
# Fake feed source — returns a fresh iterator each call so we can simulate
# polling cycles. Each "page" is a (items, continuation_token) tuple.
# ---------------------------------------------------------------------------


class FakeFeed:
    """Mock for the ``JobStore.subscribe_change_feed`` callable shape.

    Each call returns the NEXT batch as a single-page iterator. After all
    batches are exhausted, returns empty pages forever (so the hub keeps
    polling without deadlocking).

    Now accepts a resume-state dict (``{"continuation": ..., "since": ...}``)
    so the harness mirrors the new ``FeedSource`` signature.
    """

    def __init__(self, batches: list[list[dict]]):
        self._batches = list(batches)
        self.calls: list[dict] = []

    def __call__(self, state: dict) -> Iterator[tuple]:
        # Snapshot a copy — the hub mutates its own state between calls
        # and we want each entry to record the values at call time.
        self.calls.append(dict(state))
        if self._batches:
            page = self._batches.pop(0)
            yield page, f"cont-{len(self.calls)}"
        else:
            yield [], None

    @property
    def last_continuation(self) -> Optional[str]:
        return self.calls[-1]["continuation"] if self.calls else None


async def _drain(sub, timeout: float = 1.0, count: int = 1) -> list[dict]:
    """Pull ``count`` events off a subscription with a per-event timeout."""
    out: list[dict] = []
    for _ in range(count):
        evt = await asyncio.wait_for(sub.queue.get(), timeout=timeout)
        out.append(evt)
    return out


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_returns_handle_with_queue():
    feed = FakeFeed([])
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    sub = await hub.subscribe("proj-A")
    assert sub.project_id == "proj-A"
    assert sub.queue.qsize() == 0
    await sub.aclose()


@pytest.mark.asyncio
async def test_close_removes_subscription_from_hub():
    feed = FakeFeed([])
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    sub = await hub.subscribe("proj-A")
    assert hub.subscriber_count == 1
    await sub.aclose()
    assert hub.subscriber_count == 0
    # Idempotent — second close is a no-op.
    await sub.aclose()
    assert hub.subscriber_count == 0


# ---------------------------------------------------------------------------
# Routing — one project's events do not leak to another's subscribers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_routed_only_to_matching_project_subscribers():
    feed = FakeFeed([
        [
            {"id": "j1", "project_id": "proj-A", "status": "pending"},
            {"id": "j2", "project_id": "proj-B", "status": "pending"},
            {"id": "j3", "project_id": "proj-A", "status": "running"},
        ],
    ])
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    sub_a = await hub.subscribe("proj-A")
    sub_b = await hub.subscribe("proj-B")
    await hub.start()
    try:
        a_events = await _drain(sub_a, count=2, timeout=2.0)
        b_events = await _drain(sub_b, count=1, timeout=2.0)
    finally:
        await hub.stop()
        await sub_a.aclose()
        await sub_b.aclose()

    assert [e["id"] for e in a_events] == ["j1", "j3"]
    assert [e["id"] for e in b_events] == ["j2"]


# ---------------------------------------------------------------------------
# Broadcast — multiple subs on the same project all receive every event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_subscribers_on_same_project_all_receive_events():
    feed = FakeFeed([
        [
            {"id": "j1", "project_id": "proj-A", "status": "pending"},
            {"id": "j1", "project_id": "proj-A", "status": "running"},
        ],
    ])
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    sub1 = await hub.subscribe("proj-A")
    sub2 = await hub.subscribe("proj-A")
    await hub.start()
    try:
        e1 = await _drain(sub1, count=2, timeout=2.0)
        e2 = await _drain(sub2, count=2, timeout=2.0)
    finally:
        await hub.stop()
        await sub1.aclose()
        await sub2.aclose()
    assert [e["id"] for e in e1] == ["j1", "j1"]
    assert [e["id"] for e in e2] == ["j1", "j1"]
    assert [e["status"] for e in e1] == ["pending", "running"]


# ---------------------------------------------------------------------------
# Order preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_delivered_in_change_feed_order():
    states = ["pending", "running", "running", "succeeded"]
    feed = FakeFeed([
        [{"id": "j", "project_id": "proj-A", "status": s} for s in states],
    ])
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    sub = await hub.subscribe("proj-A")
    await hub.start()
    try:
        out = await _drain(sub, count=4, timeout=2.0)
    finally:
        await hub.stop()
        await sub.aclose()
    assert [e["status"] for e in out] == states


# ---------------------------------------------------------------------------
# Pump survives a failing poll cycle (logged, not crashed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pump_survives_feed_source_exception(caplog):
    """A single failed poll must not kill the pump — the next poll
    proceeds and delivers events."""
    cycles = {"n": 0}

    def failing_then_ok(_state):
        cycles["n"] += 1
        if cycles["n"] == 1:
            raise RuntimeError("boom — change feed transient failure")
        yield [{"id": "j-recovered", "project_id": "proj-A", "status": "running"}], None

    hub = SSEHub(feed_source=failing_then_ok, poll_interval=0.01)
    sub = await hub.subscribe("proj-A")
    await hub.start()
    try:
        out = await _drain(sub, count=1, timeout=2.0)
    finally:
        await hub.stop()
        await sub.aclose()
    assert out[0]["id"] == "j-recovered"


# ---------------------------------------------------------------------------
# Continuation token plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_continuation_token_advances_across_polls():
    """The hub passes the most recent continuation token back into
    the feed source so JobStore can resume across polls without
    re-reading the entire history each time."""
    feed = FakeFeed([
        [{"id": "j1", "project_id": "proj-A", "status": "pending"}],
        [{"id": "j2", "project_id": "proj-A", "status": "running"}],
    ])
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    sub = await hub.subscribe("proj-A")
    await hub.start()
    try:
        await _drain(sub, count=2, timeout=2.0)
    finally:
        await hub.stop()
        await sub.aclose()

    # First call: cold start (both fields None).
    assert feed.calls[0] == {"continuation": None, "since": None}
    # By the time the second batch is consumed, the token from page 1
    # must have been passed into a later poll.
    assert any(c["continuation"] == "cont-1" for c in feed.calls[1:])


# ---------------------------------------------------------------------------
# Resume-state semantics — issue 002 PRD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_start_passes_both_fields_none():
    """First poll on a fresh hub: both resume-state fields are None
    so the closure can branch to its boot_iso fallback."""
    feed = FakeFeed([])
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    await hub.start()
    try:
        # Wait for at least one poll to land.
        for _ in range(50):
            if feed.calls:
                break
            await asyncio.sleep(0.02)
    finally:
        await hub.stop()
    assert feed.calls, "feed source was never invoked"
    assert feed.calls[0] == {"continuation": None, "since": None}


@pytest.mark.asyncio
async def test_resume_round_trip_token_appears_on_next_poll():
    """A token returned on poll N appears as state['continuation'] on
    poll N+1, with state['since'] cleared."""

    class TokenThenIdle:
        def __init__(self):
            self.calls: list[dict] = []
            self._first = True

        def __call__(self, state: dict):
            self.calls.append(dict(state))
            if self._first:
                self._first = False
                yield [{"id": "j1", "project_id": "proj-A", "status": "pending"}], "TOKEN-XYZ"
            else:
                yield [], None

    feed = TokenThenIdle()
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    sub = await hub.subscribe("proj-A")
    await hub.start()
    try:
        await _drain(sub, count=1, timeout=2.0)
        # Wait for at least one further poll after the producing one.
        for _ in range(100):
            if len(feed.calls) >= 2:
                break
            await asyncio.sleep(0.02)
    finally:
        await hub.stop()
        await sub.aclose()

    assert len(feed.calls) >= 2
    assert feed.calls[0] == {"continuation": None, "since": None}
    # The very next poll after the token-yielding one carries the token.
    assert feed.calls[1] == {"continuation": "TOKEN-XYZ", "since": None}


@pytest.mark.asyncio
async def test_since_fallback_when_items_yielded_without_token():
    """A poll that yields items but no token sets state['since'] on the
    next poll to an ISO timestamp captured immediately before the
    token-less poll began (NOT boot_iso, NOT poll-end time).

    Verified by patching ``sse_hub.time.strftime`` *only on the
    sse_hub module's bound reference* so the global ``time`` module
    is untouched.
    """
    import backend.core.sse_hub as sse_hub_mod

    class FakeTime:
        """Stand-in for the ``time`` module rebinding inside sse_hub."""

        def __init__(self, real):
            self._real = real
            self.n = 0

        def strftime(self, fmt, t=None):
            self.n += 1
            if self.n == 1:
                # First strftime call is poll_start_iso for the
                # token-less producing poll.
                return "2026-05-02T00:00:00Z"
            return self._real.strftime(fmt, t) if t is not None else self._real.strftime(fmt, self._real.gmtime())

        def gmtime(self, *a, **kw):
            return self._real.gmtime(*a, **kw)

    class ItemsNoToken:
        def __init__(self):
            self.calls: list[dict] = []
            self._first = True

        def __call__(self, state: dict):
            self.calls.append(dict(state))
            if self._first:
                self._first = False
                yield [{"id": "j1", "project_id": "proj-A", "status": "pending"}], None
            else:
                yield [], None

    feed = ItemsNoToken()
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    real_time = sse_hub_mod.time
    sse_hub_mod.time = FakeTime(real_time)
    sub = await hub.subscribe("proj-A")
    try:
        await hub.start()
        await _drain(sub, count=1, timeout=2.0)
        for _ in range(100):
            if len(feed.calls) >= 2:
                break
            await asyncio.sleep(0.02)
        await hub.stop()
    finally:
        sse_hub_mod.time = real_time
        await sub.aclose()

    assert len(feed.calls) >= 2
    assert feed.calls[0] == {"continuation": None, "since": None}
    # Next poll's `since` must be the timestamp captured BEFORE the
    # producing poll, not boot_iso or any later wall-clock value.
    assert feed.calls[1] == {"continuation": None, "since": "2026-05-02T00:00:00Z"}


@pytest.mark.asyncio
async def test_state_untouched_when_no_items_and_no_token():
    """A poll that yields nothing must NOT silently advance past unseen
    events: state passed to the next poll matches the prior poll's
    state byte-for-byte."""

    class TokenThenIdle:
        def __init__(self):
            self.calls: list[dict] = []
            self._batches = [
                ([{"id": "j1", "project_id": "proj-A", "status": "pending"}], "TOKEN-1"),
            ]

        def __call__(self, state: dict):
            self.calls.append(dict(state))
            if self._batches:
                items, token = self._batches.pop(0)
                yield items, token
            else:
                yield [], None

    feed = TokenThenIdle()
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    sub = await hub.subscribe("proj-A")
    await hub.start()
    try:
        await _drain(sub, count=1, timeout=2.0)
        # Let several idle polls happen.
        for _ in range(100):
            if len(feed.calls) >= 4:
                break
            await asyncio.sleep(0.02)
    finally:
        await hub.stop()
        await sub.aclose()

    assert len(feed.calls) >= 4
    # After the producing poll set continuation=TOKEN-1, every subsequent
    # idle poll must see the same dict — the empty pages must not bump
    # state forward.
    after_producing = feed.calls[1:]
    for entry in after_producing:
        assert entry == {"continuation": "TOKEN-1", "since": None}


# ---------------------------------------------------------------------------
# Items missing project_id are silently dropped (defensive)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_items_without_project_id_are_dropped():
    feed = FakeFeed([
        [
            {"id": "good", "project_id": "proj-A", "status": "pending"},
            {"id": "bad", "status": "pending"},  # no project_id
        ],
    ])
    hub = SSEHub(feed_source=feed, poll_interval=0.01)
    sub = await hub.subscribe("proj-A")
    await hub.start()
    try:
        evt = await _drain(sub, count=1, timeout=2.0)
        # Confirm no second event arrives within a reasonable wait.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sub.queue.get(), timeout=0.2)
    finally:
        await hub.stop()
        await sub.aclose()
    assert evt[0]["id"] == "good"


# ---------------------------------------------------------------------------
# stop() is idempotent + safe before start()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_before_start_is_noop():
    hub = SSEHub(feed_source=FakeFeed([]), poll_interval=0.01)
    await hub.stop()  # no exception


@pytest.mark.asyncio
async def test_double_start_is_noop():
    hub = SSEHub(feed_source=FakeFeed([]), poll_interval=0.01)
    await hub.start()
    await hub.start()  # no exception, single task
    await hub.stop()
