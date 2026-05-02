"""``SSEHub`` — per-replica in-memory pub/sub for image-job state events
(issue 005, image-pipeline-and-project-ux-overhaul PRD).

One ``SSEHub`` per worker / API replica. A single background task pumps
the Cosmos ``jobs`` change-feed (via ``JobStore.subscribe_change_feed``);
each item is dispatched to the set of subscribers registered for that
``project_id``. The ``/jobs/stream`` SSE endpoint constructs one
``Subscription`` per connected client and forwards events to the wire.

Design choices

  - **Decoupled from JobStore.** ``feed_source`` is a callable
    ``(continuation_token | None) -> Iterable[(items, token)]`` so the
    hub is testable in isolation. The lazy singleton wires it up to the
    real ``JobStore``.
  - **Polling pump runs in a worker thread** via ``asyncio.to_thread``
    because the Cosmos SDK's change-feed iterator is synchronous.
    Dispatch happens back on the event loop so subscriber queues are
    only ever touched from the loop — no cross-thread queue races.
  - **Bounded subscriber queues** (``maxsize=256``). A slow consumer
    drops events instead of OOMing the replica; the next change-feed
    poll will deliver the latest doc state anyway since subscribers
    only need eventual consistency for UI rendering.
  - **Pump survives transient errors** — a single failing iteration is
    logged and the next poll proceeds. Otherwise a transient Cosmos
    blip would orphan every connected client.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# Sync iterator shape returned by ``JobStore.subscribe_change_feed``:
#   yield (items: list[dict], continuation_token: str | None)
#
# The hub passes a resume-state dict on each call:
#   {"continuation": Optional[str], "since": Optional[str]}
# The source decides which (if either) to forward to the SDK.
FeedSource = Callable[[dict], Iterable[tuple]]


_QUEUE_MAXSIZE = 256


class Subscription:
    """Per-client handle returned by ``SSEHub.subscribe``.

    The endpoint reads from ``self.queue`` and calls ``aclose`` when the
    client disconnects (FastAPI propagates the disconnect as a
    ``CancelledError`` into the response generator).
    """

    def __init__(self, hub: "SSEHub", project_id: str, queue: asyncio.Queue):
        self._hub = hub
        self.project_id = project_id
        self.queue = queue
        self._closed = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hub._unsubscribe(self.project_id, self.queue)


class SSEHub:
    """Single-replica change-feed → per-project subscriber fan-out."""

    def __init__(
        self,
        feed_source: FeedSource,
        *,
        poll_interval: float = 1.0,
    ):
        self._feed_source = feed_source
        self._poll_interval = poll_interval
        # project_id → set of subscriber queues. Touched only on the
        # event loop, so no lock needed.
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # Resume state — touched only on the worker thread inside
        # ``_collect_once``, so no lock needed.
        self._continuation: Optional[str] = None
        self._since: Optional[str] = None

    # ------------------------------------------------------------------
    # Subscriber surface
    # ------------------------------------------------------------------

    async def subscribe(self, project_id: str) -> Subscription:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subs[project_id].add(queue)
        return Subscription(self, project_id, queue)

    def _unsubscribe(self, project_id: str, queue: asyncio.Queue) -> None:
        bucket = self._subs.get(project_id)
        if not bucket:
            return
        bucket.discard(queue)
        if not bucket:
            self._subs.pop(project_id, None)

    @property
    def subscriber_count(self) -> int:
        return sum(len(s) for s in self._subs.values())

    # ------------------------------------------------------------------
    # Background pump
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="sse-hub-feed")

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                items = await asyncio.to_thread(self._collect_once)
                for item in items:
                    self._dispatch(item)
            except Exception as exc:  # noqa: BLE001
                logger.exception("sse_hub.poll_failed: %s", exc)
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._poll_interval
                )
            except asyncio.TimeoutError:
                pass

    def _collect_once(self) -> list[dict]:
        """Run one poll: snapshot the resume state, drain the iterator
        returned by the feed source, and update resume state per the
        priority rules below.

        State-update rules (applied once per poll):

        - Any continuation token observed during iteration → set
          ``_continuation`` to the latest token and clear ``_since``.
          A token always trumps a timestamp.
        - Items observed but no token → clear ``_continuation`` and
          set ``_since`` to the ISO timestamp captured immediately
          *before* the feed source was invoked. Using the pre-call
          timestamp ensures the next poll catches anything written
          while the prior poll was in flight.
        - No items and no token → leave both fields untouched. The
          previously-stored resume marker is preserved so we don't
          silently advance past unseen events.

        A failure inside the feed source is contained here so the
        next poll cycle still runs. (Issue 003 will make the failure
        path return ``(items, err)`` instead of swallowing.)
        """
        poll_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state = {"continuation": self._continuation, "since": self._since}
        out: list[dict] = []
        latest_token: Optional[str] = None
        token_seen = False
        try:
            for items, token in self._feed_source(state):
                out.extend(items)
                if token is not None:
                    latest_token = token
                    token_seen = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("sse_hub.feed_iter_failed: %s", exc)

        if token_seen:
            self._continuation = latest_token
            self._since = None
        elif out:
            self._continuation = None
            self._since = poll_start_iso
        # else: state untouched
        return out

    def _dispatch(self, item: dict[str, Any]) -> None:
        pid = item.get("project_id")
        if not pid:
            return
        bucket = self._subs.get(pid)
        if not bucket:
            return
        for queue in list(bucket):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                logger.warning(
                    "sse_hub.subscriber_queue_full project_id=%s "
                    "(slow client; dropping event)",
                    pid,
                )


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_hub_instance: Optional[SSEHub] = None
_hub_lock = asyncio.Lock()


async def get_sse_hub() -> SSEHub:
    """Return (creating on first call) the per-replica ``SSEHub``.

    The singleton wires its feed source to a ``JobStore`` instance.
    The hub passes a resume-state dict on every poll; the closure
    forwards exactly one resume marker to ``subscribe_change_feed``
    in priority order:

      1. ``state["continuation"]`` truthy → pass as ``continuation=``
         (Cosmos resumes precisely where the prior poll left off).
      2. else ``state["since"]`` truthy → pass as ``start_time=``
         (resume by timestamp captured before the prior poll began).
      3. else cold start → pass ``start_time=boot_iso`` so newly-
         connected EventSource clients only see events that landed
         AFTER this replica started serving.
    """
    global _hub_instance
    if _hub_instance is not None:
        return _hub_instance
    async with _hub_lock:
        if _hub_instance is not None:
            return _hub_instance
        from backend.core.job_store import JobStore  # local to avoid import cycles

        store = JobStore()
        boot_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        def _feed(state: dict):
            continuation = state.get("continuation")
            since = state.get("since")
            if continuation:
                return store.subscribe_change_feed(continuation=continuation)
            if since:
                return store.subscribe_change_feed(start_time=since)
            return store.subscribe_change_feed(start_time=boot_iso)

        hub = SSEHub(feed_source=_feed)
        await hub.start()
        _hub_instance = hub
        return _hub_instance


async def reset_sse_hub_for_tests() -> None:
    """Test helper: stop and clear the singleton so tests can install a
    fresh feed source via ``app.dependency_overrides``."""
    global _hub_instance
    if _hub_instance is not None:
        await _hub_instance.stop()
        _hub_instance = None
