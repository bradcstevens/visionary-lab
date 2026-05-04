"""Unit tests for ``backend.core.job_queue.JobQueue``.

Public-interface contract pinned by these tests (per PRD § JobQueue +
issue 002 AC):

  - ``enqueue(job_id, project_id)`` writes a JSON message
    ``{"job_id": ..., "project_id": ...}`` to the ``imagejobs`` queue.
    Per PRD: "queue carries only a small reference (pointer)" — the
    real state lives in JobStore.

  - ``dequeue(visibility_timeout=90)`` peeks one message and returns
    it as a ``JobMessage`` exposing ``job_id``, ``project_id``,
    ``dequeue_count``, plus the underlying SDK message handle the
    queue needs to delete/update.

  - ``complete(message)`` deletes the message from the queue
    (success path).

  - ``abandon(message)`` is the failure path. If the message has been
    dequeued ``MAX_DEQUEUE_COUNT`` (3) times already, route to the
    ``imagejobs-poison`` queue and delete from the main queue.
    Otherwise, make it visible again immediately so the next replica
    can pick it up.

  - Auth uses ``DefaultAzureCredential`` (managed identity), never a
    connection string. AC: "No connection strings in code or config —
    managed identity only".

Tests use ``unittest.mock`` only.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_queue(main=None, poison=None):
    """Return a JobQueue wired to mock QueueClient instances."""
    from backend.core.job_queue import JobQueue

    return JobQueue(
        main_client=main or MagicMock(),
        poison_client=poison or MagicMock(),
    )


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------


def test_enqueue_sends_json_pointer_message_to_main_queue():
    main = MagicMock()
    queue = _make_queue(main=main)

    queue.enqueue(job_id="proj-1:room-1:var-1:0", project_id="proj-1")

    main.send_message.assert_called_once()
    body = main.send_message.call_args.kwargs.get("content") or main.send_message.call_args.args[0]
    payload = json.loads(body)
    assert payload == {"job_id": "proj-1:room-1:var-1:0", "project_id": "proj-1"}


def test_enqueue_sets_seven_day_message_ttl():
    """Per PRD § Infrastructure: 'message TTL 7 days'."""
    main = MagicMock()
    queue = _make_queue(main=main)

    queue.enqueue(job_id="j1", project_id="p1")

    kwargs = main.send_message.call_args.kwargs
    assert kwargs.get("time_to_live") == 7 * 24 * 60 * 60


# ---------------------------------------------------------------------------
# Dequeue
# ---------------------------------------------------------------------------


def test_dequeue_returns_none_when_queue_empty():
    main = MagicMock()
    main.receive_messages.return_value = iter([])
    queue = _make_queue(main=main)

    assert queue.dequeue() is None


def test_dequeue_returns_job_message_with_parsed_fields():
    main = MagicMock()
    raw = MagicMock()
    raw.content = json.dumps({"job_id": "j1", "project_id": "p1"})
    raw.dequeue_count = 1
    raw.id = "msg-id"
    raw.pop_receipt = "pop"
    main.receive_messages.return_value = iter([raw])
    queue = _make_queue(main=main)

    msg = queue.dequeue()

    assert msg is not None
    assert msg.job_id == "j1"
    assert msg.project_id == "p1"
    assert msg.dequeue_count == 1
    assert msg.raw is raw


def test_dequeue_uses_ninety_second_visibility_timeout_by_default():
    """Per PRD § Infrastructure: 'visibility timeout 90s'."""
    main = MagicMock()
    main.receive_messages.return_value = iter([])
    queue = _make_queue(main=main)

    queue.dequeue()

    kwargs = main.receive_messages.call_args.kwargs
    assert kwargs.get("visibility_timeout") == 90


# ---------------------------------------------------------------------------
# Complete (delete)
# ---------------------------------------------------------------------------


def test_complete_deletes_message_from_main_queue():
    main = MagicMock()
    queue = _make_queue(main=main)
    raw = MagicMock()
    raw.id = "m1"
    raw.pop_receipt = "pr1"
    raw.content = json.dumps({"job_id": "j", "project_id": "p"})
    raw.dequeue_count = 1
    from backend.core.job_queue import JobMessage

    queue.complete(JobMessage(job_id="j", project_id="p", dequeue_count=1, raw=raw))

    main.delete_message.assert_called_once_with(raw)


# ---------------------------------------------------------------------------
# Abandon — retry vs poison
# ---------------------------------------------------------------------------


def _msg(dequeue_count: int):
    from backend.core.job_queue import JobMessage

    raw = MagicMock()
    raw.id = "m1"
    raw.pop_receipt = "pr1"
    raw.content = json.dumps({"job_id": "j", "project_id": "p"})
    raw.dequeue_count = dequeue_count
    return JobMessage(
        job_id="j", project_id="p", dequeue_count=dequeue_count, raw=raw
    ), raw


def test_abandon_under_max_dequeue_makes_message_visible_again():
    main = MagicMock()
    poison = MagicMock()
    queue = _make_queue(main=main, poison=poison)

    msg, raw = _msg(dequeue_count=1)
    queue.abandon(msg)

    main.update_message.assert_called_once()
    args = main.update_message.call_args
    # visibility_timeout=0 → re-deliver immediately
    assert args.kwargs.get("visibility_timeout") == 0
    poison.send_message.assert_not_called()
    main.delete_message.assert_not_called()


def test_abandon_at_max_dequeue_routes_to_poison_and_deletes_from_main():
    """Per AC: 'routes a 3rd-failure message to imagejobs-poison'.

    Dequeue count = 3 means this is the 3rd attempt that just failed —
    next visibility-timeout re-delivery would be the 4th, exceeding
    the policy. So we poison NOW.
    """
    main = MagicMock()
    poison = MagicMock()
    queue = _make_queue(main=main, poison=poison)

    msg, raw = _msg(dequeue_count=3)
    queue.abandon(msg)

    poison.send_message.assert_called_once()
    body = poison.send_message.call_args.kwargs.get("content") or poison.send_message.call_args.args[0]
    payload = json.loads(body)
    assert payload == {"job_id": "j", "project_id": "p"}
    main.delete_message.assert_called_once_with(raw)
    main.update_message.assert_not_called()


def test_abandon_above_max_dequeue_also_poisons():
    """Defensive: if dequeue_count somehow exceeds 3 (race, replica
    crash mid-update), still route to poison rather than infinite-loop."""
    main = MagicMock()
    poison = MagicMock()
    queue = _make_queue(main=main, poison=poison)

    msg, raw = _msg(dequeue_count=5)
    queue.abandon(msg)

    poison.send_message.assert_called_once()
    main.delete_message.assert_called_once_with(raw)


def test_max_dequeue_count_constant_is_three():
    """Pin the policy constant — KEDA / queue-service config and the
    JobQueue must agree on the 3-attempt limit."""
    from backend.core.job_queue import MAX_DEQUEUE_COUNT

    assert MAX_DEQUEUE_COUNT == 3


# ---------------------------------------------------------------------------
# Auth — managed identity only (AC: 'No connection strings')
# ---------------------------------------------------------------------------


def test_default_construction_uses_managed_identity_no_connection_string(monkeypatch):
    """Constructing JobQueue() with no clients must build QueueClient
    instances from the storage account URL + DefaultAzureCredential.
    A connection-string code path would be a security regression."""
    from backend.core import job_queue as jq_mod

    captured = {}

    class FakeQueueClient:
        def __init__(self, account_url, queue_name, credential):
            captured.setdefault("calls", []).append(
                {"account_url": account_url, "queue_name": queue_name, "credential": credential}
            )

        def create_queue(self):
            pass

    monkeypatch.setattr(jq_mod, "QueueClient", FakeQueueClient)
    fake_cred = object()
    monkeypatch.setattr(jq_mod, "DefaultAzureCredential", lambda: fake_cred)
    monkeypatch.setattr(
        jq_mod.settings, "AZURE_STORAGE_ACCOUNT_NAME", "teststorage", raising=False
    )

    jq_mod.JobQueue()

    queue_names = {c["queue_name"] for c in captured["calls"]}
    assert queue_names == {"imagejobs", "imagejobs-poison"}
    for c in captured["calls"]:
        assert c["account_url"] == "https://teststorage.queue.core.windows.net/"
        assert c["credential"] is fake_cred


# ---------------------------------------------------------------------------
# Extend visibility — heartbeat for long-running messages (issue 001)
# ---------------------------------------------------------------------------


def test_extend_visibility_calls_update_message_with_timeout():
    """The wrapper must drive the SDK's update_message with the requested
    visibility timeout. The Azure Storage Queue SDK uses the message's
    id + pop_receipt internally when the QueueMessage object is passed
    in directly, so we don't need to forward those explicitly."""
    main = MagicMock()
    updated = MagicMock()
    updated.pop_receipt = "fresh-receipt"
    updated.next_visible_on = "2026-05-03T20:30:00Z"
    main.update_message.return_value = updated
    queue = _make_queue(main=main)

    msg, raw = _msg(dequeue_count=1)
    raw.pop_receipt = "stale-receipt"
    raw.next_visible_on = "2026-05-03T20:00:00Z"

    queue.extend_visibility(msg, 90)

    main.update_message.assert_called_once()
    args = main.update_message.call_args
    assert args.kwargs.get("visibility_timeout") == 90


def test_extend_visibility_writes_back_pop_receipt_in_place():
    """AC: the wrapper mutates ``message.raw`` in place so a subsequent
    ``complete()`` (which delegates to ``delete_message`` using
    ``message.raw``) uses the latest receipt rather than a stale one.

    Without this write-back the rubber-duck blocking #1 finding triggers:
    after a 30s heartbeat extension, complete() would 404 and the
    Storage Queue would redeliver the message, running the project
    pipeline twice.
    """
    main = MagicMock()
    updated = MagicMock()
    updated.pop_receipt = "fresh-receipt"
    updated.next_visible_on = "2026-05-03T20:30:00Z"
    main.update_message.return_value = updated
    queue = _make_queue(main=main)

    msg, raw = _msg(dequeue_count=1)
    raw.pop_receipt = "stale-receipt"
    raw.next_visible_on = "2026-05-03T20:00:00Z"

    queue.extend_visibility(msg, 90)

    assert raw.pop_receipt == "fresh-receipt"
    assert raw.next_visible_on == "2026-05-03T20:30:00Z"
    # And the JobMessage's raw still points at the same object — no
    # replacement, in-place mutation only.
    assert msg.raw is raw


def test_complete_after_extend_visibility_uses_refreshed_pop_receipt():
    """End-to-end pin for the rubber-duck blocking #1 finding: once
    extend_visibility has run, calling complete must hand the SDK a
    message whose pop_receipt is the freshly-refreshed one. Without
    the in-place write-back, the delete would 404."""
    main = MagicMock()
    updated = MagicMock()
    updated.pop_receipt = "fresh-receipt"
    updated.next_visible_on = "2026-05-03T20:30:00Z"
    main.update_message.return_value = updated
    queue = _make_queue(main=main)

    msg, raw = _msg(dequeue_count=1)
    raw.pop_receipt = "stale-receipt"

    queue.extend_visibility(msg, 90)
    queue.complete(msg)

    main.delete_message.assert_called_once_with(raw)
    # delete_message receives the raw message object; the SDK reads
    # pop_receipt off it. The mutation in extend_visibility ensures
    # this read returns the fresh receipt.
    assert raw.pop_receipt == "fresh-receipt"
