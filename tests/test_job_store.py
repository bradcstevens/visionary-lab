"""Unit tests for ``backend.core.job_store.JobStore``.

Public-interface contract pinned by these tests (per PRD § JobStore):

  - ``deterministic_job_id(project_id, room_id, variation_id, revision)``
    returns a stable colon-joined id used as the Cosmos doc id and as
    the queue payload reference.
  - ``create_job(...)`` writes a doc with the deterministic id and is
    idempotent — re-creating the same job returns the EXISTING doc and
    does NOT raise. Implementation uses ``If-None-Match: *`` and treats
    409 Conflict as a no-op.
  - ``get_job(job_id, project_id)`` returns the doc or None.
  - ``update_job(job_id, project_id, **patches)`` merges patches and
    refreshes ``updated_at``; only known fields are accepted.
  - ``list_jobs_by_project(project_id)`` returns project-scoped docs
    in created_at descending order via a partition-scoped query (no
    cross-partition).
  - ``subscribe_change_feed(start_time=None)`` returns an iterator over
    ``(items, continuation_token)`` batches; the underlying call uses
    the Cosmos change feed API.

Tests use ``unittest.mock`` only — no real Cosmos connection. Emulator-
backed coverage lives under ``tests/integration/``.
"""
from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest
from azure.cosmos import exceptions


# ---------------------------------------------------------------------------
# Deterministic id
# ---------------------------------------------------------------------------


def test_deterministic_job_id_is_colon_joined_quad():
    from backend.core.job_store import deterministic_job_id

    job_id = deterministic_job_id("proj-1", "room-2", "var-3", 7)

    assert job_id == "proj-1:room-2:var-3:7"


def test_deterministic_job_id_is_stable_for_same_inputs():
    from backend.core.job_store import deterministic_job_id

    a = deterministic_job_id("p", "r", "v", 1)
    b = deterministic_job_id("p", "r", "v", 1)

    assert a == b


# ---------------------------------------------------------------------------
# JobStore — create / idempotency
# ---------------------------------------------------------------------------


def _make_store_with_mock_container():
    from backend.core.job_store import JobStore

    container = MagicMock()
    return JobStore(container=container), container


def test_create_job_writes_doc_with_pending_status_and_zero_progress():
    store, container = _make_store_with_mock_container()
    container.create_item.side_effect = lambda body: body

    doc = store.create_job(
        project_id="proj-1",
        room_id="room-1",
        variation_id="var-1",
        revision=0,
        kind="generate",
        payload={"prompt": "hi"},
    )

    container.create_item.assert_called_once()
    written = container.create_item.call_args.kwargs["body"]
    assert written["id"] == "proj-1:room-1:var-1:0"
    assert written["project_id"] == "proj-1"
    assert written["room_id"] == "room-1"
    assert written["variation_id"] == "var-1"
    assert written["revision"] == 0
    assert written["kind"] == "generate"
    assert written["status"] == "pending"
    assert written["progress"] == 0
    assert written["attempts"] == 0
    assert written["payload"] == {"prompt": "hi"}
    assert written["result"] is None
    assert written["error"] is None
    assert "created_at" in written and "updated_at" in written
    assert doc == written


def test_create_job_is_idempotent_on_conflict():
    """Per AC: 'deterministic-id insert is idempotent'. A second create_job
    with the same deterministic id MUST NOT raise — it returns the existing
    doc as if the create had succeeded."""
    store, container = _make_store_with_mock_container()
    existing_doc = {
        "id": "proj-1:room-1:var-1:0",
        "project_id": "proj-1",
        "status": "running",  # already past pending — proves we returned the existing one
    }
    container.create_item.side_effect = exceptions.CosmosResourceExistsError(
        status_code=409, message="Conflict"
    )
    container.read_item.return_value = existing_doc

    doc = store.create_job(
        project_id="proj-1",
        room_id="room-1",
        variation_id="var-1",
        revision=0,
        kind="generate",
        payload={"prompt": "hi"},
    )

    assert doc == existing_doc
    container.read_item.assert_called_once_with(
        item="proj-1:room-1:var-1:0", partition_key="proj-1"
    )


# ---------------------------------------------------------------------------
# JobStore — get / update / list
# ---------------------------------------------------------------------------


def test_get_job_returns_none_when_missing():
    store, container = _make_store_with_mock_container()
    container.read_item.side_effect = exceptions.CosmosResourceNotFoundError(
        status_code=404, message="not found"
    )

    assert store.get_job("missing-id", "proj-1") is None


def test_get_job_returns_doc_when_present():
    store, container = _make_store_with_mock_container()
    container.read_item.return_value = {"id": "j1", "project_id": "proj-1"}

    assert store.get_job("j1", "proj-1") == {"id": "j1", "project_id": "proj-1"}
    container.read_item.assert_called_once_with(item="j1", partition_key="proj-1")


def test_update_job_merges_patches_and_refreshes_updated_at():
    store, container = _make_store_with_mock_container()
    existing = {
        "id": "j1",
        "project_id": "proj-1",
        "status": "pending",
        "progress": 0,
        "phase": None,
        "attempts": 0,
        "updated_at": "1970-01-01T00:00:00+00:00",
    }
    container.read_item.return_value = existing
    container.replace_item.side_effect = lambda item, body: body

    updated = store.update_job(
        "j1", "proj-1", status="running", progress=42, phase="generating", attempts=1
    )

    assert updated["status"] == "running"
    assert updated["progress"] == 42
    assert updated["phase"] == "generating"
    assert updated["attempts"] == 1
    assert updated["updated_at"] != "1970-01-01T00:00:00+00:00"
    container.replace_item.assert_called_once()


def test_update_job_rejects_unknown_fields():
    store, _ = _make_store_with_mock_container()

    with pytest.raises(ValueError, match="unknown"):
        store.update_job("j1", "proj-1", bogus_field="oops")


def test_update_job_raises_when_missing():
    store, container = _make_store_with_mock_container()
    container.read_item.side_effect = exceptions.CosmosResourceNotFoundError(
        status_code=404, message="not found"
    )

    with pytest.raises(LookupError):
        store.update_job("missing", "proj-1", status="running")


def test_list_jobs_by_project_uses_partition_scoped_query():
    store, container = _make_store_with_mock_container()
    docs = [
        {"id": "j2", "project_id": "p", "created_at": "2026-01-02T00:00:00+00:00"},
        {"id": "j1", "project_id": "p", "created_at": "2026-01-01T00:00:00+00:00"},
    ]
    container.query_items.return_value = iter(docs)

    result = store.list_jobs_by_project("p")

    assert result == docs
    call = container.query_items.call_args
    # Partition-scoped → cross-partition disabled, partition_key set
    assert call.kwargs.get("partition_key") == "p"
    assert call.kwargs.get("enable_cross_partition_query") in (False, None)


# ---------------------------------------------------------------------------
# JobStore — change feed
# ---------------------------------------------------------------------------


def _make_change_feed_iterator(
    items: list[dict],
    *,
    etag: Optional[str] = None,
    continuation_token: Optional[str] = None,
) -> MagicMock:
    """Build a mock change-feed iterator with controlled token sources.

    ``response_headers`` and ``continuation_token`` are set explicitly so
    the extraction-precedence rule can be pinned without MagicMock's
    auto-attr behaviour leaking truthy sentinels.
    """
    page = MagicMock()
    page.__iter__ = lambda self: iter(list(items))
    iterator = MagicMock()
    iterator.by_page.return_value = iter([page])
    iterator.response_headers = {"etag": etag} if etag is not None else {}
    iterator.continuation_token = continuation_token
    return iterator


def test_subscribe_change_feed_yields_items_and_continuation():
    store, container = _make_store_with_mock_container()
    iterator = _make_change_feed_iterator(
        [{"id": "j1"}, {"id": "j2"}], etag="abc"
    )
    container.query_items_change_feed.return_value = iterator

    batches = list(store.subscribe_change_feed())

    assert batches == [([{"id": "j1"}, {"id": "j2"}], "abc")]
    container.query_items_change_feed.assert_called_once()


# ---------------------------------------------------------------------------
# JobStore — change feed: resume-kwarg priority (issue 001)
# ---------------------------------------------------------------------------


def test_subscribe_change_feed_continuation_takes_priority():
    """When ``continuation`` is provided, ONLY it is forwarded — no
    ``start_time`` and no ``is_start_from_beginning``. Fixes the
    once-per-second ``ValueError: is_start_from_beginning and start_time
    are exclusive`` crash."""
    store, container = _make_store_with_mock_container()
    iterator = _make_change_feed_iterator([], etag="t1")
    container.query_items_change_feed.return_value = iterator

    list(store.subscribe_change_feed(start_time="2026-01-01T00:00:00Z", continuation="abc"))

    kwargs = container.query_items_change_feed.call_args.kwargs
    assert kwargs == {"continuation": "abc"}
    assert "start_time" not in kwargs
    assert "is_start_from_beginning" not in kwargs


def test_subscribe_change_feed_start_time_when_no_continuation():
    """``start_time`` only — never together with ``is_start_from_beginning``."""
    store, container = _make_store_with_mock_container()
    iterator = _make_change_feed_iterator([], etag="t2")
    container.query_items_change_feed.return_value = iterator

    list(store.subscribe_change_feed(start_time="2026-01-01T00:00:00Z"))

    kwargs = container.query_items_change_feed.call_args.kwargs
    assert kwargs == {"start_time": "2026-01-01T00:00:00Z"}
    assert "continuation" not in kwargs
    assert "is_start_from_beginning" not in kwargs


def test_subscribe_change_feed_cold_start_uses_is_start_from_beginning():
    """No resume marker → ``is_start_from_beginning=True`` alone."""
    store, container = _make_store_with_mock_container()
    iterator = _make_change_feed_iterator([], etag="t3")
    container.query_items_change_feed.return_value = iterator

    list(store.subscribe_change_feed())

    kwargs = container.query_items_change_feed.call_args.kwargs
    assert kwargs == {"is_start_from_beginning": True}
    assert "continuation" not in kwargs
    assert "start_time" not in kwargs


def test_subscribe_change_feed_continuation_extraction_precedence():
    """``response_headers['etag']`` wins; falls back to
    ``iterator.continuation_token`` when only that is set; ``None``
    when neither is available."""
    store, container = _make_store_with_mock_container()

    # 1. etag header present → wins over continuation_token attribute
    iterator = _make_change_feed_iterator(
        [{"id": "j1"}], etag="from-header", continuation_token="from-attr"
    )
    container.query_items_change_feed.return_value = iterator
    batches = list(store.subscribe_change_feed())
    assert batches == [([{"id": "j1"}], "from-header")]

    # 2. no etag header → falls back to continuation_token attribute
    iterator = _make_change_feed_iterator(
        [{"id": "j2"}], etag=None, continuation_token="from-attr"
    )
    container.query_items_change_feed.return_value = iterator
    batches = list(store.subscribe_change_feed())
    assert batches == [([{"id": "j2"}], "from-attr")]

    # 3. neither → None
    iterator = _make_change_feed_iterator([{"id": "j3"}], etag=None, continuation_token=None)
    container.query_items_change_feed.return_value = iterator
    batches = list(store.subscribe_change_feed())
    assert batches == [([{"id": "j3"}], None)]
