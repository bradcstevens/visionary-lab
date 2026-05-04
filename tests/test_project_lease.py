"""Tests for ``backend.core.project_lease``.

Covers the three helpers that issues 005 (project dispatcher) and 006
(POST endpoint) consume:

  - ``acquire_project_lease`` — Cosmos ETag CAS on
    ``current_project_job_id``.
  - ``release_project_lease`` — clears the field only if we still
    hold it; one ETag-conflict retry.
  - ``cascade_cancel_variation_jobs`` — sets ``cancel_requested=True``
    on every non-terminal ``regenerate_variation`` job for a project.

Storage and store dependencies are passed in as keyword arguments so
every test injects a ``MagicMock`` and asserts the wire-level Cosmos
call shape (``etag=`` + ``match_condition=``).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from azure.core import MatchConditions
from azure.cosmos import exceptions as cosmos_exceptions

from backend.core.project_lease import (
    TERMINAL_JOB_STATUSES,
    acquire_project_lease,
    cascade_cancel_variation_jobs,
    release_project_lease,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(
    *, project_id: str = "proj-1", holder: str | None = None, etag: str = "etag-1"
) -> dict:
    return {
        "id": project_id,
        "doc_type": "staging_project",
        "current_project_job_id": holder,
        "_etag": etag,
        "name": "Test project",
        "prompt": "Test prompt",
    }


def _make_storage(project: dict | None = None) -> MagicMock:
    storage = MagicMock(name="StagingStorageService")
    storage.container = MagicMock(name="container")
    storage.get_project = MagicMock(return_value=project)
    return storage


def _make_store(*, jobs: dict[str, dict] | None = None) -> MagicMock:
    store = MagicMock(name="JobStore")
    jobs = jobs or {}
    store.get_job = MagicMock(side_effect=lambda jid, pid: jobs.get(jid))
    store.list_jobs_by_project = MagicMock(return_value=list(jobs.values()))
    store.update_job = MagicMock()
    return store


# ---------------------------------------------------------------------------
# Module surface / TERMINAL constant
# ---------------------------------------------------------------------------


def test_terminal_job_statuses_set():
    assert TERMINAL_JOB_STATUSES == frozenset({"succeeded", "failed", "cancelled"})


# ---------------------------------------------------------------------------
# acquire_project_lease — 8 tests
# ---------------------------------------------------------------------------


def test_acquire_raises_when_project_not_found():
    storage = _make_storage(project=None)
    store = _make_store()

    with pytest.raises(ValueError, match="Project not found"):
        acquire_project_lease(
            storage=storage, store=store, project_id="missing", job_id="job-1"
        )


def test_acquire_succeeds_when_field_is_unset():
    project = _make_project(holder=None, etag="etag-A")
    storage = _make_storage(project=project)
    store = _make_store()

    result = acquire_project_lease(
        storage=storage, store=store, project_id="proj-1", job_id="job-1"
    )

    assert result is True
    storage.container.replace_item.assert_called_once()
    written = storage.container.replace_item.call_args.kwargs["body"]
    assert written["current_project_job_id"] == "job-1"


def test_acquire_idempotent_when_field_already_equals_job_id_skips_cas_write():
    project = _make_project(holder="job-1", etag="etag-A")
    storage = _make_storage(project=project)
    store = _make_store()

    result = acquire_project_lease(
        storage=storage, store=store, project_id="proj-1", job_id="job-1"
    )

    assert result is True
    # No CAS write needed — we already hold the lease. Saves a Cosmos
    # round-trip and avoids burning a write per re-acquire.
    storage.container.replace_item.assert_not_called()


def test_acquire_returns_false_when_foreign_holder_is_non_terminal():
    project = _make_project(holder="other-job", etag="etag-A")
    storage = _make_storage(project=project)
    store = _make_store(jobs={"other-job": {"id": "other-job", "status": "running"}})

    result = acquire_project_lease(
        storage=storage, store=store, project_id="proj-1", job_id="job-1"
    )

    assert result is False
    storage.container.replace_item.assert_not_called()


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed", "cancelled"])
def test_acquire_takes_over_when_foreign_holder_is_terminal(terminal_status):
    project = _make_project(holder="other-job", etag="etag-A")
    storage = _make_storage(project=project)
    store = _make_store(
        jobs={"other-job": {"id": "other-job", "status": terminal_status}}
    )

    result = acquire_project_lease(
        storage=storage, store=store, project_id="proj-1", job_id="job-1"
    )

    assert result is True
    storage.container.replace_item.assert_called_once()
    written = storage.container.replace_item.call_args.kwargs["body"]
    assert written["current_project_job_id"] == "job-1"


def test_acquire_returns_false_when_foreign_holder_job_is_missing():
    """Conservative default per PRD safety review: a missing holder
    job is NOT proof the work finished — it could be a transient read
    anomaly. Treat as busy and let the next acquire try again."""
    project = _make_project(holder="ghost-job", etag="etag-A")
    storage = _make_storage(project=project)
    store = _make_store(jobs={})  # ghost-job not in store

    result = acquire_project_lease(
        storage=storage, store=store, project_id="proj-1", job_id="job-1"
    )

    assert result is False
    storage.container.replace_item.assert_not_called()


def test_acquire_returns_false_when_cas_loses_race():
    """ETag conflict raised by Cosmos must surface as a clean 'busy'
    return, NOT a bubbling exception — the caller (issue 005) treats
    busy as a transient signal to abandon for redelivery."""
    project = _make_project(holder=None, etag="etag-A")
    storage = _make_storage(project=project)
    storage.container.replace_item.side_effect = (
        cosmos_exceptions.CosmosAccessConditionFailedError(message="race lost")
    )
    store = _make_store()

    result = acquire_project_lease(
        storage=storage, store=store, project_id="proj-1", job_id="job-1"
    )

    assert result is False


def test_acquire_cas_write_uses_if_not_modified_with_read_etag():
    project = _make_project(holder=None, etag="etag-XYZ")
    storage = _make_storage(project=project)
    store = _make_store()

    acquire_project_lease(
        storage=storage, store=store, project_id="proj-1", job_id="job-1"
    )

    kwargs = storage.container.replace_item.call_args.kwargs
    assert kwargs["etag"] == "etag-XYZ"
    assert kwargs["match_condition"] is MatchConditions.IfNotModified
    assert kwargs["item"] == "proj-1"


# ---------------------------------------------------------------------------
# release_project_lease — 5 tests
# ---------------------------------------------------------------------------


def test_release_returns_false_and_skips_write_when_foreign_holder():
    project = _make_project(holder="someone-else", etag="etag-A")
    storage = _make_storage(project=project)

    result = release_project_lease(
        storage=storage, project_id="proj-1", job_id="job-1"
    )

    assert result is False
    storage.container.replace_item.assert_not_called()


def test_release_returns_false_when_project_missing():
    storage = _make_storage(project=None)

    result = release_project_lease(
        storage=storage, project_id="missing", job_id="job-1"
    )

    assert result is False
    storage.container.replace_item.assert_not_called()


def test_release_clears_field_when_self_owned_no_race():
    project = _make_project(holder="job-1", etag="etag-A")
    storage = _make_storage(project=project)

    result = release_project_lease(
        storage=storage, project_id="proj-1", job_id="job-1"
    )

    assert result is True
    storage.container.replace_item.assert_called_once()
    body = storage.container.replace_item.call_args.kwargs["body"]
    assert body["current_project_job_id"] is None


def test_release_retries_on_etag_conflict_when_still_self_owned_after_reread():
    """An unrelated mutation may bump the project's ETag during the
    run. If we still own the lease after the re-read, the retry
    succeeds and the field is cleared."""
    initial = _make_project(holder="job-1", etag="etag-A")
    fresh = _make_project(holder="job-1", etag="etag-B")
    storage = _make_storage(project=initial)
    storage.get_project.side_effect = [initial, fresh]

    storage.container.replace_item.side_effect = [
        cosmos_exceptions.CosmosAccessConditionFailedError(message="race"),
        MagicMock(),  # second call (retry) succeeds
    ]

    result = release_project_lease(
        storage=storage, project_id="proj-1", job_id="job-1"
    )

    assert result is True
    assert storage.container.replace_item.call_count == 2
    second_body = storage.container.replace_item.call_args_list[1].kwargs["body"]
    assert second_body["current_project_job_id"] is None
    # Retry uses the FRESH etag, not the stale one.
    assert storage.container.replace_item.call_args_list[1].kwargs["etag"] == "etag-B"


def test_release_no_ops_on_etag_conflict_when_foreign_holder_after_reread():
    """If the re-read shows a foreign holder, someone else took the
    lease over during our run. Leave their ownership untouched."""
    initial = _make_project(holder="job-1", etag="etag-A")
    fresh = _make_project(holder="someone-else", etag="etag-B")
    storage = _make_storage(project=initial)
    storage.get_project.side_effect = [initial, fresh]

    storage.container.replace_item.side_effect = (
        cosmos_exceptions.CosmosAccessConditionFailedError(message="race")
    )

    result = release_project_lease(
        storage=storage, project_id="proj-1", job_id="job-1"
    )

    assert result is False
    # Only the first (failed) attempt — no second CAS write because
    # the foreign holder check short-circuits the retry.
    assert storage.container.replace_item.call_count == 1


def test_release_returns_false_when_second_cas_fails_after_retry():
    """Two ETag conflicts in a row means the system is contended
    enough that giving up is better than spinning. The next
    dispatcher's acquire-takeover path will reclaim the lease."""
    initial = _make_project(holder="job-1", etag="etag-A")
    fresh = _make_project(holder="job-1", etag="etag-B")
    storage = _make_storage(project=initial)
    storage.get_project.side_effect = [initial, fresh]

    storage.container.replace_item.side_effect = (
        cosmos_exceptions.CosmosAccessConditionFailedError(message="race")
    )

    result = release_project_lease(
        storage=storage, project_id="proj-1", job_id="job-1"
    )

    assert result is False
    assert storage.container.replace_item.call_count == 2  # one + one retry, both failed


# ---------------------------------------------------------------------------
# cascade_cancel_variation_jobs — 5 tests
# ---------------------------------------------------------------------------


def test_cascade_cancel_returns_zero_with_no_jobs():
    store = _make_store(jobs={})

    count = cascade_cancel_variation_jobs(store=store, project_id="proj-1")

    assert count == 0
    store.update_job.assert_not_called()


def test_cascade_cancel_sets_cancel_requested_only_not_status():
    """Per PRD §regenerate_all=true: cancellations propagate through
    the existing ``is_cancelled()`` polling path. The worker owns the
    terminal status transition (it polls ``cancel_requested`` and
    transitions status itself). Setting ``status="cancelled"`` here
    would race with concurrent worker progress writes and could
    overwrite a ``status="succeeded"`` doc that flipped between our
    list query and our update."""
    store = _make_store(
        jobs={
            "var-1": {
                "id": "var-1",
                "kind": "regenerate_variation",
                "status": "running",
            }
        }
    )

    count = cascade_cancel_variation_jobs(store=store, project_id="proj-1")

    assert count == 1
    store.update_job.assert_called_once_with(
        "var-1", "proj-1", cancel_requested=True
    )


def test_cascade_cancel_skips_non_regenerate_variation_kinds():
    store = _make_store(
        jobs={
            "var-1": {
                "id": "var-1",
                "kind": "regenerate_variation",
                "status": "running",
            },
            "proj-job-1": {
                "id": "proj-job-1",
                "kind": "generate_project",
                "status": "running",
            },
            "other-1": {
                "id": "other-1",
                "kind": "some_future_kind",
                "status": "pending",
            },
        }
    )

    count = cascade_cancel_variation_jobs(store=store, project_id="proj-1")

    assert count == 1
    store.update_job.assert_called_once_with(
        "var-1", "proj-1", cancel_requested=True
    )


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed", "cancelled"])
def test_cascade_cancel_skips_terminal_statuses(terminal_status):
    store = _make_store(
        jobs={
            "var-1": {
                "id": "var-1",
                "kind": "regenerate_variation",
                "status": terminal_status,
            }
        }
    )

    count = cascade_cancel_variation_jobs(store=store, project_id="proj-1")

    assert count == 0
    store.update_job.assert_not_called()


def test_cascade_cancel_returns_count_of_jobs_touched():
    store = _make_store(
        jobs={
            "var-1": {"id": "var-1", "kind": "regenerate_variation", "status": "pending"},
            "var-2": {"id": "var-2", "kind": "regenerate_variation", "status": "running"},
            "var-3": {"id": "var-3", "kind": "regenerate_variation", "status": "succeeded"},
            "var-4": {"id": "var-4", "kind": "regenerate_variation", "status": "running"},
        }
    )

    count = cascade_cancel_variation_jobs(store=store, project_id="proj-1")

    assert count == 3
    assert store.update_job.call_count == 3
    cancelled_ids = {call.args[0] for call in store.update_job.call_args_list}
    assert cancelled_ids == {"var-1", "var-2", "var-4"}
