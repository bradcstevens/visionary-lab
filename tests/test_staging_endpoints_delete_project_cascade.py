"""Tests for the project-deletion cascade added in issue 007 of the
image-pipeline-and-project-ux-overhaul PRD.

When DELETE /projects/{id} runs, every non-terminal job for that
project must be marked ``cancel_requested=True`` via ``JobStore``
before the project document and its blobs are deleted. The
``JobWorker`` (issue 003) observes the flag at the next safe point
and transitions each job to ``cancelled``. Terminal jobs are left
alone — flipping the flag would be a no-op the worker never sees.

Best-effort: ``JobStore`` failures must NOT block the project delete.
The Cosmos document and blob cleanup happen regardless. A leaked
``cancel_requested`` write is recoverable on the next deploy; a
failed delete is a UX bug the user has to retry.

GET /jobs already returns 404 for an unknown project (issue 004
endpoint), which satisfies the AC bullet "Deleted project no longer
surfaces in GET /jobs for that id" without further work here.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def cascade_deps(app, mock_staging_deps):
    """Stack on ``mock_staging_deps`` to also override ``get_job_store``
    so the delete endpoint exercises the cascade against an in-memory
    JobStore mock instead of real Cosmos.
    """
    from backend.api.endpoints import staging as staging_module

    store = MagicMock(name="JobStore")
    store.list_jobs_by_project.return_value = []

    app.dependency_overrides[staging_module.get_job_store] = lambda: store
    try:
        yield {**mock_staging_deps, "store": store}
    finally:
        app.dependency_overrides.pop(staging_module.get_job_store, None)


def _project_doc() -> dict:
    return {"id": "proj-cascade", "name": "Cascade", "rooms": []}


# ---------------------------------------------------------------------------
# Tracer bullet: non-terminal jobs are flipped before delete completes
# ---------------------------------------------------------------------------


def test_delete_project_marks_non_terminal_jobs_cancel_requested(
    client, cascade_deps
):
    container = cascade_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cascade_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": "proj-cascade:r1:v1:0", "project_id": "proj-cascade", "status": "pending"},
        {"id": "proj-cascade:r1:v2:0", "project_id": "proj-cascade", "status": "running"},
    ]

    response = client.delete("/api/v1/staging/projects/proj-cascade")

    assert response.status_code == 200, response.text
    # Each non-terminal job got cancel_requested=True.
    assert store.update_job.call_count == 2
    update_calls = sorted(
        (c.args[0], c.kwargs.get("cancel_requested"))
        for c in store.update_job.call_args_list
    )
    assert update_calls == [
        ("proj-cascade:r1:v1:0", True),
        ("proj-cascade:r1:v2:0", True),
    ]
    # Project doc was also deleted (cascade did not short-circuit
    # the existing flow).
    container.delete_item.assert_called_once()


# ---------------------------------------------------------------------------
# Terminal jobs are skipped — flipping cancel_requested would be a
# no-op the worker never sees, and we don't want to touch finished
# records.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed", "cancelled"])
def test_delete_project_skips_terminal_jobs(client, cascade_deps, terminal_status):
    container = cascade_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cascade_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": "proj-cascade:r1:v1:0", "project_id": "proj-cascade", "status": terminal_status},
    ]

    response = client.delete("/api/v1/staging/projects/proj-cascade")

    assert response.status_code == 200
    store.update_job.assert_not_called()


def test_delete_project_partitions_terminal_and_non_terminal(
    client, cascade_deps
):
    """Mixed list: only the non-terminal jobs get flipped."""
    container = cascade_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cascade_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": "proj-cascade:r1:v1:0", "project_id": "proj-cascade", "status": "pending"},
        {"id": "proj-cascade:r1:v2:0", "project_id": "proj-cascade", "status": "succeeded"},
        {"id": "proj-cascade:r1:v3:0", "project_id": "proj-cascade", "status": "running"},
        {"id": "proj-cascade:r1:v4:0", "project_id": "proj-cascade", "status": "failed"},
    ]

    response = client.delete("/api/v1/staging/projects/proj-cascade")

    assert response.status_code == 200
    flipped = sorted(c.args[0] for c in store.update_job.call_args_list)
    assert flipped == ["proj-cascade:r1:v1:0", "proj-cascade:r1:v3:0"]


# ---------------------------------------------------------------------------
# Best-effort: a JobStore failure must NOT block the project delete.
# ---------------------------------------------------------------------------


def test_delete_project_proceeds_when_list_jobs_raises(client, cascade_deps):
    container = cascade_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cascade_deps["store"]
    store.list_jobs_by_project.side_effect = RuntimeError("cosmos transient")

    response = client.delete("/api/v1/staging/projects/proj-cascade")

    assert response.status_code == 200
    container.delete_item.assert_called_once()
    store.update_job.assert_not_called()


def test_delete_project_proceeds_when_update_job_raises(client, cascade_deps):
    container = cascade_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cascade_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": "proj-cascade:r1:v1:0", "project_id": "proj-cascade", "status": "pending"},
        {"id": "proj-cascade:r1:v2:0", "project_id": "proj-cascade", "status": "running"},
    ]
    # First update succeeds, second blows up — delete must still go
    # through and the first flip must still take effect.
    store.update_job.side_effect = [None, RuntimeError("conflict")]

    response = client.delete("/api/v1/staging/projects/proj-cascade")

    assert response.status_code == 200
    assert store.update_job.call_count == 2
    container.delete_item.assert_called_once()


# ---------------------------------------------------------------------------
# Backward compat: when FEATURE_ASYNC_QUEUE is off, the legacy
# delete path runs unchanged (no JobStore traffic at all).
# ---------------------------------------------------------------------------


def test_delete_project_skips_cascade_when_feature_flag_disabled(
    client, cascade_deps, monkeypatch
):
    from backend.core import config as config_module

    monkeypatch.setattr(config_module.settings, "FEATURE_ASYNC_QUEUE", False)
    container = cascade_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cascade_deps["store"]

    response = client.delete("/api/v1/staging/projects/proj-cascade")

    assert response.status_code == 200
    store.list_jobs_by_project.assert_not_called()
    store.update_job.assert_not_called()


# ---------------------------------------------------------------------------
# Unknown project still 404s (cascade is post-fetch, must not mask the
# existing not-found behavior).
# ---------------------------------------------------------------------------


def test_delete_project_unknown_returns_404_no_cascade(client, cascade_deps):
    container = cascade_deps["container"]
    container.read_item.return_value = None
    store = cascade_deps["store"]

    response = client.delete("/api/v1/staging/projects/missing")

    assert response.status_code == 404
    store.list_jobs_by_project.assert_not_called()
    store.update_job.assert_not_called()
