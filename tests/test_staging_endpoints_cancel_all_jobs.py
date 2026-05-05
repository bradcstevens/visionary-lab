"""Tests for ``DELETE /staging/projects/{project_id}/jobs`` — the
cancel-all endpoint shipped in issue 005 of the
active-and-queued-jobs-ux-redesign PRD.

The endpoint is the user's "give up — free the queue" escape hatch
when generation has stalled past the front-end's 120-second hard
threshold. Reuses ``_cascade_cancel_project_jobs`` so cancelling
behavior matches ``DELETE /projects/{id}`` exactly.

PRD AC:

- 202 with ``{status: "accepted", cancelled_count, project_id}``.
- Idempotent: terminal-only project => ``cancelled_count: 0``.
- 404 when project missing.
- 503 when ``FEATURE_ASYNC_QUEUE`` off.
- Best-effort cascade: a single ``update_job`` failure must NOT
  abort the loop; the response reflects the count of *successful*
  flips.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


PROJECT_ID = "proj-cancel-all"


def _project_doc() -> dict:
    return {"id": PROJECT_ID, "name": "Cancel-all", "rooms": []}


@pytest.fixture
def cancel_all_deps(app, mock_staging_deps):
    """Stack on ``mock_staging_deps`` to also override ``get_job_store``
    so the cancel-all endpoint exercises the cascade against an
    in-memory ``JobStore`` mock instead of real Cosmos.
    """
    from backend.api.endpoints import staging as staging_module

    store = MagicMock(name="JobStore")
    store.list_jobs_by_project.return_value = []

    app.dependency_overrides[staging_module.get_job_store] = lambda: store
    try:
        yield {**mock_staging_deps, "store": store}
    finally:
        app.dependency_overrides.pop(staging_module.get_job_store, None)


# ---------------------------------------------------------------------------
# Tracer bullet: happy path returns 202 with cancelled_count and project_id
# ---------------------------------------------------------------------------


def test_cancel_all_returns_202_with_cancelled_count_and_project_id(
    client, cancel_all_deps,
):
    container = cancel_all_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cancel_all_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": f"{PROJECT_ID}:r1:v1:0", "project_id": PROJECT_ID, "status": "pending"},
        {"id": f"{PROJECT_ID}:r1:v2:0", "project_id": PROJECT_ID, "status": "running"},
    ]

    response = client.delete(f"/api/v1/staging/projects/{PROJECT_ID}/jobs")

    assert response.status_code == 202, response.text
    body = response.json()
    assert body == {
        "status": "accepted",
        "cancelled_count": 2,
        "project_id": PROJECT_ID,
    }
    # Each non-terminal job got cancel_requested=True.
    assert store.update_job.call_count == 2
    flipped = sorted(
        (c.args[0], c.kwargs.get("cancel_requested"))
        for c in store.update_job.call_args_list
    )
    assert flipped == [
        (f"{PROJECT_ID}:r1:v1:0", True),
        (f"{PROJECT_ID}:r1:v2:0", True),
    ]


# ---------------------------------------------------------------------------
# Idempotent: terminal-only project => cancelled_count=0 (no error)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed", "cancelled"])
def test_cancel_all_idempotent_when_all_jobs_terminal(
    client, cancel_all_deps, terminal_status,
):
    container = cancel_all_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cancel_all_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": f"{PROJECT_ID}:r1:v1:0", "project_id": PROJECT_ID, "status": terminal_status},
    ]

    response = client.delete(f"/api/v1/staging/projects/{PROJECT_ID}/jobs")

    assert response.status_code == 202
    assert response.json() == {
        "status": "accepted",
        "cancelled_count": 0,
        "project_id": PROJECT_ID,
    }
    store.update_job.assert_not_called()


def test_cancel_all_idempotent_when_no_jobs_exist(client, cancel_all_deps):
    """A project that never enqueued a job still answers 202 with
    cancelled_count=0 — no 404, no 400."""
    container = cancel_all_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cancel_all_deps["store"]
    store.list_jobs_by_project.return_value = []

    response = client.delete(f"/api/v1/staging/projects/{PROJECT_ID}/jobs")

    assert response.status_code == 202
    assert response.json()["cancelled_count"] == 0
    store.update_job.assert_not_called()


def test_cancel_all_partitions_terminal_and_non_terminal(
    client, cancel_all_deps,
):
    """Mixed list: only the non-terminal jobs get flipped, count
    reflects only those."""
    container = cancel_all_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cancel_all_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": f"{PROJECT_ID}:r1:v1:0", "project_id": PROJECT_ID, "status": "pending"},
        {"id": f"{PROJECT_ID}:r1:v2:0", "project_id": PROJECT_ID, "status": "succeeded"},
        {"id": f"{PROJECT_ID}:r1:v3:0", "project_id": PROJECT_ID, "status": "running"},
        {"id": f"{PROJECT_ID}:r1:v4:0", "project_id": PROJECT_ID, "status": "failed"},
    ]

    response = client.delete(f"/api/v1/staging/projects/{PROJECT_ID}/jobs")

    assert response.status_code == 202
    assert response.json()["cancelled_count"] == 2
    flipped = sorted(c.args[0] for c in store.update_job.call_args_list)
    assert flipped == [f"{PROJECT_ID}:r1:v1:0", f"{PROJECT_ID}:r1:v3:0"]


# ---------------------------------------------------------------------------
# 404 when project missing — the endpoint must validate project
# existence BEFORE running the cascade so a typo'd id doesn't
# silently no-op. Mirrors DELETE /projects/{id} contract.
# ---------------------------------------------------------------------------


def test_cancel_all_404_when_project_missing(client, cancel_all_deps):
    container = cancel_all_deps["container"]
    container.read_item.return_value = None
    store = cancel_all_deps["store"]

    response = client.delete("/api/v1/staging/projects/nonexistent/jobs")

    assert response.status_code == 404
    store.list_jobs_by_project.assert_not_called()
    store.update_job.assert_not_called()


# ---------------------------------------------------------------------------
# Feature flag — endpoint is gated behind FEATURE_ASYNC_QUEUE so a
# misconfigured production deploy fails loud.
# ---------------------------------------------------------------------------


def test_cancel_all_503_when_feature_flag_off(
    client, cancel_all_deps, monkeypatch,
):
    from backend.core import config as config_module

    monkeypatch.setattr(config_module.settings, "FEATURE_ASYNC_QUEUE", False)
    container = cancel_all_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cancel_all_deps["store"]

    response = client.delete(f"/api/v1/staging/projects/{PROJECT_ID}/jobs")

    assert response.status_code == 503
    store.list_jobs_by_project.assert_not_called()
    store.update_job.assert_not_called()


# ---------------------------------------------------------------------------
# Best-effort: a single update_job failure does NOT abort the cascade.
# The response's cancelled_count reflects only the SUCCESSFUL flips
# (rubber-duck finding: the count is "what the worker will actually
# observe", not "how many we tried to flip").
# ---------------------------------------------------------------------------


def test_cancel_all_continues_when_one_update_job_raises(
    client, cancel_all_deps,
):
    container = cancel_all_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cancel_all_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": f"{PROJECT_ID}:r1:v1:0", "project_id": PROJECT_ID, "status": "pending"},
        {"id": f"{PROJECT_ID}:r1:v2:0", "project_id": PROJECT_ID, "status": "running"},
        {"id": f"{PROJECT_ID}:r1:v3:0", "project_id": PROJECT_ID, "status": "pending"},
    ]
    # Middle update raises — surrounding two succeed.
    store.update_job.side_effect = [None, RuntimeError("transient cosmos"), None]

    response = client.delete(f"/api/v1/staging/projects/{PROJECT_ID}/jobs")

    assert response.status_code == 202
    body = response.json()
    # 2 successful flips, not 3 (the middle one raised).
    assert body["cancelled_count"] == 2
    assert store.update_job.call_count == 3


def test_cancel_all_returns_zero_when_list_jobs_raises(client, cancel_all_deps):
    """Best-effort: ``list_jobs_by_project`` failure gracefully
    yields cancelled_count=0. No 5xx — the user clicked a "cancel
    everything" button and we shouldn't surface the upstream error
    as a hard failure."""
    container = cancel_all_deps["container"]
    container.read_item.return_value = _project_doc()
    store = cancel_all_deps["store"]
    store.list_jobs_by_project.side_effect = RuntimeError("transient cosmos")

    response = client.delete(f"/api/v1/staging/projects/{PROJECT_ID}/jobs")

    assert response.status_code == 202
    assert response.json()["cancelled_count"] == 0
    store.update_job.assert_not_called()
