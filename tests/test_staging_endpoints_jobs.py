"""Tests for the async-queue REST surface added in issue 004 of the
image-pipeline-and-project-ux-overhaul PRD:

- POST   /api/v1/staging/projects/{project_id}/jobs/regenerate
- GET    /api/v1/staging/projects/{project_id}/jobs
- DELETE /api/v1/staging/jobs/{job_id}

The endpoints sit on top of the deep modules shipped in 002:
``JobStore`` (Cosmos persistence) + ``JobQueue`` (Storage Queue
producer/consumer). Both are mocked at the FastAPI dependency boundary
so these tests exercise endpoint logic only — JobStore/JobQueue
behavior is pinned by ``test_job_store.py`` / ``test_job_queue.py``.

Acceptance contract (issue 004):

- All three endpoints implemented.
- ``FEATURE_ASYNC_QUEUE`` defaults true in dev/staging.
- Regenerate produces deterministic job ids and is idempotent on retry.
- Cancel returns 202 even if the job has already completed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _project_with_two_variations() -> dict:
    """Tracer-bullet project: one room, two variations. Tests can filter
    by room_id / variation_id and assert the resulting job_ids match
    the deterministic-id contract.
    """
    return {
        "id": "proj-jobs",
        "name": "Jobs API project",
        "prompt": "warm modern with greenery",
        "status": "completed",
        "rooms": [
            {
                "id": "room-A",
                "label": "Living Room",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj-jobs/originals/a.png",
                "status": "completed",
                "variations": [
                    {"id": "var-A1", "status": "completed",
                     "image_url": "https://acct/.../a1.png"},
                    {"id": "var-A2", "status": "failed",
                     "error": "previous error"},
                ],
            }
        ],
        "settings": {"variations_per_room": 2, "model": "gpt-image-2",
                     "quality": "high", "size": "auto"},
    }


@pytest.fixture
def jobs_deps(app, mock_staging_deps):
    """Stack on top of ``mock_staging_deps`` to also override
    ``get_job_store`` and ``get_job_queue`` via FastAPI's
    ``dependency_overrides`` so the endpoint code talks to in-memory
    mocks instead of Cosmos / Storage Queue (the real factories construct
    SDK clients at call time).

    Yields a dict with ``container`` (Cosmos staging-projects),
    ``store`` (mocked JobStore), and ``queue`` (mocked JobQueue).
    """
    from backend.api.endpoints import staging as staging_module

    store = MagicMock(name="JobStore")
    queue = MagicMock(name="JobQueue")

    def _create(**kw):
        project_id = kw["project_id"]
        room_id = kw["room_id"]
        variation_id = kw["variation_id"]
        revision = kw["revision"]
        return {
            "id": f"{project_id}:{room_id}:{variation_id}:{revision}",
            "project_id": project_id,
            "room_id": room_id,
            "variation_id": variation_id,
            "revision": revision,
            "kind": kw["kind"],
            "status": "pending",
            "progress": 0,
            "phase": None,
            "attempts": 0,
            "payload": kw["payload"],
            "result": None,
            "error": None,
        }
    store.create_job.side_effect = _create
    store.list_jobs_by_project.return_value = []

    app.dependency_overrides[staging_module.get_job_store] = lambda: store
    app.dependency_overrides[staging_module.get_job_queue] = lambda: queue
    try:
        yield {**mock_staging_deps, "store": store, "queue": queue}
    finally:
        app.dependency_overrides.pop(staging_module.get_job_store, None)
        app.dependency_overrides.pop(staging_module.get_job_queue, None)


# ---------------------------------------------------------------------------
# Tracer bullet: POST /jobs/regenerate happy path enqueues all variations
# ---------------------------------------------------------------------------


def test_regenerate_enqueues_one_job_per_variation_and_returns_ids(
    client, jobs_deps
):
    """POST /projects/{id}/jobs/regenerate with no filter enqueues a
    job for every variation in the project, returns the deterministic
    ids in body, and calls JobQueue.enqueue once per job.
    """
    container = jobs_deps["container"]
    container.read_item.return_value = _project_with_two_variations()

    response = client.post(
        "/api/v1/staging/projects/proj-jobs/jobs/regenerate", json={}
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body == {
        "job_ids": [
            "proj-jobs:room-A:var-A1:0",
            "proj-jobs:room-A:var-A2:0",
        ]
    }

    store = jobs_deps["store"]
    queue = jobs_deps["queue"]
    assert store.create_job.call_count == 2
    assert queue.enqueue.call_count == 2
    # Every enqueue uses the same id we returned in the response — no
    # rewriting between create_job and enqueue.
    enq_ids = sorted(c.kwargs["job_id"] for c in queue.enqueue.call_args_list)
    assert enq_ids == ["proj-jobs:room-A:var-A1:0", "proj-jobs:room-A:var-A2:0"]
    # All enqueues carry the project_id pointer for partition-routed
    # change-feed consumers.
    assert all(
        c.kwargs["project_id"] == "proj-jobs"
        for c in queue.enqueue.call_args_list
    )


# ---------------------------------------------------------------------------
# POST /jobs/regenerate — filter by room_ids / variation_ids
# ---------------------------------------------------------------------------


def test_regenerate_with_room_filter_only_enqueues_matching_rooms(
    client, jobs_deps
):
    container = jobs_deps["container"]
    project = _project_with_two_variations()
    project["rooms"].append({
        "id": "room-B",
        "label": "Kitchen",
        "original_image_url": "https://x/b.png",
        "status": "completed",
        "variations": [{"id": "var-B1", "status": "completed",
                        "image_url": "https://x/b1.png"}],
    })
    container.read_item.return_value = project

    response = client.post(
        "/api/v1/staging/projects/proj-jobs/jobs/regenerate",
        json={"room_ids": ["room-B"]},
    )
    assert response.status_code == 202
    assert response.json() == {"job_ids": ["proj-jobs:room-B:var-B1:0"]}


def test_regenerate_with_variation_filter_only_enqueues_matching_variations(
    client, jobs_deps
):
    container = jobs_deps["container"]
    container.read_item.return_value = _project_with_two_variations()

    response = client.post(
        "/api/v1/staging/projects/proj-jobs/jobs/regenerate",
        json={"variation_ids": ["var-A2"]},
    )
    assert response.status_code == 202
    assert response.json() == {"job_ids": ["proj-jobs:room-A:var-A2:0"]}


# ---------------------------------------------------------------------------
# POST /jobs/regenerate — idempotency on retry
# ---------------------------------------------------------------------------


def test_regenerate_retry_with_pending_job_returns_same_id(client, jobs_deps):
    """A second regenerate call while the first job is still pending
    must return the SAME job id (revision unchanged). This is the
    "idempotent on retry" PRD AC."""
    container = jobs_deps["container"]
    container.read_item.return_value = _project_with_two_variations()

    store = jobs_deps["store"]
    # Simulate the first call having already produced a pending job
    # at revision 0 for var-A1.
    store.list_jobs_by_project.return_value = [
        {
            "id": "proj-jobs:room-A:var-A1:0",
            "project_id": "proj-jobs",
            "room_id": "room-A",
            "variation_id": "var-A1",
            "revision": 0,
            "status": "pending",
        },
    ]

    response = client.post(
        "/api/v1/staging/projects/proj-jobs/jobs/regenerate",
        json={"variation_ids": ["var-A1"]},
    )
    assert response.status_code == 202
    # Same id — the second caller observed the in-flight job and got its id.
    assert response.json() == {"job_ids": ["proj-jobs:room-A:var-A1:0"]}


def test_regenerate_after_terminal_increments_revision(client, jobs_deps):
    """When the latest job for a variation is terminal (succeeded /
    failed / cancelled), the next regenerate request bumps revision
    by 1 — that's a "do it again" semantic, not an idempotent retry.
    """
    container = jobs_deps["container"]
    container.read_item.return_value = _project_with_two_variations()

    store = jobs_deps["store"]
    store.list_jobs_by_project.return_value = [
        {
            "id": "proj-jobs:room-A:var-A1:0",
            "room_id": "room-A",
            "variation_id": "var-A1",
            "revision": 0,
            "status": "succeeded",
        },
    ]

    response = client.post(
        "/api/v1/staging/projects/proj-jobs/jobs/regenerate",
        json={"variation_ids": ["var-A1"]},
    )
    assert response.status_code == 202
    assert response.json() == {"job_ids": ["proj-jobs:room-A:var-A1:1"]}


# ---------------------------------------------------------------------------
# POST /jobs/regenerate — error paths
# ---------------------------------------------------------------------------


def test_regenerate_unknown_project_returns_404(client, jobs_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    container = jobs_deps["container"]
    container.read_item.side_effect = CosmosResourceNotFoundError(
        message="not found"
    )

    response = client.post(
        "/api/v1/staging/projects/missing/jobs/regenerate", json={}
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /jobs — list endpoint
# ---------------------------------------------------------------------------


def test_list_jobs_returns_status_and_progress(client, jobs_deps):
    container = jobs_deps["container"]
    container.read_item.return_value = _project_with_two_variations()

    store = jobs_deps["store"]
    store.list_jobs_by_project.return_value = [
        {
            "id": "proj-jobs:room-A:var-A1:0",
            "project_id": "proj-jobs",
            "room_id": "room-A",
            "variation_id": "var-A1",
            "revision": 0,
            "kind": "regenerate_variation",
            "status": "running",
            "progress": 45,
            "phase": "generating",
            "attempts": 1,
            "error": None,
            "result": None,
            "created_at": "2026-05-01T00:00:00Z",
            "updated_at": "2026-05-01T00:00:30Z",
        },
        {
            "id": "proj-jobs:room-A:var-A2:0",
            "project_id": "proj-jobs",
            "room_id": "room-A",
            "variation_id": "var-A2",
            "revision": 0,
            "kind": "regenerate_variation",
            "status": "succeeded",
            "progress": 100,
            "phase": "finalizing",
            "attempts": 1,
            "result": {"image_url": "https://x/a2.png"},
        },
    ]

    response = client.get("/api/v1/staging/projects/proj-jobs/jobs")
    assert response.status_code == 200
    body = response.json()
    assert "jobs" in body
    assert len(body["jobs"]) == 2
    statuses = [j["status"] for j in body["jobs"]]
    progresses = [j["progress"] for j in body["jobs"]]
    assert statuses == ["running", "succeeded"]
    assert progresses == [45, 100]
    # Phase + kind surfaced for the frontend's per-image bar.
    assert body["jobs"][0]["phase"] == "generating"
    assert body["jobs"][0]["kind"] == "regenerate_variation"


def test_list_jobs_unknown_project_returns_404(client, jobs_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    container = jobs_deps["container"]
    container.read_item.side_effect = CosmosResourceNotFoundError(
        message="not found"
    )

    response = client.get("/api/v1/staging/projects/missing/jobs")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id} — cancel endpoint
# ---------------------------------------------------------------------------


def test_cancel_pending_job_sets_cancel_requested_and_returns_202(
    client, jobs_deps
):
    store = jobs_deps["store"]
    store.get_job.return_value = {
        "id": "proj-jobs:room-A:var-A1:0",
        "project_id": "proj-jobs",
        "status": "pending",
    }

    response = client.delete("/api/v1/staging/jobs/proj-jobs:room-A:var-A1:0")
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "proj-jobs:room-A:var-A1:0"
    assert body["already_terminal"] is False

    # JobStore was patched with cancel_requested=True via update_job.
    store.update_job.assert_called_once_with(
        "proj-jobs:room-A:var-A1:0", "proj-jobs", cancel_requested=True
    )


def test_cancel_running_job_sets_cancel_requested_and_returns_202(
    client, jobs_deps
):
    """Mid-dispatch: worker is running the job. Cancel still flips the
    flag — JobWorker's is_cancelled probe re-reads from JobStore each
    tick (per issue 003) and will raise JobCancelled at the next probe.
    """
    store = jobs_deps["store"]
    store.get_job.return_value = {
        "id": "proj-jobs:room-A:var-A1:0",
        "project_id": "proj-jobs",
        "status": "running",
    }

    response = client.delete("/api/v1/staging/jobs/proj-jobs:room-A:var-A1:0")
    assert response.status_code == 202
    store.update_job.assert_called_once_with(
        "proj-jobs:room-A:var-A1:0", "proj-jobs", cancel_requested=True
    )


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed", "cancelled"])
def test_cancel_terminal_job_returns_202_without_modifying(
    client, jobs_deps, terminal_status
):
    """PRD AC: returns 202 even if the job has already completed.
    The endpoint must NOT call update_job in that case — there's
    nothing to cancel and the doc is immutable from the worker's
    point of view.
    """
    store = jobs_deps["store"]
    store.get_job.return_value = {
        "id": "proj-jobs:room-A:var-A1:0",
        "project_id": "proj-jobs",
        "status": terminal_status,
    }

    response = client.delete("/api/v1/staging/jobs/proj-jobs:room-A:var-A1:0")
    assert response.status_code == 202
    assert response.json()["already_terminal"] is True
    store.update_job.assert_not_called()


def test_cancel_unknown_job_returns_404(client, jobs_deps):
    store = jobs_deps["store"]
    store.get_job.return_value = None

    response = client.delete("/api/v1/staging/jobs/proj-jobs:room-A:var-A1:0")
    assert response.status_code == 404
    store.update_job.assert_not_called()


def test_cancel_malformed_job_id_returns_400(client, jobs_deps):
    """Deterministic id format is ``{p}:{r}:{v}:{rev}`` — fewer than 4
    colon-separated segments is a client error, not a 404 (the path
    parameter never resolved to a real partition key).
    """
    response = client.delete("/api/v1/staging/jobs/not-a-real-id")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# FEATURE_ASYNC_QUEUE flag — disabled path
# ---------------------------------------------------------------------------


def test_endpoints_return_503_when_feature_flag_off(
    client, jobs_deps, monkeypatch
):
    """When ``FEATURE_ASYNC_QUEUE`` is false (production not yet flipped
    over), all three new endpoints fail loud with 503 instead of
    silently queueing into a worker that may not be deployed.
    """
    from backend.core import config as cfg
    monkeypatch.setattr(cfg.settings, "FEATURE_ASYNC_QUEUE", False)

    r1 = client.post(
        "/api/v1/staging/projects/proj-jobs/jobs/regenerate", json={}
    )
    r2 = client.get("/api/v1/staging/projects/proj-jobs/jobs")
    r3 = client.delete("/api/v1/staging/jobs/proj-jobs:room-A:var-A1:0")

    for r in (r1, r2, r3):
        assert r.status_code == 503, r.text


def test_feature_flag_defaults_true():
    """PRD § Feature flags: defaults true in dev/staging from each PR.
    Pin so a future config refactor can't quietly flip the default to
    false and disable the queue path on every environment."""
    from backend.core.config import Settings
    assert Settings().FEATURE_ASYNC_QUEUE is True
