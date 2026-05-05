"""Tests for the async-queue ``POST /jobs/generate`` endpoint added in
issue 006 of the project-generation-async-queue-cutover PRD.

The endpoint is the producer side of the async cutover:

- Inline brief composition (~30-90s blocking) so a worker retry doesn't
  recompute prompts.
- ``regenerate_all=true`` cascade-cancels in-flight ``regenerate_variation``
  jobs BEFORE creating the new job (rubber-duck-validated point of no
  return: cancellation persists even if create/enqueue then fails).
- UUID4 hex revision so two concurrent POSTs always produce DISTINCT
  deterministic job ids — the existing 409-idempotent ``create_job``
  contract MUST NOT silently collapse the second click.
- Compensation contract: if ``queue.enqueue`` fails after ``create_job``
  succeeded, mark the doc ``status="failed"`` so the SSE feed and
  ``GET /jobs`` surface the failure (no phantom pending jobs).

All four BLOCKING rubber-duck findings (compensation, cascade-as-PNR,
empty-rooms 400, StrictBool) have explicit regression tests below.

Endpoint under test:

    POST /api/v1/staging/projects/{project_id}/jobs/generate

Mock pattern mirrors ``test_staging_endpoints_jobs.py``:
``mock_staging_deps`` (Cosmos for storage) +
``app.dependency_overrides`` for ``get_job_store`` and
``get_job_queue`` so the endpoint code talks to in-memory mocks.
``BriefGeneratorService.brief_to_prompts`` is patched at class level
so we never round-trip to a real LLM.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Test data + fixtures
# ---------------------------------------------------------------------------

PROJECT_ID = "proj-gen"
ROOM_ID = "room-A"


def _project_with_brief() -> dict:
    """Tracer-bullet project: one room, two variations, design_brief +
    analyses present so the brief composition path runs.

    The design_brief shape mirrors what the existing PATCH
    /projects/{id} flow writes into Cosmos via DesignBrief.model_dump.
    Only the fields BriefGeneratorService.brief_to_prompts touches
    need to be present for the endpoint to construct the model.

    Carries an ``_etag`` field so the producer's
    ``acquire_project_lease`` step (which CAS-replaces the project
    doc) can succeed against the test fakes.
    """
    return {
        "id": PROJECT_ID,
        "name": "Generate-jobs project",
        "prompt": "warm modern with greenery",
        "status": "completed",
        "_etag": '"abcd-1234"',
        "rooms": [
            {
                "id": ROOM_ID,
                "label": "Living Room",
                "original_image_url": (
                    "https://acct.blob.core.windows.net/images/staging/"
                    "proj-gen/originals/a.png"
                ),
                "status": "completed",
                "variations": [
                    {"id": "var-A1", "status": "completed",
                     "image_url": "https://acct/.../a1.png"},
                    {"id": "var-A2", "status": "completed",
                     "image_url": "https://acct/.../a2.png"},
                ],
            }
        ],
        "settings": {"variations_per_room": 2, "model": "gpt-image-2",
                     "quality": "high", "size": "auto"},
        "design_brief": {
            "global_instructions": "Bright modern staging.",
            "preserve_elements": ["fireplace"],
            "placement_guide": {"back_row": "sofa", "middle_row": None,
                                "front_row": None},
            "objects_palette": [
                {"name": "sofa", "size": "large", "placement": "back",
                 "quantity": 1, "visual_notes": "linen"}
            ],
            "per_image_overrides": {},
            "per_image_notes": {},
        },
        "analyses": [
            {"room_id": ROOM_ID, "description": "modern living room",
             "features": ["fireplace"]}
        ],
    }


def _project_no_brief() -> dict:
    """Project with rooms but NO design_brief / analyses — the brief
    composition branch must skip and pass ``brief_prompts=None``."""
    proj = _project_with_brief()
    proj["design_brief"] = None
    proj["analyses"] = []
    return proj


@pytest.fixture
def gen_jobs_deps(app, mock_staging_deps):
    """Stack on top of ``mock_staging_deps`` to also override
    ``get_job_store`` and ``get_job_queue``.

    The ``store.create_job`` side-effect rebuilds the deterministic id
    from the kwargs so two POSTs (each with a distinct
    Idempotency-Key header) produce distinct ids — the
    test_..._distinct test pins this.

    ``store.get_job.return_value = None`` is the default so the
    producer's same-key dedupe precheck doesn't mistakenly think a
    doc already exists. Tests that exercise the dedupe branch override
    this explicitly.

    The ``mock_staging_deps`` container's ``replace_item`` also returns
    a fresh project doc so ``acquire_project_lease`` (called by the
    producer to CAS-claim ``current_project_job_id``) succeeds by
    default. Tests that exercise the CAS-lose branch can override
    ``replace_item`` to raise a CosmosAccessConditionFailedError.
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
    # Default: same-key precheck sees no existing doc.
    store.get_job.return_value = None

    # Lease acquire CAS path: replace_item returns the updated project
    # doc so ``acquire_project_lease`` reports success.
    container = mock_staging_deps["container"]
    def _replace_item_default(item, body, **_kwargs):
        return body
    container.replace_item.side_effect = _replace_item_default

    app.dependency_overrides[staging_module.get_job_store] = lambda: store
    app.dependency_overrides[staging_module.get_job_queue] = lambda: queue
    try:
        yield {**mock_staging_deps, "store": store, "queue": queue}
    finally:
        app.dependency_overrides.pop(staging_module.get_job_store, None)
        app.dependency_overrides.pop(staging_module.get_job_queue, None)


def _patched_brief(brief_prompts: dict | None = None,
                   side_effect: BaseException | None = None):
    """Return a context manager that patches BriefGeneratorService.
    brief_to_prompts at class level. Default returns a deterministic
    per-room prompt dict; pass side_effect for failure-path tests."""
    if brief_prompts is None:
        brief_prompts = {ROOM_ID: ["composed prompt 1", "composed prompt 2"]}
    mock = AsyncMock(return_value=brief_prompts)
    if side_effect is not None:
        mock = AsyncMock(side_effect=side_effect)
    return patch(
        "backend.core.brief_generator.BriefGeneratorService.brief_to_prompts",
        mock,
    ), mock


# ---------------------------------------------------------------------------
# Happy path + AC pins
# ---------------------------------------------------------------------------


def test_post_returns_202_with_job_id_on_happy_path(client, gen_jobs_deps):
    """The headlining contract: POST returns 202 + {"job_id": ...},
    with kind=generate_project, the dunder sentinel room/variation ids,
    a uuid4-hex revision, and the precomputed brief_prompts in the
    payload. Pins create_job kwargs and queue.enqueue kwargs in the
    same call so a future refactor can't silently desync them."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    assert response.status_code == 202, response.text
    body = response.json()
    assert "job_id" in body
    job_id = body["job_id"]
    assert job_id.startswith(f"{PROJECT_ID}:__project__:__project__:")

    store = gen_jobs_deps["store"]
    queue = gen_jobs_deps["queue"]
    assert store.create_job.call_count == 1
    kwargs = store.create_job.call_args.kwargs
    assert kwargs["project_id"] == PROJECT_ID
    assert kwargs["room_id"] == "__project__"
    assert kwargs["variation_id"] == "__project__"
    assert kwargs["kind"] == "generate_project"
    assert kwargs["payload"]["regenerate_all"] is False
    assert kwargs["payload"]["brief_prompts"] == {
        ROOM_ID: ["composed prompt 1", "composed prompt 2"]
    }

    assert queue.enqueue.call_count == 1
    eq = queue.enqueue.call_args.kwargs
    assert eq["job_id"] == job_id
    assert eq["project_id"] == PROJECT_ID


def test_post_503_when_feature_flag_off(client, gen_jobs_deps, monkeypatch):
    """PRD parity with /jobs/regenerate: when FEATURE_ASYNC_QUEUE is
    false, fail loud (503) so a misconfigured production deploy
    doesn't silently queue into a worker that isn't running."""
    from backend.core import config as cfg
    monkeypatch.setattr(cfg.settings, "FEATURE_ASYNC_QUEUE", False)

    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    response = client.post(
        f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate", json={}
    )
    assert response.status_code == 503
    gen_jobs_deps["store"].create_job.assert_not_called()
    gen_jobs_deps["queue"].enqueue.assert_not_called()


def test_post_404_when_project_missing(client, gen_jobs_deps):
    container = gen_jobs_deps["container"]
    container.read_item.side_effect = Exception("not found")
    # storage.get_project handles its own 404 internally returning None;
    # mock_staging_deps' container is the underlying source. The
    # StagingStorageService.get_project method catches the Cosmos
    # not-found and returns None, so we mock get_project directly:
    from backend.core.staging_storage import StagingStorageService
    with patch.object(
        StagingStorageService, "get_project", return_value=None
    ):
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    assert response.status_code == 404
    gen_jobs_deps["store"].create_job.assert_not_called()
    gen_jobs_deps["queue"].enqueue.assert_not_called()


def test_post_400_when_project_has_no_rooms(client, gen_jobs_deps):
    """Rubber-duck grill #3: preserve the legacy
    ``POST /projects/{id}/generate`` contract so a client mistake
    (project with no rooms uploaded yet) is rejected up front
    rather than burning 30-90s on brief composition + creating a
    doomed job. The empty-rooms guard MUST run BEFORE brief
    composition AND before cascade-cancel."""
    container = gen_jobs_deps["container"]
    proj = _project_with_brief()
    proj["rooms"] = []
    container.read_item.return_value = proj

    patcher, brief_mock = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={"regenerate_all": True},
        )
    assert response.status_code == 400, response.text
    # No brief composition (would have burned 30-90s for nothing).
    assert not brief_mock.called
    # No cascade — we MUST NOT pre-cancel sibling jobs for an invalid request.
    gen_jobs_deps["store"].update_job.assert_not_called()
    gen_jobs_deps["store"].create_job.assert_not_called()
    gen_jobs_deps["queue"].enqueue.assert_not_called()


def test_post_422_when_regenerate_all_is_string(client, gen_jobs_deps):
    """Rubber-duck grill #4: ``regenerate_all`` is a destructive flag
    (cancels in-flight variations + later clears project state). A
    permissive ``bool(...)`` would convert ``"yes"`` / ``"false"`` /
    ``[1]`` to True. StrictBool rejects with 422. No side effects."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    patcher, brief_mock = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={"regenerate_all": "yes"},
        )
    assert response.status_code == 422, response.text
    assert not brief_mock.called
    gen_jobs_deps["store"].update_job.assert_not_called()
    gen_jobs_deps["store"].create_job.assert_not_called()
    gen_jobs_deps["queue"].enqueue.assert_not_called()


def test_post_default_body_uses_regenerate_all_false(client, gen_jobs_deps):
    """Body can be absent or {} — defaults to regenerate_all=False."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    assert response.status_code == 202
    payload = gen_jobs_deps["store"].create_job.call_args.kwargs["payload"]
    assert payload["regenerate_all"] is False


def test_post_payload_includes_brief_prompts_and_regenerate_all_true(
    client, gen_jobs_deps
):
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    patcher, _ = _patched_brief(
        brief_prompts={ROOM_ID: ["P1", "P2", "P3", "P4"]}
    )
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={"regenerate_all": True},
        )
    assert response.status_code == 202
    payload = gen_jobs_deps["store"].create_job.call_args.kwargs["payload"]
    assert payload["regenerate_all"] is True
    assert payload["brief_prompts"] == {ROOM_ID: ["P1", "P2", "P3", "P4"]}


def test_post_revision_is_uuid_hex_not_integer(client, gen_jobs_deps):
    """PRD AC: revision MUST be uuid.uuid4().hex (a 32-char lowercase
    hex string), NOT an integer counter. Two concurrent POSTs
    relying on integer revisions would race through
    ``_select_revision_for_idempotent_regen`` and silently collapse
    the second click into the first job's id (rubber-duck blocking #2).
    """
    import re

    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    assert response.status_code == 202
    revision = gen_jobs_deps["store"].create_job.call_args.kwargs["revision"]
    assert isinstance(revision, str), (
        f"revision must be a uuid hex string, got {type(revision).__name__}"
    )
    assert re.fullmatch(r"[0-9a-f]{32}", revision), (
        f"revision must match uuid4().hex format, got {revision!r}"
    )


def test_post_brief_uses_variations_per_room_setting(client, gen_jobs_deps):
    """``n_variations`` passed to brief_to_prompts must reflect the
    project's per-room setting, NOT a hardcoded 5."""
    container = gen_jobs_deps["container"]
    proj = _project_with_brief()
    proj["settings"]["variations_per_room"] = 4
    container.read_item.return_value = proj

    patcher, brief_mock = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    assert response.status_code == 202
    assert brief_mock.called
    assert brief_mock.call_args.kwargs["n_variations"] == 4


def test_post_no_design_brief_passes_none_brief_prompts(client, gen_jobs_deps):
    """Project lacking design_brief OR analyses skips brief composition
    entirely (the legacy regenerate_room block has the same gating).
    Payload's brief_prompts is None — the dispatcher (issue 005) owns
    the None -> legacy-compute fallback in generate_project_for_job."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_no_brief()

    patcher, brief_mock = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    assert response.status_code == 202, response.text
    assert not brief_mock.called
    payload = gen_jobs_deps["store"].create_job.call_args.kwargs["payload"]
    assert payload["brief_prompts"] is None


# ---------------------------------------------------------------------------
# Failure / compensation pins
# ---------------------------------------------------------------------------


def test_post_brief_composition_failure_returns_5xx_no_job_created(
    app, gen_jobs_deps
):
    """PRD AC: brief failure path asserts no job doc was written.
    Brief runs FIRST (before cascade + before create_job) so the
    failure short-circuits cleanly with zero Cosmos / Queue side
    effects.

    Uses a dedicated TestClient with ``raise_server_exceptions=False``
    so the unhandled-exception path (RuntimeError from brief_to_prompts)
    converts to a 500 response — that's the production FastAPI
    behaviour; the default TestClient re-raises for test convenience.
    """
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    raising_client = TestClient(app, raise_server_exceptions=False)

    patcher, _ = _patched_brief(side_effect=RuntimeError("LLM down"))
    with patcher:
        response = raising_client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    # FastAPI's default-exception handler converts the unhandled
    # RuntimeError to 500. The test pin is "no job created" + 5xx.
    assert response.status_code >= 500
    gen_jobs_deps["store"].create_job.assert_not_called()
    gen_jobs_deps["queue"].enqueue.assert_not_called()


def test_post_marks_job_failed_when_enqueue_raises(client, gen_jobs_deps):
    """Rubber-duck grill #1: if create_job succeeds but enqueue fails,
    the doc would otherwise sit in 'pending' forever (no worker will
    ever pick it up; UUID4 means the client retry produces a NEW
    distinct doc, not the same one). Compensation: mark the orphan
    'failed' (with error_kind + structured error dict) so the SSE
    feed and GET /jobs surface it.

    Issue 002: response is the new structured error shape
    ``{error_kind, user_message, detail}`` (was raw ``{detail: str}``).
    A bare RuntimeError ("queue down") is classified as UNKNOWN/500
    by ``backend.core.job_errors.classify``; QUEUE_PERMISSION/502
    would require a real Azure auth error type.
    """
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    queue = gen_jobs_deps["queue"]
    queue.enqueue.side_effect = RuntimeError("queue down")

    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    # Generic RuntimeError is classified as UNKNOWN/500.
    assert response.status_code == 500, response.text
    body = response.json()
    assert body["error_kind"] == "UNKNOWN"
    assert "user_message" in body
    assert body["detail"] == {
        "type": "RuntimeError", "message": "queue down",
    }

    store = gen_jobs_deps["store"]
    # Compensation update_job MUST have been called with status=failed
    # against the doc id we just minted.
    assert store.update_job.called, (
        "Compensation update_job(status='failed') must run when "
        "enqueue raises — otherwise the orphan job sits in 'pending' "
        "forever."
    )
    compensation_call = store.update_job.call_args
    compensation_kwargs = compensation_call.kwargs
    args = compensation_call.args
    job_id = args[0] if args else compensation_kwargs.get("job_id")
    assert job_id and job_id.startswith(
        f"{PROJECT_ID}:__project__:__project__:"
    )
    assert compensation_kwargs.get("status") == "failed"
    # Issue 002: error_kind is now persisted on the doc so the front-end
    # can surface a kind-specific message via /jobs.
    assert compensation_kwargs.get("error_kind") == "UNKNOWN"
    # Issue 002: error is a structured {type, message} dict (matches the
    # worker's existing terminal-failure shape), not a raw string.
    assert compensation_kwargs.get("error") == {
        "type": "RuntimeError", "message": "queue down",
    }


def test_post_compensation_failure_still_raises_5xx(client, gen_jobs_deps):
    """If both enqueue AND the compensation update_job fail, the
    endpoint MUST still return 5xx (the user's request didn't
    succeed). Compensation is best-effort; a logging-only failure
    must not become a request-blocking exception."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    store = gen_jobs_deps["store"]
    queue = gen_jobs_deps["queue"]
    queue.enqueue.side_effect = RuntimeError("queue down")
    store.update_job.side_effect = RuntimeError("cosmos down")

    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    # 5xx range — enqueue failed, compensation also failed, but the
    # client still gets a non-2xx so the retry path can engage.
    assert response.status_code >= 500


# ---------------------------------------------------------------------------
# Cascade contract pins
# ---------------------------------------------------------------------------


def test_post_regenerate_all_cascades_cancel_to_inflight_variation_jobs(
    client, gen_jobs_deps
):
    """When regenerate_all=true, every NON-TERMINAL regenerate_variation
    job for the project gets cancel_requested=True flipped via the
    cascade-cancel helper from issue 002.

    The helper MUST skip:
    - terminal jobs (already done — no need to re-cancel)
    - generate_project jobs (different kind; the lease guarantees
      only one runs at a time, no need to pre-empt)
    """
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    store = gen_jobs_deps["store"]
    store.list_jobs_by_project.return_value = [
        # Two in-flight regenerate_variation jobs — should be cancelled.
        {"id": f"{PROJECT_ID}:room-A:var-A1:0",
         "project_id": PROJECT_ID, "kind": "regenerate_variation",
         "status": "pending"},
        {"id": f"{PROJECT_ID}:room-A:var-A2:0",
         "project_id": PROJECT_ID, "kind": "regenerate_variation",
         "status": "running"},
        # Terminal regenerate_variation — should be SKIPPED.
        {"id": f"{PROJECT_ID}:room-A:var-A3:0",
         "project_id": PROJECT_ID, "kind": "regenerate_variation",
         "status": "succeeded"},
        # Another generate_project (different kind) — should be SKIPPED.
        {"id": f"{PROJECT_ID}:__project__:__project__:abc",
         "project_id": PROJECT_ID, "kind": "generate_project",
         "status": "pending"},
    ]

    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={"regenerate_all": True},
        )
    assert response.status_code == 202

    # Exactly the two in-flight regenerate_variation jobs should have
    # been touched with cancel_requested=True.
    cancel_calls = [
        c for c in store.update_job.call_args_list
        if c.kwargs.get("cancel_requested") is True
    ]
    assert len(cancel_calls) == 2, (
        f"Expected exactly 2 cancel_requested=True update_job calls "
        f"(one per in-flight regenerate_variation), got {len(cancel_calls)}: "
        f"{cancel_calls}"
    )
    cancelled_ids = {c.args[0] for c in cancel_calls}
    assert cancelled_ids == {
        f"{PROJECT_ID}:room-A:var-A1:0",
        f"{PROJECT_ID}:room-A:var-A2:0",
    }


def test_post_regenerate_all_false_does_not_cascade_cancel(
    client, gen_jobs_deps
):
    """Default regenerate_all=False MUST NOT cascade. The dispatcher's
    project lease (issue 002) serializes runs against in-flight
    variations; cascading is only justified when the user explicitly
    asked to throw away in-flight work."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    store = gen_jobs_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": f"{PROJECT_ID}:room-A:var-A1:0",
         "project_id": PROJECT_ID, "kind": "regenerate_variation",
         "status": "pending"},
    ]

    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={"regenerate_all": False},
        )
    assert response.status_code == 202
    cancel_calls = [
        c for c in store.update_job.call_args_list
        if c.kwargs.get("cancel_requested") is True
    ]
    assert cancel_calls == [], (
        "regenerate_all=False MUST NOT cancel sibling variation jobs."
    )


def test_post_regenerate_all_cancellation_persists_when_create_job_fails(
    app, gen_jobs_deps
):
    """Rubber-duck grill #2: cascade-cancel is a POINT OF NO RETURN.
    If create_job fails AFTER cascade ran, the cancellation persists
    — there is no rollback. Rolling back ``cancel_requested=False``
    would race other concurrent requests and could undo a legitimate
    user-initiated cancel.

    Uses ``raise_server_exceptions=False`` because the create_job
    RuntimeError is unhandled (no compensation path for create_job
    failure — the doc never made it to the queue).
    """
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    store = gen_jobs_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": f"{PROJECT_ID}:room-A:var-A1:0",
         "project_id": PROJECT_ID, "kind": "regenerate_variation",
         "status": "pending"},
    ]
    # Cascade goes through update_job; create_job fails AFTER cascade.
    store.create_job.side_effect = RuntimeError("cosmos down")

    raising_client = TestClient(app, raise_server_exceptions=False)
    patcher, _ = _patched_brief()
    with patcher:
        response = raising_client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={"regenerate_all": True},
        )
    assert response.status_code >= 500

    # Cascade still fired before create_job blew up.
    cancel_calls = [
        c for c in store.update_job.call_args_list
        if c.kwargs.get("cancel_requested") is True
    ]
    assert len(cancel_calls) == 1
    # And no rollback uncancel was issued (no cancel_requested=False call).
    uncancel_calls = [
        c for c in store.update_job.call_args_list
        if c.kwargs.get("cancel_requested") is False
    ]
    assert uncancel_calls == [], (
        "Cascade is point-of-no-return: there must be NO rollback "
        "of cancel_requested when create_job subsequently fails."
    )


def test_post_regenerate_all_cancellation_persists_when_enqueue_fails(
    client, gen_jobs_deps
):
    """Symmetric grill #2 pin: cancellation persists even when
    cascade succeeded, create_job succeeded, and enqueue fails."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    store = gen_jobs_deps["store"]
    store.list_jobs_by_project.return_value = [
        {"id": f"{PROJECT_ID}:room-A:var-A1:0",
         "project_id": PROJECT_ID, "kind": "regenerate_variation",
         "status": "pending"},
    ]
    queue = gen_jobs_deps["queue"]
    queue.enqueue.side_effect = RuntimeError("queue down")

    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={"regenerate_all": True},
        )
    assert response.status_code >= 500

    cancel_calls = [
        c for c in store.update_job.call_args_list
        if c.kwargs.get("cancel_requested") is True
    ]
    assert len(cancel_calls) == 1, (
        "Cascade ran before enqueue failed and the cancellation must persist."
    )
    uncancel_calls = [
        c for c in store.update_job.call_args_list
        if c.kwargs.get("cancel_requested") is False
    ]
    assert uncancel_calls == []


# ---------------------------------------------------------------------------
# Concurrent / race pin
# ---------------------------------------------------------------------------


def test_post_two_concurrent_calls_produce_two_distinct_job_ids(
    client, gen_jobs_deps
):
    """PRD AC + rubber-duck blocking #2: two concurrent POSTs MUST
    produce two distinct job documents. Integer-counter revisions
    would race through ``_select_revision_for_idempotent_regen`` and
    silently collapse the second into the first; UUID4 hex
    revisions can never collide.

    The TestClient is synchronous, so we exercise the contract by
    issuing two sequential calls — the deterministic id format is
    ``{project_id}:__project__:__project__:{uuid4().hex}``, so
    distinct revisions imply distinct ids regardless of call
    ordering.
    """
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    patcher, _ = _patched_brief()
    with patcher:
        r1 = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
        r2 = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    assert r1.status_code == 202
    assert r2.status_code == 202

    id_1 = r1.json()["job_id"]
    id_2 = r2.json()["job_id"]
    assert id_1 != id_2, (
        "Two POSTs MUST produce DISTINCT job ids (UUID4 hex revisions). "
        "If these ever collide, the integer-counter regression has "
        "snuck back in via _select_revision_for_idempotent_regen."
    )

    # Both create_job calls actually ran — no idempotency collapse.
    assert gen_jobs_deps["store"].create_job.call_count == 2
    revisions = [
        c.kwargs["revision"]
        for c in gen_jobs_deps["store"].create_job.call_args_list
    ]
    assert revisions[0] != revisions[1]


# ---------------------------------------------------------------------------
# Issue 002 — Idempotency-Key + dedupe + structured error shape
# ---------------------------------------------------------------------------


def test_post_uses_idempotency_key_header_as_revision(client, gen_jobs_deps):
    """Issue 002 contract: when the front-end sends an
    ``Idempotency-Key`` header, the producer uses it verbatim as the
    deterministic-id revision component. Transport-layer retries that
    re-send the SAME header collapse onto the SAME doc id."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    key = "abc123-DEADBEEF_42"
    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
            headers={"Idempotency-Key": key},
        )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["job_id"] == f"{PROJECT_ID}:__project__:__project__:{key}"
    assert body["already_in_flight"] is False
    revision = gen_jobs_deps["store"].create_job.call_args.kwargs["revision"]
    assert revision == key


def test_post_same_idempotency_key_returns_200_already_in_flight(
    client, gen_jobs_deps
):
    """Issue 002 headline contract: a transport-layer retry that
    re-sends the SAME Idempotency-Key gets 200 (not 202) +
    ``already_in_flight=true`` AND the same job_id. No second
    ``create_job`` call is made (idempotent retry semantics).
    """
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    key = "samekey1234567890"
    seeded_id = f"{PROJECT_ID}:__project__:__project__:{key}"
    store = gen_jobs_deps["store"]
    # Same-key precheck finds the existing doc — produces dedupe path.
    store.get_job.return_value = {
        "id": seeded_id,
        "project_id": PROJECT_ID,
        "kind": "generate_project",
        "status": "pending",
    }

    patcher, brief_mock = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
            headers={"Idempotency-Key": key},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job_id"] == seeded_id
    assert body["already_in_flight"] is True
    # Dedupe must short-circuit BEFORE brief composition + create_job.
    store.create_job.assert_not_called()
    gen_jobs_deps["queue"].enqueue.assert_not_called()
    assert not brief_mock.called


def test_post_invalid_idempotency_key_returns_422(client, gen_jobs_deps):
    """Issue 002 server-side validation: keys must match the regex
    ``^[A-Za-z0-9_-]{1,128}$``. Empty / >128 / non-ASCII / colon /
    slash / dot — all rejected with 422 BEFORE any side effects."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    response = client.post(
        f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
        json={},
        headers={"Idempotency-Key": "has space"},
    )
    assert response.status_code == 422, response.text
    gen_jobs_deps["store"].create_job.assert_not_called()
    gen_jobs_deps["queue"].enqueue.assert_not_called()


def test_post_missing_idempotency_key_falls_back_to_server_minted_revision(
    client, gen_jobs_deps
):
    """Backwards-compat: callers that don't send the header still
    succeed. The server mints a uuid4().hex as the revision. The
    response body is still the new ``{job_id, already_in_flight}``
    shape (NOT a legacy single-key body)."""
    import re
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    patcher, _ = _patched_brief()
    with patcher:
        response = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )  # no Idempotency-Key header
    assert response.status_code == 202, response.text
    body = response.json()
    assert "job_id" in body
    assert body["already_in_flight"] is False
    revision = gen_jobs_deps["store"].create_job.call_args.kwargs["revision"]
    assert re.fullmatch(r"[0-9a-f]{32}", revision), (
        f"Server-minted revision must be uuid4().hex, got {revision!r}"
    )


def test_post_brief_composition_failure_returns_structured_error_kind(
    app, gen_jobs_deps
):
    """Issue 002 error-shape contract: brief composition failure no
    longer raises an unhandled exception — it's classified as
    BRIEF_FAILED/502 with the structured ``{error_kind, user_message,
    detail}`` body so the front-end can surface a kind-specific
    message instead of a raw string.
    """
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    patcher, _ = _patched_brief(side_effect=RuntimeError("LLM down"))
    raising_client = TestClient(app, raise_server_exceptions=False)
    with patcher:
        response = raising_client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
        )
    assert response.status_code == 502, response.text
    body = response.json()
    assert body["error_kind"] == "BRIEF_FAILED"
    assert "user_message" in body and body["user_message"]
    # detail carries the wrapping ``BriefCompositionFailed`` type +
    # the underlying message (preserved via __cause__). The wrapper
    # type is what the classifier dispatches on, so it's the
    # canonical "type" for the structured error.
    assert body["detail"]["type"] == "BriefCompositionFailed"
    assert "LLM down" in body["detail"]["message"]
    # No job created — brief failure is a clean short-circuit.
    gen_jobs_deps["store"].create_job.assert_not_called()
    gen_jobs_deps["queue"].enqueue.assert_not_called()


def test_post_two_distinct_keys_produce_distinct_jobs(client, gen_jobs_deps):
    """Frontend always mints a fresh ``crypto.randomUUID()`` per
    button click. Two distinct keys produce two distinct doc ids
    (the dedupe path doesn't fire). This pins that the
    Idempotency-Key wire format actually round-trips through the
    revision component."""
    container = gen_jobs_deps["container"]
    container.read_item.return_value = _project_with_brief()

    patcher, _ = _patched_brief()
    with patcher:
        r1 = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
            headers={"Idempotency-Key": "key-one"},
        )
        r2 = client.post(
            f"/api/v1/staging/projects/{PROJECT_ID}/jobs/generate",
            json={},
            headers={"Idempotency-Key": "key-two"},
        )
    assert r1.status_code == 202
    assert r2.status_code == 202
    assert r1.json()["job_id"] != r2.json()["job_id"]
    assert r1.json()["job_id"].endswith(":key-one")
    assert r2.json()["job_id"].endswith(":key-two")
