"""Tests for ``PATCH /api/v1/staging/projects/{id}`` — project-level
partial-update endpoint added in issue 002 of the projects-page-
improvements PRD.

Endpoint contract (asserted by tests below):

    PATCH /api/v1/staging/projects/{id}
    Body shape: ``{name?, prompt?, settings?, design_brief?}``

    - Each of the four fields is optional (omit to leave untouched).
    - ``settings`` is MERGED onto the persisted settings, key-by-key.
      Sending ``{settings: {variations_per_room: 3}}`` updates only that
      key and leaves the persisted ``model``/``quality``/``size`` intact.
    - ``design_brief`` accepts ``null`` explicitly to CLEAR the brief.
      The other three fields treat ``null`` as a 422 validation error
      (clients shouldn't send null for required project state).
    - The endpoint NEVER modifies ``rooms``, ``analyses``, or ``status``.
    - The endpoint NEVER triggers any generation (no SSE response, no
      pipeline invocation).
    - Returns the updated full ``ProjectResponse`` in plain JSON.

The endpoint runs its read-modify-write inside the per-project
``_get_project_lock`` to serialize with the regen finalizers and
pipeline workers. The race regression test
``test_pipeline_persist_does_not_clobber_patch_top_level_fields`` pins
the joint behavior with the pipeline's ``_persist_project_locked``,
which (per this PRD slice) was changed to merge only its own pipeline-
owned fields (``rooms`` + ``status``) so a concurrent worker write
cannot clobber the user-owned scalars (``name``/``prompt``/``settings``/
``design_brief``).
"""
import json

import pytest
from unittest.mock import MagicMock


def _project_with_two_rooms_for_patch_project() -> dict:
    """Two-room completed project. Tests assert (a) only the field they
    PATCH gets touched and (b) ``rooms``/``analyses``/``status`` are
    preserved byte-for-byte across the write."""
    return {
        "id": "proj-pp",
        "name": "Original Name",
        "prompt": "modern minimalist",
        "status": "completed",
        "rooms": [
            {
                "id": "room-A",
                "label": "Living Room",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/a.png",
                "status": "completed",
                "variations": [
                    {
                        "id": "var-A1",
                        "status": "completed",
                        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-A/v1.png",
                        "generation_metadata": {
                            "model": "gpt-image-2",
                            "adapted_prompt": "earlier prompt for A",
                        },
                    }
                ],
            },
            {
                "id": "room-B",
                "label": "Kitchen",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/b.png",
                "status": "completed",
                "prompt_addendum": "B's existing addendum",
                "variations": [
                    {
                        "id": "var-B1",
                        "status": "completed",
                        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-B/v1.png",
                    }
                ],
            },
        ],
        "settings": {
            "variations_per_room": 5,
            "model": "gpt-image-2",
            "quality": "high",
            "size": "auto",
        },
        "analyses": [
            {"room_id": "room-A", "summary": "warm wood and greenery"},
            {"room_id": "room-B", "summary": "stainless and white tile"},
        ],
        "design_brief": {
            "global_instructions": "preserve the pergola",
            "object_palette": [],
        },
    }


def _captured_replace_body(mock_container: MagicMock) -> dict:
    """Pull the body kwarg / positional arg from the most recent
    ``replace_item`` call on the mocked Cosmos container. The Cosmos SDK
    accepts both call shapes; the storage layer happens to use kwargs
    today but the test isn't coupled to that choice."""
    call = mock_container.replace_item.call_args
    return call.kwargs.get("body") or call.args[1]


# ---------------------------------------------------------------------------
# Tracer bullet: name-only update (the simplest possible PATCH).
# ---------------------------------------------------------------------------


def test_patch_project_name_only_updates_only_name(client, mock_staging_deps):
    """The simplest tracer: PATCH ``{name: "X"}`` updates the name and
    leaves every other field untouched. Sibling keys (rooms, analyses,
    status, prompt, settings, design_brief) must be byte-identical."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch_project()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    rooms_before = json.dumps(project_data["rooms"], sort_keys=True)
    analyses_before = json.dumps(project_data["analyses"], sort_keys=True)
    status_before = project_data["status"]
    prompt_before = project_data["prompt"]
    settings_before = json.dumps(project_data["settings"], sort_keys=True)
    brief_before = json.dumps(project_data["design_brief"], sort_keys=True)

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"name": "Renamed Project"},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["name"] == "Renamed Project"
    assert json.dumps(persisted["rooms"], sort_keys=True) == rooms_before
    assert json.dumps(persisted["analyses"], sort_keys=True) == analyses_before
    assert persisted["status"] == status_before
    assert persisted["prompt"] == prompt_before
    assert json.dumps(persisted["settings"], sort_keys=True) == settings_before
    assert json.dumps(persisted["design_brief"], sort_keys=True) == brief_before


def test_patch_project_prompt_only_updates_only_prompt(client, mock_staging_deps):
    """``prompt`` is the second-most-likely user edit. Same byte-for-byte
    invariants as the name test, just for a different scalar."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch_project()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    name_before = project_data["name"]
    rooms_before = json.dumps(project_data["rooms"], sort_keys=True)
    settings_before = json.dumps(project_data["settings"], sort_keys=True)

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"prompt": "warm wood, lots of greenery"},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["prompt"] == "warm wood, lots of greenery"
    assert persisted["name"] == name_before
    assert json.dumps(persisted["rooms"], sort_keys=True) == rooms_before
    assert json.dumps(persisted["settings"], sort_keys=True) == settings_before


def test_patch_project_settings_partial_merges_onto_persisted(client, mock_staging_deps):
    """Critical contract: a partial settings PATCH like
    ``{settings: {variations_per_room: 3}}`` must MERGE onto the
    persisted settings, NOT replace the whole object. Without this
    behavior, the absent keys (model/quality/size) would silently revert
    to ``StagingSettings`` defaults and the user would lose their
    previously-chosen image config.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch_project()
    # Pre-set non-default settings so we can assert they SURVIVE the
    # partial PATCH. If the endpoint full-replaced settings, model
    # would be reset to "gpt-image-2" and quality to "high" — both of
    # which happen to match the defaults, so we set distinct values
    # here to make the test discriminating.
    project_data["settings"] = {
        "variations_per_room": 5,
        "model": "flux-kontext-pro",
        "quality": "low",
        "size": "1536x1024",
    }
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"settings": {"variations_per_room": 3}},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["settings"] == {
        "variations_per_room": 3,
        "model": "flux-kontext-pro",  # preserved
        "quality": "low",  # preserved
        "size": "1536x1024",  # preserved
    }


def test_patch_project_design_brief_set(client, mock_staging_deps):
    """A non-null ``design_brief`` value is persisted as-is."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch_project()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "different brief",
        "object_palette": [{"id": "obj-1", "name": "couch", "category": "furniture"}],
    }

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"design_brief": new_brief},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["design_brief"] == new_brief


def test_patch_project_design_brief_explicit_null_clears(client, mock_staging_deps):
    """Sending ``design_brief: null`` explicitly clears the persisted
    brief — distinct from the "absent" case, which leaves it untouched.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch_project()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"design_brief": None},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["design_brief"] is None


def test_patch_project_all_fields_atomic(client, mock_staging_deps):
    """A PATCH with every editable field updates them atomically — one
    storage write, all fields applied. This is the most-realistic
    payload from the frontend Settings sheet (Save with multiple dirty
    fields).

    Note on the both-present prompt+brief assertion below: issue 001
    of the project-settings-completeness PRD added a mirror that makes
    ``design_brief.global_instructions`` win on ``project.prompt`` when
    both are sent in the same PATCH. So the persisted ``prompt`` ends
    up equal to ``new_brief["global_instructions"]`` (``"all-fields"``),
    NOT the user-supplied ``"all-fields prompt"``. The full mirror
    contract has dedicated coverage in
    ``test_staging_endpoints_prompt_brief_mirror.py``.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch_project()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {"global_instructions": "all-fields", "object_palette": []}

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={
            "name": "All Fields",
            "prompt": "all-fields prompt",
            "settings": {"variations_per_room": 7, "model": "gpt-image-2", "quality": "medium", "size": "1024x1024"},
            "design_brief": new_brief,
        },
    )
    assert response.status_code == 200, response.text
    # Atomic: a single replace_item call.
    assert mock_container.replace_item.call_count == 1

    persisted = _captured_replace_body(mock_container)
    assert persisted["name"] == "All Fields"
    # Brief wins on prompt (mirror — see docstring above).
    assert persisted["prompt"] == "all-fields"
    assert persisted["settings"] == {
        "variations_per_room": 7,
        "model": "gpt-image-2",
        "quality": "medium",
        "size": "1024x1024",
    }
    assert persisted["design_brief"] == new_brief


def test_patch_project_empty_body_is_noop_passthrough(client, mock_staging_deps):
    """An empty body ``{}`` is a valid PATCH that doesn't change
    anything. The response still echoes the project so frontend code
    that always reloads-from-response stays consistent."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch_project()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    name_before = project_data["name"]
    rooms_before = json.dumps(project_data["rooms"], sort_keys=True)

    response = client.patch("/api/v1/staging/projects/proj-pp", json={})
    assert response.status_code == 200

    persisted = _captured_replace_body(mock_container)
    assert persisted["name"] == name_before
    assert json.dumps(persisted["rooms"], sort_keys=True) == rooms_before


# ---------------------------------------------------------------------------
# Validation: explicit-null on required fields → 422.
# ---------------------------------------------------------------------------


def test_patch_project_name_explicit_null_returns_422(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms_for_patch_project()

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"name": None},
    )
    assert response.status_code == 422
    # And nothing was persisted.
    assert not mock_container.replace_item.called


def test_patch_project_prompt_explicit_null_returns_422(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms_for_patch_project()

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"prompt": None},
    )
    assert response.status_code == 422


def test_patch_project_settings_explicit_null_returns_422(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms_for_patch_project()

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"settings": None},
    )
    assert response.status_code == 422


def test_patch_project_name_empty_string_returns_422(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms_for_patch_project()

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"name": "   "},
    )
    assert response.status_code == 422


def test_patch_project_invalid_settings_returns_422(client, mock_staging_deps):
    """``StagingSettings`` validates ``variations_per_room`` between
    1 and 10. An out-of-range value must surface as a 422 from
    Pydantic, NOT silently coerce or accept."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms_for_patch_project()

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"settings": {"variations_per_room": 99}},
    )
    assert response.status_code == 422
    assert not mock_container.replace_item.called


# ---------------------------------------------------------------------------
# 404 + 405-related sanity.
# ---------------------------------------------------------------------------


def test_patch_project_returns_404_when_project_missing(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    mock_container = mock_staging_deps["container"]
    mock_container.read_item.side_effect = CosmosResourceNotFoundError(
        status_code=404, message="Not found"
    )

    response = client.patch(
        "/api/v1/staging/projects/nope",
        json={"name": "x"},
    )
    assert response.status_code == 404
    assert not mock_container.replace_item.called


# ---------------------------------------------------------------------------
# Pipeline isolation: PATCH must NEVER trigger any generation.
# ---------------------------------------------------------------------------


def test_patch_project_does_not_invoke_pipeline(client, mock_staging_deps):
    """PATCH is a pure metadata write. It must not touch the pipeline:
    no ``process_room``, no ``generate_project``, no
    ``process_single_variation`` calls — and the response must be plain
    JSON, not an SSE stream."""
    mock_container = mock_staging_deps["container"]
    mock_pipeline = mock_staging_deps["pipeline"]
    project_data = _project_with_two_rooms_for_patch_project()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"name": "PATCH-no-pipeline"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    assert not mock_pipeline.process_room.called
    assert not mock_pipeline.generate_project.called
    assert not mock_pipeline.process_single_variation.called


def test_patch_project_returns_updated_project_payload(client, mock_staging_deps):
    """The response payload includes the freshly-updated project so the
    frontend can swap local state without an extra GET round-trip."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch_project()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-pp",
        json={"name": "Echoed Back"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "project" in body
    assert body["project"]["name"] == "Echoed Back"
    # Existing rooms are echoed back too.
    assert len(body["project"]["rooms"]) == 2
    assert {r["id"] for r in body["project"]["rooms"]} == {"room-A", "room-B"}


# ---------------------------------------------------------------------------
# Race regression: PATCH'd top-level fields must survive a later pipeline
# ``_persist_project_locked`` call. Pre-fix, the pipeline serialized the
# WHOLE in-memory project (including stale ``name``/``prompt``/
# ``settings``/``design_brief``) and merge-overwrote storage on every
# room-finish persist. With the fix that scopes pipeline writes to
# ``{rooms, status}`` only, the PATCH-touched user-owned scalars survive.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_persist_does_not_clobber_patch_top_level_fields(
    monkeypatch,
):
    """Drive the actual race by hand. A pipeline worker holds a stale
    in-memory ``project`` snapshot from before a PATCH lands. The
    pipeline's ``_persist_project_locked`` writes back AFTER the PATCH.
    The PATCH-touched ``name``/``prompt``/``settings``/``design_brief``
    must survive because pipeline writes are now scoped to
    ``{rooms, status}`` only.

    This is the race the rubber-duck flagged as blocking. Pre-fix this
    test fails because the pipeline's full-project serialization
    overwrites the PATCH'd name back to its stale value.
    """
    from backend.core.staging_pipeline import StagingPipeline
    from backend.models.staging import (
        ItemStatus,
        ProjectStatus,
        Room,
        StagingProject,
        StagingSettings,
        Variation,
    )

    # In-process fake storage that mimics the real Cosmos read-modify-
    # write merge: ``update_project`` reads existing, dict-merges
    # ``updates`` over it, persists the whole doc.
    persisted_doc: dict = {
        "id": "race-proj",
        "name": "Original Name",
        "prompt": "original prompt",
        "status": "processing",
        "settings": {
            "variations_per_room": 5,
            "model": "gpt-image-2",
            "quality": "high",
            "size": "auto",
        },
        "design_brief": {"global_instructions": "original brief"},
        "analyses": [],
        "rooms": [
            {
                "id": "room-1",
                "label": "Room 1",
                "original_image_url": "https://acct.blob.core.windows.net/img/originals/r1.png",
                "status": "processing",
                "prompt_addendum": None,
                "variations": [],
            }
        ],
    }

    class FakeStorage:
        def get_project(self, project_id):
            # Always return a fresh copy so the pipeline's "stale" snapshot
            # remains genuinely stale relative to PATCH writes.
            import copy
            return copy.deepcopy(persisted_doc)

        def update_project(self, project_id, updates):
            existing = persisted_doc
            existing.update(updates)
            return existing

    fake_storage = FakeStorage()

    # Build the pipeline's in-memory project snapshot. This is the
    # "stale" state — captured BEFORE the PATCH lands.
    stale_project = StagingProject(
        id="race-proj",
        name="Original Name",
        prompt="original prompt",
        status=ProjectStatus.PROCESSING,
        settings=StagingSettings(
            variations_per_room=5, model="gpt-image-2", quality="high", size="auto"
        ),
        design_brief={"global_instructions": "original brief"},
        rooms=[
            Room(
                id="room-1",
                label="Room 1",
                original_image_url="https://acct.blob.core.windows.net/img/originals/r1.png",
                status=ItemStatus.PROCESSING,
                variations=[],
            )
        ],
    )

    # Mock all the heavy pipeline dependencies — we only care about
    # ``_persist_project_locked``.
    pipeline = StagingPipeline.__new__(StagingPipeline)
    pipeline.storage_service = fake_storage
    pipeline.azure_storage = MagicMock()
    pipeline.client = MagicMock()
    pipeline.deployment = "gpt-image-2"
    pipeline.brief_service = MagicMock()

    # Step 1: Simulate a PATCH that ran BEFORE the pipeline persists.
    persisted_doc["name"] = "PATCHED Name"
    persisted_doc["prompt"] = "PATCHED prompt"
    persisted_doc["settings"] = {
        "variations_per_room": 3,
        "model": "gpt-image-2",
        "quality": "high",
        "size": "auto",
    }
    persisted_doc["design_brief"] = {"global_instructions": "PATCHED brief"}

    # Step 2: Pipeline finishes a room and persists. With the surgical
    # fix in place, this should ONLY update ``rooms`` + ``status``.
    # The PATCH-touched user-owned scalars must survive.
    stale_project.rooms[0].status = ItemStatus.COMPLETED
    stale_project.status = ProjectStatus.COMPLETED
    await pipeline._persist_project_locked(stale_project)

    # The PATCH-touched fields are still PATCHED — pipeline writes
    # didn't clobber them with the stale snapshot.
    assert persisted_doc["name"] == "PATCHED Name"
    assert persisted_doc["prompt"] == "PATCHED prompt"
    assert persisted_doc["settings"]["variations_per_room"] == 3
    assert persisted_doc["design_brief"] == {"global_instructions": "PATCHED brief"}

    # And the pipeline-owned fields ARE updated to the worker's values.
    assert persisted_doc["status"] == "completed"
    assert persisted_doc["rooms"][0]["status"] == "completed"
