"""Tests for ``DELETE /api/v1/staging/projects/{id}/rooms/{rid}`` —
room-level cascading delete endpoint.

Issue history:

- Added in issue 005 of the project-settings-completeness PRD to back
  the new "delete room with confirm" affordance in
  ``ProjectRoomsManager``. The PRD/issue text describes this endpoint
  as "existing" but no DELETE-room endpoint existed on ``main`` —
  same adaptation pattern used in 001/002/003/004 (the PRD was
  written against a stale worktree branch). This slice ADDS the
  endpoint, the API client function, and the inline UI together so
  the slice is end-to-end testable at the HTTP boundary.

Endpoint contract (asserted by the tests below):

    DELETE /api/v1/staging/projects/{id}/rooms/{rid}
    Body: none
    Response: 200 with ``{project: <updated StagingProject>}``

    - Removes the target room from ``project.rooms``.
    - PRUNES room-keyed metadata in the same write so the project doc
      stays internally consistent (rubber-duck blocker — without this,
      stale references would leak into future brief/regenerate flows):
        * ``project.analyses[*]`` entries where ``room_id == room_id``.
        * ``project.design_brief.per_image_notes[room_id]``.
        * ``project.design_brief.per_image_objects[room_id]``.
    - REJECTS with 409 Conflict when ``project.status == "processing"``
      (rubber-duck blocker — the lock alone does not protect against
      a stale ``rooms`` snapshot from an in-flight pipeline worker
      reintroducing the deleted room when the worker eventually
      writes its accumulated state back).
    - 404 on unknown project_id.
    - 404 on unknown room_id (project state untouched).
    - Sibling rooms preserved BYTE-FOR-BYTE.
    - Project-level scalars preserved
      (name / prompt / settings / status — the brief structure
      itself is preserved; only the per-image-* keys for the deleted
      room are pruned).
    - Best-effort blob cleanup runs OUTSIDE the project lock so blob
      I/O latency doesn't block other room edits / regens on the
      project (rubber-duck non-blocking finding). Blob cleanup
      failures are LOGGED but do NOT bubble — the metadata delete
      still returns 200 (mirrors ``delete_project``'s try/except).

The read-modify-write is wrapped in the per-project asyncio lock from
``staging_pipeline._get_project_lock`` so concurrent edits across
different rooms (or a parallel update_room finalizer) cannot clobber
each other through Cosmos's full-doc replacement semantics.
"""
import copy
import json
from unittest.mock import MagicMock, patch

import pytest


def _project_with_two_rooms_and_metadata() -> dict:
    """Two-room project with analyses + design_brief metadata. Used
    by the metadata-pruning tests to assert that only the deleted
    room's keys are pruned and the surviving room's keys are preserved.
    """
    return {
        "id": "proj-del",
        "name": "Delete Test Project",
        "prompt": "modern minimalist",
        "status": "completed",
        "rooms": [
            {
                "id": "room-A",
                "label": "Living Room",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj-del/originals/a.png",
                "original_thumbnail_url": "https://acct.blob.core.windows.net/images/staging/proj-del/originals/a-thumb.png",
                "status": "completed",
                "prompt_addendum": None,
                "variations": [
                    {
                        "id": "var-A1",
                        "status": "completed",
                        "image_url": "https://acct.blob.core.windows.net/images/staging/proj-del/variations/room-A/v1.png",
                    },
                    {
                        "id": "var-A2",
                        "status": "completed",
                        "image_url": "https://acct.blob.core.windows.net/images/staging/proj-del/variations/room-A/v2.png",
                    },
                ],
            },
            {
                "id": "room-B",
                "label": "Kitchen",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj-del/originals/b.png",
                "original_thumbnail_url": "https://acct.blob.core.windows.net/images/staging/proj-del/originals/b-thumb.png",
                "status": "completed",
                "prompt_addendum": "B's existing addendum",
                "variations": [
                    {
                        "id": "var-B1",
                        "status": "completed",
                        "image_url": "https://acct.blob.core.windows.net/images/staging/proj-del/variations/room-B/v1.png",
                    },
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
            {"room_id": "room-A", "label": "Living Room", "description": "A bright living room."},
            {"room_id": "room-A", "label": "Living Room (regen)", "description": "Updated description."},
            {"room_id": "room-B", "label": "Kitchen", "description": "A kitchen."},
        ],
        "design_brief": {
            "global_instructions": "modern minimalist",
            "object_palette": [],
            "placement_guide": {"back_row": ""},
            "per_image_notes": {
                "room-A": "Note for room A",
                "room-B": "Note for room B",
            },
            "per_image_objects": {
                "room-A": [{"name": "couch"}],
                "room-B": [{"name": "stove"}],
            },
            "preserve_elements": [],
            "settings": {
                "variations_per_room": 5,
                "model": "gpt-image-2",
                "quality": "high",
                "size": "auto",
            },
        },
    }


def _captured_replace_body(mock_container: MagicMock) -> dict:
    """Return the most recently persisted document body."""
    call = mock_container.replace_item.call_args
    return call.kwargs.get("body") or call.args[1]


def _find_room(project_data: dict, room_id: str):
    """Find a room by id, returning None if missing."""
    return next((r for r in project_data["rooms"] if r["id"] == room_id), None)


# ---------------------------------------------------------------------------
# Happy path + structural preservation
# ---------------------------------------------------------------------------


def test_delete_room_happy_path_returns_updated_project_without_room(client, mock_staging_deps):
    """Tracer: DELETE on a known room returns 200 with the updated
    project body, and the persisted document no longer contains the
    deleted room.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 200, response.text

    body = response.json()
    assert "project" in body
    rooms = body["project"]["rooms"]
    assert len(rooms) == 1
    assert rooms[0]["id"] == "room-B"

    persisted = _captured_replace_body(mock_container)
    assert _find_room(persisted, "room-A") is None
    assert _find_room(persisted, "room-B") is not None


def test_delete_room_preserves_sibling_rooms_byte_for_byte(client, mock_staging_deps):
    """Sibling rooms must survive the delete BYTE-FOR-BYTE — no
    accidental clobber of any field on rooms we didn't touch.

    Uses a deep-copy of the original room body so the comparison can't
    false-pass against a mutated shared object (rubber-duck guard).
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    room_b_before = copy.deepcopy(_find_room(project_data, "room-B"))
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert json.dumps(_find_room(persisted, "room-B"), sort_keys=True) == json.dumps(
        room_b_before, sort_keys=True
    )


def test_delete_room_preserves_project_level_scalars(client, mock_staging_deps):
    """Project-level scalars (name, prompt, settings, status) must
    survive the delete byte-for-byte. Only the rooms list and the
    room-keyed metadata sub-dicts change.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    name_before = project_data["name"]
    prompt_before = project_data["prompt"]
    status_before = project_data["status"]
    settings_before = json.dumps(project_data["settings"], sort_keys=True)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["name"] == name_before
    assert persisted["prompt"] == prompt_before
    assert persisted["status"] == status_before
    assert json.dumps(persisted["settings"], sort_keys=True) == settings_before


def test_delete_last_room_returns_empty_rooms_list(client, mock_staging_deps):
    """Deleting the only remaining room is a normal 200 — the project
    document survives with an empty rooms list. No status flip.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    project_data["rooms"] = [project_data["rooms"][1]]  # keep only room-B
    project_data["analyses"] = [
        e for e in project_data["analyses"] if e["room_id"] == "room-B"
    ]
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-B")
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["rooms"] == []


# ---------------------------------------------------------------------------
# Error paths (404 / 409)
# ---------------------------------------------------------------------------


def test_delete_room_unknown_project_returns_404(client, mock_staging_deps):
    """An unknown project_id returns 404. No write is attempted."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = None

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete(
            "/api/v1/staging/projects/proj-DOES-NOT-EXIST/rooms/room-A"
        )
    assert response.status_code == 404, response.text
    mock_container.replace_item.assert_not_called()


def test_delete_room_unknown_room_returns_404_and_project_untouched(
    client, mock_staging_deps
):
    """An unknown room_id under a known project returns 404 and the
    project document is NOT modified.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    mock_container.read_item.return_value = project_data

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete(
            "/api/v1/staging/projects/proj-del/rooms/room-DOES-NOT-EXIST"
        )
    assert response.status_code == 404, response.text
    mock_container.replace_item.assert_not_called()


def test_delete_room_returns_409_when_project_processing(client, mock_staging_deps):
    """LOAD-BEARING REGRESSION (rubber-duck blocker for issue 005):

    The endpoint must REJECT delete attempts while the project is
    actively generating. The lock alone is insufficient — an in-flight
    pipeline worker that started BEFORE the delete carries a stale
    ``rooms`` snapshot in memory and will reintroduce the deleted room
    when it eventually writes its accumulated state back.

    The frontend's issue 007 will also disable the affordance during
    processing, but the backend guard is the authoritative protection
    against a programmatic / racing client that bypasses the UI.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    project_data["status"] = "processing"
    mock_container.read_item.return_value = project_data

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 409, response.text
    mock_container.replace_item.assert_not_called()


# ---------------------------------------------------------------------------
# Metadata pruning (rubber-duck blocker — without this, stale references
# leak into future brief/regenerate flows)
# ---------------------------------------------------------------------------


def test_delete_room_prunes_analyses_for_that_room_only(client, mock_staging_deps):
    """All ``analyses`` entries with ``room_id == <deleted>`` are
    removed. Other rooms' analyses survive.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    # Baseline sanity:
    assert sum(1 for e in project_data["analyses"] if e["room_id"] == "room-A") == 2
    assert sum(1 for e in project_data["analyses"] if e["room_id"] == "room-B") == 1
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    persisted_analyses = persisted.get("analyses", [])
    assert all(e["room_id"] != "room-A" for e in persisted_analyses)
    # Surviving room's analyses preserved.
    surviving = [e for e in persisted_analyses if e["room_id"] == "room-B"]
    assert len(surviving) == 1
    assert surviving[0]["description"] == "A kitchen."


def test_delete_room_prunes_design_brief_per_image_notes_for_that_room_only(
    client, mock_staging_deps
):
    """``design_brief.per_image_notes[<deleted>]`` is removed. Other
    rooms' notes and the brief structure itself are preserved.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    notes = persisted["design_brief"]["per_image_notes"]
    assert "room-A" not in notes
    assert notes["room-B"] == "Note for room B"
    # Brief structural fields preserved.
    assert persisted["design_brief"]["global_instructions"] == "modern minimalist"


def test_delete_room_prunes_design_brief_per_image_objects_for_that_room_only(
    client, mock_staging_deps
):
    """``design_brief.per_image_objects[<deleted>]`` is removed. Other
    rooms' object lists are preserved.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    objects = persisted["design_brief"]["per_image_objects"]
    assert "room-A" not in objects
    assert objects["room-B"] == [{"name": "stove"}]


def test_delete_room_with_no_brief_or_analyses_does_not_crash(client, mock_staging_deps):
    """Pruning is defensive: if the project has no brief or no
    analyses (legacy / unmigrated projects), the delete still
    succeeds without an attribute error.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    project_data["analyses"] = None
    project_data["design_brief"] = None
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    with patch("backend.api.endpoints.staging.AzureBlobStorageService"):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Blob cleanup (rubber-duck guard: assert specific delete calls so the
# test can't false-pass against a no-op cleanup mock)
# ---------------------------------------------------------------------------


def test_delete_room_invokes_blob_cleanup_for_originals_and_variation_prefix(
    client, mock_staging_deps
):
    """Blob cleanup deletes:

      - The room's ``original_image_url`` blob.
      - The room's ``original_thumbnail_url`` blob (when present).
      - All blobs under the ``staging/{project_id}/variations/{room_id}/``
        prefix (covers all variations even if some have null
        ``image_url``).

    Asserted by inspecting the recorded delete calls — guards against
    a silently-broken cleanup that would false-pass on the 200-status
    check alone.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    fake_blob_service = MagicMock()
    fake_container_client = MagicMock()
    fake_blob_service.blob_service_client.get_container_client.return_value = (
        fake_container_client
    )
    # Simulate the prefix sweep finding the two variation blobs.
    fake_container_client.list_blobs.return_value = [
        MagicMock(name="staging/proj-del/variations/room-A/v1.png"),
        MagicMock(name="staging/proj-del/variations/room-A/v2.png"),
    ]
    # Configure the .name attributes (MagicMock ``name`` arg sets the
    # repr, not an attribute).
    for blob_mock, name in zip(
        fake_container_client.list_blobs.return_value,
        [
            "staging/proj-del/variations/room-A/v1.png",
            "staging/proj-del/variations/room-A/v2.png",
        ],
    ):
        blob_mock.name = name

    with patch(
        "backend.api.endpoints.staging.AzureBlobStorageService",
        return_value=fake_blob_service,
    ):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 200, response.text

    # Prefix sweep used the right prefix.
    fake_container_client.list_blobs.assert_any_call(
        name_starts_with="staging/proj-del/variations/room-A/"
    )
    # The two variation blobs (from the prefix sweep) AND the original
    # blob AND the thumbnail blob were deleted. We don't assert order
    # — only that each expected blob name was passed to delete_blob.
    delete_calls = [c.args[0] for c in fake_container_client.delete_blob.call_args_list]
    assert "staging/proj-del/variations/room-A/v1.png" in delete_calls
    assert "staging/proj-del/variations/room-A/v2.png" in delete_calls
    assert "staging/proj-del/originals/a.png" in delete_calls
    assert "staging/proj-del/originals/a-thumb.png" in delete_calls


def test_delete_room_blob_cleanup_failure_does_not_block_metadata_delete(
    client, mock_staging_deps
):
    """Best-effort blob cleanup: even if the blob service raises during
    the cleanup pass, the metadata delete already succeeded and the
    response is still 200. Mirrors ``delete_project``'s try/except.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_and_metadata()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    fake_blob_service = MagicMock()
    # Any blob-service interaction raises.
    fake_blob_service.blob_service_client.get_container_client.side_effect = RuntimeError(
        "blob service offline"
    )

    with patch(
        "backend.api.endpoints.staging.AzureBlobStorageService",
        return_value=fake_blob_service,
    ):
        response = client.delete("/api/v1/staging/projects/proj-del/rooms/room-A")
    assert response.status_code == 200, response.text
    # Metadata delete persisted regardless.
    persisted = _captured_replace_body(mock_container)
    assert _find_room(persisted, "room-A") is None
