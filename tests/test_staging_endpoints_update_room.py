"""Tests for ``PATCH /api/v1/staging/projects/{id}/rooms/{rid}`` —
room-level partial-update endpoint.

History:
- Originally added in issue 003 of the projects-page-improvements PRD
  to edit only ``prompt_addendum``.
- Extended in issue 004 of the project-settings-completeness PRD to
  also accept ``label`` so the new ``ProjectRoomsManager`` UI on the
  Project Settings sheet can rename rooms in place.

Endpoint contract (asserted by the tests below):

    PATCH /api/v1/staging/projects/{id}/rooms/{rid}
    Body shape: ``{label?, prompt_addendum?}``

    - Both fields are optional. The endpoint applies updates ONLY to
      the fields the client actually sends — ``__fields_set__``-aware
      for BOTH ``label`` and ``prompt_addendum`` (the latter is
      load-bearing: a label-only PATCH must NOT silently clear an
      existing addendum, which would happen if the handler defaulted
      the absent field to None and unconditionally wrote it back).
    - ``label``: required project state when present. Explicit ``null``
      raises 422 (clients shouldn't clear required fields). Empty /
      whitespace-only also 422. The persisted value is trimmed.
    - ``prompt_addendum``: ``None`` and empty/whitespace-only are
      both meaningful — they explicitly clear the addendum. The
      persisted value is trimmed when non-empty.
    - The endpoint NEVER modifies sibling rooms (byte-for-byte
      preservation), or project-level scalars
      (``name``/``prompt``/``status``/``settings``/``design_brief``).
    - The endpoint NEVER triggers any generation.
    - Returns the updated full ``ProjectResponse`` in plain JSON.

The read-modify-write is wrapped in the per-project asyncio lock
from ``staging_pipeline._get_project_lock`` so concurrent edits across
different rooms (or a parallel regen finalizer) cannot clobber each
other on the way out — that lock semantic was added in issue 002 of
the projects-page-improvements PRD and remains in force here.
"""
import json

import pytest
from unittest.mock import MagicMock


def _project_with_two_rooms() -> dict:
    """Two-room project. Room A has no addendum; Room B has an existing
    addendum so the load-bearing "label-only PATCH preserves addendum"
    regression is straightforward to assert.
    """
    return {
        "id": "proj-rr",
        "name": "Rooms Test Project",
        "prompt": "modern minimalist",
        "status": "completed",
        "rooms": [
            {
                "id": "room-A",
                "label": "Living Room",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/a.png",
                "status": "completed",
                "prompt_addendum": None,
                "variations": [
                    {
                        "id": "var-A1",
                        "status": "completed",
                        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-A/v1.png",
                    }
                ],
            },
            {
                "id": "room-B",
                "label": "Kitchen",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/b.png",
                "status": "completed",
                "prompt_addendum": "B's existing addendum (must survive label-only PATCH)",
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
        "analyses": [],
        "design_brief": None,
    }


def _captured_replace_body(mock_container: MagicMock) -> dict:
    """Pull the body from the most recent ``replace_item`` call. Same
    shape as the helper in ``test_staging_endpoints_patch_project.py``
    so a future reader can compare."""
    call = mock_container.replace_item.call_args
    return call.kwargs.get("body") or call.args[1]


def _find_room(project_data: dict, room_id: str) -> dict:
    return next(r for r in project_data["rooms"] if r["id"] == room_id)


# ---------------------------------------------------------------------------
# Issue 003 (projects-page-improvements) regression coverage:
# addendum-only PATCH continues to behave as before.
# ---------------------------------------------------------------------------


def test_patch_room_addendum_only_updates_only_addendum(client, mock_staging_deps):
    """Pre-issue-004 behavior: addendum-only PATCH updates the addendum
    on the target room and leaves label / sibling room / project-level
    state byte-identical.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    name_before = project_data["name"]
    prompt_before = project_data["prompt"]
    settings_before = json.dumps(project_data["settings"], sort_keys=True)
    room_b_before = json.dumps(_find_room(project_data, "room-B"), sort_keys=True)

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-A",
        json={"prompt_addendum": "fresh addendum"},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    room_a = _find_room(persisted, "room-A")
    assert room_a["label"] == "Living Room"
    assert room_a["prompt_addendum"] == "fresh addendum"
    # Sibling room untouched.
    assert json.dumps(_find_room(persisted, "room-B"), sort_keys=True) == room_b_before
    # Project-level scalars untouched.
    assert persisted["name"] == name_before
    assert persisted["prompt"] == prompt_before
    assert json.dumps(persisted["settings"], sort_keys=True) == settings_before


def test_patch_room_addendum_explicit_null_clears_addendum(client, mock_staging_deps):
    """Explicit ``null`` for ``prompt_addendum`` clears it. The
    normalization at the handler converts empty/whitespace/null to
    None so the persisted shape stays consistent with the composer's
    "absent" rule.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-B",
        json={"prompt_addendum": None},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert _find_room(persisted, "room-B")["prompt_addendum"] is None


# ---------------------------------------------------------------------------
# Issue 004 (project-settings-completeness): label is now editable.
# ---------------------------------------------------------------------------


def test_patch_room_label_only_updates_label(client, mock_staging_deps):
    """Tracer: a label-only PATCH updates the label on the target room.

    Pre-issue-004 the handler explicitly left ``label`` untouched; this
    test pins the new behavior. The post-fix handler trims the value
    and writes it to ``room["label"]``.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-A",
        json={"label": "Master Bedroom"},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert _find_room(persisted, "room-A")["label"] == "Master Bedroom"


def test_patch_room_label_only_preserves_existing_addendum(client, mock_staging_deps):
    """LOAD-BEARING REGRESSION (rubber-duck blocker for issue 004):

    A label-only PATCH on a room that has an existing
    ``prompt_addendum`` MUST NOT silently clear the addendum. The
    pre-fix handler unconditionally wrote ``body.prompt_addendum``
    to the room — which defaults to ``None`` when the client didn't
    send the field — silently clearing an existing addendum on every
    label rename. The post-fix handler is ``__fields_set__``-aware
    for BOTH ``label`` and ``prompt_addendum``, so omitted fields
    leave the persisted value untouched.

    Without this regression, the rooms manager UI would clear every
    room's addendum on every rename. This is the test that catches
    that bug.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    addendum_before = _find_room(project_data, "room-B")["prompt_addendum"]
    assert addendum_before is not None  # baseline sanity
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-B",
        json={"label": "Renamed Kitchen"},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    room_b = _find_room(persisted, "room-B")
    assert room_b["label"] == "Renamed Kitchen"
    assert room_b["prompt_addendum"] == addendum_before


def test_patch_room_addendum_only_preserves_existing_label(client, mock_staging_deps):
    """Symmetric regression: an addendum-only PATCH must NOT clear or
    rewrite the room's label. Already covered by the addendum-only
    test above, but pinned explicitly here so a future change to make
    the handler "label-aware" doesn't accidentally overwrite the
    label with None when it's absent from the body.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    label_before = _find_room(project_data, "room-B")["label"]
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-B",
        json={"prompt_addendum": "another addendum"},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert _find_room(persisted, "room-B")["label"] == label_before


def test_patch_room_label_and_addendum_in_one_request(client, mock_staging_deps):
    """When both fields are present, both are applied. No precedence
    weirdness — the two fields are independent."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-A",
        json={"label": "Den", "prompt_addendum": "with leather chairs"},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    room_a = _find_room(persisted, "room-A")
    assert room_a["label"] == "Den"
    assert room_a["prompt_addendum"] == "with leather chairs"


def test_patch_room_label_is_trimmed(client, mock_staging_deps):
    """Surrounding whitespace is stripped before persisting. Mirrors
    the existing addendum-trimming behavior; matches the ``name`` /
    ``prompt`` rules in ``UpdateProjectRequest``.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-A",
        json={"label": "  Trimmed Label  "},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert _find_room(persisted, "room-A")["label"] == "Trimmed Label"


def test_patch_room_label_explicit_null_returns_422(client, mock_staging_deps):
    """``label`` cannot be cleared. Explicit ``null`` raises 422 at
    parse time — clients should omit the field to leave it unchanged.
    Mirrors the ``name`` and ``prompt`` rules in
    ``UpdateProjectRequest``.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    mock_container.read_item.return_value = project_data

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-A",
        json={"label": None},
    )
    assert response.status_code == 422, response.text
    # No write occurred.
    mock_container.replace_item.assert_not_called()


@pytest.mark.parametrize("bad_label", ["", "   ", "\t\n  "])
def test_patch_room_label_empty_or_whitespace_returns_422(client, mock_staging_deps, bad_label):
    """Empty / whitespace-only label is a 422 — same rule as
    project-level ``name``/``prompt``."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    mock_container.read_item.return_value = project_data

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-A",
        json={"label": bad_label},
    )
    assert response.status_code == 422, response.text
    mock_container.replace_item.assert_not_called()


def test_patch_room_unknown_room_returns_404(client, mock_staging_deps):
    """An unknown room id under a known project still returns 404."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    mock_container.read_item.return_value = project_data

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-DOES-NOT-EXIST",
        json={"label": "Whatever"},
    )
    assert response.status_code == 404, response.text


def test_patch_room_empty_body_is_noop(client, mock_staging_deps):
    """An empty body changes neither label nor addendum on the target
    room. Pre-issue-004 the handler still persisted (writing
    addendum=None unconditionally — that's the bug this slice fixes);
    post-fix the handler may either write the document unchanged
    (modulo the storage layer's automatic ``updated_at`` bump) or
    skip the write entirely. We assert directly on the load-bearing
    fields rather than on a full byte-for-byte equality so the
    handler can take either correct path.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms()
    label_before = _find_room(project_data, "room-B")["label"]
    addendum_before = _find_room(project_data, "room-B")["prompt_addendum"]
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-rr/rooms/room-B",
        json={},
    )
    assert response.status_code == 200, response.text

    if mock_container.replace_item.called:
        persisted = _captured_replace_body(mock_container)
        room_b = _find_room(persisted, "room-B")
        assert room_b["label"] == label_before
        assert room_b["prompt_addendum"] == addendum_before
