"""Tests for staging API endpoints."""
import pytest
from unittest.mock import MagicMock, patch


def test_create_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.create_item.return_value = {
        "id": "proj-123",
        "name": "Test Project",
        "prompt": "Modern minimalist",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "created_at": "2026-04-26T00:00:00Z",
        "updated_at": "2026-04-26T00:00:00Z",
        "doc_type": "staging_project",
    }

    response = client.post("/api/v1/staging/projects", json={
        "name": "Test Project",
        "prompt": "Modern minimalist",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["project"]["name"] == "Test Project"
    assert data["project"]["status"] == "uploading"


def test_list_projects(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.query_items.return_value = []
    
    # Mock count query - returns list with a single integer result
    def mock_query_items(query=None, **kwargs):
        if "SELECT VALUE COUNT(1)" in query:
            return [0]  # Count query returns list with integer
        return []
    
    mock_container.query_items = mock_query_items

    response = client.get("/api/v1/staging/projects")
    assert response.status_code == 200
    data = response.json()
    assert data["projects"] == []
    assert data["total"] == 0


def test_get_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-123",
        "name": "Test",
        "prompt": "Test prompt",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    response = client.get("/api/v1/staging/projects/proj-123")
    assert response.status_code == 200
    assert response.json()["project"]["id"] == "proj-123"


def test_get_project_not_found(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.side_effect = CosmosResourceNotFoundError(status_code=404, message="Not found")
    response = client.get("/api/v1/staging/projects/nonexistent")
    assert response.status_code == 404


def test_delete_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {"id": "proj-123"}  # Found
    mock_container.delete_item.return_value = None  # Success
    response = client.delete("/api/v1/staging/projects/proj-123")
    assert response.status_code == 200


def test_delete_project_not_found(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]
    # The delete endpoint calls storage.delete_project which tries to delete directly
    mock_container.delete_item.side_effect = CosmosResourceNotFoundError(status_code=404, message="Not found")
    response = client.delete("/api/v1/staging/projects/nonexistent")
    assert response.status_code == 404


# --- Variation regeneration tests ---

def _project_with_completed_variation():
    """Helper: project with one room, one completed variation with metadata."""
    return {
        "id": "proj-123",
        "name": "Test",
        "prompt": "Modern minimalist",
        "status": "completed",
        "rooms": [{
            "id": "room-1",
            "label": "Living Room",
            "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
            "status": "completed",
            "variations": [{
                "id": "var-1",
                "status": "completed",
                "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
                "generation_metadata": {
                    "model": "gpt-image-2",
                    "adapted_prompt": "Add a cozy reading nook with warm lighting",
                    "generation_time_ms": 5000,
                },
            }],
        }],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }


def test_regenerate_variation_not_found_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = None
    response = client.post("/api/v1/staging/projects/nope/rooms/room-1/variations/var-1/regenerate?strategy=fresh")
    assert response.status_code == 404


def test_regenerate_variation_not_found_room(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_completed_variation()
    response = client.post("/api/v1/staging/projects/proj-123/rooms/bad-room/variations/var-1/regenerate?strategy=fresh")
    assert response.status_code == 404


def test_regenerate_variation_not_found_variation(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_completed_variation()
    response = client.post("/api/v1/staging/projects/proj-123/rooms/room-1/variations/bad-var/regenerate?strategy=fresh")
    assert response.status_code == 404


def test_regenerate_variation_invalid_strategy(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_completed_variation()
    response = client.post("/api/v1/staging/projects/proj-123/rooms/room-1/variations/var-1/regenerate?strategy=invalid")
    assert response.status_code == 400


def test_regenerate_variation_already_processing(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_completed_variation()
    project_data["rooms"][0]["variations"][0]["status"] = "processing"
    mock_container.read_item.return_value = project_data
    response = client.post("/api/v1/staging/projects/proj-123/rooms/room-1/variations/var-1/regenerate?strategy=fresh")
    assert response.status_code == 409


def test_regenerate_variation_preflight_preserves_image_url(client, mock_staging_deps):
    """Issue 002: the endpoint preflight must NOT clear `variation.image_url`.

    The pipeline captures the prior URL for failure rollback and old-blob
    cleanup, so wiping it in the preflight write would defeat that contract.
    Verifies the preflight write keeps the prior image_url intact and sets
    status=processing (not pending).
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_completed_variation()
    prior_image_url = project_data["rooms"][0]["variations"][0]["image_url"]
    mock_container.read_item.return_value = project_data
    # Echo replace_item back so update_project returns valid data
    mock_container.replace_item.side_effect = lambda item, body: body

    # The pipeline's process_single_variation is awaited inside the SSE
    # generator. We don't need it to do anything — the preflight write happens
    # before the generator yields, so we let the response stream to completion
    # but the pipeline is mocked to return an empty async generator.
    mock_pipeline = mock_staging_deps["pipeline"]

    async def _empty_async_gen(*_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    mock_pipeline.process_single_variation = _empty_async_gen

    with client.stream(
        "POST",
        "/api/v1/staging/projects/proj-123/rooms/room-1/variations/var-1/regenerate?strategy=retry",
    ) as response:
        assert response.status_code == 200
        # Drain the stream so the preflight + final writes complete.
        for _ in response.iter_bytes():
            pass

    # The first replace_item call is the preflight write.
    assert mock_container.replace_item.call_count >= 1
    first_call = mock_container.replace_item.call_args_list[0]
    persisted_body = first_call.kwargs.get("body") or first_call.args[1]
    persisted_variation = persisted_body["rooms"][0]["variations"][0]
    # Critical: the prior image_url is preserved through the preflight write.
    assert persisted_variation["image_url"] == prior_image_url, \
        "Preflight write must NOT clear variation.image_url; the pipeline " \
        "needs it for failure rollback and old-blob cleanup (issue 002)."
    # And the variation is now PROCESSING (not PENDING) so the 409 mutex works.
    assert persisted_variation["status"] == "processing"
    # The error field is cleared (in case the variation was previously FAILED).
    assert persisted_variation.get("error") is None

# ============================================================================
# Legacy plant_palette → object_palette migration on read.
#
# Verifies: per-image-object-quantities issue 001 — old persisted briefs are
# transparently migrated when surfaced via the GET project endpoints AND the
# migrated dict is written back so the next read is a no-op.
# ============================================================================


def _legacy_brief_payload():
    """A persisted-shape design brief from the pre-migration era."""
    return {
        "global_instructions": "Add evergreens",
        "plant_palette": [
            {
                "species": "Sequoia",
                "botanical_name": "Sequoiadendron giganteum",
                "quantity": 2,
                "size": "20 ft tall",
                "placement": "north fence",
                "visual_notes": "tall, conical",
            }
        ],
        "placement_guide": {"back_row": "Tall conifers"},
        "preserve_elements": ["patio"],
    }


def _migrated_brief_payload():
    """A persisted-shape design brief that's already been migrated.

    No legacy keys; ``object_palette`` and ``per_image_objects`` set.
    """
    return {
        "global_instructions": "Add evergreens",
        "object_palette": [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "name": "Sequoia",
                "description": "Sequoiadendron giganteum",
                "category": "tree",
                "default_quantity": 2,
                "size": "20 ft tall",
                "placement": "north fence",
                "visual_notes": "tall, conical",
            }
        ],
        "placement_guide": {"back_row": "Tall conifers"},
        "preserve_elements": ["patio"],
        "per_image_objects": {},
    }


def test_get_project_migrates_legacy_plant_palette_and_writes_back(client, mock_staging_deps):
    """A project persisted with the old plant_palette shape is auto-migrated
    on read AND the migrated dict is written back so the next read is a no-op."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-legacy",
        "name": "Legacy",
        "prompt": "Test",
        "status": "completed",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": _legacy_brief_payload(),
    }
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.get("/api/v1/staging/projects/proj-legacy")
    assert response.status_code == 200
    body = response.json()["project"]

    # Response carries object_palette, NOT plant_palette.
    brief = body["design_brief"]
    assert "object_palette" in brief
    assert "plant_palette" not in brief
    assert len(brief["object_palette"]) == 1
    obj = brief["object_palette"][0]
    assert obj["name"] == "Sequoia"
    assert obj["category"] == "tree"
    assert obj["default_quantity"] == 2
    assert obj["description"] == "Sequoiadendron giganteum"
    # Migration assigns a UUID id, regardless of what was persisted before.
    assert isinstance(obj["id"], str) and len(obj["id"]) > 0

    # And the migrated doc was persisted (writeback) — this is what makes the
    # next read a no-op rather than re-running the migration.
    assert mock_container.replace_item.call_count >= 1
    last_call = mock_container.replace_item.call_args_list[-1]
    persisted_body = last_call.kwargs.get("body") or last_call.args[1]
    assert "object_palette" in persisted_body["design_brief"]
    assert "plant_palette" not in persisted_body["design_brief"]


def test_get_project_no_writeback_when_brief_already_migrated(client, mock_staging_deps):
    """Already-migrated briefs MUST NOT trigger an opportunistic writeback —
    avoids needless Cosmos writes on every read."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-already-migrated",
        "name": "Modern",
        "prompt": "Test",
        "status": "completed",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": _migrated_brief_payload(),
    }

    response = client.get("/api/v1/staging/projects/proj-already-migrated")
    assert response.status_code == 200
    body = response.json()["project"]
    assert "object_palette" in body["design_brief"]

    # No-op writeback — no replace_item calls (reconcile_project also returned
    # False because rooms=[] and the project is already terminal).
    assert mock_container.replace_item.call_count == 0


def test_get_project_no_design_brief_does_not_crash(client, mock_staging_deps):
    """Projects without a design_brief (e.g. uploading state) MUST NOT crash
    or trigger a writeback — the migration helper must short-circuit."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-no-brief",
        "name": "Brand new",
        "prompt": "Test",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    response = client.get("/api/v1/staging/projects/proj-no-brief")
    assert response.status_code == 200
    assert mock_container.replace_item.call_count == 0


def test_list_projects_migrates_legacy_plant_palette_and_writes_back(client, mock_staging_deps):
    """list_projects also runs the migration so legacy keys can't leak via
    list endpoints (defense-in-depth — issue 001 explicitly mentions
    surfacing only the new shape)."""
    mock_container = mock_staging_deps["container"]
    legacy_doc = {
        "id": "proj-legacy",
        "name": "Legacy",
        "prompt": "Test",
        "status": "completed",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": _legacy_brief_payload(),
    }

    def mock_query_items(query, parameters=None, enable_cross_partition_query=None):
        if "COUNT" in (query or "").upper():
            return iter([1])
        return iter([legacy_doc])

    mock_container.query_items = mock_query_items
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.get("/api/v1/staging/projects?limit=10")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    project = payload["projects"][0]

    brief = project["design_brief"]
    assert "object_palette" in brief
    assert "plant_palette" not in brief

    # Writeback fired so the next list read is a no-op for this project.
    assert mock_container.replace_item.call_count >= 1


def test_put_brief_round_trip_preserves_typed_per_image_objects(client, mock_staging_deps):
    """Issue 003 of the per-image-object-quantities-design PRD: a brief with
    typed `per_image_objects` overrides survives a PUT round trip with all
    fields intact (object_id, quantity, placement, enabled).
    """
    mock_container = mock_staging_deps["container"]

    project_id = "proj-pio"
    obj_id = "obj-lavender"

    initial = {
        "id": project_id,
        "name": "Per-Image Test",
        "prompt": "",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": None,
    }
    mock_container.read_item.return_value = initial

    # Capture the payload that would be persisted.
    persisted = {}

    def fake_replace_item(item, body):
        persisted["body"] = body
        return body

    mock_container.replace_item.side_effect = fake_replace_item

    brief_payload = {
        "global_instructions": "Lush greenery",
        "object_palette": [
            {
                "id": obj_id,
                "name": "Lavender",
                "description": "Lavandula",
                "category": "plant",
                "default_quantity": 3,
                "size": "2 ft",
                "placement": "front row",
                "visual_notes": None,
            }
        ],
        "placement_guide": {"back_row": "Tall grasses"},
        "preserve_elements": [],
        "per_image_notes": {"room-1": "Heavy reds"},
        "per_image_objects": {
            "room-1": [
                {
                    "object_id": obj_id,
                    "quantity": 7,
                    "placement": "back row",
                    "enabled": True,
                }
            ],
            "room-2": [
                {
                    "object_id": obj_id,
                    "quantity": 0,
                    "placement": None,
                    "enabled": False,
                }
            ],
        },
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    response = client.put(f"/api/v1/staging/projects/{project_id}/brief", json=brief_payload)
    assert response.status_code == 200, response.text

    returned = response.json()["brief"]
    # Round-trip: full structure preserved.
    assert returned["per_image_objects"]["room-1"] == [
        {"object_id": obj_id, "quantity": 7, "placement": "back row", "enabled": True}
    ]
    assert returned["per_image_objects"]["room-2"] == [
        {"object_id": obj_id, "quantity": 0, "placement": None, "enabled": False}
    ]
    assert returned["per_image_notes"] == {"room-1": "Heavy reds"}

    # Persisted Cosmos doc carries the same typed structure (not a free-form
    # `Dict[str, Any]` blob).
    assert persisted["body"]["design_brief"]["per_image_objects"]["room-1"][0]["quantity"] == 7
    assert persisted["body"]["design_brief"]["per_image_objects"]["room-2"][0]["enabled"] is False


def test_put_brief_normalises_placement_whitespace(client, mock_staging_deps):
    """ImageObjectOverride's placement validator (mode='before') strips
    whitespace and turns empty/whitespace-only strings into None. This is
    enforced at the model boundary; assert it actually fires through the
    HTTP layer too.
    """
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-ws",
        "name": "WS",
        "prompt": "",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": None,
    }
    mock_container.replace_item.side_effect = lambda item, body: body

    obj_id = "obj-1"
    brief_payload = {
        "global_instructions": "X",
        "object_palette": [
            {
                "id": obj_id,
                "name": "Lavender",
                "description": "",
                "category": "plant",
                "default_quantity": 3,
                "size": "2 ft",
                "placement": "front",
                "visual_notes": None,
            }
        ],
        "placement_guide": {"back_row": "ZZ"},
        "preserve_elements": [],
        "per_image_notes": {},
        "per_image_objects": {
            "room-1": [
                {"object_id": obj_id, "quantity": 5, "placement": "   ", "enabled": True},
                {"object_id": obj_id + "-dummy-ignored", "quantity": 1, "placement": "  back  row  ", "enabled": True},
            ]
        },
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    response = client.put("/api/v1/staging/projects/proj-ws/brief", json=brief_payload)
    assert response.status_code == 200, response.text

    overrides = response.json()["brief"]["per_image_objects"]["room-1"]
    # whitespace-only → None
    assert overrides[0]["placement"] is None
    # interior whitespace preserved; edges stripped
    assert overrides[1]["placement"] == "back  row"


# ---------------------------------------------------------------------------
# POST /projects/{id}/brief — issue 004 of the per-image-object-quantities PRD.
# The endpoint accepts an optional ``previous_brief`` and returns a
# ``reconciliation_summary`` body field. The wizard's regenerate flow uses
# both fields together to carry forward per-image quantity overrides across
# regeneration.
# ---------------------------------------------------------------------------


def test_post_brief_returns_reconciliation_summary_zero_zero_when_no_previous_brief(
    client, mock_staging_deps
):
    """Smoke-check: the new ``reconciliation_summary`` field is always
    present in the response, even on first generation when no
    ``previous_brief`` is supplied. carried_forward / dropped both 0.
    """
    import json as _json
    from unittest.mock import AsyncMock, MagicMock as _MagicMock

    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-pb-1",
        "name": "X",
        "prompt": "",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": None,
        "analyses": [
            {"room_id": "room-1", "description": "back yard", "features": [], "zones": []}
        ],
    }
    mock_container.replace_item.side_effect = lambda item, body: body

    fake_llm_response = _MagicMock(
        choices=[_MagicMock(message=_MagicMock(content=_json.dumps({
            "global_instructions": "x",
            "object_palette": [
                {
                    "name": "Lavender",
                    "category": "plant",
                    "default_quantity": 3,
                    "size": "2 ft",
                    "placement": "front",
                    "visual_notes": None,
                    "description": None,
                }
            ],
            "placement_guide": {"back_row": "z"},
            "per_image_notes": {},
            "preserve_elements": [],
        })))]
    )
    fake_llm = AsyncMock()
    fake_llm.chat.completions.create.return_value = fake_llm_response

    with patch("backend.core.async_llm_client", fake_llm):
        response = client.post(
            "/api/v1/staging/projects/proj-pb-1/brief",
            json={"conversation_history": [{"role": "user", "content": "Add lavender"}]},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "brief" in body
    assert body["reconciliation_summary"] == {"carried_forward": 0, "dropped": 0}


def test_post_brief_reconciles_previous_brief_overrides_by_name(
    client, mock_staging_deps
):
    """When the request body carries a ``previous_brief``, surviving
    per-image overrides are carried forward by case-insensitive,
    whitespace-trimmed name match against the new palette and the
    ``reconciliation_summary`` reports counts.
    """
    import json as _json
    from unittest.mock import AsyncMock, MagicMock as _MagicMock

    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-pb-2",
        "name": "X",
        "prompt": "",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": None,
        "analyses": [
            {"room_id": "room-1", "description": "back yard", "features": [], "zones": []}
        ],
    }
    mock_container.replace_item.side_effect = lambda item, body: body

    # Previous brief: user manually set Lavender qty=8 in room-1 and
    # also has an orphan "Pine" override that won't survive (Pine isn't
    # in the regenerated palette).
    prev_lav_id = "prev-lav-uuid"
    prev_pine_id = "prev-pine-uuid"
    previous_brief = {
        "global_instructions": "x",
        "object_palette": [
            {"id": prev_lav_id, "name": "Lavender", "category": "plant", "default_quantity": 3, "size": "2 ft", "placement": "front", "visual_notes": None, "description": None},
            {"id": prev_pine_id, "name": "Pine", "category": "tree", "default_quantity": 2, "size": "8 ft", "placement": "back", "visual_notes": None, "description": None},
        ],
        "placement_guide": {"back_row": "z"},
        "per_image_notes": {},
        "preserve_elements": [],
        "per_image_objects": {
            "room-1": [
                {"object_id": prev_lav_id, "quantity": 8, "placement": None, "enabled": True},
                {"object_id": prev_pine_id, "quantity": 5, "placement": None, "enabled": True},
            ]
        },
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    # New LLM response: Lavender survives (different UUID, same name),
    # Pine is gone.
    fake_llm_response = _MagicMock(
        choices=[_MagicMock(message=_MagicMock(content=_json.dumps({
            "global_instructions": "x",
            "object_palette": [
                {
                    "name": "lavender",  # different case, same normalized name.
                    "category": "plant",
                    "default_quantity": 3,
                    "size": "2 ft",
                    "placement": "front",
                    "visual_notes": None,
                    "description": None,
                }
            ],
            "placement_guide": {"back_row": "z"},
            "per_image_notes": {},
            "preserve_elements": [],
        })))]
    )
    fake_llm = AsyncMock()
    fake_llm.chat.completions.create.return_value = fake_llm_response

    with patch("backend.core.async_llm_client", fake_llm):
        response = client.post(
            "/api/v1/staging/projects/proj-pb-2/brief",
            json={
                "conversation_history": [{"role": "user", "content": "regenerate"}],
                "previous_brief": previous_brief,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    # Reconciliation surfaces: 1 carried (Lavender), 1 dropped (Pine).
    assert body["reconciliation_summary"] == {"carried_forward": 1, "dropped": 1}
    # The carried-forward override now points at the NEW palette UUID.
    new_lavender_id = body["brief"]["object_palette"][0]["id"]
    assert new_lavender_id != prev_lav_id
    overrides = body["brief"]["per_image_objects"]["room-1"]
    assert len(overrides) == 1
    assert overrides[0]["object_id"] == new_lavender_id
    assert overrides[0]["quantity"] == 8


# ============================================================================
# Issue 003 of single-variation-regeneration PRD: prompt_diversity threading.
#
# When the user clicks "Try Something New" (strategy=fresh) on a variation
# that has a previously rejected ``adapted_prompt`` in ``generation_metadata``,
# the prior prompt MUST flow through to the LLM call site as negative
# context. These tests drive the regen endpoint and assert the LLM mock
# receives ``messages[0].content`` containing the prior prompt — the
# integration acceptance criterion.
#
# Test pattern: the existing ``mock_staging_deps`` fixture replaces the
# pipeline with a MagicMock. We configure that mock's blob/analyzer to
# return real data, and replace ``adapt_prompt`` with a thin wrapper that
# delegates to a REAL ``StagingPipeline`` instance whose ``async_llm_client``
# is a captured ``AsyncMock``. The brief path uses ``BriefGeneratorService``
# constructed inside the endpoint with ``backend.core.async_llm_client`` —
# patching that module attribute routes the brief LLM call to the captured
# mock. ``process_single_variation`` is stubbed to an empty async generator
# so the test focuses on the prompt-generation seam.
# ============================================================================

import json as _json_for_regen_tests
from unittest.mock import AsyncMock, MagicMock as _MM, patch as _patch


_PRIOR_PROMPT = "MAGENTA-AND-CHROME MAXIMALIST AESTHETIC FROM REJECTED VARIATION"


def _project_fresh_regen_payload(*, with_brief: bool):
    """Project containing one room with one previously-rejected variation.

    The variation's ``generation_metadata.adapted_prompt`` is the prompt
    the user just rejected. ``with_brief=True`` exercises the
    BriefGeneratorService path; False exercises the no-brief
    (``adapt_prompt``) path.
    """
    project = {
        "id": "proj-regen-fresh",
        "name": "Fresh Regen Test",
        "prompt": "USER_INTENT_SENTINEL — Modern minimalist",
        "status": "completed",
        "rooms": [{
            "id": "room-1",
            "label": "Living Room",
            "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
            "status": "completed",
            "analysis": {"description": "A sunlit living room with hardwood floors", "features": ["floor", "window"]},
            "variations": [{
                "id": "var-1",
                "status": "completed",
                "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
                "generation_metadata": {
                    "model": "gpt-image-2",
                    "adapted_prompt": _PRIOR_PROMPT,
                    "generation_time_ms": 5000,
                },
            }],
        }],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "analyses": [{"room_id": "room-1", "description": "A sunlit living room with hardwood floors", "features": ["floor", "window"]}],
    }
    if with_brief:
        project["design_brief"] = {
            "global_instructions": "BRIEF_INTENT_SENTINEL — warm scandinavian palette",
            "object_palette": [
                {"name": "Sofa", "category": "furniture", "default_quantity": 1, "size": "3-seater", "placement": "facing window"},
            ],
            "placement_guide": {"back_row": "abstract art"},
            "per_image_notes": {},
            "preserve_elements": ["hardwood floor"],
            "per_image_objects": {},
        }
    return project


def _make_captured_llm():
    """Build an AsyncMock LLM client that records every chat completion."""
    llm = AsyncMock()
    llm.chat.completions.create.return_value = _MM(
        choices=[_MM(message=_MM(content=_json_for_regen_tests.dumps({"prompts": ["new direction"]})))]
    )
    return llm


def test_fresh_regen_threads_prior_prompt_to_brief_llm_call(client, mock_staging_deps):
    """Brief path: the rejected ``adapted_prompt`` from
    ``generation_metadata`` must appear in ``messages[0].content`` sent
    to the LLM by ``BriefGeneratorService.brief_to_prompts``.

    Patches ``backend.core.async_llm_client`` so the BriefGeneratorService
    constructed inside the endpoint routes through our captured mock.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_fresh_regen_payload(with_brief=True)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    # ``process_single_variation`` is patched at the class level so the
    # request returns immediately after prompt-generation — we don't want
    # it to run real image-gen.
    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-regen-fresh/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    assert captured_llm.chat.completions.create.called, (
        "Brief path must reach the LLM for prompt generation"
    )
    sent = captured_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert _PRIOR_PROMPT in sent, (
        f"Rejected prior prompt missing from LLM call site. Got: {sent[:400]!r}"
    )
    assert "REJECTED_PRIOR_DIRECTION" in sent
    assert "BRIEF_INTENT_SENTINEL" in sent  # brief intent survives steering


def test_fresh_regen_threads_prior_prompt_to_no_brief_llm_call(client, mock_staging_deps):
    """No-brief path: the rejected ``adapted_prompt`` must appear in the
    LLM's system message via ``StagingPipeline.adapt_prompt``.

    Patches ``backend.core.async_llm_client`` so the pipeline's LLM
    client is the captured mock; patches
    ``AzureBlobStorageService.get_asset_content`` to serve a real bytes
    tuple so ``base64.b64encode`` doesn't crash on the auto-MagicMock
    chain; replaces ``StagingPipeline.analyze_room`` with a stub so we
    don't need to wire the analyzer.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_fresh_regen_payload(with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "ROOM_ANALYSIS_SENTINEL — sunlit living room", "features": ["floor", "window"]}

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-regen-fresh/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    assert captured_llm.chat.completions.create.called, (
        "No-brief path must reach the LLM for prompt adaptation"
    )
    sent = captured_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert _PRIOR_PROMPT in sent, (
        f"Rejected prior prompt missing from no-brief LLM call site. Got: {sent[:400]!r}"
    )
    assert "REJECTED_PRIOR_DIRECTION" in sent
    # User intent survives the steering wrapper.
    assert "USER_INTENT_SENTINEL" in sent


def test_fresh_regen_no_prior_metadata_does_not_inject_steering(client, mock_staging_deps):
    """First-ever generation defense: when ``generation_metadata`` lacks an
    ``adapted_prompt`` (e.g. an old variation persisted before issue 002),
    the LLM call must NOT contain the steering block — defaulting to
    ``rejected_prompt=None`` is the no-op contract."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_fresh_regen_payload(with_brief=False)
    project_data["rooms"][0]["variations"][0]["generation_metadata"] = {"model": "gpt-image-2"}
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A sunlit living room", "features": []}

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-regen-fresh/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    sent = captured_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "REJECTED_PRIOR_DIRECTION" not in sent
    assert "REGENERATION STEERING" not in sent


# ============================================================================
# Issue 004 — Retry-to-fresh fallback signaling
# ----------------------------------------------------------------------------
# When a user picks ``Retry Same Prompt`` on a variation that has no prior
# ``adapted_prompt`` recorded (legacy variation, or one that errored before
# issue 001 closed the metadata-persistence gap), the backend silently falls
# back to fresh prompt generation. This slice surfaces the fallback as a
# dedicated ``variation_fallback`` SSE event so the frontend can toast the
# user "no previous prompt found — generating a fresh take instead."
#
# Backend contract:
#  - ``strategy=retry`` AND ``generation_metadata.adapted_prompt`` is missing
#    → emit ``variation_fallback`` BEFORE the fresh-fallback prompt
#    generation work begins. Payload:
#       {"type": "variation_fallback",
#        "room_id": "...", "variation_id": "...",
#        "reason": "no_prior_prompt"}
#  - ``strategy=retry`` AND prior prompt exists → no fallback event.
#  - ``strategy=fresh`` (user explicit choice) → no fallback event.
# ============================================================================


def _parse_sse_stream(byte_chunks):
    """Parse SSE-formatted bytes into a list of ``{type, ...data}`` dicts.

    Joins the chunks into a single buffer and walks the standard
    ``event:`` / ``data:`` framing emitted by ``_sse_event`` in the
    endpoint. Used by the issue 004 fallback tests below.
    """
    buf = b"".join(byte_chunks).decode("utf-8")
    events = []
    current_event = None
    current_data = None
    for raw in buf.split("\n"):
        line = raw.rstrip("\r")
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "":
            if current_event and current_data is not None:
                try:
                    parsed = _json_for_regen_tests.loads(current_data)
                except _json_for_regen_tests.JSONDecodeError:
                    parsed = {"raw": current_data}
                if not isinstance(parsed, dict):
                    parsed = {"value": parsed}
                events.append({"type": current_event, **parsed})
            current_event = None
            current_data = None
    return events


def _project_for_fallback_test(*, with_prior_prompt: bool, with_brief: bool = False):
    """Project containing one variation whose ``generation_metadata`` may or
    may not include ``adapted_prompt``. Used by the fallback-event tests."""
    project = {
        "id": "proj-fallback",
        "name": "Fallback Test",
        "prompt": "Modern minimalist living room",
        "status": "completed",
        "rooms": [{
            "id": "room-1",
            "label": "Living Room",
            "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
            "status": "completed",
            "analysis": {"description": "A sunlit living room", "features": ["floor"]},
            "variations": [{
                "id": "var-1",
                "status": "completed",
                "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
                "generation_metadata": (
                    {"model": "gpt-image-2", "adapted_prompt": "PRIOR PROMPT TEXT", "generation_time_ms": 5000}
                    if with_prior_prompt
                    else {"model": "gpt-image-2", "generation_time_ms": 5000}
                ),
            }],
        }],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "analyses": [{"room_id": "room-1", "description": "A sunlit living room", "features": ["floor"]}],
    }
    if with_brief:
        project["design_brief"] = {
            "global_instructions": "warm scandinavian palette",
            "object_palette": [],
            "placement_guide": {},
            "per_image_notes": {},
            "preserve_elements": [],
            "per_image_objects": {},
        }
    return project


def test_retry_no_prior_prompt_emits_variation_fallback_then_continues_normally(
    client, mock_staging_deps,
):
    """Retry against a variation with no ``adapted_prompt`` must emit a
    ``variation_fallback`` SSE event before doing fresh-fallback work, and
    must continue to a terminal ``project_completed`` event (no early
    termination)."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=False, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A sunlit living room", "features": []}

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    chunks: list[bytes] = []
    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=retry",
        ) as response:
            assert response.status_code == 200, response.text
            for chunk in response.iter_bytes():
                chunks.append(chunk)

    events = _parse_sse_stream(chunks)
    types = [e["type"] for e in events]

    fallback_events = [e for e in events if e["type"] == "variation_fallback"]
    assert len(fallback_events) == 1, (
        f"Expected exactly one variation_fallback event; got types={types}"
    )
    fb = fallback_events[0]
    assert fb["room_id"] == "room-1"
    assert fb["variation_id"] == "var-1"
    assert fb["reason"] == "no_prior_prompt"

    assert "project_completed" in types, (
        f"Stream must reach terminal project_completed; got types={types}"
    )

    assert types.index("variation_fallback") < types.index("project_completed"), (
        "variation_fallback must precede the terminal project_completed event"
    )


def test_retry_with_prior_prompt_does_not_emit_variation_fallback(
    client, mock_staging_deps,
):
    """Retry against a variation whose ``generation_metadata.adapted_prompt``
    is present must NOT emit ``variation_fallback`` — the prior prompt is
    used as-is and there is no fallback."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=True, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline

    chunks: list[bytes] = []
    with _patch.object(StagingPipeline, "process_single_variation", _empty_psv):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=retry",
        ) as response:
            assert response.status_code == 200, response.text
            for chunk in response.iter_bytes():
                chunks.append(chunk)

    events = _parse_sse_stream(chunks)
    types = [e["type"] for e in events]
    assert "variation_fallback" not in types, (
        f"Retry with valid prior prompt must NOT emit fallback event; "
        f"got types={types}"
    )


def test_fresh_strategy_does_not_emit_variation_fallback(client, mock_staging_deps):
    """``strategy=fresh`` is the user's explicit choice — it must not be
    confused with a retry-fallback. Even when there is no prior prompt,
    fresh must emit no fallback event."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=False, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A sunlit living room", "features": []}

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    chunks: list[bytes] = []
    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for chunk in response.iter_bytes():
                chunks.append(chunk)

    events = _parse_sse_stream(chunks)
    types = [e["type"] for e in events]
    assert "variation_fallback" not in types, (
        f"strategy=fresh must NOT emit fallback event; got types={types}"
    )
